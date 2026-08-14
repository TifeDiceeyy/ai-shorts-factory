"""Phase 1 retrieval/claim-extraction/verification tests.

Uses a test-only FakeSearchProvider with a small, realistic multi-domain
corpus instead of hitting the live Tavily API — keeps the suite fast and
network-independent while still exercising the real chunking/extraction/
verification logic. The corpus is deliberately constructed so that one claim
is genuinely corroborated across independent domains, and one is fabricated
and appears in exactly one domain — this is what the adversarial
fabricated-claim-rejection test (CLAUDE.md Phase 1 gate) checks.
"""
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.search import SearchProvider, SearchResult
from shorts_factory.retrieval import (
    extract_candidate_claims,
    ingest_search_results,
    retrieve_for_topic,
    verify_claim,
    write_citation_store,
)

SOAP_DOMAIN_A_TEXT = (
    "Soap is produced through saponification, a chemical reaction in which a "
    "fat or oil reacts with an alkali such as sodium hydroxide to produce soap "
    "and glycerol.\n\n"
    "Before manufactured lye was available, people commonly leached wood ash "
    "with water to produce an alkaline lye solution for making soap."
)

SOAP_DOMAIN_B_TEXT = (
    "Traditional soap making relies on saponification, the reaction between "
    "fats and an alkali like sodium hydroxide, which yields soap along with "
    "glycerol as a byproduct.\n\n"
    "Historical soap makers often rendered tallow from cattle or sheep fat as "
    "their primary source of fat for the reaction."
)

# Deliberately fabricated, absurd, and NOT corroborated anywhere else in this
# fixture corpus — this is the claim the adversarial test expects rejected.
SOAP_DOMAIN_C_FABRICATED_TEXT = (
    "Historians confirm that ancient Romans invented liquid soap using "
    "synthetic detergents imported from Mesopotamia in 300 BC.\n\n"
    "This fabricated claim about Roman synthetic detergent imports appears "
    "in no other independent source."
)


class FakeSearchProvider(SearchProvider):
    """Test-only — canned results, no network. Not a runtime config option;
    see providers/search.py for why there's no real stub for search."""

    name = "fake"

    def __init__(self, results_by_query: dict[str, list[SearchResult]]):
        self.results_by_query = results_by_query
        self.call_log: list[str] = []

    def search(self, query, max_results, cost_tracker):
        self.call_log.append(query)
        cost_tracker.check_budget(f"fake.search[{query}]", 0.0)
        cost_tracker.record("fake", f"fake.search[{query}]", 0.0, 0.0, is_stub=True)
        return self.results_by_query.get(query, [])[:max_results]


def _soap_corpus_provider() -> FakeSearchProvider:
    return FakeSearchProvider({
        "soap making saponification history": [
            SearchResult(url="https://chem-reference.example/soap", title="Saponification Chemistry",
                         content=SOAP_DOMAIN_A_TEXT, domain="chem-reference.example"),
            SearchResult(url="https://craft-history.example/soap", title="Traditional Soap Making",
                         content=SOAP_DOMAIN_B_TEXT, domain="craft-history.example"),
            SearchResult(url="https://myth-site.example/soap", title="Surprising Soap Facts",
                         content=SOAP_DOMAIN_C_FABRICATED_TEXT, domain="myth-site.example"),
        ],
    })


def test_retrieve_for_topic_returns_chunks_from_multiple_domains():
    provider = _soap_corpus_provider()
    tracker = CostTracker(budget_cap_usd=2.00)

    chunks = retrieve_for_topic("soap", ["soap making saponification history"], provider, tracker)

    domains = {c.source_domain for c in chunks}
    assert domains == {"chem-reference.example", "craft-history.example", "myth-site.example"}
    assert len(provider.call_log) == 1


def test_claim_extraction_picks_up_soap_sentences():
    provider = _soap_corpus_provider()
    tracker = CostTracker(budget_cap_usd=2.00)
    chunks = retrieve_for_topic("soap", ["soap making saponification history"], provider, tracker)

    claims = extract_candidate_claims(chunks, "soap", ["soap", "saponification", "lye"])

    assert len(claims) >= 3
    assert any("saponification" in c.text.lower() for c in claims)


def test_genuine_claim_is_verified_via_independent_domain_corroboration():
    provider = _soap_corpus_provider()
    tracker = CostTracker(budget_cap_usd=2.00)
    chunks = retrieve_for_topic("soap", ["soap making saponification history"], provider, tracker)
    claims = extract_candidate_claims(chunks, "soap", ["soap", "saponification"])

    saponification_claim = next(c for c in claims if "saponification" in c.text.lower() and c.origin_domain == "chem-reference.example")
    citation = verify_claim(saponification_claim, chunks)

    assert citation.verified is True
    assert citation.independent_domain_count >= 2
    corroborating_domains = {s["domain"] for s in citation.sources}
    assert "craft-history.example" in corroborating_domains


def test_fabricated_claim_is_rejected_for_lack_of_independent_corroboration():
    """The adversarial check CLAUDE.md Phase 1 gate requires: a fabricated
    claim present in only one source must NOT be independently verified."""
    provider = _soap_corpus_provider()
    tracker = CostTracker(budget_cap_usd=2.00)
    chunks = retrieve_for_topic("soap", ["soap making saponification history"], provider, tracker)
    claims = extract_candidate_claims(chunks, "soap", ["soap", "roman", "detergent"])

    fabricated = next(c for c in claims if "mesopotamia" in c.text.lower() or "synthetic detergents" in c.text.lower())
    citation = verify_claim(fabricated, chunks)

    assert citation.verified is False, (
        "a fabricated claim appearing in exactly one source must not be "
        "independently verified — the verifier rubber-stamped it instead"
    )
    assert citation.independent_domain_count == 1
    assert citation.confidence < 0.5


def test_citation_store_writes_expected_structure(tmp_path):
    provider = _soap_corpus_provider()
    tracker = CostTracker(budget_cap_usd=2.00)
    chunks = retrieve_for_topic("soap", ["soap making saponification history"], provider, tracker)
    claims = extract_candidate_claims(chunks, "soap", ["soap", "saponification", "lye", "tallow"])
    citations = [verify_claim(c, chunks) for c in claims]

    out_path = tmp_path / "soap.citations.json"
    payload = write_citation_store(citations, out_path)

    assert out_path.exists()
    assert payload["citation_count"] == len(claims)
    assert payload["verified_count"] >= 1
    for entry in payload["citations"]:
        assert "sources" in entry and len(entry["sources"]) >= 1
        assert "confidence" in entry
        for src in entry["sources"]:
            assert src["url"].startswith("https://")
            assert src["domain"]
