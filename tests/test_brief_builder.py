"""Phase 2: prove that only VERIFIED citations reach a brief, and that the
resulting brief is schema-valid and flows cleanly through the existing
Phase 0 script-generation path unchanged."""
import pytest
from shorts_factory.brief_builder import InsufficientVerifiedClaims, build_brief_from_citations
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.llm import StubLLMProvider
from shorts_factory.schema_validate import validate_brief, validate_script_against_brief


def _citation(claim_id, text, verified, confidence, domain="a.example"):
    return {
        "claim_id": claim_id,
        "claim_text": text,
        "sources": [{"title": f"Source for {claim_id}", "url": f"https://{domain}/{claim_id}", "domain": domain}],
        "independent_domain_count": 2 if verified else 1,
        "confidence": confidence,
        "verified": verified,
    }


def _store(citations):
    return {
        "citation_count": len(citations),
        "verified_count": sum(1 for c in citations if c["verified"]),
        "citations": citations,
    }


VERIFIED_CLAIMS = [
    _citation("soap-claim-01", "Soap forms via saponification of fat and alkali.", True, 1.0),
    _citation("soap-claim-02", "Wood-ash lye was used before manufactured lye existed.", True, 0.7),
    _citation("soap-claim-03", "Tallow was a common traditional soap-making fat.", True, 0.7),
    _citation("soap-claim-04", "Lye is caustic and must be handled with protection.", True, 1.0),
    _citation("soap-claim-05", "Soap reaches trace when the mixture thickens enough to hold shape.", True, 0.7),
]
FABRICATED_CLAIM = _citation(
    "soap-claim-99", "Ancient Romans invented liquid soap using synthetic detergents.", False, 0.2,
)


def test_brief_built_only_from_verified_claims():
    store = _store(VERIFIED_CLAIMS + [FABRICATED_CLAIM])
    brief = build_brief_from_citations("soap", store, safety_class="yellow", caution="Lye is caustic.")

    claim_texts = [c["claim"] for c in brief["claims"]]
    assert FABRICATED_CLAIM["claim_text"] not in claim_texts
    assert len(brief["claims"]) == 5
    assert all(c["id"].startswith("claim-") for c in brief["claims"])


def test_brief_from_citations_passes_schema_validation():
    store = _store(VERIFIED_CLAIMS)
    brief = build_brief_from_citations("soap", store, safety_class="yellow", caution="Lye is caustic.")
    validate_brief(brief)  # must not raise


def test_insufficient_verified_claims_raises():
    store = _store(VERIFIED_CLAIMS[:2] + [FABRICATED_CLAIM])  # only 2 verified, need >=4
    with pytest.raises(InsufficientVerifiedClaims):
        build_brief_from_citations("soap", store, safety_class="yellow")


def test_low_confidence_verified_claim_excluded_by_default_bar():
    borderline = _citation("soap-claim-06", "A borderline low-confidence claim.", True, 0.3)
    store = _store(VERIFIED_CLAIMS + [borderline])
    brief = build_brief_from_citations("soap", store, safety_class="yellow")
    claim_texts = [c["claim"] for c in brief["claims"]]
    assert borderline["claim_text"] not in claim_texts


def test_citation_derived_brief_flows_through_existing_script_generation():
    """End to end: citations -> brief -> StubLLMProvider (unchanged) ->
    schema-valid script with storyboard fields, every claim traceable."""
    store = _store(VERIFIED_CLAIMS)
    brief = build_brief_from_citations("soap", store, safety_class="yellow", caution="Lye is caustic.")

    tracker = CostTracker(budget_cap_usd=2.00)
    script = StubLLMProvider().generate_script(brief, "English", "illustrated realism", tracker)
    validate_script_against_brief(script, brief)  # must not raise

    for scene in script["scenes"]:
        assert "camera" in scene and scene["camera"]
        assert "sfx" in scene  # present (may be None)
