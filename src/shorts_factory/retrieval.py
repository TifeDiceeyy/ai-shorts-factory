"""Phase 1: knowledge & research.

ingest (chunk fetched documents) -> retrieve (tag by topic) -> extract
candidate claims -> verify each against an independent (different-domain)
source -> store citation + confidence.

No LLM is used for claim extraction or verification — LLM_PROVIDER is still
stub (no real provider approved), and inventing a "smart" extraction step
that's secretly just as unverified as the thing it's supposed to check would
defeat the point. Everything here is deliberately simple, rule-based, and
auditable: sentence splitting + keyword-overlap corroboration across
independently-fetched, different-domain sources. This is honest about what
it is — a real, if unsophisticated, independent-source check — not a
simulation of one.
"""
from __future__ import annotations

import datetime
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .cost_tracker import CostTracker
from .providers.search import SearchProvider, SearchResult

REPO_ROOT = Path(__file__).resolve().parents[2]

MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 900

MIN_CLAIM_CHARS = 30
MAX_CLAIM_CHARS = 320

# Minimal built-in stopword list — enough to strip function words for a
# bag-of-content-words overlap check. Not exhaustive; this is a deliberately
# simple heuristic, not an NLP pipeline.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "of",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "this",
    "that", "these", "those", "it", "its", "as", "can", "could", "may",
    "might", "will", "would", "should", "not", "no", "nor", "which", "who",
    "whom", "what", "when", "where", "why", "how", "also", "such", "there",
    "their", "they", "you", "your", "we", "our", "i", "he", "she", "his",
    "her", "them", "more", "most", "other", "some", "any", "each", "into",
}

NAV_BOILERPLATE_PREFIXES = (
    "cookie", "subscribe", "advertisement", "click here", "sign up",
    "log in", "menu", "navigation", "copyright", "all rights reserved",
    "share this", "related articles", "read more",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


def content_words(text: str) -> set[str]:
    words = (w.lower() for w in WORD_RE.findall(text))
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str
    source_title: str
    source_domain: str
    fetched_at: str


@dataclass
class Claim:
    claim_id: str
    text: str
    topic: str
    origin_chunk_id: str
    origin_domain: str
    origin_url: str
    origin_title: str


@dataclass
class Citation:
    claim_id: str
    claim_text: str
    topic: str
    sources: list[dict] = field(default_factory=list)  # [{title, url, domain, accessed_at}]
    independent_domain_count: int = 0
    confidence: float = 0.0
    verified: bool = False


def chunk_text(text: str, source_meta: dict, chunk_prefix: str) -> list[Chunk]:
    """Generic paragraph-based chunking. Works on any plain text — a fetched
    web page today, a real BOOK_FILE later through the same path."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    idx = 0

    def flush(buf: str) -> None:
        nonlocal idx
        buf = buf.strip()
        if len(buf) < MIN_CHUNK_CHARS:
            return
        idx += 1
        chunks.append(
            Chunk(
                chunk_id=f"{chunk_prefix}-{idx:03d}",
                text=buf,
                source_url=source_meta["url"],
                source_title=source_meta["title"],
                source_domain=source_meta["domain"],
                fetched_at=source_meta["fetched_at"],
            )
        )

    for para in paragraphs:
        if len(buf) + len(para) + 1 <= MAX_CHUNK_CHARS:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            flush(buf)
            buf = para
    flush(buf)
    return chunks


def ingest_search_results(
    results: list[SearchResult],
    topic: str,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i, r in enumerate(results):
        if not r.content or not r.content.strip():
            continue
        source_meta = {"url": r.url, "title": r.title, "domain": r.domain, "fetched_at": fetched_at}
        prefix = f"{topic}-src{i:02d}"
        all_chunks.extend(chunk_text(r.content, source_meta, prefix))
    return all_chunks


def retrieve_for_topic(
    topic: str,
    queries: list[str],
    search_provider: SearchProvider,
    cost_tracker: CostTracker,
    max_results_per_query: int = 4,
) -> list[Chunk]:
    """Tagged retrieval: run each query, ingest+chunk every result, tag with topic."""
    all_chunks: list[Chunk] = []
    for query in queries:
        results = search_provider.search(query, max_results_per_query, cost_tracker)
        all_chunks.extend(ingest_search_results(results, topic))
    return all_chunks


def _is_boilerplate(sentence: str) -> bool:
    s = sentence.strip().lower()
    return any(s.startswith(p) for p in NAV_BOILERPLATE_PREFIXES)


def extract_candidate_claims(chunks: list[Chunk], topic: str, topic_keywords: list[str]) -> list[Claim]:
    """Rule-based claim extraction: sentences that mention a topic keyword,
    look like a factual statement (not a question, not nav boilerplate), and
    fall in a reasonable length range. No LLM — see module docstring."""
    keywords_lower = [k.lower() for k in topic_keywords]
    claims: list[Claim] = []
    seen_normalized: set[str] = set()

    for chunk in chunks:
        sentences = SENTENCE_SPLIT_RE.split(chunk.text.replace("\n", " "))
        for sentence in sentences:
            s = sentence.strip()
            if not (MIN_CLAIM_CHARS <= len(s) <= MAX_CLAIM_CHARS):
                continue
            if s.endswith("?"):
                continue
            if _is_boilerplate(s):
                continue
            s_lower = s.lower()
            if not any(kw in s_lower for kw in keywords_lower):
                continue

            normalized = " ".join(s_lower.split())
            if normalized in seen_normalized:
                continue
            seen_normalized.add(normalized)

            claims.append(
                Claim(
                    claim_id=f"{topic}-claim-{len(claims) + 1:02d}",
                    text=s,
                    topic=topic,
                    origin_chunk_id=chunk.chunk_id,
                    origin_domain=chunk.source_domain,
                    origin_url=chunk.source_url,
                    origin_title=chunk.source_title,
                )
            )
    return claims


def verify_claim(claim: Claim, chunks: list[Chunk], overlap_threshold: float = 0.5) -> Citation:
    """A claim is independently verified only if a chunk from a DIFFERENT
    domain than its origin shares enough content words with it. This is a
    real, if simple, corroboration check — not a rubber stamp. See module
    docstring for why it's rule-based rather than LLM-based."""
    claim_words = content_words(claim.text)
    sources = [{
        "title": claim.origin_title,
        "url": claim.origin_url,
        "domain": claim.origin_domain,
    }]

    corroborating_domains: dict[str, dict] = {}
    if claim_words:
        for chunk in chunks:
            if chunk.source_domain == claim.origin_domain:
                continue
            chunk_words = content_words(chunk.text)
            if not chunk_words:
                continue
            overlap = len(claim_words & chunk_words) / len(claim_words)
            if overlap >= overlap_threshold and chunk.source_domain not in corroborating_domains:
                corroborating_domains[chunk.source_domain] = {
                    "title": chunk.source_title,
                    "url": chunk.source_url,
                    "domain": chunk.source_domain,
                }

    sources.extend(corroborating_domains.values())
    independent_domain_count = 1 + len(corroborating_domains)
    verified = len(corroborating_domains) >= 1
    confidence = round(min(1.0, 0.4 + 0.3 * len(corroborating_domains)), 2) if verified else 0.2

    return Citation(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        topic=claim.topic,
        sources=sources,
        independent_domain_count=independent_domain_count,
        confidence=confidence,
        verified=verified,
    )


def write_citation_store(citations: list[Citation], out_path: Path) -> dict:
    payload = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "citation_count": len(citations),
        "verified_count": sum(1 for c in citations if c.verified),
        "citations": [
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "topic": c.topic,
                "sources": c.sources,
                "independent_domain_count": c.independent_domain_count,
                "confidence": c.confidence,
                "verified": c.verified,
            }
            for c in citations
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_retrieval_for_topic(
    topic: str,
    search_provider: SearchProvider,
    cost_tracker: CostTracker,
    book_file: str = "",
) -> dict:
    """The real, end-to-end Phase 1 pipeline for one topic: retrieve -> extract
    -> verify -> store. Requires a real, keyed search_provider — see
    providers/search.py, there is no stub for this step."""
    from .topics import TOPIC_KEYWORDS, TOPIC_QUERIES

    queries = TOPIC_QUERIES.get(topic)
    keywords = TOPIC_KEYWORDS.get(topic)
    if not queries or not keywords:
        raise ValueError(f"no configured queries/keywords for topic {topic!r} — add to topics.py first")

    queries = list(queries)
    book_refs: list[str] = []
    if book_file:
        from .book_ingest import build_private_index, research_terms, retrieve_private_chunks

        index_path = REPO_ROOT / "data" / "private" / "book-index.json"
        index = build_private_index(Path(book_file), index_path)
        private_chunks = retrieve_private_chunks(index, topic)
        book_refs = [chunk["id"] for chunk in private_chunks]
        terms = research_terms(private_chunks, topic)
        if terms:
            queries.append(f"{topic} {' '.join(terms)} reliable source")

    chunks = retrieve_for_topic(topic, queries, search_provider, cost_tracker)
    claims = extract_candidate_claims(chunks, topic, keywords)
    citations = [verify_claim(c, chunks) for c in claims]

    out_path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
    payload = write_citation_store(citations, out_path)
    payload["_meta"] = {
        "topic": topic,
        "chunk_count": len(chunks),
        "claim_count": len(claims),
        "queries_used": queries,
        "output_path": str(out_path),
        "private_book_chunk_ids": book_refs,
        "book_text_exposed": False,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str]) -> int:
    from .config import BudgetApprovalRequired, load_settings, require_budget_approval_if_paid
    from .providers.search import get_search_provider

    if len(argv) < 2:
        print("usage: python -m shorts_factory.retrieval <topic>", file=sys.stderr)
        return 2
    topic = argv[1]

    settings = load_settings()
    try:
        require_budget_approval_if_paid(settings)
    except BudgetApprovalRequired as e:
        print(f"BUDGET APPROVAL REQUIRED: {e}", file=sys.stderr)
        return 1

    if settings.search.is_stub:
        print(
            "SEARCH NOT CONFIGURED: set SEARCH_PROVIDER=tavily and SEARCH_API_KEY "
            "in .env — there is no stub option for retrieval (see providers/search.py).",
            file=sys.stderr,
        )
        return 1

    search_provider = get_search_provider(settings.search.provider, settings.search_api_key)
    cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)

    result = run_retrieval_for_topic(topic, search_provider, cost_tracker, book_file=settings.book_file)

    cost_report_path = REPO_ROOT / "data" / topic.replace(" ", "_") / "retrieval-cost-report.json"
    cost_tracker.write_report(cost_report_path)

    print(f"topic={topic}")
    print(f"chunks={result['_meta']['chunk_count']} claims={result['_meta']['claim_count']}")
    print(f"verified={result['verified_count']}/{result['citation_count']}")
    print(f"citation store: {result['_meta']['output_path']}")
    print(f"cost: ${cost_tracker.total_spent_usd:.4f} / ${settings.budget_cap_usd:.2f} cap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
