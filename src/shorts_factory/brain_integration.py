"""Adapter between the local, zero-cost `brain` knowledge base
(src/shorts_factory/brain — a book-grounded local index over "The
Knowledge", "How to Invent Everything", and a third rebuild-civilization
text) and this pipeline's brief/claim schema (brief.schema.json,
brief_builder.build_brief_from_citations' output shape).

User request 2026-08-28: "wire the brain into the pipeline... questions
should go through brain first" — every real topic now tries the brain
BEFORE falling back to real, paid Tavily retrieval (run_pipeline's existing
citations.json path, unchanged as the fallback for topics the brain doesn't
cover well).

Brain claims skip the independent-domain-corroboration "verified" gate that
Tavily-sourced claims go through in retrieval.py — there's no second
independent domain to corroborate against inside one local library. That's
a deliberate choice (matches the earlier-session decision that the brain can
fully replace retrieval for well-covered topics), not an oversight: the
source books are real, purchased, human-authored non-fiction, trusted by
construction, not scraped web content of unknown provenance.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .brief_builder import MAX_CLAIMS_FOR_BRIEF, MIN_CLAIMS_FOR_BRIEF

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAIN_DATA_DIR = REPO_ROOT / "data" / "brain"

# Empirically calibrated 2026-08-28 (real local queries, zero cost) against a
# mix of well-covered and clearly-uncovered topics: furnace/concrete/water
# filtration/electricity/soap/charcoal all scored 13.8-20.9 (average of the
# top 5 BM25 result scores); genuinely uncovered modern topics (wifi
# routers, social media algorithms, 5g cellular networks, video game
# consoles) scored 0-6.9. NOT a perfect signal — it's BM25 keyword overlap,
# not semantic judgment: "pottery"/"rope" (real registered topics, plausibly
# covered by the books) scored below this bar, and "electric cars battery
# tech" scored above it as a likely false positive from generic
# electricity/battery terms. When the heuristic is wrong, the topic simply
# falls back to the existing Tavily retrieval path unchanged — this never
# silently ships bad content, it just misses the brain-first optimization
# in either direction sometimes.
BRAIN_MIN_AVG_TOP5_SCORE = 12.0

# Brain's extractive fact-picking occasionally lets raw OCR/citation
# artifacts through (a stray "(page 158)", a book's own "Key Point:"
# callout) — harmless in a book margin, but jarring if read verbatim as a
# spoken narration claim. Drop those rather than trying to clean them.
_NOISE_RE = re.compile(r"\(page\s+\d+\)|key point\s*:|critical thinking", re.IGNORECASE)
# A fact whose underlying source chunk ends mid-sentence (the chunk's own
# character-count cutoff landing before the next '.'/'!'/'?', not a real
# sentence boundary) survives brain's own sentence-splitting as a truncated
# fragment — confirmed for real 2026-08-28 against a "furnace" query
# ("...is a long, hot, d"). A real sentence always ends in terminal
# punctuation; anything that doesn't is a cut fragment, not a usable claim.
_TERMINAL_PUNCT = (".", "!", '"', "'", ")")


def load_brain():
    """Returns a Brain instance over the already-built local index, or None
    if it hasn't been built on this machine (a missing brain is a
    legitimate, unexceptional state, not an error — the brain has no
    ingestion capability of its own anymore; see brain/__init__.py's
    docstring). Brain's own default data_dir resolves to
    src/shorts_factory/data (wrong — that directory doesn't exist in this
    repo; the real built index lives at REPO_ROOT/data/brain), so
    BRAIN_DATA_DIR must be passed explicitly."""
    from .brain import Brain

    try:
        return Brain(data_dir=BRAIN_DATA_DIR)
    except RuntimeError:
        return None


def _clean_facts(facts: list[str]) -> list[str]:
    return [
        f for f in facts
        if not _NOISE_RE.search(f) and f.strip().endswith(_TERMINAL_PUNCT)
    ]


def brain_covers_topic(brain, topic: str) -> tuple[bool, dict[str, Any] | None]:
    """Queries the brain for `topic` and judges whether the result is
    strong enough to fully replace external retrieval for this run. Returns
    (covered, research) — research is brain.research_brief()'s own return
    value (with key_facts already noise-filtered) when covered, else
    (False, None)."""
    research = brain.research_brief(topic, top_k=10)
    facts = _clean_facts(research.get("key_facts", []))
    if len(facts) < MIN_CLAIMS_FOR_BRIEF:
        return False, None
    sources = research.get("sources", [])[:5]
    if not sources:
        return False, None
    avg_top5 = sum(s.get("score", 0.0) for s in sources) / len(sources)
    if avg_top5 < BRAIN_MIN_AVG_TOP5_SCORE:
        return False, None
    return True, {**research, "key_facts": facts}


def build_brief_from_brain(
    topic: str,
    research: dict[str, Any],
    safety_class: str,
    caution: str | None = None,
    idea: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Same output shape as brief_builder.build_brief_from_citations — a
    schema-valid brief.schema.json object — built from a brain
    research_brief() result instead of a Tavily citation store. `research`
    should already have noise-filtered key_facts (see brain_covers_topic)."""
    facts = research["key_facts"][:MAX_CLAIMS_FOR_BRIEF]
    books = sorted({s["book"] for s in research.get("sources", []) if s.get("book")})
    source_desc = f"{', '.join(books)} (local knowledge base)" if books else "local knowledge base"

    claims = [
        {"id": f"claim-{i:02d}", "claim": fact, "source": source_desc}
        for i, fact in enumerate(facts, start=1)
    ]

    brief: dict[str, Any] = {"topic": topic, "safety_class": safety_class, "claims": claims}
    if caution:
        brief["caution"] = caution
    if idea:
        if idea.get("concept"):
            brief["concept"] = idea["concept"]
        if idea.get("angle"):
            brief["angle"] = idea["angle"]
        if idea.get("hooks"):
            brief["chosen_hook"] = idea["hooks"][0]["text"]
        if idea.get("payoff"):
            brief["payoff"] = idea["payoff"]
    return brief
