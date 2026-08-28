"""Tests for brain_integration.py — the adapter that lets run_pipeline try
the local, zero-cost brain knowledge base before falling back to real, paid
Tavily retrieval (user request 2026-08-28: "questions should go through
brain first").

Most tests here use a FAKE brain object (duck-typed: only needs
.research_brief(topic, top_k=...)) rather than the real one, since the real
brain's data (data/brain/, built from two copyrighted books) is gitignored
— present on this development machine, but not guaranteed to exist on a
fresh checkout or CI. Real-brain tests are explicitly skipped when that
data isn't present, so this file stays portable."""
import pytest

from shorts_factory.brain_integration import (
    BRAIN_DATA_DIR,
    BRAIN_MIN_AVG_TOP5_SCORE,
    build_brief_from_brain,
    brain_covers_topic,
    load_brain,
)
from shorts_factory.schema_validate import validate_brief

REAL_BRAIN_AVAILABLE = (BRAIN_DATA_DIR / "chunks.json").exists()


class FakeBrain:
    """Returns a canned research_brief() result regardless of the query —
    lets these tests exercise brain_covers_topic()'s own coverage logic in
    isolation from the real BM25 retriever/book corpus."""

    def __init__(self, key_facts, sources):
        self._key_facts = key_facts
        self._sources = sources

    def research_brief(self, topic, top_k=10):
        return {
            "topic": topic,
            "key_facts": list(self._key_facts),
            "sources": list(self._sources),
            "full_context": "",
            "word_count": 0,
        }


def _sources(scores, book="Test Book"):
    return [{"score": s, "book": book, "pages": "1-2", "text": "x", "matched_terms": []} for s in scores]


GOOD_FACTS = [
    "Roman concrete used volcanic ash mixed with lime to form a durable binder.",
    "The material could set even underwater, unlike many later cements.",
    "Modern Portland cement lacks the same long-term self-healing property.",
    "Ancient builders layered the mixture with aggregate to pour massive domes.",
    "Analysis of harbor structures shows the concrete strengthening over centuries.",
]


def test_brain_covers_topic_true_when_facts_and_score_both_clear_the_bar():
    brain = FakeBrain(GOOD_FACTS, _sources([20.0, 19.0, 18.0, 17.0, 16.0]))
    covered, research = brain_covers_topic(brain, "roman concrete")
    assert covered is True
    assert research is not None
    assert len(research["key_facts"]) == len(GOOD_FACTS)


def test_brain_covers_topic_false_when_score_is_below_the_bar():
    """Confirms the empirically-calibrated score floor actually gates
    something — plenty of facts, but weak (BM25 keyword-overlap-only)
    relevance, matching the real "wifi routers"/"video game consoles"
    case measured against the real brain (both scored well under 10)."""
    brain = FakeBrain(GOOD_FACTS, _sources([5.0, 4.0, 3.0, 2.0, 1.0]))
    covered, research = brain_covers_topic(brain, "some modern topic")
    assert covered is False
    assert research is None


def test_brain_covers_topic_false_when_too_few_facts_survive():
    """Even with a strong score, fewer than MIN_CLAIMS_FOR_BRIEF (4) usable
    facts isn't enough to build a schema-valid brief (minItems: 4)."""
    brain = FakeBrain(GOOD_FACTS[:2], _sources([20.0, 19.0, 18.0]))
    covered, research = brain_covers_topic(brain, "roman concrete")
    assert covered is False
    assert research is None


def test_brain_covers_topic_filters_noisy_and_truncated_facts_before_counting():
    """Real bug found against the real brain (2026-08-28): a "furnace"
    query returned a fact truncated mid-sentence at a chunk boundary
    ("...is a long, hot, d") and another with an inline "(page 158)"
    citation artifact. Both must be dropped BEFORE the minimum-fact-count
    check, not after — otherwise a topic could look "covered" while half
    its claims are unusable fragments."""
    facts = GOOD_FACTS[:3] + [
        "This fact cites a page reference (page 158) inline.",
        "This one just trails off mid-sentence without any real en",
    ]
    brain = FakeBrain(facts, _sources([20.0, 19.0, 18.0, 17.0, 16.0]))
    covered, research = brain_covers_topic(brain, "roman concrete")
    assert covered is False  # only 3 clean facts survive, below MIN_CLAIMS_FOR_BRIEF
    assert research is None


def test_build_brief_from_brain_is_schema_valid_and_cites_the_books():
    brain = FakeBrain(GOOD_FACTS, _sources([20.0, 19.0], book="The Knowledge"))
    covered, research = brain_covers_topic(brain, "roman concrete")
    assert covered is True

    brief = build_brief_from_brain("roman concrete", research, "green")
    validate_brief(brief)
    assert brief["topic"] == "roman concrete"
    assert brief["safety_class"] == "green"
    assert 4 <= len(brief["claims"]) <= 6
    for claim in brief["claims"]:
        assert claim["claim"] in GOOD_FACTS
        assert "The Knowledge" in claim["source"]
        assert "local knowledge base" in claim["source"]


def test_build_brief_from_brain_carries_caution_and_idea_through():
    brain = FakeBrain(GOOD_FACTS, _sources([20.0, 19.0]))
    _covered, research = brain_covers_topic(brain, "roman concrete")
    idea = {
        "concept": "The lost self-healing concrete",
        "angle": "myth-busting historical explainer",
        "hooks": [{"text": "Rome's concrete outlives ours — here's why."}],
        "payoff": "Viewer learns the real mechanism.",
    }
    brief = build_brief_from_brain(
        "roman concrete", research, "yellow", caution="Handle lime safely.", idea=idea,
    )
    validate_brief(brief)
    assert brief["caution"] == "Handle lime safely."
    assert brief["concept"] == idea["concept"]
    assert brief["angle"] == idea["angle"]
    assert brief["chosen_hook"] == idea["hooks"][0]["text"]
    assert brief["payoff"] == idea["payoff"]


def test_load_brain_returns_none_when_data_dir_is_missing(tmp_path, monkeypatch):
    """A missing brain (the common case on a fresh checkout — data/brain/
    is gitignored) must be a quiet, handled state, not a crash — the
    pipeline needs to fall through to citations.json in that case."""
    import shorts_factory.brain_integration as bi

    monkeypatch.setattr(bi, "BRAIN_DATA_DIR", tmp_path / "nonexistent")
    assert load_brain() is None


@pytest.mark.skipif(not REAL_BRAIN_AVAILABLE, reason="data/brain/ not built on this machine (gitignored)")
def test_real_brain_covers_a_well_known_topic_for_real():
    """One real, zero-cost (fully local) integration check against the
    actual book corpus — confirms the whole real path (BM25 retrieval,
    fact extraction, noise filtering, score threshold) produces a usable,
    schema-valid brief for a topic genuinely covered by the source books."""
    brain = load_brain()
    assert brain is not None
    covered, research = brain_covers_topic(brain, "roman concrete")
    assert covered is True
    brief = build_brief_from_brain("roman concrete", research, "green")
    validate_brief(brief)
    assert BRAIN_MIN_AVG_TOP5_SCORE > 0  # sanity: the constant is actually a positive bar


@pytest.mark.skipif(not REAL_BRAIN_AVAILABLE, reason="data/brain/ not built on this machine (gitignored)")
def test_real_brain_does_not_cover_a_clearly_unrelated_topic():
    brain = load_brain()
    assert brain is not None
    covered, research = brain_covers_topic(brain, "wifi routers")
    assert covered is False
