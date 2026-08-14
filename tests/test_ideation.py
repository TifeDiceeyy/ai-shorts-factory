import pytest
from shorts_factory.ideation import generate_ideas, load_idea_history, record_idea_chosen
from shorts_factory.safety import TopicBlocked


def test_generate_ideas_returns_five_hooks_each():
    ideas = generate_ideas("roman concrete")
    assert len(ideas) >= 1
    for idea in ideas:
        assert len(idea.hooks) == 5
        assert all("roman concrete" in h.text for h in idea.hooks)


def test_red_topic_blocked_before_any_idea_generated():
    with pytest.raises(TopicBlocked):
        generate_ideas("gunpowder")


def test_ideas_are_ranked_highest_first():
    ideas = generate_ideas("soap")
    scores = [i.rank_score for i in ideas]
    assert scores == sorted(scores, reverse=True)


def test_source_availability_reflects_missing_citation_store():
    # "rope" is a known GREEN topic (passes the safety gate) with no citation
    # store built yet (Phase 1's live retrieval never ran for it).
    ideas = generate_ideas("rope")
    assert ideas[0].source_availability["available"] is False
    assert ideas[0].rank_score < 1.0  # source_ratio term contributes 0


def test_repeated_concept_increases_similarity_to_recent(tmp_path, monkeypatch):
    history_path = tmp_path / "idea_history.json"
    monkeypatch.setattr("shorts_factory.ideation.IDEA_HISTORY_PATH", history_path)

    first_batch = generate_ideas("charcoal")
    assert load_idea_history() == []  # nothing recorded yet

    record_idea_chosen(first_batch[0])
    history = load_idea_history()
    assert len(history) == 1

    second_batch = generate_ideas("charcoal")
    matching = [i for i in second_batch if i.concept == first_batch[0].concept]
    assert matching, "expected the same concept template to reappear for the same topic"
    assert matching[0].similarity_to_recent > 0.5
