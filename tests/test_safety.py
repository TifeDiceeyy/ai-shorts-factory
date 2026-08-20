import pytest
from shorts_factory.safety import SafetyClass, TopicBlocked, classify_topic, enforce_not_blocked


def test_soap_is_yellow():
    assert classify_topic("soap") == SafetyClass.YELLOW


def test_roman_concrete_is_green():
    assert classify_topic("roman concrete") == SafetyClass.GREEN


def test_gunpowder_is_red():
    assert classify_topic("gunpowder") == SafetyClass.RED


def test_gunpowder_blocked_before_pipeline_touches_anything():
    """Adversarial check (CLAUDE.md rule 6): prove the RED topic is actually
    refused, not just classified."""
    with pytest.raises(TopicBlocked):
        enforce_not_blocked("gunpowder")


def test_how_to_make_gunpowder_phrasing_also_blocked():
    """Red must catch obvious rephrasing, not just the exact table entry."""
    with pytest.raises(TopicBlocked):
        enforce_not_blocked("how to make gunpowder at home")


def test_unknown_topic_fails_closed_to_red():
    """A topic never seen before must NOT default to green/yellow — it must
    be blocked until a real classifier (Phase 1+) proves it safe."""
    assert classify_topic("some totally novel topic nobody classified yet") == SafetyClass.RED
    with pytest.raises(TopicBlocked):
        enforce_not_blocked("some totally novel topic nobody classified yet")


def test_yellow_topic_is_not_blocked():
    enforce_not_blocked("soap")  # must not raise


def test_retrieval_refuses_red_topic():
    from shorts_factory.cost_tracker import CostTracker
    from shorts_factory.providers.search import SearchProvider
    from shorts_factory.retrieval import run_retrieval_for_topic

    class DummySearch(SearchProvider):
        name = "dummy"
        def search(self, query, cost_tracker):
            return []

    with pytest.raises(TopicBlocked):
        run_retrieval_for_topic("gunpowder", DummySearch(), CostTracker(1.0))

