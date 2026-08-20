import pytest
from shorts_factory.safety import SafetyClass, TopicBlocked, classify_topic, enforce_not_blocked
from shorts_factory.topic_registry import get_topic, load_registry, register_topic


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("shorts_factory.topic_registry.REGISTRY_PATH", tmp_path / "topic_registry.json")


def test_seed_registry_is_written_on_first_load():
    registry = load_registry()
    assert registry["soap"]["safety_class"] == "yellow"
    assert registry["roman concrete"]["safety_class"] == "green"


def test_seed_registry_preserves_existing_safety_classification():
    # These must keep classifying exactly as they did before the registry
    # refactor (safety.py used to hardcode these directly).
    assert classify_topic("soap") == SafetyClass.YELLOW
    assert classify_topic("roman concrete") == SafetyClass.GREEN
    assert classify_topic("gunpowder") == SafetyClass.RED
    assert classify_topic("some totally novel topic") == SafetyClass.RED


def test_register_topic_persists_and_is_immediately_visible():
    register_topic("candle making", ["candle history"], ["candle", "wax"], "green")
    entry = get_topic("candle making")
    assert entry == {
        "queries": ["candle history"],
        "keywords": ["candle", "wax"],
        "safety_class": "green",
        "caution": None,
    }
    # Immediately visible to safety.classify_topic in the same process,
    # without needing a restart.
    assert classify_topic("candle making") == SafetyClass.GREEN
    enforce_not_blocked("candle making")  # must not raise


def test_register_topic_refuses_red():
    with pytest.raises(ValueError):
        register_topic("gunpowder", ["q"], ["k"], "red")
    assert get_topic("gunpowder") is None


def test_register_topic_normalizes_name():
    register_topic("  Candle MAKING  ", ["q"], ["k"], "yellow", caution="hot wax")
    assert get_topic("candle making")["safety_class"] == "yellow"
    assert get_topic("  CANDLE making")["safety_class"] == "yellow"


def test_unregistered_topic_still_fails_closed():
    with pytest.raises(TopicBlocked):
        enforce_not_blocked("candle making")  # not registered in this test's isolated registry
