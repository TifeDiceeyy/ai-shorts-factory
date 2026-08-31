import pytest
from shorts_factory.telegram_bot import TelegramController


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr("shorts_factory.topic_registry.REGISTRY_PATH", tmp_path / "topic_registry.json")


@pytest.fixture
def controller(tmp_path):
    return TelegramController((1,), tmp_path)


def test_topic_status_registered(controller):
    assert controller.topic_status("soap") == {"state": "registered", "topic": "soap", "safety_class": "yellow"}


def test_topic_status_red_by_keyword(controller):
    assert controller.topic_status("how to make gunpowder") == {"state": "red", "topic": "how to make gunpowder"}


def test_topic_status_unknown(controller):
    assert controller.topic_status("candle making") == {"state": "unknown", "topic": "candle making"}


def test_propose_new_topic_uses_stub_llm_and_returns_a_plan(controller):
    proposal = controller.propose_new_topic("candle making")
    assert proposal["topic"] == "candle making"
    assert proposal["safety_class"] in ("green", "yellow", "red")
    assert proposal["queries"]
    assert proposal["keywords"]


def test_propose_new_topic_overrides_llm_for_red_keyword_topics(controller, monkeypatch):
    # Force the stub LLM to (wrongly) call this green, to prove the bot's own
    # RED_KEYWORDS re-check overrides it rather than trusting the LLM alone.
    from shorts_factory.providers.llm import StubLLMProvider

    def fake_propose_topic(self, topic, cost_tracker):
        return {"safety_class": "green", "reasoning": "stub", "queries": ["q"], "keywords": ["k"], "caution": None}

    monkeypatch.setattr(StubLLMProvider, "propose_topic", fake_propose_topic)
    proposal = controller.propose_new_topic("homemade gunpowder recipe")
    assert proposal["safety_class"] == "red"


def test_confirm_new_topic_registers_green_or_yellow(controller):
    proposal = {
        "topic": "candle making",
        "safety_class": "green",
        "queries": ["candle history"],
        "keywords": ["candle", "wax"],
        "caution": None,
    }
    reply = controller.confirm_new_topic(proposal)
    assert reply == "Registered 'candle making' as GREEN."
    assert controller.topic_status("candle making")["state"] == "registered"


def test_confirm_new_topic_refuses_red_proposal(controller):
    proposal = {"topic": "candle making", "safety_class": "red", "queries": [], "keywords": [], "caution": None}
    with pytest.raises(ValueError):
        controller.confirm_new_topic(proposal)
    assert controller.topic_status("candle making")["state"] == "unknown"


def test_needs_retrieval_true_when_no_citation_store(controller):
    assert controller.needs_retrieval("a topic with definitely no citations yet") is True


def test_needs_retrieval_false_when_citation_store_exists(controller, tmp_path, monkeypatch):
    monkeypatch.setattr("shorts_factory.telegram_bot.REPO_ROOT", tmp_path)
    citation_dir = tmp_path / "data" / "candle_making"
    citation_dir.mkdir(parents=True)
    (citation_dir / "candle_making.citations.json").write_text("{}")
    assert controller.needs_retrieval("candle making") is False


def test_needs_retrieval_false_when_the_brain_covers_the_topic(controller, monkeypatch):
    """Real gap found 2026-08-29 via a live Telegram test: this gate only
    ever checked for a citations.json file, so the bot kept prompting "Run
    retrieval now?" even for topics the brain already covers for free —
    out of sync with run_pipeline's own brain-first check (pipeline.py)."""
    import shorts_factory.brain_integration as brain_integration

    monkeypatch.setattr(brain_integration, "load_brain", lambda: object())
    monkeypatch.setattr(brain_integration, "brain_covers_topic", lambda brain, topic: (True, {"key_facts": ["x"]}))
    assert controller.needs_retrieval("a topic with definitely no citations yet") is False


def test_needs_retrieval_true_when_brain_does_not_cover_and_no_citations(controller, monkeypatch):
    import shorts_factory.brain_integration as brain_integration

    monkeypatch.setattr(brain_integration, "load_brain", lambda: object())
    monkeypatch.setattr(brain_integration, "brain_covers_topic", lambda brain, topic: (False, None))
    assert controller.needs_retrieval("a topic with definitely no citations yet") is True


def test_needs_retrieval_true_when_brain_is_not_built_and_no_citations(controller, monkeypatch):
    import shorts_factory.brain_integration as brain_integration

    monkeypatch.setattr(brain_integration, "load_brain", lambda: None)
    assert controller.needs_retrieval("a topic with definitely no citations yet") is True
