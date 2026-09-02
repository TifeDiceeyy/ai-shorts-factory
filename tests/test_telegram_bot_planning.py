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


def test_length_choices_stay_inside_the_shorts_range():
    """The offered lengths must be ones the pacing was actually tuned for.

    One scene runs ~3s (brief_builder.SECONDS_PER_SCENE), and the duration
    budget was calibrated against real renders in this range. Anything
    longer is untested pacing and proportionally more spend, since video
    generation dominates cost.
    """
    from shorts_factory.telegram_bot import DEFAULT_LENGTH_SECONDS, LENGTH_CHOICES

    seconds = [s for s, _label in LENGTH_CHOICES]
    assert seconds == sorted(seconds), "offer lengths in ascending order"
    assert all(15 <= s <= 180 for s in seconds), "outside the brief schema's range"
    assert DEFAULT_LENGTH_SECONDS in seconds


def test_requested_length_drives_scene_count_and_validation_window():
    """Asking for 30s or 60s has to change BOTH the scene count and the
    window the script is validated against.

    Without moving the window too, a 30s script would be generated and then
    rejected against the fixed 40-50s default — and the caller silently
    falls back to the deterministic stub script, so choosing a length would
    have appeared to do nothing at all.
    """
    import json
    from pathlib import Path

    from shorts_factory.brief_builder import build_brief_from_citations
    from shorts_factory.cost_tracker import CostTracker
    from shorts_factory.providers.llm import StubLLMProvider
    from shorts_factory.schema_validate import (
        script_duration_window,
        validate_script_against_brief,
    )

    store = json.loads(
        Path("data/roman_concrete/roman_concrete.citations.json").read_text(encoding="utf-8")
    )
    counts = {}
    for seconds in (30, 45, 60):
        brief = build_brief_from_citations(
            "roman concrete", store, "green", target_seconds=seconds
        )
        assert brief["target_seconds"] == seconds
        low, high = script_duration_window(brief)
        assert low < seconds < high

        script = StubLLMProvider().generate_script(
            brief, "English", "flat", CostTracker(budget_cap_usd=1)
        )
        validate_script_against_brief(script, brief)
        total = sum(s["duration"] for s in script["scenes"])
        assert low <= total <= high, f"{seconds}s request produced {total:.1f}s"
        counts[seconds] = len(script["scenes"])

    assert counts[30] < counts[45] < counts[60], counts


def test_no_requested_length_keeps_the_original_behaviour():
    """Every existing caller, and every brief written before this field
    existed, must validate against the unchanged 40-50s window."""
    from shorts_factory.schema_validate import (
        SCRIPT_MAX_TOTAL_SECONDS,
        SCRIPT_MIN_TOTAL_SECONDS,
        script_duration_window,
    )

    for brief in ({}, {"target_seconds": None}, {"target_seconds": 0}):
        assert script_duration_window(brief) == (
            SCRIPT_MIN_TOTAL_SECONDS,
            SCRIPT_MAX_TOTAL_SECONDS,
        )


def test_every_route_to_generation_passes_through_the_length_picker():
    """A topic that already has verified sources skips retrieval — and used
    to skip the length picker with it, jumping straight to the confirm
    step. That silently affected MOST topics, since anything retrieved
    before takes that branch (found in real use 2026-09-02).

    Asserted structurally: enter_generate_confirm is reachable only from
    the length-picker callback, never called directly by a routing helper.
    """
    import inspect
    import re

    from shorts_factory import telegram_bot

    source = inspect.getsource(telegram_bot)
    # Calls to enter_generate_confirm, excluding its own definition.
    calls = [
        line.strip()
        for line in source.splitlines()
        if "enter_generate_confirm(" in line and "async def" not in line
    ]
    assert len(calls) == 1, (
        f"enter_generate_confirm should be reached from exactly one place "
        f"(the length callback); found {len(calls)}: {calls}"
    )
    # And that one place must be inside the choosing_length handling.
    picker_block = source.split("choosing_length.state", 1)
    assert len(picker_block) == 2, "the length-picker state handler is missing"
    assert "enter_generate_confirm(" in picker_block[1].split("confirming_generate.state")[0], (
        "the only call must sit in the choosing_length branch"
    )


def test_progress_reporting_can_never_break_a_run():
    """A generation costs real money and takes ~20 minutes. A slip in the
    REPORTING path — a formatting error, a dropped Telegram call — must not
    lose it, so every callback failure is swallowed deliberately."""
    from shorts_factory.pipeline import _progress_reporter

    exploding = _progress_reporter(lambda *args: 1 / 0)
    exploding("Drawing scenes", 1, 15)          # must not raise

    seen = []
    _progress_reporter(lambda s, d, t: seen.append((s, d, t)))("Animating scenes", 4, 9)
    assert seen == [("Animating scenes", 4, 9)]

    _progress_reporter(None)("no callback configured")   # must not raise


def test_every_reported_stage_is_known_to_the_progress_bar():
    """The bar positions itself by finding the stage in PROGRESS_STAGES. A
    stage the pipeline reports but the bar doesn't know silently renders as
    0% for its whole duration."""
    import inspect
    import re

    from shorts_factory import pipeline
    from shorts_factory.telegram_bot import PROGRESS_STAGES

    reported = set(re.findall(r'report\(\s*"([^"]+)"', inspect.getsource(pipeline)))
    assert reported, "no progress calls found in the pipeline"
    unknown = reported - set(PROGRESS_STAGES)
    assert not unknown, f"pipeline reports stages the bar doesn't know: {sorted(unknown)}"


def test_progress_text_shows_stage_scene_and_elapsed():
    from shorts_factory.telegram_bot import _duration, _progress_text

    text = _progress_text("Animating scenes", 4, 9, 626)
    assert "Animating scenes" in text
    assert "5 of 9" in text, "scene counter should be 1-based for a reader"
    assert "10m 26s" in text
    assert "█" in text and "░" in text

    # A stage with no per-item count omits the counter line entirely.
    assert "of" not in _progress_text("Assembling the video", 0, 0, 10).split("\n")[1]

    assert _duration(0) == "0s"
    assert _duration(59) == "59s"
    assert _duration(61) == "1m 01s"


def test_bar_advances_monotonically_through_the_stages():
    """A bar that jumps backwards reads as a failure."""
    from shorts_factory.telegram_bot import PROGRESS_STAGES, _progress_text

    filled = [_progress_text(s, 0, 0, 0).count("█") for s in PROGRESS_STAGES]
    assert filled == sorted(filled), filled
    assert filled[0] == 0 and filled[-1] == 12
