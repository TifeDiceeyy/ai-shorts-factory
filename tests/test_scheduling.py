"""Phase 7: prove the autonomy gate is structurally inert right now, and
that it would correctly permanently-gate non-green topics even with enough
reviewed videos."""
import pytest
from shorts_factory.scheduling import AutonomyNotEligible, attempt_autonomous_run, is_autonomy_eligible


def test_currently_zero_reviewed_videos_blocks_every_topic():
    # Real state, not mocked: as of this test, no video has actually been
    # published (Phase 5 was built but never live-run), so this must be 0.
    eligible, reason = is_autonomy_eligible("roman concrete")
    assert eligible is False
    assert "human-reviewed" in reason


def test_attempt_autonomous_run_raises_without_touching_pipeline():
    with pytest.raises(AutonomyNotEligible):
        attempt_autonomous_run("roman concrete")


def test_yellow_topic_permanently_gated_even_with_enough_reviews(monkeypatch):
    monkeypatch.setattr("shorts_factory.scheduling.human_reviewed_count", lambda: 50)
    eligible, reason = is_autonomy_eligible("soap")  # soap is yellow
    assert eligible is False
    assert "human-gated permanently" in reason


def test_red_topic_permanently_gated_even_with_enough_reviews(monkeypatch):
    monkeypatch.setattr("shorts_factory.scheduling.human_reviewed_count", lambda: 50)
    eligible, reason = is_autonomy_eligible("gunpowder")
    assert eligible is False


def test_green_topic_eligible_once_threshold_met(monkeypatch):
    monkeypatch.setattr("shorts_factory.scheduling.human_reviewed_count", lambda: 30)
    eligible, reason = is_autonomy_eligible("roman concrete")  # green
    assert eligible is True


def test_green_topic_still_blocked_just_under_threshold(monkeypatch):
    monkeypatch.setattr("shorts_factory.scheduling.human_reviewed_count", lambda: 29)
    eligible, reason = is_autonomy_eligible("roman concrete")
    assert eligible is False
