import pytest
from shorts_factory.dashboard import review_state


def test_default_state_is_pending(tmp_path):
    state = review_state.load(tmp_path)
    assert state.status == "pending"


def test_approve_then_schedule(tmp_path):
    review_state.approve(tmp_path, notes="looks good")
    state = review_state.schedule(tmp_path)
    assert state.status == "scheduled"
    assert len(state.history) == 2


def test_cannot_schedule_without_approval_first():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            review_state.schedule(Path(d))


def test_reject_records_notes(tmp_path):
    state = review_state.reject(tmp_path, notes="hook is weak")
    assert state.status == "rejected"
    assert state.notes == "hook is weak"


def test_state_persists_across_loads(tmp_path):
    review_state.approve(tmp_path)
    reloaded = review_state.load(tmp_path)
    assert reloaded.status == "approved"
