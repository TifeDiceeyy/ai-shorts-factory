"""Phase 5: prove the human-approval gate can't be bypassed by calling
publish_to_youtube() directly, and that it refuses cleanly without OAuth
config — same honest "no stub" pattern as search (providers/search.py)."""
import pytest
from shorts_factory.dashboard import review_state
from shorts_factory.pipeline import REPO_ROOT
from shorts_factory.publish import NotApproved, publish_to_youtube
from shorts_factory.providers.youtube import YouTubeNotConfigured


def test_publish_refuses_unapproved_video():
    artifacts_dir = REPO_ROOT / "artifacts" / "soap"
    review_state.reset_to_pending(artifacts_dir)
    with pytest.raises(NotApproved):
        publish_to_youtube("soap")


def test_publish_refuses_rejected_video():
    artifacts_dir = REPO_ROOT / "artifacts" / "charcoal"
    review_state.reject(artifacts_dir, notes="not good enough")
    with pytest.raises(NotApproved):
        publish_to_youtube("charcoal")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_when_youtube_not_configured():
    artifacts_dir = REPO_ROOT / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    with pytest.raises(YouTubeNotConfigured):
        publish_to_youtube("soap")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_missing_topic():
    with pytest.raises(FileNotFoundError):
        publish_to_youtube("a-topic-that-was-never-rendered-xyz")
