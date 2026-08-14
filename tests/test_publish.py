"""Phase 5: prove the human-approval gate can't be bypassed by calling
publish_to_youtube() directly, and that it refuses cleanly without OAuth
config — same honest "no stub" pattern as search (providers/search.py)."""
import pytest
from shorts_factory.dashboard import review_state
import shorts_factory.publish as publish_module
from shorts_factory.publish import NotApproved, publish_to_youtube
from shorts_factory.providers.youtube import YouTubeNotConfigured

@pytest.fixture(autouse=True)
def isolated_repo(tmp_path, monkeypatch):
    for topic in ("soap", "charcoal"):
        (tmp_path / "artifacts" / topic).mkdir(parents=True)
    monkeypatch.setattr(publish_module, "REPO_ROOT", tmp_path)
    return tmp_path


def test_publish_refuses_unapproved_video(isolated_repo):
    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.reset_to_pending(artifacts_dir)
    with pytest.raises(NotApproved):
        publish_to_youtube("soap")


def test_publish_refuses_rejected_video(isolated_repo):
    artifacts_dir = isolated_repo / "artifacts" / "charcoal"
    review_state.reject(artifacts_dir, notes="not good enough")
    with pytest.raises(NotApproved):
        publish_to_youtube("charcoal")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_when_youtube_not_configured(isolated_repo):
    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    with pytest.raises(YouTubeNotConfigured):
        publish_to_youtube("soap")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_missing_topic():
    with pytest.raises(FileNotFoundError):
        publish_to_youtube("a-topic-that-was-never-rendered-xyz")
