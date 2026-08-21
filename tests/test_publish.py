"""Phase 5: prove the human-approval gate can't be bypassed by calling
publish_to_youtube() directly, and that it refuses cleanly without OAuth
config — same honest "no stub" pattern as search (providers/search.py)."""
import json

import pytest
from shorts_factory.dashboard import review_state
import shorts_factory.publish as publish_module
from shorts_factory.publish import NotApproved, VerificationFailed, publish_to_youtube
from shorts_factory.providers.youtube import YouTubeNotConfigured

@pytest.fixture(autouse=True)
def isolated_repo(tmp_path, monkeypatch):
    for topic in ("soap", "charcoal"):
        (tmp_path / "artifacts" / topic).mkdir(parents=True)
    monkeypatch.setattr(publish_module, "REPO_ROOT", tmp_path)
    return tmp_path


def _write_passing_verification(artifacts_dir):
    (artifacts_dir / "verification-report.json").write_text(
        json.dumps({"overall_pass": True, "checks": []}), encoding="utf-8"
    )


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


def test_publish_refuses_when_verification_report_missing(isolated_repo):
    """Regression test: an approved video with no verification-report.json
    at all (e.g. approval happened before verification ran) used to sail
    straight through to the upload attempt — approved status alone was
    trusted, never cross-checked against whether the render actually
    passed (confirmed real 2026-08-21 review)."""
    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    with pytest.raises(VerificationFailed):
        publish_to_youtube("soap")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_when_verification_failed(isolated_repo):
    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    (artifacts_dir / "verification-report.json").write_text(
        json.dumps({"overall_pass": False, "checks": [{"criterion": "x", "passed": False}]}), encoding="utf-8"
    )
    with pytest.raises(VerificationFailed):
        publish_to_youtube("soap")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_when_youtube_not_configured(isolated_repo):
    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    _write_passing_verification(artifacts_dir)
    with pytest.raises(YouTubeNotConfigured):
        publish_to_youtube("soap")
    review_state.reset_to_pending(artifacts_dir)


def test_publish_refuses_missing_topic():
    with pytest.raises(FileNotFoundError):
        publish_to_youtube("a-topic-that-was-never-rendered-xyz")


def test_youtube_upload_raises_when_disclosure_not_confirmed(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from shorts_factory.providers.youtube import DisclosureNotConfirmed, YouTubeUploadProvider

    fake_secrets = tmp_path / "secrets.json"
    fake_secrets.write_text("{}", encoding="utf-8")
    provider = YouTubeUploadProvider(client_secrets_file=str(fake_secrets), token_file=str(tmp_path / "token.json"))
    mock_service = MagicMock()
    monkeypatch.setattr(provider, "_get_service", lambda: mock_service)

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"dummy mp4")

    # Mock insert request & response
    mock_insert_request = MagicMock()
    mock_insert_request.execute.return_value = {"id": "vid123", "status": {"privacyStatus": "private"}}
    mock_service.videos().insert.return_value = mock_insert_request

    # Mock list request & confirmation response with missing containsSyntheticMedia
    mock_list_request = MagicMock()
    mock_list_request.execute.return_value = {
        "items": [{"id": "vid123", "status": {"containsSyntheticMedia": False}}]
    }
    mock_service.videos().list.return_value = mock_list_request

    with pytest.raises(DisclosureNotConfirmed) as exc_info:
        provider.upload_video(fake_video, "title", "desc", contains_synthetic_media=True)

    assert exc_info.value.video_id == "vid123"

