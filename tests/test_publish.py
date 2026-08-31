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
    """The insert response itself is the source of truth (see the real bug
    below) — an insert response missing containsSyntheticMedia entirely
    must still refuse, exactly as if a confirmation read had reported it
    missing."""
    from unittest.mock import MagicMock
    from shorts_factory.providers.youtube import DisclosureNotConfirmed, YouTubeUploadProvider

    fake_secrets = tmp_path / "secrets.json"
    fake_secrets.write_text("{}", encoding="utf-8")
    provider = YouTubeUploadProvider(client_secrets_file=str(fake_secrets), token_file=str(tmp_path / "token.json"))
    mock_service = MagicMock()
    monkeypatch.setattr(provider, "_get_service", lambda: mock_service)

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"dummy mp4")

    mock_insert_request = MagicMock()
    mock_insert_request.execute.return_value = {"id": "vid123", "status": {"privacyStatus": "private"}}
    mock_service.videos().insert.return_value = mock_insert_request

    with pytest.raises(DisclosureNotConfirmed) as exc_info:
        provider.upload_video(fake_video, "title", "desc", contains_synthetic_media=True)

    assert exc_info.value.video_id == "vid123"


def test_youtube_upload_confirms_disclosure_directly_from_the_insert_response(tmp_path, monkeypatch):
    """Real bug found 2026-08-30 against a live YouTube account: a separate
    videos().list() call made immediately after videos().insert() to
    "confirm" containsSyntheticMedia unreliably returned a status missing
    the field entirely — an eventual-consistency lag on YouTube's own
    backend, verified directly against the live API (insert's own response
    echoed containsSyntheticMedia=true correctly and immediately; the very
    next list() call for the same video did not). Confirming from the
    insert response itself avoids this race — and this test proves the
    fix never even calls list() at all."""
    from unittest.mock import MagicMock
    from shorts_factory.providers.youtube import YouTubeUploadProvider

    fake_secrets = tmp_path / "secrets.json"
    fake_secrets.write_text("{}", encoding="utf-8")
    provider = YouTubeUploadProvider(client_secrets_file=str(fake_secrets), token_file=str(tmp_path / "token.json"))
    mock_service = MagicMock()
    monkeypatch.setattr(provider, "_get_service", lambda: mock_service)

    fake_video = tmp_path / "video.mp4"
    fake_video.write_bytes(b"dummy mp4")

    mock_insert_request = MagicMock()
    mock_insert_request.execute.return_value = {
        "id": "vid456",
        "status": {"privacyStatus": "private", "containsSyntheticMedia": True},
    }
    mock_service.videos().insert.return_value = mock_insert_request
    # Deliberately wired to look "unconfirmed" if it were ever consulted —
    # proves the fix doesn't depend on this call succeeding or agreeing.
    mock_service.videos().list.return_value.execute.return_value = {
        "items": [{"id": "vid456", "status": {}}]
    }

    result = provider.upload_video(fake_video, "title", "desc", contains_synthetic_media=True)

    assert result.video_id == "vid456"
    assert result.contains_synthetic_media is True
    mock_service.videos().list.assert_not_called()


def test_topic_tags_includes_topic_and_registered_keywords(monkeypatch):
    from shorts_factory import topic_registry

    monkeypatch.setattr(
        topic_registry, "load_registry",
        lambda: {"candle making": {"queries": ["q"], "keywords": ["candle", "Wax"], "safety_class": "green"}},
    )
    tags = publish_module._topic_tags("candle making")
    assert tags[0] == "candle making"
    assert "candle" in tags
    assert "wax" in tags  # lowercased, same as the topic itself


def test_topic_tags_handles_a_topic_with_no_registry_entry(monkeypatch):
    from shorts_factory import topic_registry

    monkeypatch.setattr(topic_registry, "load_registry", lambda: {})
    assert publish_module._topic_tags("soap") == ["soap"]


def test_description_hashtags_always_includes_shorts_and_camel_cases_multiword_tags(monkeypatch):
    from shorts_factory import topic_registry

    monkeypatch.setattr(
        topic_registry, "load_registry",
        lambda: {"water filtration": {"queries": [], "keywords": ["slaked lime"], "safety_class": "green"}},
    )
    hashtags = publish_module._description_hashtags("water filtration")
    assert "#Shorts" in hashtags
    assert "#WaterFiltration" in hashtags
    assert "#SlakedLime" in hashtags


def test_publish_passes_topic_tags_and_hashtags_through_to_the_real_upload_call(isolated_repo, monkeypatch):
    """Real gap fixed 2026-08-29: publish_to_youtube() always sent an empty
    tags=[] to the YouTube API and never added any hashtags to the
    description at all — nothing tied the upload back to its own topic."""
    from types import SimpleNamespace
    from shorts_factory import topic_registry
    from shorts_factory.providers.youtube import UploadResult

    monkeypatch.setattr(
        topic_registry, "load_registry",
        lambda: {"soap": {"queries": [], "keywords": ["lye", "fat"], "safety_class": "yellow"}},
    )

    artifacts_dir = isolated_repo / "artifacts" / "soap"
    review_state.approve(artifacts_dir, notes="test")
    _write_passing_verification(artifacts_dir)
    (artifacts_dir / "soap.script.json").write_text(
        json.dumps({"scenes": [{"caption": "How soap is made", "narration": "Soap starts with fat and lye."}]}),
        encoding="utf-8",
    )
    (artifacts_dir / "soap.mp4").write_bytes(b"fake mp4 bytes")

    monkeypatch.setattr(
        publish_module, "load_settings",
        lambda: SimpleNamespace(
            youtube_configured=True, youtube_client_secrets_file="secrets.json", youtube_token_file="token.json",
        ),
    )
    monkeypatch.setattr(publish_module, "record_publish", lambda **kwargs: None)

    captured: dict = {}

    class FakeProvider:
        def upload_video(self, **kwargs):
            captured.update(kwargs)
            return UploadResult(video_id="vid1", privacy_status="private", contains_synthetic_media=True, raw_response={})

    monkeypatch.setattr(publish_module, "get_youtube_provider", lambda *a, **k: FakeProvider())

    result = publish_to_youtube("soap")

    assert result["video_id"] == "vid1"
    assert captured["tags"] == ["soap", "lye", "fat"]
    assert "#Shorts" in captured["description"]
    assert "#Soap" in captured["description"]
    review_state.reset_to_pending(artifacts_dir)

