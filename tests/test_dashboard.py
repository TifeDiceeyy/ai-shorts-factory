"""Phase 4: dashboard route tests via FastAPI's TestClient — real HTTP
request/response cycle against the actual app, not a mock."""
import json
import pytest
from fastapi.testclient import TestClient
from shorts_factory.dashboard.app import app
from shorts_factory.dashboard import app as dashboard_app
from shorts_factory.dashboard import review_state

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    for topic in ("soap", "charcoal"):
        topic_dir = root / topic
        topic_dir.mkdir(parents=True)
        (topic_dir / f"{topic}.script.json").write_text(json.dumps({
            "topic": topic,
            "language": "English",
            "visual_style": "test style",
            "scenes": [{
                "narration": "Saponification turns fats and alkali into soap.",
                "caption": "Saponification turns fats and alkali into soap.",
                "duration": 7.6,
                "visual_prompt": "a workshop scene",
                "source_claim_id": "claim-01",
                "camera": "static wide shot",
                "sfx": None,
            }]
        }))
        (topic_dir / f"{topic}.mp4").write_bytes(b"test-video")
        (topic_dir / "verification-report.json").write_text(json.dumps({"overall_pass": True, "checks": []}))
    monkeypatch.setattr(dashboard_app, "ARTIFACTS_ROOT", root)
    yield root


def test_index_lists_known_topics():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "soap" in resp.text
    assert "charcoal" in resp.text


def test_video_detail_shows_stage_and_scenes():
    resp = client.get("/video/soap")
    assert resp.status_code == 200
    assert "soap" in resp.text.lower()
    assert "saponification" in resp.text.lower()  # a real scene's narration text


def test_video_detail_404_for_unknown_topic():
    resp = client.get("/video/definitely-not-a-real-topic-xyz")
    assert resp.status_code == 404


def test_media_route_serves_the_actual_mp4():
    resp = client.get("/video/soap/media/soap.mp4")
    assert resp.status_code == 200
    assert resp.headers["content-type"] in ("video/mp4", "application/octet-stream")


def test_index_and_review_render_when_verification_failed(isolated_artifacts):
    """Regression test: both index.html and review.html computed the
    stage-progress dots/steps via a bare
    ['script','render','verify','ready'].index(stage) — "verify-failed" (the
    real value _stage_for returns whenever a video's verification actually
    fails, see app.py) isn't in that list, so Jinja2's list.index() raised
    ValueError and the page 500'd instead of showing the failure state.
    Confirmed for real: the only existing fixture in this file always wrote
    overall_pass=True, so this path was never exercised before."""
    (isolated_artifacts / "soap" / "verification-report.json").write_text(
        json.dumps({"overall_pass": False, "checks": [{"criterion": "x", "passed": False}]})
    )
    no_raise_client = TestClient(app, raise_server_exceptions=False)

    resp = no_raise_client.get("/")
    assert resp.status_code == 200
    assert "verify-failed" in resp.text

    resp = no_raise_client.get("/video/soap")
    assert resp.status_code == 200
    # review.html doesn't print the raw "verify-failed" stage value as text
    # (only index.html's stage-label does) — it renders the fail state via
    # this CSS class on the "verify" step.
    assert "stage-item--fail" in resp.text


def test_media_route_rejects_path_traversal():
    resp = client.get("/video/soap/media/..%2F..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (404, 400)


def test_approve_reject_schedule_flow(isolated_artifacts):
    artifacts_dir = isolated_artifacts / "soap"
    review_state.reset_to_pending(artifacts_dir)  # clean slate for this test

    resp = client.post("/video/soap/schedule", data={"notes": "too early"})
    assert resp.status_code == 400  # can't schedule before approval

    resp = client.post("/video/soap/approve", data={"notes": "looks good"}, follow_redirects=False)
    assert resp.status_code == 303
    assert review_state.load(artifacts_dir).status == "approved"

    resp = client.post("/video/soap/schedule", data={"notes": "go"}, follow_redirects=False)
    assert resp.status_code == 303
    assert review_state.load(artifacts_dir).status == "scheduled"

    review_state.reset_to_pending(artifacts_dir)  # leave state clean for other tests
