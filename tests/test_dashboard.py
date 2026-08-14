"""Phase 4: dashboard route tests via FastAPI's TestClient — real HTTP
request/response cycle against the actual app, not a mock."""
from fastapi.testclient import TestClient
from shorts_factory.dashboard.app import app
from shorts_factory.dashboard import review_state
from shorts_factory.pipeline import REPO_ROOT

client = TestClient(app)


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


def test_media_route_rejects_path_traversal():
    resp = client.get("/video/soap/media/..%2F..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (404, 400)


def test_approve_reject_schedule_flow():
    artifacts_dir = REPO_ROOT / "artifacts" / "soap"
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
