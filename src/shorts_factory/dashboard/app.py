"""Phase 4: review/publish dashboard. "Control Room" design (signed off).

Single-operator review queue: pipeline-stage overview, per-video review
(preview, edit hook/caption, regenerate one scene, inspect sources),
approve/reject, and a schedule action gated behind approval. Publishing
itself (Phase 5, real YouTube upload) isn't connected yet — schedule() marks
intent only, clearly labeled as such, not a fake upload.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import review_state
from ..pipeline import REPO_ROOT, regenerate_scene, run_pipeline

APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"

app = FastAPI(title="AI Shorts Factory — Review Queue")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _stage_for(topic: str, artifacts_dir: Path) -> str:
    if not (artifacts_dir / f"{topic}.script.json").exists():
        return "script"
    if not (artifacts_dir / f"{topic}.mp4").exists():
        return "render"
    vpath = artifacts_dir / "verification-report.json"
    if not vpath.exists():
        return "verify"
    verification = json.loads(vpath.read_text(encoding="utf-8"))
    return "ready" if verification.get("overall_pass") else "verify-failed"


def _list_videos() -> list[dict[str, Any]]:
    if not ARTIFACTS_ROOT.exists():
        return []
    videos = []
    for topic_dir in sorted(ARTIFACTS_ROOT.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue
        topic = topic_dir.name
        stage = _stage_for(topic, topic_dir)
        state = review_state.load(topic_dir)
        cost = None
        cost_path = topic_dir / "cost-report.json"
        if cost_path.exists():
            cost = json.loads(cost_path.read_text(encoding="utf-8")).get("total_spent_usd")
        videos.append({
            "topic": topic,
            "stage": stage,
            "status": state.status,
            "cost": cost,
        })
    return videos


def _load_video(topic: str) -> dict[str, Any]:
    artifacts_dir = ARTIFACTS_ROOT / topic
    if not artifacts_dir.exists():
        raise HTTPException(404, f"no artifacts for topic {topic!r}")

    script_path = artifacts_dir / f"{topic}.script.json"
    script = json.loads(script_path.read_text(encoding="utf-8")) if script_path.exists() else None

    captions_meta_path = artifacts_dir / "captions.meta.json"
    captions_meta = json.loads(captions_meta_path.read_text(encoding="utf-8")) if captions_meta_path.exists() else None

    verification_path = artifacts_dir / "verification-report.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else None

    cost_path = artifacts_dir / "cost-report.json"
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else None

    citations_path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
    citations = json.loads(citations_path.read_text(encoding="utf-8")) if citations_path.exists() else None

    brief_path = REPO_ROOT / "data" / topic / f"{topic}.brief.json"
    brief = json.loads(brief_path.read_text(encoding="utf-8")) if brief_path.exists() else None

    state = review_state.load(artifacts_dir)
    has_video = (artifacts_dir / f"{topic}.mp4").exists()

    return {
        "topic": topic,
        "stage": _stage_for(topic, artifacts_dir),
        "state": state,
        "script": script,
        "captions_meta": captions_meta,
        "verification": verification,
        "cost": cost,
        "citations": citations,
        "brief": brief,
        "has_video": has_video,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"videos": _list_videos()})


@app.get("/video/{topic}", response_class=HTMLResponse)
def video_detail(request: Request, topic: str):
    return templates.TemplateResponse(request, "review.html", _load_video(topic))


@app.get("/video/{topic}/media/{filename}")
def video_media(topic: str, filename: str):
    topic_dir = (ARTIFACTS_ROOT / topic).resolve()
    path = (topic_dir / filename).resolve()
    # Reject path traversal (e.g. "../../../etc/passwd") — the resolved file
    # must land directly inside this topic's own artifacts directory.
    if topic_dir not in path.parents or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.post("/video/{topic}/scene/{scene_index}/regenerate")
def do_regenerate_scene(topic: str, scene_index: int, narration: str = Form(...)):
    try:
        regenerate_scene(topic, scene_index, new_narration=narration or None)
    except (FileNotFoundError, IndexError, ValueError) as e:
        raise HTTPException(400, str(e))
    return RedirectResponse(f"/video/{topic}", status_code=303)


@app.post("/video/{topic}/approve")
def do_approve(topic: str, notes: str = Form("")):
    artifacts_dir = ARTIFACTS_ROOT / topic
    if not artifacts_dir.exists():
        raise HTTPException(404, "not found")
    review_state.approve(artifacts_dir, notes)
    return RedirectResponse(f"/video/{topic}", status_code=303)


@app.post("/video/{topic}/reject")
def do_reject(topic: str, notes: str = Form("")):
    artifacts_dir = ARTIFACTS_ROOT / topic
    if not artifacts_dir.exists():
        raise HTTPException(404, "not found")
    review_state.reject(artifacts_dir, notes)
    return RedirectResponse(f"/video/{topic}", status_code=303)


@app.post("/video/{topic}/schedule")
def do_schedule(topic: str, notes: str = Form("")):
    artifacts_dir = ARTIFACTS_ROOT / topic
    if not artifacts_dir.exists():
        raise HTTPException(404, "not found")
    try:
        review_state.schedule(artifacts_dir, notes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return RedirectResponse(f"/video/{topic}", status_code=303)
