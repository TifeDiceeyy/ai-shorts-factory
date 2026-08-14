"""Phase 4: human-approval state per video. Filesystem-JSON, same pattern as
everything else in this repo — no database yet (CLAUDE.md's target
architecture only introduces Postgres after extracting from a proven path).

Publishing stays human-approved: nothing in this module can move a video to
"scheduled" except an explicit approve() call from the dashboard.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["pending", "approved", "rejected", "scheduled"]


@dataclass
class ReviewState:
    status: Status = "pending"
    updated_at: str = ""
    notes: str = ""
    history: list[dict] = field(default_factory=list)


def _path(artifacts_dir: Path) -> Path:
    return artifacts_dir / "review-state.json"


def load(artifacts_dir: Path) -> ReviewState:
    p = _path(artifacts_dir)
    if not p.exists():
        return ReviewState()
    data = json.loads(p.read_text(encoding="utf-8"))
    return ReviewState(**data)


def _save(artifacts_dir: Path, state: ReviewState) -> None:
    _path(artifacts_dir).write_text(
        json.dumps(
            {"status": state.status, "updated_at": state.updated_at, "notes": state.notes, "history": state.history},
            indent=2,
        ),
        encoding="utf-8",
    )


def _transition(artifacts_dir: Path, new_status: Status, notes: str = "") -> ReviewState:
    state = load(artifacts_dir)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state.history.append({"from": state.status, "to": new_status, "at": now, "notes": notes})
    state.status = new_status
    state.updated_at = now
    state.notes = notes
    _save(artifacts_dir, state)
    return state


def approve(artifacts_dir: Path, notes: str = "") -> ReviewState:
    return _transition(artifacts_dir, "approved", notes)


def reject(artifacts_dir: Path, notes: str = "") -> ReviewState:
    return _transition(artifacts_dir, "rejected", notes)


def schedule(artifacts_dir: Path, notes: str = "") -> ReviewState:
    """Only callable meaningfully from an already-approved state — the
    dashboard route enforces this before calling in, this function itself
    also refuses so it can't be bypassed by calling it directly."""
    state = load(artifacts_dir)
    if state.status != "approved":
        raise ValueError(f"cannot schedule from status {state.status!r} — must be 'approved' first")
    return _transition(artifacts_dir, "scheduled", notes)


def reset_to_pending(artifacts_dir: Path, notes: str = "") -> ReviewState:
    return _transition(artifacts_dir, "pending", notes)
