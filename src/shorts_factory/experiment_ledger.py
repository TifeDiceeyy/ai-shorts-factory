"""Phase 6: experiment ledger + rule-based scoring.

No ML — CLAUDE.md is explicit: "No ML until enough videos exist." The score
is a documented, simple weighted formula anyone can reproduce by hand:

    score = 0.7 * (average_view_percentage / 100) + 0.3 * min(1, subscribers_gained / views * 100)

Retention (average_view_percentage) is weighted higher than sub-conversion
because for a Shorts-style series, "did people watch it" is the more
reliable, less noisy signal at low view counts — sub counts can be zero for
many perfectly fine videos. Revisit this weighting once real data exists;
right now it's a starting assumption, not a fitted model.
"""
from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .providers.youtube_analytics import VideoMetrics

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "data" / "experiment_ledger.json"

RETENTION_WEIGHT = 0.7
SUB_CONVERSION_WEIGHT = 0.3


def compute_score(metrics: VideoMetrics) -> float:
    retention_component = metrics.average_view_percentage / 100.0
    sub_rate = (metrics.subscribers_gained / metrics.views) if metrics.views else 0.0
    sub_component = min(1.0, sub_rate * 100)
    return round(RETENTION_WEIGHT * retention_component + SUB_CONVERSION_WEIGHT * sub_component, 4)


@dataclass
class ExperimentEntry:
    topic: str
    video_id: str
    concept: str
    hook_variant_index: int
    published_at: str
    series: str | None = None
    metrics: dict[str, Any] | None = None
    score: float | None = None


def load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _save(entries: list[dict]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def record_publish(
    topic: str,
    video_id: str,
    concept: str,
    hook_variant_index: int,
    series: str | None = None,
) -> dict:
    entries = load_ledger()
    entry = ExperimentEntry(
        topic=topic,
        video_id=video_id,
        concept=concept,
        hook_variant_index=hook_variant_index,
        published_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        series=series,
    )
    entries.append(asdict(entry))
    _save(entries)
    return entries[-1]


def ingest_metrics(video_id: str, metrics: VideoMetrics) -> dict:
    entries = load_ledger()
    for entry in entries:
        if entry["video_id"] == video_id:
            entry["metrics"] = {
                "views": metrics.views,
                "estimated_minutes_watched": metrics.estimated_minutes_watched,
                "average_view_duration_seconds": metrics.average_view_duration_seconds,
                "average_view_percentage": metrics.average_view_percentage,
                "subscribers_gained": metrics.subscribers_gained,
                "subscribers_lost": metrics.subscribers_lost,
            }
            entry["score"] = compute_score(metrics)
            _save(entries)
            return entry
    raise ValueError(f"no ledger entry for video_id={video_id!r} — call record_publish() first")


def human_reviewed_count() -> int:
    """Count of videos that have gone through an actual human decision
    (approved or rejected) — this is what Phase 7's 30-50 video gate counts,
    not just "rendered" or "published"."""
    from .dashboard import review_state

    artifacts_root = REPO_ROOT / "artifacts"
    if not artifacts_root.exists():
        return 0
    count = 0
    for topic_dir in artifacts_root.iterdir():
        if not topic_dir.is_dir():
            continue
        state = review_state.load(topic_dir)
        if state.status in ("approved", "rejected", "scheduled"):
            count += 1
    return count
