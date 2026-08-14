"""Fetch real YouTube metrics for published experiment-ledger entries."""
from __future__ import annotations

import datetime as dt
import sys

from .config import load_settings
from .experiment_ledger import ingest_metrics, load_ledger
from .providers.youtube import YouTubeNotConfigured
from .providers.youtube_analytics import get_analytics_provider


def ingest_all(end_date: str | None = None) -> list[dict]:
    settings = load_settings()
    if not settings.youtube_configured:
        raise YouTubeNotConfigured("YOUTUBE_CLIENT_SECRETS_FILE is not configured")
    provider = get_analytics_provider(settings.youtube_client_secrets_file, settings.youtube_token_file)
    end = end_date or dt.datetime.now(dt.timezone.utc).date().isoformat()
    updated = []
    for entry in load_ledger():
        published = entry.get("published_at", "")[:10]
        video_id = entry.get("video_id")
        if not video_id or not published:
            continue
        metrics = provider.get_video_metrics(video_id, published, end)
        updated.append(ingest_metrics(video_id, metrics))
    return updated


def main() -> int:
    try:
        updated = ingest_all()
    except Exception as exc:
        print(f"Analytics refused: {exc}", file=sys.stderr)
        return 1
    print(f"Updated {len(updated)} video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
