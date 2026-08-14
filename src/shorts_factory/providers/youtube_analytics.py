"""Phase 6: YouTube Analytics ingestion.

Same OAuth token as Phase 5's upload (SCOPES in providers/youtube.py already
covers yt-analytics.readonly). No stub, same reasoning as search and upload:
there is no meaningful fake retention/subscriber data, and Phase 6's whole
point is real data landing per video.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .youtube import API_VERSION, YouTubeNotConfigured, get_credentials

ANALYTICS_SERVICE_NAME = "youtubeAnalytics"
ANALYTICS_API_VERSION = "v2"

METRICS = [
    "views",
    "estimatedMinutesWatched",
    "averageViewDuration",
    "averageViewPercentage",
    "subscribersGained",
    "subscribersLost",
]


@dataclass
class VideoMetrics:
    video_id: str
    start_date: str
    end_date: str
    views: int
    estimated_minutes_watched: float
    average_view_duration_seconds: float
    average_view_percentage: float
    subscribers_gained: int
    subscribers_lost: int
    raw_response: dict[str, Any]
    retention_curve: list[dict[str, float]] = field(default_factory=list)


class YouTubeAnalyticsProvider:
    name = "youtube_analytics"

    def __init__(self, client_secrets_file: str, token_file: str):
        if not client_secrets_file:
            raise YouTubeNotConfigured("YOUTUBE_CLIENT_SECRETS_FILE not set — Phase 6 analytics not configured.")
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self._service = None

    def _get_service(self):
        if self._service is None:
            from googleapiclient.discovery import build
            creds = get_credentials(self.client_secrets_file, self.token_file)
            self._service = build(ANALYTICS_SERVICE_NAME, ANALYTICS_API_VERSION, credentials=creds)
        return self._service

    def get_video_metrics(self, video_id: str, start_date: str, end_date: str) -> VideoMetrics:
        """start_date/end_date as 'YYYY-MM-DD'."""
        service = self._get_service()
        response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=",".join(METRICS),
            filters=f"video=={video_id}",
        ).execute()

        rows = response.get("rows") or []
        if not rows:
            row = [0] * len(METRICS)
        else:
            row = rows[0]
        values = dict(zip(METRICS, row))

        retention_response = service.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            dimensions="elapsedVideoTimeRatio",
            metrics="audienceWatchRatio",
            filters=f"video=={video_id}",
            sort="elapsedVideoTimeRatio",
        ).execute()
        retention_curve = [
            {"elapsed_video_time_ratio": float(point[0]), "audience_watch_ratio": float(point[1])}
            for point in (retention_response.get("rows") or [])
        ]

        return VideoMetrics(
            video_id=video_id,
            start_date=start_date,
            end_date=end_date,
            views=int(values.get("views", 0)),
            estimated_minutes_watched=float(values.get("estimatedMinutesWatched", 0)),
            average_view_duration_seconds=float(values.get("averageViewDuration", 0)),
            average_view_percentage=float(values.get("averageViewPercentage", 0)),
            subscribers_gained=int(values.get("subscribersGained", 0)),
            subscribers_lost=int(values.get("subscribersLost", 0)),
            raw_response=response,
            retention_curve=retention_curve,
        )


def get_analytics_provider(client_secrets_file: str, token_file: str) -> YouTubeAnalyticsProvider:
    return YouTubeAnalyticsProvider(client_secrets_file, token_file)
