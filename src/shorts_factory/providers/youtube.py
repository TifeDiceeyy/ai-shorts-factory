"""Phase 5: YouTube upload + synthetic-content disclosure.

No stub — same reasoning as search (providers/search.py): a fake upload
can't prove anything real, and "disclosure flag confirmed via API response"
is meaningless without a real API response to check. This refuses to run
without real OAuth credentials.

Field confirmed against live Google documentation (2026-08-14), not assumed:
`status.containsSyntheticMedia` was added to the YouTube Data API v3 on
2024-10-30 specifically for declaring realistic altered/synthetic content.
Re-verify against https://developers.google.com/youtube/v3/revision_history
before relying on this if it's been a while — CLAUDE.md's own rule: YouTube
API shapes change, confirm against live docs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/yt-analytics.readonly",  # Phase 6
]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

# Applies to realistic Altered/Synthetic content: making a real person appear
# to say/do something they didn't, altering footage of a real event/place,
# or generating a realistic scene that didn't occur. Our content is
# synthesized narration + AI-generated imagery depicting historical/how-to
# scenes — always disclose, never leave this False by default.
DEFAULT_CONTAINS_SYNTHETIC_MEDIA = True


class YouTubeNotConfigured(Exception):
    pass


class DisclosureNotConfirmed(Exception):
    """Raised if the API response doesn't echo back containsSyntheticMedia as
    requested — CLAUDE.md requires this be CONFIRMED from the response, not
    assumed from the request succeeding."""


@dataclass
class UploadResult:
    video_id: str
    privacy_status: str
    contains_synthetic_media: bool
    raw_response: dict[str, Any]


def require_client_secrets(client_secrets_file: str) -> None:
    if not client_secrets_file or not Path(client_secrets_file).exists():
        raise YouTubeNotConfigured(
            f"YOUTUBE_CLIENT_SECRETS_FILE not found: {client_secrets_file!r}. "
            "Create an OAuth client (Desktop app type) in Google Cloud Console, "
            "enable the YouTube Data API v3, and download the client secrets JSON."
        )


def get_credentials(client_secrets_file: str, token_file: str):
    """Shared OAuth flow for both upload (Phase 5) and analytics (Phase 6) —
    SCOPES above covers both, so one consent grant serves both phases."""
    require_client_secrets(client_secrets_file)
    from google.auth.transport.requests import Request as GoogleRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    token_path = Path(token_file)
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleRequest())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


class YouTubeUploadProvider:
    name = "youtube"

    def __init__(self, client_secrets_file: str, token_file: str):
        require_client_secrets(client_secrets_file)
        self.client_secrets_file = client_secrets_file
        self.token_file = token_file
        self._service = None

    def _get_service(self):
        if self._service is None:
            from googleapiclient.discovery import build
            creds = get_credentials(self.client_secrets_file, self.token_file)
            self._service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)
        return self._service

    def upload_video(
        self,
        mp4_path: Path,
        title: str,
        description: str,
        tags: list[str] | None = None,
        category_id: str = "27",  # Education
        privacy_status: str = "private",
        contains_synthetic_media: bool = DEFAULT_CONTAINS_SYNTHETIC_MEDIA,
    ) -> UploadResult:
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
                "containsSyntheticMedia": contains_synthetic_media,
            },
        }
        media = MediaFileUpload(str(mp4_path), chunksize=-1, resumable=True, mimetype="video/mp4")
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()

        actual_flag = response.get("status", {}).get("containsSyntheticMedia")
        if actual_flag != contains_synthetic_media:
            raise DisclosureNotConfirmed(
                f"requested containsSyntheticMedia={contains_synthetic_media}, but the API "
                f"response reports {actual_flag!r} — upload succeeded but disclosure is NOT "
                f"confirmed. Do not treat this video as compliantly disclosed."
            )

        return UploadResult(
            video_id=response["id"],
            privacy_status=response.get("status", {}).get("privacyStatus", "unknown"),
            contains_synthetic_media=actual_flag,
            raw_response=response,
        )


def get_youtube_provider(client_secrets_file: str, token_file: str) -> YouTubeUploadProvider:
    return YouTubeUploadProvider(client_secrets_file, token_file)
