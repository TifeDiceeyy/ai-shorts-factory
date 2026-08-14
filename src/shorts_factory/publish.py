"""Phase 5: publish an approved video to YouTube (private/unlisted).

Human-in-the-loop is enforced here, not just in the dashboard UI — this
function itself refuses to upload anything whose review_state isn't
"approved", so the gate can't be bypassed by calling in directly (same
defensive pattern as review_state.schedule()).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import load_settings
from .dashboard import review_state
from .providers.youtube import DisclosureNotConfirmed, YouTubeNotConfigured, get_youtube_provider

REPO_ROOT = Path(__file__).resolve().parents[2]


class NotApproved(Exception):
    def __init__(self, topic: str, status: str):
        super().__init__(
            f"refusing to publish {topic!r}: review status is {status!r}, not 'approved'. "
            "Publishing stays human-approved (CLAUDE.md §5) — approve it in the dashboard first."
        )


def publish_to_youtube(topic: str, privacy_status: str = "private") -> dict:
    artifacts_dir = REPO_ROOT / "artifacts" / topic
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"no artifacts for topic {topic!r}")

    state = review_state.load(artifacts_dir)
    if state.status != "approved":
        raise NotApproved(topic, state.status)

    settings = load_settings()
    if not settings.youtube_configured:
        raise YouTubeNotConfigured(
            "YOUTUBE_CLIENT_SECRETS_FILE not set in .env — Phase 5 upload isn't configured yet."
        )

    script_path = artifacts_dir / f"{topic}.script.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    mp4_path = artifacts_dir / f"{topic}.mp4"
    if not mp4_path.exists():
        raise FileNotFoundError(f"no rendered video at {mp4_path}")

    title = script["scenes"][0]["caption"][:95] if script["scenes"] else topic.title()
    description = "\n\n".join(s["narration"] for s in script["scenes"])
    description += "\n\nAI-generated content. Sources available on request."

    provider = get_youtube_provider(settings.youtube_client_secrets_file, settings.youtube_token_file)
    result = provider.upload_video(
        mp4_path=mp4_path,
        title=title,
        description=description,
        privacy_status=privacy_status,
        contains_synthetic_media=True,
    )

    upload_record = {
        "video_id": result.video_id,
        "privacy_status": result.privacy_status,
        "contains_synthetic_media": result.contains_synthetic_media,
    }
    (artifacts_dir / "youtube-upload-result.json").write_text(json.dumps(upload_record, indent=2), encoding="utf-8")

    review_state.schedule(artifacts_dir, notes=f"uploaded as {result.video_id}")

    return upload_record


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m shorts_factory.publish <topic>", file=sys.stderr)
        return 2
    topic = argv[1]
    try:
        result = publish_to_youtube(topic)
    except (NotApproved, YouTubeNotConfigured, DisclosureNotConfirmed, FileNotFoundError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"uploaded: video_id={result['video_id']} privacy={result['privacy_status']} "
          f"synthetic_disclosure={result['contains_synthetic_media']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
