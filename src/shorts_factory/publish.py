"""Phase 5: publish an approved video to YouTube (private/unlisted).

Human-in-the-loop is enforced here, not just in the dashboard UI — this
function itself refuses to upload anything whose review_state isn't
"approved", so the gate can't be bypassed by calling in directly (same
defensive pattern as review_state.schedule()).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .config import load_settings
from .dashboard import review_state
from .daily_publish import DailyPublishLedger, DailyPublishLimitReached
from .experiment_ledger import record_publish
from .providers.youtube import DisclosureNotConfirmed, YouTubeNotConfigured, get_youtube_provider
from .topic_registry import get_topic

REPO_ROOT = Path(__file__).resolve().parents[2]

# YouTube tag/hashtag ceilings, kept well under the API's own real limits
# (500 total characters across snippet.tags; a wall of hashtags in the
# description reads as spammy) rather than dumping every registered keyword.
MAX_TOPIC_TAGS = 15
MAX_DESCRIPTION_HASHTAGS = 8


def _topic_tags(topic: str) -> list[str]:
    """Plain keyword phrases for YouTube's own snippet.tags field — the
    topic itself plus its registered search keywords (topic_registry.py),
    the same terms already used for retrieval/safety classification, so
    there's no new keyword-authoring step and nothing to keep in sync by
    hand. Deduplicated, order-preserving, topic first."""
    entry = get_topic(topic)
    keywords = (entry.get("keywords") if entry else []) or []
    seen: list[str] = []
    for raw in [topic, *keywords]:
        cleaned = raw.strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen[:MAX_TOPIC_TAGS]


def _description_hashtags(topic: str) -> str:
    """#Shorts (the standard Shorts-discoverability tag) plus one CamelCase
    hashtag per topic tag — YouTube hashtags can't contain spaces, so
    multi-word tags/topics get concatenated (e.g. "water filtration" ->
    #WaterFiltration)."""
    words = ["Shorts"] + [
        "".join(w.capitalize() for w in re.findall(r"[A-Za-z0-9]+", tag))
        for tag in _topic_tags(topic)
    ]
    seen: list[str] = []
    for w in words:
        if w and w not in seen:
            seen.append(w)
    return " ".join(f"#{w}" for w in seen[:MAX_DESCRIPTION_HASHTAGS])


class NotApproved(Exception):
    def __init__(self, topic: str, status: str):
        super().__init__(
            f"refusing to publish {topic!r}: review status is {status!r}, not 'approved'. "
            "Publishing stays human-approved (CLAUDE.md §5) — approve it in the dashboard first."
        )


class VerificationFailed(Exception):
    """A human can click "approve" independently of whether verify.py's
    checks actually passed (e.g. approval happened before verification ran,
    or the report is stale/missing) — approved status alone was letting a
    failed or unverified render ship to YouTube (confirmed real 2026-08-21
    review). This is a second, independent gate: both approved AND
    overall_pass=True are required, not just one."""
    def __init__(self, topic: str, reason: str):
        super().__init__(
            f"refusing to publish {topic!r}: verification did not pass ({reason}). "
            "Re-run the pipeline or regenerate the failing scene(s) first."
        )


def publish_to_youtube(topic: str, privacy_status: str = "private") -> dict:
    artifacts_dir = REPO_ROOT / "artifacts" / topic
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"no artifacts for topic {topic!r}")

    state = review_state.load(artifacts_dir)
    if state.status != "approved":
        raise NotApproved(topic, state.status)

    verification_path = artifacts_dir / "verification-report.json"
    if not verification_path.exists():
        raise VerificationFailed(topic, "no verification-report.json found")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not verification.get("overall_pass"):
        raise VerificationFailed(topic, "overall_pass is not true in verification-report.json")

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
    hashtags = _description_hashtags(topic)
    if hashtags:
        description += f"\n\n{hashtags}"
    tags = _topic_tags(topic)

    provider = get_youtube_provider(settings.youtube_client_secrets_file, settings.youtube_token_file)
    ledger = DailyPublishLedger(REPO_ROOT / "data" / "publish-ledger.json")
    reservation = ledger.reserve(topic)
    try:
        result = provider.upload_video(
            mp4_path=mp4_path,
            tags=tags,
            title=title,
            description=description,
            privacy_status=privacy_status,
            contains_synthetic_media=True,
        )
    except DisclosureNotConfirmed as exc:
        # Upload may already exist. Persist the ID so an operator can inspect
        # or remove the private orphan instead of silently losing track of it.
        failure = {
            "video_id": exc.video_id,
            "privacy_status_requested": privacy_status,
            "contains_synthetic_media_confirmed": False,
            "error": str(exc),
        }
        (artifacts_dir / "youtube-upload-failed-verification.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        ledger.mark_failed(reservation, str(exc))
        raise
    except Exception as exc:
        ledger.mark_failed(reservation, str(exc))
        raise

    upload_record = {
        "video_id": result.video_id,
        "privacy_status": result.privacy_status,
        "contains_synthetic_media": result.contains_synthetic_media,
    }
    (artifacts_dir / "youtube-upload-result.json").write_text(json.dumps(upload_record, indent=2), encoding="utf-8")

    ledger.mark_published(reservation, result.video_id)
    record_publish(
        topic=topic,
        video_id=result.video_id,
        concept=title,
        hook_variant_index=1,
        series="reinvent-it",
    )
    review_state.schedule(artifacts_dir, notes=f"uploaded as {result.video_id}")

    return upload_record


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m shorts_factory.publish <topic>", file=sys.stderr)
        return 2
    topic = argv[1]
    try:
        result = publish_to_youtube(topic)
    except (
        NotApproved, VerificationFailed, YouTubeNotConfigured,
        DisclosureNotConfirmed, FileNotFoundError, DailyPublishLimitReached,
    ) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"uploaded: video_id={result['video_id']} privacy={result['privacy_status']} "
          f"synthetic_disclosure={result['contains_synthetic_media']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
