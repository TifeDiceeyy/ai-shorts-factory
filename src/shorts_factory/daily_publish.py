"""Cross-process one-successful-upload-per-day gate."""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import uuid
from pathlib import Path


class DailyPublishLimitReached(Exception):
    pass


class DailyPublishLedger:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    @staticmethod
    def _today() -> str:
        return dt.datetime.now(dt.timezone.utc).date().isoformat()

    def _locked_update(self, callback):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            entries = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
            result = callback(entries)
            self.path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            return result

    def reserve(self, topic: str) -> str:
        today = self._today()
        token = str(uuid.uuid4())

        def update(entries):
            active = [e for e in entries if e.get("date") == today and e.get("status") in ("pending", "published")]
            if active:
                raise DailyPublishLimitReached(
                    f"daily publish limit reached for {today}: {active[0].get('topic')} is {active[0].get('status')}"
                )
            entries.append({"token": token, "date": today, "topic": topic, "status": "pending"})
            return token

        return self._locked_update(update)

    def mark_published(self, token: str, video_id: str) -> None:
        def update(entries):
            for entry in entries:
                if entry.get("token") == token:
                    entry.update({"status": "published", "video_id": video_id})
                    return
            raise ValueError("publish reservation not found")
        self._locked_update(update)

    def mark_failed(self, token: str, error: str) -> None:
        def update(entries):
            for entry in entries:
                if entry.get("token") == token:
                    entry.update({"status": "failed", "error": error[:500]})
                    return
        self._locked_update(update)
