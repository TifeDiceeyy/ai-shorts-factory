"""Supervised Telegram control surface.

The bot cannot generate content. It exposes review state, approval/rejection,
private preview delivery, and an explicit publish command. Every command is
restricted to configured Telegram user IDs; an empty allowlist refuses to run.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

from .config import load_settings
from .dashboard import review_state
from .pipeline import REPO_ROOT
from .publish import publish_to_youtube

TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,79}$", re.IGNORECASE)


class TelegramNotConfigured(Exception):
    pass


class TelegramBot:
    def __init__(self, token: str, allowed_user_ids: tuple[int, ...], artifacts_root: Path | None = None):
        if not token:
            raise TelegramNotConfigured("TELEGRAM_BOT_TOKEN is not configured")
        if not allowed_user_ids:
            raise TelegramNotConfigured("TELEGRAM_ALLOWED_USER_IDS must contain at least one trusted user ID")
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.allowed_user_ids = set(allowed_user_ids)
        self.artifacts_root = artifacts_root or REPO_ROOT / "artifacts"

    def _request(self, method: str, **kwargs) -> dict:
        response = requests.post(f"{self.base_url}/{method}", timeout=60, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {payload.get('description', 'unknown error')}")
        return payload

    def send_message(self, chat_id: int, text: str) -> None:
        self._request("sendMessage", data={"chat_id": chat_id, "text": text})

    def send_video(self, chat_id: int, video_path: Path) -> None:
        with video_path.open("rb") as video:
            self._request("sendVideo", data={"chat_id": chat_id}, files={"video": video})

    def _topic_dir(self, topic: str) -> Path:
        topic = topic.strip()
        if not TOPIC_RE.fullmatch(topic):
            raise ValueError("invalid topic; use letters, numbers, spaces, dashes, or underscores")
        path = (self.artifacts_root / topic).resolve()
        if self.artifacts_root.resolve() not in path.parents or not path.is_dir():
            raise FileNotFoundError(f"no artifacts found for {topic!r}")
        return path

    def status_text(self) -> str:
        rows = []
        if self.artifacts_root.exists():
            for topic_dir in sorted(p for p in self.artifacts_root.iterdir() if p.is_dir()):
                state = review_state.load(topic_dir)
                verified = False
                report = topic_dir / "verification-report.json"
                if report.exists():
                    verified = bool(json.loads(report.read_text(encoding="utf-8")).get("overall_pass"))
                rows.append(f"{topic_dir.name}: {state.status}; verified={'yes' if verified else 'no'}")
        return "\n".join(rows) if rows else "No rendered videos."

    def handle_update(self, update: dict) -> None:
        message = update.get("message") or {}
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or user_id is None or not text:
            return
        if int(user_id) not in self.allowed_user_ids:
            self.send_message(int(chat_id), "Unauthorized.")
            return

        command, _, arguments = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        try:
            if command in ("/start", "/help"):
                reply = (
                    "/status\n/video <topic>\n/approve <topic>\n"
                    "/reject <topic> [reason]\n/publish <topic>\n\n"
                    "Publishing still requires prior approval and uploads privately by default."
                )
            elif command == "/status":
                reply = self.status_text()
            elif command == "/video":
                topic_dir = self._topic_dir(arguments)
                video_path = topic_dir / f"{topic_dir.name}.mp4"
                if not video_path.is_file():
                    raise FileNotFoundError(f"video is missing for {topic_dir.name!r}")
                self.send_video(int(chat_id), video_path)
                return
            elif command == "/approve":
                topic_dir = self._topic_dir(arguments)
                review_state.approve(topic_dir, notes=f"approved by Telegram user {user_id}")
                reply = f"Approved {topic_dir.name}. It has not been published."
            elif command == "/reject":
                topic, _, reason = arguments.partition(" ")
                topic_dir = self._topic_dir(topic)
                review_state.reject(topic_dir, notes=reason or f"rejected by Telegram user {user_id}")
                reply = f"Rejected {topic_dir.name}."
            elif command == "/publish":
                topic_dir = self._topic_dir(arguments)
                result = publish_to_youtube(topic_dir.name, privacy_status="private")
                reply = f"Uploaded privately: {result['video_id']}; disclosure confirmed."
            else:
                reply = "Unknown command. Send /help."
        except Exception as exc:
            reply = f"Refused: {exc}"
        self.send_message(int(chat_id), reply)

    def run_polling(self) -> None:
        offset = 0
        while True:
            payload = self._request("getUpdates", data={"timeout": 50, "offset": offset})
            for update in payload.get("result", []):
                offset = max(offset, int(update["update_id"]) + 1)
                self.handle_update(update)


def main() -> int:
    settings = load_settings()
    try:
        TelegramBot(settings.telegram_bot_token, settings.telegram_allowed_user_ids).run_polling()
    except (TelegramNotConfigured, KeyboardInterrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
