"""Supervised Telegram control surface built on aiogram 3."""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.types import FSInputFile, Message

from .config import load_settings
from .dashboard import review_state
from .pipeline import REPO_ROOT
from .publish import publish_to_youtube

TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,79}$", re.IGNORECASE)


class TelegramNotConfigured(Exception):
    pass


class TelegramController:
    """Synchronous business rules kept separate from aiogram transport."""

    def __init__(self, allowed_user_ids: tuple[int, ...], artifacts_root: Path | None = None):
        if not allowed_user_ids:
            raise TelegramNotConfigured("TELEGRAM_ALLOWED_USER_IDS must contain at least one trusted user ID")
        self.allowed_user_ids = set(allowed_user_ids)
        self.artifacts_root = artifacts_root or REPO_ROOT / "artifacts"

    def authorized(self, user_id: int | None) -> bool:
        return user_id is not None and int(user_id) in self.allowed_user_ids

    def topic_dir(self, topic: str) -> Path:
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

    def approve(self, topic: str, user_id: int) -> str:
        topic_dir = self.topic_dir(topic)
        review_state.approve(topic_dir, notes=f"approved by Telegram user {user_id}")
        return f"Approved {topic_dir.name}. It has not been published."

    def reject(self, arguments: str, user_id: int) -> str:
        topic, separator, reason = arguments.partition("|")
        topic_dir = self.topic_dir(topic)
        review_state.reject(topic_dir, notes=reason.strip() if separator else f"rejected by Telegram user {user_id}")
        return f"Rejected {topic_dir.name}."

    def video_path(self, topic: str) -> Path:
        topic_dir = self.topic_dir(topic)
        path = topic_dir / f"{topic_dir.name}.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"video is missing for {topic_dir.name!r}")
        return path


HELP = (
    "/status\n/video <topic>\n/approve <topic>\n"
    "/reject <topic> | <reason>\n/publish <topic>\n\n"
    "Publishing requires prior approval and uploads privately by default."
)


def build_router(controller: TelegramController) -> Router:
    router = Router(name="shorts_factory_review")

    @router.message()
    async def handle_message(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not controller.authorized(user_id):
            await message.answer("Unauthorized.")
            return
        text = (message.text or "").strip()
        if not text:
            return
        command, _, arguments = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        try:
            if command in ("/start", "/help"):
                reply = HELP
            elif command == "/status":
                reply = controller.status_text()
            elif command == "/video":
                await message.answer_video(FSInputFile(controller.video_path(arguments)))
                return
            elif command == "/approve":
                reply = controller.approve(arguments, int(user_id))
            elif command == "/reject":
                reply = controller.reject(arguments, int(user_id))
            elif command == "/publish":
                topic_dir = controller.topic_dir(arguments)
                result = await asyncio.to_thread(publish_to_youtube, topic_dir.name, "private")
                reply = f"Uploaded privately: {result['video_id']}; disclosure confirmed."
            else:
                reply = "Unknown command. Send /help."
        except Exception as exc:
            reply = f"Refused: {exc}"
        await message.answer(reply)

    return router


def create_dispatcher(controller: TelegramController) -> Dispatcher:
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(controller))
    return dispatcher


async def run() -> None:
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise TelegramNotConfigured("TELEGRAM_BOT_TOKEN is not configured")
    controller = TelegramController(settings.telegram_allowed_user_ids)
    bot = Bot(settings.telegram_bot_token)
    dispatcher = create_dispatcher(controller)
    await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())


def main() -> int:
    try:
        asyncio.run(run())
    except (TelegramNotConfigured, KeyboardInterrupt) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
