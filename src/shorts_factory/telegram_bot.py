"""Supervised Telegram control surface built on aiogram 3.

Two kinds of surface live here:
1. The original review/approve/publish commands over already-rendered videos.
2. A guided planning flow (aiogram FSM) that drives the rest of the pipeline
   from Telegram: pick or propose a topic, run retrieval, then generate the
   video — each step is a human-confirmed button press, never automatic.
   See PlanningStates below. (The idea/hook-selection step that used to sit
   between topic and retrieval was removed 2026-08-28, per explicit user
   request — "once a story question is prompted... check the brain and
   write story," skipping the intermediate concept/angle choice.)

Only one retrieval-or-generate job (the two expensive, blocking calls) runs
at a time, guarded by _job_lock — there is no job queue; a second request
while one is in flight is refused with a "try again shortly" reply.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)

from .config import load_settings, require_budget_approval_if_paid
from .cost_tracker import CostTracker
from .dashboard import review_state
from .pipeline import REPO_ROOT, PipelineResult, run_pipeline
from .providers.fal import FalGateway
from .providers.llm import get_llm_provider
from .providers.search import get_search_provider
from .publish import publish_to_youtube
from .retrieval import run_retrieval_for_topic
from .safety import is_explicitly_red
from .topic_registry import get_topic, load_registry, normalize_topic, register_topic

TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,79}$", re.IGNORECASE)
logger = logging.getLogger(__name__)


class TelegramNotConfigured(Exception):
    pass


class PlanningStates(StatesGroup):
    choosing_topic = State()
    confirming_new_topic = State()
    confirming_retrieval = State()
    choosing_length = State()
    confirming_generate = State()
    reviewing_generated = State()


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

    # --- Planning: topic registry / safety -------------------------------

    def known_topics(self) -> list[str]:
        registry = load_registry()
        return sorted(name for name, entry in registry.items() if entry.get("queries") and entry.get("keywords"))

    def topic_status(self, topic: str) -> dict:
        """Classify a topic without touching an LLM: 'registered' (has
        retrieval config already), 'red' (explicitly blocked), or 'unknown'
        (needs a new-topic proposal)."""
        normalized = normalize_topic(topic)
        entry = get_topic(normalized)
        if entry and entry.get("queries") and entry.get("keywords"):
            return {"state": "registered", "topic": normalized, "safety_class": entry["safety_class"]}
        if is_explicitly_red(normalized):
            return {"state": "red", "topic": normalized}
        return {"state": "unknown", "topic": normalized}

    def propose_new_topic(self, topic: str) -> dict:
        """LLM call (caller must run this via asyncio.to_thread) proposing a
        safety class + Phase 1 retrieval config for an unregistered topic.
        Raises BudgetApprovalRequired if a real provider needs approval first.
        Never trusted alone for RED: the result is re-checked against
        is_explicitly_red() here, and confirm_new_topic() below refuses
        to persist anything not classified green/yellow."""
        settings = load_settings()
        require_budget_approval_if_paid(settings)
        cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
        uses_fal = settings.llm.provider.strip().lower() == "fal"
        fal_gateway = FalGateway(settings.fal_key) if uses_fal else None
        llm = get_llm_provider(
            settings.llm.provider,
            settings.credential_for(settings.llm),
            settings.llm.model_or_voice,
            settings.llm_cost_per_script_usd,
            gateway=fal_gateway,
            endpoint=settings.fal_llm_endpoint,
        )
        proposal = llm.propose_topic(topic, cost_tracker)
        normalized = normalize_topic(topic)
        proposal["topic"] = normalized
        if is_explicitly_red(normalized):
            proposal["safety_class"] = "red"
        return proposal

    def confirm_new_topic(self, proposal: dict) -> str:
        if proposal.get("safety_class") not in ("green", "yellow"):
            raise ValueError("refusing to register a topic that isn't classified green or yellow")
        register_topic(
            proposal["topic"],
            proposal.get("queries") or [],
            proposal.get("keywords") or [],
            proposal["safety_class"],
            proposal.get("caution"),
        )
        return f"Registered {proposal['topic']!r} as {proposal['safety_class'].upper()}."

    def needs_retrieval(self, topic: str) -> bool:
        """False (no retrieval needed) when either a citations.json already
        exists on disk, OR the local brain covers the topic well enough to
        skip paid Tavily retrieval entirely — mirrors run_pipeline's own
        brain-first check (pipeline.py), which this gate had drifted out of
        sync with: it only ever checked for citations.json, so the bot kept
        asking to run (paid) retrieval even for topics the brain already
        covers for free. Real gap found 2026-08-29 via a live Telegram
        test — a brain-covered topic still prompted "No verified sources
        yet... Run retrieval now?"."""
        path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
        if path.exists():
            return False
        from .brain_integration import brain_covers_topic, load_brain

        brain = load_brain()
        if brain is not None:
            covered, _research = brain_covers_topic(brain, topic)
            if covered:
                return False
        return True

    # --- Retrieval / generate (blocking; caller must run via asyncio.to_thread) --

    def run_retrieval(self, topic: str) -> dict:
        settings = load_settings()
        require_budget_approval_if_paid(settings)
        if settings.search.is_stub:
            raise RuntimeError(
                "SEARCH_PROVIDER is not configured — set SEARCH_PROVIDER=tavily and "
                "SEARCH_API_KEY in .env, there is no stub option for retrieval"
            )
        search_provider = get_search_provider(settings.search.provider, settings.search_api_key)
        cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
        result = run_retrieval_for_topic(topic, search_provider, cost_tracker, book_file=settings.book_file)
        cost_report_path = REPO_ROOT / "data" / topic.replace(" ", "_") / "retrieval-cost-report.json"
        cost_tracker.write_report(cost_report_path)
        return {"result": result, "spent": cost_tracker.total_spent_usd, "cap": settings.budget_cap_usd}

    def run_generate(
        self, topic: str, idea: dict | None = None, target_seconds: float | None = None
    ) -> PipelineResult:
        return run_pipeline(topic, idea=idea, target_seconds=target_seconds)


HELP = (
    "/plan — pick or propose a topic, generate a video\n"
    "/cancel — abort the current /plan flow\n"
    "/status\n/video <topic>\n/approve <topic>\n"
    "/reject <topic> | <reason>\n/publish <topic>\n\n"
    "Publishing requires prior approval and uploads privately by default."
)


# Offered video lengths, in seconds. Kept inside the Shorts range the
# pacing was actually tuned against: one scene runs ~3s (brief_builder.
# SECONDS_PER_SCENE), so these map to ~10/15/20 scenes. Video generation is
# the dominant cost, so the label carries a rough estimate — a longer video
# is proportionally more expensive, and that should not be a surprise.
LENGTH_CHOICES: tuple[tuple[int, str], ...] = (
    (30, "30s · ~10 scenes"),
    (45, "45s · ~15 scenes"),
    (60, "60s · ~20 scenes"),
)
DEFAULT_LENGTH_SECONDS = 45


def _length_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"length_{seconds}")]
            for seconds, label in LENGTH_CHOICES
        ]
        + [[InlineKeyboardButton(text="Cancel", callback_data="length_cancel")]]
    )


def _confirm_cancel_kb(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="Confirm", callback_data=f"{prefix}_confirm"),
            InlineKeyboardButton(text="Cancel", callback_data=f"{prefix}_cancel"),
        ]]
    )


def _approve_publish_kb() -> InlineKeyboardMarkup:
    # No topic embedded in callback_data (Telegram caps it at 64 bytes, and
    # TOPIC_RE allows up to 80 chars — "review_publish:" + a long topic
    # could silently exceed that). The topic is read back from FSM state
    # data instead, same as every other button in this file (generate_confirm
    # etc. all resolve their topic via state.get_data(), never callback_data).
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="✅ Approve", callback_data="review_approve"),
            InlineKeyboardButton(text="📤 Publish", callback_data="review_publish"),
        ]]
    )


def _main_kb() -> ReplyKeyboardMarkup:
    """Persistent bottom keyboard for the no-argument top-level commands.
    Button text is the literal command string, so tapping one just sends
    that text — the existing command dispatch in handle_message() below
    handles it exactly like it was typed, no separate parsing needed."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/plan"), KeyboardButton(text="/status")],
            [KeyboardButton(text="/cancel"), KeyboardButton(text="/help")],
        ],
        resize_keyboard=True,
    )


class _JobState:
    topic: str | None = None


def build_router(controller: TelegramController) -> Router:
    router = Router(name="shorts_factory_review")
    job_lock = asyncio.Lock()
    job_state = _JobState()

    async def enter_length_choice(message: Message, state: FSMContext, topic: str) -> None:
        await state.set_state(PlanningStates.choosing_length)
        await state.update_data(topic=topic)
        await message.answer(
            f"How long should {topic!r} be? One scene runs about 3 seconds, so a longer "
            "video means more scenes — and more scenes cost proportionally more.",
            reply_markup=_length_kb(),
        )

    async def enter_generate_confirm(message: Message, state: FSMContext, topic: str) -> None:
        stored = await state.get_data()
        seconds = stored.get("target_seconds") or DEFAULT_LENGTH_SECONDS
        await state.set_state(PlanningStates.confirming_generate)
        await state.update_data(topic=topic)
        await message.answer(
            f"Ready to generate {topic!r} at ~{seconds}s — the mascot is chosen automatically from the "
            "story. Real cost is ~$0.30/video once real providers are configured (stub providers cost $0) "
            "— enforced against BUDGET_CAP_USD either way. Generate now?",
            reply_markup=_confirm_cancel_kb("generate"),
        )

    async def enter_retrieval_or_generate(message: Message, state: FSMContext, topic: str) -> None:
        if controller.needs_retrieval(topic):
            await state.set_state(PlanningStates.confirming_retrieval)
            await state.update_data(topic=topic)
            await message.answer(
                "No verified sources yet for this topic. Run retrieval now? "
                "(uses your search provider; refused if SEARCH_PROVIDER isn't configured)",
                reply_markup=_confirm_cancel_kb("retrieval"),
            )
        else:
            await enter_generate_confirm(message, state, topic)

    async def handle_topic_entry(message: Message, state: FSMContext, text: str) -> None:
        topic = text.strip()
        if not TOPIC_RE.fullmatch(topic):
            await message.answer("Invalid topic name — use letters, numbers, spaces, dashes, or underscores (max 80 chars).")
            return
        status = controller.topic_status(topic)
        if status["state"] == "registered":
            await enter_retrieval_or_generate(message, state, status["topic"])
            return
        if status["state"] == "red":
            await message.answer(f"Topic {topic!r} is classified RED — permanently blocked, nothing registered.")
            await state.clear()
            return
        await message.answer(f"Classifying new topic {topic!r}…")
        try:
            proposal = await asyncio.to_thread(controller.propose_new_topic, topic)
        except Exception as exc:
            await message.answer(f"Refused: {exc}")
            await state.clear()
            return
        if proposal["safety_class"] not in ("green", "yellow"):
            await message.answer(
                f"Topic {proposal['topic']!r} is classified RED — permanently blocked, nothing registered.\n"
                f"Reasoning: {proposal.get('reasoning', '')}"
            )
            await state.clear()
            return
        lines = [
            f"New topic {proposal['topic']!r} — proposed classification: {proposal['safety_class'].upper()}",
            f"Reasoning: {proposal.get('reasoning', '')}",
            f"Search queries: {', '.join(proposal.get('queries') or [])}",
            f"Keywords: {', '.join(proposal.get('keywords') or [])}",
        ]
        if proposal.get("caution"):
            lines.append(f"Caution: {proposal['caution']}")
        lines.append("\nRegister this topic?")
        await state.set_state(PlanningStates.confirming_new_topic)
        await state.update_data(proposal=proposal)
        await message.answer("\n".join(lines), reply_markup=_confirm_cancel_kb("newtopic"))

    async def run_locked_retrieval(message: Message, state: FSMContext, topic: str) -> None:
        if job_lock.locked():
            await message.answer(f"A job is already running ({job_state.topic}). Try again shortly.")
            return
        async with job_lock:
            job_state.topic = f"retrieval:{topic}"
            await message.answer(f"Running retrieval for {topic!r}…")
            try:
                outcome = await asyncio.to_thread(controller.run_retrieval, topic)
            except Exception as exc:
                await message.answer(f"Refused: {exc}")
                await state.clear()
                return
            finally:
                job_state.topic = None
        result = outcome["result"]
        await message.answer(
            f"Retrieval done: {result['_meta']['chunk_count']} chunks, {result['_meta']['claim_count']} claims, "
            f"{result['verified_count']}/{result['citation_count']} verified. "
            f"Spent ${outcome['spent']:.4f} / ${outcome['cap']:.2f} cap."
        )
        await enter_length_choice(message, state, topic)

    async def run_locked_generate(
        message: Message, state: FSMContext, topic: str, target_seconds: float | None = None
    ) -> None:
        if job_lock.locked():
            await message.answer(f"A job is already running ({job_state.topic}). Try again shortly.")
            return
        async with job_lock:
            job_state.topic = f"generate:{topic}"
            await message.answer(f"Generating {topic!r}… this can take a few minutes.")
            try:
                result = await asyncio.to_thread(
                    controller.run_generate, topic, None, target_seconds
                )
            except Exception as exc:
                logger.exception("Telegram generation failed for topic %r", topic)
                await message.answer(f"Refused: {exc}")
                await state.clear()
                return
            finally:
                job_state.topic = None
        if result.budget_approval_blocked:
            await message.answer(f"Refused: {result.budget_approval_block_reason}")
        elif result.blocked:
            await message.answer(f"Blocked: {result.block_reason}")
        elif result.budget_exceeded:
            # result.verification stays None in this case (the run
            # returned early, before verification could run) — this branch
            # used to fall through to the "Generated ..." success message
            # below regardless, telling the operator a run had succeeded
            # when it had actually failed partway through on the budget cap
            # (confirmed real 2026-08-21 review).
            lines = [f"Budget exceeded generating {topic!r}: {result.budget_exceeded_reason}"]
            if result.cost_report:
                lines.append(
                    f"Cost: ${result.cost_report['total_spent_usd']:.4f} / "
                    f"${result.cost_report['budget_cap_usd']:.2f} cap"
                )
            await message.answer("\n".join(lines))
        else:
            lines = [f"Generated {topic!r}."]
            if result.cost_report:
                lines.append(
                    f"Cost: ${result.cost_report['total_spent_usd']:.4f} / "
                    f"${result.cost_report['budget_cap_usd']:.2f} cap"
                )
            if result.verification:
                lines.append(f"Verification: {'PASS' if result.verification['overall_pass'] else 'FAIL'}")
            lines.append("Or use /approve, /reject, /publish directly.")
            await message.answer("\n".join(lines))
            # Send the actual video file too, not just the status text —
            # previously the operator had to know to separately type
            # /video <topic> to ever see it (confirmed real UX gap 2026-08-28:
            # user generated a real video and it never appeared in chat).
            # Same lookup /video itself uses (controller.video_path); if the
            # file is somehow missing despite a "successful" result, let it
            # raise into the outer handler's "Refused: ..." reporting rather
            # than silently skip sending anything.
            await message.answer_video(FSInputFile(controller.video_path(topic)))
            # Approve/Publish buttons right after the video, same one-tap
            # pattern as the generate/retrieval confirm buttons — direct
            # user feedback 2026-08-30: typing /approve <topic> then
            # /publish <topic> by hand was more friction than necessary.
            await state.set_state(PlanningStates.reviewing_generated)
            await state.update_data(topic=topic)
            await message.answer("Approve or publish this video?", reply_markup=_approve_publish_kb())
            return
        await state.clear()

    @router.message()
    async def handle_message(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not controller.authorized(user_id):
            print(f"[telegram_bot] Unauthorized message from user_id={user_id}")
            await message.answer("Unauthorized.")
            return
        text = (message.text or "").strip()
        if not text:
            return

        current_state = await state.get_state()
        if current_state == PlanningStates.choosing_topic.state and not text.startswith("/"):
            await handle_topic_entry(message, state, text)
            return

        command, _, arguments = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        reply_markup = None
        try:
            if command in ("/start", "/help"):
                reply = HELP
                reply_markup = _main_kb()
            elif command == "/plan":
                await state.set_state(PlanningStates.choosing_topic)
                known = ", ".join(controller.known_topics()) or "none yet"
                reply = (
                    f"Send a topic name to plan.\nKnown topics: {known}\n"
                    "Or type a new one — I'll propose a safety class and research queries."
                )
                reply_markup = _main_kb()
            elif command == "/cancel":
                await state.clear()
                reply = "Cancelled."
                reply_markup = _main_kb()
            elif command == "/status":
                reply = controller.status_text()
                if job_state.topic:
                    reply += f"\n\nActive job: {job_state.topic}"
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
        await message.answer(reply, reply_markup=reply_markup)

    @router.callback_query()
    async def handle_callback(callback: CallbackQuery, state: FSMContext) -> None:
        user_id = callback.from_user.id if callback.from_user else None
        if not controller.authorized(user_id):
            await callback.answer("Unauthorized.", show_alert=True)
            return
        # Telegram callback queries must be acknowledged within seconds. The
        # generation/retrieval handlers can run for many minutes, so answering
        # at the end makes Telegram reject the stale query even when the job
        # itself completed successfully.
        await callback.answer()
        current_state = await state.get_state()
        data = callback.data or ""
        message = callback.message
        try:
            if current_state == PlanningStates.confirming_new_topic.state:
                stored = await state.get_data()
                proposal = stored.get("proposal") or {}
                if data == "newtopic_confirm":
                    reply = controller.confirm_new_topic(proposal)
                    await message.answer(reply)
                    await enter_retrieval_or_generate(message, state, proposal["topic"])
                elif data == "newtopic_cancel":
                    await message.answer("Cancelled.")
                    await state.clear()
            elif current_state == PlanningStates.confirming_retrieval.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "retrieval_confirm":
                    await run_locked_retrieval(message, state, topic)
                elif data == "retrieval_cancel":
                    await message.answer("Skipped retrieval.")
                    await enter_length_choice(message, state, topic)
            elif current_state == PlanningStates.choosing_length.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "length_cancel":
                    await message.answer("Cancelled.")
                    await state.clear()
                elif data.startswith("length_"):
                    try:
                        seconds = int(data.split("_", 1)[1])
                    except ValueError:
                        await message.answer("Unrecognised length.")
                        return
                    # Only ever accept a length we actually offered — the
                    # callback payload is attacker-controllable in principle,
                    # and an arbitrary value would drive both scene count and
                    # spend.
                    if seconds not in {opt for opt, _label in LENGTH_CHOICES}:
                        await message.answer("Unrecognised length.")
                        return
                    await state.update_data(target_seconds=seconds)
                    await enter_generate_confirm(message, state, topic)
            elif current_state == PlanningStates.confirming_generate.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "generate_confirm":
                    await run_locked_generate(
                        message, state, topic, stored.get("target_seconds")
                    )
                elif data == "generate_cancel":
                    await message.answer("Cancelled.")
                    await state.clear()
            elif current_state == PlanningStates.reviewing_generated.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "review_approve":
                    reply = controller.approve(topic, int(user_id))
                    await message.answer(reply)
                    # State (and the buttons) stay active — Publish still
                    # needs its own tap right after Approve, same as
                    # /approve then /publish as two separate commands.
                elif data == "review_publish":
                    result = await asyncio.to_thread(publish_to_youtube, topic, "private")
                    await message.answer(f"Uploaded privately: {result['video_id']}; disclosure confirmed.")
                    await state.clear()
        except Exception as exc:
            await message.answer(f"Refused: {exc}")

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
