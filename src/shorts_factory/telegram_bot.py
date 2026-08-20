"""Supervised Telegram control surface built on aiogram 3.

Two kinds of surface live here:
1. The original review/approve/publish commands over already-rendered videos.
2. A guided planning flow (aiogram FSM) that drives the rest of the pipeline
   from Telegram: pick or propose a topic, generate ideas/hooks, run
   retrieval, then generate the video — each step is a human-confirmed
   button press, never automatic. See PlanningStates below.

Only one retrieval-or-generate job (the two expensive, blocking calls) runs
at a time, guarded by _job_lock — there is no job queue; a second request
while one is in flight is refused with a "try again shortly" reply.
"""
from __future__ import annotations

import asyncio
import json
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
from .ideation import Hook, Idea, generate_ideas, ideas_to_dicts, record_idea_chosen
from .mascots import get_mascot, list_mascots
from .pipeline import REPO_ROOT, PipelineResult, run_pipeline
from .providers.fal import FalGateway
from .providers.llm import get_llm_provider
from .providers.search import get_search_provider
from .publish import publish_to_youtube
from .retrieval import run_retrieval_for_topic
from .safety import RED_KEYWORDS, RED_TOPICS
from .topic_registry import get_topic, load_registry, normalize_topic, register_topic

TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,79}$", re.IGNORECASE)


class TelegramNotConfigured(Exception):
    pass


class PlanningStates(StatesGroup):
    choosing_topic = State()
    confirming_new_topic = State()
    choosing_idea = State()
    choosing_mascot = State()
    confirming_retrieval = State()
    confirming_generate = State()


def _idea_from_dict(d: dict) -> Idea:
    """Inverse of ideation.ideas_to_dicts — reconstructs one Idea so
    record_idea_chosen() can be called after a round-trip through FSM state."""
    return Idea(
        topic=d["topic"],
        concept=d["concept"],
        angle=d["angle"],
        hooks=[Hook(text=h["text"], variant_index=h["variant_index"]) for h in d["hooks"]],
        payoff=d["payoff"],
        series=d["series"],
        safety_class=d["safety_class"],
        visual_potential_score=d["visual_potential_score"],
        source_availability=d["source_availability"],
        similarity_to_recent=d["similarity_to_recent"],
        rank_score=d["rank_score"],
    )


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
        if normalized in RED_TOPICS or any(kw in normalized for kw in RED_KEYWORDS):
            return {"state": "red", "topic": normalized}
        return {"state": "unknown", "topic": normalized}

    def propose_new_topic(self, topic: str) -> dict:
        """LLM call (caller must run this via asyncio.to_thread) proposing a
        safety class + Phase 1 retrieval config for an unregistered topic.
        Raises BudgetApprovalRequired if a real provider needs approval first.
        Never trusted alone for RED: the result is re-checked against
        RED_TOPICS/RED_KEYWORDS here, and confirm_new_topic() below refuses
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
        if normalized in RED_TOPICS or any(kw in normalized for kw in RED_KEYWORDS):
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

    # --- Ideation ----------------------------------------------------------

    def ideate(self, topic: str, n: int = 3) -> list[Idea]:
        return generate_ideas(topic, n)

    def choose_idea(self, idea_dict: dict) -> str:
        idea = _idea_from_dict(idea_dict)
        record_idea_chosen(idea)
        return f"Chosen: {idea.concept}"

    def needs_retrieval(self, topic: str) -> bool:
        path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
        return not path.exists()

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

    def mascots_text(self) -> str:
        lines = ["🎭 Selectable Mascots:"]
        for m in list_mascots():
            lines.append(f"• {m.name}: {m.short_desc}")
        return "\n".join(lines)

    def run_generate(
        self, topic: str, idea: dict | None = None, mascot_id: str | None = None
    ) -> PipelineResult:
        return run_pipeline(topic, idea=idea, mascot_id=mascot_id)


HELP = (
    "/plan — pick or propose a topic, choose mascot, get ideas, generate a video\n"
    "/mascots — view and preview available character mascots (1 to 5)\n"
    "/cancel — abort the current /plan flow\n"
    "/status\n/video <topic>\n/approve <topic>\n"
    "/reject <topic> | <reason>\n/publish <topic>\n\n"
    "Publishing requires prior approval and uploads privately by default."
)


def _confirm_cancel_kb(prefix: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="Confirm", callback_data=f"{prefix}_confirm"),
            InlineKeyboardButton(text="Cancel", callback_data=f"{prefix}_cancel"),
        ]]
    )


def _mascot_kb() -> InlineKeyboardMarkup:
    rows = []
    for m in list_mascots():
        rows.append([InlineKeyboardButton(text=m.name, callback_data=f"mascot_{m.id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _idea_kb(idea_dicts: list[dict]) -> InlineKeyboardMarkup:
    rows = []
    for i, idea in enumerate(idea_dicts):
        label = idea["hooks"][0]["text"] if idea["hooks"] else idea["concept"]
        rows.append([InlineKeyboardButton(text=label[:64], callback_data=f"idea_{i}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


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

    async def enter_ideation(message: Message, state: FSMContext, topic: str) -> None:
        try:
            ideas = await asyncio.to_thread(controller.ideate, topic, 3)
        except Exception as exc:
            await message.answer(f"Refused: {exc}")
            await state.clear()
            return
        if not ideas:
            await message.answer("No ideas generated.")
            await state.clear()
            return
        idea_dicts = ideas_to_dicts(ideas)
        await state.set_state(PlanningStates.choosing_idea)
        await state.update_data(topic=topic, ideas=idea_dicts)
        lines = [f"Ideas for {topic!r}:"]
        for i, idea in enumerate(idea_dicts):
            hook = idea["hooks"][0]["text"] if idea["hooks"] else idea["concept"]
            lines.append(f"{i + 1}. {idea['concept']} — \"{hook}\"")
        await message.answer("\n".join(lines), reply_markup=_idea_kb(idea_dicts))

    async def enter_mascot_selection(message: Message, state: FSMContext, topic: str) -> None:
        await state.set_state(PlanningStates.choosing_mascot)
        await state.update_data(topic=topic)
        await message.answer(
            "🎭 Choose a character mascot for this video (Mascot 1 to 5):",
            reply_markup=_mascot_kb(),
        )

    async def enter_generate_confirm(message: Message, state: FSMContext, topic: str) -> None:
        await state.set_state(PlanningStates.confirming_generate)
        await state.update_data(topic=topic)
        stored = await state.get_data()
        mascot_id = stored.get("chosen_mascot_id")
        mascot = get_mascot(mascot_id)
        await message.answer(
            f"Ready to generate {topic!r} using {mascot.name}. Real cost is ~$0.30/video once real providers are "
            "configured (stub providers cost $0) — enforced against BUDGET_CAP_USD either way. Generate now?",
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
            await enter_ideation(message, state, status["topic"])
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
        await enter_generate_confirm(message, state, topic)

    async def run_locked_generate(message: Message, state: FSMContext, topic: str) -> None:
        if job_lock.locked():
            await message.answer(f"A job is already running ({job_state.topic}). Try again shortly.")
            return
        stored = await state.get_data()
        chosen_idea = stored.get("chosen_idea")
        chosen_mascot_id = stored.get("chosen_mascot_id")
        async with job_lock:
            job_state.topic = f"generate:{topic}"
            await message.answer(f"Generating {topic!r}… this can take a few minutes.")
            try:
                result = await asyncio.to_thread(controller.run_generate, topic, chosen_idea, chosen_mascot_id)
            except Exception as exc:
                await message.answer(f"Refused: {exc}")
                await state.clear()
                return
            finally:
                job_state.topic = None
        if result.budget_approval_blocked:
            await message.answer(f"Refused: {result.budget_approval_block_reason}")
        elif result.blocked:
            await message.answer(f"Blocked: {result.block_reason}")
        else:
            lines = [f"Generated {topic!r}."]
            if result.cost_report:
                lines.append(
                    f"Cost: ${result.cost_report['total_spent_usd']:.4f} / "
                    f"${result.cost_report['budget_cap_usd']:.2f} cap"
                )
            if result.verification:
                lines.append(f"Verification: {'PASS' if result.verification['overall_pass'] else 'FAIL'}")
            lines.append("Use /video, /approve, /reject, /publish to continue.")
            await message.answer("\n".join(lines))
        await state.clear()

    @router.message()
    async def handle_message(message: Message, state: FSMContext) -> None:
        user_id = message.from_user.id if message.from_user else None
        if not controller.authorized(user_id):
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
            elif command in ("/mascot", "/mascots"):
                reply = controller.mascots_text()
                reply_markup = _mascot_kb()
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
                    await enter_ideation(message, state, proposal["topic"])
                elif data == "newtopic_cancel":
                    await message.answer("Cancelled.")
                    await state.clear()
            elif current_state == PlanningStates.choosing_idea.state:
                stored = await state.get_data()
                ideas = stored.get("ideas") or []
                topic = stored.get("topic")
                if data.startswith("idea_"):
                    idx = int(data.split("_", 1)[1])
                    chosen_idea = ideas[idx]
                    reply = controller.choose_idea(chosen_idea)
                    await message.answer(reply)
                    await state.update_data(chosen_idea=chosen_idea)
                    await enter_mascot_selection(message, state, topic)
            elif current_state == PlanningStates.choosing_mascot.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data.startswith("mascot_"):
                    mascot_id = data.replace("mascot_", "", 1)
                    mascot = get_mascot(mascot_id)
                    await state.update_data(chosen_mascot_id=mascot.id)
                    await message.answer(f"Selected {mascot.name}.")
                    await enter_retrieval_or_generate(message, state, topic)
            elif current_state == PlanningStates.confirming_retrieval.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "retrieval_confirm":
                    await run_locked_retrieval(message, state, topic)
                elif data == "retrieval_cancel":
                    await message.answer("Skipped retrieval.")
                    await enter_generate_confirm(message, state, topic)
            elif current_state == PlanningStates.confirming_generate.state:
                stored = await state.get_data()
                topic = stored.get("topic")
                if data == "generate_confirm":
                    await run_locked_generate(message, state, topic)
                elif data == "generate_cancel":
                    await message.answer("Cancelled.")
                    await state.clear()
            elif data.startswith("mascot_"):
                mascot_id = data.replace("mascot_", "", 1)
                mascot = get_mascot(mascot_id)
                await message.answer(f"🎭 {mascot.name}\n{mascot.short_desc}\n\nStyle Description:\n{mascot.visual_style}")
        except Exception as exc:
            await message.answer(f"Refused: {exc}")
        await callback.answer()

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
