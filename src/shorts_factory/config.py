"""Resolves CLAUDE.md §1 inputs from the environment (.env), stubbing anything
unset. Every field reports whether it's a real, approved value or a stub —
callers use that to decide whether a step is allowed to make a paid call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

STUB = "stub"


@dataclass(frozen=True)
class ProviderConfig:
    kind: str  # "llm" | "tts" | "image"
    provider: str
    model_or_voice: str

    @property
    def is_stub(self) -> bool:
        return self.provider.strip().lower() in ("", STUB)


@dataclass(frozen=True)
class Settings:
    book_file: str
    output_language: str
    visual_style: str
    budget_cap_usd: float
    budget_cap_is_stub: bool
    music_sfx_source: str

    llm: ProviderConfig
    tts: ProviderConfig
    image: ProviderConfig
    video: ProviderConfig
    search: ProviderConfig
    search_api_key: str

    fal_key: str
    fal_llm_endpoint: str
    tts_voice: str
    image_style: str
    llm_cost_per_script_usd: float
    tts_cost_per_1k_chars_usd: float
    image_cost_per_image_usd: float
    video_cost_per_second_usd: float

    youtube_client_secrets_file: str
    youtube_token_file: str

    telegram_bot_token: str
    telegram_allowed_user_ids: tuple[int, ...]
    # Pinned rather than randomized per video — direct user instruction
    # 2026-09-01 ("this should only be the caption style"), matching the bold
    # orange Impact + heavy black stroke look they supplied. Has a default so
    # existing Settings(...) constructions (tests included) keep working.
    caption_style: str = "comic_punch_orange"
    # Synthesized pops/whooshes under the narration. On by default: the
    # reference short carries ~40 transients across 43s and the pop is what
    # sells an ingredient appearing, since it snaps on with no motion path.
    # Optional extras, defaulted to the stub so that an install which
    # configures neither behaves exactly as it did before they existed:
    # captions fall back to the length-weighted estimate and the video
    # ships with no music bed. Defaulted also means every existing caller
    # constructing Settings(...) keeps working unchanged.
    stt: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(kind="stt", provider=STUB, model_or_voice="")
    )
    music: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(kind="music", provider=STUB, model_or_voice="")
    )
    stt_cost_per_minute_usd: float = 0.0
    music_cost_per_bed_usd: float = 0.0
    sfx_enabled: bool = True
    # Rigged puppet mascot (mascot_rig.py). On by default, but every failure
    # point falls back to the previous static behaviour, so a bad character
    # sheet can never make output worse than it was.
    mascot_rig_enabled: bool = True
    default_mascot_id: str = "mascot_4"
    # "sticker" (default, 2026-08-27): still images pop-in/hard-cut, motion-
    # graphics style — no video provider call, zero video spend. "ai_video":
    # legacy continuous I2V animation via VIDEO_PROVIDER (Kling/Hailuo) —
    # kept working, opt-in only; real Kling output was confirmed frozen for
    # its entire raw duration on half a test video's scenes even with
    # aggressive motion prompting, which the sticker approach sidesteps
    # entirely by not depending on AI-generated motion at all.
    animation_mode: str = "sticker"

    stub_fields: list[str] = field(default_factory=list)

    @property
    def youtube_configured(self) -> bool:
        return bool(self.youtube_client_secrets_file)

    @property
    def any_provider_is_real(self) -> bool:
        return not (
            self.llm.is_stub and self.tts.is_stub and self.image.is_stub
            and self.video.is_stub and self.search.is_stub
        )

    def credential_for(self, provider_config: ProviderConfig) -> str:
        """The right credential for whichever provider name is actually
        configured for this kind. fal.ai is the only paid generation gateway;
        search retains its separate Tavily credential."""
        name = provider_config.provider.strip().lower()
        # Every fal-backed kind must be listed here. A kind that is missing
        # gets an empty credential and fails with "FAL_KEY is required" even
        # though the key is present and correct — which is exactly what
        # happened when stt/music were first added.
        return (
            self.fal_key
            if name == "fal"
            and provider_config.kind in ("llm", "tts", "image", "video", "stt", "music")
            else ""
        )


class BudgetApprovalRequired(Exception):
    """Raised when a real (non-stub) provider is configured but no explicit
    BUDGET_CAP_USD was ever set — a real paid call must never run against a
    silently-defaulted budget. CLAUDE.md §0: 'Do not make paid API calls
    until I approve the providers and budget.' The $2.00 default in
    load_settings() exists only so the free stub pipeline has a cap to test
    against; it is not an approval."""

    def __init__(self, real_providers: list[str]):
        self.real_providers = real_providers
        super().__init__(
            f"Refusing to run: {', '.join(real_providers)} are configured as real "
            "(non-stub) providers, but BUDGET_CAP_USD was never explicitly set — "
            "the pipeline would silently fall back to the $2.00 stub default instead "
            "of requiring your explicit budget approval. Set BUDGET_CAP_USD in .env "
            "before connecting any real provider."
        )


def require_budget_approval_if_paid(settings: Settings) -> None:
    if not settings.any_provider_is_real:
        return  # everything is stub — zero cost, nothing to approve
    if settings.budget_cap_is_stub:
        real = [
            p.kind.upper()
            for p in (settings.llm, settings.tts, settings.image, settings.video, settings.search)
            if not p.is_stub
        ]
        raise BudgetApprovalRequired(real)


def _env(name: str, default: str = "") -> str:
    """Real bug found 2026-08-29: os.environ.get(name, default) only falls
    back to `default` when the key is entirely ABSENT from the environment
    — a .env line like `YOUTUBE_TOKEN_FILE=` (present, explicitly empty)
    returns "" instead, silently skipping a documented default (".env's own
    comment says 'Defaults to <repo root>/.youtube_token.json if unset' —
    an explicitly-blank line is not meaningfully different from unset for
    every other var in this file). Treat present-but-blank the same as
    absent."""
    value = os.environ.get(name, "").strip()
    return value if value else default


def load_settings() -> Settings:
    stub_fields: list[str] = []

    book_file = _env("BOOK_FILE")
    if not book_file:
        stub_fields.append("BOOK_FILE (not needed until Phase 1 retrieval)")

    output_language = _env("OUTPUT_LANGUAGE", "English")

    visual_style = _env(
        "VISUAL_STYLE",
        "Flat 2D hand-inked explainer-cartoon style: thick, slightly uneven black outlines, flat "
        "cel-shading with a light grainy paper texture, muted post-collapse workshop palette (rust "
        "orange, clay red, moss green, bone white, charcoal) on a plain warm off-white background. "
        "Naive, low-detail sticker aesthetic, tutorial/how-to composition centered on the subject. "
        "Recurring mascot: a stocky, round-headed scavenger-tinkerer with simple black dot eyes and "
        "one expressive brow line, wearing salvaged welding goggles pushed up on the forehead and a "
        "patched leather work apron over a plain tunic, short stubby legs, rounded boots, holding a "
        "charcoal marking-stick to point at diagrams.",
    )

    budget_raw = _env("BUDGET_CAP_USD")
    budget_cap_is_stub = not budget_raw
    if budget_cap_is_stub:
        stub_fields.append("BUDGET_CAP_USD (defaulted to $2.00 stub — needs real approval before any paid run)")
    try:
        budget_cap_usd = float(budget_raw) if budget_raw else 2.00
    except ValueError:
        budget_cap_usd = 2.00
        budget_cap_is_stub = True

    music_sfx_source = _env("MUSIC_SFX_SOURCE")
    if not music_sfx_source:
        stub_fields.append("MUSIC_SFX_SOURCE (optional for Phase 0 — no music/SFX track in the skeleton)")

    def provider_cfg(kind: str, provider_var: str, model_var: str) -> ProviderConfig:
        provider = _env(provider_var, STUB)
        model = _env(model_var)
        if provider.lower() in ("", STUB):
            stub_fields.append(f"{provider_var} (using local stub provider — zero network, zero cost)")
        return ProviderConfig(kind=kind, provider=provider, model_or_voice=model)

    llm = provider_cfg("llm", "LLM_PROVIDER", "LLM_MODEL")
    tts = provider_cfg("tts", "TTS_PROVIDER", "TTS_MODEL")
    image = provider_cfg("image", "IMAGE_PROVIDER", "IMAGE_MODEL")
    video = provider_cfg("video", "VIDEO_PROVIDER", "VIDEO_MODEL")
    search = provider_cfg("search", "SEARCH_PROVIDER", "SEARCH_MODEL")
    stt = provider_cfg("stt", "STT_PROVIDER", "STT_MODEL")
    music = provider_cfg("music", "MUSIC_PROVIDER", "MUSIC_MODEL")

    search_api_key = _env("SEARCH_API_KEY") or _env("TAVILY_API_KEY")
    if not search.is_stub and not search_api_key:
        stub_fields.append("SEARCH_API_KEY (SEARCH_PROVIDER is set but no key was provided)")

    youtube_client_secrets_file = _env("YOUTUBE_CLIENT_SECRETS_FILE")
    youtube_token_file = _env("YOUTUBE_TOKEN_FILE", str(REPO_ROOT / ".youtube_token.json"))
    if not youtube_client_secrets_file:
        stub_fields.append("YOUTUBE_CLIENT_SECRETS_FILE (Phase 5 upload not configured)")

    def money(name: str) -> float:
        raw = _env(name, "0")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative number") from exc
        if value < 0:
            raise ValueError(f"{name} must be a non-negative number")
        return value

    telegram_ids: list[int] = []
    for raw_id in _env("TELEGRAM_ALLOWED_USER_IDS").split(","):
        if raw_id.strip():
            telegram_ids.append(int(raw_id.strip()))

    return Settings(
        book_file=book_file,
        output_language=output_language,
        visual_style=visual_style,
        budget_cap_usd=budget_cap_usd,
        budget_cap_is_stub=budget_cap_is_stub,
        music_sfx_source=music_sfx_source,
        llm=llm,
        tts=tts,
        image=image,
        video=video,
        search=search,
        stt=stt,
        music=music,
        search_api_key=search_api_key,
        fal_key=_env("FAL_KEY"),
        fal_llm_endpoint=_env("FAL_LLM_ENDPOINT", "openrouter/router"),
        # Pinned, not randomized — direct user instruction 2026-09-01
        # ("this should only be the caption style"), matching the
        # bold orange Impact + heavy black stroke look they sent.
        caption_style=_env("CAPTION_STYLE", "comic_punch_orange"),
        sfx_enabled=_env("SFX_ENABLED", "true").strip().lower() not in ("0", "false", "no"),
        mascot_rig_enabled=_env("MASCOT_RIG_ENABLED", "true").strip().lower() not in ("0", "false", "no"),
        tts_voice=_env("TTS_VOICE", "Rachel"),
        image_style=_env("IMAGE_STYLE", "digital_illustration/hand_drawn_outline"),
        llm_cost_per_script_usd=money("LLM_COST_PER_SCRIPT_USD"),
        tts_cost_per_1k_chars_usd=money("TTS_COST_PER_1K_CHARS_USD"),
        image_cost_per_image_usd=money("IMAGE_COST_PER_IMAGE_USD"),
        video_cost_per_second_usd=money("VIDEO_COST_PER_SECOND_USD"),
        stt_cost_per_minute_usd=money("STT_COST_PER_MINUTE_USD"),
        music_cost_per_bed_usd=money("MUSIC_COST_PER_BED_USD"),
        youtube_client_secrets_file=youtube_client_secrets_file,
        youtube_token_file=youtube_token_file,
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN"),
        telegram_allowed_user_ids=tuple(telegram_ids),
        default_mascot_id=_env("DEFAULT_MASCOT", "mascot_4"),
        animation_mode=_env("ANIMATION_MODE", "sticker").strip().lower(),
        stub_fields=stub_fields,
    )
