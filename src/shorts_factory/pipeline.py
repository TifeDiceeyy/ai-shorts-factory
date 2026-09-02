"""Phase 0 walking-skeleton orchestrator.

One command, one hardcoded-shape run: topic -> brief -> script -> placeholder
render (zero image spend, proves assembly) -> generated-image render (stub
provider stands in until a real one is approved) -> soap.mp4 -> verification
report. See CLAUDE.md for the full spec this implements.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import assembly, verify
from .config import BudgetApprovalRequired, load_settings, require_budget_approval_if_paid
from .cost_tracker import BudgetExceeded, CostTracker
from .dashboard import review_state
from .mascots import Mascot, generate_custom_mascot, get_mascot, select_mascot_for_story
from .providers.image import get_image_provider
from .providers.llm import LLMResponseFormatError, StubLLMProvider, get_llm_provider
from .providers.music import build_mood_prompt, get_music_provider
from .providers.stt import get_stt_provider
from .providers.tts import get_tts_provider
from .providers.video import get_video_provider
from .providers.fal import FalGateway
from .safety import TopicBlocked, caution_caption, caution_line, enforce_not_blocked
from .schema_validate import ValidationError, validate_brief, validate_script_against_brief

REPO_ROOT = Path(__file__).resolve().parents[2]

# Composited onto only the last scene's final cue — see
# assembly.build_timed_caption_overlays' subscribe_cta_text docstring.
SUBSCRIBE_CTA_TEXT = "SUBSCRIBE!"


class PipelineResult:
    def __init__(self):
        self.topic: str | None = None
        self.safety_class: str | None = None
        self.mascot_id: str | None = None
        self.blocked: bool = False
        self.block_reason: str | None = None
        self.budget_approval_blocked: bool = False
        self.budget_approval_block_reason: str | None = None
        self.budget_exceeded: bool = False
        self.budget_exceeded_reason: str | None = None
        self.artifacts_dir: Path | None = None
        self.verification: dict[str, Any] | None = None
        self.cost_report: dict[str, Any] | None = None
        self.script: dict[str, Any] | None = None
        self.error: str | None = None


def _generate_script_with_fallback(
    llm,
    brief: dict[str, Any],
    language: str,
    visual_style: str,
    cost_tracker: CostTracker,
) -> tuple[dict[str, Any], dict[str, str] | None, dict[str, Any] | None]:
    """Generate a strict script, falling back locally on model format drift.

    The fallback is deterministic and builds one scene directly from each
    verified brief claim, so it never guesses a missing citation ID. It is
    used only after a real provider returned malformed JSON or a script that
    failed strict schema/citation/duration validation; provider/network errors
    still propagate normally.
    """
    rejected_script = None
    try:
        script = llm.generate_script(brief, language, visual_style, cost_tracker)
        rejected_script = script
        validate_script_against_brief(script, brief)
        return script, None, None
    except (LLMResponseFormatError, ValidationError) as exc:
        if getattr(llm, "name", "") == "stub":
            raise
        fallback = StubLLMProvider().generate_script(brief, language, visual_style, cost_tracker)
        validate_script_against_brief(fallback, brief)
        warning = {
            "provider_error_type": type(exc).__name__,
            "provider_error": str(exc),
            "fallback": "deterministic_verified_claim_script",
        }
        return fallback, warning, rejected_script


def get_scene_image_prompt(scene: dict[str, Any], mascot: Mascot) -> str:
    """Builds the base image prompt for a scene. If structured fields are present,
    uses mascot.build_scene_prompt(); otherwise falls back to scene's visual_prompt."""
    if any(k in scene for k in ("scene_type", "mascot_role", "mascot_emotion", "layout", "props", "fx")):
        # `or ""` not `.get(k, "")`: these fields are legitimately null on a
        # character-free scene, and a null reaching build_scene_prompt renders
        # the literal string "None" into the image prompt.
        return mascot.build_scene_prompt(
            scene_role=scene.get("mascot_role") or "",
            action=scene.get("action") or "",
            emotion=scene.get("mascot_emotion") or "",
            props=scene.get("props"),
            layout=scene.get("layout", "auto"),
            scene_type=scene.get("scene_type", "mascot"),
            fx=scene.get("fx"),
            # The script's own description of the shot. Previously dropped
            # whenever any structured field was present, which is what made
            # images generic and off-concept — see build_scene_prompt's
            # `subject` docstring for the measured case.
            subject=scene.get("visual_prompt") or "",
        )
    return scene.get("visual_prompt", mascot.hero_image_prompt)


def get_scene_motion_prompt(scene: dict[str, Any], mascot: Mascot) -> str:
    """Motion prompt for ai_video mode's continuous animation (Kling/
    Hailuo) — deliberately SEPARATE from get_scene_image_prompt() (the
    still-image composition prompt). Reusing the still-image prompt as the
    motion source (the prior approach, see Q2 in the 2026-08-21 external
    review) fixed a prompt/image mismatch but introduced a different bug:
    that prompt only describes a static composition, so the video model had
    nothing telling it what should actually move — real output showed the
    mascot bouncing/hopping while props stayed completely frozen. See
    mascots.Mascot.build_scene_motion_prompt()'s docstring for the fix."""
    # Prefer the scene's OWN authored motion directive. The generic builder
    # below matches a scene against 8 fixed keyword categories
    # (smoke/water/fire/...) and can only produce text like "steam gently
    # rising" — which describes nothing actually in the shot, and is why real
    # Kling clips came back frozen for their whole duration. The script's
    # `motion` field is written by the model that knows this topic, this
    # claim, these props and this action, so it is the only source that can
    # say what specifically should move. Falls back to the generic builder
    # for older scripts that predate the field.
    authored = (scene.get("motion") or "").strip()
    if authored:
        return authored
    return mascot.build_scene_motion_prompt(
        scene_type=scene.get("scene_type", "mascot"),
        props=scene.get("props"),
        fx=scene.get("fx"),
        action=scene.get("action", ""),
        narration=scene.get("narration", ""),
    )


def _hero_cache_key(mascot: Mascot, image_model: str, image_style: str) -> str:
    """Short hash covering everything that changes what the hero image
    looks like: the mascot's own hero_prompt text, the image model, and the
    style preset. mascot.id alone isn't enough — artifacts/<topic>/_work
    persists across separate runs of the same topic, so switching
    IMAGE_MODEL (e.g. Recraft -> Nano Banana) or editing a mascot's
    hero_prompt while keeping the same mascot_id would otherwise silently
    keep reusing a hero image rendered under the old model/prompt."""
    raw = f"{mascot.hero_image_prompt}|{image_model}|{image_style}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def _get_or_create_hero_image(image_provider, mascot: Mascot, hero_path: Path, cost_tracker: CostTracker) -> Path:
    """Generates the shared hero character image once and reuses it on every
    later call (within or across run_pipeline/regenerate_scene) — this is
    what keeps the mascot's appearance consistent across every animated
    scene. Lazy: a script made entirely of ingredient_grid/process_action
    scenes never calls this and never pays for a hero image it wouldn't use.

    hero_path MUST be keyed by both mascot.id AND _hero_cache_key() by the
    caller (e.g. generated/hero_mascot_4_a1b2c3d4e5.png, not a fixed
    hero.png or a mascot.id-only hero_mascot_4.png) — artifacts/<topic>
    persists across separate runs of the same topic, so a filename that
    doesn't account for the mascot's prompt text and image model would let
    a stale image rendered under a different one get silently reused
    indefinitely."""
    if not hero_path.exists():
        hero_path.parent.mkdir(parents=True, exist_ok=True)
        image_provider.generate_scene_image({"visual_prompt": mascot.hero_image_prompt}, "hero", hero_path, cost_tracker)
    return hero_path


def _align_scene_captions(settings, scene_audio, cost_tracker) -> list[list] | None:
    """Word timings per scene, or None to keep the estimate.

    Every failure mode — stub provider, provider error, an alignment whose
    word count doesn't match the narration — resolves to "use the estimate".
    Captions are a finishing touch; none of this is worth losing a paid
    render over.
    """
    if settings.stt.is_stub:
        return None
    try:
        provider = get_stt_provider(
            settings.stt.provider,
            settings.credential_for(settings.stt),
            settings.stt.model_or_voice,
            settings.stt_cost_per_minute_usd,
        )
        timings = [provider.align(audio.path, cost_tracker) for audio in scene_audio]
    except Exception as err:  # noqa: BLE001 - deliberately non-fatal
        print(f"caption alignment unavailable ({err}) — using estimated timings", file=sys.stderr)
        return None
    return timings if any(timings) else None


def _get_or_create_music_bed(settings, topic: str, workdir: Path, cost_tracker) -> Path | None:
    """The topic's music bed, generated once and reused.

    An explicit MUSIC_SFX_SOURCE always wins — if someone has supplied a
    real track, never spend money generating one over the top of it.
    """
    explicit = (settings.music_sfx_source or "").strip()
    if explicit and Path(explicit).exists():
        return Path(explicit)
    if settings.music.is_stub:
        return None
    bed_path = workdir / "music_bed.wav"
    if bed_path.exists():
        return bed_path
    try:
        provider = get_music_provider(
            settings.music.provider,
            settings.credential_for(settings.music),
            settings.music.model_or_voice,
            settings.music_cost_per_bed_usd,
        )
        return provider.generate_bed(
            build_mood_prompt(topic, settings.visual_style), bed_path, cost_tracker
        )
    except Exception as err:  # noqa: BLE001 - deliberately non-fatal
        print(f"music bed unavailable ({err}) — rendering without one", file=sys.stderr)
        return None


def _scene_base_image_path(
    image_provider,
    mascot: Mascot,
    hero_path: Path,
    scene: dict[str, Any],
    scene_index: int,
    generated_dir: Path,
    cost_tracker: CostTracker,
    character_free: bool = False,
    shot_index: int = 0,
) -> Path:
    """Generates one image for a scene — shot_index selects which shot of it.

    shot_index 0 is the scene's main image. Higher indices are additional
    SHOTS within the same scene, framed differently via
    SHOT_FRAMING_VARIANTS so cutting between them reads as an edit.

    character_free forces the character OUT of the render, for scenes where
    the rigged puppet (mascot_rig.py) will supply an ANIMATED mascot on top
    instead. Without it the scene would carry a second, frozen copy of the
    character underneath the animated one.

    ingredient_grid/process_action scenes render with no character at all —
    build_scene_prompt() omits the mascot from these entirely. Every other
    scene_type IS a mascot scene: it also renders fresh, per-scene (so pose,
    composition, and layout can genuinely vary — small-and-pointing,
    big-and-reacting, split-canvas, ...), but ANCHORED on the shared hero
    image via image-to-image editing when the model supports it
    (supports_reference_edit), so the character stays recognizable across
    scenes instead of drifting.

    Reusing the literal hero image for every mascot scene (the prior
    approach) guaranteed character consistency but collapsed every mascot
    scene into one identical frozen pose/composition — confirmed for real
    2026-08-21 against actual generated output, flagged by the user
    ("always big... should sometimes be small... sometimes not in the
    scene at all"). Editing from the hero image gets both: real per-scene
    composition variety AND character consistency, instead of trading one
    for the other."""
    stype = scene.get("scene_type", "mascot")
    if character_free:
        # Reuse the character-free branch that ingredient_grid/process_action
        # already take, rather than inventing a second no-mascot prompt.
        scene_prompt = get_scene_image_prompt({**scene, "scene_type": "process_action"}, mascot)
    else:
        scene_prompt = get_scene_image_prompt(scene, mascot)
    if shot_index:
        scene_prompt = f"{scene_prompt} {SHOT_FRAMING_VARIANTS[shot_index % len(SHOT_FRAMING_VARIANTS)]}"
    suffix = "" if not shot_index else f"_{shot_index}"
    out_path = generated_dir / "raw" / f"raw_{scene_index:02d}{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference_image_path = None
    if not character_free and stype not in ("ingredient_grid", "process_action"):
        reference_image_path = _get_or_create_hero_image(image_provider, mascot, hero_path, cost_tracker)
    # Keep the plain int id for a scene's main image: callers and tests
    # distinguish real per-scene renders from the hero/rig-sheet calls by
    # that type, and stringifying every id silently hid them all. Only the
    # extra shots get the "<index>_<shot>" form.
    image_id = scene_index if not shot_index else f"{scene_index}{suffix}"
    image_provider.generate_scene_image(
        {"visual_prompt": scene_prompt}, image_id, out_path, cost_tracker,
        reference_image_path=reference_image_path,
    )
    return out_path


# Appended to the prompt for a scene's SECOND and THIRD shot. Two images of
# the same subject drawn from the same prompt come back near-identical, and
# cutting between near-identical frames reads as a glitch, not an edit — the
# shot has to actually change. These say how, without changing what the shot
# is about.
SHOT_FRAMING_VARIANTS = (
    "",
    "Draw this as a closer view: the key object or detail large in frame, the rest cropped out.",
    "Draw this as a wider view: the whole scene smaller in frame with more empty space around it.",
)


# Direct user feedback (2026-08-28): once a real clip's motion runs out
# mid-scene, the next beat should also be real motion where reasonable, not
# an immediate fall back to a static hold. Bounded at 2 — not unlimited —
# specifically to cap the worst-case cost impact (each extra clip is a real
# paid Kling/Hailuo call); the static sticker-style cut-in (see
# assembly.build_scene_video_segment_from_clip) still absorbs whatever's
# left after that.
MAX_REAL_CLIPS_PER_SCENE = 2

# Generation-speed fix (2026-08-29): bounded worker counts for the
# per-scene image/TTS calls below, matching the same "cap concurrency,
# don't flood the provider" reasoning as MAX_REAL_CLIPS_PER_SCENE's own
# ThreadPoolExecutor above it — lighter/faster calls than video generation,
# so a higher cap than that one's is reasonable.
IMAGE_MAX_WORKERS = 3
TTS_MAX_WORKERS = 3


def _render_scene_clips(
    video_provider,
    scene: dict[str, Any],
    source_image_path: Path,
    target_duration: float,
    scene_index: int,
    generated_dir: Path,
    cost_tracker: CostTracker,
    motion_prompt: str,
) -> list[Path]:
    """Generates up to MAX_REAL_CLIPS_PER_SCENE real video clips for one
    scene, continuing from where the previous clip's real motion left off
    (its own last frame, extracted locally — zero extra cost) rather than
    re-animating the same starting pose. Stops early once the accumulated
    real motion (after each clip's own leading-freeze-skip and the same
    MAX_CLIP_STRETCH_FACTOR-capped stretch build_scene_video_segment_from_clip
    will apply) already covers target_duration — no point paying for a
    second clip nothing will end up using. Returns every clip actually
    generated, in order; build_scene_video_segment_from_clip does the same
    per-clip accounting again when it assembles them, so this is a cost
    decision, not a duplicate of the assembly-time timing logic."""
    clip_paths: list[Path] = []
    remaining = target_duration
    current_source = source_image_path

    for attempt in range(MAX_REAL_CLIPS_PER_SCENE):
        tmp_clip_path = generated_dir / "raw" / f"clip_{scene_index:02d}_{attempt}.mp4"
        clip_path = video_provider.generate_scene_video(
            scene, current_source, scene_index, tmp_clip_path, cost_tracker, motion_prompt=motion_prompt
        )
        clip_paths.append(clip_path)

        usable = assembly.usable_clip_seconds(clip_path)
        stretch_factor = min(remaining / usable, assembly.MAX_CLIP_STRETCH_FACTOR)
        played = usable * stretch_factor
        remaining = max(0.0, remaining - played)

        if remaining <= assembly.CUT_IN_MIN_PAD_SECONDS or attempt == MAX_REAL_CLIPS_PER_SCENE - 1:
            break

        last_frame_path = generated_dir / "raw" / f"clip_{scene_index:02d}_{attempt}_last.png"
        current_source = assembly.extract_last_frame(clip_path, last_frame_path)

    return clip_paths


# Scene types where the rigged mascot replaces the drawn-in character.
# ingredient_grid/process_action already render character-free by design.
RIG_SCENE_TYPES = ("mascot", "mascot_reaction", "split_canvas")


# --- Which scenes are worth paying to animate -------------------------------
#
# Measured on the reference short (2026-09-01): 9 of its 15 shots carry real
# animation and the other 6 are held drawings. Animating everything is both
# off-style and the dominant cost — video is 83% of a run's spend ($3.60 of
# $4.34 at 16 clips), so every scene left static saves ~$0.22 AND removes a
# chance for the model to invent motion the scene never called for.
ANIMATED_SCENE_FRACTION = 0.6

# A 2x2 grid of labelled icons has nothing to move; animating one only lets
# the model drift and morph the items. process_action is the opposite case —
# it exists to depict a physical process, so it always earns a clip.
ALWAYS_STATIC_SCENE_TYPES = ("ingredient_grid",)
ALWAYS_ANIMATED_SCENE_TYPES = ("process_action",)

# Scene types that render the mascot large and centred, so its face — and
# therefore its mouth — is big enough on screen to read.
#
# These are never animated (2026-09-02). Kling opens the mascot's mouth
# partway through any clip containing a legible face: four separate levers
# were tried (prompt wording, negative prompt, a closed-mouth source image,
# and cfg_scale in both directions) and the best of them only held the mouth
# shut for roughly the first 60% of a clip. Measured on real base images,
# split_canvas renders the mascot substantially smaller than
# mascot/mascot_reaction do, and Kling further reframes toward a close-up
# once it starts from a large subject — so the reliable fix is to spend
# clips on the smaller-mascot and character-free shots instead.
#
# This costs nothing: on the 15-scene script there are 7 split_canvas + 2
# process_action scenes, exactly filling the 9-clip budget. It is also
# closer to the reference short, which never animates a close-up — its
# mascot sits at ~25% of frame height in every single shot.
FACE_CLOSEUP_SCENE_TYPES = ("mascot", "mascot_reaction")

# Verbs describing something physically moving or transforming on screen —
# the shots animation actually buys something on. Worth more than a gesture.
_TRANSFORM_MOTION_VERBS = (
    "pour", "mix", "stir", "shake", "fall", "drop", "rise", "spin", "rotate",
    "flow", "bubble", "boil", "erupt", "crack", "shatter", "crumble", "collapse",
    "melt", "burn", "smoke", "splash", "spray", "swirl", "hammer", "carve", "dig",
    "assemble", "stack", "tumble", "drip", "sink", "float", "fly", "grow", "shrink",
    "roll", "slide", "throw", "jump", "dance", "walk", "run", "climb", "spill",
    "harden", "set", "cure", "dry", "freeze", "crush", "grind", "sift", "knead",
)
# Character gestures: real movement, but small and localized. Enough to make a
# scene animatable, not enough to outrank a scene where the subject transforms.
_GESTURE_MOTION_VERBS = (
    "raise", "gesture", "tilt", "point", "bow", "lean", "nod", "recoil", "wave",
    "hold", "bring", "swing", "turn", "reach", "lift", "push", "pull", "open",
    "close", "step", "shrug", "clap", "grab", "place", "set down", "present",
)

# Matched on WORD BOUNDARIES, with a trailing -s/-es/-ed/-ing allowed. Plain
# substring matching silently fired on the wrong words — "run" inside
# "around", "roll" inside "controlled", "set" inside "sunset" — which would
# have bought paid clips for scenes that describe no motion at all.
_MOTION_VERB_RE_CACHE: dict[tuple[str, ...], "re.Pattern[str]"] = {}


def _motion_verb_pattern(verbs: tuple[str, ...]) -> "re.Pattern[str]":
    cached = _MOTION_VERB_RE_CACHE.get(verbs)
    if cached is None:
        alternation = "|".join(re.escape(v) for v in sorted(verbs, key=len, reverse=True))
        cached = re.compile(rf"\b(?:{alternation})(?:s|es|ed|ing)?\b")
        _MOTION_VERB_RE_CACHE[verbs] = cached
    return cached


def scene_motion_score(scene: dict[str, Any]) -> int:
    """How much real movement this scene's own text asks for.

    Transform verbs (the subject itself moves or changes) count double;
    character gestures count single. A scene that says "pours the mixture as
    steam rises" outranks one that says "raises an arm", which in turn
    outranks "stands looking concerned" — that last scores 0 and is never
    animated, because paying to animate a shot whose script describes no
    movement is exactly what produced invented, off-concept motion.
    """
    haystack = " ".join(
        str(scene.get(field) or "") for field in ("motion", "action", "fx", "sfx")
    ).lower()
    transforms = set(_motion_verb_pattern(_TRANSFORM_MOTION_VERBS).findall(haystack))
    gestures = set(_motion_verb_pattern(_GESTURE_MOTION_VERBS).findall(haystack))
    return 2 * len(transforms) + len(gestures)


def choose_animated_scenes(
    scenes: list[dict[str, Any]], fraction: float = ANIMATED_SCENE_FRACTION
) -> list[bool]:
    """Decide, per scene, whether to spend a video clip on it.

    Scene type settles the obvious cases; everything else is ranked by
    scene_motion_score and the best-scoring scenes fill the remaining budget.
    Ties break by scene order, so the choice is deterministic for a given
    script — the determinism tests depend on that.

    A scene that scores zero is never animated even if the budget has room:
    paying to animate a shot whose own script describes no movement is what
    produced the invented, off-concept motion in the first place. Neither is
    a scene that puts the mascot's face large on screen — see
    FACE_CLOSEUP_SCENE_TYPES for why, and note that leaving budget unspent
    is the intended outcome there, not a bug.
    """
    budget = max(1, round(len(scenes) * fraction))
    decisions = [False] * len(scenes)

    forced = [
        i for i, s in enumerate(scenes)
        if (s.get("scene_type") or "") in ALWAYS_ANIMATED_SCENE_TYPES
    ]
    never = set(ALWAYS_STATIC_SCENE_TYPES) | set(FACE_CLOSEUP_SCENE_TYPES)
    eligible = [
        i for i, s in enumerate(scenes)
        if i not in forced
        and (s.get("scene_type") or "") not in never
        and scene_motion_score(s) > 0
    ]

    for i in forced:
        decisions[i] = True
    remaining = max(0, budget - len(forced))
    ranked = sorted(eligible, key=lambda i: (-scene_motion_score(scenes[i]), i))
    for i in ranked[:remaining]:
        decisions[i] = True
    return decisions


def _build_mascot_rig(settings, mascot, generated_dir: Path, image_provider, cost_tracker):
    """Generate + slice the mascot's parts once per video.

    Returns None on ANY problem, and the caller then renders exactly as it did
    before the rig existed. That fallback is the point: a character sheet is a
    generative output and cannot be relied on, so the rig must be a strict
    improvement or a no-op — never a regression.
    """
    if not settings.mascot_rig_enabled:
        return None
    try:
        from .mascot_rig import PART_NAMES, character_sheet_prompt, extract_parts_from_sheet
        from PIL import Image as _Image

        sheet = generated_dir / f"rig_sheet_{mascot.id}.png"
        if not sheet.exists():
            image_provider.generate_scene_image(
                {"visual_prompt": character_sheet_prompt(mascot)}, "rig_sheet", sheet, cost_tracker,
            )
        parts_dir = generated_dir / f"rig_parts_{mascot.id}"
        paths = extract_parts_from_sheet(sheet, parts_dir)
        missing = [n for n in PART_NAMES if n not in paths]
        if missing:
            print(f"mascot rig: sheet yielded no {missing} — falling back to static mascot")
            return None
        return {name: _Image.open(paths[name]).convert("RGBA") for name in PART_NAMES}
    except Exception as exc:                     # noqa: BLE001 - never fail a paid run over the rig
        print(f"mascot rig unavailable ({exc}) — falling back to static mascot")
        return None



def run_pipeline(
    topic: str,
    idea: dict[str, Any] | None = None,
    artifacts_root: Path | None = None,
    target_seconds: float | None = None,
) -> PipelineResult:
    """idea, if given, is a {concept, angle, chosen_hook, payoff} dict that
    steers script framing (brief_builder.build_brief_from_citations), never
    facts. No caller currently populates it (the interactive idea-selection
    step was removed 2026-08-28); None preserves the topic-only behavior.

    The mascot is always chosen automatically from the story (see
    select_mascot_for_story()/generate_custom_mascot() below) — there is no
    way to override it manually; that override existed once but was removed
    per explicit user request (2026-08-28).

    artifacts_root, if given, overrides where output is written (default
    REPO_ROOT / "artifacts") — tests MUST pass a tmp_path here. Without
    this, a test run and a real production run for the same topic write to
    the exact same directory and silently clobber each other (confirmed:
    this happened for real 2026-08-17, a stub test run overwrote a $1.72
    real animated video with a free demo one)."""
    result = PipelineResult()
    result.topic = topic

    settings = load_settings()

    # --- Budget-approval gate: a real (non-stub) provider must never run
    # against a silently-defaulted budget cap. Must run before the safety
    # gate even touches a provider — this is a config-level refusal, not a
    # per-topic one. ---
    try:
        require_budget_approval_if_paid(settings)
    except BudgetApprovalRequired as e:
        result.budget_approval_blocked = True
        result.budget_approval_block_reason = str(e)
        return result

    # --- Safety gate: must run before brief/script/render, and before any
    # provider is touched. A RED topic must not reach this far. ---
    try:
        safety_class = enforce_not_blocked(topic)
    except TopicBlocked as e:
        result.blocked = True
        result.block_reason = str(e)
        return result
    result.safety_class = safety_class.value

    artifacts_dir = (artifacts_root or REPO_ROOT / "artifacts") / topic
    workdir = artifacts_dir / "_work"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result.artifacts_dir = artifacts_dir

    # "sticker" (default): still images pop-in/hard-cut, no video provider
    # call at all. "ai_video" (opt-in, legacy): continuous I2V animation via
    # VIDEO_PROVIDER. See Settings.animation_mode's docstring for why sticker
    # is now the default.
    sticker_mode = settings.animation_mode == "sticker" and not settings.image.is_stub
    ai_video_mode = settings.animation_mode == "ai_video" and not settings.video.is_stub
    animate = sticker_mode or ai_video_mode
    if ai_video_mode and settings.image.is_stub:
        raise ValueError(
            "VIDEO_PROVIDER is real but IMAGE_PROVIDER is stub — animation needs a real "
            "hero image to animate; set IMAGE_PROVIDER=fal too"
        )

    cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
    uses_fal = any(
        p.provider.strip().lower() == "fal" for p in (settings.llm, settings.tts, settings.image, settings.video)
    )
    fal_gateway = FalGateway(settings.fal_key) if uses_fal else None

    stage = "brief"
    try:
        # Real runs must consume a verified citation store OR sufficient
        # local brain coverage. Hand-authored briefs remain available only
        # to the zero-cost Phase 0 renderer test.
        citation_path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
        if settings.any_provider_is_real:
            brief = None
            # "Questions should go through brain first" — try the local,
            # zero-cost book knowledge base before falling back to real,
            # paid Tavily retrieval. See brain_integration.brain_covers_topic's
            # docstring for the (real but imperfect, keyword-based) coverage
            # heuristic; when it's wrong, this just falls through to the
            # citations.json path below, unchanged from before.
            from .brain_integration import brain_covers_topic, build_brief_from_brain, load_brain

            brain = load_brain()
            if brain is not None:
                covered, research = brain_covers_topic(brain, topic)
                if covered:
                    brief = build_brief_from_brain(
                        topic, research, safety_class.value, caution=caution_line(topic), idea=idea,
                    )
                    # The brain path builds its own brief, so the requested
                    # length has to be stamped on here too — otherwise asking
                    # for 30s does nothing whenever the brain covers the topic.
                    if target_seconds:
                        brief["target_seconds"] = float(target_seconds)
                    # One scene per claim, so a short brief is literally a
                    # short video. Real case 2026-09-01: brain covers "roman
                    # concrete" well enough to use, but its extractive
                    # fact-picking only yielded 9 usable facts, so the video
                    # came out 9 scenes instead of the intended 15 while the
                    # committed citation store for the same topic held 63
                    # verified claims. Brain stays FIRST (it's free), but when
                    # it can't fill the scene budget, fall back to citations —
                    # and only if those genuinely do better, so a thin
                    # citation store can never make the video WORSE than the
                    # free brain brief we already have in hand.
                    from .brief_builder import (
                        MAX_CLAIMS_FOR_BRIEF,
                        InsufficientVerifiedClaims,
                        build_brief_from_citations,
                    )

                    if len(brief.get("claims", [])) < MAX_CLAIMS_FOR_BRIEF and citation_path.exists():

                        try:
                            richer = build_brief_from_citations(
                                topic,
                                json.loads(citation_path.read_text(encoding="utf-8")),
                                safety_class.value,
                                caution=caution_line(topic),
                                idea=idea,
                                target_seconds=target_seconds,
                            )
                        except InsufficientVerifiedClaims:
                            richer = None
                        if richer is not None and len(richer.get("claims", [])) > len(brief["claims"]):
                            brief = richer
            if brief is None:
                if not citation_path.exists():
                    raise FileNotFoundError(
                        f"real generation requires verified citations at {citation_path} "
                        "(brain coverage was insufficient for this topic); run retrieve.sh first"
                    )
                from .brief_builder import build_brief_from_citations

                citation_store = json.loads(citation_path.read_text(encoding="utf-8"))
                brief = build_brief_from_citations(
                    topic,
                    citation_store,
                    safety_class.value,
                    caution=caution_line(topic),
                    idea=idea,
                    target_seconds=target_seconds,
                )
        else:
            brief_path = REPO_ROOT / "data" / topic / f"{topic}.brief.json"
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
        validate_brief(brief)
        if brief["safety_class"] != safety_class.value:
            raise ValidationError(
                f"brief declares safety_class={brief['safety_class']!r} but the "
                f"classifier says {safety_class.value!r} — refusing to proceed on a mismatch"
            )

        # --- LLM provider (stub by default) — constructed here, before
        # mascot resolution, because select_mascot_for_story()'s no-match
        # case needs it to design a brand-new custom mascot. ---
        llm = get_llm_provider(
            settings.llm.provider,
            settings.credential_for(settings.llm),
            settings.llm.model_or_voice,
            settings.llm_cost_per_script_usd,
            gateway=fal_gateway,
            endpoint=settings.fal_llm_endpoint,
        )

        # Mascot resolution happens here (after the brief exists, not right
        # after the safety gate) so story-matching can use the brief's
        # concept/angle/claims text, not just the bare topic string. Always
        # automatic — no human override (see run_pipeline's docstring).
        mascot = select_mascot_for_story(topic, brief=brief)
        if mascot is None:
            # Nothing among the 5 registered mascots or any previously
            # generated custom one fits this topic at all — design and
            # persist a brand-new one instead of forcing an unrelated
            # mascot onto a story it doesn't suit.
            mascot = generate_custom_mascot(topic, brief, llm, cost_tracker)
        result.mascot_id = mascot.id

        effective_visual_style = (
            f"{mascot.name} Template. Style DNA: {mascot.visual_style}. "
            f"Scene Adaptive Direction: {mascot.scene_role_template}"
        )
        stage = "script_generation"
        script, script_warning, rejected_script = _generate_script_with_fallback(
            llm, brief, settings.output_language, effective_visual_style, cost_tracker
        )
        if rejected_script is not None:
            rejected_out = artifacts_dir / f"{topic}.script.rejected.json"
            rejected_out.write_text(json.dumps(rejected_script, indent=2), encoding="utf-8")
        if script_warning is not None:
            warning_out = artifacts_dir / "script-generation-warning.json"
            warning_out.write_text(json.dumps(script_warning, indent=2), encoding="utf-8")
        script["mascot_id"] = mascot.id
        # Persisted (not baked into any scene's caption — see
        # assembly.assemble()'s caution_text param) so regenerate_scene can
        # re-apply the same badge if the last scene gets regenerated.
        # Overwriting scenes[-1]["caption"] with this used to silently
        # replace every yellow topic's real payoff line (confirmed for real
        # 2026-08-21) — the badge is composited on top of the real caption
        # instead, never instead of it.
        script["caution_text"] = caution_caption(topic)
        # One caption style, persisted so regenerate_scene reads the same
        # value back and a single-scene regeneration never mismatches the
        # rest of the video. No longer randomized per video: the user picked
        # one look (bold orange Impact, heavy black stroke) and asked for it
        # everywhere. Override with CAPTION_STYLE in .env.
        script["caption_style"] = settings.caption_style
        # Keep the provider result available when strict validation refuses
        # it. Previously the only copy was discarded, leaving Telegram with
        # an error but no artifact that explained which field was malformed.
        script_draft_out = artifacts_dir / f"{topic}.script.draft.json"
        script_draft_out.write_text(json.dumps(script, indent=2), encoding="utf-8")
        stage = "script_validation"
        validate_script_against_brief(script, brief)
        result.script = script

        script_out = artifacts_dir / f"{topic}.script.json"
        script_out.write_text(json.dumps(script, indent=2), encoding="utf-8")

        # --- Narration: synthesized ONCE per scene, shared across every render
        # stage below. A real TTS provider must never be charged twice for the
        # same narration just because the pipeline renders the video twice.
        # Actual audio duration (measured via ffprobe inside synthesize_scenes)
        # is the source of truth from here on — the script's `duration` field is
        # only ever a nominal estimate, and real narration will not match it
        # exactly. Everything downstream (video segments, captions, the
        # verification target duration) is driven by the measured value so
        # nothing can drift out of sync with what's actually on the timeline. ---
        stage = "tts"

        # A factory, not a shared instance — synthesize_scenes() now renders
        # scenes concurrently, and fal_client's synchronous client is not
        # thread-safe (same constraint documented on render_clip/
        # render_scene_image above); each worker needs its own gateway.
        def make_tts_provider():
            return get_tts_provider(
                settings.tts.provider,
                settings.credential_for(settings.tts),
                settings.tts.model_or_voice,
                settings.tts_voice,
                settings.tts_cost_per_1k_chars_usd,
                gateway=FalGateway(settings.fal_key) if uses_fal else None,
            )

        scene_audio = assembly.synthesize_scenes(make_tts_provider, script["scenes"], workdir / "audio", cost_tracker)

        # Caption timing from the actual audio, where an STT provider is
        # configured. Falls back silently to the length-weighted estimate:
        # slightly-off captions are worth far more than a failed render, so
        # nothing here is allowed to break the run.
        scene_word_timings = _align_scene_captions(settings, scene_audio, cost_tracker)

        # One music bed per topic, cached in the topic's own workdir. The
        # reference runs a continuous bed under the whole voiceover, and
        # that bed is also what gives it a 1.7 LU loudness range where ours
        # measures 3.5 (verified 2026-09-02).
        music_bed_path = _get_or_create_music_bed(settings, topic, workdir, cost_tracker)
        actual_durations = [a.duration for a in scene_audio]
        scripted_durations = [a.scripted_duration for a in scene_audio]
        actual_total = sum(actual_durations)

        # --- Captions (timing derived from ACTUAL audio duration, not the script's estimate) ---
        captions_srt = artifacts_dir / "captions.srt"
        assembly.write_captions_srt(script["scenes"], actual_durations, captions_srt)

        # --- Stage 1: deterministic placeholder render — zero image spend.
        # Proves FFmpeg/assembly correctness before any image provider is touched. ---
        placeholder_dir = workdir / "placeholder"
        placeholder_mp4 = artifacts_dir / f"{topic}.placeholder.mp4"
        stage = "placeholder_assembly"
        placeholder_result = assembly.assemble(
            scenes=script["scenes"],
            frame_source=lambda i, scene: assembly.solid_color_frame(i),
            audio=scene_audio,
            workdir=placeholder_dir,
            out_mp4=placeholder_mp4,
            caption_style=script["caption_style"],
            caution_text=script["caution_text"],
        )
        assembly.write_captions_meta(
            script["scenes"], actual_durations, placeholder_result["caption_boxes"],
            artifacts_dir / "captions.placeholder.meta.json", scripted_durations=scripted_durations,
        )

        # --- Stage 2: hybrid hero + per-scene generation (stub image provider stands in
        # until a real one is approved) — this produces the final video.
        # 1. ONE hero character image is generated once from mascot.hero_prompt into
        #    generated/hero_<mascot.id>.png (keyed by mascot so switching mascots can
        #    never silently reuse a stale image from a different one).
        # 2. Every scene renders its OWN base image, one call per scene (see
        #    _scene_base_image_path): ingredient_grid/process_action scenes render
        #    with no character; every other scene_type is a mascot scene, edited FROM
        #    the hero image (image-to-image, when supported) so pose/composition can
        #    genuinely vary per scene (small-and-pointing, big-and-reacting,
        #    split-canvas, ...) while the character itself stays recognizable.
        # 3. When animation is enabled (animate=True), each base image is then
        #    animated into a video clip via FalVideoProvider using the LLM's
        #    visual_prompt as motion guidance.
        # Cost model: 1 hero image + 1 image per scene + (N video clips, if animating). ---
        image_provider = get_image_provider(
            settings.image.provider,
            settings.credential_for(settings.image),
            settings.image.model_or_voice,
            settings.image_cost_per_image_usd,
            gateway=fal_gateway,
            # Not mascot.visual_style here: get_scene_image_prompt()/
            # build_scene_prompt() already embed the right style into every
            # prompt per scene_type (full character description for mascot
            # scenes, a character-free style for ingredient_grid/
            # process_action). Force-appending the character-laden
            # mascot.visual_style here too bled the mascot into
            # ingredient/process shots that are supposed to show zero
            # character (confirmed for real 2026-08-21).
            visual_style="" if mascot else settings.visual_style,
            style_preset=settings.image_style,
        )
        generated_dir = workdir / "generated"
        final_mp4 = artifacts_dir / f"{topic}.mp4"

        stage = "image_video_generation" if animate else "image_generation"
        if sticker_mode:
            # No video provider at all — see assembly.assemble_stickers()'s
            # docstring for why. Base images are the same per-scene renders
            # the ai_video path used as I2V source frames; here they ARE the
            # final visual content, popped in and held rather than animated.
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"

            # Generation-speed fix, 2026-08-29: per-scene image calls used to
            # run one at a time (a plain for-loop) — real wall time for a
            # 6-scene video is dominated by these sequential network calls,
            # not by the (fast, local) ffmpeg assembly after them. Parallelized
            # the same way _render_scene_clips already parallelizes ai_video's
            # clip rendering: the shared hero reference must exist BEFORE any
            # worker starts (created here, synchronously, exactly once — the
            # same lazy condition _scene_base_image_path itself uses, so a
            # script with zero mascot-type scenes still never pays for a hero
            # it wouldn't use), and every worker gets its OWN image provider
            # (and FalGateway) rather than sharing one across threads —
            # fal_client's synchronous client is not thread-safe (the same
            # constraint _render_scene_clips's own worker_gateway already
            # documents for video).
            if any(
                scene.get("scene_type", "mascot") not in ("ingredient_grid", "process_action")
                for scene in script["scenes"]
            ):
                _get_or_create_hero_image(image_provider, mascot, hero_path, cost_tracker)

            # One character sheet per video buys unlimited articulated motion.
            # Returns None on any problem, and every use below is guarded, so
            # the run degrades to exactly the previous static behaviour.
            rig_parts = _build_mascot_rig(
                settings, mascot, generated_dir, image_provider, cost_tracker,
            )

            def render_scene_image(i: int, scene: dict[str, Any]) -> Path:
                worker_gateway = FalGateway(settings.fal_key) if uses_fal else None
                worker_image_provider = get_image_provider(
                    settings.image.provider,
                    settings.credential_for(settings.image),
                    settings.image.model_or_voice,
                    settings.image_cost_per_image_usd,
                    gateway=worker_gateway,
                    visual_style="" if mascot else settings.visual_style,
                    style_preset=settings.image_style,
                )
                return _scene_base_image_path(
                    worker_image_provider, mascot, hero_path, scene, i, generated_dir, cost_tracker,
                    # With a rig available the mascot is animated on top, so
                    # the background must NOT also contain a drawn one —
                    # otherwise the scene carries a second, frozen character.
                    character_free=bool(rig_parts)
                    and scene.get("scene_type", "mascot") in RIG_SCENE_TYPES,
                )

            base_image_paths: list[Path | None] = [None] * len(script["scenes"])
            with ThreadPoolExecutor(max_workers=min(IMAGE_MAX_WORKERS, len(script["scenes"]))) as executor:
                pending = {
                    executor.submit(render_scene_image, i, scene): i
                    for i, scene in enumerate(script["scenes"])
                }
                for future in as_completed(pending):
                    base_image_paths[pending[future]] = future.result()

            def sticker_image_source(i: int, _scene: dict[str, Any]) -> Path:
                path = base_image_paths[i]
                if path is None:
                    raise RuntimeError(f"scene image {i} did not complete")
                return path

            generated_result = assembly.assemble_stickers(
                scenes=script["scenes"],
                image_source=sticker_image_source,
                audio=scene_audio,
                workdir=generated_dir,
                out_mp4=final_mp4,
                caption_style=script["caption_style"],
                caution_text=script["caution_text"],
                subscribe_cta_text=SUBSCRIBE_CTA_TEXT,
                rig_parts=rig_parts,
                rig_scene_types=RIG_SCENE_TYPES if rig_parts else (),
                sfx_enabled=settings.sfx_enabled,
                music_path=music_bed_path,
                word_timings=scene_word_timings,
            )
        elif ai_video_mode:
            video_provider = get_video_provider(
                settings.video.provider,
                settings.credential_for(settings.video),
                settings.video.model_or_voice,
                settings.video_cost_per_second_usd,
                gateway=fal_gateway,
            )

            # Keyed by mascot.id AND _hero_cache_key(), not a fixed "hero.png"
            # or a mascot.id-only filename — artifacts/<topic>/_work persists
            # across separate runs of the same topic (including runs days
            # apart with a different mascot/DEFAULT_MASCOT, IMAGE_MODEL, or
            # hero_prompt text). A filename that didn't account for all of
            # that let a stale hero image get silently reused (confirmed for
            # real 2026-08-20/21: a flat-2D mascot from three days earlier
            # got reused as scene 0's I2V source in a video whose other
            # scenes correctly used the new 3D "Bearded Dwarf" mascot).
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"

            # Build base images serially so the shared hero reference is
            # created exactly once, then animate two scenes concurrently.
            # Video generation dominates wall time (the six real clips in a
            # measured run took ~34 minutes serially); bounded parallelism
            # roughly halves that without flooding the provider.
            # Not every scene earns a paid clip — see choose_animated_scenes.
            # A scene left static still renders: assemble_animated hands an
            # empty clip list to build_scene_video_segment_from_clip, which
            # falls through to the pop-in-and-hold beat on the scene's own
            # base image (the same visual language the sticker path uses).
            animate_scene = choose_animated_scenes(script["scenes"])

            base_image_paths = [
                _scene_base_image_path(
                    image_provider, mascot, hero_path, scene, i, generated_dir, cost_tracker
                )
                for i, scene in enumerate(script["scenes"])
            ]

            # Extra SHOTS for the scenes that get no clip. A static scene
            # used to hold one image for its whole length — measured 19.5s
            # of a 47.5s render with nothing changing on screen at all,
            # against a reference whose shots run 0.7-2.0s. Each extra image
            # is one more paid call (~$0.04), so they are only bought for
            # scenes that are actually static and long enough to cut.
            scene_shot_paths: list[list[Path]] = [[p] for p in base_image_paths]
            for i, scene in enumerate(script["scenes"]):
                if animate_scene[i]:
                    continue
                wanted = len(assembly.plan_shot_durations(scene_audio[i].duration))
                for shot in range(1, min(wanted, len(SHOT_FRAMING_VARIANTS))):
                    scene_shot_paths[i].append(
                        _scene_base_image_path(
                            image_provider, mascot, hero_path, scene, i, generated_dir,
                            cost_tracker, shot_index=shot,
                        )
                    )
            # Worst case, per scene: MAX_REAL_CLIPS_PER_SCENE clips (see
            # _render_scene_clips) — an early sanity check against that
            # ceiling, not the only guard: check_budget() also runs before
            # every individual real clip call inside generate_scene_video().
            batch_estimate = getattr(video_provider, "cost", 0.0) * len(script["scenes"]) * MAX_REAL_CLIPS_PER_SCENE
            cost_tracker.check_budget("video.generate_batch", batch_estimate)

            def render_clip(i: int, scene: dict[str, Any]) -> list[Path]:
                if not animate_scene[i]:
                    return []
                # fal_client's synchronous client is not shared across worker
                # threads; each concurrent request gets its own gateway.
                worker_gateway = FalGateway(settings.fal_key)
                worker_provider = get_video_provider(
                    settings.video.provider,
                    settings.credential_for(settings.video),
                    settings.video.model_or_voice,
                    settings.video_cost_per_second_usd,
                    gateway=worker_gateway,
                )
                # A dedicated MOTION prompt, not the still-image composition
                # prompt (get_scene_image_prompt) — see get_scene_motion_prompt's
                # docstring: reusing the image prompt as the motion source
                # produced a bouncing mascot and completely frozen props.
                motion_prompt = get_scene_motion_prompt(scene, mascot)
                return _render_scene_clips(
                    worker_provider, scene, base_image_paths[i], scene_audio[i].duration,
                    i, generated_dir, cost_tracker, motion_prompt,
                )

            clip_paths_per_scene: list[list[Path] | None] = [None] * len(script["scenes"])
            with ThreadPoolExecutor(max_workers=min(2, len(script["scenes"]))) as executor:
                pending = {
                    executor.submit(render_clip, i, scene): i
                    for i, scene in enumerate(script["scenes"])
                }
                for future in as_completed(pending):
                    clip_paths_per_scene[pending[future]] = future.result()

            def clip_source(i: int, _scene: dict[str, Any]) -> list[Path]:
                paths = clip_paths_per_scene[i]
                if paths is None:
                    raise RuntimeError(f"animated clip {i} did not complete")
                return paths

            generated_result = assembly.assemble_animated(
                scenes=script["scenes"],
                clip_source=clip_source,
                audio=scene_audio,
                workdir=generated_dir,
                out_mp4=final_mp4,
                caption_style=script["caption_style"],
                caution_text=script["caution_text"],
                subscribe_cta_text=SUBSCRIBE_CTA_TEXT,
                image_source=lambda i, _scene: scene_shot_paths[i],
                sfx_enabled=settings.sfx_enabled,
                music_path=music_bed_path,
                word_timings=scene_word_timings,
            )
        else:
            # Keyed by mascot.id + _hero_cache_key() — see _get_or_create_hero_image's docstring.
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"

            def image_frame_source(i: int, scene: dict[str, Any]):
                from PIL import Image
                base_image_path = _scene_base_image_path(
                    image_provider, mascot, hero_path, scene, i, generated_dir, cost_tracker
                )
                return Image.open(base_image_path).convert("RGB")

            generated_result = assembly.assemble(
                scenes=script["scenes"],
                frame_source=image_frame_source,
                audio=scene_audio,
                workdir=generated_dir,
                out_mp4=final_mp4,
                caption_style=script["caption_style"],
                caution_text=script["caution_text"],
                subscribe_cta_text=SUBSCRIBE_CTA_TEXT,
            )
        captions_meta = artifacts_dir / "captions.meta.json"
        assembly.write_captions_meta(
            script["scenes"], actual_durations, generated_result["caption_boxes"],
            captions_meta, scripted_durations=scripted_durations,
        )

        # --- Cost report ---
        cost_report_path = artifacts_dir / "cost-report.json"
        cost_report = cost_tracker.write_report(cost_report_path)
        result.cost_report = cost_report

        # --- Verification (final mp4 = the generated-image stage output).
        # Target duration is the ACTUAL total from real audio, not the script's
        # nominal total — with the stub this is identical by construction; with
        # a real TTS provider it's the only correct target to check against. ---
        stage = "verification"
        verification = verify.run_verification(
            mp4_path=final_mp4,
            scripted_total_seconds=actual_total,
            captions_meta_path=captions_meta,
            cost_report_path=cost_report_path,
            budget_cap_usd=settings.budget_cap_usd,
            artifacts_dir=artifacts_dir,
        )
        verification_path = artifacts_dir / "verification-report.json"
        verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
        result.verification = verification

        return result
    except BudgetExceeded as e:
        cost_report_path = artifacts_dir / "cost-report.json"
        cost_report = cost_tracker.write_report(cost_report_path)
        result.cost_report = cost_report
        result.budget_exceeded = True
        result.budget_exceeded_reason = str(e)
        result.error = str(e)
        return result
    except Exception as e:
        # Paid provider calls can succeed before a later validation/render
        # step fails. Persist their cost and the failing stage before
        # propagating the exception to Telegram.
        cost_report_path = artifacts_dir / "cost-report.json"
        cost_tracker.write_report(cost_report_path)
        error_report_path = artifacts_dir / "generation-error.json"
        error_report_path.write_text(
            json.dumps(
                {"stage": stage, "error_type": type(e).__name__, "error": str(e)},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def regenerate_scene(
    topic: str, scene_index: int, new_narration: str | None = None, artifacts_root: Path | None = None
) -> PipelineResult:
    """Phase 4: regenerate ONE scene's expensive steps (TTS + image gen) and
    cheaply reassemble the final video, instead of re-rendering every scene.
    Requires a prior full run_pipeline(topic) — reuses its on-disk audio/
    segment files for every scene except scene_index. artifacts_root: see
    run_pipeline's docstring — tests MUST pass a tmp_path here."""
    from PIL import Image

    from .providers.llm import MAX_SCENE_SECONDS, MIN_SCENE_SECONDS, _caption_from_claim

    result = PipelineResult()
    result.topic = topic

    artifacts_dir = (artifacts_root or REPO_ROOT / "artifacts") / topic
    workdir = artifacts_dir / "_work"
    script_path = artifacts_dir / f"{topic}.script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"no existing script for {topic!r} at {script_path} — run the full pipeline first")

    # Enforce safety check before touching any provider or spending
    try:
        safety_class = enforce_not_blocked(topic)
    except TopicBlocked as e:
        result.blocked = True
        result.block_reason = str(e)
        return result
    result.safety_class = safety_class.value

    settings = load_settings()
    try:
        require_budget_approval_if_paid(settings)
    except BudgetApprovalRequired as e:
        result.budget_approval_blocked = True
        result.budget_approval_block_reason = str(e)
        return result

    script = json.loads(script_path.read_text(encoding="utf-8"))
    scenes = script["scenes"]
    if not (0 <= scene_index < len(scenes)):
        raise IndexError(f"scene_index {scene_index} out of range (0..{len(scenes) - 1})")
    result.script = script

    mascot_id = script.get("mascot_id") or settings.default_mascot_id
    mascot = get_mascot(mascot_id)
    result.mascot_id = mascot.id
    # Reuse the SAME caption style the original full run picked (persisted in
    # script.json) — a regenerated scene must not end up with a different
    # font/color/casing than the rest of the video. None (older script.json
    # from before this field existed) falls back to draw_caption's own
    # per-caption-text default, same as it always did.
    caption_style = script.get("caption_style")
    # Same caution badge the original full run would have applied — only
    # relevant if the scene being regenerated is the LAST one (see
    # assembly.assemble()'s caution_text docstring: composited on top of
    # the real caption, never instead of it).
    caution_text = script.get("caution_text")
    is_last_scene = scene_index == len(scenes) - 1

    if new_narration:
        scenes[scene_index]["narration"] = new_narration
        scenes[scene_index]["caption"] = _caption_from_claim(new_narration)
        # Re-estimate the nominal duration too (same heuristic StubLLMProvider
        # uses) so script.json doesn't keep a stale estimate around — the
        # ACTUAL duration below is still what really drives the render either way.
        word_count = len(new_narration.split())
        scenes[scene_index]["duration"] = round(
            max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, 2.5 + 0.35 * word_count)), 1
        )
        script_path.write_text(json.dumps(script, indent=2), encoding="utf-8")

    # Cumulative cost across this video's full lifetime (original render +
    # every prior regeneration), not just this one call — the budget cap
    # must see the true total spend on this video.
    cost_report_path = artifacts_dir / "cost-report.json"
    cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
    if cost_report_path.exists():
        prior = json.loads(cost_report_path.read_text(encoding="utf-8"))
        for e in prior.get("entries", []):
            cost_tracker.record(e["provider"], e["operation"], e["estimated_cost_usd"], e["actual_cost_usd"], e["is_stub"])

    audio_dir = workdir / "audio"
    generated_dir = workdir / "generated"
    sticker_mode = settings.animation_mode == "sticker" and not settings.image.is_stub
    ai_video_mode = settings.animation_mode == "ai_video" and not settings.video.is_stub
    animate = sticker_mode or ai_video_mode
    uses_fal = any(p.provider.strip().lower() == "fal" for p in (settings.tts, settings.image, settings.video))
    fal_gateway = FalGateway(settings.fal_key) if uses_fal else None

    try:
        tts = get_tts_provider(
            settings.tts.provider,
            settings.credential_for(settings.tts),
            settings.tts.model_or_voice,
            settings.tts_voice,
            settings.tts_cost_per_1k_chars_usd,
            gateway=fal_gateway,
        )
        scene = scenes[scene_index]
        new_audio_path = assembly.build_scene_audio(tts, scene, scene_index, audio_dir, cost_tracker)
        new_duration = assembly.probe_duration(new_audio_path)

        image_provider = get_image_provider(
            settings.image.provider,
            settings.credential_for(settings.image),
            settings.image.model_or_voice,
            settings.image_cost_per_image_usd,
            gateway=fal_gateway,
            # Not mascot.visual_style here: get_scene_image_prompt()/
            # build_scene_prompt() already embed the right style into every
            # prompt per scene_type (full character description for mascot
            # scenes, a character-free style for ingredient_grid/
            # process_action). Force-appending the character-laden
            # mascot.visual_style here too bled the mascot into
            # ingredient/process shots that are supposed to show zero
            # character (confirmed for real 2026-08-21).
            visual_style="" if mascot else settings.visual_style,
            style_preset=settings.image_style,
        )

        if sticker_mode:
            # See assembly.assemble_stickers()'s docstring — no video provider.
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"
            base_image_path = _scene_base_image_path(
                image_provider, mascot, hero_path, scene, scene_index, generated_dir, cost_tracker
            )
            timed_overlays, new_box = assembly.build_timed_caption_overlays(
                scene["narration"],
                new_duration,
                caption_style=caption_style,
                caution_text=caution_text if is_last_scene else None,
            )
            new_seg_path = assembly.build_scene_video_segment_from_still(
                base_image_path,
                new_duration,
                scene_index,
                generated_dir / "segments",
                timed_caption_overlays=timed_overlays,
            )
        elif ai_video_mode:
            video_provider = get_video_provider(
                settings.video.provider,
                settings.credential_for(settings.video),
                settings.video.model_or_voice,
                settings.video_cost_per_second_usd,
                gateway=fal_gateway,
            )
            # Keyed by mascot.id + _hero_cache_key() — see _get_or_create_hero_image's docstring.
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"
            base_image_path = _scene_base_image_path(
                image_provider, mascot, hero_path, scene, scene_index, generated_dir, cost_tracker
            )

            # A dedicated MOTION prompt, not the still-image composition
            # prompt — see get_scene_motion_prompt's docstring.
            motion_prompt = get_scene_motion_prompt(scene, mascot)
            clip_paths = _render_scene_clips(
                video_provider, scene, base_image_path, new_duration,
                scene_index, generated_dir, cost_tracker, motion_prompt,
            )
            timed_overlays, new_box = assembly.build_timed_caption_overlays(
                scene["narration"],
                new_duration,
                caption_style=caption_style,
                caution_text=caution_text if is_last_scene else None,
            )
            new_seg_path = assembly.build_scene_video_segment_from_clip(
                clip_paths,
                new_duration,
                None,
                scene_index,
                generated_dir / "segments",
                timed_caption_overlays=timed_overlays,
                image_path=base_image_path,
            )
        else:
            # Keyed by mascot.id + _hero_cache_key() — see _get_or_create_hero_image's docstring.
            hero_cache_key = _hero_cache_key(mascot, settings.image.model_or_voice, settings.image_style)
            hero_path = generated_dir / f"hero_{mascot.id}_{hero_cache_key}.png"
            base_image_path = _scene_base_image_path(
                image_provider, mascot, hero_path, scene, scene_index, generated_dir, cost_tracker
            )
            base_image = Image.open(base_image_path).convert("RGB")
            new_frame_path, new_box = assembly.build_scene_frame(
                scene, scene_index, base_image, generated_dir / "frames", caption_style=caption_style
            )
            if caution_text and is_last_scene:
                badged, _caution_box = assembly.draw_caution_badge(Image.open(new_frame_path), caution_text)
                badged.save(new_frame_path)
            new_seg_path = assembly.build_scene_video_segment(new_frame_path, new_duration, scene_index, generated_dir / "segments")

        prior_captions_meta = json.loads((artifacts_dir / "captions.meta.json").read_text(encoding="utf-8"))["scenes"]

        all_audio_paths, all_durations, all_segment_paths, all_boxes = [], [], [], []
        for i in range(len(scenes)):
            if i == scene_index:
                all_audio_paths.append(new_audio_path)
                all_durations.append(new_duration)
                all_segment_paths.append(new_seg_path)
                all_boxes.append(new_box)
            else:
                existing_audio = audio_dir / f"scene_{i:02d}.wav"
                existing_seg = generated_dir / "segments" / f"seg_{i:02d}.mp4"
                if not existing_audio.exists() or not existing_seg.exists():
                    raise FileNotFoundError(f"missing prior render artifacts for scene {i} — run the full pipeline first")
                all_audio_paths.append(existing_audio)
                all_durations.append(assembly.probe_duration(existing_audio))
                all_segment_paths.append(existing_seg)
                all_boxes.append(prior_captions_meta[i])  # reuse the previously-computed box, don't re-derive it

        final_mp4 = artifacts_dir / f"{topic}.mp4"
        assembly.concat_and_mux(all_segment_paths, all_audio_paths, generated_dir, final_mp4)

        captions_meta = artifacts_dir / "captions.meta.json"
        assembly.write_captions_meta(scenes, all_durations, all_boxes, captions_meta)
        captions_srt = artifacts_dir / "captions.srt"
        assembly.write_captions_srt(scenes, all_durations, captions_srt)

        cost_report = cost_tracker.write_report(cost_report_path)
        result.cost_report = cost_report

        verification = verify.run_verification(
            mp4_path=final_mp4,
            scripted_total_seconds=sum(all_durations),
            captions_meta_path=captions_meta,
            cost_report_path=cost_report_path,
            budget_cap_usd=settings.budget_cap_usd,
            artifacts_dir=artifacts_dir,
        )
        (artifacts_dir / "verification-report.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
        result.verification = verification
        result.artifacts_dir = artifacts_dir
        # The final .mp4 just changed — any prior approval/schedule was for
        # the OLD content, not this one. Reset here (not just in the
        # dashboard route) so it can't be bypassed by calling regenerate_scene
        # directly (same defensive pattern as review_state.schedule()'s own
        # approved-only guard) — confirmed real 2026-08-21 review: a
        # regenerated scene silently kept its prior "approved"/"scheduled"
        # status, so a fix could ship without a fresh human look.
        review_state.reset_to_pending(artifacts_dir, notes=f"scene {scene_index} regenerated — needs re-review")
        return result
    except BudgetExceeded as e:
        cost_report = cost_tracker.write_report(cost_report_path)
        result.cost_report = cost_report
        result.budget_exceeded = True
        result.budget_exceeded_reason = str(e)
        result.error = str(e)
        return result


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m shorts_factory.pipeline <topic>", file=sys.stderr)
        return 2

    topic = argv[1]
    result = run_pipeline(topic)

    if result.budget_approval_blocked:
        print(f"BUDGET APPROVAL REQUIRED: {result.budget_approval_block_reason}")
        return 1

    if result.blocked:
        print(f"BLOCKED: {result.block_reason}")
        return 1

    if result.budget_exceeded:
        # result.verification stays None when this happens (the run
        # returned early, before verification could run) — the old code
        # fell straight through the `if result.verification:` block below
        # to the final `return 0`, reporting SUCCESS for a run that had
        # actually failed partway through on the budget cap (confirmed
        # real 2026-08-21 review).
        print(f"BUDGET EXCEEDED: {result.budget_exceeded_reason}", file=sys.stderr)
        if result.cost_report:
            print(f"cost: ${result.cost_report['total_spent_usd']:.4f} / ${result.cost_report['budget_cap_usd']:.2f} cap")
        return 1

    print(f"topic={result.topic} safety_class={result.safety_class}")
    print(f"artifacts: {result.artifacts_dir}")
    if result.cost_report:
        print(f"cost: ${result.cost_report['total_spent_usd']:.4f} / ${result.cost_report['budget_cap_usd']:.2f} cap")
    if result.verification:
        print(f"verification overall_pass={result.verification['overall_pass']}")
        for c in result.verification["checks"]:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"  [{mark}] {c['criterion']}")
        return 0 if result.verification["overall_pass"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
