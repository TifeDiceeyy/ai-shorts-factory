"""Phase 0 walking-skeleton orchestrator.

One command, one hardcoded-shape run: topic -> brief -> script -> placeholder
render (zero image spend, proves assembly) -> generated-image render (stub
provider stands in until a real one is approved) -> soap.mp4 -> verification
report. See CLAUDE.md for the full spec this implements.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import assembly, verify
from .captions import get_random_caption_style_name
from .config import BudgetApprovalRequired, load_settings, require_budget_approval_if_paid
from .cost_tracker import BudgetExceeded, CostTracker
from .dashboard import review_state
from .mascots import Mascot, generate_custom_mascot, get_mascot, select_mascot_for_story
from .providers.image import get_image_provider
from .providers.llm import LLMResponseFormatError, StubLLMProvider, get_llm_provider
from .providers.tts import get_tts_provider
from .providers.video import get_video_provider
from .providers.fal import FalGateway
from .safety import TopicBlocked, caution_caption, caution_line, enforce_not_blocked
from .schema_validate import ValidationError, validate_brief, validate_script_against_brief

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        return mascot.build_scene_prompt(
            scene_role=scene.get("mascot_role", ""),
            action=scene.get("action", ""),
            emotion=scene.get("mascot_emotion", ""),
            props=scene.get("props"),
            layout=scene.get("layout", "auto"),
            scene_type=scene.get("scene_type", "mascot"),
            fx=scene.get("fx"),
        )
    return scene.get("visual_prompt", mascot.hero_prompt)


def _hero_cache_key(mascot: Mascot, image_model: str, image_style: str) -> str:
    """Short hash covering everything that changes what the hero image
    looks like: the mascot's own hero_prompt text, the image model, and the
    style preset. mascot.id alone isn't enough — artifacts/<topic>/_work
    persists across separate runs of the same topic, so switching
    IMAGE_MODEL (e.g. Recraft -> Nano Banana) or editing a mascot's
    hero_prompt while keeping the same mascot_id would otherwise silently
    keep reusing a hero image rendered under the old model/prompt."""
    raw = f"{mascot.hero_prompt}|{image_model}|{image_style}"
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
        image_provider.generate_scene_image({"visual_prompt": mascot.hero_prompt}, "hero", hero_path, cost_tracker)
    return hero_path


def _scene_base_image_path(
    image_provider,
    mascot: Mascot,
    hero_path: Path,
    scene: dict[str, Any],
    scene_index: int,
    generated_dir: Path,
    cost_tracker: CostTracker,
) -> Path:
    """Generates this scene's base image, one call per scene, every scene.

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
    scene_prompt = get_scene_image_prompt(scene, mascot)
    out_path = generated_dir / "raw" / f"raw_{scene_index:02d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reference_image_path = None
    if stype not in ("ingredient_grid", "process_action"):
        reference_image_path = _get_or_create_hero_image(image_provider, mascot, hero_path, cost_tracker)
    image_provider.generate_scene_image(
        {"visual_prompt": scene_prompt}, scene_index, out_path, cost_tracker,
        reference_image_path=reference_image_path,
    )
    return out_path


def run_pipeline(
    topic: str,
    idea: dict[str, Any] | None = None,
    artifacts_root: Path | None = None,
    mascot_id: str | None = None,
) -> PipelineResult:
    """idea, if given, is the concept/angle/hook the human picked during
    /plan's ideation step (as a dict, see ideation.ideas_to_dicts) — it
    steers script framing (brief_builder.build_brief_from_citations),
    never facts. None preserves the old topic-only behavior.

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
        # Real runs must consume a verified citation store. Hand-authored briefs
        # remain available only to the zero-cost Phase 0 renderer test.
        citation_path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
        if settings.any_provider_is_real:
            if not citation_path.exists():
                raise FileNotFoundError(
                    f"real generation requires verified citations at {citation_path}; run retrieve.sh first"
                )
            from .brief_builder import build_brief_from_citations

            citation_store = json.loads(citation_path.read_text(encoding="utf-8"))
            brief = build_brief_from_citations(
                topic,
                citation_store,
                safety_class.value,
                caution=caution_line(topic),
                idea=idea,
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
        # after the safety gate) so story-matching (raw_mascot_id in
        # ("auto", "random", "story") or simply not given) can use the
        # brief's concept/angle/claims text, not just the bare topic string.
        raw_mascot_id = mascot_id or (idea.get("mascot_id") if idea else None)
        if raw_mascot_id and str(raw_mascot_id).strip().lower() not in ("auto", "random", "story", ""):
            mascot = get_mascot(raw_mascot_id)
        else:
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
        # One caption style per video, chosen once and persisted — not
        # re-randomized per scene/per render stage. regenerate_scene reads
        # this same value back so a single-scene regeneration never mismatches
        # the rest of the video's captions.
        script["caption_style"] = get_random_caption_style_name()
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
        tts = get_tts_provider(
            settings.tts.provider,
            settings.credential_for(settings.tts),
            settings.tts.model_or_voice,
            settings.tts_voice,
            settings.tts_cost_per_1k_chars_usd,
            gateway=fal_gateway,
        )
        scene_audio = assembly.synthesize_scenes(tts, script["scenes"], workdir / "audio", cost_tracker)
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

            def sticker_image_source(i: int, scene: dict[str, Any]) -> Path:
                return _scene_base_image_path(
                    image_provider, mascot, hero_path, scene, i, generated_dir, cost_tracker
                )

            generated_result = assembly.assemble_stickers(
                scenes=script["scenes"],
                image_source=sticker_image_source,
                audio=scene_audio,
                workdir=generated_dir,
                out_mp4=final_mp4,
                caption_style=script["caption_style"],
                caution_text=script["caution_text"],
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
            base_image_paths = [
                _scene_base_image_path(
                    image_provider, mascot, hero_path, scene, i, generated_dir, cost_tracker
                )
                for i, scene in enumerate(script["scenes"])
            ]
            batch_estimate = getattr(video_provider, "cost", 0.0) * len(script["scenes"])
            cost_tracker.check_budget("video.generate_batch", batch_estimate)

            def render_clip(i: int, scene: dict[str, Any]) -> Path:
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
                tmp_clip_path = generated_dir / "raw" / f"clip_{i:02d}.mp4"
                # Same prompt the base image was actually built from — not
                # scene["visual_prompt"] (see providers/video.py's docstring
                # on why animating with that raw, separate field risks
                # describing a different shot than what's in the frame).
                motion_prompt = get_scene_image_prompt(scene, mascot)
                return worker_provider.generate_scene_video(
                    scene, base_image_paths[i], i, tmp_clip_path, cost_tracker, motion_prompt=motion_prompt
                )

            clip_paths: list[Path | None] = [None] * len(script["scenes"])
            with ThreadPoolExecutor(max_workers=min(2, len(script["scenes"]))) as executor:
                pending = {
                    executor.submit(render_clip, i, scene): i
                    for i, scene in enumerate(script["scenes"])
                }
                for future in as_completed(pending):
                    clip_paths[pending[future]] = future.result()

            def clip_source(i: int, _scene: dict[str, Any]) -> Path:
                path = clip_paths[i]
                if path is None:
                    raise RuntimeError(f"animated clip {i} did not complete")
                return path

            generated_result = assembly.assemble_animated(
                scenes=script["scenes"],
                clip_source=clip_source,
                audio=scene_audio,
                workdir=generated_dir,
                out_mp4=final_mp4,
                caption_style=script["caption_style"],
                caution_text=script["caution_text"],
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

            tmp_clip_path = generated_dir / "raw" / f"clip_{scene_index:02d}.mp4"
            motion_prompt = get_scene_image_prompt(scene, mascot)
            clip_path = video_provider.generate_scene_video(
                scene, base_image_path, scene_index, tmp_clip_path, cost_tracker, motion_prompt=motion_prompt
            )
            timed_overlays, new_box = assembly.build_timed_caption_overlays(
                scene["narration"],
                new_duration,
                caption_style=caption_style,
                caution_text=caution_text if is_last_scene else None,
            )
            new_seg_path = assembly.build_scene_video_segment_from_clip(
                clip_path,
                new_duration,
                None,
                scene_index,
                generated_dir / "segments",
                timed_caption_overlays=timed_overlays,
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
                badged = assembly.draw_caution_badge(Image.open(new_frame_path), caution_text)
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
