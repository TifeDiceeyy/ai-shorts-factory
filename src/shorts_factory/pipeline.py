"""Phase 0 walking-skeleton orchestrator.

One command, one hardcoded-shape run: topic -> brief -> script -> placeholder
render (zero image spend, proves assembly) -> generated-image render (stub
provider stands in until a real one is approved) -> soap.mp4 -> verification
report. See CLAUDE.md for the full spec this implements.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import assembly, verify
from .config import BudgetApprovalRequired, load_settings, require_budget_approval_if_paid
from .cost_tracker import CostTracker
from .mascots import get_mascot
from .providers.image import get_image_provider
from .providers.llm import get_llm_provider
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
        self.artifacts_dir: Path | None = None
        self.verification: dict[str, Any] | None = None
        self.cost_report: dict[str, Any] | None = None
        self.script: dict[str, Any] | None = None
        self.error: str | None = None


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

    selected_mascot_id = mascot_id or (idea.get("mascot_id") if idea else None) or settings.default_mascot_id
    mascot = get_mascot(selected_mascot_id)
    result.mascot_id = mascot.id

    artifacts_dir = (artifacts_root or REPO_ROOT / "artifacts") / topic
    workdir = artifacts_dir / "_work"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    result.artifacts_dir = artifacts_dir

    animate = not settings.video.is_stub
    if animate and settings.image.is_stub:
        raise ValueError(
            "VIDEO_PROVIDER is real but IMAGE_PROVIDER is stub — animation needs a real "
            "hero image to animate; set IMAGE_PROVIDER=fal too"
        )

    cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
    uses_fal = any(
        p.provider.strip().lower() == "fal" for p in (settings.llm, settings.tts, settings.image, settings.video)
    )
    fal_gateway = FalGateway(settings.fal_key) if uses_fal else None

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

    # --- Script generation (LLM provider, stub by default) ---
    llm = get_llm_provider(
        settings.llm.provider,
        settings.credential_for(settings.llm),
        settings.llm.model_or_voice,
        settings.llm_cost_per_script_usd,
        gateway=fal_gateway,
        endpoint=settings.fal_llm_endpoint,
    )
    effective_visual_style = (
        f"{mascot.name} Template. Style DNA: {mascot.visual_style}. "
        f"Scene Adaptive Direction: {mascot.scene_role_template}"
    )
    script = llm.generate_script(brief, settings.output_language, effective_visual_style, cost_tracker)
    warning = caution_caption(topic)
    if warning and script.get("scenes"):
        script["scenes"][-1]["caption"] = warning
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
    placeholder_result = assembly.assemble(
        scenes=script["scenes"],
        frame_source=lambda i, scene: assembly.solid_color_frame(i),
        audio=scene_audio,
        workdir=placeholder_dir,
        out_mp4=placeholder_mp4,
    )
    assembly.write_captions_meta(
        script["scenes"], actual_durations, placeholder_result["caption_boxes"],
        artifacts_dir / "captions.placeholder.meta.json", scripted_durations=scripted_durations,
    )

    # --- Stage 2: swap in generated images (stub image provider stands in
    # until a real one is approved) — this produces the final soap.mp4.
    # If VIDEO_PROVIDER is real, every scene is animated instead (see
    # providers/video.py): ONE hero character image is generated once and
    # reused as the source for every scene's image-to-video call — that
    # shared source is what keeps the character consistent across scenes,
    # which independent per-scene image generation could never guarantee. ---
    image_provider = get_image_provider(
        settings.image.provider,
        settings.credential_for(settings.image),
        settings.image.model_or_voice,
        settings.image_cost_per_image_usd,
        gateway=fal_gateway,
        visual_style=mascot.visual_style if mascot else settings.visual_style,
        style_preset=settings.image_style,
    )
    generated_dir = workdir / "generated"
    final_mp4 = artifacts_dir / f"{topic}.mp4"

    if animate:
        video_provider = get_video_provider(
            settings.video.provider,
            settings.credential_for(settings.video),
            settings.video.model_or_voice,
            settings.video_cost_per_second_usd,
            gateway=fal_gateway,
        )
        hero_path = generated_dir / "hero.png"
        hero_scene = {
            "visual_prompt": mascot.hero_prompt
        }
        image_provider.generate_scene_image(hero_scene, "hero", hero_path, cost_tracker)

        def clip_source(i: int, scene: dict[str, Any]) -> Path:
            tmp_path = generated_dir / "raw" / f"clip_{i:02d}.mp4"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            return video_provider.generate_scene_video(scene, hero_path, i, tmp_path, cost_tracker)

        generated_result = assembly.assemble_animated(
            scenes=script["scenes"],
            clip_source=clip_source,
            audio=scene_audio,
            workdir=generated_dir,
            out_mp4=final_mp4,
        )
    else:

        def image_frame_source(i: int, scene: dict[str, Any]):
            tmp_path = generated_dir / "raw" / f"raw_{i:02d}.png"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            image_provider.generate_scene_image(scene, i, tmp_path, cost_tracker)
            from PIL import Image
            return Image.open(tmp_path).convert("RGB")

        generated_result = assembly.assemble(
            scenes=script["scenes"],
            frame_source=image_frame_source,
            audio=scene_audio,
            workdir=generated_dir,
            out_mp4=final_mp4,
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

    settings = load_settings()
    try:
        require_budget_approval_if_paid(settings)
    except BudgetApprovalRequired as e:
        result.budget_approval_blocked = True
        result.budget_approval_block_reason = str(e)
        return result

    artifacts_dir = (artifacts_root or REPO_ROOT / "artifacts") / topic
    workdir = artifacts_dir / "_work"
    script_path = artifacts_dir / f"{topic}.script.json"
    if not script_path.exists():
        raise FileNotFoundError(f"no existing script for {topic!r} at {script_path} — run the full pipeline first")

    script = json.loads(script_path.read_text(encoding="utf-8"))
    scenes = script["scenes"]
    if not (0 <= scene_index < len(scenes)):
        raise IndexError(f"scene_index {scene_index} out of range (0..{len(scenes) - 1})")
    result.script = script

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
    uses_fal = any(p.provider.strip().lower() == "fal" for p in (settings.tts, settings.image))
    fal_gateway = FalGateway(settings.fal_key) if uses_fal else None

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
        visual_style=settings.visual_style,
        style_preset=settings.image_style,
    )
    raw_path = generated_dir / "raw" / f"raw_{scene_index:02d}.png"
    image_provider.generate_scene_image(scene, scene_index, raw_path, cost_tracker)
    base_image = Image.open(raw_path).convert("RGB")
    new_frame_path, new_box = assembly.build_scene_frame(scene, scene_index, base_image, generated_dir / "frames")
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
    result.safety_class = enforce_not_blocked(topic).value
    result.artifacts_dir = artifacts_dir
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
