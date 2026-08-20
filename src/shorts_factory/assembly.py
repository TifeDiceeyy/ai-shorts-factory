"""Deterministic FFmpeg assembly: per-scene frames + per-scene TTS audio ->
one soap.mp4 with exactly one video and one audio stream.

Determinism strategy: no wall-clock metadata (`-fflags +bitexact`, explicit
`creation_time`), single-threaded x264 (`-threads 1`) so encoder output
doesn't depend on scheduling, and every visual/audio input is generated from
the scene data alone (no randomness). This is verified, not assumed — see
tests/test_determinism.py, which hashes two independent runs.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .captions import FRAME_HEIGHT, FRAME_WIDTH, CaptionBox, caption_overlay_png, draw_caption
from .cost_tracker import CostTracker
from .providers.tts import TTSProvider

FPS = 30
LOUDNORM_TARGET_I = -14.0
LOUDNORM_TARGET_TP = -1.5
LOUDNORM_TARGET_LRA = 11.0

# Flat placeholder palette (step 5) — deliberately distinct from
# providers/image.py's gradient palette so the two pipeline stages are
# visually and programmatically distinguishable.
SOLID_PALETTE = [
    (61, 64, 91), (91, 71, 61), (61, 91, 78), (91, 61, 84), (74, 61, 91), (91, 87, 61),
]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstderr:\n{result.stderr}")
    return result


def probe_duration(path: Path) -> float:
    """Actual duration of a media file, via ffprobe / ffmpeg — never trust a
    provider's requested/nominal duration, measure what it actually produced."""
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return float(result.stdout.strip())
            except ValueError:
                pass

    # Fallback to ffmpeg -i parsing if ffprobe is unavailable
    cmd = ["ffmpeg", "-i", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if m:
        hours, mins, secs = m.groups()
        return int(hours) * 3600 + int(mins) * 60 + float(secs)
    raise RuntimeError(f"Could not probe duration for {path}:\n{result.stderr}")


def solid_color_frame(index: int) -> Image.Image:
    color = SOLID_PALETTE[index % len(SOLID_PALETTE)]
    return Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), color)


@dataclass
class SceneAudio:
    """The ACTUAL measured duration of a scene's synthesized audio — never
    the script's nominal `duration` field, which is only an estimate until
    real narration exists. This is what drives video-segment length,
    caption timing, and the final assembled duration."""
    path: Path
    duration: float
    scripted_duration: float

    @property
    def drift_seconds(self) -> float:
        return self.duration - self.scripted_duration


@dataclass
class SceneArtifacts:
    index: int
    frame_path: Path
    audio_path: Path
    duration: float
    caption_box: CaptionBox
    segment_video_path: Path


def build_scene_frame(
    scene: dict[str, Any],
    index: int,
    base_image: Image.Image,
    frames_dir: Path,
) -> tuple[Path, CaptionBox]:
    composited, box = draw_caption(base_image, scene["caption"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"scene_{index:02d}.png"
    composited.save(frame_path)
    return frame_path, box


def build_scene_audio(
    tts_provider: TTSProvider,
    scene: dict[str, Any],
    index: int,
    audio_dir: Path,
    cost_tracker: CostTracker,
) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"scene_{index:02d}.wav"
    return tts_provider.synthesize_scene(scene, index, out_path, cost_tracker)


def build_scene_video_segment(frame_path: Path, duration: float, index: int, segments_dir: Path) -> Path:
    segments_dir.mkdir(parents=True, exist_ok=True)
    out_path = segments_dir / f"seg_{index:02d}.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(frame_path),
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-c:v", "libx264", "-preset", "medium", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        "-an",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def build_scene_video_segment_from_clip(
    clip_path: Path,
    duration: float,
    caption_overlay: Image.Image,
    index: int,
    segments_dir: Path,
) -> Path:
    """Same role as build_scene_video_segment(), for an animated clip
    instead of a static frame: scales to frame size, composites the caption
    overlay (see captions.caption_overlay_png — a transparent RGBA layer,
    since there's no single base image to burn it into ahead of time), and
    holds the last frame (tpad, generously over-padded then hard-trimmed by
    -t) to reach the scene's actual scripted duration — image-to-video
    models return a fixed clip length (e.g. Hailuo's 6s minimum) that will
    essentially never match the scripted duration exactly."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = segments_dir / f"caption_overlay_{index:02d}.png"
    caption_overlay.save(overlay_path)
    out_path = segments_dir / f"seg_{index:02d}.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(clip_path),
        "-loop", "1", "-i", str(overlay_path),
        "-filter_complex",
        f"[0:v]scale={FRAME_WIDTH}:{FRAME_HEIGHT},tpad=stop_mode=clone:stop_duration=15[base];"
        "[base][1:v]overlay=0:0[out]",
        "-map", "[out]",
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def concat_video_segments(segment_paths: list[Path], workdir: Path, out_path: Path) -> Path:
    list_file = workdir / "segments.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "libx264", "-preset", "medium", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        "-an",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def concat_audio(audio_paths: list[Path], workdir: Path, out_path: Path) -> Path:
    list_file = workdir / "audio_segments.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in audio_paths:
            f.write(f"file '{p.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(out_path),
    ]
    _run(cmd)
    return out_path


def loudnorm_measure(in_path: Path) -> dict:
    """First pass: measure only, returns the JSON stats block ffmpeg prints."""
    filt = f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:print_format=json"
    cmd = ["ffmpeg", "-y", "-loglevel", "info", "-i", str(in_path), "-af", filt, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"could not parse loudnorm stats from ffmpeg output:\n{stderr}")
    return json.loads(stderr[start:end + 1])


def loudnorm_apply(in_path: Path, out_path: Path, measured: dict) -> None:
    """Two-pass apply: measured is loudnorm_measure()'s pass-1 JSON stats fed
    back in as measured_I/TP/LRA/thresh + offset with linear=true. ffmpeg's
    own docs are explicit that single-pass loudnorm (the filter applied
    without these) uses a dynamic compressor and is meaningfully less
    accurate — confirmed in practice: a real run measured -16.24 LUFS
    against a -14 +/-1 target using single-pass before this fix."""
    filt = (
        f"loudnorm=I={LOUDNORM_TARGET_I}:TP={LOUDNORM_TARGET_TP}:LRA={LOUDNORM_TARGET_LRA}:"
        f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(in_path),
        "-af", filt,
        "-ar", "48000",
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(out_path),
    ]
    _run(cmd)


def mux_final(video_path: Path, audio_path: Path, out_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest",
        "-fflags", "+bitexact",
        "-map_metadata", "-1",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        str(out_path),
    ]
    _run(cmd)


def write_captions_srt(scenes: list[dict[str, Any]], durations: list[float], out_path: Path) -> None:
    """durations must be the ACTUAL per-scene audio durations (SceneAudio.duration),
    not the script's nominal `duration` field — captions must track what's
    really on the timeline, not what the LLM guessed it would be."""
    def fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    cursor = 0.0
    for i, (scene, duration) in enumerate(zip(scenes, durations), start=1):
        start = cursor
        end = cursor + duration
        lines.append(str(i))
        lines.append(f"{fmt(start)} --> {fmt(end)}")
        lines.append(scene["caption"])
        lines.append("")
        cursor = end
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _box_fields(box: CaptionBox | dict) -> tuple[dict, bool]:
    """Accepts either a fresh CaptionBox (from draw_caption) or a plain dict
    reloaded from a previous captions.meta.json (see regenerate_scene, which
    reuses unchanged scenes' boxes instead of re-deriving them)."""
    if isinstance(box, CaptionBox):
        return (
            {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom},
            box.inside_safe_area(),
        )
    return box["box"] if "box" in box else box, box.get("inside_safe_area", True)


def write_captions_meta(
    scenes: list[dict[str, Any]],
    durations: list[float],
    caption_boxes: list[CaptionBox | dict],
    out_path: Path,
    scripted_durations: list[float] | None = None,
) -> None:
    """durations must be ACTUAL per-scene audio durations. scripted_durations
    (the script's nominal estimate), if given, is recorded alongside so any
    drift between estimate and reality is visible in the artifact, not hidden.
    caption_boxes entries may be CaptionBox objects or plain dicts (see
    _box_fields) — the latter lets regenerate_scene reuse unchanged scenes'
    previously-computed boxes without re-running draw_caption on them."""
    cursor = 0.0
    entries = []
    scripted_durations = scripted_durations or [None] * len(scenes)
    for scene, duration, scripted, box in zip(scenes, durations, scripted_durations, caption_boxes):
        start = cursor
        end = cursor + duration
        box_dict, inside_safe_area = _box_fields(box)
        entries.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "actual_duration": round(duration, 3),
                "scripted_duration": round(scripted, 3) if scripted is not None else None,
                "drift_seconds": round(duration - scripted, 3) if scripted is not None else None,
                "caption": scene["caption"],
                "source_claim_id": scene["source_claim_id"],
                "box": box_dict,
                "inside_safe_area": inside_safe_area,
            }
        )
        cursor = end
    out_path.write_text(json.dumps({"frame_width": FRAME_WIDTH, "frame_height": FRAME_HEIGHT, "scenes": entries}, indent=2), encoding="utf-8")


def synthesize_scenes(
    tts_provider: TTSProvider,
    scenes: list[dict[str, Any]],
    audio_dir: Path,
    cost_tracker: CostTracker,
) -> list[SceneAudio]:
    """Synthesize narration audio ONCE per scene, then MEASURE its actual
    duration — never trust the script's nominal `duration` field once real
    audio exists. Narration doesn't depend on which visual stage (placeholder
    vs. generated-image) is being assembled, so the caller synthesizes once
    and reuses the same SceneAudio list for every stage — a real TTS provider
    must not be charged twice for identical narration just because the
    pipeline renders the video twice."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for i, scene in enumerate(scenes):
        path = build_scene_audio(tts_provider, scene, i, audio_dir, cost_tracker)
        actual_duration = probe_duration(path)
        result.append(SceneAudio(path=path, duration=actual_duration, scripted_duration=scene["duration"]))
    return result


def assemble(
    scenes: list[dict[str, Any]],
    frame_source: Callable[[int, dict[str, Any]], Image.Image],
    audio: list[SceneAudio],
    workdir: Path,
    out_mp4: Path,
) -> dict[str, Any]:
    """Runs the full assembly for one stage (placeholder or generated-image).
    frame_source(index, scene) -> a base PIL Image (pre-caption) for that scene.
    audio must already exist — see synthesize_scenes(), called once and shared
    across every stage so narration is never re-synthesized (and never
    re-charged) per stage. Video segment length is driven by audio[i].duration
    (the MEASURED actual audio length), never scene["duration"] (the script's
    nominal estimate) — this is what keeps video/audio in sync once a real,
    variable-length TTS provider replaces the stub.
    Returns a dict of intermediate paths + caption boxes, useful for verification.
    """
    frames_dir = workdir / "frames"
    segments_dir = workdir / "segments"
    workdir.mkdir(parents=True, exist_ok=True)

    if len(audio) != len(scenes):
        raise ValueError(f"expected {len(scenes)} audio entries, got {len(audio)}")

    frame_paths: list[Path] = []
    caption_boxes: list[CaptionBox] = []
    segment_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        base_image = frame_source(i, scene)
        frame_path, box = build_scene_frame(scene, i, base_image, frames_dir)
        frame_paths.append(frame_path)
        caption_boxes.append(box)

        seg_path = build_scene_video_segment(frame_path, audio[i].duration, i, segments_dir)
        segment_paths.append(seg_path)

    tail = concat_and_mux(segment_paths, [a.path for a in audio], workdir, out_mp4)
    return {
        "frame_paths": frame_paths,
        "caption_boxes": caption_boxes,
        **tail,
    }


def assemble_animated(
    scenes: list[dict[str, Any]],
    clip_source: Callable[[int, dict[str, Any]], Path],
    audio: list[SceneAudio],
    workdir: Path,
    out_mp4: Path,
) -> dict[str, Any]:
    """Animated-scene counterpart to assemble(): clip_source(index, scene) ->
    a raw, uncaptioned, arbitrary-duration animated clip path for that scene
    (see providers/video.py). Captioning and duration-fitting happen here —
    same division of responsibility as assemble()'s frame_source, the caller
    only supplies the per-scene visual content."""
    segments_dir = workdir / "segments"
    workdir.mkdir(parents=True, exist_ok=True)

    if len(audio) != len(scenes):
        raise ValueError(f"expected {len(scenes)} audio entries, got {len(audio)}")

    caption_boxes: list[CaptionBox] = []
    segment_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        clip_path = clip_source(i, scene)
        overlay, box = caption_overlay_png(scene["caption"])
        caption_boxes.append(box)

        seg_path = build_scene_video_segment_from_clip(clip_path, audio[i].duration, overlay, i, segments_dir)
        segment_paths.append(seg_path)

    tail = concat_and_mux(segment_paths, [a.path for a in audio], workdir, out_mp4)
    return {
        "caption_boxes": caption_boxes,
        **tail,
    }


def concat_and_mux(
    segment_paths: list[Path],
    audio_paths: list[Path],
    workdir: Path,
    out_mp4: Path,
) -> dict[str, Any]:
    """The reassembly tail shared by both a full assemble() run and a
    single-scene regeneration (Phase 4): concat all video segments, concat
    all audio, loudnorm, mux. Cheap — this is what "without a full re-render"
    means: the expensive per-scene steps (image gen, TTS) aren't repeated for
    scenes that didn't change, only this reassembly is."""
    video_only = workdir / "video_only.mp4"
    concat_video_segments(segment_paths, workdir, video_only)

    narration_raw = workdir / "narration_raw.wav"
    concat_audio(audio_paths, workdir, narration_raw)

    loudnorm_stats_pass1 = loudnorm_measure(narration_raw)

    narration_normalized = workdir / "narration_normalized.wav"
    loudnorm_apply(narration_raw, narration_normalized, measured=loudnorm_stats_pass1)

    mux_final(video_only, narration_normalized, out_mp4)

    return {
        "audio_paths": audio_paths,
        "video_only": video_only,
        "narration_raw": narration_raw,
        "narration_normalized": narration_normalized,
        "loudnorm_measure_pass1": loudnorm_stats_pass1,
        "out_mp4": out_mp4,
    }
