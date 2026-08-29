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
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .captions import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SUBSCRIBE_CTA_STACK_GAP,
    CaptionBox,
    caption_overlay_png,
    caution_badge_overlay_png,
    draw_caption,
    draw_caution_badge,
    draw_subscribe_cta,
    subscribe_cta_overlay_png,
)
from .cost_tracker import CostTracker
from .media_probe import probe_duration
from .providers.tts import TTSProvider

FPS = 30
LOUDNORM_TARGET_I = -14.0
LOUDNORM_TARGET_TP = -1.5
LOUDNORM_TARGET_LRA = 11.0
# verify.py's actual gate is +/-1.0 LU on the FINAL .mp4. Correct proactively
# at a tighter internal margin so a single post-mux correction pass reliably
# lands inside that gate rather than skating its edge.
LOUDNORM_CORRECTION_MARGIN_LU = 0.5

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


@dataclass(frozen=True)
class CaptionCue:
    text: str
    start: float
    end: float


@dataclass
class TimedCaptionOverlay:
    image: Image.Image
    start: float
    end: float
    box: CaptionBox


def narration_caption_cues(
    narration: str,
    duration: float,
    max_words: int = 3,
    max_chars: int = 22,
) -> list[CaptionCue]:
    """Split the exact narration into short, contiguous timed captions.

    Timing is proportional to spoken-character weight within the measured
    TTS duration. This keeps every displayed word identical to the voiceover
    without requiring a second speech-to-text provider call.
    """
    words = narration.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and (len(current) >= max_words or len(candidate) > max_chars):
            chunks.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current))

    weights = [max(1, len(re.sub(r"[^A-Za-z0-9]", "", chunk))) for chunk in chunks]
    total_weight = sum(weights)
    cursor = 0.0
    cues: list[CaptionCue] = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        end = duration if index == len(chunks) - 1 else cursor + duration * weight / total_weight
        cues.append(CaptionCue(text=chunk, start=cursor, end=end))
        cursor = end
    return cues


def build_timed_caption_overlays(
    narration: str,
    duration: float,
    caption_style: str | None = None,
    caution_text: str | None = None,
    subscribe_cta_text: str | None = None,
) -> tuple[list[TimedCaptionOverlay], CaptionBox]:
    """subscribe_cta_text, unlike caution_text, is composited onto only the
    LAST cue's overlay (not every cue) — it's an end-of-video call to
    action, meant to appear once in the closing seconds, not repeated
    across a whole scene."""
    overlays: list[TimedCaptionOverlay] = []
    caution_box: CaptionBox | None = None
    for cue in narration_caption_cues(narration, duration):
        image, box = caption_overlay_png(cue.text, style=caption_style)
        if caution_text:
            caution_overlay, caution_box = caution_badge_overlay_png(caution_text)
            image = Image.alpha_composite(image, caution_overlay)
        overlays.append(TimedCaptionOverlay(image=image, start=cue.start, end=cue.end, box=box))
    if not overlays:
        image, box = caption_overlay_png("…", style=caption_style)
        overlays.append(TimedCaptionOverlay(image=image, start=0.0, end=duration, box=box))
    if subscribe_cta_text:
        last = overlays[-1]
        # Stack above the caution badge instead of both anchoring to the
        # same bottom spot — real bug found 2026-08-29 on a real
        # yellow-safety-class video: they overlapped.
        bottom_limit = caution_box.top - SUBSCRIBE_CTA_STACK_GAP if caution_box else None
        cta_overlay = subscribe_cta_overlay_png(subscribe_cta_text, bottom_limit=bottom_limit)
        composited = Image.alpha_composite(last.image, cta_overlay)
        overlays[-1] = TimedCaptionOverlay(image=composited, start=last.start, end=last.end, box=last.box)
    union_box = CaptionBox(
        left=min(item.box.left for item in overlays),
        top=min(item.box.top for item in overlays),
        right=max(item.box.right for item in overlays),
        bottom=max(item.box.bottom for item in overlays),
    )
    return overlays, union_box


def build_scene_frame(
    scene: dict[str, Any],
    index: int,
    base_image: Image.Image,
    frames_dir: Path,
    caption_style: str | None = None,
) -> tuple[Path, CaptionBox]:
    composited, box = draw_caption(base_image, scene["caption"], style=caption_style)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frames_dir / f"scene_{index:02d}.png"
    composited.save(frame_path)
    return frame_path, box


# User feedback 2026-08-28: narration reads a bit slow across the whole
# video. Applied as a pitch-preserving tempo change (ffmpeg `atempo`), not a
# raw resample (which would also raise pitch — "chipmunk" effect) and not an
# ElevenLabs API `speed` param (fal.ai's eleven-v3 endpoint schema doesn't
# reliably expose one, unlike some other TTS models' endpoints there).
# `atempo` accepts 0.5-2.0 in a single filter node with no chaining needed
# for this modest a change. Applied globally (every scene, every provider)
# since the request was for the whole video, not one clip.
NARRATION_SPEED_FACTOR = 1.10


def _apply_narration_speed(path: Path, factor: float = NARRATION_SPEED_FACTOR) -> None:
    if factor == 1.0:
        return
    tmp_path = path.with_suffix(".sped.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-af", f"atempo={factor:.4f}",
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(tmp_path),
    ]
    _run(cmd)
    tmp_path.replace(path)


def _trim_edge_silence(path: Path) -> None:
    """Trims leading/trailing near-silence from a narration clip in place,
    leaving an 80ms buffer at each edge — NOT internal pauses (natural
    speech prosody), only the dead air at the very start/end that each
    independently-synthesized scene clip tends to carry. Concatenating N
    scenes back-to-back without this stacks the trailing silence of scene i
    against the leading silence of scene i+1 into one noticeably longer
    pause at every cut, confirmed by measuring real ElevenLabs narration
    (2026-08-21): a 7s clip had ~2s of near-silent tail past the last
    detected speech, silencedetect only ever finding a single leading/
    trailing gap wide enough to matter — internal pauses stayed much
    shorter and are left untouched here (start_periods=1 only strips the
    very first/last matching span, not every one throughout).
    No-op for StubTTSProvider's output (a continuous sine tone, never dips
    below the threshold), so this never touches deterministic test audio."""
    tmp_path = path.with_suffix(".trimmed.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path),
        "-af",
        "silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:start_silence=0.08,areverse,"
        "silenceremove=start_periods=1:start_duration=0:start_threshold=-40dB:start_silence=0.08,areverse",
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        str(tmp_path),
    ]
    _run(cmd)
    tmp_path.replace(path)


def build_scene_audio(
    tts_provider: TTSProvider,
    scene: dict[str, Any],
    index: int,
    audio_dir: Path,
    cost_tracker: CostTracker,
) -> Path:
    audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = audio_dir / f"scene_{index:02d}.wav"
    result = tts_provider.synthesize_scene(scene, index, out_path, cost_tracker)
    _trim_edge_silence(result)
    _apply_narration_speed(result)
    return result


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


_FREEZE_EVENT_RE = re.compile(
    r"lavfi\.freezedetect\.freeze_(start|duration|end):\s*([\d.]+)"
)

# Beyond ~1.8x, stretching a short usable clip into slow motion starts
# reading as barely-moving rather than as real playback speed.
MAX_CLIP_STRETCH_FACTOR = 1.8
# Zoom increment applied every output frame (30fps) by the always-on Ken
# Burns pan in build_scene_video_segment_from_clip. Chosen empirically
# (2026-08-27): a subtler rate (~0.0003-0.0006/frame) still reads as
# perceptibly static over 1-2s windows on real footage — confirmed via direct
# frame comparison, not just freezedetect. This rate produces clearly
# noticeable zoom progression by 5s in.
KEN_BURNS_ZOOM_PER_FRAME = 0.0012
# zoompan crops from an oversized source; the crop coordinates round to whole
# pixels each frame, so too little headroom relative to clip length means
# consecutive frames can round to the identical crop and look duplicated.
# 1.4x gives enough pixel budget across a realistic ~8s scene to avoid that.
KEN_BURNS_HEADROOM = 1.4


def _leading_freeze_seconds(clip_path: Path, clip_duration: float) -> float:
    """How long clip_path holds a static (near-zero-motion) frame starting
    from time 0 — real image-to-video models (confirmed for real 2026-08-21,
    measured against actual paid Kling clips from the electricity video)
    routinely hold the source pose for a beat, sometimes over half the
    clip, before any real motion begins. That dead time landing right after
    a scene cut is what reads as "a pause between scenes" even though the
    narration audio is already talking — trimming it (see
    build_scene_video_segment_from_clip) gets straight to the motion, and
    tpad's existing end-of-clip hold absorbs the same amount of frozen time
    at the scene's END instead, which reads as "holding on the result"
    rather than "nothing is happening yet".

    Detects via ffmpeg's freezedetect filter, coalescing any back-to-back
    freeze intervals starting at t=0 into one combined leading-freeze
    length (a single real freeze sometimes gets reported as 2-3 adjacent
    intervals). Capped at 70% of the clip's own duration so a clip that's
    frozen almost throughout still keeps a meaningful slice of real motion
    rather than being trimmed to nothing."""
    cmd = [
        "ffmpeg", "-i", str(clip_path),
        "-vf", "freezedetect=n=-30dB:d=0.15",
        "-an", "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    events: list[tuple[str, float]] = [
        (kind, float(value)) for kind, value in _FREEZE_EVENT_RE.findall(result.stderr)
    ]
    # Group the flat (kind, value) stream back into (start, duration, end) triples.
    triples: list[tuple[float, float, float]] = []
    current: dict[str, float] = {}
    for kind, value in events:
        current[kind] = value
        if {"start", "duration", "end"}.issubset(current):
            triples.append((current["start"], current["duration"], current["end"]))
            current = {}

    leading_end = 0.0
    for start, _duration, end in triples:
        if abs(start - leading_end) < 0.05:
            leading_end = end
        else:
            break
    return min(leading_end, clip_duration * 0.7)


# Below this, a padded remainder gets a hard cut to a sticker-style beat
# instead of a held-frame Ken Burns fudge — a cut this short would read as a
# glitch, not an edit, so it's not worth the extra concat complexity.
CUT_IN_MIN_PAD_SECONDS = 0.75


def usable_clip_seconds(clip_path: Path) -> float:
    """How much of this raw clip is real, non-frozen motion — the leading
    freeze skipped, everything after it counted. Public so pipeline.py can
    decide whether a scene needs a second real clip without reaching into
    this module's freeze-detection internals."""
    clip_duration = probe_duration(clip_path)
    skip_seconds = _leading_freeze_seconds(clip_path, clip_duration)
    return max(1.0 / FPS, clip_duration - skip_seconds)


def extract_last_frame(clip_path: Path, out_path: Path) -> Path:
    """Grabs a frame near the raw clip's end as a still image — used as the
    continuation source for a second real video-generation call when the
    first clip's motion doesn't cover the whole scene (see
    pipeline._render_scene_clips).

    Seeking to within one frame of the exact end (clip_duration - 1/FPS)
    silently produced zero output frames on a real Kling clip (ffmpeg exited
    0 with no error, no file written) — the seek landed past the last frame
    ffmpeg would actually decode. 0.15s of margin (well over one frame at
    30fps) fixed it; still verified explicitly below rather than trusting
    exit code 0 to mean a frame was actually written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clip_duration = probe_duration(clip_path)
    timestamp = max(0.0, clip_duration - 0.15)
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", str(clip_path),
        "-frames:v", "1", str(out_path),
    ])
    if not out_path.exists():
        raise RuntimeError(f"extract_last_frame produced no output for {clip_path} at t={timestamp:.3f}")
    return out_path


def build_scene_video_segment_from_clip(
    clip_paths: list[Path],
    duration: float,
    caption_overlay: Image.Image | None,
    index: int,
    segments_dir: Path,
    *,
    timed_caption_overlays: list[TimedCaptionOverlay] | None = None,
    image_path: Path | None = None,
) -> Path:
    """Same role as build_scene_video_segment(), for one or more animated
    clips instead of a static frame: skips any leading static/frozen hold in
    each source clip (see _leading_freeze_seconds), scales to frame size,
    composites timed caption overlays, and retimes the usable source motion
    to the scene's measured narration duration.

    clip_paths is an ORDERED list (usually 1, sometimes 2 — see
    pipeline._render_scene_clips): each clip's usable motion plays up to
    MAX_CLIP_STRETCH_FACTOR stretched against whatever time is still left
    ("try to continue reasonably" stops there) before moving to the next
    clip in the list. Only once every clip is exhausted and real time is
    still left over does it hard-cut to a static sticker-style pop-in-and-
    hold beat on the scene's own base image (image_path) — the same visual
    language build_scene_video_segment_from_still() uses — rather than
    holding a clip's last frame indefinitely (a real Kling clip observed
    2026-08-27/28 spent 6.29s of an 8.5s scene in exactly that held-frame
    state — most of the shot). image_path is optional (and
    CUT_IN_MIN_PAD_SECONDS gates it even when given), so callers/tests that
    don't supply a scene image keep the older held-frame-only behavior for
    whatever's left after the last clip."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    out_path = segments_dir / f"seg_{index:02d}.mp4"

    if timed_caption_overlays is None:
        if caption_overlay is None:
            raise ValueError("caption_overlay or timed_caption_overlays is required")
        fallback_box = CaptionBox(0, 0, FRAME_WIDTH, FRAME_HEIGHT)
        timed_caption_overlays = [
            TimedCaptionOverlay(caption_overlay, 0.0, duration, fallback_box)
        ]

    overlay_paths: list[Path] = []
    for cue_index, timed in enumerate(timed_caption_overlays):
        path = segments_dir / f"caption_overlay_{index:02d}_{cue_index:02d}.png"
        timed.image.save(path)
        overlay_paths.append(path)

    oversized_w = round(FRAME_WIDTH * KEN_BURNS_HEADROOM)
    oversized_h = round(FRAME_HEIGHT * KEN_BURNS_HEADROOM)

    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    motion_filters: list[str] = []
    motion_labels: list[str] = []
    remaining = duration

    for clip_path in clip_paths:
        if remaining <= 0.05:
            break
        cmd.extend(["-i", str(clip_path)])
        clip_index = len(motion_labels)
        clip_duration = probe_duration(clip_path)
        skip_seconds = _leading_freeze_seconds(clip_path, clip_duration)
        usable_duration = max(1.0 / FPS, clip_duration - skip_seconds)
        natural_stretch = remaining / usable_duration
        stretch_factor = min(natural_stretch, MAX_CLIP_STRETCH_FACTOR)
        played_duration = usable_duration * stretch_factor
        remaining = max(0.0, remaining - played_duration)

        label = f"motion{clip_index}"
        motion_labels.append(label)
        # A continuous slow zoom guarantees visible motion for a held tail
        # (a held frame is not literally static on screen) and adds a small
        # floor of motion even where a clip's own animation is too subtle
        # to read as movement.
        motion_filters.append(
            f"[{clip_index}:v]trim=start={skip_seconds:.3f}:duration={usable_duration:.3f},"
            f"setpts={stretch_factor:.8f}*(PTS-STARTPTS),"
            f"scale={oversized_w}:{oversized_h},fps={FPS},"
            f"zoompan=z='1.0+{KEN_BURNS_ZOOM_PER_FRAME}*on':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:fps={FPS}:s={FRAME_WIDTH}x{FRAME_HEIGHT},setsar=1[{label}]"
        )

    pad_duration = remaining
    clips_used = len(motion_labels)
    use_cut_in = image_path is not None and pad_duration > CUT_IN_MIN_PAD_SECONDS

    if use_cut_in:
        cmd.extend(["-loop", "1", "-framerate", str(FPS), "-i", str(image_path)])
        motion_filters.append(
            f"[{clips_used}:v]pad={oversized_w}:{oversized_h}:(ow-{FRAME_WIDTH})/2:(oh-{FRAME_HEIGHT})/2:color=white,"
            f"zoompan=z='{_pop_in_zoom_expr()}':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d=1:fps={FPS}:s={FRAME_WIDTH}x{FRAME_HEIGHT},setsar=1,"
            f"trim=duration={pad_duration:.3f}[cutin0]"
        )
        motion_labels.append("cutin0")
        clips_used += 1
    elif pad_duration > 0.05 and motion_filters:
        # Sub-threshold remainder (or no image_path given): extend the LAST
        # clip's own hold via tpad instead of a separate cut — a near-
        # invisible extra cut isn't worth it. Insert right before that
        # clip's zoompan stage so the Ken Burns floor also covers the hold.
        last_filter = motion_filters[-1]
        insertion_point = last_filter.index(",zoompan=")
        motion_filters[-1] = (
            last_filter[:insertion_point]
            + f",tpad=stop_mode=clone:stop_duration={pad_duration:.3f}"
            + last_filter[insertion_point:]
        )

    if len(motion_labels) > 1:
        concat_inputs = "".join(f"[{label}]" for label in motion_labels)
        base_filter = ";".join([*motion_filters, f"{concat_inputs}concat=n={len(motion_labels)}:v=1:a=0[base0]"])
    else:
        base_filter = motion_filters[0].replace(f"[{motion_labels[0]}]", "[base0]")

    next_input_index = clips_used
    filters = [base_filter]
    prior_label = "base0"
    for cue_index, (timed, overlay_path) in enumerate(zip(timed_caption_overlays, overlay_paths)):
        cue_label, next_input_index = _append_caption_cue_stage(
            cmd, filters, cue_index, timed, overlay_path, segments_dir, index, next_input_index
        )
        out_label = f"captioned{cue_index}"
        filters.append(
            f"[{prior_label}][{cue_label}]overlay=0:0:"
            f"enable='gte(t,{timed.start:.3f})*lt(t,{timed.end:.3f})'[{out_label}]"
        )
        prior_label = out_label

    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", f"[{prior_label}]",
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        str(out_path),
    ])
    _run(cmd)
    return out_path


# Motion-graphics "sticker" compositor — replaces continuous AI-video (Kling/
# Hailuo) animation. Root cause of the switch (2026-08-27): a real reference
# short in this niche uses static character/prop stills with a snappy
# scale-bounce pop-in and hard cuts, not continuous AI "performance" —
# confirmed against real Kling output that was frozen for its ENTIRE raw
# duration on 3 of 6 scenes even with aggressive motion prompting. A pop-in
# on a still image is guaranteed to never read as static (it's not depending
# on an AI model's motion at all) and costs zero video-generation spend.
POP_IN_UP_FRAMES = 5          # 0 -> overshoot, in output frames (30fps => 0.167s)
POP_IN_SETTLE_FRAMES = 8      # overshoot -> rest, in output frames (30fps => 0.267s)
POP_IN_START_SCALE = 0.70
POP_IN_OVERSHOOT_SCALE = 1.12
POP_IN_REST_SCALE = 1.00
# zoompan crops from an oversized, white-padded canvas; needs enough margin
# to support the smallest (zoomed-out) pop-in scale without running out of
# source pixels. 1/POP_IN_START_SCALE = 1.43x is the hard minimum; 1.6x
# leaves real margin.
STICKER_HEADROOM = 1.6


def _pop_in_zoom_expr() -> str:
    up_slope = (POP_IN_OVERSHOOT_SCALE - POP_IN_START_SCALE) / POP_IN_UP_FRAMES
    settle_slope = (POP_IN_OVERSHOOT_SCALE - POP_IN_REST_SCALE) / (POP_IN_SETTLE_FRAMES - POP_IN_UP_FRAMES)
    return (
        f"if(lt(on,{POP_IN_UP_FRAMES}),{POP_IN_START_SCALE}+{up_slope}*on,"
        f"if(lt(on,{POP_IN_SETTLE_FRAMES}),{POP_IN_OVERSHOOT_SCALE}-{settle_slope}*(on-{POP_IN_UP_FRAMES}),"
        f"{POP_IN_REST_SCALE}))"
    )


# Per-cue caption scale-punch: each caption cue briefly overshoots-then-
# settles (same curve as the image pop-in) so captions read as "popping"
# stickers rather than static karaoke text. Rendered as pre-baked PIL frames,
# NOT an ffmpeg zoompan expression — a live test proved zoompan's crop
# window clamps once the requested zoom-out exceeds the source's available
# margin toward whichever edge the anchor is closer to, and this pipeline's
# captions default to position="top" (close to the top edge), so the
# effective anchor silently shifted and the caption visibly drifted down as
# it scaled. Plain PIL resize+paste has no such source-bounds restriction:
# verified directly (a top-positioned caption's alpha-bbox center moved by
# <2px across the full 0.70x-1.12x scale range).
CAPTION_PUNCH_FRAMES = POP_IN_SETTLE_FRAMES
CAPTION_PUNCH_SECONDS = CAPTION_PUNCH_FRAMES / FPS
# Below this cue length there isn't enough time left after the punch for a
# meaningful hold — fall back to the plain static overlay instead.
CAPTION_PUNCH_MIN_CUE_SECONDS = 0.45
# Crop margin around the caption's own box so the overshoot scale (1.12x)
# doesn't clip glyphs at the crop edge.
CAPTION_PUNCH_CROP_MARGIN = 1.5


def _pop_scale_at_frame(frame_idx: int) -> float:
    if frame_idx < POP_IN_UP_FRAMES:
        return POP_IN_START_SCALE + (POP_IN_OVERSHOOT_SCALE - POP_IN_START_SCALE) * frame_idx / POP_IN_UP_FRAMES
    settled = frame_idx - POP_IN_UP_FRAMES
    return POP_IN_OVERSHOOT_SCALE - (POP_IN_OVERSHOOT_SCALE - POP_IN_REST_SCALE) * settled / (
        POP_IN_SETTLE_FRAMES - POP_IN_UP_FRAMES
    )


def _caption_punch_frame_paths(overlay_image: Image.Image, box: CaptionBox, out_dir: Path, prefix: str) -> list[Path]:
    """CAPTION_PUNCH_FRAMES full-canvas frames, each identical to
    overlay_image except a crop around `box` (the caption's own bounding
    box — NOT the whole canvas) is resized/re-pasted anchored on its own
    center. Scaling only that crop, against a base canvas that's otherwise
    an exact copy of overlay_image, means any unrelated content already
    baked into overlay_image elsewhere (e.g. the last scene's Subscribe CTA,
    which is composited near the bottom regardless of where the caption
    itself sits) is carried over untouched rather than being dragged along
    by the caption's own scale/anchor."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cx = (box.left + box.right) / 2
    cy = (box.top + box.bottom) / 2
    half_w = (box.right - box.left) / 2 * CAPTION_PUNCH_CROP_MARGIN
    half_h = (box.bottom - box.top) / 2 * CAPTION_PUNCH_CROP_MARGIN
    crop_box = (
        max(0, round(cx - half_w)), max(0, round(cy - half_h)),
        min(overlay_image.width, round(cx + half_w)), min(overlay_image.height, round(cy + half_h)),
    )
    crop = overlay_image.crop(crop_box)
    crop_cx = cx - crop_box[0]
    crop_cy = cy - crop_box[1]
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]

    paths: list[Path] = []
    for frame_idx in range(CAPTION_PUNCH_FRAMES):
        scale = _pop_scale_at_frame(frame_idx)
        new_w = max(1, round(crop.width * scale))
        new_h = max(1, round(crop.height * scale))
        resized = crop.resize((new_w, new_h), Image.LANCZOS)
        canvas = overlay_image.copy()
        # Clear the original (unscaled) crop region first — otherwise a
        # smaller (scale<1) resized paste leaves the original larger content
        # still visible underneath around its edges (ghosting).
        canvas.paste(Image.new("RGBA", (crop_w, crop_h), (0, 0, 0, 0)), (crop_box[0], crop_box[1]))
        paste_x = round(crop_box[0] + crop_cx - crop_cx * scale)
        paste_y = round(crop_box[1] + crop_cy - crop_cy * scale)
        canvas.paste(resized, (paste_x, paste_y), resized)
        path = out_dir / f"{prefix}_{frame_idx:03d}.png"
        canvas.save(path)
        paths.append(path)
    return paths


def _append_caption_cue_stage(
    cmd: list[str],
    filters: list[str],
    cue_index: int,
    timed: TimedCaptionOverlay,
    overlay_path: Path,
    segments_dir: Path,
    index: int,
    next_input_index: int,
) -> tuple[str, int]:
    """Appends the ffmpeg input(s) and filter stage that produce one caption
    cue's own video branch (a scale-punch prefix + static hold, or a plain
    static branch for cues too short to punch) — shared by
    build_scene_video_segment_from_still and build_scene_video_segment_from_clip,
    which otherwise have separate overlay-compositing loops. Returns
    (output_label, next_input_index) for the caller to overlay onto its main
    composite and keep tracking ffmpeg's positional input indices."""
    label = f"cue{cue_index}"
    cue_duration = timed.end - timed.start

    if cue_duration < CAPTION_PUNCH_MIN_CUE_SECONDS:
        cmd.extend(["-loop", "1", "-i", str(overlay_path)])
        filters.append(f"[{next_input_index}:v]format=rgba[{label}]")
        return label, next_input_index + 1

    punch_dir = segments_dir / "caption_punches"
    prefix = f"punch_{index:02d}_{cue_index:02d}"
    _caption_punch_frame_paths(timed.image, timed.box, punch_dir, prefix)
    punch_pattern = punch_dir / f"{prefix}_%03d.png"

    cmd.extend(["-framerate", str(FPS), "-i", str(punch_pattern)])
    punch_input_index = next_input_index
    next_input_index += 1
    cmd.extend(["-loop", "1", "-i", str(overlay_path)])
    hold_input_index = next_input_index
    next_input_index += 1

    hold_duration = max(0.0, cue_duration - CAPTION_PUNCH_SECONDS)
    filters.append(f"[{punch_input_index}:v]format=rgba[{label}_punch]")
    filters.append(f"[{hold_input_index}:v]format=rgba,trim=duration={hold_duration:.3f}[{label}_hold]")
    filters.append(
        f"[{label}_punch][{label}_hold]concat=n=2:v=1:a=0,setpts=PTS+{timed.start:.3f}/TB[{label}]"
    )
    return label, next_input_index


def build_scene_video_segment_from_still(
    image_path: Path,
    duration: float,
    index: int,
    segments_dir: Path,
    *,
    timed_caption_overlays: list[TimedCaptionOverlay],
) -> Path:
    """Sticker-style scene segment: a still image pops in (scale-bounce
    overshoot, ~0.27s) then holds for the rest of the scene's duration —
    the motion-graphics grammar (static art + snap transitions), not a
    continuous AI-generated "performance". Never reads as frozen/static in
    the way a stalled I2V clip can, because there's no continuous motion
    being depended on in the first place."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    out_path = segments_dir / f"seg_{index:02d}.mp4"

    overlay_paths: list[Path] = []
    for cue_index, timed in enumerate(timed_caption_overlays):
        path = segments_dir / f"caption_overlay_{index:02d}_{cue_index:02d}.png"
        timed.image.save(path)
        overlay_paths.append(path)

    oversized_w = round(FRAME_WIDTH * STICKER_HEADROOM)
    oversized_h = round(FRAME_HEIGHT * STICKER_HEADROOM)
    # Pop once per scene (when the image itself changes), not per caption
    # cue — reverted 2026-08-29 per direct user feedback watching a real
    # video: re-popping the whole image on every cue (cues landed every
    # ~0.8-1.5s for punchy narration, not the ~2-3s originally targeted)
    # read as constant zooming/fidgeting rather than a deliberate beat. The
    # caption's own per-cue scale-punch (below) is untouched — new caption
    # text still gets its own small pop as it appears, which is a much
    # smaller, less distracting effect than re-zooming the whole frame.
    base_filter = (
        f"[0:v]pad={oversized_w}:{oversized_h}:(ow-{FRAME_WIDTH})/2:(oh-{FRAME_HEIGHT})/2:color=white,"
        f"zoompan=z='{_pop_in_zoom_expr()}':"
        "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:fps={FPS}:s={FRAME_WIDTH}x{FRAME_HEIGHT}[base0]"
    )

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(FPS), "-i", str(image_path),
    ]
    next_input_index = 1
    filters = [base_filter]
    prior_label = "base0"
    for cue_index, (timed, overlay_path) in enumerate(zip(timed_caption_overlays, overlay_paths)):
        cue_label, next_input_index = _append_caption_cue_stage(
            cmd, filters, cue_index, timed, overlay_path, segments_dir, index, next_input_index
        )
        out_label = f"captioned{cue_index}"
        filters.append(
            f"[{prior_label}][{cue_label}]overlay=0:0:"
            f"enable='gte(t,{timed.start:.3f})*lt(t,{timed.end:.3f})'[{out_label}]"
        )
        prior_label = out_label

    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", f"[{prior_label}]",
        "-t", f"{duration:.3f}",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-threads", "1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact",
        "-metadata", "creation_time=1970-01-01T00:00:00Z",
        str(out_path),
    ])
    _run(cmd)
    return out_path


def assemble_stickers(
    scenes: list[dict[str, Any]],
    image_source: Callable[[int, dict[str, Any]], Path],
    audio: list[SceneAudio],
    workdir: Path,
    out_mp4: Path,
    caption_style: str | None = None,
    caution_text: str | None = None,
    subscribe_cta_text: str | None = None,
) -> dict[str, Any]:
    """Sticker/motion-graphics counterpart to assemble()/assemble_animated():
    image_source(index, scene) -> that scene's already-generated still image
    (no video provider involved at all — see build_scene_video_segment_from_still).
    Same caption/duration-fitting responsibility split as the other two
    assemble variants."""
    segments_dir = workdir / "segments"
    workdir.mkdir(parents=True, exist_ok=True)

    if len(audio) != len(scenes):
        raise ValueError(f"expected {len(scenes)} audio entries, got {len(audio)}")

    caption_boxes: list[CaptionBox] = []
    segment_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        image_path = image_source(i, scene)
        timed_overlays, box = build_timed_caption_overlays(
            scene["narration"],
            audio[i].duration,
            caption_style=caption_style,
            caution_text=caution_text if i == len(scenes) - 1 else None,
            subscribe_cta_text=subscribe_cta_text if i == len(scenes) - 1 else None,
        )
        caption_boxes.append(box)

        seg_path = build_scene_video_segment_from_still(
            image_path,
            audio[i].duration,
            i,
            segments_dir,
            timed_caption_overlays=timed_overlays,
        )
        segment_paths.append(seg_path)

    tail = concat_and_mux(segment_paths, [a.path for a in audio], workdir, out_mp4)
    return {
        "caption_boxes": caption_boxes,
        **tail,
    }


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
    cue_number = 1
    for scene, duration in zip(scenes, durations):
        for cue in narration_caption_cues(scene["narration"], duration):
            lines.append(str(cue_number))
            lines.append(f"{fmt(cursor + cue.start)} --> {fmt(cursor + cue.end)}")
            lines.append(cue.text)
            lines.append("")
            cue_number += 1
        cursor += duration
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
                "spoken_narration": scene["narration"],
                "caption_cues": [
                    {
                        "text": cue.text,
                        "start": round(start + cue.start, 3),
                        "end": round(start + cue.end, 3),
                    }
                    for cue in narration_caption_cues(scene["narration"], duration)
                ],
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
    caption_style: str | None = None,
    caution_text: str | None = None,
    subscribe_cta_text: str | None = None,
) -> dict[str, Any]:
    """Runs the full assembly for one stage (placeholder or generated-image).
    frame_source(index, scene) -> a base PIL Image (pre-caption) for that scene.
    audio must already exist — see synthesize_scenes(), called once and shared
    across every stage so narration is never re-synthesized (and never
    re-charged) per stage. Video segment length is driven by audio[i].duration
    (the MEASURED actual audio length), never scene["duration"] (the script's
    nominal estimate) — this is what keeps video/audio in sync once a real,
    variable-length TTS provider replaces the stub.
    caution_text, if given, is composited as a small bottom-of-frame badge
    onto the LAST scene ONLY, on top of that scene's real caption — it must
    never replace a scene's own caption (confirmed for real 2026-08-21: every
    yellow-topic video was silently ending on a fixed caution string instead
    of its actual payoff line).
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
        frame_path, box = build_scene_frame(scene, i, base_image, frames_dir, caption_style=caption_style)
        caution_box = None
        if caution_text and i == len(scenes) - 1:
            badged, caution_box = draw_caution_badge(Image.open(frame_path), caution_text)
            badged.save(frame_path)
        if subscribe_cta_text and i == len(scenes) - 1:
            # Stack above the caution badge instead of both anchoring to
            # the same bottom spot — see build_timed_caption_overlays'
            # matching fix for the real bug this addresses.
            bottom_limit = caution_box.top - SUBSCRIBE_CTA_STACK_GAP if caution_box else None
            cta_frame = draw_subscribe_cta(Image.open(frame_path), subscribe_cta_text, bottom_limit=bottom_limit)
            cta_frame.save(frame_path)
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
    clip_source: Callable[[int, dict[str, Any]], list[Path]],
    audio: list[SceneAudio],
    workdir: Path,
    out_mp4: Path,
    caption_style: str | None = None,
    caution_text: str | None = None,
    image_source: Callable[[int, dict[str, Any]], Path] | None = None,
    subscribe_cta_text: str | None = None,
) -> dict[str, Any]:
    """Animated-scene counterpart to assemble(): clip_source(index, scene) ->
    an ordered list of one or more raw, uncaptioned animated clip paths for
    that scene (see pipeline._render_scene_clips, providers/video.py).
    Captioning and duration-fitting happen here — same division of
    responsibility as assemble()'s frame_source, the caller only supplies
    the per-scene visual content.
    image_source(index, scene), if given, returns that scene's already-
    generated base image — passed through to build_scene_video_segment_from_clip
    as the hard-cut-in beat for whatever time is left once a clip's real
    motion runs out (see that function's docstring). Optional: omitting it
    keeps the older held-frame-only behavior for callers that don't have a
    scene image on hand.
    caution_text: see assemble()'s docstring — composited onto the LAST
    scene only, on top of (never instead of) its real caption."""
    segments_dir = workdir / "segments"
    workdir.mkdir(parents=True, exist_ok=True)

    if len(audio) != len(scenes):
        raise ValueError(f"expected {len(scenes)} audio entries, got {len(audio)}")

    caption_boxes: list[CaptionBox] = []
    segment_paths: list[Path] = []

    for i, scene in enumerate(scenes):
        clip_paths = clip_source(i, scene)
        timed_overlays, box = build_timed_caption_overlays(
            scene["narration"],
            audio[i].duration,
            caption_style=caption_style,
            caution_text=caution_text if i == len(scenes) - 1 else None,
            subscribe_cta_text=subscribe_cta_text if i == len(scenes) - 1 else None,
        )
        caption_boxes.append(box)

        seg_path = build_scene_video_segment_from_clip(
            clip_paths,
            audio[i].duration,
            None,
            i,
            segments_dir,
            timed_caption_overlays=timed_overlays,
            image_path=image_source(i, scene) if image_source else None,
        )
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

    # The WAV lands within ~0.05 LU of target after the two-pass normalize
    # above, but mux_final's AAC re-encode (lossy, 160kbps) can shift the
    # audio's MEASURED integrated loudness by up to ~1.3 LU relative to the
    # source WAV — confirmed in practice: a WAV measured exactly -14.00
    # LUFS produced a final .mp4 measuring -15.08 to -15.29. verify.py (and
    # any real platform) measures the final container's audio, not the
    # pre-encode WAV, so that's what must be corrected against. A single
    # linear-gain nudge is enough since the residual is small and AAC's
    # further shift on a re-encode at a slightly different level is noise-level.
    final_stats = loudnorm_measure(out_mp4)
    final_measured_i = float(final_stats["input_i"])
    if abs(final_measured_i - LOUDNORM_TARGET_I) > LOUDNORM_CORRECTION_MARGIN_LU:
        correction_db = LOUDNORM_TARGET_I - final_measured_i
        narration_corrected = workdir / "narration_corrected.wav"
        _run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(narration_normalized),
            "-af", f"volume={correction_db:.3f}dB",
            "-ar", "48000",
            "-fflags", "+bitexact", "-flags:a", "+bitexact",
            str(narration_corrected),
        ])
        mux_final(video_only, narration_corrected, out_mp4)
        narration_normalized = narration_corrected
        final_stats = loudnorm_measure(out_mp4)
        final_measured_i = float(final_stats["input_i"])

    return {
        "audio_paths": audio_paths,
        "video_only": video_only,
        "narration_raw": narration_raw,
        "narration_normalized": narration_normalized,
        "loudnorm_measure_pass1": loudnorm_stats_pass1,
        "final_loudness_i": final_measured_i,
        "out_mp4": out_mp4,
    }
