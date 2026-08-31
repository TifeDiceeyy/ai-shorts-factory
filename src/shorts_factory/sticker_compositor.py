"""Layered sticker compositor — renders 12–15 isolated sticker assets per video
onto a white canvas with fade/slide entrances and semantic idle motion (no bounce)."""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from .captions import FRAME_HEIGHT, FRAME_WIDTH

ENTRANCE_FRAMES = 8
FADE_START_SCALE = 0.94
FADE_END_SCALE = 1.0
SLIDE_OFFSET_PX = 28
FLOAT_AMPLITUDE_PX = 4.0
FLOAT_PERIOD_S = 2.4
SPIN_AMPLITUDE_DEG = 6.0
SPIN_PERIOD_S = 3.0
BREATHE_SCALE_DELTA = 0.015
BREATHE_PERIOD_S = 2.5
FLICKER_PERIOD_S = 0.18
DRIFT_AMPLITUDE_PX = 5.0
DRIFT_PERIOD_S = 1.6

POSITION_ANCHORS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.52),
    "top_left": (0.28, 0.34),
    "top_right": (0.72, 0.34),
    "bottom_left": (0.30, 0.68),
    "bottom_right": (0.70, 0.68),
}


def _content_bbox(image: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    r, g, b = rgb.split()
    below = lambda ch: ch.point(lambda p: 255 if p < threshold else 0)
    mask = ImageChops.lighter(ImageChops.lighter(below(r), below(g)), below(b))
    if ImageStat.Stat(mask).mean[0] < 2.0:
        return None
    return mask.getbbox()


def _load_sticker_crop(path: Path, max_height_frac: float = 0.38) -> Image.Image | None:
    img = Image.open(path).convert("RGBA")
    box = _content_bbox(img)
    if box is None:
        return None
    crop = img.crop(box)
    max_h = int(FRAME_HEIGHT * max_height_frac)
    if crop.height > max_h:
        scale = max_h / crop.height
        crop = crop.resize((max(1, round(crop.width * scale)), max_h), Image.LANCZOS)
    return crop


def _ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _entrance_transform(entrance: str, local_t: float) -> tuple[float, float, float]:
    """Returns (scale, alpha, slide_y_px) for entrance phase."""
    progress = _ease_out(local_t / (ENTRANCE_FRAMES / 30.0))
    scale = FADE_START_SCALE + (FADE_END_SCALE - FADE_START_SCALE) * progress
    alpha = progress
    slide_y = 0.0
    if entrance == "slide_up":
        slide_y = SLIDE_OFFSET_PX * (1.0 - progress)
    elif entrance == "slide_left":
        slide_y = 0.0
    return scale, alpha, slide_y


def _idle_offset(idle: str, local_t: float) -> tuple[float, float, float]:
    """Returns (dx, dy, rotation_deg) for idle loop after entrance settles."""
    if idle == "hold":
        return 0.0, 0.0, 0.0
    if idle == "float":
        dy = FLOAT_AMPLITUDE_PX * math.sin(2 * math.pi * local_t / FLOAT_PERIOD_S)
        return 0.0, dy, 0.0
    if idle == "drift":
        dy = DRIFT_AMPLITUDE_PX * math.sin(2 * math.pi * local_t / DRIFT_PERIOD_S)
        dx = DRIFT_AMPLITUDE_PX * 0.4 * math.cos(2 * math.pi * local_t / DRIFT_PERIOD_S)
        return dx, dy, 0.0
    if idle == "spin":
        rot = SPIN_AMPLITUDE_DEG * math.sin(2 * math.pi * local_t / SPIN_PERIOD_S)
        return 0.0, 0.0, rot
    if idle == "breathe":
        return 0.0, 0.0, 0.0
    if idle == "flicker":
        return 0.0, 0.0, 0.0
    dy = FLOAT_AMPLITUDE_PX * 0.6 * math.sin(2 * math.pi * local_t / FLOAT_PERIOD_S)
    return 0.0, dy, 0.0


def _idle_alpha(idle: str, local_t: float) -> float:
    if idle == "flicker":
        wave = math.sin(2 * math.pi * local_t / FLICKER_PERIOD_S)
        return 0.82 + 0.18 * (wave * 0.5 + 0.5)
    return 1.0


def _idle_scale(idle: str, local_t: float) -> float:
    if idle == "breathe":
        return 1.0 + BREATHE_SCALE_DELTA * math.sin(2 * math.pi * local_t / BREATHE_PERIOD_S)
    return 1.0


def _paste_sticker(
    canvas: Image.Image,
    sticker: Image.Image,
    position: str,
    scale: float,
    alpha: float,
    dx: float,
    dy: float,
    rotation_deg: float,
) -> None:
    anchor_x, anchor_y = POSITION_ANCHORS.get(position, POSITION_ANCHORS["center"])
    w = max(1, round(sticker.width * scale))
    h = max(1, round(sticker.height * scale))
    resized = sticker.resize((w, h), Image.LANCZOS)
    if abs(rotation_deg) > 0.05:
        resized = resized.rotate(rotation_deg, resample=Image.BICUBIC, expand=True)
    if alpha < 1.0:
        r, g, b, a = resized.split()
        a = a.point(lambda p: int(p * alpha))
        resized = Image.merge("RGBA", (r, g, b, a))
    cx = int(FRAME_WIDTH * anchor_x + dx)
    cy = int(FRAME_HEIGHT * anchor_y + dy)
    paste_x = cx - resized.width // 2
    paste_y = cy - resized.height // 2
    canvas.paste(resized, (paste_x, paste_y), resized)


def render_sticker_scene_frame(
    stickers: list[dict[str, Any]],
    sticker_images: dict[str, Path],
    t: float,
) -> Image.Image:
    crops = {sid: _load_sticker_crop(path) for sid, path in sticker_images.items() if path.exists()}
    return _render_sticker_scene_frame_cached(stickers, crops, t)


def write_sticker_scene_video(
    stickers: list[dict[str, Any]],
    sticker_images: dict[str, Path],
    duration: float,
    out_path: Path,
    fps: int = 30,
) -> Path:
    """Render sticker layers straight into an mp4 via a rawvideo pipe — avoids
    writing hundreds of intermediate PNGs per scene."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration * fps)))
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    crops = {
        sid: _load_sticker_crop(path)
        for sid, path in sticker_images.items()
        if path.exists()
    }

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}", "-r", str(fps),
        "-i", "-",
        "-t", f"{duration:.3f}",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "ultrafast", "-threads", "1",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            frame = _render_sticker_scene_frame_cached(stickers, crops, t)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.wait() != 0:
            raise RuntimeError(f"sticker scene encode failed: {stderr}")
    except Exception:
        proc.kill()
        raise
    return out_path


def _render_sticker_scene_frame_cached(
    stickers: list[dict[str, Any]],
    crops: dict[str, Image.Image | None],
    t: float,
) -> Image.Image:
    canvas = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    for spec in sorted(stickers, key=lambda s: s.get("appear_at", 0.0)):
        appear_at = float(spec.get("appear_at", 0.0))
        if t < appear_at:
            continue
        crop = crops.get(spec["id"])
        if crop is None:
            continue
        local_t = t - appear_at
        entrance = spec.get("entrance", "fade_in")
        idle = spec.get("idle", "float")
        position = spec.get("position", "center")
        entrance_t = ENTRANCE_FRAMES / 30.0
        if local_t < entrance_t:
            scale, alpha, slide_y = _entrance_transform(entrance, local_t)
            dx, dy, rot = 0.0, slide_y, 0.0
        else:
            idle_t = local_t - entrance_t
            scale = 1.0
            alpha = _idle_alpha(idle, idle_t)
            scale *= _idle_scale(idle, idle_t)
            dx, dy, rot = _idle_offset(idle, idle_t)
        _paste_sticker(canvas, crop, position, scale, alpha, dx, dy, rot)
    return canvas
