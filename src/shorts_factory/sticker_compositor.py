"""Layered sticker compositor — renders 12–15 isolated sticker assets per video
onto a white canvas with fade/slide entrances and semantic idle motion (no bounce)."""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat

from .captions import FRAME_HEIGHT, FRAME_WIDTH, load_font_by_family

ENTRANCE_FRAMES = 8
FADE_START_SCALE = 0.94
FADE_END_SCALE = 1.0
SLIDE_OFFSET_PX = 28
FLOAT_AMPLITUDE_PX = 7.0
FLOAT_PERIOD_S = 2.4
SPIN_AMPLITUDE_DEG = 8.0
SPIN_PERIOD_S = 3.0
BREATHE_SCALE_DELTA = 0.022
BREATHE_PERIOD_S = 2.5
FLICKER_PERIOD_S = 0.18
DRIFT_AMPLITUDE_PX = 8.0
DRIFT_PERIOD_S = 1.6
MOTION_SCALE = 1.0
COLLISION_PAD_PX = 18
SAFE_BOTTOM_GUARD = 280

POSITION_ANCHORS: dict[str, tuple[float, float]] = {
    "center": (0.5, 0.52),
    "top_left": (0.28, 0.34),
    "top_right": (0.72, 0.34),
    "bottom_left": (0.30, 0.72),
    "bottom_right": (0.70, 0.72),
}


@dataclass
class PlacedRect:
    left: int
    top: int
    right: int
    bottom: int


def configure_motion(scale: float = 1.0) -> None:
    global MOTION_SCALE
    MOTION_SCALE = max(0.5, scale)


def _scaled(value: float) -> float:
    return value * MOTION_SCALE


def _content_bbox(image: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
    rgb = image.convert("RGB")
    for t in (threshold, threshold - 8, threshold - 15):
        r, g, b = rgb.split()
        below = lambda ch, thr=t: ch.point(lambda p, thr=thr: 255 if p < thr else 0)
        mask = ImageChops.lighter(ImageChops.lighter(below(r), below(g)), below(b))
        if ImageStat.Stat(mask).mean[0] >= 2.0:
            return mask.getbbox()
    return None


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


def _entrance_transform(entrance: str, local_t: float) -> tuple[float, float, float, float]:
    progress = _ease_out(local_t / (ENTRANCE_FRAMES / 30.0))
    scale = FADE_START_SCALE + (FADE_END_SCALE - FADE_START_SCALE) * progress
    alpha = progress
    slide_x = 0.0
    slide_y = 0.0
    if entrance == "slide_up":
        slide_y = _scaled(SLIDE_OFFSET_PX) * (1.0 - progress)
    elif entrance == "slide_left":
        slide_x = _scaled(SLIDE_OFFSET_PX) * (1.0 - progress)
    return scale, alpha, slide_x, slide_y


def _idle_offset(idle: str, local_t: float) -> tuple[float, float, float]:
    if idle == "hold":
        return 0.0, 0.0, 0.0
    if idle == "float":
        dy = _scaled(FLOAT_AMPLITUDE_PX) * math.sin(2 * math.pi * local_t / FLOAT_PERIOD_S)
        return 0.0, dy, 0.0
    if idle == "drift":
        dy = _scaled(DRIFT_AMPLITUDE_PX) * math.sin(2 * math.pi * local_t / DRIFT_PERIOD_S)
        dx = _scaled(DRIFT_AMPLITUDE_PX) * 0.4 * math.cos(2 * math.pi * local_t / DRIFT_PERIOD_S)
        return dx, dy, 0.0
    if idle == "spin":
        rot = _scaled(SPIN_AMPLITUDE_DEG) * math.sin(2 * math.pi * local_t / SPIN_PERIOD_S)
        return 0.0, 0.0, rot
    if idle == "breathe":
        return 0.0, 0.0, 0.0
    if idle == "flicker":
        return 0.0, 0.0, 0.0
    dy = _scaled(FLOAT_AMPLITUDE_PX) * 0.6 * math.sin(2 * math.pi * local_t / FLOAT_PERIOD_S)
    return 0.0, dy, 0.0


def _idle_alpha(idle: str, local_t: float) -> float:
    if idle == "flicker":
        wave = math.sin(2 * math.pi * local_t / FLICKER_PERIOD_S)
        return 0.82 + 0.18 * (wave * 0.5 + 0.5)
    return 1.0


def _idle_scale(idle: str, local_t: float) -> float:
    if idle == "breathe":
        return 1.0 + _scaled(BREATHE_SCALE_DELTA) * math.sin(2 * math.pi * local_t / BREATHE_PERIOD_S)
    return 1.0


def _sticker_rect(position: str, sticker: Image.Image, scale: float, dx: float, dy: float) -> PlacedRect:
    anchor_x, anchor_y = POSITION_ANCHORS.get(position, POSITION_ANCHORS["center"])
    w = max(1, round(sticker.width * scale))
    h = max(1, round(sticker.height * scale))
    cx = int(FRAME_WIDTH * anchor_x + dx)
    cy = int(FRAME_HEIGHT * anchor_y + dy)
    paste_x = cx - w // 2
    paste_y = cy - h // 2
    return PlacedRect(paste_x, paste_y, paste_x + w, paste_y + h)


def _overlaps(a: PlacedRect, b: PlacedRect, pad: int = COLLISION_PAD_PX) -> bool:
    return not (
        a.right + pad <= b.left
        or b.right + pad <= a.left
        or a.bottom + pad <= b.top
        or b.bottom + pad <= a.top
    )


def _resolve_collision(position: str, rect: PlacedRect, occupied: list[PlacedRect]) -> str:
    if not any(_overlaps(rect, other) for other in occupied):
        return position
    fallbacks = {
        "center": ["bottom_left", "top_right", "bottom_right", "top_left"],
        "top_left": ["top_right", "bottom_left", "bottom_right"],
        "top_right": ["top_left", "bottom_right", "bottom_left"],
        "bottom_left": ["bottom_right", "top_left", "top_right"],
        "bottom_right": ["bottom_left", "top_left", "top_right"],
    }
    for candidate in fallbacks.get(position, ["bottom_right", "top_left"]):
        anchor_x, anchor_y = POSITION_ANCHORS[candidate]
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        cx = int(FRAME_WIDTH * anchor_x)
        cy = int(FRAME_HEIGHT * anchor_y)
        trial = PlacedRect(cx - w // 2, cy - h // 2, cx - w // 2 + w, cy - h // 2 + h)
        if not any(_overlaps(trial, other) for other in occupied):
            return candidate
    return position


def _draw_label(canvas: Image.Image, text: str, anchor_rect: PlacedRect) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font_by_family("heavy_sans", 42)
    label = text.upper().strip()
    bbox = draw.textbbox((0, 0), label, font=font, stroke_width=4)
    tw = bbox[2] - bbox[0]
    x = anchor_rect.left + (anchor_rect.right - anchor_rect.left - tw) // 2
    y = min(FRAME_HEIGHT - SAFE_BOTTOM_GUARD, anchor_rect.bottom + 8)
    draw.text((x, y), label, font=font, fill=(30, 120, 255, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
    canvas.alpha_composite(overlay)


def _paste_sticker(
    canvas: Image.Image,
    sticker: Image.Image,
    position: str,
    scale: float,
    alpha: float,
    dx: float,
    dy: float,
    rotation_deg: float,
    occupied: list[PlacedRect],
) -> PlacedRect | None:
    trial_rect = _sticker_rect(position, sticker, scale, dx, dy)
    position = _resolve_collision(position, trial_rect, occupied)
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
    rect = PlacedRect(paste_x, paste_y, paste_x + resized.width, paste_y + resized.height)
    occupied.append(rect)
    return rect


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(round(duration * fps)))
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    crops = {sid: _load_sticker_crop(path) for sid, path in sticker_images.items() if path.exists()}

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-t",
        f"{duration:.3f}",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-threads",
        "1",
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
    canvas = Image.new("RGBA", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255, 255))
    occupied: list[PlacedRect] = []
    label_jobs: list[tuple[str, PlacedRect]] = []

    for spec in sorted(stickers, key=lambda s: s.get("appear_at", 0.0)):
        if spec.get("is_label"):
            continue
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
            scale, alpha, slide_x, slide_y = _entrance_transform(entrance, local_t)
            dx, dy, rot = slide_x, slide_y, 0.0
        else:
            idle_t = local_t - entrance_t
            scale = 1.0
            alpha = _idle_alpha(idle, idle_t)
            scale *= _idle_scale(idle, idle_t)
            dx, dy, rot = _idle_offset(idle, idle_t)
        rect = _paste_sticker(canvas, crop, position, scale, alpha, dx, dy, rot, occupied)
        if rect and spec.get("label"):
            label_jobs.append((spec["label"], rect))

    for label_text, rect in label_jobs:
        _draw_label(canvas, label_text, rect)

    return canvas.convert("RGB")
