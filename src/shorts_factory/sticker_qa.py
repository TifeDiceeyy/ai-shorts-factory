"""Post-generation QA for isolated sticker PNGs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


@dataclass(frozen=True)
class StickerQAResult:
    ok: bool
    white_ratio: float
    content_fraction: float
    reason: str = ""


def _white_ratio(rgb: Image.Image, threshold: int = 245) -> float:
    r, g, b = rgb.split()
    below = lambda ch: ch.point(lambda p: 255 if p < threshold else 0)
    mask = ImageChops.lighter(ImageChops.lighter(below(r), below(g)), below(b))
    mean = ImageStat.Stat(mask).mean[0]
    return 1.0 - (mean / 255.0)


def _content_bbox(rgb: Image.Image, threshold: int = 245) -> tuple[int, int, int, int] | None:
    r, g, b = rgb.split()
    below = lambda ch: ch.point(lambda p: 255 if p < threshold else 0)
    mask = ImageChops.lighter(ImageChops.lighter(below(r), below(g)), below(b))
    if ImageStat.Stat(mask).mean[0] < 2.0:
        return None
    return mask.getbbox()


def validate_sticker_image(
    path: Path,
    *,
    min_white_ratio: float = 0.55,
    min_content_fraction: float = 0.004,
    max_content_fraction: float = 0.65,
) -> StickerQAResult:
    if not path.exists():
        return StickerQAResult(False, 0.0, 0.0, "missing file")
    img = Image.open(path).convert("RGB")
    white = _white_ratio(img)
    box = _content_bbox(img)
    if box is None:
        return StickerQAResult(False, white, 0.0, "no isolated content detected")
    content_pixels = (box[2] - box[0]) * (box[3] - box[1])
    total = img.width * img.height
    fraction = content_pixels / total if total else 0.0
    if white < min_white_ratio:
        return StickerQAResult(False, white, fraction, f"background not white enough ({white:.2f})")
    if fraction < min_content_fraction:
        return StickerQAResult(False, white, fraction, "content too small")
    if fraction > max_content_fraction:
        return StickerQAResult(False, white, fraction, "content fills too much of frame")
    return StickerQAResult(True, white, fraction)
