"""Caption rendering onto a 1080x1920 frame, kept inside a safe-margin box.

Safe margins exist so captions never sit under a platform's own UI chrome
(status bar area up top, like/comment/share rail and username strip down
the bottom on Shorts/Reels/TikTok). These pixel values are conservative
defaults for 1080x1920, not derived from a specific platform's published
spec — good enough for Phase 0's own rendering pipeline (nothing here is
composited under someone else's live UI), revisit if real-platform preview
testing says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920

# Safe-area margins (pixels) — caption text must stay fully inside this box.
SAFE_TOP = 220
SAFE_BOTTOM = 320
SAFE_SIDES = 70

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]
FONT_SIZE = 54
LINE_SPACING = 12
PADDING = 28
BG_COLOR = (0, 0, 0, 165)  # translucent black card behind the text
TEXT_COLOR = (255, 255, 255, 255)


def _load_font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, FONT_SIZE)
    return ImageFont.load_default(size=FONT_SIZE)


_FONT = None


def get_font() -> ImageFont.FreeTypeFont:
    global _FONT
    if _FONT is None:
        _FONT = _load_font()
    return _FONT


@dataclass
class CaptionBox:
    """The exact pixel rectangle the caption card was drawn in — used by the
    verification step to assert it's fully inside the safe-margin box."""
    left: int
    top: int
    right: int
    bottom: int

    def inside_safe_area(self) -> bool:
        return (
            self.left >= SAFE_SIDES
            and self.right <= FRAME_WIDTH - SAFE_SIDES
            and self.top >= SAFE_TOP
            and self.bottom <= FRAME_HEIGHT - SAFE_BOTTOM
        )


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        w = draw.textlength(candidate, font=font)
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_caption(base: Image.Image, text: str) -> tuple[Image.Image, CaptionBox]:
    """Returns a new RGB image with the caption composited, plus the exact
    box it was drawn in (for the safe-margin assertion)."""
    img = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font()

    max_text_width = FRAME_WIDTH - 2 * SAFE_SIDES - 2 * PADDING
    lines = _wrap_text(text, font, max_text_width, draw)

    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    text_block_width = max(line_widths) if line_widths else 0
    text_block_height = sum(line_heights) + LINE_SPACING * (len(lines) - 1 if lines else 0)

    card_width = text_block_width + 2 * PADDING
    card_height = text_block_height + 2 * PADDING

    # Lower third of the safe area, horizontally centered.
    safe_area_bottom = FRAME_HEIGHT - SAFE_BOTTOM
    card_bottom = safe_area_bottom
    card_top = card_bottom - card_height
    card_left = (FRAME_WIDTH - card_width) // 2
    card_right = card_left + card_width

    # Clamp defensively — verified by inside_safe_area() below regardless.
    card_top = max(card_top, SAFE_TOP)

    draw.rounded_rectangle(
        [card_left, card_top, card_right, card_bottom],
        radius=18,
        fill=BG_COLOR,
    )

    y = card_top + PADDING
    for line, lh in zip(lines, line_heights):
        lw = draw.textlength(line, font=font)
        x = card_left + (card_width - lw) / 2
        draw.text((x, y), line, font=font, fill=TEXT_COLOR)
        y += lh + LINE_SPACING

    composited = Image.alpha_composite(img, overlay).convert("RGB")
    box = CaptionBox(left=card_left, top=card_top, right=card_right, bottom=card_bottom)
    return composited, box
