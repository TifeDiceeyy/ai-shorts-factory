"""Caption rendering onto a 1080x1920 frame, kept inside a safe-margin box.

Safe margins exist so captions never sit under a platform's own UI chrome
(status bar area up top, like/comment/share rail and username strip down
the bottom on Shorts/Reels/TikTok).

Supports rich kinetic typography styles, randomized fonts/colors/casings,
highlighter badges, stroke outlines, drop shadows, and dual-tone word accents.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920

# Safe-area margins (pixels) — caption text must stay fully inside this box.
SAFE_TOP = 220
SAFE_BOTTOM = 320
SAFE_SIDES = 70

FONT_FAMILIES: dict[str, list[str]] = {
    "impact": [
        "C:/Windows/Fonts/impact.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "heavy_sans": [
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/segoeuiz.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "comic": [
        "C:/Windows/Fonts/comicbd.ttf",
        "C:/Windows/Fonts/comic.ttf",
        "C:/Windows/Fonts/mvboli.ttf",
        "/System/Library/Fonts/Supplemental/Comic Sans MS Bold.ttf",
        "C:/Windows/Fonts/Candarab.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    "modern_clean": [
        "C:/Windows/Fonts/trebucbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/Candarab.ttf",
        "C:/Windows/Fonts/corbelb.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "dramatic_serif": [
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/palab.ttf",
        "C:/Windows/Fonts/cambriab.ttf",
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    "technical_din": [
        "C:/Windows/Fonts/bahnschrift.ttf",
        "C:/Windows/Fonts/framd.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
}

# Each CaptionStyle carries its OWN font_size which overrides this, so
# changing this alone does nothing — the catalog sizes below were raised
# together with it.
DEFAULT_FONT_SIZE = 96
LINE_SPACING = 14
PADDING = 24

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def load_font_by_family(family: str = "impact", size: int = DEFAULT_FONT_SIZE) -> ImageFont.FreeTypeFont:
    """Finds and caches a font matching the requested family and size."""
    cache_key = (family, size)
    if cache_key in _FONT_CACHE:
        return _FONT_CACHE[cache_key]

    candidates = FONT_FAMILIES.get(family, FONT_FAMILIES["heavy_sans"])
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[cache_key] = font
                return font
            except Exception:
                continue

    # Fallback to general candidates
    for fam_candidates in FONT_FAMILIES.values():
        for path in fam_candidates:
            if Path(path).exists():
                try:
                    font = ImageFont.truetype(path, size)
                    _FONT_CACHE[cache_key] = font
                    return font
                except Exception:
                    continue

    font = ImageFont.load_default(size=size)
    _FONT_CACHE[cache_key] = font
    return font


def get_font(size: int = DEFAULT_FONT_SIZE, family: str = "impact") -> ImageFont.FreeTypeFont:
    return load_font_by_family(family=family, size=size)


@dataclass
class CaptionStyle:
    name: str
    font_family: str = "impact"
    font_size: int = 64
    text_color: tuple[int, int, int, int] = (255, 107, 0, 255)  # Vibrant orange (#FF6B00)
    stroke_color: tuple[int, int, int, int] = (0, 0, 0, 255)    # Solid black
    stroke_width: int = 7
    bg_color: tuple[int, int, int, int] | None = None          # Card / Pill background
    bg_radius: int = 18
    casing: str = "upper"                                      # "upper", "title", "original"
    position: str = "top"                                    # "top", "middle", "bottom"
    shadow: bool = True
    shadow_offset: tuple[int, int] = (4, 4)
    shadow_color: tuple[int, int, int, int] = (0, 0, 0, 180)
    accent_color: tuple[int, int, int, int] | None = None       # Highlight keyword color
    dual_tone: bool = False                                    # Split color emphasis for multi-words


# Curated catalog of high-retention TikTok/Shorts kinetic caption styles
CAPTION_STYLES: dict[str, CaptionStyle] = {
    "comic_punch_orange": CaptionStyle(
        name="comic_punch_orange",
        font_family="impact",
        font_size=99,
        text_color=(255, 107, 0, 255),      # Punchy Orange
        stroke_color=(0, 0, 0, 255),
        stroke_width=7,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "electric_neon_yellow": CaptionStyle(
        name="electric_neon_yellow",
        font_family="heavy_sans",
        font_size=99,
        text_color=(255, 230, 0, 255),      # Electric Neon Yellow
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "cyber_cyan_ice": CaptionStyle(
        name="cyber_cyan_ice",
        font_family="technical_din",
        font_size=96,
        text_color=(0, 229, 255, 255),      # Vivid Cyan / Ice Aqua
        stroke_color=(0, 10, 25, 255),
        stroke_width=7,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "hot_magenta_fire": CaptionStyle(
        name="hot_magenta_fire",
        font_family="comic",
        font_size=96,
        text_color=(255, 0, 128, 255),      # Hot Magenta / Pink
        stroke_color=(0, 0, 0, 255),
        stroke_width=7,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "toxic_lime_surge": CaptionStyle(
        name="toxic_lime_surge",
        font_family="heavy_sans",
        font_size=96,
        text_color=(0, 255, 102, 255),      # Toxic Lime Green
        stroke_color=(0, 20, 10, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "pure_white_punch": CaptionStyle(
        name="pure_white_punch",
        font_family="impact",
        font_size=102,
        text_color=(255, 255, 255, 255),    # Clean Pure White
        stroke_color=(0, 0, 0, 255),
        stroke_width=9,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "crimson_alert": CaptionStyle(
        name="crimson_alert",
        font_family="impact",
        font_size=99,
        text_color=(255, 45, 45, 255),      # Bright Crimson Alert
        stroke_color=(0, 0, 0, 255),
        stroke_width=7,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "highlighter_yellow_pill": CaptionStyle(
        name="highlighter_yellow_pill",
        font_family="heavy_sans",
        font_size=96,
        text_color=(255, 230, 0, 255),      # Bright Yellow text (was a pill background)
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "dark_glass_badge": CaptionStyle(
        name="dark_glass_badge",
        font_family="modern_clean",
        font_size=93,
        text_color=(255, 255, 255, 255),    # Crisp White (was a glass-panel background)
        stroke_color=(0, 0, 0, 255),
        stroke_width=7,
        casing="title",
        position="top",
        shadow=True,
    ),
    "royal_blue_pill": CaptionStyle(
        name="royal_blue_pill",
        font_family="heavy_sans",
        font_size=96,
        text_color=(60, 140, 255, 255),     # Royal Blue text (was a pill background)
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "danger_red_badge": CaptionStyle(
        name="danger_red_badge",
        font_family="impact",
        font_size=96,
        text_color=(255, 45, 45, 255),      # Crimson Red text (was a badge background)
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
    ),
    "dual_tone_fire": CaptionStyle(
        name="dual_tone_fire",
        font_family="impact",
        font_size=99,
        text_color=(255, 255, 255, 255),    # Base White
        accent_color=(255, 107, 0, 255),    # Accent Orange first word
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
        dual_tone=True,
    ),
    "dual_tone_electric": CaptionStyle(
        name="dual_tone_electric",
        font_family="heavy_sans",
        font_size=99,
        text_color=(255, 255, 255, 255),    # Base White
        accent_color=(255, 230, 0, 255),    # Accent Electric Yellow first word
        stroke_color=(0, 0, 0, 255),
        stroke_width=8,
        casing="upper",
        position="top",
        shadow=True,
        dual_tone=True,
    ),
}

STYLE_NAMES = list(CAPTION_STYLES.keys())


def get_random_caption_style_name(seed: Any = None, exclude: list[str] | None = None) -> str:
    """Returns the name key of a randomly selected style from CAPTION_STYLES
    (not a randomized/jittered CaptionStyle instance like
    get_random_caption_style() below) — for cases that need a plain string
    to persist (e.g. script.json's caption_style field) and re-resolve later
    via resolve_caption_style(name), rather than a one-off in-memory object."""
    rng = random.Random(seed) if seed is not None else random
    available_keys = [k for k in STYLE_NAMES if not exclude or k not in exclude]
    return rng.choice(available_keys)


def get_random_caption_style(seed: Any = None, exclude: list[str] | None = None) -> CaptionStyle:
    """Returns a randomized, dynamic CaptionStyle with variations in font, color, and casing."""
    rng = random.Random(seed) if seed is not None else random
    available_keys = [k for k in STYLE_NAMES if not exclude or k not in exclude]
    chosen_key = rng.choice(available_keys)
    base_style = CAPTION_STYLES[chosen_key]

    # Dynamically randomize font size slightly (+/- 4px) and pick compatible font family variation
    families = list(FONT_FAMILIES.keys())
    random_family = rng.choice([base_style.font_family, rng.choice(families)])
    size_variation = rng.choice([-4, -2, 0, 2, 4])
    resolved_size = max(52, min(74, base_style.font_size + size_variation))

    return CaptionStyle(
        name=f"{base_style.name}_rand",
        font_family=random_family,
        font_size=resolved_size,
        text_color=base_style.text_color,
        stroke_color=base_style.stroke_color,
        stroke_width=base_style.stroke_width,
        bg_color=base_style.bg_color,
        bg_radius=base_style.bg_radius,
        casing=base_style.casing,
        position=base_style.position,
        shadow=base_style.shadow,
        shadow_offset=base_style.shadow_offset,
        shadow_color=base_style.shadow_color,
        accent_color=base_style.accent_color,
        dual_tone=base_style.dual_tone,
    )


def resolve_caption_style(style: CaptionStyle | str | None = None, seed: Any = None) -> CaptionStyle:
    """Resolves an input into a concrete CaptionStyle."""
    if isinstance(style, CaptionStyle):
        return style
    if isinstance(style, str):
        if style in CAPTION_STYLES:
            return CAPTION_STYLES[style]
        if style == "random":
            return get_random_caption_style(seed=seed)
        if style in ("comic_top", "default"):
            # Provide randomized stylish selection
            return get_random_caption_style(seed=seed)
        if style == "card_bottom":
            return CAPTION_STYLES["dark_glass_badge"]
    # Default to randomized rich style on every generation (seeded by text/caller for run determinism)
    return get_random_caption_style(seed=seed)


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


def _format_text(text: str, casing: str) -> str:
    cleaned = text.strip()
    if casing == "upper":
        return cleaned.upper()
    if casing == "title":
        return cleaned.title()
    return cleaned


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


def _build_caption_overlay(
    size: tuple[int, int],
    text: str,
    style: CaptionStyle | str | None = None,
    bottom_limit: int | None = None,
) -> tuple[Image.Image, CaptionBox]:
    """Builds a dynamic caption overlay adhering to the chosen CaptionStyle.

    bottom_limit, if given, overrides where a position="bottom" style's
    card BOTTOM edge sits (default: FRAME_HEIGHT - SAFE_BOTTOM) — lets a
    caller stack a second bottom-anchored element (e.g. the Subscribe CTA)
    directly above a first one (e.g. the caution badge) instead of both
    independently anchoring to the same spot and overlapping. Real bug
    found 2026-08-29: a real yellow-safety-class video's last frame showed
    the caution badge and Subscribe CTA overlapping — both used
    position="bottom" with no awareness of each other."""
    st = resolve_caption_style(style, seed=text)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = load_font_by_family(family=st.font_family, size=st.font_size)

    max_text_width = FRAME_WIDTH - 2 * SAFE_SIDES - 2 * PADDING
    formatted = _format_text(text, st.casing)
    lines = _wrap_text(formatted, font, max_text_width, draw)

    line_heights = []
    line_widths = []
    for line in lines:
        stroke_w = st.stroke_width
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
        line_widths.append(bbox[2] - bbox[0])
        # Use font.getbbox or textbbox to accurately measure line ascender + descender height
        line_heights.append(bbox[3] - bbox[1])

    text_block_width = min(max(line_widths) if line_widths else 0, max_text_width)
    max_safe_height = (FRAME_HEIGHT - SAFE_BOTTOM - SAFE_TOP) - 2 * PADDING
    text_block_height = min(sum(line_heights) + LINE_SPACING * (len(lines) - 1 if lines else 0), max_safe_height)

    # Allow a small padding buffer for ascenders/descenders/strokes so they stay strictly inside card
    effective_padding = max(PADDING, st.stroke_width + 4)
    card_width = min(text_block_width + 2 * effective_padding, FRAME_WIDTH - 2 * SAFE_SIDES)
    card_height = min(text_block_height + 2 * effective_padding, FRAME_HEIGHT - SAFE_BOTTOM - SAFE_TOP)

    # Determine vertical placement based on style.position
    bottom_edge = bottom_limit if bottom_limit is not None else (FRAME_HEIGHT - SAFE_BOTTOM)
    if st.position == "top":
        card_top = SAFE_TOP + 15
    elif st.position == "middle":
        card_top = (FRAME_HEIGHT - card_height) // 2
    else:  # bottom
        card_top = bottom_edge - card_height

    # Shift card vertically fully inside safe area without shrinking
    max_top = bottom_edge - card_height
    if card_top > max_top:
        card_top = max_top
    if card_top < SAFE_TOP:
        card_top = SAFE_TOP
    card_bottom = card_top + card_height

    # Center horizontally and shift fully inside safe area
    card_left = (FRAME_WIDTH - card_width) // 2
    max_left = (FRAME_WIDTH - SAFE_SIDES) - card_width
    if card_left > max_left:
        card_left = max_left
    if card_left < SAFE_SIDES:
        card_left = SAFE_SIDES
    card_right = card_left + card_width

    # 1. Draw background card/pill if configured
    if st.bg_color is not None:
        draw.rounded_rectangle(
            [card_left, card_top, card_right, card_bottom],
            radius=st.bg_radius,
            fill=st.bg_color,
        )

    # 2. Draw text lines with strokes, shadows, or dual-tone keywords
    final_card_width = card_right - card_left
    y = card_top + effective_padding
    for line_idx, (line, lh) in enumerate(zip(lines, line_heights)):
        lw = draw.textlength(line, font=font)
        x = card_left + (final_card_width - lw) / 2

        # Optional drop shadow for stroke text without background card
        if st.shadow and st.bg_color is None:
            sx = x + st.shadow_offset[0]
            sy = y + st.shadow_offset[1]
            draw.text(
                (sx, sy),
                line,
                font=font,
                fill=st.shadow_color,
                stroke_width=st.stroke_width,
                stroke_fill=st.shadow_color,
            )

        if st.dual_tone and st.accent_color and line_idx == 0 and " " in line:
            # First word gets accent color, remaining words get base text color
            words = line.split(" ", 1)
            first_word = words[0]
            rest_of_line = " " + words[1]

            w_first = draw.textlength(first_word, font=font)

            draw.text(
                (x, y),
                first_word,
                font=font,
                fill=st.accent_color,
                stroke_width=st.stroke_width,
                stroke_fill=st.stroke_color,
            )
            draw.text(
                (x + w_first, y),
                rest_of_line,
                font=font,
                fill=st.text_color,
                stroke_width=st.stroke_width,
                stroke_fill=st.stroke_color,
            )
        else:
            draw.text(
                (x, y),
                line,
                font=font,
                fill=st.text_color,
                stroke_width=st.stroke_width,
                stroke_fill=st.stroke_color if st.stroke_width > 0 else None,
            )

        y += lh + LINE_SPACING

    box = CaptionBox(left=card_left, top=card_top, right=card_right, bottom=card_bottom)
    return overlay, box


def draw_caption(
    base: Image.Image,
    text: str,
    style: CaptionStyle | str | None = None,
) -> tuple[Image.Image, CaptionBox]:
    """Returns a new RGB image with the caption composited, plus the exact
    box it was drawn in (for the safe-margin assertion)."""
    img = base.convert("RGBA")
    overlay, box = _build_caption_overlay(img.size, text, style=style)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return composited, box


def caption_overlay_png(
    text: str,
    style: CaptionStyle | str | None = None,
) -> tuple[Image.Image, CaptionBox]:
    """Same caption card, as a standalone transparent RGBA image — for
    compositing onto an animated video clip via ffmpeg's overlay filter."""
    return _build_caption_overlay((FRAME_WIDTH, FRAME_HEIGHT), text, style=style)


CAUTION_BADGE_STYLE = CaptionStyle(
    name="caution_badge",
    font_family="modern_clean",
    font_size=51,
    text_color=(255, 255, 255, 255),
    stroke_color=(0, 0, 0, 255),
    stroke_width=5,
    casing="original",
    position="bottom",
    shadow=True,
)


def draw_caution_badge(base: Image.Image, text: str) -> tuple[Image.Image, CaptionBox]:
    """Composites a small safety-disclaimer line at the BOTTOM of the frame,
    ON TOP OF whatever else is already drawn (the scene's own real caption
    included) — this must never replace a scene's actual caption. Small,
    stroke-only (no background card, same as every other style), positioned
    opposite the main "middle" caption so the two never collide. Returns
    the badge's own box too, so a caller stacking the Subscribe CTA above
    it (see subscribe_cta_overlay_png's bottom_limit) knows where it landed."""
    img = base.convert("RGBA")
    overlay, box = _build_caption_overlay(img.size, text, style=CAUTION_BADGE_STYLE)
    return Image.alpha_composite(img, overlay).convert("RGB"), box


def caution_badge_overlay_png(text: str) -> tuple[Image.Image, CaptionBox]:
    """Same as draw_caution_badge, as a standalone transparent RGBA layer —
    for compositing onto an animated video clip via ffmpeg's overlay filter."""
    return _build_caption_overlay((FRAME_WIDTH, FRAME_HEIGHT), text, style=CAUTION_BADGE_STYLE)


SUBSCRIBE_CTA_STYLE = CaptionStyle(
    name="subscribe_cta",
    font_family="impact",
    font_size=87,
    text_color=(255, 45, 45, 255),   # bright red, standard "subscribe" urgency color
    stroke_color=(255, 255, 255, 255),
    stroke_width=8,
    casing="upper",
    position="bottom",
    shadow=True,
)

# Vertical gap between the Subscribe CTA and whatever bottom-anchored
# element it's stacked above (currently only the caution badge).
SUBSCRIBE_CTA_STACK_GAP = 14


def subscribe_cta_overlay_png(text: str = "SUBSCRIBE!", bottom_limit: int | None = None) -> Image.Image:
    """A bold end-of-video Subscribe call-to-action, as a standalone
    transparent RGBA layer — same compositing pattern as
    caution_badge_overlay_png(), meant for the LAST scene's final seconds
    only (see assemble()/assemble_stickers()/assemble_animated()'s
    subscribe_cta_text param).

    bottom_limit, if given, stacks the CTA above that y-coordinate instead
    of at the frame's own bottom-safe-area edge — pass the caution badge's
    own box.top (minus SUBSCRIBE_CTA_STACK_GAP) when both are present on
    the same frame, or they silently land in the same spot and overlap
    (real bug, confirmed 2026-08-29 on a real yellow-safety-class video)."""
    overlay, _box = _build_caption_overlay(
        (FRAME_WIDTH, FRAME_HEIGHT), text, style=SUBSCRIBE_CTA_STYLE, bottom_limit=bottom_limit,
    )
    return overlay


def draw_subscribe_cta(base: Image.Image, text: str = "SUBSCRIBE!", bottom_limit: int | None = None) -> Image.Image:
    """Same role as draw_caution_badge() but for the Subscribe CTA — used by
    the static assemble() path (which draws directly onto a PIL frame,
    unlike the timed-overlay paths). See subscribe_cta_overlay_png's
    docstring for bottom_limit's stacking purpose."""
    img = base.convert("RGBA")
    overlay, _box = _build_caption_overlay(img.size, text, style=SUBSCRIBE_CTA_STYLE, bottom_limit=bottom_limit)
    return Image.alpha_composite(img, overlay).convert("RGB")
