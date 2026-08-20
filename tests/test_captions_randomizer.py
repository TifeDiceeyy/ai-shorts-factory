import pytest
from PIL import Image

from shorts_factory.captions import (
    CAPTION_STYLES,
    CaptionStyle,
    draw_caption,
    caption_overlay_png,
    get_random_caption_style,
    resolve_caption_style,
    SAFE_TOP,
    SAFE_BOTTOM,
    SAFE_SIDES,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


def test_caption_styles_catalog_is_rich_and_valid():
    assert len(CAPTION_STYLES) >= 10
    for name, st in CAPTION_STYLES.items():
        assert isinstance(st, CaptionStyle)
        assert st.name == name
        assert st.font_size >= 50
        assert len(st.text_color) == 4


def test_caption_style_randomizer_produces_varied_styles():
    styles = [get_random_caption_style() for _ in range(20)]
    names = {s.name for s in styles}
    font_families = {s.font_family for s in styles}
    colors = {s.text_color for s in styles}

    # Proves randomized variety across runs
    assert len(names) >= 4
    assert len(font_families) >= 2
    assert len(colors) >= 3


def test_all_caption_styles_stay_inside_safe_area():
    base = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    sample_texts = [
        "SHORT HOOK",
        "THIS IS A MEDIUM LENGTH EXPLAINER CAPTION",
        "POUR THICK SLURRY DIRECTLY INTO CLAMPED WOODEN CASTING MOLD WITH REBAR",
    ]

    for style_name, style_obj in CAPTION_STYLES.items():
        for text in sample_texts:
            comp, box = draw_caption(base, text, style=style_obj)
            assert box.inside_safe_area() is True
            assert box.top >= SAFE_TOP
            assert box.bottom <= FRAME_HEIGHT - SAFE_BOTTOM
            assert box.left >= SAFE_SIDES
            assert box.right <= FRAME_WIDTH - SAFE_SIDES


def test_dual_tone_and_highlighter_pill_rendering():
    base = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))

    # Dual-tone style
    comp_dual, box_dual = draw_caption(base, "REINVENT ROMAN CONCRETE", style="dual_tone_fire")
    assert box_dual.inside_safe_area() is True

    # Highlighter yellow pill style
    comp_pill, box_pill = draw_caption(base, "STEP ONE: COOK LIMESTONE", style="highlighter_yellow_pill")
    assert box_pill.inside_safe_area() is True


def test_caption_overlay_png_returns_rgba_and_valid_box():
    overlay, box = caption_overlay_png("ALIVE AND HEALING", style="cyber_cyan_ice")
    assert overlay.mode == "RGBA"
    assert overlay.size == (FRAME_WIDTH, FRAME_HEIGHT)
    assert box.inside_safe_area() is True


def test_long_captions_at_top_middle_bottom_stay_inside_box_and_safe_margins():
    long_text = "THIS IS A VERY LONG SCRIPT CAPTION DESIGNED TO WRAP INTO MULTIPLE LINES AND TEST BOUNDING BOX SHIFTING ACCURACY"
    for pos in ("top", "middle", "bottom"):
        custom_style = CaptionStyle(
            name=f"test_{pos}",
            font_family="impact",
            font_size=64,
            text_color=(255, 255, 255, 255),
            stroke_color=(0, 0, 0, 255),
            stroke_width=6,
            bg_color=(0, 0, 0, 180),
            bg_radius=18,
            casing="upper",
            position=pos,
        )
        overlay, box = caption_overlay_png(long_text, style=custom_style)
        assert box.inside_safe_area() is True
        assert box.top >= SAFE_TOP
        assert box.bottom <= FRAME_HEIGHT - SAFE_BOTTOM
        assert box.left >= SAFE_SIDES
        assert box.right <= FRAME_WIDTH - SAFE_SIDES

        # Verify all rendered pixels (alpha > 0) are strictly inside the safe margins and bounded by box
        bbox = overlay.getbbox()
        assert bbox is not None
        render_left, render_top, render_right, render_bottom = bbox

        assert render_left >= SAFE_SIDES
        assert render_right <= FRAME_WIDTH - SAFE_SIDES
        assert render_top >= SAFE_TOP
        assert render_bottom <= FRAME_HEIGHT - SAFE_BOTTOM + 1

        assert render_left >= box.left - 2
        assert render_right <= box.right + 2
        assert render_top >= box.top - 2
        assert render_bottom <= box.bottom + 2


