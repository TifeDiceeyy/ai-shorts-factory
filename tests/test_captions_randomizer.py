import pytest
from PIL import Image

from shorts_factory.captions import (
    CAPTION_STYLES,
    CaptionStyle,
    STYLE_NAMES,
    SUBSCRIBE_CTA_STACK_GAP,
    draw_caption,
    caption_overlay_png,
    caution_badge_overlay_png,
    draw_caution_badge,
    draw_subscribe_cta,
    subscribe_cta_overlay_png,
    get_random_caption_style,
    get_random_caption_style_name,
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


def test_no_caption_style_has_a_background_card():
    """Regression test: 4 catalog styles (highlighter_yellow_pill,
    dark_glass_badge, royal_blue_pill, danger_red_badge) used to draw a
    solid/translucent card behind the text. User feedback: captions should
    never have a background — converted to stroke-only text (same color
    identity, no bg_color) so every style in the catalog renders the same
    way now: bold outlined text directly over the footage."""
    for name, st in CAPTION_STYLES.items():
        assert st.bg_color is None, f"{name} still has a background card"


def test_every_caption_style_sits_in_the_top_headspace_not_over_the_character():
    """Regression test, reversed twice by direct user feedback: captions
    were originally "top", then changed to "middle" (overlay the mascot) per
    earlier feedback, then reverted back to "top" 2026-08-27 after the user
    provided a real reference short — captions there always sit in the empty
    space above the character/props, never across the torso, matching
    professional practice for this format."""
    for name, st in CAPTION_STYLES.items():
        assert st.position == "top", f"{name} is not positioned in the top headspace"


def test_caption_style_randomizer_produces_varied_styles():
    styles = [get_random_caption_style() for _ in range(20)]
    names = {s.name for s in styles}
    font_families = {s.font_family for s in styles}
    colors = {s.text_color for s in styles}

    # Proves randomized variety across runs
    assert len(names) >= 4
    assert len(font_families) >= 2
    assert len(colors) >= 3


def test_get_random_caption_style_name_returns_a_real_catalog_key():
    """get_random_caption_style_name() must return a plain string key
    (unlike get_random_caption_style(), which returns a jittered
    CaptionStyle instance) — this is what gets persisted as script.json's
    caption_style field and re-resolved later via resolve_caption_style()."""
    for _ in range(20):
        name = get_random_caption_style_name()
        assert isinstance(name, str)
        assert name in STYLE_NAMES
        # Must resolve to the EXACT unmodified catalog style, not a jittered
        # variant — a persisted name has to look the same every time it's
        # re-resolved (e.g. once per scene across a whole video).
        assert resolve_caption_style(name) is CAPTION_STYLES[name]


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




def test_draw_caution_badge_adds_to_the_image_not_replaces_it():
    """Regression test: the caution badge must be an ADDITIONAL element
    composited on top of the existing frame — it must never be used as a
    substitute for a scene's real caption (that was the actual bug:
    pipeline.py used to overwrite scenes[-1]["caption"] with a fixed
    warning string, silently deleting every yellow topic's real payoff
    line)."""
    base = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    real_caption_composited, box = draw_caption(base, "THE REAL PAYOFF LINE", style="comic_punch_orange")

    badged, _box = draw_caution_badge(real_caption_composited, "CAUTION: educational overview only.")

    # The real caption's own rendered pixels must still be present — the
    # badge must not have overwritten/erased the frame's existing content,
    # just added to it. Sample within the real caption's own returned box
    # (position-independent — the badge sits at the bottom regardless of
    # where the main caption is configured to render): pixel data must
    # differ from the plain white background, proving the caption text is
    # still there underneath.
    sample_y = (box.top + box.bottom) // 2
    diff_found = any(
        badged.getpixel((x, sample_y)) != (255, 255, 255)
        for x in range(0, FRAME_WIDTH, 20)
    )
    assert diff_found, "the real caption's content is gone after adding the caution badge"


def test_caution_badge_sits_at_the_bottom_not_over_the_main_middle_caption():
    """The badge must be positioned opposite the main caption (bottom vs
    middle) so the two never visually collide."""
    overlay, _box = caution_badge_overlay_png("CAUTION: educational overview only.")
    bbox = overlay.getbbox()
    assert bbox is not None
    _left, top, _right, bottom = bbox
    # Must land in the bottom safe region, well below the vertical middle
    # of the frame where the main caption is drawn.
    assert top > FRAME_HEIGHT // 2
    assert bottom <= FRAME_HEIGHT - SAFE_BOTTOM + 1


def test_subscribe_cta_stacks_above_the_caution_badge_without_overlapping():
    """Real bug found 2026-08-29 on a real yellow-safety-class video
    ("furnace"): the caution badge and Subscribe CTA both independently
    used position="bottom", so on the last scene's final cue (where both
    apply) they landed in the exact same spot and visually overlapped —
    the caution text was even partly obscured/cut off. Fixed via
    bottom_limit: the CTA must now stack entirely above the caution
    badge's own top edge, with a real gap between them, not just avoid
    exact pixel collision."""
    text = "CAUTION: educational overview only."
    _caution_overlay, caution_box = caution_badge_overlay_png(text)

    bottom_limit = caution_box.top - SUBSCRIBE_CTA_STACK_GAP
    cta_overlay = subscribe_cta_overlay_png("SUBSCRIBE!", bottom_limit=bottom_limit)
    cta_bbox = cta_overlay.getbbox()
    assert cta_bbox is not None
    _left, _top, _right, cta_bottom = cta_bbox

    assert cta_bottom <= caution_box.top, (
        f"Subscribe CTA (bottom={cta_bottom}) must not extend past the caution badge's top "
        f"({caution_box.top}) — they would visually overlap"
    )
    assert caution_box.top - cta_bottom >= SUBSCRIBE_CTA_STACK_GAP - 2, (
        "expected a real visual gap between the two, not just a non-overlapping touch"
    )


def test_draw_subscribe_cta_without_bottom_limit_still_lands_at_the_frame_bottom():
    """No caution badge on this frame (e.g. a green-safety-class topic) —
    the CTA must fall back to its normal bottom-of-frame position, same as
    before this fix, not silently vanish or misplace."""
    overlay = subscribe_cta_overlay_png("SUBSCRIBE!")
    bbox = overlay.getbbox()
    assert bbox is not None
    _left, top, _right, bottom = bbox
    assert top > FRAME_HEIGHT // 2
    assert bottom <= FRAME_HEIGHT - SAFE_BOTTOM + 1
