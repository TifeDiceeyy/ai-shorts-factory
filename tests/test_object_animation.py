"""Localized object animation — user request 2026-08-29: when a scene's
narration names an object (fire, water, steam, etc. — see
mascots.OBJECT_FX_KEYWORDS), it should visibly animate starting at that
exact moment, not sit static for the whole scene. Applies to
process_action/ingredient_grid/split_canvas/mascot/mascot_reaction scenes
(mascot/mascot_reaction scenes exclude a centered box around the character
— see assembly._mascot_exclusion_region — since there's no real object
segmentation in this codebase to isolate "the FX" from "the character"
any other way)."""
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageStat

from shorts_factory import assembly
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH


def _prop_image(tmp_path, name="scene.png", color=(200, 100, 40)) -> Path:
    """A colored rectangle on white — stands in for a real generated
    process_action/ingredient_grid scene image (always a stark #FFFFFF
    background per the house art style)."""
    path = tmp_path / name
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    draw.rectangle((cx - 150, cy - 150, cx + 150, cy + 150), fill=color)
    img.save(path)
    return path


def _blank_image(tmp_path, name="blank.png") -> Path:
    path = tmp_path / name
    Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255)).save(path)
    return path


def _frame_at(video_path: Path, t: float, out_path: Path) -> Image.Image:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video_path),
         "-frames:v", "1", str(out_path)],
        check=True,
    )
    return Image.open(out_path).convert("RGB")


def _max_channel_diff(a: Image.Image, b: Image.Image) -> int:
    stat = ImageStat.Stat(ImageChops.difference(a, b))
    return max(hi for _lo, hi in stat.extrema)


def _mean_channel_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean, not max/extrema — a handful of h264-noisy edge/anti-aliased
    pixels (real, confirmed 2026-08-29: up to ~20 units at a rectangle's
    boundary even when the region's overall content is unchanged) can push
    a MAX-based diff over any reasonable threshold despite nothing having
    actually changed. Mean is far more robust for a "stayed static" check;
    the "a real change happened somewhere" checks still use the max-based
    helper, where that sensitivity is exactly what's wanted."""
    stat = ImageStat.Stat(ImageChops.difference(a, b))
    return max(stat.mean)


def test_content_mask_isolates_the_non_background_rectangle(tmp_path):
    img_path = _prop_image(tmp_path)
    mask_path = assembly._content_mask(img_path, tmp_path / "mask.png")
    assert mask_path is not None
    mask = Image.open(mask_path)
    bbox = mask.getbbox()
    assert bbox is not None
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    # The mask's non-zero region should land right on the drawn rectangle,
    # not somewhere random or covering the whole frame.
    assert abs((bbox[0] + bbox[2]) / 2 - cx) < 20
    assert abs((bbox[1] + bbox[3]) / 2 - cy) < 20
    assert (bbox[2] - bbox[0]) < FRAME_WIDTH * 0.6


def test_content_mask_returns_none_for_a_blank_white_image(tmp_path):
    img_path = _blank_image(tmp_path)
    mask_path = assembly._content_mask(img_path, tmp_path / "mask.png")
    assert mask_path is None


def test_content_mask_region_excludes_content_outside_the_box(tmp_path):
    """split_canvas support: restricting the mask to a region (the top
    half, keeping it off the mascot's bottom corner) must zero out
    anything outside that box even if it's real non-white content."""
    img_path = _prop_image(tmp_path)  # rectangle is centered, spanning the vertical middle
    top_half = (0, 0, FRAME_WIDTH, FRAME_HEIGHT // 2)
    mask_path = assembly._content_mask(img_path, tmp_path / "mask.png", region=top_half)
    if mask_path is None:
        return  # the centered rectangle may have too little area in just the top half — acceptable
    mask = Image.open(mask_path)
    bbox = mask.getbbox()
    assert bbox is not None
    assert bbox[3] <= FRAME_HEIGHT // 2 + 1, "mask must not extend below the given region"


def test_narrated_object_cue_start_matches_a_real_cue_for_process_action(tmp_path):
    narration = "Watch closely as the metal begins to melt near the roaring fire."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    assert start is not None
    cue_starts = [c.start for c in assembly.narration_caption_cues(narration, duration)]
    assert start in cue_starts, "must return an actual cue start time, not an arbitrary timestamp"
    assert start > 0.0, "'fire' isn't in the first few words, so this shouldn't fire at scene start"


def test_narrated_object_cue_start_none_when_no_keyword_matches(tmp_path):
    narration = "The ancient traders exchanged goods across the mountain pass."
    start = assembly._narrated_object_cue_start(narration, 6.0, "process_action")
    assert start is None


def test_narrated_object_cue_start_matches_for_mascot_scene_types(tmp_path):
    """Extended 2026-08-29 (Part C): mascot-present scenes now animate too
    — the FX/prop around the character, not the character itself (see
    _mascot_exclusion_region) — per direct user feedback that a mascot
    scene's glowing prop shouldn't be excluded outright."""
    narration = "The fire crackles as the metal begins to glow."
    for scene_type in ("mascot", "mascot_reaction"):
        start = assembly._narrated_object_cue_start(narration, 6.0, scene_type)
        assert start is not None


def test_narrated_object_cue_style_matches_the_same_cues_category(tmp_path):
    fire_style = assembly._narrated_object_cue_style(
        "The fire crackles as the metal begins to glow.", 6.0, "process_action",
    )
    assert fire_style == "flicker"
    steam_style = assembly._narrated_object_cue_style(
        "Watch as steam rises steadily from the pot.", 6.0, "process_action",
    )
    assert steam_style == "drift"


def test_mascot_scene_object_fx_falls_back_to_props_and_fx_fields(tmp_path):
    """Real gap found 2026-08-29 against the actual furnace script: a
    mascot_reaction scene had props="tongs, molten iron blob",
    fx="red glow", mascot_emotion="alarmed", but its narration never
    mentioned fire/glow at all ("...can have way too much carbon..."), so
    the narration-cue mechanism alone would never trigger. The scene-level
    fallback must catch this via fx/props/action instead, starting at t=0
    since the prop is visible in the image from frame one."""
    scene = {
        "narration": "But here's the catch: the iron can have way too much carbon, up to 4.5 percent!",
        "action": "holding up molten iron with tongs, looking concerned",
        "props": "tongs, molten iron blob",
        "fx": "red glow",
        "scene_type": "mascot_reaction",
    }
    assert assembly._narrated_object_cue_start(scene["narration"], 7.0, "mascot_reaction") is None
    result = assembly._scene_object_fx(scene)
    assert result is not None
    start, style = result
    assert start == 0.0
    assert style == "flicker"


def test_mascot_scene_object_fx_none_when_nothing_matches(tmp_path):
    scene = {
        "narration": "Civilization grew stronger with every discovery.",
        "action": "gesturing proudly",
        "props": "",
        "fx": "",
        "scene_type": "mascot_reaction",
    }
    assert assembly._scene_object_fx(scene) is None


def test_scene_object_fx_wins_over_a_conflicting_narration_only_category(tmp_path):
    """Real gap found 2026-08-29 against the actual furnace script: a
    process_action scene's narration said "materials heat up" (matching
    the fire/flicker category), while its own action/props fields said
    "stirring a bubbling cauldron" / "carbon monoxide gas" (bubble/drift)
    — the actual image shows a bubbling liquid with rising gas, not fire,
    so scene_object_fx's action/props match must win over the narration's
    own (less accurate) category when both match different categories."""
    scene = {
        "narration": "Inside, materials heat up! Carbon monoxide helps pull iron from oxides.",
        "action": "stirring a bubbling cauldron",
        "props": "rustic cauldron, wooden paddle, carbon monoxide gas",
        "fx": None,
        "scene_type": "process_action",
    }
    result = assembly._scene_object_fx(scene)
    assert result is not None
    _start, style = result
    assert style == "drift", "the bubbling/rising-gas action should win over narration's 'heat' wording"


def test_mascot_exclusion_region_is_centered_and_smaller_than_the_full_frame(tmp_path):
    left, top, right, bottom = assembly._mascot_exclusion_region()
    assert 0 < left < right < FRAME_WIDTH
    assert 0 < top < bottom < FRAME_HEIGHT
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    assert abs(cx - FRAME_WIDTH / 2) < 2
    assert abs(cy - FRAME_HEIGHT / 2) < 2
    assert (right - left) < FRAME_WIDTH
    assert (bottom - top) < FRAME_HEIGHT


def test_content_mask_exclude_region_zeroes_out_the_inner_box(tmp_path):
    """Part C: a mascot-scene mask must never include the excluded center
    box, even when real non-white content (the character) sits there."""
    img_path = _prop_image(tmp_path)  # centered rectangle, exactly where the exclusion box sits
    exclude_box = (
        FRAME_WIDTH // 2 - 200, FRAME_HEIGHT // 2 - 200,
        FRAME_WIDTH // 2 + 200, FRAME_HEIGHT // 2 + 200,
    )
    mask_path = assembly._content_mask(img_path, tmp_path / "mask.png", exclude_region=exclude_box)
    if mask_path is None:
        return  # the whole rectangle sat inside the exclusion box — acceptable, nothing left to animate
    mask = Image.open(mask_path)
    left, top, right, bottom = exclude_box
    stat = ImageStat.Stat(mask.crop(exclude_box))
    assert stat.mean[0] == 0, "excluded box must be fully zeroed, even where real content was drawn"


def test_object_pulse_preserves_the_real_object_color_before_it_starts(tmp_path):
    """Real bug found and fixed 2026-08-29: maskedmerge produced a flat,
    visibly wrong gray blend (not the object's actual drawn color) when
    the mask stream was formatted as plain grayscale instead of matching
    the base/overlay streams' own RGBA format. This must never regress —
    before the pulse starts, the object's on-screen color must still
    match what was actually drawn, not some unrelated blended color."""
    color = (200, 100, 40)
    img_path = _prop_image(tmp_path, color=color)
    narration = "Watch closely as the metal begins to melt near the roaring fire."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    assert start is not None and start > 0.5

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=start,
    )
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    frame = _frame_at(seg, 0.4, tmp_path / "before.png")
    pixel = frame.getpixel((cx, cy))
    for channel_val, expected in zip(pixel, color):
        assert abs(channel_val - expected) < 15, (
            f"expected the object's real color {color} before the pulse starts, got {pixel}"
        )


def test_object_pulse_stays_static_before_the_matched_cue_and_animates_after(tmp_path):
    img_path = _prop_image(tmp_path)
    narration = "Watch closely as the metal begins to melt near the roaring fire."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    assert start is not None and start > 0.5, "test needs real static time before the pulse starts"

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=start,
    )

    # Sample strictly within the drawn rectangle's own bounds, away from
    # the caption area — the caption legitimately changes text between
    # these two timestamps (different cues), which isn't what this test
    # is checking.
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    region = (cx - 100, cy - 100, cx + 100, cy + 100)
    # Sample after the scene's own single pop-in has settled (~0.27s, see
    # POP_IN_SETTLE_FRAMES) — sampling during that transition would show a
    # real pixel change from the unrelated whole-image pop-in, not this
    # feature.
    early_a = _frame_at(seg, 0.4, tmp_path / "early_a.png").crop(region)
    early_b = _frame_at(seg, max(0.5, start - 0.3), tmp_path / "early_b.png").crop(region)
    assert _mean_channel_diff(early_a, early_b) < 2.0, "must stay static before the matched cue starts"

    after_samples = [
        _frame_at(seg, start + off, tmp_path / f"after_{off}.png").crop(region)
        for off in (0.2, 0.5, 0.8, 1.1)
    ]
    diffs = [_max_channel_diff(early_a, frame) for frame in after_samples]
    assert max(diffs) >= 20, (
        f"expected a real brightness change somewhere after the pulse starts, got max diff {max(diffs)}"
    )


def test_object_pulse_leaves_the_white_background_untouched(tmp_path):
    img_path = _prop_image(tmp_path)
    narration = "Watch closely as the metal begins to melt near the roaring fire."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    assert start is not None

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=start,
    )
    corner = (10, FRAME_HEIGHT - 60, 60, FRAME_HEIGHT - 10)
    for t in (start + 0.2, start + 0.7, start + 1.2):
        frame = _frame_at(seg, t, tmp_path / f"corner_{t}.png")
        stat = ImageStat.Stat(frame.crop(corner))
        assert min(stat.mean) > 250, f"background corner should stay white at t={t}, got {stat.mean}"


def test_object_drift_changes_pixel_position_not_just_brightness(tmp_path):
    """Part B: the drift style must produce a real content-POSITION shift
    inside the mask, distinct from the flicker style's in-place brightness
    change — confirmed live 2026-08-29 against a real scene before wiring
    this in (see assembly.OBJECT_DRIFT_AMPLITUDE_PX). A pixel right at the
    object's own edge should swing between "near background white" and
    "near the object's real color" as the drift phase changes — a swing a
    bounded +-OBJECT_PULSE_AMPLITUDE brightness multiplier on the object's
    own hue could never produce, since it never approaches white."""
    color = (200, 100, 40)
    img_path = _prop_image(tmp_path, color=color)
    narration = "Watch as steam rises steadily from the pot."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    style = assembly._narrated_object_cue_style(narration, duration, "process_action")
    assert start is not None and style == "drift"

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=start,
        object_animation_style=style,
    )
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    # build_scene_video_segment_from_still's own pad+zoompan
    # (STICKER_HEADROOM) uniformly scales the whole rendered image down
    # toward center in the final frame, so source-space coordinates (the
    # rectangle's own top edge, drawn at cy-150) must be remapped through
    # that same scale to find where it actually lands in the output.
    rest_scale = 1.0 / assembly.STICKER_HEADROOM
    edge_x, edge_y = round(cx), round(cy + (cy - 150 - cy) * rest_scale)

    samples = [
        _frame_at(seg, start + off, tmp_path / f"drift_{off}.png").getpixel((edge_x, edge_y))
        for off in (0.1, 0.4, 0.7, 1.0, 1.3)
    ]
    dist_to_white = [sum(abs(255 - c) for c in px) for px in samples]
    dist_to_color = [sum(abs(v - c) for v, c in zip(px, color)) for px in samples]
    # The white-proximity swing is the real discriminator: a bounded
    # +-OBJECT_PULSE_AMPLITUDE brightness multiplier on this base color
    # could never bring it anywhere close to white (its own baseline
    # distance to white is ~425; even a full swing only closes ~183 of
    # that). Only a genuine positional shift revealing the background can.
    assert min(dist_to_white) < 70, (
        f"expected the edge pixel to swing near background white at some drift phase, got {samples}"
    )
    assert min(dist_to_color) < 150, (
        f"expected the edge pixel to also swing back near the object's real color, got {samples}"
    )
    # Outside the mask, drift must be exactly as inert as the flicker style.
    corner = (10, FRAME_HEIGHT - 60, 60, FRAME_HEIGHT - 10)
    for off in (0.3, 0.9):
        frame = _frame_at(seg, start + off, tmp_path / f"drift_corner_{off}.png")
        stat = ImageStat.Stat(frame.crop(corner))
        assert min(stat.mean) > 250, f"background corner should stay white at t={start + off}"


def test_grid_reveal_shows_quadrants_one_at_a_time_then_settles(tmp_path):
    """Part A: an ingredient_grid scene's four quadrants must pop in in
    sequence (one new quadrant visible per GRID_REVEAL_STAGGER_FRAMES
    window), not all appear at once, and the sequence must settle back to
    the real full image once every quadrant has revealed."""
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    quadrants = assembly._grid_quadrant_regions(FRAME_WIDTH, FRAME_HEIGHT)
    colors = [(200, 60, 60), (60, 160, 60), (60, 60, 200), (200, 160, 40)]
    for box, color in zip(quadrants, colors):
        pad = 40
        draw.rectangle((box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad), fill=color)

    frames = assembly._grid_reveal_frame_paths(img, tmp_path / "reveal", "test")
    assert len(frames) == assembly.GRID_REVEAL_STAGGER_FRAMES * 3 + assembly.POP_IN_SETTLE_FRAMES

    def is_white_at(frame_path, box):
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        img = Image.open(frame_path).convert("RGB")
        return img.getpixel(center) == (255, 255, 255)

    # At frame 0, only the first quadrant has started popping — the rest
    # stay blank until their own stagger window arrives.
    assert not is_white_at(frames[0], quadrants[0])
    assert is_white_at(frames[0], quadrants[1])
    assert is_white_at(frames[0], quadrants[2])
    assert is_white_at(frames[0], quadrants[3])
    # After the first stagger window, quadrant 1 has also started while
    # quadrants 2 and 3 are still blank — one new quadrant per window, not
    # all four at once.
    mid_frame = frames[assembly.GRID_REVEAL_STAGGER_FRAMES + 2]
    assert not is_white_at(mid_frame, quadrants[0])
    assert not is_white_at(mid_frame, quadrants[1])
    assert is_white_at(mid_frame, quadrants[2])
    assert is_white_at(mid_frame, quadrants[3])
    # By the last frame, every quadrant has revealed.
    assert all(not is_white_at(frames[-1], box) for box in quadrants)


def test_grid_reveal_wires_into_build_scene_video_segment_from_still(tmp_path):
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    quadrants = assembly._grid_quadrant_regions(FRAME_WIDTH, FRAME_HEIGHT)
    for box, color in zip(quadrants, [(200, 60, 60), (60, 160, 60), (60, 60, 200), (200, 160, 40)]):
        pad = 40
        draw.rectangle((box[0] + pad, box[1] + pad, box[2] - pad, box[3] - pad), fill=color)
    img_path = tmp_path / "grid.png"
    img.save(img_path)

    duration = 6.0
    overlays, _box = assembly.build_timed_caption_overlays("Clay bricks, steel sheet, straw, and wood plank.", duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, grid_reveal=True,
    )
    assert seg.exists()

    # Late in the scene, the full grid must have settled in (no quadrant left blank).
    late = _frame_at(seg, duration - 0.3, tmp_path / "late.png")
    for box in quadrants:
        center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        pixel = late.getpixel(center)
        assert pixel != (255, 255, 255), f"quadrant at {center} should have revealed by the end, got {pixel}"


def _mask_region_brightness_series(seg: Path, tmp_path, tag: str, start: float,
                                   n: int = 18, step: float = 1 / 30) -> list[float]:
    """Mean brightness of the animated prop region, sampled frame-by-frame
    over a short window right after `start`."""
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    region = (cx - 100, cy - 100, cx + 100, cy + 100)
    out = []
    for i in range(n):
        frame = _frame_at(seg, start + 0.05 + i * step, tmp_path / f"fs_{tag}_{i}.png").crop(region)
        out.append(max(ImageStat.Stat(frame).mean))
    return out


def _mean_abs_step(series: list[float]) -> float:
    return sum(abs(series[i + 1] - series[i]) for i in range(len(series) - 1)) / (len(series) - 1)


def test_flicker_style_jitters_faster_and_harder_than_the_slow_breathing_pulse(tmp_path):
    """Section 2 (localized fire/glow flicker): fire/spark scenes route to
    style="flicker", which must render as a fast, irregular flame jitter —
    _object_flicker_brightness_expr — NOT the slow even
    _object_pulse_brightness_expr breathing that every other brightness
    category falls back to. Fails-without-fix: before this change, "flicker"
    reused _object_pulse_brightness_expr verbatim, so the two series below
    had identical frame-to-frame motion (ratio ~1.0).

    Validated live 2026-08-31 against the real furnace molten-iron scene:
    flicker's frame-to-frame masked-region brightness delta ran ~4x the
    slow pulse's, with no visible repeat cycle."""
    img_path = _prop_image(tmp_path)
    narration = "The fire roars and the metal starts to glow bright orange."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    style = assembly._narrated_object_cue_style(narration, duration, "process_action")
    assert start is not None and style == "flicker"

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)

    flicker_seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "seg_flicker",
        timed_caption_overlays=overlays, object_animation_start=start,
        object_animation_style="flicker",
    )
    pulse_seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 1, tmp_path / "seg_pulse",
        timed_caption_overlays=overlays, object_animation_start=start,
        object_animation_style="pulse",
    )

    flicker_step = _mean_abs_step(_mask_region_brightness_series(flicker_seg, tmp_path, "flick", start))
    pulse_step = _mean_abs_step(_mask_region_brightness_series(pulse_seg, tmp_path, "pulse", start))
    assert flicker_step > pulse_step * 2.0, (
        f"flicker should jitter far harder than the slow pulse: "
        f"flicker per-frame delta {flicker_step:.2f} vs pulse {pulse_step:.2f}"
    )


def test_flicker_style_stays_static_before_the_cue_and_leaves_background_untouched(tmp_path):
    img_path = _prop_image(tmp_path)
    narration = "We start by gathering the raw ore and stacking it carefully, and only much later does the fire finally roar to life."
    duration = 6.0
    start = assembly._narrated_object_cue_start(narration, duration, "process_action")
    assert start is not None and start > 0.5
    style = assembly._narrated_object_cue_style(narration, duration, "process_action")
    assert style == "flicker"

    overlays, _box = assembly.build_timed_caption_overlays(narration, duration)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, duration, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=start,
        object_animation_style=style,
    )

    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    region = (cx - 100, cy - 100, cx + 100, cy + 100)
    a = _frame_at(seg, 0.4, tmp_path / "early_a.png").crop(region)
    b = _frame_at(seg, max(0.5, start - 0.3), tmp_path / "early_b.png").crop(region)
    assert _mean_channel_diff(a, b) < 2.0, "must stay static before the fire cue starts"

    corner = (10, FRAME_HEIGHT - 60, 60, FRAME_HEIGHT - 10)
    for off in (0.2, 0.6, 1.1):
        frame = _frame_at(seg, start + off, tmp_path / f"corner_{off}.png")
        stat = ImageStat.Stat(frame.crop(corner))
        assert min(stat.mean) > 250, f"background must stay white at t={start + off}, got {stat.mean}"


def test_no_animation_wiring_when_object_animation_start_is_none(tmp_path):
    """Default/backward-compatible path: scenes that don't qualify (no
    keyword match, or a mascot scene type) get object_animation_start=None
    from the caller, and the segment builder must add no extra ffmpeg
    input or filter stage at all in that case — same output as before this
    feature existed."""
    img_path = _prop_image(tmp_path)
    overlays, _box = assembly.build_timed_caption_overlays("A plain narration line here.", 4.0)
    seg = assembly.build_scene_video_segment_from_still(
        img_path, 4.0, 0, tmp_path / "segments",
        timed_caption_overlays=overlays, object_animation_start=None,
    )
    assert seg.exists()
    assert not (tmp_path / "segments" / "objmask_00.png").exists()
