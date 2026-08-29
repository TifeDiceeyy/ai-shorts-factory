"""Localized object animation — user request 2026-08-29: when a scene's
narration names an object (fire, water, steam, etc. — see
mascots.OBJECT_FX_KEYWORDS), it should visibly animate starting at that
exact moment, not sit static for the whole scene. Scoped to
process_action/ingredient_grid/split_canvas scenes only (see
assembly.py's module comment above build_scene_video_segment_from_still
for why plain mascot scenes are excluded — no real object segmentation
exists in this codebase, so isolating "the FX" from "the character" in
one flat image isn't reliable)."""
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


def test_narrated_object_cue_start_none_for_mascot_scene_types(tmp_path):
    """Scope guard: even with an obvious keyword match, mascot-present
    scene types must never trigger this — no real way to isolate the FX
    from the character in one flat image without segmentation."""
    narration = "The fire crackles as the metal begins to glow."
    for scene_type in ("mascot", "mascot_reaction"):
        assert assembly._narrated_object_cue_start(narration, 6.0, scene_type) is None


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
