"""Regression coverage for trimming a raw animated clip's leading freeze.
User feedback ("little pause between scenes it shouldn't be so") led to
measuring real, already-paid-for Kling clips from the electricity video:
several held the static source pose for well over half the clip before any
real motion began — that dead time landing right after a scene cut is what
reads as a pause, even though narration audio is already talking.
build_scene_video_segment_from_clip() now skips that leading freeze via
_leading_freeze_seconds(), so the visible clip gets to real motion
immediately; tpad's existing end-of-clip hold absorbs the same amount of
frozen time at the scene's END instead."""
import subprocess

import pytest
from PIL import Image

from shorts_factory import assembly, captions
from shorts_factory.media_probe import probe_duration


def _frozen_then_moving_clip(path, freeze_seconds=3.0, motion_seconds=2.0):
    """A real (not faked) clip: a static color for freeze_seconds, then
    genuine per-frame random noise for motion_seconds — a faithful stand-in
    for a Kling clip that holds the source pose before animating, with a
    sharp, unambiguous boundary so the detected freeze length can be
    checked precisely."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={freeze_seconds}",
            "-f", "lavfi", "-i", f"nullsrc=s=320x240:d={motion_seconds}:r=30",
            "-filter_complex",
            "[1:v]geq=random(1)*255:random(2)*255:random(3)*255[moving];"
            "[0:v][moving]concat=n=2:v=1:a=0[out]",
            "-map", "[out]", "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


def _continuously_moving_clip(path, duration=5.0):
    """Per-frame random noise — guarantees a large frame-to-frame pixel
    difference throughout, unlike testsrc's subtle continuous gradient
    (which can register as "frozen" at the same noise tolerance real Kling
    output needs — AI-video encoder grain/dither means a genuinely held
    pose isn't perfectly pixel-identical frame to frame either, so the
    threshold has to tolerate small deltas; that just makes testsrc's
    subtle motion a bad, unrepresentative test fixture, not a reason to
    loosen production's threshold)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"nullsrc=s=320x240:d={duration}:r=30",
            "-vf", "geq=random(1)*255:random(2)*255:random(3)*255",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )


def test_leading_freeze_seconds_detects_a_real_static_intro(tmp_path):
    clip = tmp_path / "clip.mp4"
    _frozen_then_moving_clip(clip, freeze_seconds=3.0, motion_seconds=2.0)
    skip = assembly._leading_freeze_seconds(clip, probe_duration(clip))
    assert 2.5 <= skip <= 3.2, f"expected ~3s leading freeze, got {skip}"


def test_leading_freeze_seconds_is_zero_for_continuous_motion(tmp_path):
    clip = tmp_path / "clip.mp4"
    _continuously_moving_clip(clip, duration=5.0)
    skip = assembly._leading_freeze_seconds(clip, probe_duration(clip))
    assert skip == 0.0


def test_leading_freeze_seconds_caps_at_70_percent_of_clip_duration(tmp_path):
    """A clip frozen almost throughout must still keep a meaningful slice
    of real source material, not get trimmed to nothing."""
    clip = tmp_path / "clip.mp4"
    _frozen_then_moving_clip(clip, freeze_seconds=4.8, motion_seconds=0.3)
    total = probe_duration(clip)
    skip = assembly._leading_freeze_seconds(clip, total)
    assert skip <= total * 0.7 + 0.05


def test_build_scene_video_segment_from_clip_skips_the_leading_freeze(tmp_path):
    """Wiring test: the assembled segment's very first frame must already
    be past the frozen intro, not the static source pose. Samples a strip
    of pixels near the very top of the FULL FRAME_WIDTH x FRAME_HEIGHT
    output (build_scene_video_segment_from_clip always scales up to that,
    regardless of the small test clip's own resolution) — well above where
    a "middle"-positioned caption card would sit, so this measures the
    underlying video content, not caption text/background."""
    from shorts_factory.captions import FRAME_WIDTH

    clip = tmp_path / "clip.mp4"
    _frozen_then_moving_clip(clip, freeze_seconds=3.0, motion_seconds=2.0)
    overlay, _box = captions.caption_overlay_png("x", style="comic_punch_orange")

    seg = assembly.build_scene_video_segment_from_clip(clip, duration=4.0, caption_overlay=overlay, index=0, segments_dir=tmp_path)

    frame_path = tmp_path / "frame.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "0.05", "-i", str(seg), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    frame = Image.open(frame_path).convert("RGB")
    # The frozen intro is solid blue (0, 0, 255) across the ENTIRE frame;
    # the noise-based "moving" segment varies wildly pixel to pixel. Sample
    # a spread of points near the top (above any caption) and check they
    # are NOT all uniform solid blue — if the segment still opens on the
    # frozen intro, every one of these would read (0, 0, 255).
    samples = [frame.getpixel((x, 40)) for x in range(50, FRAME_WIDTH - 50, 100)]
    assert len(set(samples)) > 1, f"segment still opens on the frozen blue intro frame: {samples}"
