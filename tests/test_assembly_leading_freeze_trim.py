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
from PIL import Image, ImageChops

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


def _solid_still_image(path, color=(0, 200, 0)):
    """A distinctive, recognizable still — used to prove the padded
    remainder shows THIS image (a real cut) rather than a held/zoomed frame
    from the noise-based clip."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1080, 1920), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((300, 700, 780, 1200), fill=color)
    img.save(path)


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


def test_duration_fitting_keeps_motion_changing_through_final_second(tmp_path):
    """A short source clip stretched to longer narration must not freeze its
    final frame. The old tpad path cloned the last frame for the remainder."""
    clip = tmp_path / "clip.mp4"
    _continuously_moving_clip(clip, duration=2.0)
    overlay, _box = captions.caption_overlay_png("moving", style="comic_punch_orange")
    seg = assembly.build_scene_video_segment_from_clip(
        clip, duration=5.0, caption_overlay=overlay, index=0, segments_dir=tmp_path
    )

    frames = []
    for n, timestamp in enumerate((4.0, 4.8)):
        path = tmp_path / f"tail_{n}.png"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(timestamp), "-i", str(seg), "-frames:v", "1", str(path)],
            check=True,
        )
        frames.append(Image.open(path).convert("RGB").crop((0, 0, 1080, 500)))
    assert ImageChops.difference(frames[0], frames[1]).getbbox() is not None


def test_padded_remainder_hard_cuts_to_the_scene_image_instead_of_holding_the_clip(tmp_path):
    """Regression test, direct user feedback (2026-08-28): once real usable
    motion runs out and can't reasonably continue, the scene should hard-cut
    to something else rather than "wait it out" holding the same shot. A
    real Kling clip observed 2026-08-27/28 spent 6.29s of an 8.5s scene in
    the old held-frame (tpad + Ken Burns) pad — most of the shot. When an
    image_path is supplied and the remainder is worth cutting for
    (> CUT_IN_MIN_PAD_SECONDS), the remainder must show the scene's own
    still image (a real edit cut), not the clip's last frame held/zoomed."""
    clip = tmp_path / "clip.mp4"
    _continuously_moving_clip(clip, duration=2.0)  # usable_duration ~2.0s
    image_path = tmp_path / "scene.png"
    _solid_still_image(image_path, color=(0, 200, 0))
    overlay, _box = captions.caption_overlay_png("x", style="comic_punch_orange")

    # duration=6.0, MAX_CLIP_STRETCH_FACTOR=1.8 -> played_duration=3.6s,
    # pad_duration=2.4s (well above the 0.75s cut-in threshold).
    seg = assembly.build_scene_video_segment_from_clip(
        clip, duration=6.0, caption_overlay=overlay, index=0, segments_dir=tmp_path, image_path=image_path
    )

    frame_path = tmp_path / "post_cut.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", "5.0", "-i", str(seg), "-frames:v", "1", str(frame_path)],
        check=True,
    )
    frame = Image.open(frame_path).convert("RGB")
    # t=5.0 is well inside the pad window [3.6, 6.0] and well past the
    # ~0.27s pop-in settle — the frame center should show the still image's
    # green rectangle, not noise.
    r, g, b = frame.getpixel((frame.width // 2, frame.height // 2))
    assert g > r + 40 and g > b + 40, (
        f"expected the padded remainder to show the scene's still image (greenish), got RGB=({r},{g},{b}) "
        "— looks like the old clip-hold behavior instead of a hard cut"
    )
