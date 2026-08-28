"""Sticker/motion-graphics compositor — replaces continuous AI-video (Kling/
Hailuo) animation as the default (2026-08-27). See assembly.py's
build_scene_video_segment_from_still docstring for why: a real reference
short in this niche uses static stills with a snap pop-in, not continuous
AI "performance", and real Kling output was confirmed frozen for its entire
raw duration on half a test video's scenes even with aggressive motion
prompting.
"""
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from shorts_factory import assembly
from shorts_factory.assembly import SceneAudio
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH


def _still_image(tmp_path, color=(60, 90, 140)) -> Path:
    """A solid fill has no spatial features — zooming into one produces the
    exact same pixels everywhere, which would make a pop-in scale change
    invisible to a pixel-diff test even if it's working correctly. Draw a
    centered rectangle "subject" on a white ground so scaling it is actually
    detectable, matching the real shape of a scene image (a subject centered
    on white)."""
    from PIL import ImageDraw
    path = tmp_path / "scene.png"
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    draw.rectangle((cx - 150, cy - 150, cx + 150, cy + 150), fill=color)
    img.save(path)
    return path


def _frame_at(video_path: Path, t: float, out_path: Path) -> Image.Image:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", str(video_path),
         "-frames:v", "1", str(out_path)],
        check=True,
    )
    return Image.open(out_path).convert("RGB")


def test_pop_in_produces_visible_size_change_then_holds_steady(tmp_path):
    """The pop-in must actually move (overshoot then settle) in its first
    ~0.27s, then hold at a constant size for the rest of the scene — proving
    the zoompan expression is evaluated per-frame, not frozen at its initial
    value (a real bug hit while building this: ffmpeg's crop filter silently
    evaluates w/h expressions once at init unless the whole per-frame
    machinery lines up right; zoompan does not have that failure mode)."""
    img = _still_image(tmp_path)
    overlays, _box = assembly.build_timed_caption_overlays("test caption text", 2.0)
    seg = assembly.build_scene_video_segment_from_still(
        img, 2.0, 0, tmp_path / "segments", timed_caption_overlays=overlays
    )

    early = _frame_at(seg, 0.1, tmp_path / "f_early.png")
    settled_a = _frame_at(seg, 0.5, tmp_path / "f_settled_a.png")
    settled_b = _frame_at(seg, 1.5, tmp_path / "f_settled_b.png")

    # Crop to a region away from the caption overlay (top strip) so we're
    # only measuring the pop-in's own scale change, not caption timing.
    region = (0, FRAME_HEIGHT // 2, FRAME_WIDTH, FRAME_HEIGHT)
    assert ImageChops.difference(early.crop(region), settled_a.crop(region)).getbbox() is not None, (
        "no visible change between the pop-in's overshoot phase and its settled state"
    )
    assert ImageChops.difference(settled_a.crop(region), settled_b.crop(region)).getbbox() is None, (
        "the held phase must be pixel-stable once the pop-in settles (no drift/idle motion in v1)"
    )


def test_output_duration_matches_requested_duration(tmp_path):
    img = _still_image(tmp_path)
    overlays, _box = assembly.build_timed_caption_overlays("short narration here", 3.25)
    seg = assembly.build_scene_video_segment_from_still(
        img, 3.25, 0, tmp_path / "segments", timed_caption_overlays=overlays
    )
    assert assembly.probe_duration(seg) == pytest.approx(3.25, abs=0.12)


def test_assemble_stickers_end_to_end_no_video_provider_involved(tmp_path):
    """Free, stub-only proof that the whole sticker path produces a
    correctly-sized, correctly-timed final video purely from still images —
    image_source is never asked for anything but a path, no video-generation
    concept exists in this path at all."""
    from shorts_factory.providers.tts import StubTTSProvider
    from shorts_factory.cost_tracker import CostTracker

    scenes = [
        {"narration": "Could you make a metal tool from scratch?", "caption": "Could you?", "duration": 4.0},
        {"narration": "Ancient humans used smelting to purify ore.", "caption": "Smelting!", "duration": 4.0},
    ]
    tracker = CostTracker(budget_cap_usd=1.0)
    audio = assembly.synthesize_scenes(StubTTSProvider(), scenes, tmp_path / "audio", tracker)

    img = _still_image(tmp_path, color=(200, 100, 50))
    image_calls = []

    def image_source(i, scene):
        image_calls.append(i)
        return img

    out_mp4 = tmp_path / "out.mp4"
    result = assembly.assemble_stickers(scenes, image_source, audio, tmp_path / "work", out_mp4, caption_style="comic_punch_orange")

    assert out_mp4.exists()
    assert image_calls == [0, 1]
    assert len(result["caption_boxes"]) == 2
    total_audio = sum(a.duration for a in audio)
    assert assembly.probe_duration(out_mp4) == pytest.approx(total_audio, abs=0.3)


def test_sticker_mode_is_the_default_when_image_is_real_and_costs_no_video_spend(tmp_path, monkeypatch):
    """Regression test: ANIMATION_MODE defaults to "sticker" — a real
    IMAGE_PROVIDER with no ANIMATION_MODE set must NOT call the video
    provider at all, and must NOT require VIDEO_PROVIDER to be configured."""
    from shorts_factory import pipeline
    from shorts_factory.config import ProviderConfig, Settings
    from shorts_factory.providers.image import ImageProvider

    class FakeImageProvider(ImageProvider):
        name = "fake_img"

        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker, reference_image_path=None):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (60, 90, 140)).save(out_path)
            return out_path

    def refuses_if_called(*a, **k):
        raise AssertionError("get_video_provider must never be called in sticker mode")

    monkeypatch.setattr(pipeline, "get_image_provider", lambda *a, **k: FakeImageProvider())
    monkeypatch.setattr(pipeline, "get_video_provider", refuses_if_called)

    test_settings = Settings(
        book_file="stub", output_language="English", visual_style="stub",
        budget_cap_usd=5.0, budget_cap_is_stub=False, music_sfx_source="stub",
        llm=ProviderConfig("llm", "stub", "stub"), tts=ProviderConfig("tts", "stub", "stub"),
        image=ProviderConfig("image", "fal", "fal-ai/nano-banana"),
        video=ProviderConfig("video", "stub", "stub"),
        search=ProviderConfig("search", "stub", "stub"), search_api_key="", fal_key="fake_key",
        fal_llm_endpoint="", tts_voice="", image_style="",
        llm_cost_per_script_usd=0.0, tts_cost_per_1k_chars_usd=0.0, image_cost_per_image_usd=0.0,
        video_cost_per_second_usd=0.05, youtube_client_secrets_file="", youtube_token_file="",
        telegram_bot_token="", telegram_allowed_user_ids=(),
        # animation_mode intentionally omitted — must default to "sticker"
    )
    assert test_settings.animation_mode == "sticker"
    monkeypatch.setattr(pipeline, "load_settings", lambda: test_settings)

    result = pipeline.run_pipeline("soap", mascot_id="mascot_4", artifacts_root=tmp_path)

    assert result.verification is not None
    assert result.cost_report is not None
    assert all(e["provider"] != "fake_vid" for e in result.cost_report["entries"])
