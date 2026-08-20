from pathlib import Path

import pytest
from PIL import Image

from shorts_factory.assembly import assemble_animated, probe_duration, synthesize_scenes
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.tts import StubTTSProvider
from shorts_factory.providers.video import HAILUO_CLIP_SECONDS, StubVideoProvider


def _hero_image(tmp_path) -> Path:
    path = tmp_path / "hero.png"
    Image.new("RGB", (1024, 1024), (200, 100, 50)).save(path)
    return path


def test_stub_video_provider_produces_a_clip_of_the_fixed_clip_length(tmp_path):
    hero = _hero_image(tmp_path)
    provider = StubVideoProvider()
    tracker = CostTracker(budget_cap_usd=1.0)

    out = provider.generate_scene_video({"visual_prompt": "x"}, hero, 0, tmp_path / "clip.mp4", tracker)

    assert probe_duration(out) == pytest.approx(HAILUO_CLIP_SECONDS, abs=0.1)
    assert tracker.total_spent_usd == 0.0


def test_assemble_animated_pads_short_clips_and_burns_captions(tmp_path):
    """End-to-end (free, stub-only) proof that the animate path produces a
    correctly-sized, correctly-timed final video with captions — the same
    mechanics a real fal video run depends on, just with a local clip
    standing in for Hailuo's output."""
    hero = _hero_image(tmp_path)
    video_provider = StubVideoProvider()
    tracker = CostTracker(budget_cap_usd=1.0)
    tts = StubTTSProvider()

    scenes = [
        {"narration": "Scene one narration here.", "caption": "Scene one", "duration": 8.0},
        {"narration": "Scene two narration here.", "caption": "Scene two", "duration": 8.0},
    ]
    audio = synthesize_scenes(tts, scenes, tmp_path / "audio", tracker)

    def clip_source(i, scene):
        return video_provider.generate_scene_video(scene, hero, i, tmp_path / f"raw_{i}.mp4", tracker)

    result = assemble_animated(
        scenes=scenes, clip_source=clip_source, audio=audio, workdir=tmp_path / "work", out_mp4=tmp_path / "out.mp4"
    )

    assert len(result["caption_boxes"]) == 2
    final_duration = probe_duration(tmp_path / "out.mp4")
    expected_total = sum(a.duration for a in audio)
    assert final_duration == pytest.approx(expected_total, abs=0.2)
