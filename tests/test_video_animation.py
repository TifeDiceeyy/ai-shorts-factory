from pathlib import Path

import pytest
from PIL import Image

from shorts_factory.assembly import assemble_animated, probe_duration, synthesize_scenes
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.tts import StubTTSProvider
from shorts_factory.config import ProviderConfig, Settings
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


def test_animate_path_generates_hero_once_and_reuses_for_mascot_scenes(tmp_path, monkeypatch):
    from shorts_factory import assembly, pipeline
    from shorts_factory.config import Settings
    from shorts_factory.providers.image import ImageProvider
    from shorts_factory.providers.video import VideoProvider

    image_calls = []
    class FakeImageProvider(ImageProvider):
        name = "fake_img"
        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker):
            image_calls.append((scene_index, scene.get("visual_prompt", "")))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake_image_bytes")
            return out_path

    video_calls = []
    class FakeVideoProvider(VideoProvider):
        name = "fake_vid"
        def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
            video_calls.append((scene_index, hero_image_path.name))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake_video_bytes")
            return out_path

    monkeypatch.setattr(pipeline, "get_image_provider", lambda *args, **kwargs: FakeImageProvider())
    monkeypatch.setattr(pipeline, "get_video_provider", lambda *args, **kwargs: FakeVideoProvider())
    def fake_assemble_animated(scenes, clip_source, audio, workdir, out_mp4):
        boxes = []
        for i, s in enumerate(scenes):
            clip_source(i, s)
            boxes.append(assembly.CaptionBox(100, 300, 900, 500))
        return {"caption_boxes": boxes}

    monkeypatch.setattr(assembly, "assemble_animated", fake_assemble_animated)
    monkeypatch.setattr(assembly, "assemble", lambda *args, **kwargs: {"caption_boxes": []})
    monkeypatch.setattr(assembly, "synthesize_scenes", lambda tts, scenes, audio_dir, cost_tracker: [
        assembly.SceneAudio(path=audio_dir / f"s{i}.wav", duration=5.0, scripted_duration=5.0) for i in range(len(scenes))
    ])
    monkeypatch.setattr(pipeline.verify, "run_verification", lambda **kwargs: {"overall_pass": True})

    test_settings = Settings(
        book_file="stub",
        output_language="English",
        visual_style="stub",
        budget_cap_usd=5.0,
        budget_cap_is_stub=False,
        music_sfx_source="stub",
        llm=ProviderConfig("llm", "stub", "stub"),
        tts=ProviderConfig("tts", "stub", "stub"),
        image=ProviderConfig("image", "fal", "fal-ai/imagen3"),
        video=ProviderConfig("video", "fal", "fal-ai/kling-video/v1.5/pro/image-to-video"),
        search=ProviderConfig("search", "stub", "stub"),
        search_api_key="",
        fal_key="fake_key",
        fal_llm_endpoint="",
        tts_voice="",
        image_style="",
        llm_cost_per_script_usd=0.0,
        tts_cost_per_1k_chars_usd=0.0,
        image_cost_per_image_usd=0.0,
        video_cost_per_second_usd=0.05,
        youtube_client_secrets_file="",
        youtube_token_file="",
        telegram_bot_token="",
        telegram_allowed_user_ids=(),
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: test_settings)

    res = pipeline.run_pipeline("soap", mascot_id="mascot_4", artifacts_root=tmp_path)
    assert res.mascot_id == "mascot_4"

    # 1. Hero image was generated once with scene_index "hero"
    hero_gen_calls = [c for c in image_calls if c[0] == "hero"]
    assert len(hero_gen_calls) == 1

    # 2. Video calls were made for all scenes using hero.png for mascot scenes
    assert len(video_calls) > 0
    for s_idx, hero_name in video_calls:
        # Default soap scenes are mascot scenes, so they all reuse hero.png
        assert hero_name == "hero.png"


def test_regenerate_scene_animate_path_does_not_crash(tmp_path, monkeypatch):
    """Regression test: regenerate_scene()'s animate branch used to call
    assembly.burn_caption_into_clip(), a function that has never existed
    (the real one is assembly.build_scene_video_segment_from_clip) — any
    real regenerate call with VIDEO_PROVIDER set would crash with
    AttributeError after already paying for TTS/image/video. Unlike the
    test above, assemble_animated/build_scene_video_segment_from_clip are
    NOT mocked here — real ffmpeg runs against real (tiny) media files, so
    this actually exercises the code path that was broken."""
    import subprocess

    from shorts_factory import pipeline
    from shorts_factory.config import ProviderConfig, Settings
    from shorts_factory.providers.image import ImageProvider
    from shorts_factory.providers.video import VideoProvider

    class FakeImageProvider(ImageProvider):
        name = "fake_img"

        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (60, 90, 140)).save(out_path)
            return out_path

    class FakeVideoProvider(VideoProvider):
        name = "fake_vid"

        def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=6",
                    "-pix_fmt", "yuv420p", str(out_path),
                ],
                check=True,
            )
            return out_path

    monkeypatch.setattr(pipeline, "get_image_provider", lambda *a, **k: FakeImageProvider())
    monkeypatch.setattr(pipeline, "get_video_provider", lambda *a, **k: FakeVideoProvider())

    test_settings = Settings(
        book_file="stub",
        output_language="English",
        visual_style="stub",
        budget_cap_usd=5.0,
        budget_cap_is_stub=False,
        music_sfx_source="stub",
        llm=ProviderConfig("llm", "stub", "stub"),
        tts=ProviderConfig("tts", "stub", "stub"),
        image=ProviderConfig("image", "fal", "fal-ai/imagen3"),
        video=ProviderConfig("video", "fal", "fal-ai/kling-video/v1.5/pro/image-to-video"),
        search=ProviderConfig("search", "stub", "stub"),
        search_api_key="",
        fal_key="fake_key",
        fal_llm_endpoint="",
        tts_voice="",
        image_style="",
        llm_cost_per_script_usd=0.0,
        tts_cost_per_1k_chars_usd=0.0,
        image_cost_per_image_usd=0.0,
        video_cost_per_second_usd=0.05,
        youtube_client_secrets_file="",
        youtube_token_file="",
        telegram_bot_token="",
        telegram_allowed_user_ids=(),
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: test_settings)

    full = pipeline.run_pipeline("soap", artifacts_root=tmp_path)
    assert full.verification is not None

    # This call is what used to raise AttributeError.
    result = pipeline.regenerate_scene("soap", 0, artifacts_root=tmp_path)
    assert result.verification is not None
    final_mp4 = tmp_path / "soap" / "soap.mp4"
    assert final_mp4.exists()

