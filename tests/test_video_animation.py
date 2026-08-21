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
        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker, reference_image_path=None):
            image_calls.append((scene_index, reference_image_path is not None))
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
    def fake_assemble_animated(scenes, clip_source, audio, workdir, out_mp4, caption_style=None):
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

    # 2. Every scene renders its own base image (one call per scene, not
    # shared/reused) — mascot-type scenes edit FROM the hero image
    # (reference_image_path is set); ingredient_grid/process_action scenes
    # render fresh with no reference (StubLLMProvider cycles scene_type
    # across scenes, see providers/llm.py).
    scenes = res.script["scenes"]
    per_scene_image_calls = [c for c in image_calls if c[0] != "hero"]
    assert len(per_scene_image_calls) == len(scenes)
    for s_idx, had_reference in per_scene_image_calls:
        stype = scenes[s_idx].get("scene_type")
        if stype in ("ingredient_grid", "process_action"):
            assert had_reference is False
        else:
            assert had_reference is True

    # 3. Video calls were made for all scenes, each animating that scene's
    # own freshly-generated raw image (not the literal hero image directly)
    # — this is what lets composition (small-and-pointing, big-and-reacting,
    # split-canvas) actually vary per scene instead of every mascot scene
    # animating one identical frozen pose.
    assert len(video_calls) == len(scenes)
    for s_idx, base_image_name in video_calls:
        assert base_image_name == f"raw_{s_idx:02d}.png"


def test_hero_cache_key_changes_with_model_or_style_or_prompt():
    """Regression test: the hero image cache was keyed on mascot.id alone.
    That's not enough — switching IMAGE_MODEL (e.g. Recraft/Imagen ->
    Nano Banana) or editing a mascot's hero_prompt text while keeping the
    same mascot_id would silently keep reusing a hero image rendered under
    the old model/prompt, since the cache only ever checked "does a file
    named hero_<mascot.id>.png already exist" — it never checked whether
    that file still matches the CURRENT model/prompt/style."""
    from shorts_factory.mascots import get_mascot
    from shorts_factory.pipeline import _hero_cache_key

    m4 = get_mascot("mascot_4")
    key_a = _hero_cache_key(m4, "fal-ai/recraft-v3", "digital_illustration/hand_drawn_outline")
    key_b = _hero_cache_key(m4, "fal-ai/nano-banana", "digital_illustration/hand_drawn_outline")
    key_c = _hero_cache_key(m4, "fal-ai/recraft-v3", "")
    assert key_a != key_b, "changing the image model must change the cache key"
    assert key_a != key_c, "changing the style preset must change the cache key"

    class _FakeMascot:
        hero_prompt = "a different hero prompt text"
        id = "mascot_4"

    key_d = _hero_cache_key(_FakeMascot(), "fal-ai/recraft-v3", "digital_illustration/hand_drawn_outline")
    assert key_a != key_d, "changing the mascot's hero_prompt text must change the cache key"


def test_switching_mascot_never_reuses_a_different_mascots_hero_image(tmp_path, monkeypatch):
    """Regression test: artifacts/<topic>/_work persists across separate
    run_pipeline() calls for the same topic. A fixed "hero.png" filename let
    a stale hero image from a since-changed mascot get silently reused —
    confirmed for real 2026-08-20/21: a flat-2D mascot from 3 days earlier
    was reused as scene 0's I2V source in a video whose other scenes
    correctly used the new mascot. hero_path must be keyed by mascot.id."""
    from shorts_factory import assembly, pipeline
    from shorts_factory.config import Settings
    from shorts_factory.providers.image import ImageProvider
    from shorts_factory.providers.video import VideoProvider

    image_calls = []

    class FakeImageProvider(ImageProvider):
        name = "fake_img"

        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker, reference_image_path=None):
            image_calls.append(scene_index)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake_image_bytes")
            return out_path

    class FakeVideoProvider(VideoProvider):
        name = "fake_vid"

        def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake_video_bytes")
            return out_path

    monkeypatch.setattr(pipeline, "get_image_provider", lambda *a, **k: FakeImageProvider())
    monkeypatch.setattr(pipeline, "get_video_provider", lambda *a, **k: FakeVideoProvider())

    def fake_assemble_animated(scenes, clip_source, audio, workdir, out_mp4, caption_style=None):
        for i, s in enumerate(scenes):
            clip_source(i, s)
        return {"caption_boxes": [assembly.CaptionBox(100, 300, 900, 500) for _ in scenes]}

    monkeypatch.setattr(assembly, "assemble_animated", fake_assemble_animated)
    monkeypatch.setattr(assembly, "assemble", lambda *a, **k: {"caption_boxes": []})
    monkeypatch.setattr(
        assembly,
        "synthesize_scenes",
        lambda tts, scenes, audio_dir, cost_tracker: [
            assembly.SceneAudio(path=audio_dir / f"s{i}.wav", duration=5.0, scripted_duration=5.0)
            for i in range(len(scenes))
        ],
    )
    monkeypatch.setattr(pipeline.verify, "run_verification", lambda **kwargs: {"overall_pass": True})

    test_settings = Settings(
        book_file="stub", output_language="English", visual_style="stub",
        budget_cap_usd=5.0, budget_cap_is_stub=False, music_sfx_source="stub",
        llm=ProviderConfig("llm", "stub", "stub"), tts=ProviderConfig("tts", "stub", "stub"),
        image=ProviderConfig("image", "fal", "fal-ai/nano-banana"),
        video=ProviderConfig("video", "fal", "fal-ai/kling-video/v1.5/pro/image-to-video"),
        search=ProviderConfig("search", "stub", "stub"), search_api_key="", fal_key="fake_key",
        fal_llm_endpoint="", tts_voice="", image_style="",
        llm_cost_per_script_usd=0.0, tts_cost_per_1k_chars_usd=0.0, image_cost_per_image_usd=0.0,
        video_cost_per_second_usd=0.05, youtube_client_secrets_file="", youtube_token_file="",
        telegram_bot_token="", telegram_allowed_user_ids=(),
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: test_settings)

    # Same topic, same artifacts_root (tmp_path) — mimics regenerating the
    # same video days apart after the mascot changed.
    pipeline.run_pipeline("soap", mascot_id="mascot_1", artifacts_root=tmp_path)
    hero_calls_after_first = image_calls.count("hero")
    assert hero_calls_after_first == 1

    pipeline.run_pipeline("soap", mascot_id="mascot_2", artifacts_root=tmp_path)
    # A second "hero" image call must have happened for mascot_2 — if the
    # stale-reuse bug were back, this would stay at 1 (mascot_1's hero
    # silently reused) instead of becoming 2.
    assert image_calls.count("hero") == 2

    from shorts_factory.mascots import get_mascot
    from shorts_factory.pipeline import _hero_cache_key

    generated_dir = tmp_path / "soap" / "_work" / "generated"
    key1 = _hero_cache_key(get_mascot("mascot_1"), test_settings.image.model_or_voice, test_settings.image_style)
    key2 = _hero_cache_key(get_mascot("mascot_2"), test_settings.image.model_or_voice, test_settings.image_style)
    assert (generated_dir / f"hero_mascot_1_{key1}.png").exists()
    assert (generated_dir / f"hero_mascot_2_{key2}.png").exists()


def test_static_image_path_anchors_mascot_scenes_on_hero_via_reference(tmp_path, monkeypatch):
    """Regression test: the non-animate (VIDEO_PROVIDER=stub, IMAGE_PROVIDER=fal)
    path independently called the image model once per mascot scene using only
    a text prompt — no shared reference image. A stochastic image model given
    the same character description twice can render a visibly different
    character each time. Confirmed for real 2026-08-21: an unrelated reference
    video showed 4 different character designs across mascot-labeled scenes
    generated this way. The fix is NOT to reuse one frozen hero image for
    every mascot scene either (that guarantees consistency but kills all
    per-scene pose/composition variety — also flagged for real by the user,
    "should sometimes be small... sometimes not in the scene at all"): every
    scene must render its own image, with mascot-type scenes anchored on the
    hero image via a reference (image-to-image edit) instead of a bare prompt."""
    from shorts_factory import assembly, pipeline
    from shorts_factory.config import Settings
    from shorts_factory.providers.image import ImageProvider

    image_calls = []

    class FakeImageProvider(ImageProvider):
        name = "fake_img"

        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker, reference_image_path=None):
            image_calls.append((scene_index, reference_image_path))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1024, 1024), (10, 20, 30)).save(out_path)
            return out_path

    monkeypatch.setattr(pipeline, "get_image_provider", lambda *a, **k: FakeImageProvider())

    def fake_assemble(scenes, frame_source, audio, workdir, out_mp4, caption_style=None):
        for i, s in enumerate(scenes):
            frame_source(i, s)
        return {"caption_boxes": [assembly.CaptionBox(100, 300, 900, 500) for _ in scenes]}

    monkeypatch.setattr(assembly, "assemble", fake_assemble)
    monkeypatch.setattr(
        assembly,
        "synthesize_scenes",
        lambda tts, scenes, audio_dir, cost_tracker: [
            assembly.SceneAudio(path=audio_dir / f"s{i}.wav", duration=5.0, scripted_duration=5.0)
            for i in range(len(scenes))
        ],
    )
    monkeypatch.setattr(pipeline.verify, "run_verification", lambda **kwargs: {"overall_pass": True})

    test_settings = Settings(
        book_file="stub", output_language="English", visual_style="stub",
        budget_cap_usd=5.0, budget_cap_is_stub=False, music_sfx_source="stub",
        llm=ProviderConfig("llm", "stub", "stub"), tts=ProviderConfig("tts", "stub", "stub"),
        image=ProviderConfig("image", "fal", "fal-ai/nano-banana"),
        video=ProviderConfig("video", "stub", "stub"),
        search=ProviderConfig("search", "stub", "stub"), search_api_key="", fal_key="fake_key",
        fal_llm_endpoint="", tts_voice="", image_style="",
        llm_cost_per_script_usd=0.0, tts_cost_per_1k_chars_usd=0.0, image_cost_per_image_usd=0.0,
        video_cost_per_second_usd=0.0, youtube_client_secrets_file="", youtube_token_file="",
        telegram_bot_token="", telegram_allowed_user_ids=(),
    )
    monkeypatch.setattr(pipeline, "load_settings", lambda: test_settings)

    res = pipeline.run_pipeline("soap", mascot_id="mascot_4", artifacts_root=tmp_path)

    # Hero image generated exactly once.
    hero_calls = [c for c in image_calls if c[0] == "hero"]
    assert len(hero_calls) == 1
    from shorts_factory.mascots import get_mascot
    from shorts_factory.pipeline import _hero_cache_key

    generated_dir = tmp_path / "soap" / "_work" / "generated"
    hero_key = _hero_cache_key(get_mascot("mascot_4"), test_settings.image.model_or_voice, test_settings.image_style)
    real_hero_path = generated_dir / f"hero_mascot_4_{hero_key}.png"
    assert real_hero_path.exists()

    # StubLLMProvider cycles scene_type across scenes (see providers/llm.py).
    # Every scene must render its OWN image exactly once. Mascot-type scenes
    # must be anchored on the hero image (reference_image_path set to it);
    # ingredient_grid/process_action scenes must NOT get a character reference.
    scenes = res.script["scenes"]
    mascot_scene_indices = [
        i for i, s in enumerate(scenes) if s.get("scene_type") not in ("ingredient_grid", "process_action")
    ]
    fresh_scene_indices = [
        i for i, s in enumerate(scenes) if s.get("scene_type") in ("ingredient_grid", "process_action")
    ]
    assert mascot_scene_indices, "test script must contain at least one mascot-type scene"
    per_scene_calls = {idx: ref for idx, ref in image_calls if idx != "hero"}
    assert sorted(per_scene_calls) == list(range(len(scenes)))
    for i in mascot_scene_indices:
        assert per_scene_calls[i] == real_hero_path
    for i in fresh_scene_indices:
        assert per_scene_calls[i] is None


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

        def generate_scene_image(self, scene, scene_index, out_path, cost_tracker, reference_image_path=None):
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

