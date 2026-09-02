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
from PIL import Image, ImageChops, ImageStat

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
    # Mean, not an exact getbbox() check. Word-for-word captions (2026-08-31)
    # add one overlay stage per word, and the extra filter stages perturb a
    # single h264 macroblock in an otherwise identical frame — measured
    # mean 0.0000 / max 8 over one 8x8 block, i.e. encoder noise, while the
    # image itself is genuinely still. Same reasoning as
    # test_object_animation._mean_channel_diff.
    held_diff = max(ImageStat.Stat(
        ImageChops.difference(settled_a.crop(region), settled_b.crop(region))
    ).mean)
    assert held_diff < 1.0, (
        f"the held phase must stay steady once the pop-in settles "
        f"(no drift/idle motion in v1) — got mean delta {held_diff:.3f}"
    )


def _max_channel_diff(a: Image.Image, b: Image.Image) -> int:
    from PIL import ImageStat
    stat = ImageStat.Stat(ImageChops.difference(a, b))
    return max(hi for _lo, hi in stat.extrema)


def test_pop_in_fires_once_per_scene_not_on_every_caption_cue(tmp_path):
    """Reverted 2026-08-29 per direct user feedback watching a real
    generated video: an earlier "editing rhythm" pass made the image
    re-pop at every caption cue start, but real caption cues for punchy
    narration land every ~0.8-1.5s — re-zooming the whole image that often
    read as constant zooming/fidgeting, not a deliberate beat. The image
    must now pop exactly once, at scene start, and hold steady through
    every later cue boundary — the caption's own per-cue scale-punch
    was removed entirely on 2026-08-31 (see
    test_caption_does_not_bounce_within_its_own_cue)."""
    img = _still_image(tmp_path)
    narration = "one two three four five six seven eight nine ten eleven twelve"
    overlays, _box = assembly.build_timed_caption_overlays(narration, 6.0)
    assert len(overlays) >= 3, "test needs multiple cues to prove later ones stay settled"
    seg = assembly.build_scene_video_segment_from_still(
        img, 6.0, 0, tmp_path / "segments", timed_caption_overlays=overlays
    )

    region = (0, FRAME_HEIGHT // 2, FRAME_WIDTH, FRAME_HEIGHT)
    noise_floor = 50  # well above the 1-2 unit encoder noise floor, well below a real ~210 pop delta
    settled_at_scene_start = _frame_at(seg, 0.55, tmp_path / "settled_0.png").crop(region)
    for next_cue in overlays[1:]:
        just_after = _frame_at(
            seg, next_cue.start + 0.05, tmp_path / f"after_{next_cue.start:.2f}.png"
        ).crop(region)
        diff = _max_channel_diff(settled_at_scene_start, just_after)
        assert diff < noise_floor, (
            f"expected the image to stay settled through cue start {next_cue.start:.2f}s "
            f"(image pops once, at scene start, not per cue) — got a real content change (delta {diff})"
        )


def test_caption_does_not_bounce_within_its_own_cue(tmp_path):
    """Inverted 2026-08-31 on direct user instruction that captions must not
    bounce any more. A caption used to scale-punch (overshoot then settle)
    as it appeared; now each cue is a single word that cuts straight in and
    holds perfectly still for its whole slot. Sampled twice WITHIN one cue —
    across cues the word itself legitimately changes, which is the intended
    motion and not what this test is about."""
    img = _still_image(tmp_path)
    narration = "one two three four five six seven eight nine ten eleven twelve"
    overlays, _box = assembly.build_timed_caption_overlays(narration, 6.0)
    # Need a cue long enough to sample twice strictly inside it.
    cue = next((c for c in overlays[1:] if (c.end - c.start) > 0.35), None)
    assert cue is not None, "test needs a cue longer than 0.35s to sample inside"
    seg = assembly.build_scene_video_segment_from_still(
        img, 6.0, 0, tmp_path / "segments", timed_caption_overlays=overlays
    )
    pad = 60
    region = (
        max(0, cue.box.left - pad), max(0, cue.box.top - pad),
        min(FRAME_WIDTH, cue.box.right + pad), min(FRAME_HEIGHT, cue.box.bottom + pad),
    )
    early = _frame_at(seg, cue.start + 0.04, tmp_path / "cap_early.png").crop(region)
    later = _frame_at(seg, cue.start + 0.30, tmp_path / "cap_later.png").crop(region)
    # Mean, not max: a few h264-noisy glyph-edge pixels are expected even
    # when nothing moved (same reasoning as test_object_animation's helper).
    mean_diff = max(ImageStat.Stat(ImageChops.difference(early, later)).mean)
    assert mean_diff < 2.0, (
        f"caption must hold still inside its own cue (no bounce), got mean delta {mean_diff:.2f}"
    )


def test_captions_type_out_word_by_word_then_clear_each_line(tmp_path):
    """Typewriter reveal, per direct user instruction 2026-09-01: a lone word
    replaced on every beat vanished too fast to read ("what about slow
    readers"), so each word ADDS to the line and the line only clears every
    CAPTION_WORDS_PER_LINE words.

    Still one cue per word (the reveal is in step with the voiceover), the
    words are still exactly the narration in order, and the cues still cover
    the full scene duration."""
    narration = "Roman concrete repairs itself when rain gets into a crack."
    words = narration.split()
    n = assembly.CAPTION_WORDS_PER_LINE
    cues = assembly.narration_caption_cues(narration, 6.0)

    assert len(cues) == len(words), "one cue per word"
    for i, cue in enumerate(cues):
        line_start = (i // n) * n
        assert cue.text == " ".join(words[line_start:i + 1]), (
            f"cue {i} should show the line accumulated so far, got {cue.text!r}"
        )
    # every word appears, in order, exactly once as a newly-added word
    assert [c.text.split()[-1] for c in cues] == words
    # a line grows to n words and then resets
    assert len(cues[n - 1].text.split()) == n
    assert len(cues[n].text.split()) == 1, "line must clear after CAPTION_WORDS_PER_LINE"
    assert cues[0].start == 0.0 and abs(cues[-1].end - 6.0) < 1e-6


def test_pop_in_does_not_fire_on_every_scene(tmp_path):
    """Direct user instruction 2026-09-01: the whole-image pop "should not
    happen on every scene" — at 15 scenes it was a bounce every ~4s. Only
    every POP_IN_EVERY_N_SCENES-th scene pops; the rest open already settled.

    Compared against a scene rendered WITH the pop, so this can't pass just
    because both happen to look static."""
    img = _still_image(tmp_path)
    overlays, _box = assembly.build_timed_caption_overlays("a caption line here", 2.0)
    region = (0, FRAME_HEIGHT // 2, FRAME_WIDTH, FRAME_HEIGHT)

    def opening_movement(pop: bool, idx: int) -> float:
        seg = assembly.build_scene_video_segment_from_still(
            img, 2.0, idx, tmp_path / f"seg_{idx}",
            timed_caption_overlays=overlays, pop_in=pop,
        )
        settled = _frame_at(seg, 1.2, tmp_path / f"s_{idx}.png").crop(region)
        # Scan the whole pop window rather than one instant: ffmpeg's fast
        # seek quantizes several early requests onto the same decoded frame,
        # so a single sample can miss the overshoot entirely.
        deltas = []
        for t in (0.10, 0.15, 0.20, 0.25):
            f = _frame_at(seg, t, tmp_path / f"e_{idx}_{t}.png").crop(region)
            deltas.append(max(ImageStat.Stat(ImageChops.difference(f, settled)).mean))
        return max(deltas)

    popped = opening_movement(True, 0)
    flat = opening_movement(False, 1)
    # Measured separation on this fixture is ~0.89 vs exactly 0.000 — the
    # subject is a small part of a mostly-white half-frame, so the mean is
    # diluted. Absolute size doesn't matter; the two must be distinguishable.
    assert popped > 0.3, f"a popping scene must visibly scale in, got {popped:.3f}"
    assert flat < 0.05, f"a non-popping scene must open already settled, got {flat:.3f}"
    assert popped > flat * 10, f"pop {popped:.3f} vs no-pop {flat:.3f}"


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

    # No manual mascot override exists anymore — "soap" is one of mascot_4's
    # own keywords, so story-matching resolves to it deterministically.
    result = pipeline.run_pipeline("soap", artifacts_root=tmp_path)

    assert result.verification is not None
    assert result.cost_report is not None
    assert all(e["provider"] != "fake_vid" for e in result.cost_report["entries"])


def test_subscribe_cta_composited_only_onto_final_cue_overlay(tmp_path):
    """subscribe_cta_text (unlike caution_text) must land on the LAST cue's
    overlay only — it's an end-of-video call to action, not a repeated
    badge. Build the same narration/duration with and without a CTA and
    diff every cue: every cue except the last must be pixel-identical."""
    narration = "This is a longer sentence with several distinct words to caption across many cues."
    duration = 8.0
    baseline_overlays, _ = assembly.build_timed_caption_overlays(narration, duration)
    cta_overlays, _ = assembly.build_timed_caption_overlays(
        narration, duration, subscribe_cta_text="SUBSCRIBE!"
    )
    assert len(baseline_overlays) == len(cta_overlays)
    assert len(baseline_overlays) >= 2, "test needs multiple cues to prove the CTA isn't applied to all of them"

    for i in range(len(baseline_overlays) - 1):
        assert ImageChops.difference(baseline_overlays[i].image, cta_overlays[i].image).getbbox() is None, (
            f"cue {i} (not the last cue) must be unaffected by subscribe_cta_text"
        )
    assert ImageChops.difference(baseline_overlays[-1].image, cta_overlays[-1].image).getbbox() is not None, (
        "the last cue must visibly differ once the CTA is composited onto it"
    )


def test_subscribe_cta_appears_near_the_end_of_the_assembled_video_not_earlier(tmp_path):
    """End-to-end proof through assemble_stickers: a frame near the very end
    of the final video shows the red CTA text, an early frame from the same
    scene does not."""
    from shorts_factory.providers.tts import StubTTSProvider
    from shorts_factory.cost_tracker import CostTracker

    scenes = [
        {"narration": "Could you make a metal tool from scratch right now today?", "caption": "Could you?", "duration": 4.0},
        {"narration": "Ancient humans used smelting to purify raw ore into metal.", "caption": "Smelting!", "duration": 4.0},
    ]
    tracker = CostTracker(budget_cap_usd=1.0)
    audio = assembly.synthesize_scenes(StubTTSProvider(), scenes, tmp_path / "audio", tracker)

    img = _still_image(tmp_path, color=(200, 100, 50))

    def image_source(i, scene):
        return img

    out_mp4 = tmp_path / "out.mp4"
    assembly.assemble_stickers(
        scenes, image_source, audio, tmp_path / "work", out_mp4,
        caption_style="comic_punch_orange", subscribe_cta_text="SUBSCRIBE!",
    )

    total_duration = assembly.probe_duration(out_mp4)
    late_frame = _frame_at(out_mp4, max(0.0, total_duration - 0.3), tmp_path / "f_late.png")
    early_frame = _frame_at(out_mp4, 0.5, tmp_path / "f_early.png")

    # SUBSCRIBE_CTA_STYLE is bright red (255, 45, 45) text near the bottom —
    # look for a strong-red pixel cluster in the bottom third only.
    bottom = (0, int(FRAME_HEIGHT * 0.66), FRAME_WIDTH, FRAME_HEIGHT)

    def has_strong_red(frame: Image.Image) -> bool:
        region = frame.crop(bottom)
        px = region.load()
        for y in range(0, region.height, 4):
            for x in range(0, region.width, 4):
                r, g, b = px[x, y]
                if r > 180 and g < 110 and b < 110:
                    return True
        return False

    assert has_strong_red(late_frame), "expected the red SUBSCRIBE CTA near the end of the video"
    assert not has_strong_red(early_frame), "the CTA must not appear on an early frame"
