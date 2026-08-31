"""Tests for Roman-concrete-style pipeline upgrades."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from shorts_factory.assembly import narration_caption_cues
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH, word_chunk_overlay_frames
from shorts_factory.providers.llm import repair_sticker_manifest
from shorts_factory.sticker_compositor import configure_motion, write_sticker_scene_video
from shorts_factory.sticker_qa import validate_sticker_image
from shorts_factory.sticker_timing import sync_sticker_appear_times


def _white_sticker(path: Path, color=(200, 80, 40)) -> None:
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((120, 120, 392, 392), fill=color)
    img.save(path)


def test_sticker_qa_accepts_isolated_white_background(tmp_path):
    path = tmp_path / "good.png"
    _white_sticker(path)
    result = validate_sticker_image(path)
    assert result.ok, result.reason


def test_sticker_qa_rejects_full_frame_fill(tmp_path):
    path = tmp_path / "bad.png"
    Image.new("RGB", (512, 512), (200, 80, 40)).save(path)
    result = validate_sticker_image(path)
    assert not result.ok


def test_build_stickers_trigger_words_fit_schema():
    from shorts_factory.providers.llm import build_stickers_for_scene
    from shorts_factory.schema_validate import validate_script_shape

    scene = {
        "scene_type": "mascot_reaction",
        "props": "hammer, anvil",
        "action": "striking hot metal on the anvil",
        "duration": 8.0,
    }
    stickers = build_stickers_for_scene(scene, 8.0, "flat cartoon", ["stk-001", "stk-002", "stk-003"])
    for sticker in stickers:
        tw = sticker.get("trigger_words")
        if tw is not None:
            assert len(tw) <= 3
        if sticker.get("uses_hero"):
            assert "trigger_words" not in sticker

    script = {
        "topic": "metal",
        "language": "English",
        "visual_style": "flat cartoon",
        "scenes": [
            {
                "narration": "n",
                "caption": "c",
                "duration": 8.0,
                "visual_prompt": "v",
                "source_claim_id": "claim-01",
                "scene_type": "ingredient_grid",
                "props": "limestone, volcanic ash, gravel",
                "stickers": stickers,
            }
        ],
    }
    validate_script_shape(script)


def test_repair_sticker_manifest_fills_missing_count():
    script = {
        "visual_style": "flat cartoon",
        "scenes": [
            {"duration": 8.0, "scene_type": "mascot_reaction", "props": "hammer", "stickers": []},
            {"duration": 8.0, "scene_type": "ingredient_grid", "props": "lime, ash", "stickers": []},
            {"duration": 8.0, "scene_type": "process_action", "props": "fire", "stickers": []},
            {"duration": 8.0, "scene_type": "mascot_reaction", "props": "mix", "stickers": []},
            {"duration": 8.0, "scene_type": "split_canvas", "props": "pour", "stickers": []},
        ],
    }
    repair_sticker_manifest(script, target_min=12, target_max=15)
    image_count = sum(
        1 for scene in script["scenes"] for sticker in scene["stickers"] if not sticker.get("is_label")
    )
    assert 12 <= image_count <= 15


def test_sync_sticker_appear_times_uses_narration_keywords():
    scenes = [
        {
            "scene_type": "ingredient_grid",
            "narration": "You need limestone and volcanic ash to start.",
            "duration": 6.0,
            "props": "limestone, volcanic ash",
            "stickers": [
                {
                    "id": "stk-001",
                    "visual_prompt": "flat sticker of limestone on pure white",
                    "trigger_words": ["limestone"],
                    "appear_at": 99.0,
                    "entrance": "fade_in",
                    "idle": "float",
                    "position": "top_left",
                },
                {
                    "id": "stk-002",
                    "visual_prompt": "flat sticker of volcanic ash on pure white",
                    "trigger_words": ["volcanic", "ash"],
                    "appear_at": 99.0,
                    "entrance": "fade_in",
                    "idle": "float",
                    "position": "top_right",
                },
            ],
        }
    ]
    sync_sticker_appear_times(scenes, [6.0])
    lime_t = scenes[0]["stickers"][0]["appear_at"]
    ash_t = scenes[0]["stickers"][1]["appear_at"]
    assert lime_t < ash_t
    assert lime_t >= 0.0
    # "limestone" is the 3rd spoken word — should not appear at scene start.
    assert lime_t > 0.5


def test_stone_sticker_appears_when_stone_is_spoken():
    """Regression: subject noun timing, not a fixed delay after the cue chunk."""
    from shorts_factory.assembly import narration_word_timings

    narration = "First gather your stone then mix sand."
    duration = 4.0
    stone_start = next(start for word, start, _ in narration_word_timings(narration, duration) if word == "stone")
    scenes = [
        {
            "scene_type": "process_action",
            "narration": narration,
            "duration": duration,
            "props": "stone, sand",
            "stickers": [
                {
                    "id": "stk-001",
                    "visual_prompt": "isolated sticker of rough gray stone on pure white",
                    "trigger_words": ["stone"],
                    "appear_at": 0.0,
                    "entrance": "fade_in",
                    "idle": "float",
                    "position": "center",
                },
            ],
        }
    ]
    sync_sticker_appear_times(scenes, [duration])
    assert scenes[0]["stickers"][0]["appear_at"] == pytest.approx(stone_start, abs=0.05)


def test_word_chunk_overlay_fades_in():
    frames, _box = word_chunk_overlay_frames("HEAT THE LIME", 1.0, style="comic_punch_orange", fps=30)
    assert len(frames) >= 10
    early_alpha = frames[0].convert("RGBA").split()[-1].getextrema()[1]
    late_alpha = frames[-1].convert("RGBA").split()[-1].getextrema()[1]
    assert late_alpha >= early_alpha


def test_sticker_compositor_writes_video_with_motion(tmp_path):
    configure_motion(1.75)
    sticker_path = tmp_path / "stk-001.png"
    _white_sticker(sticker_path)
    stickers = [
        {
            "id": "stk-001",
            "appear_at": 0.0,
            "entrance": "fade_in",
            "idle": "float",
            "position": "center",
        }
    ]
    out = tmp_path / "scene.mp4"
    write_sticker_scene_video(stickers, {"stk-001": sticker_path}, 1.5, out, fps=30)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_narration_caption_cues_word_chunks():
    cues = narration_caption_cues("one two three four five six seven eight", 8.0, max_words=3)
    assert all(len(c.text.split()) <= 3 for c in cues)
    assert cues[0].start == 0.0
    assert cues[-1].end == pytest.approx(8.0)
