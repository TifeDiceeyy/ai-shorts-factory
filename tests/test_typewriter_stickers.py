"""Typewriter lyrics animation and layered sticker compositor tests."""
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from shorts_factory import assembly
from shorts_factory.assembly import SceneAudio
from shorts_factory.captions import FRAME_HEIGHT, FRAME_WIDTH, typewriter_overlay_frames
from shorts_factory.providers.tts import StubTTSProvider
from shorts_factory.sticker_compositor import render_sticker_scene_frame


def _sticker_png(tmp_path: Path, color=(200, 80, 60), name: str = "sticker.png") -> Path:
    path = tmp_path / name
    img = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
    draw.rectangle((cx - 120, cy - 120, cx + 120, cy + 120), fill=color)
    img.save(path)
    return path


def test_typewriter_frames_reveal_more_characters_over_time():
    frames, _box = typewriter_overlay_frames("HELLO WORLD", 2.0, style="comic_punch_orange", fps=30)
    assert len(frames) > 5
    early_alpha = frames[2].convert("RGBA").split()[-1]
    late_alpha = frames[-2].convert("RGBA").split()[-1]
    assert early_alpha.getbbox() is not None
    assert late_alpha.getbbox() is not None
    assert ImageChops.difference(frames[2], frames[-2]).getbbox() is not None


def test_sticker_scene_frame_has_motion_after_entrance(tmp_path):
    sticker = {
        "id": "stk-001",
        "appear_at": 0.0,
        "entrance": "fade_in",
        "idle": "float",
        "position": "center",
    }
    path = _sticker_png(tmp_path)
    early = render_sticker_scene_frame([sticker], {"stk-001": path}, 0.5)
    later = render_sticker_scene_frame([sticker], {"stk-001": path}, 2.0)
    region = (0, FRAME_HEIGHT // 3, FRAME_WIDTH, 2 * FRAME_HEIGHT // 3)
    assert ImageChops.difference(early.crop(region), later.crop(region)).getbbox() is not None


def test_assemble_stickers_with_layered_stickers(tmp_path):
    scenes = [
        {
            "narration": "Could you make a metal tool from scratch?",
            "caption": "Could you?",
            "duration": 4.0,
            "stickers": [
                {
                    "id": "stk-001",
                    "visual_prompt": "rock sticker",
                    "appear_at": 0.0,
                    "entrance": "fade_in",
                    "idle": "float",
                    "position": "center",
                },
                {
                    "id": "stk-002",
                    "visual_prompt": "splash sticker",
                    "appear_at": 1.0,
                    "entrance": "slide_up",
                    "idle": "drift",
                    "position": "bottom_left",
                },
            ],
        },
        {
            "narration": "Ancient humans used smelting to purify ore.",
            "caption": "Smelting!",
            "duration": 4.0,
            "stickers": [
                {
                    "id": "stk-003",
                    "visual_prompt": "furnace sticker",
                    "appear_at": 0.0,
                    "entrance": "fade_in",
                    "idle": "flicker",
                    "position": "center",
                },
                {
                    "id": "stk-004",
                    "visual_prompt": "ore sticker",
                    "appear_at": 1.2,
                    "entrance": "fade_in",
                    "idle": "float",
                    "position": "top_right",
                },
            ],
        },
    ]
    tracker = __import__("shorts_factory.cost_tracker", fromlist=["CostTracker"]).CostTracker(1.0)
    audio = assembly.synthesize_scenes(StubTTSProvider(), scenes, tmp_path / "audio", tracker)
    sticker_paths = {
        "stk-001": _sticker_png(tmp_path, color=(200, 80, 60), name="stk-001.png"),
        "stk-002": _sticker_png(tmp_path, color=(80, 120, 200), name="stk-002.png"),
        "stk-003": _sticker_png(tmp_path, color=(120, 200, 80), name="stk-003.png"),
        "stk-004": _sticker_png(tmp_path, color=(200, 120, 80), name="stk-004.png"),
    }
    out_mp4 = tmp_path / "layered.mp4"
    result = assembly.assemble_stickers(
        scenes,
        lambda i, scene: sticker_paths["stk-001"],
        audio,
        tmp_path / "work",
        out_mp4,
        sticker_image_paths=sticker_paths,
        caption_animation_mode="typewriter",
    )
    assert out_mp4.exists()
    assert len(result["caption_boxes"]) == 2
