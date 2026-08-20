import json
import pytest

from shorts_factory.mascots import MASCOTS, get_mascot, list_mascots, Mascot
from shorts_factory.pipeline import run_pipeline
from shorts_factory.telegram_bot import TelegramController


def test_all_five_mascots_registered_and_structured():
    mascots = list_mascots()
    assert len(mascots) == 5
    for i, m in enumerate(mascots, start=1):
        assert isinstance(m, Mascot)
        assert m.id == f"mascot_{i}"
        assert f"Mascot {i}" in m.name
        assert len(m.short_desc) > 5
        assert len(m.hero_prompt) > 20
        assert "white background" in m.hero_prompt.lower()
        assert len(m.visual_style) > 10
        assert len(m.motion_instruction) > 10
        assert len(m.scene_role_template) > 10


def test_scene_adaptive_script_schema_validation():
    from shorts_factory.schema_validate import validate_script_shape

    script = {
        "topic": "soap",
        "language": "English",
        "visual_style": "3D cartoon on white background",
        "scenes": [
            {
                "narration": "What if civilization collapsed tomorrow? How would you make soap?",
                "caption": "How to make soap if society ends",
                "duration": 8.0,
                "visual_prompt": "3D Tinkerer mascot with wide shocked eyes reacting to muddy hands",
                "source_claim_id": "claim-01",
                "camera": "static wide shot",
                "sfx": "whoosh",
                "mascot_role": "Shocked Survivor",
                "mascot_emotion": "astonished gasp with wide eyes",
                "props": "muddy bowl",
            },
            {
                "narration": "Ancient people boiled animal fat and mixed it with wood ash.",
                "caption": "Boil fat with wood ash",
                "duration": 9.0,
                "visual_prompt": "3D Tinkerer mascot in artisan apron stirring wood ash into boiling cauldron",
                "source_claim_id": "claim-02",
                "camera": "close-up",
                "sfx": "sizzle",
                "mascot_role": "Ancient Alchemist",
                "mascot_emotion": "focused determination",
                "props": "wooden paddle and smoking cauldron",
            },
            {
                "narration": "Wood ash contains potassium hydroxide, which turns oil into soap.",
                "caption": "Potassium hydroxide reacts with oil",
                "duration": 9.0,
                "visual_prompt": "3D Tinkerer mascot wearing brass loupe holding a clear beaker of bubbling lye",
                "source_claim_id": "claim-03",
                "camera": "overhead",
                "sfx": "bubbling",
                "mascot_role": "Master Chemist",
                "mascot_emotion": "curious excitement",
                "props": "steaming glass retort",
            },
            {
                "narration": "Pour the mixture into a wooden mold and let it cure for four weeks.",
                "caption": "Pour into mold and let cure",
                "duration": 9.0,
                "visual_prompt": "3D Tinkerer mascot pouring thick creamy soap mixture into wooden mold",
                "source_claim_id": "claim-04",
                "camera": "close-up",
                "sfx": "tap",
                "mascot_role": "Workshop Artisan",
                "mascot_emotion": "steady careful focus",
                "props": "wooden rectangular mold and ladle",
            },
            {
                "narration": "And just like that, you have real antimicrobial soap that saved millions.",
                "caption": "Real antimicrobial soap",
                "duration": 9.0,
                "visual_prompt": "3D Tinkerer mascot smiling proudly, holding up a sparkling clean bar of soap",
                "source_claim_id": "claim-05",
                "camera": "push-in",
                "sfx": "ding",
                "mascot_role": "Triumphant Creator",
                "mascot_emotion": "beaming victory smile with sparkle",
                "props": "stamped clean soap bar and lather bubbles",
            },
        ],
    }
    validate_script_shape(script)  # must not raise schema ValidationError


def test_get_mascot_aliases_and_fallback():
    assert get_mascot("mascot_1").id == "mascot_1"
    assert get_mascot("mascot 1").id == "mascot_1"
    assert get_mascot("1").id == "mascot_1"
    assert get_mascot("mascot_3").id == "mascot_3"
    assert get_mascot("3").id == "mascot_3"
    assert get_mascot("mascot_5").id == "mascot_5"
    assert get_mascot("5").id == "mascot_5"
    # Fallback to default Mascot 4
    assert get_mascot(None).id == "mascot_4"
    assert get_mascot("unknown_mascot").id == "mascot_4"


def test_telegram_controller_mascots_text(tmp_path):
    controller = TelegramController((1,), tmp_path)
    text = controller.mascots_text()
    assert "Mascot 1" in text
    assert "Mascot 2" in text
    assert "Mascot 3" in text
    assert "Mascot 4" in text
    assert "Mascot 5" in text


def test_pipeline_records_chosen_mascot(tmp_path, monkeypatch):
    # Mock assembly.assemble to avoid needing external ffmpeg in quick unit tests
    from shorts_factory import assembly

    def fake_assemble(scenes, frame_source, audio, workdir, out_mp4):
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.write_bytes(b"dummy_mp4")
        return {"caption_boxes": []}

    def fake_synthesize(tts_provider, scenes, audio_dir, cost_tracker):
        return [
            assembly.SceneAudio(path=audio_dir / "s0.wav", duration=5.0, scripted_duration=5.0)
            for _ in scenes
        ]

    def fake_verify(mp4_path, scripted_total_seconds, captions_meta_path, cost_report_path, budget_cap_usd, artifacts_dir):
        return {"overall_pass": True}

    monkeypatch.setattr(assembly, "assemble", fake_assemble)
    monkeypatch.setattr(assembly, "synthesize_scenes", fake_synthesize)
    from shorts_factory import verify
    monkeypatch.setattr(verify, "run_verification", fake_verify)

    result = run_pipeline("soap", artifacts_root=tmp_path, mascot_id="mascot_2")
    assert result.mascot_id == "mascot_2"
    assert (tmp_path / "soap" / "soap.script.json").exists()
