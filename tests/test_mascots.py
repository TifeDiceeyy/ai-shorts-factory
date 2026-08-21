import json
import pytest

from shorts_factory.mascots import MASCOTS, get_mascot, list_mascots, select_mascot_for_story, Mascot
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


def test_select_mascot_for_story_matches_topic_keywords():
    # Direct topic keyword hits (see MASCOT_STORY_KEYWORDS) — deterministic
    # since a single-candidate match never reaches the random tie-break.
    assert select_mascot_for_story("soap").id == "mascot_4"
    assert select_mascot_for_story("charcoal").id == "mascot_4"
    assert select_mascot_for_story("roman concrete").id == "mascot_1"
    assert select_mascot_for_story("apple cider vinegar").id == "mascot_3"


def test_select_mascot_for_story_uses_brief_text_too_not_just_topic():
    # A generic/ambiguous topic string alone matches nothing, but the
    # brief's concept/angle/claims carry the real thematic signal — must be
    # searched too, not just the bare topic.
    brief = {
        "concept": "How Roman engineers made waterproof concrete",
        "angle": "an ancient engineering explainer",
        "claims": [{"claim": "Pozzolana ash was mixed with lime to bind aqueduct stone."}],
    }
    assert select_mascot_for_story("ancient engineering", brief=brief).id == "mascot_1"


def test_select_mascot_for_story_falls_back_to_random_for_unmatched_topics():
    # No keyword hits at all -> random among all 5, not a silent crash or a
    # hardcoded always-the-same-mascot fallback.
    chosen_ids = {select_mascot_for_story("xyzzy nonsense topic", seed=i).id for i in range(30)}
    assert chosen_ids.issubset(set(MASCOTS.keys()))
    assert len(chosen_ids) > 1, "expected variety across seeds when nothing matches"


def test_telegram_controller_mascots_text(tmp_path):
    controller = TelegramController((1,), tmp_path)
    text = controller.mascots_text()
    assert "Mascot 1" in text
    assert "Mascot 2" in text
    assert "Mascot 3" in text
    assert "Mascot 4" in text
    assert "Mascot 5" in text


def test_mascot_build_scene_prompt_split_canvas_and_centered():
    m = get_mascot("mascot_4")

    # Explainer with prop -> split canvas
    prompt_split = m.build_scene_prompt(
        scene_role="Chemical Artisan",
        action="stirring wood ash lye",
        emotion="wide curious eyes",
        props="bubbling glass beaker with amber lye",
        layout="split_bottom_left",
    )
    assert "Split-canvas" in prompt_split
    assert "bottom-left" in prompt_split
    assert "upper-right" in prompt_split
    assert "bubbling glass beaker" in prompt_split
    assert "white background" in prompt_split.lower()

    # Hook with hazard FX
    prompt_fx = m.build_scene_prompt(
        scene_role="Shocked Survivor",
        action="reacting in horror to caustic splash",
        emotion="wide eyes and open mouth",
        fx="green toxic chemical smoke",
        layout="centered",
        scene_type="mascot",
    )
    assert "green toxic chemical smoke" in prompt_fx
    assert "centered vertically in frame" in prompt_fx

    # Ingredient grid
    prompt_grid = m.build_scene_prompt(
        scene_type="ingredient_grid",
        grid_items=["Limestone rock", "Volcanic ash", "Water in bronze pot", "Crushed gravel"],
    )
    assert "ingredient recipe grid" in prompt_grid
    assert "Limestone rock" in prompt_grid
    assert "Volcanic ash" in prompt_grid

    # Process action
    prompt_action = m.build_scene_prompt(
        scene_type="process_action",
        action="pouring thick grey slurry from a clay bowl directly into a clamped wooden mold with rebar",
    )
    assert "process demonstration" in prompt_action
    assert "clamped wooden mold" in prompt_action


def test_get_scene_image_prompt_carries_the_scenes_action_through():
    """Regression test: pipeline.get_scene_image_prompt() reads
    scene.get("action", "") and build_scene_prompt()'s process_action
    branch uses it as the shot's main content — but neither the real LLM's
    system prompt (providers/llm.py) nor script.schema.json ever requested/
    allowed an "action" field, so it was always "" in practice and every
    process_action scene silently fell back to the generic hardcoded
    'pouring mixture into mold', regardless of what the scene was actually
    about. Confirmed by adding action to both and checking it now survives
    the full get_scene_image_prompt() call, not just a direct
    build_scene_prompt() call (already covered by
    test_mascot_build_scene_prompt_split_canvas_and_centered above)."""
    from shorts_factory.pipeline import get_scene_image_prompt
    from shorts_factory.schema_validate import validate_script_shape

    mascot = get_mascot("mascot_4")
    scene = {
        "narration": "Lye breaks down the fatty acid bonds.",
        "caption": "Lye breaks down the fatty acid bonds",
        "duration": 7.0,
        "visual_prompt": "placeholder, must be overridden by structured fields",
        "source_claim_id": "claim-01",
        "scene_type": "process_action",
        "action": "stirring the boiling cauldron with a long wooden paddle",
        "props": "cauldron, wooden paddle",
    }
    prompt = get_scene_image_prompt(scene, mascot)
    assert "stirring the boiling cauldron with a long wooden paddle" in prompt
    assert "pouring mixture into mold" not in prompt

    # And the field is schema-legal, not silently rejected by the strict
    # additionalProperties:false scene schema.
    validate_script_shape({
        "topic": "soap", "language": "English", "visual_style": "test",
        "scenes": [scene],
    })


def test_pipeline_records_chosen_mascot(tmp_path, monkeypatch):
    # Mock assembly.assemble to avoid needing external ffmpeg in quick unit tests
    from shorts_factory import assembly

    def fake_assemble(scenes, frame_source, audio, workdir, out_mp4, caption_style=None):
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
