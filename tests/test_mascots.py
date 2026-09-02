import json
import pytest

from shorts_factory.mascots import (
    DEFAULT_MASCOT_ID,
    HOUSE_MASCOTS_ONLY,
    HOUSE_MASCOT_IDS,
    MASCOTS,
    custom_mascot_slug,
    generate_custom_mascot,
    get_mascot,
    list_mascots,
    load_custom_mascots,
    register_custom_mascot,
    select_mascot_for_story,
    Mascot,
)
from shorts_factory.pipeline import run_pipeline


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
    assert get_mascot(None).id == DEFAULT_MASCOT_ID
    assert get_mascot("unknown_mascot").id == DEFAULT_MASCOT_ID


@pytest.mark.skipif(
    HOUSE_MASCOTS_ONLY,
    reason="HOUSE_MASCOTS_ONLY: the two house mascots carry every story, so themed "
           "keyword selection and the custom-mascot fallback are bypassed by design. "
           "This test describes the themed path and applies again if the flag is cleared.",
)
def test_select_mascot_for_story_matches_topic_keywords():
    # Direct topic keyword hits (see MASCOT_STORY_KEYWORDS) — deterministic
    # since a single-candidate match never reaches the random tie-break.
    assert select_mascot_for_story("soap").id == "mascot_4"
    assert select_mascot_for_story("charcoal").id == "mascot_4"
    assert select_mascot_for_story("roman concrete").id == "mascot_1"
    assert select_mascot_for_story("apple cider vinegar").id == "mascot_3"


@pytest.mark.skipif(
    HOUSE_MASCOTS_ONLY,
    reason="HOUSE_MASCOTS_ONLY: the two house mascots carry every story, so themed "
           "keyword selection and the custom-mascot fallback are bypassed by design. "
           "This test describes the themed path and applies again if the flag is cleared.",
)
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


@pytest.mark.skipif(
    HOUSE_MASCOTS_ONLY,
    reason="HOUSE_MASCOTS_ONLY: the two house mascots carry every story, so themed "
           "keyword selection and the custom-mascot fallback are bypassed by design. "
           "This test describes the themed path and applies again if the flag is cleared.",
)
def test_select_mascot_for_story_returns_none_when_nothing_matches_at_all():
    # No keyword hits at all -> None, signaling the caller (run_pipeline) to
    # generate a brand-new custom mascot instead of forcing an unrelated one
    # or silently defaulting. Not a crash, not a random pick among the 5.
    for seed in range(10):
        assert select_mascot_for_story("xyzzy nonsense topic", seed=seed) is None


@pytest.mark.skipif(
    HOUSE_MASCOTS_ONLY,
    reason="HOUSE_MASCOTS_ONLY: the two house mascots carry every story, so themed "
           "keyword selection and the custom-mascot fallback are bypassed by design. "
           "This test describes the themed path and applies again if the flag is cleared.",
)
def test_select_mascot_for_story_main_mascot_wins_ties():
    # Mascot 4 ("Main Mascot") wins ties among the 5 registered mascots
    # instead of a random pick — an exact tie: "soap" (mascot_4, +1, only in
    # the brief text) vs "roman" (mascot_1, +1, only in the brief text);
    # neither appears in the bare topic string, so neither gets the
    # topic-text double-count bonus, and the tie is genuine.
    brief = {"concept": "soap made near roman ruins", "angle": "", "claims": []}
    for seed in range(10):
        assert select_mascot_for_story("a history lesson", brief=brief, seed=seed).id == "mascot_4"


def _custom_mascot_design(topic: str = "deep sea diving") -> dict:
    return {
        "name": f"Mascot: {topic.title()} Explorer",
        "short_desc": f"A {topic}-themed explorer.",
        "hero_prompt": (
            "Full-body 3D CGI cartoon mascot, centered vertically occupying 60% of frame, "
            "fully clothed, no bare skin except face/forearms/calves, not shirtless or undressed, "
            "stark pure solid white background (#FFFFFF), zero shadows, sticker framing."
        ),
        "visual_style": "High-end 3D CGI cartoon render. Stark white background, no text.",
        "motion_instruction": "Describe dynamic actions and emotions for this mascot.",
        "scene_role_template": "Hook/Discovery/Process/Challenge/Payoff arc.",
        "keywords": ["diving", "submarine", "ocean", "deep", "sea"],
    }


def test_register_custom_mascot_persists_and_get_mascot_resolves_it(tmp_path, monkeypatch):
    import shorts_factory.mascots as mascots_module
    monkeypatch.setattr(mascots_module, "CUSTOM_MASCOT_REGISTRY_PATH", tmp_path / "custom_mascots.json")

    mascot_id = custom_mascot_slug("Deep Sea Diving!!")
    assert mascot_id == "mascot_custom_deep_sea_diving"

    registered = register_custom_mascot(mascot_id, _custom_mascot_design())
    assert registered.id == mascot_id
    assert registered.name == "Mascot: Deep Sea Diving Explorer"

    # Persisted to disk, not just held in memory.
    on_disk = load_custom_mascots()
    assert mascot_id in on_disk

    # get_mascot() must resolve it, not silently fall back to the default.
    resolved = get_mascot(mascot_id)
    assert resolved.id == mascot_id
    assert resolved.name == registered.name


def test_generate_custom_mascot_uses_llm_and_registers_it(tmp_path, monkeypatch):
    import shorts_factory.mascots as mascots_module
    from shorts_factory.providers.llm import StubLLMProvider
    from shorts_factory.cost_tracker import CostTracker
    monkeypatch.setattr(mascots_module, "CUSTOM_MASCOT_REGISTRY_PATH", tmp_path / "custom_mascots.json")

    llm = StubLLMProvider()
    tracker = CostTracker(budget_cap_usd=1.0)
    mascot = generate_custom_mascot("deep sea diving", None, llm, tracker)

    assert mascot.id == custom_mascot_slug("deep sea diving")
    on_disk = load_custom_mascots()
    assert mascot.id in on_disk
    assert on_disk[mascot.id]["keywords"]


@pytest.mark.skipif(
    HOUSE_MASCOTS_ONLY,
    reason="HOUSE_MASCOTS_ONLY: the two house mascots carry every story, so themed "
           "keyword selection and the custom-mascot fallback are bypassed by design. "
           "This test describes the themed path and applies again if the flag is cleared.",
)
def test_select_mascot_for_story_reuses_a_previously_generated_custom_mascot(tmp_path, monkeypatch):
    """Regression test: the whole point of persisting a custom mascot is
    that a FUTURE similar topic finds and reuses it via the same
    keyword-scoring path, instead of never being found again (which would
    mean paying to generate a brand-new one every single time)."""
    import shorts_factory.mascots as mascots_module
    monkeypatch.setattr(mascots_module, "CUSTOM_MASCOT_REGISTRY_PATH", tmp_path / "custom_mascots.json")

    mascot_id = custom_mascot_slug("deep sea diving")
    register_custom_mascot(mascot_id, _custom_mascot_design())

    found = select_mascot_for_story("submarine ocean exploration")
    assert found is not None
    assert found.id == mascot_id


def test_build_scene_prompt_expands_fear_emotions_into_a_vivid_expression():
    """Real gap found 2026-08-29: a bare 'Emotion: alarmed.' label
    under-rendered with the image model — a real generated frame's mascot
    didn't read as scared despite the script correctly asking for it.
    Fear/danger/shock emotion words must expand into an explicit physical
    expression description instead of passing the bare adjective through."""
    m = get_mascot("mascot_4")

    centered = m.build_scene_prompt(emotion="alarmed", layout="centered")
    assert "Emotion: alarmed." not in centered
    assert "wide terrified eyes" in centered

    split = m.build_scene_prompt(emotion="scared", layout="split_bottom_left", props="a rock")
    assert "wide terrified eyes" in split


def test_build_scene_prompt_leaves_non_fear_emotions_unchanged():
    m = get_mascot("mascot_4")
    centered = m.build_scene_prompt(emotion="proud", layout="centered")
    assert "Emotion: proud." in centered
    assert "wide terrified eyes" not in centered


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


def test_build_scene_motion_prompt_forbids_repeating_bounce():
    """User feedback 2026-08-28: the mascot should stay planted (tiny
    breath+blink only) and never perform a repeating bounce/hop/bob — the
    motion prompt sent to the video model must say so explicitly, for both
    mascot scenes and object-only scenes (ingredient_grid/process_action)."""
    m = get_mascot("mascot_4")

    mascot_prompt = m.build_scene_motion_prompt(
        scene_type="mascot", action="pointing forward", fx="green toxic chemical smoke",
    )
    assert "planted" in mascot_prompt
    assert "no hopping, bobbing, squash-and-stretch, or repeating idle bounce" in mascot_prompt
    assert "never a second bounce" in mascot_prompt
    assert "never a repeating springy idle" in mascot_prompt

    grid_prompt = m.build_scene_motion_prompt(scene_type="ingredient_grid", props="Limestone rock, Volcanic ash")
    assert "No bounce or wobble after each item's initial pop-in" in grid_prompt


def test_build_scene_motion_prompt_asks_for_object_fx_when_mascot_matches_a_category():
    """When the mascot is mostly still, the frame must stay alive through
    the PROPS/environment instead — generic keyword matching (not
    hardcoded to any one topic), covering volcano/ash, water/drip,
    lime/powder, and mix/slurry as explicitly named in the user's brief."""
    m = get_mascot("mascot_4")

    volcano = m.build_scene_motion_prompt(scene_type="mascot", fx="volcanic ash cloud")
    assert "volcanic ash drifting" in volcano

    water = m.build_scene_motion_prompt(scene_type="mascot", props="a dripping water jug")
    assert "water rippling, dripping, or pouring" in water

    lime = m.build_scene_motion_prompt(scene_type="mascot", props="a sack of lime powder")
    assert "powder/dust settling" in lime

    mix = m.build_scene_motion_prompt(scene_type="mascot", action="mixing the slurry with a paddle")
    assert "mixture slowly swirling and blending" in mix

    # No matching category and no props at all -> no fabricated object FX,
    # falls back to the mascot's own subtle motion only.
    plain = m.build_scene_motion_prompt(scene_type="mascot", action="nodding")
    assert "keep the frame alive through the props/environment" not in plain


def test_object_fx_style_for_distinguishes_flicker_from_drift():
    """New 2026-08-29 for the sticker-mode localized object-pulse feature:
    fire/spark categories get "flicker" (brightness alone is a reasonable
    fire metaphor); smoke/steam/water/dust/mix categories get "drift" (a
    real position-offset animation) since flat brightness oscillation
    doesn't read as "moving" for those — a real gap flagged by the user
    watching a bubbling-cauldron scene."""
    from shorts_factory.mascots import object_fx_style_for

    assert object_fx_style_for("the fire crackles") == "flicker"
    assert object_fx_style_for("sparks fly everywhere") == "flicker"
    assert object_fx_style_for("steam rises from the pot") == "drift"
    assert object_fx_style_for("water drips steadily") == "drift"
    assert object_fx_style_for("nothing relevant here") is None


def test_build_scene_motion_prompt_object_only_scenes_have_no_character():
    """ingredient_grid/process_action scenes never show the mascot (see
    build_scene_prompt) — the motion prompt must say so too, so the video
    model doesn't invent one, and must describe the equipment/materials as
    the thing carrying all the motion."""
    m = get_mascot("mascot_4")
    prompt = m.build_scene_motion_prompt(
        scene_type="process_action", action="pouring thick slurry into a wooden mold",
    )
    assert "No character present" in prompt
    assert "pouring thick slurry into a wooden mold" in prompt


def test_get_scene_motion_prompt_differs_from_the_still_image_prompt():
    """Regression test for the pipeline-wiring half of the fix: the motion
    prompt used for ai_video generation must be a genuinely different,
    dedicated prompt — not the still-image composition prompt reused
    verbatim (the prior bug: reusing it produced a bouncing mascot and
    completely frozen props, since that prompt only describes a static
    composition)."""
    from shorts_factory.pipeline import get_scene_image_prompt, get_scene_motion_prompt

    mascot = get_mascot("mascot_4")
    scene = {
        "narration": "The dwarf mixes lime powder into the bubbling cauldron.",
        "caption": "Mixing lime powder",
        "duration": 6.0,
        "scene_type": "mascot",
        "mascot_role": "chemist",
        "mascot_emotion": "focused",
        "action": "stirring the cauldron",
        "props": "lime powder, wooden paddle",
        "fx": None,
    }
    image_prompt = get_scene_image_prompt(scene, mascot)
    motion_prompt = get_scene_motion_prompt(scene, mascot)
    assert motion_prompt != image_prompt
    assert "planted" in motion_prompt
    assert "powder/dust settling" in motion_prompt
    # The still-image prompt never mentions the bounce/planted rules at all.
    assert "planted" not in image_prompt


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

    def fake_assemble(scenes, frame_source, audio, workdir, out_mp4, caption_style=None, caution_text=None, subscribe_cta_text=None):
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

    # No manual mascot override exists anymore (removed 2026-08-28, per
    # explicit user request) — the mascot is always resolved automatically
    # via select_mascot_for_story(). "soap" is one of mascot_4's own
    # keywords (MASCOT_STORY_KEYWORDS), so it deterministically resolves to
    # mascot_4 rather than needing a seeded random pick.
    result = run_pipeline("soap", artifacts_root=tmp_path)
    # A house mascot, not a themed one — see HOUSE_MASCOTS_ONLY. Which of
    # the two depends on the topic's keyword lean, so assert the group.
    assert result.mascot_id in HOUSE_MASCOT_IDS
    assert (tmp_path / "soap" / "soap.script.json").exists()


def test_every_mascot_image_prompt_demands_a_closed_mouth():
    """The lip-sync fix lives in the IMAGE prompt, not the video prompt.

    Root cause found 2026-09-01: the generated hero and scene images carried
    an open, teeth-showing grin, and Kling was then asked to animate them.
    In a character close-up the mouth is the most animatable thing in frame,
    so it moved, and the mascot appeared to lip-sync the narration. Two
    earlier fixes only added wording to the Kling prompt and both failed.
    The mouth has to be shut in the source image so there is nothing there
    to animate — on the hero AND on every mascot scene, since scenes are
    edited from the hero but re-state the expression themselves.
    """
    from shorts_factory.mascots import CLOSED_MOUTH_RULE

    for mascot in MASCOTS.values():
        assert CLOSED_MOUTH_RULE in mascot.hero_image_prompt, f"{mascot.id} hero"
        for layout in ("centered", "split_bottom_left"):
            prompt = mascot.build_scene_prompt(
                scene_role="explainer", emotion="curious", layout=layout
            )
            assert CLOSED_MOUTH_RULE in prompt, f"{mascot.id} {layout}"


def test_closed_mouth_rule_leads_the_prompt_and_avoids_bare_negation():
    """Two real generations were lost to prompt mechanics, not to intent.

    The rule was first APPENDED after the long style/background clauses and
    was ignored outright — the model rendered the usual open grin. It was
    also phrased as negations ("no open mouth, no teeth showing"), which
    image models notoriously render rather than omit. Front-loading it and
    restating it as a positive drawing instruction ("drawn as one simple
    thin closed curved line") is what actually produced a closed mouth.
    """
    from shorts_factory.mascots import CLOSED_MOUTH_RULE

    assert CLOSED_MOUTH_RULE.startswith("Draw the mouth as"), (
        "the rule must lead with what to DRAW, not with what to omit"
    )
    # The rule is short enough to stay effective without leading the whole
    # prompt. It leads the HERO prompt (one character, nothing competing);
    # in a scene prompt the SUBJECT leads, because burying what the picture
    # is actually of behind boilerplate is what made shots generic.
    assert len(CLOSED_MOUTH_RULE.split()) <= 20, "rule must stay compact"
    for mascot in MASCOTS.values():
        assert mascot.hero_image_prompt.startswith(CLOSED_MOUTH_RULE), f"{mascot.id}"
        assert CLOSED_MOUTH_RULE in mascot.build_scene_prompt(scene_role="x")


def test_fear_scenes_do_not_ask_for_a_grimace():
    """A cartoon grimace is drawn with bared, gritted teeth.

    The fear expansion used to end in a "worried grimace" and a real
    generation rendered exactly that — defeating the closed-mouth rule on
    the alarmed/shocked scenes that open most videos.
    """
    from shorts_factory.mascots import _FEAR_EMOTION_KEYWORDS, _expand_emotion

    for keyword in _FEAR_EMOTION_KEYWORDS:
        expanded = _expand_emotion(keyword).lower()
        for banned in ("grimace", "teeth", "gasp", "open mouth", "shout", "scream"):
            assert banned not in expanded, f"{keyword!r} expansion still says {banned!r}"


def test_text_dependent_props_get_a_pictorial_instruction():
    """Asking for a timeline while forbidding text is a contradiction.

    Measured 2026-09-02: 8 of 73 scenes across all scripts request a visual
    that only conveys meaning through writing ("timeline graphic",
    "three-field rotation diagram", "blast furnace diagram") in a prompt
    that also says "do not render any text, words, letters, labels, or
    signs". The model resolves that by drawing a meaningless graphic shape
    — the "scenes out of concept" failure. The no-text rule stays (dropping
    it puts unreadable invented lettering back on screen); the prompt
    instead gains a positive instruction for depicting the idea in pictures.
    """
    mascot = get_mascot("mascot_1")
    prompt = mascot.build_scene_prompt(
        scene_role="Discovery", emotion="thoughtful",
        props="timeline graphic", layout="split_bottom_left",
    )
    assert "pictures only" in prompt
    assert "row of small illustrated scenes joined by arrows" in prompt


def test_ordinary_physical_props_are_left_alone():
    """The hint must not fire on props that are already drawable, or every
    prompt grows for no reason."""
    from shorts_factory.mascots import _no_text_depiction_hint

    for ordinary in ("a clay pot", "volcanic ash and lime", "a bronze helmet", "seawater"):
        assert _no_text_depiction_hint(ordinary) == [], ordinary
    for texty in ("timeline graphic", "a blast furnace diagram", "floating list of items"):
        assert _no_text_depiction_hint(texty), texty


def test_split_canvas_role_and_emotion_are_grammatical():
    """Interpolating the raw fields mid-sentence produced "stands looking
    and pointing up with thoughtful as Discovery" — the values are
    adjectives and story-beat names, not sentence fragments, and the fear
    expansion returns a whole multi-clause phrase that broke the sentence
    outright."""
    mascot = get_mascot("mascot_1")
    for emotion in ("thoughtful", "alarmed"):
        prompt = mascot.build_scene_prompt(
            scene_role="Discovery", emotion=emotion, layout="split_bottom_left",
        )
        assert "up with " not in prompt, "role/emotion must not be spliced mid-sentence"
        assert "Role: Discovery." in prompt
        assert "Emotion: " in prompt


def test_the_character_is_never_placed_twice():
    """A real render came back with TWO mascots in one frame (2026-09-02).

    The script's own visual_prompt already says where the character is
    ("The Red-Cap Elder stands centered"), and the layout template then
    placed him again somewhere else ("in the bottom-left quadrant"). The
    prompt asked for the mascot seven times in conflicting positions and
    the model drew him twice.
    """
    mascot = get_mascot("mascot_6")
    authored = "The Red-Cap Elder stands centered, demonstrating with his stick"

    for layout in ("split_bottom_left", "centered"):
        prompt = mascot.build_scene_prompt(
            scene_role="Process", emotion="demonstrative", layout=layout,
            props="a rotating coil", subject=authored,
        )
        assert "exactly ONE character" in prompt, layout
        assert "stands looking and pointing up" not in prompt, (
            f"{layout}: the template placed the character a second time"
        )

    # With no authored description the template MUST still place him,
    # otherwise a character-free prompt would render an empty frame.
    plain = mascot.build_scene_prompt(scene_role="Process", layout="split_bottom_left")
    assert "quadrant" in plain and "exactly ONE character" not in plain


def test_character_detection_covers_names_and_possessives():
    """The first version of this list had neither "elder" nor any
    possessive, so a description naming the character read as
    character-free and the double-placement bug slipped through."""
    from shorts_factory.mascots import _mentions_character

    for text in (
        "The Red-Cap Elder stands centered",
        "demonstrating with his stick",
        "the presenter points at it",
        "she raises a hand",
        "their hands are visible",
    ):
        assert _mentions_character(text), text

    for text in (
        "A rotating coil of wire with a magnet inside",
        "A simple kiln with limestone entering",
        "four ingredients arranged in a grid",
    ):
        assert not _mentions_character(text), text
