"""Phase 4: prove regenerate_scene() only redoes the ONE changed scene's
expensive steps (TTS + image gen) and reuses everything else on disk,
rather than silently doing a full re-render behind a misleading name."""
import hashlib
from pathlib import Path

import pytest

from shorts_factory.mascots import HOUSE_MASCOT_IDS
from shorts_factory.pipeline import regenerate_scene, run_pipeline


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_regenerate_scene_reuses_untouched_scenes_on_disk(tmp_path):
    full = run_pipeline("charcoal", artifacts_root=tmp_path)
    assert full.verification["overall_pass"] is True

    workdir = tmp_path / "charcoal" / "_work"
    untouched_audio = workdir / "audio" / "scene_01.wav"
    untouched_seg = workdir / "generated" / "segments" / "seg_01.mp4"
    assert untouched_audio.exists() and untouched_seg.exists()

    hash_before_audio = _sha256(untouched_audio)
    hash_before_seg = _sha256(untouched_seg)
    prior_cost_entries = len(full.cost_report["entries"])

    result = regenerate_scene(
        "charcoal", 0,
        new_narration="Charcoal starts as ordinary wood, heated without enough air to burn.",
        artifacts_root=tmp_path,
    )

    assert result.verification["overall_pass"] is True

    # The scene we DIDN'T touch must be byte-identical to before — proof this
    # wasn't a full re-render wearing a "regenerate one scene" label.
    assert _sha256(untouched_audio) == hash_before_audio
    assert _sha256(untouched_seg) == hash_before_seg

    # The edited scene's narration/caption actually changed in script.json.
    script = result.script
    assert script["scenes"][0]["narration"].startswith("Charcoal starts as ordinary wood")

    # Cost accumulates cumulatively (prior full run + this one regen call),
    # it doesn't get wiped and restarted. The regenerated scene always adds
    # exactly 1 new TTS entry + 1 new image entry (every scene renders its
    # own image — see pipeline._scene_base_image_path); no 3rd entry for the
    # hero image since the prior full run already generated and cached it.
    assert len(result.cost_report["entries"]) == prior_cost_entries + 2
    assert result.cost_report["total_spent_usd"] <= result.cost_report["budget_cap_usd"]


def test_regenerate_scene_requires_prior_full_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        regenerate_scene("charcoal-never-rendered-xyz", 0, artifacts_root=tmp_path)


def test_run_pipeline_persists_the_auto_selected_mascot_and_regenerate_reads_it_back(tmp_path, monkeypatch):
    """No manual mascot override exists anymore (removed 2026-08-28, per
    explicit user request) — the mascot is always resolved automatically
    via select_mascot_for_story(topic, brief=...), using the brief's
    concept/angle/claims text, not just the bare topic string. "charcoal"
    is one of the house mascots' keywords (MASCOT_STORY_KEYWORDS), so this
    resolves deterministically rather than needing a seeded random pick.
    regenerate_scene must then read that same mascot back from script.json
    (history), not re-resolve it."""
    import json
    from shorts_factory import pipeline

    calls = []
    real_select = pipeline.select_mascot_for_story

    def spy_select(topic, brief=None, seed=None):
        calls.append((topic, brief))
        return real_select(topic, brief=brief, seed=seed)

    monkeypatch.setattr(pipeline, "select_mascot_for_story", spy_select)

    full = run_pipeline("charcoal", artifacts_root=tmp_path)
    # A house mascot — see mascots.HOUSE_MASCOTS_ONLY. Which of the two is
    # keyword-dependent, so assert the group rather than pinning one.
    assert full.mascot_id in HOUSE_MASCOT_IDS
    assert len(calls) == 1
    assert calls[0][0] == "charcoal"
    assert calls[0][1] is not None  # the brief was passed, not just the bare topic

    script_file = tmp_path / "charcoal" / "charcoal.script.json"
    assert script_file.exists()
    script_data = json.loads(script_file.read_text(encoding="utf-8"))
    assert script_data.get("mascot_id") in HOUSE_MASCOT_IDS

    regen = regenerate_scene("charcoal", 0, artifacts_root=tmp_path)
    assert regen.mascot_id in HOUSE_MASCOT_IDS


def test_run_pipeline_generates_a_custom_mascot_when_nothing_matches(tmp_path, monkeypatch):
    """Regression test: when select_mascot_for_story() finds no match at
    all (returns None), run_pipeline must design and use a brand-new custom
    mascot (mascots.generate_custom_mascot) instead of crashing on
    `None.id` or silently forcing an unrelated mascot onto the story."""
    import json
    from shorts_factory import mascots, pipeline

    monkeypatch.setattr(pipeline, "select_mascot_for_story", lambda topic, brief=None, seed=None: None)

    design_calls = []
    real_generate = pipeline.generate_custom_mascot

    def spy_generate(topic, brief, llm, cost_tracker):
        design_calls.append(topic)
        return real_generate(topic, brief, llm, cost_tracker)

    monkeypatch.setattr(pipeline, "generate_custom_mascot", spy_generate)

    result = run_pipeline("charcoal", artifacts_root=tmp_path)
    assert design_calls == ["charcoal"]
    assert result.mascot_id == mascots.custom_mascot_slug("charcoal")

    script_data = json.loads((tmp_path / "charcoal" / "charcoal.script.json").read_text(encoding="utf-8"))
    assert script_data["mascot_id"] == result.mascot_id


def test_caption_style_persists_and_regenerate_reuses_it(tmp_path):
    """Regression test: caption_style is chosen once per video and must
    survive a single-scene regeneration unchanged — a regenerated scene
    must never end up with a different font/color/casing than the rest of
    the video just because regenerate_scene() re-ran a fresh random pick."""
    import json
    from shorts_factory.captions import STYLE_NAMES

    full = run_pipeline("charcoal", artifacts_root=tmp_path)
    script_file = tmp_path / "charcoal" / "charcoal.script.json"
    script_data = json.loads(script_file.read_text(encoding="utf-8"))
    assert script_data.get("caption_style") in STYLE_NAMES
    original_style = script_data["caption_style"]

    regen = regenerate_scene("charcoal", 0, artifacts_root=tmp_path)
    assert regen.script["caption_style"] == original_style



def test_yellow_topic_last_scenes_real_caption_survives(tmp_path):
    """Regression test: pipeline.py used to overwrite
    script["scenes"][-1]["caption"] with a fixed caution string for every
    yellow-classified topic, silently deleting the actual payoff line the
    LLM wrote. caution_text is now persisted separately and composited as
    an additional badge (see captions.draw_caution_badge), never assigned
    over the real caption."""
    import json

    full = run_pipeline("charcoal", artifacts_root=tmp_path)  # charcoal is yellow
    assert full.safety_class == "yellow"

    script_data = json.loads((tmp_path / "charcoal" / "charcoal.script.json").read_text(encoding="utf-8"))
    last_caption = script_data["scenes"][-1]["caption"]
    assert last_caption != "CAUTION: Educational overview — follow current expert safety guidance."
    assert script_data["caution_text"] == "CAUTION: Educational overview — follow current expert safety guidance."

    # Regenerating the last scene must still not clobber its real caption.
    regen = regenerate_scene(
        "charcoal", len(script_data["scenes"]) - 1,
        new_narration="A brand new payoff line about charcoal.",
        artifacts_root=tmp_path,
    )
    assert regen.script["scenes"][-1]["caption"] != "CAUTION: Educational overview — follow current expert safety guidance."
    assert "brand new payoff" in regen.script["scenes"][-1]["narration"].lower()


def test_regenerate_scene_resets_approval_to_pending(tmp_path):
    """Regression test: the final .mp4 changes every time a scene is
    regenerated, but review status (approved/scheduled) used to survive
    untouched — meaning a video could ship to YouTube reflecting an old,
    never-re-reviewed approval decision made about DIFFERENT content
    (confirmed real 2026-08-21 review)."""
    from shorts_factory.dashboard import review_state

    full = run_pipeline("charcoal", artifacts_root=tmp_path)
    artifacts_dir = tmp_path / "charcoal"
    review_state.approve(artifacts_dir, notes="looked good")
    assert review_state.load(artifacts_dir).status == "approved"

    regenerate_scene("charcoal", 0, new_narration="Something different now.", artifacts_root=tmp_path)

    assert review_state.load(artifacts_dir).status == "pending"
