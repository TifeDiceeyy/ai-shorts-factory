"""Phase 4: prove regenerate_scene() only redoes the ONE changed scene's
expensive steps (TTS + image gen) and reuses everything else on disk,
rather than silently doing a full re-render behind a misleading name."""
import hashlib
from pathlib import Path

import pytest

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


def test_run_pipeline_persists_mascot_and_regenerate_uses_it(tmp_path):
    import json
    # Run pipeline with explicit non-default mascot
    full = run_pipeline("charcoal", mascot_id="mascot_2", artifacts_root=tmp_path)
    assert full.mascot_id == "mascot_2"

    script_file = tmp_path / "charcoal" / "charcoal.script.json"
    assert script_file.exists()
    script_data = json.loads(script_file.read_text(encoding="utf-8"))
    assert script_data.get("mascot_id") == "mascot_2"

    # Regenerate scene and assert it picks up mascot_2 from script.json
    regen = regenerate_scene("charcoal", 0, artifacts_root=tmp_path)
    assert regen.mascot_id == "mascot_2"


def test_run_pipeline_picks_mascot_via_story_matching_when_none_given(tmp_path, monkeypatch):
    """Regression test: an explicit mascot_id must skip story-matching
    entirely; leaving it unset must NOT silently fall back to a fixed
    default — it must go through select_mascot_for_story(topic, brief=...)."""
    from shorts_factory import pipeline

    calls = []
    real_select = pipeline.select_mascot_for_story

    def spy_select(topic, brief=None, seed=None):
        calls.append((topic, brief))
        return real_select(topic, brief=brief, seed=seed)

    monkeypatch.setattr(pipeline, "select_mascot_for_story", spy_select)

    run_pipeline("charcoal", artifacts_root=tmp_path)
    assert len(calls) == 1
    assert calls[0][0] == "charcoal"
    assert calls[0][1] is not None  # the brief was passed, not just the bare topic

    calls.clear()
    run_pipeline("charcoal", mascot_id="mascot_2", artifacts_root=(tmp_path / "explicit"))
    assert calls == [], "an explicit mascot_id must not trigger story-matching at all"


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

