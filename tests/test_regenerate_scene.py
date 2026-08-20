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
    # it doesn't get wiped and restarted.
    assert len(result.cost_report["entries"]) == prior_cost_entries + 2  # +1 TTS +1 image for the one scene
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

