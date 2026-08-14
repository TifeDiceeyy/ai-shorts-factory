"""Integration-level tests against the real run_pipeline() entrypoint."""
from pathlib import Path

from shorts_factory.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gunpowder_blocked_with_no_artifacts_created():
    """Adversarial check: a RED topic must be blocked before brief/script/
    render/provider calls, not just flagged after the fact."""
    artifacts_dir = REPO_ROOT / "artifacts" / "gunpowder"
    assert not artifacts_dir.exists(), "test precondition: no leftover artifacts from a prior run"

    result = run_pipeline("gunpowder")

    assert result.blocked is True
    assert result.script is None
    assert result.verification is None
    assert not artifacts_dir.exists(), "a blocked topic must not create any artifacts"


def test_soap_full_pipeline_passes_verification():
    """The real end-to-end acceptance proof: run the actual Phase 0 skeleton
    for the soap topic and check every criterion, not just that it didn't crash."""
    result = run_pipeline("soap")

    assert result.blocked is False
    assert result.safety_class == "yellow"
    assert result.script is not None
    assert result.verification is not None
    assert result.verification["overall_pass"] is True, result.verification["checks"]

    mp4 = result.artifacts_dir / "soap.mp4"
    assert mp4.exists()
    assert (result.artifacts_dir / "soap.script.json").exists()
    assert (result.artifacts_dir / "cost-report.json").exists()
    assert (result.artifacts_dir / "verification-report.json").exists()
    assert (result.artifacts_dir / "captions.srt").exists()

    assert result.cost_report["total_spent_usd"] <= result.cost_report["budget_cap_usd"]


def test_tts_is_charged_once_per_scene_not_once_per_render_stage():
    """Regression test: the pipeline renders each scene twice (placeholder
    stage, then generated-image stage), but narration audio must be
    synthesized ONCE and reused — a real paid TTS provider must not be
    charged twice for identical narration just because the video renders
    twice. Caught by adversarial review; fixed by sharing audio_paths
    across both assembly() calls instead of letting each one drive its own
    TTS pass."""
    result = run_pipeline("soap")
    scene_count = len(result.script["scenes"])

    entries = result.cost_report["entries"]
    tts_entries = [e for e in entries if e["operation"].startswith("tts.synthesize_scene")]
    image_entries = [e for e in entries if e["operation"].startswith("image.generate_scene_image")]
    llm_entries = [e for e in entries if e["operation"] == "llm.generate_script"]

    assert len(tts_entries) == scene_count, (
        f"expected exactly {scene_count} TTS calls (one per scene), got {len(tts_entries)} "
        "— narration is being re-synthesized per render stage"
    )
    assert len(image_entries) == scene_count
    assert len(llm_entries) == 1
    assert len(entries) == scene_count * 2 + 1
