"""Integration-level tests against the real run_pipeline() entrypoint."""
import pytest

from shorts_factory.pipeline import run_pipeline


def test_animate_with_stub_image_refuses(tmp_path, monkeypatch):
    """A real hero image is required to animate — silently animating a
    meaningless stub gradient would waste real video-generation spend on
    nothing worth looking at."""
    monkeypatch.setenv("VIDEO_PROVIDER", "fal")
    monkeypatch.setenv("VIDEO_MODEL", "fal-ai/minimax/hailuo-02/standard/image-to-video")
    monkeypatch.setenv("IMAGE_PROVIDER", "stub")
    monkeypatch.setenv("BUDGET_CAP_USD", "20")
    with pytest.raises(ValueError, match="IMAGE_PROVIDER is stub"):
        run_pipeline("soap", artifacts_root=tmp_path)


def test_gunpowder_blocked_with_no_artifacts_created(tmp_path):
    """Adversarial check: a RED topic must be blocked before brief/script/
    render/provider calls, not just flagged after the fact."""
    artifacts_dir = tmp_path / "gunpowder"

    result = run_pipeline("gunpowder", artifacts_root=tmp_path)

    assert result.blocked is True
    assert result.script is None
    assert result.verification is None
    assert not artifacts_dir.exists(), "a blocked topic must not create any artifacts"


def test_soap_full_pipeline_passes_verification(tmp_path):
    """The real end-to-end acceptance proof: run the actual Phase 0 skeleton
    for the soap topic and check every criterion, not just that it didn't crash."""
    result = run_pipeline("soap", artifacts_root=tmp_path)

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


def test_tts_is_charged_once_per_scene_not_once_per_render_stage(tmp_path):
    """Regression test: the pipeline renders each scene twice (placeholder
    stage, then generated-image stage), but narration audio must be
    synthesized ONCE and reused — a real paid TTS provider must not be
    charged twice for identical narration just because the video renders
    twice. Caught by adversarial review; fixed by sharing audio_paths
    across both assembly() calls instead of letting each one drive its own
    TTS pass."""
    result = run_pipeline("soap", artifacts_root=tmp_path)
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
