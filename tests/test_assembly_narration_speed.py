"""Regression coverage for narration speed-up: user feedback (2026-08-28)
was that the narration reads a bit slow across the whole video. Applied as
a pitch-preserving ffmpeg `atempo` pass (assembly._apply_narration_speed),
wired into build_scene_audio() right after the existing edge-silence trim —
NOT an ElevenLabs API `speed` param, since fal.ai's eleven-v3 endpoint
schema doesn't reliably expose one."""
import subprocess

import pytest

from shorts_factory import assembly
from shorts_factory.media_probe import probe_duration


def _tone_wav(path, duration=3.0, freq=440):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={duration}:sample_rate=48000",
            str(path),
        ],
        check=True,
    )


def test_apply_narration_speed_shortens_audio_by_the_configured_factor(tmp_path):
    path = tmp_path / "scene_00.wav"
    _tone_wav(path, duration=6.0)
    before = probe_duration(path)

    assembly._apply_narration_speed(path)

    after = probe_duration(path)
    expected = before / assembly.NARRATION_SPEED_FACTOR
    assert after == pytest.approx(expected, abs=0.15)


def test_apply_narration_speed_preserves_sample_rate_not_a_naive_resample(tmp_path):
    """A naive resample-based speedup would also raise pitch (the
    "chipmunk" effect) — ffmpeg's atempo is a dedicated tempo-only filter
    that keeps the sample rate (and thus pitch) unchanged. Confirmed via
    ffprobe on real output, not assumed from the filter's name alone."""
    path = tmp_path / "scene_00.wav"
    _tone_wav(path, duration=3.0)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    before_rate = probe.stdout.strip()

    assembly._apply_narration_speed(path)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    after_rate = probe.stdout.strip()
    assert after_rate == before_rate


def test_apply_narration_speed_is_a_noop_at_factor_one(tmp_path):
    path = tmp_path / "scene_00.wav"
    _tone_wav(path, duration=2.0)
    before = probe_duration(path)
    before_mtime = path.stat().st_mtime_ns

    assembly._apply_narration_speed(path, factor=1.0)

    assert probe_duration(path) == pytest.approx(before, abs=0.01)
    assert path.stat().st_mtime_ns == before_mtime, "factor=1.0 must skip the ffmpeg pass entirely, not run a no-op re-encode"


def test_build_scene_audio_wires_in_the_speed_up(tmp_path):
    """Regression test for the wiring, not just the standalone function:
    build_scene_audio() must call _apply_narration_speed() on whatever
    comes back from TTS (after the existing silence trim), not just have
    the function exist unused somewhere in the module."""
    from shorts_factory.cost_tracker import CostTracker
    from shorts_factory.providers.tts import TTSProvider

    class FakeTTSProvider(TTSProvider):
        name = "fake"

        def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
            _tone_wav(out_path, duration=6.0)
            return out_path

    result_path = assembly.build_scene_audio(
        FakeTTSProvider(), {"narration": "x", "caption": "x", "duration": 6.0}, 0,
        tmp_path, CostTracker(budget_cap_usd=1.0),
    )
    after = probe_duration(result_path)
    # ~6.0s tone, minus a hair from edge-silence trim's own tiny buffer
    # effect (negligible on a continuous tone), then divided by the speed
    # factor — should land well under the original 6.0s.
    assert after < 6.0 / assembly.NARRATION_SPEED_FACTOR + 0.2
