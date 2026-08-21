"""Regression coverage for per-scene edge-silence trimming: user feedback
was that there's a noticeable pause between scenes. Root cause, confirmed by
measuring real ElevenLabs narration audio (2026-08-21): each independently-
synthesized scene clip carries its own leading/trailing near-silence, so
concatenating scene i's trailing silence directly against scene i+1's
leading silence produces a longer, more noticeable pause at every cut than
either clip has on its own. build_scene_audio() now trims that edge silence
(leaving a small buffer) right after synthesis, before the clip's duration
is measured and used to drive that scene's video-clip length."""
import subprocess

import pytest

from shorts_factory import assembly
from shorts_factory.media_probe import probe_duration


def _silence_tone_silence_wav(path, leading=0.3, tone=1.0, trailing=2.0):
    """Builds leading_silence + tone + trailing_silence as one real WAV —
    a faithful stand-in for a real TTS clip's dead-air-padded shape, not a
    faked/mocked duration."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
            "-filter_complex",
            f"[0]atrim=0:{leading}[s1];[1]atrim=0:{tone}[tone];[0]atrim=0:{trailing}[s2];"
            "[s1][tone][s2]concat=n=3:v=0:a=1[out]",
            "-map", "[out]",
            str(path),
        ],
        check=True,
    )


def test_trim_edge_silence_removes_leading_and_trailing_dead_air(tmp_path):
    path = tmp_path / "scene_00.wav"
    _silence_tone_silence_wav(path, leading=0.3, tone=1.0, trailing=2.0)
    before = probe_duration(path)
    assert before > 3.0  # 0.3 + 1.0 + 2.0

    assembly._trim_edge_silence(path)

    after = probe_duration(path)
    # Real spoken content (the 1s tone) survives, plus the small buffer left
    # at each edge — nowhere close to the original 3.3s of mostly dead air.
    assert 1.0 <= after <= 1.5, f"expected ~1.0-1.5s after trimming, got {after}"


def test_trim_edge_silence_is_a_noop_on_continuous_signal(tmp_path):
    """StubTTSProvider synthesizes one continuous tone per scene, sized
    exactly to the scripted duration — the trim must never shrink that,
    or every stub-driven test's scene timing would silently drift."""
    path = tmp_path / "scene_00.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
            str(path),
        ],
        check=True,
    )
    before = probe_duration(path)
    assembly._trim_edge_silence(path)
    after = probe_duration(path)
    assert after == pytest.approx(before, abs=0.05)


def test_build_scene_audio_actually_wires_in_the_trim(tmp_path):
    """Regression test for the wiring, not just the standalone trim
    function: build_scene_audio() must call _trim_edge_silence() on
    whatever the TTS provider hands back, not just have the function exist
    unused somewhere in the module."""
    from shorts_factory.cost_tracker import CostTracker
    from shorts_factory.providers.tts import TTSProvider

    class PaddedFakeTTSProvider(TTSProvider):
        name = "fake"

        def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
            _silence_tone_silence_wav(out_path, leading=0.3, tone=1.0, trailing=2.0)
            return out_path

    result_path = assembly.build_scene_audio(
        PaddedFakeTTSProvider(), {"narration": "x", "caption": "x", "duration": 3.3}, 0,
        tmp_path, CostTracker(budget_cap_usd=1.0),
    )
    assert probe_duration(result_path) <= 1.5
