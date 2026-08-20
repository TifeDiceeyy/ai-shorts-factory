"""Regression coverage for the post-mux loudness correction: the two-pass
loudnorm gets the pre-mux WAV to essentially exact -14.00 LUFS, but
mux_final's AAC re-encode (lossy, 160kbps) can shift the FINAL container's
MEASURED integrated loudness outside the +/-1 LU tolerance verify.py
actually checks against — confirmed for real: a WAV measured exactly
-14.00 LUFS produced a final .mp4 measuring -15.08 to -15.29. concat_and_mux
must measure and correct against the final container, not just the WAV.
"""
import subprocess

from shorts_factory import assembly


def _make_inputs(tmp_path, color: str):
    seg = tmp_path / "seg_00.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=2",
            "-pix_fmt", "yuv420p", str(seg),
        ],
        check=True,
    )
    audio = tmp_path / "a_00.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-ar", "48000", str(audio),
        ],
        check=True,
    )
    return seg, audio


def test_concat_and_mux_corrects_final_loudness_after_aac_shift(tmp_path, monkeypatch):
    seg, audio = _make_inputs(tmp_path, "red")

    # Real mux_final, wrapped to also apply a real, deliberate -1.3dB
    # attenuation to the audio on every call — a faithful (not just faked)
    # stand-in for what AAC encoding actually does: every real muxed output,
    # including the post-correction remux, experiences it, so the test
    # proves the correction genuinely compensates rather than just reading
    # a lied-about number.
    real_mux_final = assembly.mux_final
    mux_calls = []

    def shifted_mux_final(video_path, audio_path, out_path):
        mux_calls.append(out_path.name)
        shifted_audio = audio_path.parent / f"shifted_{audio_path.name}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(audio_path),
                "-af", "volume=-1.3dB",
                "-ar", "48000", "-fflags", "+bitexact", "-flags:a", "+bitexact",
                str(shifted_audio),
            ],
            check=True,
        )
        real_mux_final(video_path, shifted_audio, out_path)

    monkeypatch.setattr(assembly, "mux_final", shifted_mux_final)

    out_mp4 = tmp_path / "out.mp4"
    result = assembly.concat_and_mux([seg], [audio], tmp_path, out_mp4)

    assert out_mp4.exists()
    assert abs(result["final_loudness_i"] - assembly.LOUDNORM_TARGET_I) <= 1.0
    # Muxed once, corrected because the first measurement was outside the
    # margin, remuxed once more with the compensating gain applied.
    assert len(mux_calls) == 2


def test_concat_and_mux_skips_correction_when_already_within_margin(tmp_path, monkeypatch):
    """No wasted extra ffmpeg pass when the first measurement is already fine."""
    seg, audio = _make_inputs(tmp_path, "green")

    real_mux_final = assembly.mux_final
    mux_calls = []

    def counting_mux_final(video_path, audio_path, out_path):
        mux_calls.append(out_path.name)
        real_mux_final(video_path, audio_path, out_path)

    monkeypatch.setattr(assembly, "mux_final", counting_mux_final)

    out_mp4 = tmp_path / "out.mp4"
    result = assembly.concat_and_mux([seg], [audio], tmp_path, out_mp4)

    assert abs(result["final_loudness_i"] - assembly.LOUDNORM_TARGET_I) <= 1.0
    assert len(mux_calls) == 1
