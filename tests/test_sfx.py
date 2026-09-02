"""Synthesized SFX + music-bed mixing.

Driven by measurement of a real reference short (2026-09-01): ~40 discrete
transients across 43s, 7 of 14 cuts sound-accented, ~14 hits during the
ingredient accumulation (about one per item), and an audio floor of -53.6dB
with only 1 of 434 windows below -45dB — i.e. a continuous music bed under
the voice, where ours dropped to true silence between sentences.

Effects are synthesized rather than sampled: no asset library ships with this
project, and generated audio is free, licence-free and deterministic.
"""
import wave

import numpy as np
import pytest

from shorts_factory import sfx


def _read(path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def test_every_effect_is_normalized_and_audible():
    """Each effect must peak at 1.0 before gain, so SFX_GAIN_DB means the
    same loudness for all of them. Raw synths landed between 0.99 and 4.13."""
    for name, make in sfx.SFX_LIBRARY.items():
        samples = make()
        peak = float(np.abs(samples).max())
        assert 0.99 <= peak <= 1.001, f"{name} peaks at {peak:.3f}, not normalized"
        assert len(samples) > 1000, f"{name} is suspiciously short"


def test_effects_are_deterministic():
    """The determinism tests hash whole renders — a random noise seed would
    make two identical scripts produce different videos."""
    assert np.array_equal(sfx.SFX_LIBRARY["whoosh"](), sfx.SFX_LIBRARY["whoosh"]())


def test_cues_fire_on_scene_changes_and_once_per_ingredient(tmp_path):
    """Mirrors the reference's own pattern: a hit at each scene change, plus
    one per ingredient as it appears."""
    scenes = [
        {"scene_type": "mascot", "stickers": []},
        {"scene_type": "ingredient_grid",
         "stickers": [{"appear_at": 0.5}, {"appear_at": 1.5}, {"appear_at": 2.5}]},
    ]
    cues = sfx.scene_sfx_cues(scenes, [3.0, 4.0])
    assert ("whoosh" in [n for _t, n in cues]), "a scene change must be accented"
    pops = [t for t, n in cues if n == "pop"]
    assert len(pops) == 3, "one pop per ingredient"
    assert pops == [3.5, 4.5, 5.5], "pops must be offset by the scene's start time"


def test_ingredient_grid_without_sticker_timings_still_pops():
    """Older scripts carry no per-sticker appear_at. The grid must still land
    audibly item by item rather than falling silent."""
    cues = sfx.scene_sfx_cues([{"scene_type": "ingredient_grid"}], [4.0])
    assert len([n for _t, n in cues if n == "pop"]) == 4


def test_track_places_every_cue_and_never_clips(tmp_path):
    scenes = [{"scene_type": "ingredient_grid",
               "stickers": [{"appear_at": t} for t in (0.4, 1.2, 2.0, 2.8)]}]
    cues = sfx.scene_sfx_cues(scenes, [4.0])
    track = sfx.build_sfx_track(cues, 4.0, tmp_path / "t.wav")
    data = _read(track)
    assert abs(len(data) / sfx.SAMPLE_RATE - 4.0) < 0.05
    assert np.abs(data).max() <= 0.995, "summed hits must be limited, not clipped"
    hop = int(sfx.SAMPLE_RATE * 0.01)
    env = np.array([np.abs(data[i:i + hop]).max() for i in range(0, len(data) - hop, hop)])
    for start, _name in cues:
        window = env[int(start * 100): int(start * 100) + 15]
        assert window.max() > 0.01, f"no audible hit at {start}s"


def test_mix_returns_narration_untouched_when_there_is_nothing_to_add(tmp_path):
    """No SFX and no music must not trigger a pointless re-encode of the
    narration — that could only degrade it."""
    narration = tmp_path / "n.wav"
    sfx._write_wav(np.zeros(sfx.SAMPLE_RATE, dtype=np.float32), narration)
    out = sfx.mix_audio_layers(narration, tmp_path / "out.wav")
    assert out == narration


def test_music_bed_is_looped_and_cut_to_the_narration(tmp_path):
    """A bed shorter than the narration must loop to fill it, and one longer
    must never extend the video."""
    pytest.importorskip("numpy")
    narration = tmp_path / "n.wav"
    music = tmp_path / "m.wav"
    t = np.arange(int(sfx.SAMPLE_RATE * 3.0), dtype=np.float32) / sfx.SAMPLE_RATE
    sfx._write_wav(0.4 * np.sin(2 * np.pi * 220 * t), narration)      # 3.0s
    t2 = np.arange(int(sfx.SAMPLE_RATE * 0.7), dtype=np.float32) / sfx.SAMPLE_RATE
    sfx._write_wav(0.4 * np.sin(2 * np.pi * 110 * t2), music)          # 0.7s, shorter
    out = sfx.mix_audio_layers(narration, tmp_path / "mixed.wav", music=music)
    data = _read(out)
    assert abs(len(data) / sfx.SAMPLE_RATE - 3.0) < 0.05, "output must match the narration length"
    # The bed must be present in the tail, where the 0.7s source would have
    # run out had it not looped.
    assert np.abs(data[int(sfx.SAMPLE_RATE * 2.5):]).max() > 0.01


def test_both_assembly_entry_points_accept_sfx_and_music():
    """SFX were wired into the sticker path only.

    assemble_stickers() got sfx_enabled/music_path when the effects were
    added, but assemble_animated() — the ai_video path — did not, so every
    Kling render shipped with no effects and no music bed, its audio bare
    narration dropping to true silence between sentences. Nothing about the
    audio layering is mode-specific, so both entry points must expose it.
    """
    import inspect

    from shorts_factory import assembly

    for name in ("assemble_stickers", "assemble_animated"):
        params = inspect.signature(getattr(assembly, name)).parameters
        assert "sfx_enabled" in params, f"{name} cannot enable SFX"
        assert "music_path" in params, f"{name} cannot take a music bed"
        # Both must default to off, so a caller that doesn't opt in keeps
        # exactly the audio it had before.
        assert params["sfx_enabled"].default is False
        assert params["music_path"].default is None


def test_pop_brightness_matches_the_reference_transients():
    """The pop fires once per ingredient, so it is the most-heard effect.

    Measured on the reference short (2026-09-01): its item-appearance
    transients have spectral centroids of 2345-6938 Hz. The first version of
    this effect measured 395 Hz — roughly eleven times too dark, a dull
    thump where the reference snaps.
    """
    x = sfx.SFX_LIBRARY["pop"]()
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sfx.SAMPLE_RATE)
    centroid = float((spectrum * freqs).sum() / spectrum.sum())
    assert 2300 <= centroid <= 6900, f"pop centroid {centroid:.0f} Hz is outside the reference band"
