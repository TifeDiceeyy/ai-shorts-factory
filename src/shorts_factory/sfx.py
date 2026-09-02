"""Synthesized sound effects and music-bed mixing.

Why synthesized rather than sampled: the pipeline ships to a customer with no
audio asset library, and a licensed SFX pack is a dependency this project
does not have. These are generated from scratch with numpy, so they are free,
deterministic (same script -> same audio, which the determinism tests rely on)
and carry no licensing question.

The need came from measuring a real reference short (2026-09-01): it carries
~40 discrete transients across 43s (roughly one a second), 7 of its 14 cuts
are sound-accented, and ~14 hits land during the ingredient-accumulation
sequence — about one per item appearing. Critically, its audio floor sits at
-53.6dB with only 1 of 434 windows below -45dB, meaning a CONTINUOUS music
bed under the voice; a speech-only track drops to true silence between
sentences, and ours did.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100

# Well below the narration. The reference sits at -19.4 LUFS overall with a
# -53.6dB floor; the bed has to be present without competing with the voice.
MUSIC_BED_GAIN_DB = -26.0
SFX_GAIN_DB = -18.0


def _envelope(n: int, attack: float, decay: float) -> np.ndarray:
    """Fast attack, exponential decay — the shape of a struck/plucked sound."""
    a = max(1, int(n * attack))
    env = np.ones(n, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    tail = np.exp(-np.linspace(0.0, 1.0, n - a, dtype=np.float32) / max(decay, 1e-3))
    env[a:] = tail
    return env


def pop(duration: float = 0.11, pitch: float = 3000.0, click: float = 1.2) -> np.ndarray:
    """The sound of a sticker/ingredient snapping into place.

    A short pitched blip with a rapid downward sweep plus a very fast noise
    click — reads as "appeared" rather than "hit". This is the one that
    fires per ingredient, so it is the most-heard effect in the video.

    Tuned against the reference short (2026-09-01): its item-appearance
    transients measure a spectral centroid of 2345-6938 Hz (median ~4336).
    The first version of this effect was pitched at 620Hz with no click and
    measured 395 Hz — about eleven times too dark, reading as a dull thump
    where the reference has a bright snap. Pitch 3000 plus the noise click
    puts it at ~4765 Hz, inside the measured band.
    """
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    sweep = pitch * np.exp(-t * 9.0)
    wave_ = np.sin(2 * np.pi * sweep * t)
    wave_ += 0.35 * np.sin(4 * np.pi * sweep * t)
    # Fixed seed: the determinism tests require identical audio per script.
    rng = np.random.default_rng(3)
    wave_ = wave_ + click * rng.standard_normal(n).astype(np.float32) * np.exp(-t * 120.0)
    return (wave_ * _envelope(n, 0.01, 0.14)).astype(np.float32)


def whoosh(duration: float = 0.22) -> np.ndarray:
    """Filtered noise sweep — a scene transition / something moving past."""
    n = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng(7)          # fixed seed: audio must be deterministic
    noise = rng.standard_normal(n).astype(np.float32)
    # Cheap one-pole low-pass whose cutoff opens then closes across the sound.
    out = np.zeros(n, dtype=np.float32)
    acc = 0.0
    sweep = np.sin(np.linspace(0, np.pi, n)) * 0.45 + 0.05
    for i in range(n):
        acc += sweep[i] * (noise[i] - acc)
        out[i] = acc
    return (out * _envelope(n, 0.25, 0.30) * 3.0).astype(np.float32)


def thud(duration: float = 0.18, pitch: float = 130.0) -> np.ndarray:
    """Low body hit — a landing, or a heavy object arriving."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    body = np.sin(2 * np.pi * pitch * np.exp(-t * 5.0) * t)
    return (body * _envelope(n, 0.01, 0.14)).astype(np.float32)


def _normalized(fn):
    """Each effect peaks at exactly 1.0 before gain.

    Without this the raw synths land anywhere from 0.99 (thud) to 4.13
    (whoosh), so SFX_GAIN_DB would mean a different loudness per effect and
    the mix would be unbalanced by accident rather than by choice.
    """
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        peak = float(np.abs(out).max())
        return out / peak if peak > 0 else out
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


SFX_LIBRARY = {"pop": _normalized(pop), "whoosh": _normalized(whoosh), "thud": _normalized(thud)}


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def build_sfx_track(cues: list[tuple[float, str]], duration: float, out_path: Path) -> Path:
    """Render timed SFX hits onto one silent track of `duration` seconds.

    cues are (start_seconds, effect_name). Hits are summed, so two effects
    landing together reinforce rather than truncating one another; the result
    is soft-limited at the end because a pile-up can otherwise clip.
    """
    total = max(1, int(SAMPLE_RATE * duration))
    track = np.zeros(total, dtype=np.float32)
    for start, name in cues:
        make = SFX_LIBRARY.get(name)
        if make is None:
            continue
        sample = make()
        at = int(start * SAMPLE_RATE)
        if at >= total:
            continue
        end = min(total, at + len(sample))
        track[at:end] += sample[: end - at]

    track *= _db_to_gain(SFX_GAIN_DB)
    peak = float(np.abs(track).max())
    if peak > 0.99:
        track *= 0.99 / peak
    _write_wav(track, out_path)
    return out_path


def _write_wav(samples: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def mix_audio_layers(
    narration: Path,
    out_path: Path,
    sfx: Path | None = None,
    music: Path | None = None,
    music_gain_db: float = MUSIC_BED_GAIN_DB,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Mix narration + optional SFX + optional looping music bed.

    The music is looped and cut to the narration's length, so any track works
    regardless of its own duration. `amix` with `duration=first` keeps the
    narration authoritative — a long music file must never extend the video.
    Narration is never attenuated: the bed and effects are placed under it.
    """
    layers = [narration]
    filters = ["[0:a]volume=1.0[voice]"]
    mix_inputs = ["[voice]"]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(narration)]

    if sfx is not None and sfx.exists():
        cmd.extend(["-i", str(sfx)])
        idx = len(layers)
        layers.append(sfx)
        filters.append(f"[{idx}:a]volume=1.0[sfxa]")
        mix_inputs.append("[sfxa]")

    if music is not None and music.exists():
        cmd.extend(["-stream_loop", "-1", "-i", str(music)])
        idx = len(layers)
        layers.append(music)
        filters.append(f"[{idx}:a]volume={_db_to_gain(music_gain_db):.5f}[bed]")
        mix_inputs.append("[bed]")

    if len(mix_inputs) == 1:
        # Nothing to add — hand the narration straight back rather than
        # burning a re-encode that could only degrade it.
        return narration

    filters.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:"
        f"dropout_transition=0:normalize=0[mixed]"
    )
    cmd.extend(["-filter_complex", ";".join(filters), "-map", "[mixed]",
                "-c:a", "pcm_s16le", str(out_path)])
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def scene_sfx_cues(scenes: list[dict], durations: list[float]) -> list[tuple[float, str]]:
    """Derive SFX timings from the script itself.

    Modelled on the reference's own pattern: a hit on each scene change, and
    one per ingredient as it appears (its accumulation sequence carried ~14
    hits across 13.6s, about one per item).
    """
    cues: list[tuple[float, str]] = []
    cursor = 0.0
    for scene, duration in zip(scenes, durations):
        if cursor > 0.0:
            cues.append((cursor, "whoosh"))
        stickers = scene.get("stickers") or []
        if stickers:
            for sticker in stickers:
                at = cursor + float(sticker.get("appear_at", 0.0))
                if at < cursor + duration:
                    cues.append((at, "pop"))
        elif scene.get("scene_type") == "ingredient_grid":
            # No per-sticker timings available — space pops across the scene
            # so the grid still lands audibly item by item.
            for i in range(4):
                cues.append((cursor + duration * (i + 1) / 5.0, "pop"))
        cursor += duration
    return cues
