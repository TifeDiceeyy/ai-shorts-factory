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

import re
import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100

# Well below the narration. The reference sits at -19.4 LUFS overall with a
# -53.6dB floor; the bed has to be present without competing with the voice.
#
# Measured against a real generated bed (2026-09-02), gain vs the resulting
# mix floor and the share of near-silent windows:
#     -26 dB -> floor -44.4 dB, 0.0% silent   <- chosen
#     -30 dB -> floor -47.4 dB, 0.6% silent
#     -34 dB -> floor -49.7 dB, 1.0% silent
#     -38 dB -> floor -51.1 dB, 1.0% silent
# Quieter settings land nearer the reference's floor but let dead air back
# in, because a generated bed dips where the reference's does not. "No
# silence anywhere" is the property worth having, so -26 stays.
#
# Note the bed does NOT close the loudness-range gap: mixing one in moved
# LRA 3.6 -> 3.5 against the reference's 1.7. That gap has some other cause,
# still unidentified — don't assume the bed or the loudnorm LRA parameter
# will fix it, both have been measured and neither does.
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


def ding(duration: float = 0.35, pitch: float = 1760.0) -> np.ndarray:
    """Bright bell — a point being made, a fact landing.

    One of the names the script's own `sfx` field actually asks for.
    """
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    bell = np.sin(2 * np.pi * pitch * t)
    bell += 0.5 * np.sin(2 * np.pi * pitch * 2.76 * t)   # inharmonic partial
    bell += 0.25 * np.sin(2 * np.pi * pitch * 5.40 * t)
    return (bell * _envelope(n, 0.005, 0.22)).astype(np.float32)


def sizzle(duration: float = 0.45) -> np.ndarray:
    """Sustained high noise — heating, burning, a reaction running."""
    n = int(SAMPLE_RATE * duration)
    rng = np.random.default_rng(11)           # fixed seed: audio is deterministic
    noise = rng.standard_normal(n).astype(np.float32)
    # High-pass by subtracting a slow-moving average, so it hisses rather
    # than rumbles.
    out = np.zeros(n, dtype=np.float32)
    acc = 0.0
    for i in range(n):
        acc += 0.02 * (noise[i] - acc)
        out[i] = noise[i] - acc
    env = _envelope(n, 0.10, 0.9)
    return (out * env).astype(np.float32)


def sparkle(duration: float = 0.40) -> np.ndarray:
    """Three rising blips — something appearing or being revealed."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    for k, pitch in enumerate((1400.0, 2100.0, 2800.0)):
        start = int(n * 0.18 * k)
        blip = pop(duration=0.10, pitch=pitch, click=0.4)
        end = min(n, start + len(blip))
        out[start:end] += blip[: end - start] * (1.0 - 0.2 * k)
    return out.astype(np.float32)


def zap(duration: float = 0.22) -> np.ndarray:
    """Electric arc/spark — buzzing crackle with a bright snap."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(17)          # fixed seed: audio is deterministic
    # A fast square-ish buzz gives the electrical character; noise bursts on
    # top give the crackle.
    buzz = np.sign(np.sin(2 * np.pi * 140.0 * t)) * 0.4
    crackle = rng.standard_normal(n).astype(np.float32) * np.exp(-t * 14.0)
    high = np.sin(2 * np.pi * 3200.0 * t) * np.exp(-t * 30.0) * 0.6
    return ((buzz + crackle + high) * _envelope(n, 0.005, 0.20)).astype(np.float32)


def splash(duration: float = 0.35) -> np.ndarray:
    """Liquid — pouring, splashing, gurgling. Filtered noise with a wobble."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(23)
    noise = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    acc = 0.0
    # Cutoff wobbles, which is what separates "liquid" from flat hiss.
    sweep = 0.10 + 0.06 * np.sin(2 * np.pi * 11.0 * t)
    for i in range(n):
        acc += sweep[i] * (noise[i] - acc)
        out[i] = acc
    return (out * _envelope(n, 0.08, 0.5) * 4.0).astype(np.float32)


def hum(duration: float = 0.5, pitch: float = 110.0) -> np.ndarray:
    """Steady electrical/mechanical drone — a machine running."""
    n = int(SAMPLE_RATE * duration)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * pitch * t) + 0.4 * np.sin(2 * np.pi * pitch * 2 * t)
    tone += 0.2 * np.sin(2 * np.pi * pitch * 3 * t)
    # Slow tremolo so it reads as a running machine, not a test tone.
    tone *= 1.0 + 0.15 * np.sin(2 * np.pi * 6.0 * t)
    return (tone * _envelope(n, 0.12, 0.8)).astype(np.float32)


def tada(duration: float = 0.6) -> np.ndarray:
    """Short rising fanfare — the payoff / it worked."""
    n = int(SAMPLE_RATE * duration)
    out = np.zeros(n, dtype=np.float32)
    # A major triad arpeggio: the cheapest thing that reads as "success".
    for k, pitch in enumerate((523.25, 659.25, 783.99, 1046.5)):
        start = int(n * 0.13 * k)
        note = ding(duration=0.30, pitch=pitch)
        end = min(n, start + len(note))
        out[start:end] += note[: end - start] * (0.7 + 0.1 * k)
    return out.astype(np.float32)


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


SFX_LIBRARY = {
    "pop": _normalized(pop),
    "whoosh": _normalized(whoosh),
    "thud": _normalized(thud),
    "ding": _normalized(ding),
    "sizzle": _normalized(sizzle),
    "sparkle": _normalized(sparkle),
    "zap": _normalized(zap),
    "splash": _normalized(splash),
    "hum": _normalized(hum),
    "tada": _normalized(tada),
}

# The script's own per-scene `sfx` field is free text written by the model,
# so it names sounds we don't synthesize. This maps what it ACTUALLY writes —
# collected from every script this project has generated — onto the closest
# effect we have.
#
# Measured 2026-09-02: 20 of the 33 distinct hints the model had written
# resolved to nothing, so those scenes fell back to the generic cut whoosh
# and 85% of a finished video's cues were the same sound. Two more resolved
# WRONGLY because matching was naive substring: "grating" matched "ting"
# inside it and became a bell; "crackling" became a thud.
_SFX_ALIASES = {
    # transitions
    "whoosh": "whoosh", "swoosh": "whoosh", "swipe": "whoosh", "wind": "whoosh",
    "transition": "whoosh", "rush": "whoosh", "sweep": "whoosh",
    # bright accents / a point landing
    "ding": "ding", "chime": "ding", "bell": "ding", "ting": "ding", "clink": "ding",
    "clang": "ding", "coin": "ding", "coin_jingle": "ding", "jingle": "ding",
    "magnify": "ding", "reveal_tone": "ding",
    # appearance / small impacts
    "pop": "pop", "click": "pop", "tap": "pop", "snap": "pop", "appear": "pop",
    "blip": "pop", "plop": "pop",
    # shimmer / discovery
    "sparkle": "sparkle", "twinkle": "sparkle", "shimmer": "sparkle",
    "magic": "sparkle", "reveal": "sparkle", "glow": "sparkle", "power_up": "sparkle",
    "powerup": "sparkle",
    # heat / reaction / texture
    "sizzle": "sizzle", "hiss": "sizzle", "fizz": "sizzle", "burn": "sizzle",
    "crackle": "sizzle", "crackling": "sizzle", "fire": "sizzle", "steam": "sizzle",
    "bubble": "sizzle", "bubbling": "sizzle", "boil": "sizzle", "boiling": "sizzle",
    "grate": "sizzle", "grating": "sizzle", "scrape": "sizzle", "stir": "sizzle",
    "stirring": "sizzle", "rustle": "sizzle",
    # heavy impacts
    "thud": "thud", "thump": "thud", "boom": "thud", "crash": "thud", "impact": "thud",
    "bang": "thud", "crack": "thud", "crumble": "thud", "crumble_impact": "thud",
    "rumble": "thud", "land": "thud", "clank": "thud", "clunk": "thud", "drop": "thud",
    # electricity
    "zap": "zap", "spark": "zap", "electric": "zap", "electric_arc": "zap",
    "arc": "zap", "buzz": "zap", "static": "zap", "shock": "zap", "current": "zap",
    # liquid
    "splash": "splash", "pour": "splash", "pouring": "splash", "gurgle": "splash",
    "water": "splash", "drip": "splash", "liquid": "splash", "flow": "splash",
    "trickle": "splash",
    # machinery drone
    "hum": "hum", "humming": "hum", "electric_hum": "hum", "drone": "hum",
    "motor": "hum", "engine": "hum", "whir": "hum", "whirr": "hum", "vibrate": "hum",
    # success
    "tada": "tada", "fanfare": "tada", "success": "tada", "cheer": "tada",
    "applause": "tada", "win": "tada", "triumph": "tada", "celebrate": "tada",
}

# Hints that name a HUMAN vocalisation. Deliberately unmapped: we synthesize
# no voice, and substituting a mechanical sound for "gasp" is worse than
# staying silent — the scene simply gets no accent of its own.
_IGNORED_SFX_HINTS = frozenset({"gasp", "sigh", "laugh", "scream", "shout", "breath", "none", "null", "silence"})


def resolve_sfx_name(raw: str | None) -> str | None:
    """Map a script's free-text sfx hint onto an effect we can synthesize.

    Matches on whole word-ish TOKENS, longest first, rather than raw
    substrings: naive `in` matching turned "grating" into a bell (it
    contains "ting") and "crackling" into a thud.
    """
    text = (raw or "").strip().lower()
    if not text or text in _IGNORED_SFX_HINTS:
        return None
    if text in SFX_LIBRARY:
        return text
    if text in _SFX_ALIASES:
        return _SFX_ALIASES[text]

    tokens = [t for t in re.split(r"[^a-z]+", text) if t]
    if any(t in _IGNORED_SFX_HINTS for t in tokens):
        return None
    for token in tokens:
        if token in SFX_LIBRARY:
            return token
        if token in _SFX_ALIASES:
            return _SFX_ALIASES[token]
    # Last resort: a token that STARTS with a known alias ("bubbles" ->
    # "bubble"). Longest alias first so "crackling" cannot match "crack".
    for alias in sorted(_SFX_ALIASES, key=len, reverse=True):
        if any(t.startswith(alias) for t in tokens):
            return _SFX_ALIASES[alias]
    return None


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

    Every scene also contributes its OWN authored `sfx` hint, placed just
    after the cut so it reads as belonging to that shot rather than to the
    transition. Without this the ai_video path produced almost no sound at
    all: it has no `stickers` and rarely an ingredient_grid, so the whole
    generator collapsed to one whoosh per cut — measured 0.09 hits/sec on a
    real render against the reference's 1.73 (2026-09-02), while the script
    had written whoosh/ding/sizzle/pop/sparkle on its five scenes and every
    one was ignored.
    """
    cues: list[tuple[float, str]] = []
    cursor = 0.0
    for scene, duration in zip(scenes, durations):
        authored = resolve_sfx_name(scene.get("sfx"))

        # Accent the cut ONLY when the scene brings no sound of its own.
        # Whooshing every cut regardless is what made 85% of a finished
        # video's cues the same sound (measured 2026-09-02) — the scene's
        # own authored effect was there, but buried under a transition it
        # didn't need. The reference accents 7 of its 14 cuts, not all 14.
        if cursor > 0.0 and not authored:
            cues.append((cursor, "whoosh"))

        if authored:
            # Offset so it doesn't collide with the transition whoosh on the
            # same frame, where the two would just sum into one louder hit.
            at = cursor + min(0.25, duration * 0.15)
            cues.append((at, authored))

        # A static scene is now cut into 2-3 shots (assembly.
        # plan_shot_durations), and every one of those cuts is a moment the
        # reference would put a sound on. Without this the extra cuts would
        # be silent, leaving the track at roughly half the reference's
        # density.
        if not scene.get("stickers"):
            from .assembly import plan_shot_durations

            # Internal cuts echo the scene's OWN sound where it has one, and
            # fall back to the transition whoosh where it doesn't. Using
            # whoosh for all of them made 85% of a finished track the same
            # noise; dropping them entirely fixed the variety but halved the
            # density to 0.33 hits/sec against the reference's ~0.9. Echoing
            # keeps both: the scene gets a motif rather than a transition.
            shot_at = cursor
            for shot_duration in plan_shot_durations(duration)[:-1]:
                shot_at += shot_duration
                cues.append((shot_at, authored or "whoosh"))

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
