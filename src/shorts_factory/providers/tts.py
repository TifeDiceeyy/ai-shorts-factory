"""TTS provider interface.

StubTTSProvider synthesizes a real (non-silent) sine-wave tone per scene via
ffmpeg's lavfi source, sized exactly to the scene's scripted duration. Using
an actual audio signal (not silence) matters: it lets the loudness-normalize
step do real, verifiable work (silence has no meaningful LUFS to normalize
toward -14). Frequency is derived deterministically from the scene index —
no randomness, so assembly stays reproducible. Zero network, zero cost.

A real provider (ElevenLabs, OpenAI TTS, ...) plugs in behind the same
synthesize_scene() signature once TTS_PROVIDER is approved and configured.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..cost_tracker import CostTracker

SAMPLE_RATE = 48000
BASE_FREQUENCY_HZ = 300
FREQUENCY_STEP_HZ = 40
MAX_FREQUENCY_HZ = 900


class TTSProvider(ABC):
    name: str

    @abstractmethod
    def synthesize_scene(
        self,
        scene: dict[str, Any],
        scene_index: int,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        ...


class StubTTSProvider(TTSProvider):
    name = "stub"

    def synthesize_scene(
        self,
        scene: dict[str, Any],
        scene_index: int,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        operation = f"tts.synthesize_scene[{scene_index}]"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)

        duration = scene["duration"]
        freq = min(MAX_FREQUENCY_HZ, BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * scene_index)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration}:sample_rate={SAMPLE_RATE}",
            "-ac", "1",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed generating stub TTS for scene {scene_index}: {result.stderr}")

        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return out_path


def get_tts_provider(provider_name: str) -> TTSProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubTTSProvider()
    raise NotImplementedError(
        f"TTS provider {provider_name!r} is not wired up yet — Phase 0 only "
        "implements the stub provider until a real provider/voice is approved "
        "and a credential is supplied."
    )
