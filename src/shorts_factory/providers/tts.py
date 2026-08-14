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
from .fal import FalGateway, media_url

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


class FalTTSProvider(TTSProvider):
    """Configurable fal text-to-speech endpoint returning `audio.url`."""

    name = "fal"

    def __init__(
        self,
        gateway: FalGateway,
        endpoint: str,
        voice_id: str,
        cost_per_1k_chars_usd: float,
    ):
        if not endpoint or not voice_id:
            raise ValueError("fal TTS requires TTS_MODEL and TTS_VOICE")
        if cost_per_1k_chars_usd <= 0:
            raise ValueError("Set TTS_COST_PER_1K_CHARS_USD to a conservative positive estimate")
        self.gateway = gateway
        self.endpoint = endpoint
        self.voice_id = voice_id
        self.rate = cost_per_1k_chars_usd

    def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
        narration = scene["narration"]
        estimate = max(0.000001, len(narration) / 1000 * self.rate)
        operation = f"tts.synthesize_scene[{scene_index}]"
        cost_tracker.check_budget(operation, estimate)

        data = self.gateway.run(
            self.endpoint,
            {
                "text": narration,
                "voice": self.voice_id,
                "timestamps": False,
            },
        )
        audio_url = media_url(data, "audio")
        audio_bytes = self.gateway.download(audio_url)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        converted = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), str(out_path)],
            input=audio_bytes,
            capture_output=True,
        )
        if converted.returncode != 0:
            raise RuntimeError(f"ffmpeg failed converting fal TTS audio: {converted.stderr.decode(errors='replace')}")

        cost_tracker.record(self.name, operation, estimate, estimate, is_stub=False)
        return out_path


def get_tts_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    voice_id: str = "",
    cost_per_1k_chars_usd: float = 0.0,
    gateway: FalGateway | None = None,
) -> TTSProvider:
    name = provider_name.strip().lower()
    if name in ("", "stub"):
        return StubTTSProvider()
    if name == "fal":
        return FalTTSProvider(gateway or FalGateway(api_key), model, voice_id, cost_per_1k_chars_usd)
    raise NotImplementedError(f"Unsupported TTS provider {provider_name!r}; use 'stub' or 'fal'")
