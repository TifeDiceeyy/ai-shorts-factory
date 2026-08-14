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

import requests

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


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(self, api_key: str, voice_id: str, cost_per_1k_chars_usd: float):
        if not api_key or not voice_id:
            raise ValueError("ElevenLabs requires ELEVENLABS_API_KEY and TTS_VOICE")
        if cost_per_1k_chars_usd <= 0:
            raise ValueError("Set TTS_COST_PER_1K_CHARS_USD to a conservative positive estimate")
        self.api_key = api_key
        self.voice_id = voice_id
        self.rate = cost_per_1k_chars_usd

    def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
        narration = scene["narration"]
        estimate = max(0.000001, len(narration) / 1000 * self.rate)
        operation = f"tts.synthesize_scene[{scene_index}]"
        cost_tracker.check_budget(operation, estimate)
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            headers={"xi-api-key": self.api_key, "accept": "audio/mpeg", "content-type": "application/json"},
            json={"text": narration, "model_id": "eleven_multilingual_v2"},
            timeout=120,
        )
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        converted = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), str(out_path)],
            input=response.content,
            capture_output=True,
        )
        if converted.returncode != 0:
            raise RuntimeError(f"ffmpeg failed converting ElevenLabs audio: {converted.stderr.decode(errors='replace')}")
        cost_tracker.record(self.name, operation, estimate, estimate, is_stub=False)
        return out_path


class FalTTSProvider(TTSProvider):
    """fal.ai's hosted MiniMax Speech-02 HD (fal-ai/minimax/speech-02-hd).
    Request/response shape confirmed against live fal docs (2026-08-14):
    request takes text + voice_setting.voice_id, response returns audio.url
    (not base64) plus duration_ms. voice_id here is TTS_VOICE, e.g. 'Wise_Woman'."""

    name = "fal"

    def __init__(self, api_key: str, voice_id: str, cost_per_1k_chars_usd: float):
        if not api_key or not voice_id:
            raise ValueError("fal TTS requires FAL_KEY and TTS_VOICE (e.g. 'Wise_Woman')")
        if cost_per_1k_chars_usd <= 0:
            raise ValueError("Set TTS_COST_PER_1K_CHARS_USD to a conservative positive estimate")
        self.api_key = api_key
        self.voice_id = voice_id
        self.rate = cost_per_1k_chars_usd

    def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
        narration = scene["narration"]
        estimate = max(0.000001, len(narration) / 1000 * self.rate)
        operation = f"tts.synthesize_scene[{scene_index}]"
        cost_tracker.check_budget(operation, estimate)

        response = requests.post(
            "https://fal.run/fal-ai/minimax/speech-02-hd",
            headers={"Authorization": f"Key {self.api_key}", "content-type": "application/json"},
            json={
                "text": narration,
                "voice_setting": {"voice_id": self.voice_id},
                "output_format": "url",
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        audio_url = data.get("audio", {}).get("url")
        if not audio_url:
            raise ValueError("fal minimax response did not contain audio.url")

        downloaded = requests.get(audio_url, timeout=120)
        downloaded.raise_for_status()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        converted = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SAMPLE_RATE), str(out_path)],
            input=downloaded.content,
            capture_output=True,
        )
        if converted.returncode != 0:
            raise RuntimeError(f"ffmpeg failed converting fal TTS audio: {converted.stderr.decode(errors='replace')}")

        cost_tracker.record(self.name, operation, estimate, estimate, is_stub=False)
        return out_path


def get_tts_provider(
    provider_name: str,
    api_key: str = "",
    voice_id: str = "",
    cost_per_1k_chars_usd: float = 0.0,
) -> TTSProvider:
    name = provider_name.strip().lower()
    if name in ("", "stub"):
        return StubTTSProvider()
    if name == "elevenlabs":
        return ElevenLabsTTSProvider(api_key, voice_id, cost_per_1k_chars_usd)
    if name == "fal":
        return FalTTSProvider(api_key, voice_id, cost_per_1k_chars_usd)
    raise NotImplementedError(f"Unsupported TTS provider {provider_name!r}; use 'stub', 'elevenlabs', or 'fal'")
