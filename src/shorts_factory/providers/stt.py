"""Speech-to-text word alignment, for caption timing.

Captions were timed by weighting each word's LENGTH — a longer word was
assumed to take proportionally longer to say (see assembly.
narration_caption_cues). That is only ever an approximation: it cannot know
about pauses, emphasis, or the fact that "through" and "throughout" take
very different times despite similar length. This asks the audio instead.

The provider is optional throughout. When it is a stub, or a call fails, the
caller keeps the length-weighted estimate — a video with slightly-off
caption timing is worth far more than no video.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cost_tracker import CostTracker
from .fal import FalGateway


@dataclass(frozen=True)
class WordTiming:
    """One spoken word and when it was actually said."""

    text: str
    start: float
    end: float


class STTProvider(ABC):
    name: str

    @abstractmethod
    def align(
        self, audio_path: Path, cost_tracker: CostTracker
    ) -> list[WordTiming]:
        """Word-level timings for everything spoken in audio_path.

        Returns [] when alignment is unavailable, rather than raising — the
        caller treats that as "fall back to the estimate".
        """
        ...


class StubSTTProvider(STTProvider):
    name = "stub"

    def align(self, audio_path: Path, cost_tracker: CostTracker) -> list[WordTiming]:
        return []


class FalSTTProvider(STTProvider):
    name = "fal"

    # ElevenLabs Scribe. Confirmed live 2026-09-02: returns
    # {"text", "language_code", "words": [{"text","start","end","type"}]},
    # 281 word entries for a 47s narration.
    ENDPOINT = "fal-ai/elevenlabs/speech-to-text"

    def __init__(self, gateway: FalGateway, model: str = "", cost_per_minute_usd: float = 0.0):
        self.gateway = gateway
        # Honour STT_MODEL when set. Without this the constant below always
        # won and a configured endpoint was silently discarded.
        self.endpoint = model or self.ENDPOINT
        self.cost_per_minute_usd = cost_per_minute_usd

    def align(self, audio_path: Path, cost_tracker: CostTracker) -> list[WordTiming]:
        from ..assembly import probe_duration

        minutes = max(0.0, probe_duration(audio_path)) / 60.0
        estimated = minutes * self.cost_per_minute_usd
        cost_tracker.check_budget("stt.align", estimated_cost_usd=estimated)

        audio_url = self.gateway.upload(audio_path)
        result = self.gateway.run(self.endpoint, {"audio_url": audio_url}, timeout=600)
        cost_tracker.record(
            provider="fal", operation="stt.align",
            estimated_cost_usd=estimated, actual_cost_usd=estimated, is_stub=False,
        )
        return parse_word_timings(result)


def parse_word_timings(result: dict[str, Any]) -> list[WordTiming]:
    """Pull WordTimings out of a Scribe response.

    Entries whose type isn't "word" (spacing, audio events) are dropped, as
    are any with a non-increasing span — a zero-length or reversed timing
    would produce a caption cue that never displays.
    """
    timings: list[WordTiming] = []
    for entry in result.get("words") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") not in (None, "word"):
            continue
        text = str(entry.get("text") or "").strip()
        start, end = entry.get("start"), entry.get("end")
        if not text or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        if end <= start:
            continue
        timings.append(WordTiming(text=text, start=float(start), end=float(end)))
    return timings


def get_stt_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_minute_usd: float = 0.0,
    gateway: FalGateway | None = None,
) -> STTProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubSTTProvider()
    if provider_name.strip().lower() == "fal":
        return FalSTTProvider(gateway or FalGateway(api_key), model, cost_per_minute_usd)
    raise NotImplementedError(f"Unsupported STT provider {provider_name!r}; use 'stub' or 'fal'")
