"""Music-bed generation, one bed per story.

The reference short runs a continuous low music bed under the whole
voiceover: its audio floor sits at -53.6dB with zero true-silence windows,
and that unbroken bed is also why its loudness range measures 1.7 LU where
ours measures 3.5 (verified 2026-09-02 — re-normalising at every LRA target
from 11 down to 2 barely moved ours, because the gaps between our sentences
are near-silent and nothing fills them).

A single cached track can't serve every video: the bed has to suit the
story, so it is generated per topic and cached under that topic's artifacts
so re-running a topic is free.

Optional throughout: a stub provider, an unset model, or a failed call all
mean "no bed", and the video renders exactly as it did before.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..cost_tracker import CostTracker
from .fal import FalGateway, media_url

# Long enough to avoid an obvious loop point, short enough not to pay for
# audio the video will never reach. mix_audio_layers loops it to length.
DEFAULT_BED_SECONDS = 30


class MusicProvider(ABC):
    name: str

    @abstractmethod
    def generate_bed(
        self, mood_prompt: str, out_path: Path, cost_tracker: CostTracker,
        seconds: int = DEFAULT_BED_SECONDS,
    ) -> Path | None:
        """Writes a music bed to out_path and returns it, or None if it
        cannot supply one. Never raises for an unavailable provider — the
        caller treats None as "render without a bed"."""
        ...


class StubMusicProvider(MusicProvider):
    name = "stub"

    def generate_bed(self, mood_prompt, out_path, cost_tracker, seconds=DEFAULT_BED_SECONDS):
        return None


class FalMusicProvider(MusicProvider):
    name = "fal"

    # Confirmed live 2026-09-02: returns {"audio_file": {"url": ...}} for
    # {"prompt": str, "seconds_total": int}.
    ENDPOINT = "fal-ai/stable-audio"

    def __init__(self, gateway: FalGateway, model: str = "", cost_per_bed_usd: float = 0.0):
        self.gateway = gateway
        self.endpoint = model or self.ENDPOINT
        self.cost_per_bed_usd = cost_per_bed_usd

    def generate_bed(self, mood_prompt, out_path, cost_tracker, seconds=DEFAULT_BED_SECONDS):
        cost_tracker.check_budget("music.generate_bed", estimated_cost_usd=self.cost_per_bed_usd)
        result = self.gateway.run(
            self.endpoint, {"prompt": mood_prompt, "seconds_total": int(seconds)}, timeout=600,
        )
        url = media_url(result, "audio_file", "url")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.gateway.download(url))
        cost_tracker.record(
            provider="fal", operation="music.generate_bed",
            estimated_cost_usd=self.cost_per_bed_usd,
            actual_cost_usd=self.cost_per_bed_usd, is_stub=False,
        )
        return out_path


def build_mood_prompt(topic: str, visual_style: str = "") -> str:
    """The bed's brief, derived from the story it plays under.

    Deliberately instrumental and low-energy-forward: this sits UNDER a
    voiceover at -26dB, so anything with vocals or a strong lead line fights
    the narration instead of supporting it.
    """
    subject = (topic or "an explainer").strip()
    return (
        f"Upbeat quirky instrumental cartoon background music for a short explainer video "
        f"about {subject}. Light playful percussion, simple plucky melody, curious and "
        f"energetic but understated. Instrumental only, no vocals, no speech, evenly paced "
        f"with no dramatic swells, loopable background bed."
    )


def get_music_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_bed_usd: float = 0.0,
    gateway: FalGateway | None = None,
) -> MusicProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubMusicProvider()
    if provider_name.strip().lower() == "fal":
        return FalMusicProvider(gateway or FalGateway(api_key), model, cost_per_bed_usd)
    raise NotImplementedError(f"Unsupported music provider {provider_name!r}; use 'stub' or 'fal'")
