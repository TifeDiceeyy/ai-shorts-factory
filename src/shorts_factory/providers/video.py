"""Video provider interface: image-to-video animation for one scene.

StubVideoProvider makes a short, deterministic, zero-cost local clip (holds
the source hero image for a fixed duration) so the pipeline stays fully
testable without real network calls or spend — same house pattern as every
other provider here.

FalVideoProvider animates a single, shared "hero" character image (see
pipeline.py — generated once per video, not once per scene) via MiniMax
Hailuo-02 image-to-video, the cheapest real video model on fal.ai as of
2026-08-17 ($0.045/sec, confirmed live against fal.ai's docs). Reusing one
hero image across every scene is deliberate: it's what makes the character
consistent scene to scene, which independent per-scene image generation
could never guarantee. Every call is pinned to Hailuo's 6-second minimum
duration (the cheapest option) — assembly.py pads or trims the result to
match each scene's actual scripted duration afterward.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..cost_tracker import CostTracker
from .fal import FalGateway, media_url

HAILUO_CLIP_SECONDS = 6.0


class VideoProvider(ABC):
    name: str

    @abstractmethod
    def generate_scene_video(
        self,
        scene: dict[str, Any],
        hero_image_path: Path,
        scene_index: int,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        ...


class StubVideoProvider(VideoProvider):
    name = "stub"

    def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
        operation = f"video.generate_scene_video[{scene_index}]"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-i", str(hero_image_path),
            "-t", str(HAILUO_CLIP_SECONDS),
            "-pix_fmt", "yuv420p",
            "-fflags", "+bitexact", "-flags:v", "+bitexact",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"stub video command failed: {' '.join(cmd)}\n{result.stderr}")
        cost_tracker.record(self.name, operation, 0.0, 0.0, is_stub=True)
        return out_path


class FalVideoProvider(VideoProvider):
    name = "fal"

    def __init__(self, gateway: FalGateway, model: str, cost_per_second_usd: float):
        if not model:
            raise ValueError("fal video generation requires VIDEO_MODEL")
        if cost_per_second_usd <= 0:
            raise ValueError("Set VIDEO_COST_PER_SECOND_USD to a conservative positive estimate")
        self.gateway = gateway
        self.model = model.strip("/")
        self.cost = cost_per_second_usd * HAILUO_CLIP_SECONDS

    def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
        operation = f"video.generate_scene_video[{scene_index}]"
        cost_tracker.check_budget(operation, self.cost)
        image_url = self.gateway.upload(hero_image_path)
        # Video generation routinely takes several minutes — confirmed live
        # 2026-08-17: the shared FalGateway.run() default of 180s (fine for
        # LLM/TTS/image calls) timed out on a real call before it finished rendering.
        prompt = scene["visual_prompt"]
        if "kling" in self.model.lower():
            arguments = {
                "image_url": image_url,
                "prompt": prompt,
                "duration": "5",
                "aspect_ratio": "9:16",
            }
        elif "luma" in self.model.lower():
            arguments = {
                "image_url": image_url,
                "prompt": prompt,
                "aspect_ratio": "9:16",
            }
        else:
            # MiniMax / Hailuo default
            arguments = {
                "image_url": image_url,
                "prompt": prompt,
                "duration": "6",
                "resolution": "768P",
            }

        data = self.gateway.run(
            self.model,
            arguments,
            timeout=600,
        )
        try:
            video_url = media_url(data, "video", "url")
        except Exception:
            try:
                video_url = media_url(data, "video")
            except Exception:
                video_url = media_url(data, "output", "url")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(self.gateway.download(video_url))
        cost_tracker.record(self.name, operation, self.cost, self.cost, is_stub=False)
        return out_path


def get_video_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_second_usd: float = 0.0,
    gateway: FalGateway | None = None,
) -> VideoProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubVideoProvider()
    if provider_name.strip().lower() == "fal":
        return FalVideoProvider(gateway or FalGateway(api_key), model, cost_per_second_usd)
    raise NotImplementedError(f"Unsupported video provider {provider_name!r}; use 'stub' or 'fal'")
