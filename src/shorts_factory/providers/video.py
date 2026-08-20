"""Video provider interface: image-to-video animation for one scene.

StubVideoProvider makes a short, deterministic, zero-cost local clip (holds
the source hero image for a fixed duration) so the pipeline stays fully
testable without real network calls or spend — same house pattern as every
other provider here.

FalVideoProvider animates base images via image-to-video models on fal.ai
(e.g., Kling 1.5 Pro or MiniMax Hailuo-02).
The animation pipeline uses a hybrid design:
  - Mascot/character scenes reuse a single shared "hero" character image
    (generated once per video) to maintain visual consistency across scenes.
  - Non-mascot scenes (e.g. ingredient_grid, process_action) generate a
    per-scene base image.
  - Each scene's base image is then animated via FalVideoProvider into an MP4 clip.
  - Cost model: 1 hero image + N_non_mascot base images + N video clips.
  - Assembly pads or trims each clip to match the scene's actual audio duration.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..cost_tracker import CostTracker
from .fal import FalGateway, media_url

HAILUO_CLIP_SECONDS = 6.0
KLING_CLIP_SECONDS = 5.0
LUMA_CLIP_SECONDS = 5.0


def get_video_model_config(model: str) -> tuple[float, dict[str, Any]]:
    """Returns (clip_duration_seconds, extra_arguments_dict) for a given video model."""
    m = model.lower()
    if "kling" in m:
        return KLING_CLIP_SECONDS, {"duration": "5", "aspect_ratio": "9:16"}
    if "luma" in m:
        return LUMA_CLIP_SECONDS, {"aspect_ratio": "9:16"}
    # Default to MiniMax / Hailuo
    return HAILUO_CLIP_SECONDS, {"duration": "6", "resolution": "768P"}


def extract_video_url(data: dict[str, Any]) -> str:
    """Extracts video URL from Fal response across candidate key paths,
    raising an informative error naming top-level keys and attempted paths if all fail."""
    candidate_paths: list[tuple[str, ...]] = [
        ("video", "url"),
        ("video",),
        ("output", "url"),
    ]
    failures: list[str] = []
    for path in candidate_paths:
        try:
            return media_url(data, *path)
        except Exception as err:
            failures.append(f"path {path}: {err}")
    top_keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
    raise KeyError(
        f"Could not extract video URL from fal response. Top-level keys: {top_keys}. "
        f"Attempted paths: {'; '.join(failures)}"
    )


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
        self.clip_seconds, self.model_args = get_video_model_config(self.model)
        self.cost = cost_per_second_usd * self.clip_seconds

    def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker):
        operation = f"video.generate_scene_video[{scene_index}]"
        cost_tracker.check_budget(operation, self.cost)
        image_url = self.gateway.upload(hero_image_path)
        # Video generation routinely takes several minutes — confirmed live
        # 2026-08-17: the shared FalGateway.run() default of 180s (fine for
        # LLM/TTS/image calls) timed out on a real call before it finished rendering.
        prompt = scene["visual_prompt"]
        arguments = {
            "image_url": image_url,
            "prompt": prompt,
            **self.model_args,
        }

        data = self.gateway.run(
            self.model,
            arguments,
            timeout=600,
        )
        video_url = extract_video_url(data)
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
