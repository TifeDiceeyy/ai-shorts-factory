"""Image generation provider interface.

StubImageProvider renders a deterministic local gradient (Pillow, zero
network, zero cost) — visually distinct from the plain solid-color frames
used in the Phase 0 step-5 placeholder stage, specifically so the "swap
placeholders for generated images" step (step 8) is provably exercising a
different code path rather than silently reusing the same image.

A real provider (fal.ai, OpenAI images, ...) plugs in behind the same
generate_scene_image() signature once IMAGE_PROVIDER is approved and
configured.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw

from ..cost_tracker import CostTracker

WIDTH = 1080
HEIGHT = 1920

# Deterministic palette (not random) so re-runs are byte-identical.
GRADIENT_PALETTE = [
    ((60, 40, 20), (150, 110, 60)),   # workshop browns
    ((30, 50, 45), (90, 130, 100)),   # muted green
    ((45, 35, 55), (120, 95, 140)),   # dusk violet
    ((55, 45, 25), (170, 140, 70)),   # amber
    ((25, 35, 50), (80, 105, 140)),   # slate blue
    ((50, 30, 30), (150, 90, 70)),    # rust
]


class ImageProvider(ABC):
    name: str

    @abstractmethod
    def generate_scene_image(
        self,
        scene: dict[str, Any],
        scene_index: int,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        ...


class StubImageProvider(ImageProvider):
    name = "stub"

    def generate_scene_image(
        self,
        scene: dict[str, Any],
        scene_index: int,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        operation = f"image.generate_scene_image[{scene_index}]"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)

        top, bottom = GRADIENT_PALETTE[scene_index % len(GRADIENT_PALETTE)]
        img = Image.new("RGB", (WIDTH, HEIGHT))
        draw = ImageDraw.Draw(img)
        for y in range(HEIGHT):
            t = y / (HEIGHT - 1)
            r = int(top[0] + (bottom[0] - top[0]) * t)
            g = int(top[1] + (bottom[1] - top[1]) * t)
            b = int(top[2] + (bottom[2] - top[2]) * t)
            draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return out_path


class FalImageProvider(ImageProvider):
    name = "fal"

    def __init__(self, api_key: str, model: str, cost_per_image_usd: float):
        if not api_key or not model:
            raise ValueError("fal requires FAL_KEY and IMAGE_MODEL")
        if cost_per_image_usd <= 0:
            raise ValueError("Set IMAGE_COST_PER_IMAGE_USD to a conservative positive estimate")
        self.api_key = api_key
        self.model = model.strip("/")
        self.cost = cost_per_image_usd

    def generate_scene_image(self, scene, scene_index, out_path, cost_tracker):
        operation = f"image.generate_scene_image[{scene_index}]"
        cost_tracker.check_budget(operation, self.cost)
        response = requests.post(
            f"https://fal.run/{self.model}",
            headers={"Authorization": f"Key {self.api_key}", "content-type": "application/json"},
            json={"prompt": scene["visual_prompt"], "image_size": "portrait_16_9", "num_images": 1},
            timeout=180,
        )
        response.raise_for_status()
        images = response.json().get("images") or []
        if not images or not images[0].get("url"):
            raise ValueError("fal response did not contain images[0].url")
        downloaded = requests.get(images[0]["url"], timeout=120)
        downloaded.raise_for_status()
        from io import BytesIO
        image = Image.open(BytesIO(downloaded.content)).convert("RGB")
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        cost_tracker.record(self.name, operation, self.cost, self.cost, is_stub=False)
        return out_path


def get_image_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_image_usd: float = 0.0,
) -> ImageProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubImageProvider()
    if provider_name.strip().lower() == "fal":
        return FalImageProvider(api_key, model, cost_per_image_usd)
    raise NotImplementedError(f"Unsupported image provider {provider_name!r}; use 'stub' or 'fal'")
