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


def get_image_provider(provider_name: str) -> ImageProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubImageProvider()
    raise NotImplementedError(
        f"Image provider {provider_name!r} is not wired up yet — Phase 0 only "
        "implements the stub provider until a real provider/model is approved "
        "and a credential is supplied."
    )
