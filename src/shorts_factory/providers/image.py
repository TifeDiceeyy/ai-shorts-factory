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

from PIL import Image, ImageDraw, ImageOps

from ..cost_tracker import CostTracker
from .fal import FalGateway, media_url

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
        scene_index: int | str,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        ...


class StubImageProvider(ImageProvider):
    name = "stub"

    def generate_scene_image(
        self,
        scene: dict[str, Any],
        scene_index: int | str,
        out_path: Path,
        cost_tracker: CostTracker,
    ) -> Path:
        operation = f"image.generate_scene_image[{scene_index}]"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)

        pal_idx = scene_index if isinstance(scene_index, int) else (abs(hash(str(scene_index))) % len(GRADIENT_PALETTE))
        top, bottom = GRADIENT_PALETTE[pal_idx % len(GRADIENT_PALETTE)]
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


def get_image_model_args(model: str, style_preset: str) -> dict[str, Any]:
    """Model-specific arguments (aspect ratio/size, style) for a given
    fal.ai image model — each family has a different schema, confirmed live
    against fal.ai's docs and a real API call (2026-08-20) rather than
    assumed. Getting this wrong is exactly what's bitten this project
    before (a deprecated imagen3, a guessed-at imagen4 path that turned out
    not to exist at all)."""
    m = model.lower()
    if "nano-banana" in m or "gemini" in m or "imagen" in m:
        # Nano Banana (Gemini 2.5/3 Flash Image) and Google's Imagen family
        # both use aspect_ratio, no "style" param — confirmed via fal.ai's
        # nano-banana/api and imagen3/api docs.
        return {"aspect_ratio": "9:16"}
    # Recraft-v3 (and default/unknown): explicit image_size + style, since
    # Recraft's "style" param defaults to "realistic_image" when omitted.
    args: dict[str, Any] = {"image_size": "portrait_16_9"}
    if style_preset:
        args["style"] = style_preset
    return args


class FalImageProvider(ImageProvider):
    name = "fal"

    def __init__(
        self,
        gateway: FalGateway,
        model: str,
        cost_per_image_usd: float,
        visual_style: str = "",
        style_preset: str = "digital_illustration/hand_drawn_outline",
    ):
        if not model:
            raise ValueError("fal image generation requires IMAGE_MODEL")
        if cost_per_image_usd <= 0:
            raise ValueError("Set IMAGE_COST_PER_IMAGE_USD to a conservative positive estimate")
        self.gateway = gateway
        self.model = model.strip("/")
        self.cost = cost_per_image_usd
        self.visual_style = visual_style
        self.style_preset = style_preset

    def generate_scene_image(self, scene, scene_index, out_path, cost_tracker):
        operation = f"image.generate_scene_image[{scene_index}]"
        cost_tracker.check_budget(operation, self.cost)
        # Recraft-v3's "style" param defaults to "realistic_image" when
        # omitted (confirmed live against fal.ai's docs 2026-08-17) — a
        # prompt alone is not enough to keep it in illustration mode, and
        # the per-scene visual_prompt (written by the script-generation LLM)
        # is not reliable about restating the house style either. Both are
        # enforced here, server-side, regardless of what the LLM wrote.
        prompt = scene["visual_prompt"]
        if self.visual_style:
            prompt = f"{prompt}\n\nRequired art style, apply to every image without exception: {self.visual_style}"
        # Recraft-v3 hard-rejects prompts over 1000 characters (confirmed via
        # a real 422 error 2026-08-17: the LLM's own per-scene description
        # plus the appended house style together went over the limit).
        # Truncate from the end so the scene-specific content and the start
        # of the style reminder survive; only the least-essential trailing
        # style detail gets dropped. The no-text instruction below is added
        # AFTER truncation so it always survives regardless.
        no_text_instruction = "\n\nDo not render any text, words, letters, labels, or signs in the image."
        max_prompt_chars = 990 - len(no_text_instruction)
        if len(prompt) > max_prompt_chars:
            prompt = prompt[: max_prompt_chars - 1].rsplit(" ", 1)[0] + "…"
        # A real generation once rendered part of this character description
        # as literal text on a held sign (confirmed 2026-08-17) — the model
        # will sometimes turn descriptive language into on-screen lettering.
        # Our own captions are the only text that belongs in frame.
        prompt = f"{prompt}{no_text_instruction}"
        arguments: dict[str, Any] = {
            "prompt": prompt,
            "num_images": 1,
            **get_image_model_args(self.model, self.style_preset),
        }

        data = self.gateway.run(
            self.model,
            arguments,
        )
        image_url = media_url(data, "images", "0", "url")
        from io import BytesIO
        image = Image.open(BytesIO(self.gateway.download(image_url))).convert("RGB")
        image = ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        cost_tracker.record(self.name, operation, self.cost, self.cost, is_stub=False)
        return out_path


def get_image_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_image_usd: float = 0.0,
    gateway: FalGateway | None = None,
    visual_style: str = "",
    style_preset: str = "digital_illustration/hand_drawn_outline",
) -> ImageProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubImageProvider()
    if provider_name.strip().lower() == "fal":
        return FalImageProvider(
            gateway or FalGateway(api_key), model, cost_per_image_usd, visual_style, style_preset
        )
    raise NotImplementedError(f"Unsupported image provider {provider_name!r}; use 'stub' or 'fal'")
