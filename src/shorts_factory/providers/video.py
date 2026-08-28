"""Video provider interface: image-to-video animation for one scene.

StubVideoProvider makes a short, deterministic, zero-cost local clip (holds
the source hero image for a fixed duration) so the pipeline stays fully
testable without real network calls or spend — same house pattern as every
other provider here.

FalVideoProvider animates base images via image-to-video models on fal.ai
(e.g., Kling 1.5 Pro or MiniMax Hailuo-02).
The animation pipeline uses a hybrid design (see pipeline._scene_base_image_path):
  - ONE shared "hero" character image is generated once per video.
  - Every scene then generates its OWN base image: ingredient_grid/process_action
    scenes render with no character; mascot scenes render edited FROM the hero
    image (image-to-image, when the model supports it) so pose/composition
    varies per scene while the character stays recognizable.
  - Each scene's base image is then animated via FalVideoProvider into an MP4 clip.
  - Cost model: 1 hero image + 1 image per scene + N video clips.
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

# A real Kling 1.5 Pro call was timed at ~8-9 minutes per 5s clip (confirmed
# live, 2026-08-20) — the previous 600s (10min) timeout was too tight and a
# real generation crashed on it after 3 clips had already been paid for.
# Generous margin above the slowest observed model.
VIDEO_GEN_TIMEOUT_S = 1200
NONVERBAL_CONTINUOUS_MOTION = (
    "Permitted nonverbal movement only: subtle breathing, blinking and gaze shifts, and — where the scene's "
    "own motion instruction calls for it — hand/arm gestures, movement of a hand-held prop, or a slight leg/"
    "weight-shift. The mouth must remain fully closed at all times — no opening, no talking, no mouthing "
    "words, no lip movement, and no jaw movement whatsoever. The character must not speak or lip-sync to the "
    "narration under any circumstance: no talking mouth shapes. The frame overall must never go completely "
    "static for the whole clip — but that continuous motion should come from breathing/blinking or an "
    "animated prop/environment (per the scene's own motion instruction), not from the character bouncing, "
    "hopping, or performing repeated idle movement."
)


# Confirmed for real (2026-08-28): the NONVERBAL_CONTINUOUS_MOTION prompt
# instruction alone did not stop Kling from opening/moving the mouth in a
# real generated clip — negative_prompt is a second, independent lever
# (fal.ai's Kling docs list it separately from the main prompt) worth
# stacking rather than relying on prompt wording alone.
KLING_NEGATIVE_PROMPT = (
    "static, frozen, still image, motionless, frozen pose, no movement, paused, "
    "talking, speaking, open mouth, moving mouth, moving lips, lip sync, mouth movement, "
    "blur, distort, low quality"
)
# fal.ai's documented default is 0.5 (confirmed via API docs, 2026-08-27). Raising
# it pushes the model to follow the motion prompt (NONVERBAL_CONTINUOUS_MOTION +
# Mascot.build_scene_motion_prompt) more literally, at the cost of some prompt
# creativity — an explicit trade we want here since under-animation (not
# over-literalness) is the failure mode we've actually hit.
KLING_CFG_SCALE = 0.7


def get_video_model_config(model: str) -> tuple[float, dict[str, Any]]:
    """Returns (clip_duration_seconds, extra_arguments_dict) for a given video model."""
    m = model.lower()
    if "kling" in m:
        return KLING_CLIP_SECONDS, {
            "duration": "5",
            "aspect_ratio": "9:16",
            "negative_prompt": KLING_NEGATIVE_PROMPT,
            "cfg_scale": KLING_CFG_SCALE,
        }
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
        motion_prompt: str = "",
    ) -> Path:
        ...


class StubVideoProvider(VideoProvider):
    name = "stub"

    def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker, motion_prompt=""):
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

    def generate_scene_video(self, scene, hero_image_path, scene_index, out_path, cost_tracker, motion_prompt=""):
        operation = f"video.generate_scene_video[{scene_index}]"
        cost_tracker.check_budget(operation, self.cost)
        image_url = self.gateway.upload(hero_image_path)
        # Video generation routinely takes several minutes — confirmed live
        # 2026-08-17: the shared FalGateway.run() default of 180s (fine for
        # LLM/TTS/image calls) timed out on a real call before it finished rendering.
        # motion_prompt must describe the SAME shot the base image actually
        # shows — the caller builds it from the identical
        # get_scene_image_prompt()/mascot.build_scene_prompt() call used for
        # the image itself. scene["visual_prompt"] (the LLM's own, separate
        # free-text field) is NOT used here: pipeline.get_scene_image_prompt()
        # already discards it in favor of the reconstructed mascot prompt for
        # any scene with structured fields, so animating with the raw
        # visual_prompt could describe a different pose/composition/layout
        # than what's actually in the frame being animated (confirmed as a
        # real mismatch risk in review 2026-08-21).
        prompt = f"{motion_prompt or scene['visual_prompt']} {NONVERBAL_CONTINUOUS_MOTION}"
        arguments = {
            "image_url": image_url,
            "prompt": prompt,
            **self.model_args,
        }

        data = self.gateway.run(
            self.model,
            arguments,
            timeout=VIDEO_GEN_TIMEOUT_S,
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
