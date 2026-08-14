"""LLM provider interface for script generation.

Phase 0 ships one implementation: StubLLMProvider — deterministic, local,
zero network, zero cost. It still goes through CostTracker.check_budget()
and .record() so the budget-guard wiring is exercised and tested even
though the actual cost is $0. A real provider (Claude, OpenAI, ...) plugs
in behind the same generate_script() signature once LLM_PROVIDER is set to
something other than "stub" and a credential is present — CLAUDE.md §0
rule says don't make that paid call until approved, so it isn't wired yet.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Any

from ..cost_tracker import CostTracker
from .fal import FalGateway

TARGET_TOTAL_SECONDS = 45.0
MIN_SCENE_SECONDS = 3.0
MAX_SCENE_SECONDS = 9.5

# Cycled by scene index for basic visual variety — a template default, not a
# creative decision. A real LLM/storyboard artist would choose these
# per-shot instead of round-robin.
CAMERA_TEMPLATES = [
    "static wide shot",
    "slow push-in",
    "close-up on hands/detail",
    "overhead top-down",
    "handheld tracking shot",
]


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_script(
        self,
        brief: dict[str, Any],
        language: str,
        visual_style: str,
        cost_tracker: CostTracker,
    ) -> dict[str, Any]:
        ...


def _caption_from_claim(claim_text: str) -> str:
    """Short on-screen caption derived from a claim (schema caps at 90 chars)."""
    cap = claim_text.strip()
    if len(cap) <= 90:
        return cap
    cut = cap[:87].rsplit(" ", 1)[0]
    return cut + "..."


def _visual_prompt(claim_text: str, visual_style: str) -> str:
    return f"{visual_style}: {claim_text}"


class StubLLMProvider(LLMProvider):
    """Deterministic template generator standing in for a real LLM call.
    Turns each brief claim into exactly one scene, scaled so total duration
    lands in the 40-50s window CLAUDE.md requires for a Short."""

    name = "stub"

    def generate_script(
        self,
        brief: dict[str, Any],
        language: str,
        visual_style: str,
        cost_tracker: CostTracker,
    ) -> dict[str, Any]:
        operation = "llm.generate_script"
        cost_tracker.check_budget(operation, estimated_cost_usd=0.0)

        claims = brief["claims"]
        raw_durations = []
        for c in claims:
            word_count = len(c["claim"].split())
            raw = 2.5 + 0.35 * word_count
            raw_durations.append(max(MIN_SCENE_SECONDS, min(MAX_SCENE_SECONDS, raw)))

        raw_total = sum(raw_durations)
        scale = TARGET_TOTAL_SECONDS / raw_total if raw_total else 1.0
        durations = [round(d * scale, 1) for d in raw_durations]

        scenes = []
        for i, (claim, duration) in enumerate(zip(claims, durations)):
            scenes.append(
                {
                    "narration": claim["claim"],
                    "caption": _caption_from_claim(claim["claim"]),
                    "duration": duration,
                    "visual_prompt": _visual_prompt(claim["claim"], visual_style),
                    "source_claim_id": claim["id"],
                    "camera": CAMERA_TEMPLATES[i % len(CAMERA_TEMPLATES)],
                    "sfx": None,
                }
            )

        script = {
            "topic": brief["topic"],
            "language": language,
            "visual_style": visual_style,
            "scenes": scenes,
        }

        cost_tracker.record(
            provider=self.name,
            operation=operation,
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
            is_stub=True,
        )
        return script


def _json_object(text: str) -> dict[str, Any]:
    """Decode a provider response without accepting prose around the object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _script_prompt(brief: dict[str, Any], language: str, visual_style: str) -> str:
    return (
        "Create a factual YouTube Short script lasting 40-50 seconds. Return JSON only. "
        "Use only the supplied claims; do not add factual claims. Make the first scene a strong hook. "
        "Every scene must contain narration, caption (max 90 characters), duration (3-9.5 seconds), "
        "visual_prompt, source_claim_id, camera, and sfx (string or null). The top-level object must "
        "contain topic, language, visual_style, and scenes. Preserve claim IDs exactly. "
        f"Language: {language}. Visual style: {visual_style}. Brief: {json.dumps(brief, ensure_ascii=False)}"
    )


class FalLLMProvider(LLMProvider):
    """Routes through fal.ai's OpenRouter endpoint (openrouter/router) —
    fal-native schema, not fal-ai/any-llm (confirmed deprecated 2026-08-14,
    do not use). model is an OpenRouter-style id, e.g. "anthropic/claude-3.5-sonnet".
    The response echoes real per-call cost (usage.cost), so the ACTUAL spend
    recorded is the real figure, not just the pre-call estimate — the
    estimate is still required up front only because check_budget() has to
    run before we know the real price."""

    name = "fal"

    def __init__(self, gateway: FalGateway, model: str, cost_per_script_usd: float, endpoint: str = "openrouter/router"):
        if not model:
            raise ValueError("fal LLM requires LLM_MODEL (for example 'google/gemini-2.5-flash')")
        if cost_per_script_usd <= 0:
            raise ValueError("Set LLM_COST_PER_SCRIPT_USD to a conservative positive pre-call estimate")
        self.gateway = gateway
        self.endpoint = endpoint
        self.model = model
        self.estimate = cost_per_script_usd

    def generate_script(self, brief, language, visual_style, cost_tracker):
        operation = "llm.generate_script"
        cost_tracker.check_budget(operation, self.estimate)
        data = self.gateway.run(
            self.endpoint,
            {
                "model": self.model,
                "prompt": _script_prompt(brief, language, visual_style),
                "temperature": 0.4,
                "max_tokens": 2500,
            },
        )
        script = _json_object(data["output"])
        actual_cost = float(data.get("usage", {}).get("cost", self.estimate))
        cost_tracker.record(self.name, operation, self.estimate, actual_cost, is_stub=False)
        return script


def get_llm_provider(
    provider_name: str,
    api_key: str = "",
    model: str = "",
    cost_per_script_usd: float = 0.0,
    gateway: FalGateway | None = None,
    endpoint: str = "openrouter/router",
) -> LLMProvider:
    name = provider_name.strip().lower()
    if name in ("", "stub"):
        return StubLLMProvider()
    if name == "fal":
        return FalLLMProvider(gateway or FalGateway(api_key), model, cost_per_script_usd, endpoint)
    raise NotImplementedError(f"Unsupported LLM provider {provider_name!r}; use 'stub' or 'fal'")
