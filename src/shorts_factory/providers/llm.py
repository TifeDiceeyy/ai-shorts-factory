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
from typing import Any

from ..cost_tracker import CostTracker

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


def get_llm_provider(provider_name: str) -> LLMProvider:
    if provider_name.strip().lower() in ("", "stub"):
        return StubLLMProvider()
    raise NotImplementedError(
        f"LLM provider {provider_name!r} is not wired up yet — Phase 0 only "
        "implements the stub provider until a real provider/model is approved "
        "and a credential is supplied. See CLAUDE.md §0 rule: no paid calls "
        "without approval."
    )
