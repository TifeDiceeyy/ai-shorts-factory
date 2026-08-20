"""Phase 2: idea generator.

N ranked concepts per topic, each with 5 hook variants, a payoff, a
visual-potential score, safety class, source availability (from the Phase 1
citation store), and novelty vs. recently-generated ideas.

The creative part (concept/angle/hooks/payoff/series) comes from
LLMProvider.propose_ideas() — stub by default (deterministic templates, zero
cost, see providers/llm.py's CONCEPT_TEMPLATES/HOOK_TEMPLATES), a real LLM
once LLM_PROVIDER is configured — same real/stub split every other provider
in this codebase uses. The analytical part below (safety class,
visual-potential score, source availability, novelty-vs-recent, rank_score)
stays here regardless of which provider answered — it's scoring logic, not
creative content, and it's what keeps a real LLM's ideas accountable to the
same rules a template idea has to follow.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import load_settings, require_budget_approval_if_paid
from .cost_tracker import CostTracker
from .providers.fal import FalGateway
from .providers.llm import get_llm_provider
from .safety import enforce_not_blocked

REPO_ROOT = Path(__file__).resolve().parents[2]
IDEA_HISTORY_PATH = REPO_ROOT / "data" / "idea_history.json"

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

# Rough, honest heuristic: topics involving hands-on physical transformation
# (fire, tools, liquid-to-solid, visible reactions) tend to film better than
# purely abstract/procedural ones. Not measured against real audience data —
# there isn't any yet (Phase 6 is what would eventually replace this).
VISUAL_POTENTIAL_KEYWORDS = {
    "fire": 0.15, "heat": 0.1, "melt": 0.15, "boil": 0.1, "mix": 0.05,
    "burn": 0.1, "carve": 0.1, "weave": 0.1, "pour": 0.05, "harden": 0.05,
    "grind": 0.05, "ferment": 0.05, "dry": 0.03, "shape": 0.05,
}
BASE_VISUAL_SCORE = 0.5


@dataclass
class Hook:
    text: str
    variant_index: int


@dataclass
class Idea:
    topic: str
    concept: str
    angle: str
    hooks: list[Hook]
    payoff: str
    series: str | None
    safety_class: str
    visual_potential_score: float
    source_availability: dict
    similarity_to_recent: float
    rank_score: float


def _visual_potential_score(topic: str, citation_claims_text: str) -> float:
    text = f"{topic} {citation_claims_text}".lower()
    score = BASE_VISUAL_SCORE
    for kw, weight in VISUAL_POTENTIAL_KEYWORDS.items():
        if kw in text:
            score += weight
    return round(min(1.0, score), 3)


def _content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 2}


def load_source_availability(topic: str) -> dict:
    """Pulls from the Phase 1 citation store if one exists for this topic."""
    path = REPO_ROOT / "data" / topic.replace(" ", "_") / f"{topic.replace(' ', '_')}.citations.json"
    if not path.exists():
        return {"available": False, "verified_claim_count": 0, "total_claim_count": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "available": True,
        "verified_claim_count": payload.get("verified_count", 0),
        "total_claim_count": payload.get("citation_count", 0),
    }


def load_idea_history() -> list[dict]:
    if not IDEA_HISTORY_PATH.exists():
        return []
    return json.loads(IDEA_HISTORY_PATH.read_text(encoding="utf-8"))


def _similarity_to_recent(concept_text: str, history: list[dict]) -> float:
    if not history:
        return 0.0
    concept_words = _content_words(concept_text)
    if not concept_words:
        return 0.0
    best = 0.0
    for entry in history:
        past_words = _content_words(entry.get("concept", ""))
        if not past_words:
            continue
        overlap = len(concept_words & past_words) / len(concept_words | past_words)
        best = max(best, overlap)
    return round(best, 3)


def generate_ideas(topic: str, n: int = 3) -> list[Idea]:
    """Blocks RED topics at the source, same as the render pipeline — an
    idea must never be generated for a topic that can't legally render."""
    safety_class = enforce_not_blocked(topic)  # raises TopicBlocked for RED — must halt before any idea is generated
    availability = load_source_availability(topic)
    history = load_idea_history()

    settings = load_settings()
    require_budget_approval_if_paid(settings)
    cost_tracker = CostTracker(budget_cap_usd=settings.budget_cap_usd)
    uses_fal = settings.llm.provider.strip().lower() == "fal"
    fal_gateway = FalGateway(settings.fal_key) if uses_fal else None
    llm = get_llm_provider(
        settings.llm.provider,
        settings.credential_for(settings.llm),
        settings.llm.model_or_voice,
        settings.llm_cost_per_script_usd,
        gateway=fal_gateway,
        endpoint=settings.fal_llm_endpoint,
    )
    raw_ideas = llm.propose_ideas(topic, n, cost_tracker)

    ideas: list[Idea] = []
    for raw in raw_ideas:
        concept = raw["concept"]
        hooks = [Hook(text=t, variant_index=i) for i, t in enumerate(raw["hooks"], start=1)]
        visual_score = _visual_potential_score(topic, "")
        similarity = _similarity_to_recent(concept, history)

        source_ratio = (
            availability["verified_claim_count"] / availability["total_claim_count"]
            if availability["available"] and availability["total_claim_count"] > 0
            else 0.0
        )
        rank_score = round(
            0.4 * visual_score + 0.3 * source_ratio + 0.3 * (1 - similarity), 3
        )

        ideas.append(
            Idea(
                topic=topic,
                concept=concept,
                angle=raw.get("angle", ""),
                hooks=hooks,
                payoff=raw.get("payoff")
                or f"Viewer walks away knowing the real, historically-grounded steps behind {topic}.",
                series=raw.get("series"),
                safety_class=safety_class.value,
                visual_potential_score=visual_score,
                source_availability=availability,
                similarity_to_recent=similarity,
                rank_score=rank_score,
            )
        )

    ideas.sort(key=lambda i: i.rank_score, reverse=True)
    return ideas


def record_idea_chosen(idea: Idea) -> None:
    """Append to idea history so future similarity_to_recent scoring accounts
    for it — this is what keeps a real series from picking the same angle
    twice in a row."""
    history = load_idea_history()
    history.append({
        "topic": idea.topic,
        "concept": idea.concept,
        "rank_score": idea.rank_score,
    })
    IDEA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDEA_HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def ideas_to_dicts(ideas: list[Idea]) -> list[dict]:
    return [
        {
            "topic": i.topic,
            "concept": i.concept,
            "angle": i.angle,
            "hooks": [{"text": h.text, "variant_index": h.variant_index} for h in i.hooks],
            "payoff": i.payoff,
            "series": i.series,
            "safety_class": i.safety_class,
            "visual_potential_score": i.visual_potential_score,
            "source_availability": i.source_availability,
            "similarity_to_recent": i.similarity_to_recent,
            "rank_score": i.rank_score,
        }
        for i in ideas
    ]
