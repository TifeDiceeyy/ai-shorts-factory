"""Rule-based green/yellow/red topic classifier (CLAUDE.md §5).

Phase 0 is intentionally simple and rule-based (a real per-claim classifier
is Phase 1+ work, once retrieval/verification exists). The one property that
must hold even at this simplicity: **fail closed**. A topic this table has
never seen is treated as RED, not green — an unrecognized topic must never
silently slip into the procedural pipeline.
"""
from __future__ import annotations

from enum import Enum


class SafetyClass(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


GREEN_TOPICS = {
    "concrete", "roman concrete", "rope", "pottery", "water filtration",
    "crop rotation", "basic compass", "compass",
}

YELLOW_TOPICS = {
    "soap", "furnaces", "furnace", "electricity", "food preservation",
    "apple cider vinegar", "charcoal", "simple mechanical water pump",
    "water pump",
}

# Explicit red list (CLAUDE.md examples) plus keyword triggers, since red
# topics are the ones most likely to be phrased in a way that isn't an exact
# table match ("how to make gunpowder", "gunpowder recipe", etc).
RED_TOPICS = {
    "weapons", "explosives", "explosive", "gunpowder", "gasoline",
    "fuel synthesis", "toxic chemistry", "unsafe medicine",
}

RED_KEYWORDS = (
    "gunpowder", "explosive", "bomb", "weapon", "gun ", "firearm",
    "poison", "toxin", "nerve agent", "chemical weapon", "napalm",
    "molotov", "ammunition",
)

YELLOW_CAUTION = {
    "soap": (
        "Caution: lye (sodium/potassium hydroxide) is caustic. It burns skin "
        "and eyes on contact and releases irritating fumes when dissolved. "
        "Handle only with eye protection, gloves, and ventilation."
    ),
}


def _normalize(topic: str) -> str:
    return " ".join(topic.strip().lower().split())


def classify_topic(topic: str) -> SafetyClass:
    t = _normalize(topic)

    if t in RED_TOPICS or any(kw in t for kw in RED_KEYWORDS):
        return SafetyClass.RED

    if t in YELLOW_TOPICS:
        return SafetyClass.YELLOW

    if t in GREEN_TOPICS:
        return SafetyClass.GREEN

    # Fail closed: an unrecognized topic is not proven safe.
    return SafetyClass.RED


class TopicBlocked(Exception):
    """Raised when a topic is classified RED and the pipeline must not proceed."""

    def __init__(self, topic: str):
        self.topic = topic
        super().__init__(
            f"Topic {topic!r} is classified RED (or unrecognized, fail-closed to RED). "
            "No actionable instructions may be generated. Blocked before brief/script/render."
        )


def enforce_not_blocked(topic: str) -> SafetyClass:
    """Gate function: call before any brief/script/render step. Raises TopicBlocked
    for RED topics so the pipeline halts before touching a provider."""
    cls = classify_topic(topic)
    if cls == SafetyClass.RED:
        raise TopicBlocked(topic)
    return cls


def caution_line(topic: str) -> str | None:
    return YELLOW_CAUTION.get(_normalize(topic))
