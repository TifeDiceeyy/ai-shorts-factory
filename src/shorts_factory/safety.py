"""Rule-based green/yellow/red topic classifier (CLAUDE.md §5).

Phase 0 is intentionally simple and rule-based (a real per-claim classifier
is Phase 1+ work, once retrieval/verification exists). The one property that
must hold even at this simplicity: **fail closed**. A topic the registry
(topic_registry.py) has never seen is treated as RED, not green — an
unrecognized topic must never silently slip into the procedural pipeline.
"""
from __future__ import annotations

from enum import Enum

from .topic_registry import get_topic, normalize_topic


class SafetyClass(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Explicit red list (CLAUDE.md examples) plus keyword triggers, since red
# topics are the ones most likely to be phrased in a way that isn't an exact
# table match ("how to make gunpowder", "gunpowder recipe", etc). Deliberately
# NOT part of the registry — see topic_registry.py's docstring for why red
# topics must stay unregistered rather than stored anywhere as "known".
RED_TOPICS = {
    "weapons", "explosives", "explosive", "gunpowder", "gasoline",
    "fuel synthesis", "toxic chemistry", "unsafe medicine",
}

RED_KEYWORDS = (
    "gunpowder", "explosive", "bomb", "weapon", "gun ", "firearm",
    "poison", "toxin", "nerve agent", "chemical weapon", "napalm",
    "molotov", "ammunition",
)


def is_explicitly_red(topic: str) -> bool:
    """The keyword/exact-match half of the red check only — deliberately
    separate from classify_topic()'s fail-closed "unrecognized == red" rule,
    so callers that need to ask "does this name itself look dangerous?"
    (e.g. topic_registry.register_topic, before the topic is registered)
    don't get a false positive just because it isn't registered yet."""
    t = normalize_topic(topic)
    return t in RED_TOPICS or any(kw in t for kw in RED_KEYWORDS)


def classify_topic(topic: str) -> SafetyClass:
    t = normalize_topic(topic)

    if is_explicitly_red(t):
        return SafetyClass.RED

    entry = get_topic(t)
    if entry is not None:
        if entry.get("safety_class") == "yellow":
            return SafetyClass.YELLOW
        if entry.get("safety_class") == "green":
            return SafetyClass.GREEN

    # Fail closed: an unrecognized (or malformed-entry) topic is not proven safe.
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
    entry = get_topic(topic)
    return entry.get("caution") if entry else None


def caution_caption(topic: str) -> str | None:
    if classify_topic(topic) != SafetyClass.YELLOW:
        return None
    return "CAUTION: Educational overview — follow current expert safety guidance."
