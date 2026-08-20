"""Per-topic search queries and keywords for Phase 1 retrieval.

Backed by the runtime-editable registry in topic_registry.py
(data/topic_registry.json) so a topic registered at runtime — e.g. from the
Telegram bot's new-topic flow — is visible immediately, with no restart.
TOPIC_QUERIES/TOPIC_KEYWORDS keep the original dict-like interface (`.get`,
`in`, iteration) so retrieval.py needs no changes.
"""
from __future__ import annotations

from collections.abc import Mapping

from .topic_registry import load_registry, normalize_topic


class _RegistryField(Mapping):
    """Read-only dict-like view over one field of the topic registry, always
    reflecting the current on-disk state."""

    def __init__(self, field: str):
        self._field = field

    def __getitem__(self, topic: str) -> list[str]:
        entry = load_registry().get(normalize_topic(topic))
        value = entry.get(self._field) if entry else None
        if not value:
            raise KeyError(topic)
        return value

    def __iter__(self):
        return (name for name, entry in load_registry().items() if entry.get(self._field))

    def __len__(self) -> int:
        return sum(1 for _ in self)


TOPIC_QUERIES: Mapping[str, list[str]] = _RegistryField("queries")
TOPIC_KEYWORDS: Mapping[str, list[str]] = _RegistryField("keywords")
