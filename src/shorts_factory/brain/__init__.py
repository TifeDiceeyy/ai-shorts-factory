"""Faceless-Shorts Brain — a local, book-grounded knowledge engine.

Replaces an "internet research" step in the main pipeline
(shorts_factory.brain_integration) with a local knowledge base built from
two books (The Knowledge — Lewis Dartnell; How to Invent Everything — Ryan
North) plus a third rebuild-civilization text. The index itself is built
and stored outside this repo (data/brain/, gitignored); this package only
loads and queries an already-built index — it does not ingest PDFs itself.

Usage::

    from shorts_factory.brain import Brain

    brain = Brain(data_dir="data/brain")
    brief = brain.research_brief("electricity disappears")

Trimmed 2026-08-28: this facade used to also offer PDF/OCR ingestion
(`.build()`), idea generation (`.ideas()`), and full script generation
(`.script()`, including an optional LLM-polish mode) — all dead code, only
reachable via this same facade's own now-removed methods, never called
from the real pipeline. `research_brief()` is the only method
`brain_integration.py` actually uses.
"""
from __future__ import annotations

from pathlib import Path

from . import config
from .retrieval import Retriever
from .scripting import ScriptEngine
from .store import BrainStore

__all__ = ["Brain"]


class Brain:
    """Facade over the local book index's retrieval-brief API."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        if not self.is_built:
            raise RuntimeError(f"Brain data not found in {self.data_dir}")
        self._load()

    def _load(self) -> None:
        self.store = BrainStore.load(self.data_dir)
        self.retriever = Retriever(self.store)
        self.script_engine = ScriptEngine(self.retriever)

    @property
    def is_built(self) -> bool:
        return (self.data_dir / "chunks.json").exists()

    def stats(self) -> dict:
        return BrainStore.load(self.data_dir).stats if self.is_built else {}

    def research_brief(self, topic: str, top_k: int = 10) -> dict:
        """Drop-in replacement for an internet-research step.

        Returns key facts + full cited context ready to paste into your
        existing LLM prompt.
        """
        return self.script_engine.research_brief(topic, top_k=top_k)
