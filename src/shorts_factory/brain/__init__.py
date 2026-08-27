"""Faceless-Shorts Brain — a local, book-grounded knowledge engine.

The brain replaces an "internet research" step in a video pipeline with a
local knowledge base built from the two books:

    * The Knowledge — Lewis Dartnell
    * How to Invent Everything — Ryan North

Usage::

    from brain import Brain

    brain = Brain()                      # auto-builds on first use
    results = brain.search("what stops working when electricity dies")
    brief   = brain.research_brief("electricity disappears")
    ideas   = brain.ideas(n=10)
    script  = brain.script("What would happen if electricity suddenly disappeared?")

Everything runs locally with zero API keys. An optional OpenAI-compatible
LLM (env: BRAIN_LLM_*) can polish scripts.
"""
from __future__ import annotations

from pathlib import Path

from . import config
from .chunking import Chunk, chunk_book
from .extract import Book, extract_book, extract_book_txt
from .ideas import IdeaGenerator
from .retrieval import Retriever, SearchResult
from .scripting import ScriptEngine
from .store import BrainStore
from .topics import build_topics

__all__ = ["Brain", "Chunk", "Book", "SearchResult"]


class Brain:
    """Facade over extraction, retrieval, ideas and scripting."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        pdfs: list[str | Path] | None = None,
        txts: list[str | Path] | None = None,
        auto_build: bool = True,
    ):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.pdfs = [Path(p) for p in (pdfs or config.DEFAULT_PDFS) if Path(p).exists()]
        self.txts = [Path(p) for p in (txts or config.DEFAULT_TXTS) if Path(p).exists()]

        if auto_build and not self.is_built:
            self.build()

        if self.is_built:
            self._load()
        elif auto_build:
            raise RuntimeError(
                f"Brain data not found in {self.data_dir}. Build it first: "
                "python build_brain.py"
            )

    def _load(self) -> None:
        self.store = BrainStore.load(self.data_dir)
        self.retriever = Retriever(self.store)
        # Idea generation runs on the curated seed topics (clean, hook-friendly),
        # while store.topics remains the full browsable inventory.
        self.ideas_engine = IdeaGenerator(self.retriever)
        self.script_engine = ScriptEngine(self.retriever)

    # ------------------------------------------------------------------
    # Build / state
    # ------------------------------------------------------------------
    @property
    def is_built(self) -> bool:
        return (self.data_dir / "chunks.json").exists()

    def build(self, force: bool = False) -> dict:
        """Ingest all PDFs into the local knowledge base."""
        if self.is_built and not force:
            return self.stats()

        books: list[Book] = []
        for path in self.pdfs:
            if not path.exists():
                continue
            books.append(extract_book(path))
        for path in self.txts:
            if not path.exists():
                continue
            books.append(extract_book_txt(path))

        chunks: list[Chunk] = []
        for book in books:
            chunks.extend(chunk_book(book))

        topics = build_topics(self.pdfs, books)
        stats = {
            "built_at": _now(),
            "books": [{"title": b.title, "pages": b.n_pages, "path": b.path} for b in books],
            "n_chunks": len(chunks),
            "n_topics": len(topics),
            "total_chars": sum(len(c.text) for c in chunks),
        }
        self.store = BrainStore.build(chunks, topics, stats, self.data_dir)
        self.store.save()
        self.retriever = Retriever(self.store)
        self.ideas_engine = IdeaGenerator(self.retriever)
        self.script_engine = ScriptEngine(self.retriever)
        return stats

    def stats(self) -> dict:
        return BrainStore.load(self.data_dir).stats if self.is_built else {}

    # ------------------------------------------------------------------
    # Knowledge API (what your flow calls)
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K_DEFAULT) -> list[dict]:
        """Keyword/semantic search over the books. Returns cited chunks."""
        return [r.to_dict() for r in self.retriever.search(query, top_k=top_k)]

    def research_brief(self, topic: str, top_k: int = 10) -> dict:
        """Drop-in replacement for an internet-research step.

        Returns key facts + full cited context ready to paste into your
        existing LLM prompt.
        """
        return self.script_engine.research_brief(topic, top_k=top_k)

    def topics(self) -> list[dict]:
        return list(self.store.topics)

    # ------------------------------------------------------------------
    # Content API
    # ------------------------------------------------------------------
    def ideas(self, seed: str | None = None, n: int = 10) -> list[dict]:
        """Generate Shorts video ideas, ranked by book knowledge coverage."""
        return self.ideas_engine.generate(seed=seed, n=n)

    def script(self, topic: str, duration: int = config.SHORTS_DURATION_SECONDS) -> dict:
        """Generate a complete Shorts script (hook, beats, CTA, sources)."""
        return self.script_engine.generate(topic, duration=duration)


def _now() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")
