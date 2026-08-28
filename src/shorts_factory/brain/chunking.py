"""The Chunk data structure — a book excerpt with page provenance.

Trimmed 2026-08-28: the ingestion logic that BUILT chunks from raw book
pages (chunk_book, _split_long_text) lived here too, reachable only via
Brain.build() — dead code, nothing in the real pipeline ever called it.
Removed along with extract.py, which it depended on. The Chunk class
itself stays: store.py loads already-built chunks from data/brain/
chunks.json via Chunk.from_dict(), and retrieval.py's SearchResult wraps
one per search hit — both genuinely live, used by the real
research_brief() path."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    book: str
    pages: str            # "12-14" or "12"
    page_start: int
    page_end: int
    text: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "book": self.book,
            "pages": self.pages,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)
