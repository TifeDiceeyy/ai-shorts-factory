"""Turn extracted book pages into overlapping, searchable chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass

from . import config
from .extract import Book


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


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """Split a long string into size-limited windows with overlap."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = end - overlap
        if start <= 0:
            break
    return out


def chunk_book(book: Book) -> list[Chunk]:
    """Chunk a whole book into overlapping chunks with page provenance."""
    chunks: list[Chunk] = []
    buffer = ""
    buf_start = 0
    buf_end = 0

    def flush(buf: str, start: int, end: int):
        for piece in _split_long_text(buf, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append(
                Chunk(
                    id=f"{_slug(book.title)}-{len(chunks):04d}",
                    book=book.title,
                    pages=f"{start}" if start == end else f"{start}-{end}",
                    page_start=start,
                    page_end=end,
                    text=piece,
                )
            )

    for idx, page in enumerate(book.pages, start=1):
        page = (page or "").strip()
        if not page:
            continue
        # Keep pages separate enough for provenance: if adding this page would
        # blow the buffer, flush first, then carry a small overlap tail.
        if buffer and len(buffer) + len(page) + 2 > config.CHUNK_SIZE * 1.4:
            flush(buffer, buf_start, buf_end)
            tail = buffer[-config.CHUNK_OVERLAP:] if len(buffer) > config.CHUNK_OVERLAP else buffer
            buffer = tail
            buf_start = buf_end  # provenance for the tail
        if not buffer:
            buf_start = idx
        buffer = f"{buffer}\n{page}".strip()
        buf_end = idx

    if buffer.strip():
        flush(buffer, buf_start, buf_end)
    return chunks


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "book"
