"""PDF -> clean text extraction.

Uses pypdf (pure Python, no heavy dependencies). Also strips boilerplate
lines that repeat across many pages (running headers, page numbers).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from . import config


@dataclass
class Book:
    title: str
    path: str
    pages: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def _clean_page(text: str) -> str:
    """Normalise whitespace and drop obvious junk from one page of text."""
    if not text:
        return ""
    # Normalise unicode spaces and hyphenation artifacts
    text = text.replace("\u00a0", " ")
    # The replacement char almost always stands in for an apostrophe in
    # these PDFs ("there\ufffds" -> "there's"). Keep contractions readable.
    text = text.replace("\ufffd", "'")
    # Normalise typographic punctuation so downstream TTS/LLM steps get
    # clean ASCII text.
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u00d7", "x").replace("\u2026", "...")
    text = re.sub(r"-\s*\n\s*", "", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _filter_boilerplate(pages: list[str]) -> list[str]:
    """Remove lines that repeat across many pages (headers/footers/watermarks)."""
    if not pages:
        return pages
    line_counts = Counter()
    for page in pages:
        seen = set()
        for line in page.splitlines():
            if len(line) > 3 and line not in seen:
                line_counts[line] += 1
                seen.add(line)
    threshold = max(3, int(len(pages) * config.BOILERPLATE_LINE_RATIO))
    junk = {ln for ln, c in line_counts.items() if c >= threshold}
    if not junk:
        return pages
    out = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if ln not in junk]
        out.append("\n".join(kept))
    return out


def extract_book(path: str | Path, title: str | None = None) -> Book:
    """Extract all pages of a PDF into a :class:`Book`."""
    path = Path(path)
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(_clean_page(page.extract_text() or ""))
        except Exception:
            pages.append("")

    title = title or _title_from_filename(path.name)
    pages = [p for p in pages if len(p) >= config.MIN_CHUNK_CHARS] or pages
    pages = _filter_boilerplate(pages)
    return Book(title=title, path=str(path), pages=pages)


def extract_book_txt(path: str | Path, title: str | None = None) -> Book:
    """Extract a plain text file (pages separated by form feeds) into a Book.

    This is the ingest path for scanned PDFs that have no text layer: run
    ``python ocr_pdf.py book.pdf`` first, then feed the resulting .txt here.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    pages = [_clean_page(p) for p in raw.split("\f")]

    title = title or _title_from_filename(path.name)
    pages = [p for p in pages if len(p) >= config.MIN_CHUNK_CHARS] or pages
    pages = _filter_boilerplate(pages)
    return Book(title=title, path=str(path), pages=pages)


def extract_outline(path: str | Path) -> list[str]:
    """Pull the PDF bookmark/outline tree (chapter/section titles)."""
    path = Path(path)
    reader = PdfReader(str(path))
    titles: list[str] = []

    def walk(items, depth=0):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
            else:
                t = getattr(item, "title", None) or getattr(item, "/Title", None)
                if t and str(t).strip():
                    titles.append("  " * depth + str(t).strip())

    try:
        walk(reader.outline)
    except Exception:
        pass
    return titles


def _title_from_filename(name: str) -> str:
    name = Path(name).stem
    # Human-friendly titles for the known books
    key = name.lower().replace("_", " ")
    if "knowledge" in key:
        return "The Knowledge"
    if "invent everything" in key:
        return "How to Invent Everything"
    if "ultimate guide" in key or "rebuild" in key:
        return "The Book: Ultimate Guide to Rebuilding Civilization"
    return name[:80]
