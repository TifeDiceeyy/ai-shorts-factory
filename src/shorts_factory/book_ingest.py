"""Local, private book ingestion for research and inspiration.

The index contains bounded chunks for internal retrieval. It is never copied
into scripts or captions. Public claims still have to be independently
verified by web sources before brief_builder can select them.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


SUPPORTED_TEXT_SUFFIXES = {".txt", ".md"}


def read_book(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"BOOK_FILE does not exist: {path}")
    if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF ingestion requires pypdf; install requirements.txt") from exc
        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    raise ValueError("BOOK_FILE must be .pdf, .txt, or .md")


def chunk_private_text(text: str, max_chars: int = 1200) -> list[str]:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if buffer and len(buffer) + len(paragraph) + 2 > max_chars:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
    if buffer:
        chunks.append(buffer)
    return chunks


def build_private_index(book_path: Path, output_path: Path) -> dict:
    raw = book_path.read_bytes()
    chunks = chunk_private_text(read_book(book_path))
    payload = {
        "schema_version": 1,
        "source": {
            "filename": book_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "private": True,
            "copyright_policy": "research-only; never emit chunks to scripts or public UI",
        },
        "chunks": [
            {"id": f"book-{index:05d}", "text": chunk, "internal_only": True}
            for index, chunk in enumerate(chunks, start=1)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def retrieve_private_chunks(index: dict, topic: str, limit: int = 8) -> list[dict]:
    words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]+", topic) if len(w) > 2}
    scored = []
    for chunk in index.get("chunks", []):
        haystack = chunk.get("text", "").lower()
        score = sum(haystack.count(word) for word in words)
        if score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].get("id", "")))
    return [chunk for _, chunk in scored[:limit]]


def research_terms(chunks: list[dict], topic: str, limit: int = 5) -> list[str]:
    """Extract keywords, not passages, for source-discovery queries."""
    topic_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]+", topic)}
    stop = topic_words | {
        "that", "this", "with", "from", "were", "have", "into", "their", "which",
        "when", "then", "than", "also", "been", "being", "would", "could", "about",
        "there", "they", "them", "these", "those", "what", "where", "while", "because",
    }
    counts = Counter(
        word.lower()
        for chunk in chunks
        for word in re.findall(r"[A-Za-z][A-Za-z'-]+", chunk.get("text", ""))
        if len(word) >= 5 and word.lower() not in stop
    )
    return [word for word, _ in counts.most_common(limit)]
