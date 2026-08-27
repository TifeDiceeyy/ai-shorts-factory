"""Persistence layer: chunks + inverted index + topics, all stored as JSON.

Zero external dependencies. The store is a folder:

    data/
      chunks.json       list of chunk dicts
      index.json        token -> {chunk_id: tf}
      topics.json       curated + extracted topic inventory
      stats.json        build metadata
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config
from .chunking import Chunk

STOPWORDS = set(
    """a an and are as at be but by for from had has have he her his i if in is it
    its may not of on or our she so that the their them then there these they this
    to was we were what when where which who will with would you your""".split()
)


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    words = [w.strip("'") for w in text.split()]
    return [w for w in words if len(w) > 2 and w not in STOPWORDS]


@dataclass
class BrainStore:
    data_dir: Path
    chunks: list[Chunk]
    index: dict[str, dict[str, int]]     # token -> {chunk_id: term_freq}
    doc_len: dict[str, int]              # chunk_id -> number of tokens
    avg_len: float
    topics: list[dict]
    stats: dict

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, chunks: list[Chunk], topics: list[dict], stats: dict, data_dir: Path) -> "BrainStore":
        index: dict[str, dict[str, int]] = {}
        doc_len: dict[str, int] = {}
        total = 0
        for c in chunks:
            counts = Counter(tokenize(c.text))
            doc_len[c.id] = sum(counts.values())
            total += doc_len[c.id]
            for tok, tf in counts.items():
                index.setdefault(tok, {})[c.id] = tf
        avg = total / len(chunks) if chunks else 0.0
        return cls(
            data_dir=data_dir,
            chunks=chunks,
            index=index,
            doc_len=doc_len,
            avg_len=avg,
            topics=topics,
            stats=stats,
        )

    @classmethod
    def load(cls, data_dir: Path) -> "BrainStore":
        data_dir = Path(data_dir)
        chunks = [Chunk.from_dict(d) for d in _read_json(data_dir / "chunks.json")]
        index = _read_json(data_dir / "index.json")
        doc_len = _read_json(data_dir / "doc_len.json")
        topics = _read_json(data_dir / "topics.json")
        stats = _read_json(data_dir / "stats.json")
        avg = sum(doc_len.values()) / len(doc_len) if doc_len else 0.0
        return cls(data_dir, chunks, index, doc_len, avg, topics, stats)

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.data_dir / "chunks.json", [c.to_dict() for c in self.chunks])
        _write_json(self.data_dir / "index.json", self.index)
        _write_json(self.data_dir / "doc_len.json", self.doc_len)
        _write_json(self.data_dir / "topics.json", self.topics)
        _write_json(self.data_dir / "stats.json", self.stats)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def search_tokens(self, query: str) -> list[str]:
        return tokenize(query)

    def idf(self, token: str) -> float:
        n_docs = len(self.chunks)
        if n_docs == 0:
            return 0.0
        df = len(self.index.get(token, {}))
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def __len__(self) -> int:
        return len(self.chunks)


def _read_json(path: Path) -> object:
    if not path.exists():
        return [] if path.name != "stats.json" else {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
