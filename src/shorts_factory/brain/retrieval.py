"""Retrieval: BM25 keyword scoring with optional vector embeddings.

The default path is 100% local: BM25 over the inverted index built by
BrainStore. If sentence-transformers is installed, chunk and query vectors
are cached and blended in for semantic recall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .chunking import Chunk
from .store import BrainStore, tokenize
from .topics import topics_for_idea


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    matched_terms: list[str]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "book": self.chunk.book,
            "pages": self.chunk.pages,
            "text": self.chunk.text,
            "matched_terms": self.matched_terms[:12],
        }


class Retriever:
    """Scores chunks against a query."""

    def __init__(self, store: BrainStore):
        self.store = store
        self._embedder = _load_embedder()

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K_DEFAULT) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []
        expanded = self._expand_query(query)
        tokens = tokenize(expanded)
        if not tokens:
            return []

        bm25 = self._bm25_scores(tokens)
        vector = self._vector_scores(expanded) if self._embedder else {}
        combined: dict[str, float] = {}
        for cid in bm25:
            b = bm25[cid]
            v = vector.get(cid, 0.0)
            combined[cid] = (1 - config.EMBED_WEIGHT) * b + config.EMBED_WEIGHT * v if self._embedder else b

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        results: list[SearchResult] = []
        chunk_by_id = {c.id: c for c in self.store.chunks}
        for cid, score in ranked:
            if score <= 0:
                continue
            chunk = chunk_by_id.get(cid)
            if not chunk:
                continue
            matched = [t for t in tokens if t in chunk.text.lower()]
            results.append(SearchResult(chunk=chunk, score=score, matched_terms=matched))
        return results

    # ------------------------------------------------------------------
    def _expand_query(self, query: str) -> str:
        """Add seed-topic keywords so natural-language queries find the
        right chunks (e.g. 'what stops working when electricity dies' ->
        + power grid generator turbine)."""
        topic = topics_for_idea(query)
        if not topic:
            return query
        ql = query.lower()
        extra = [k for k in topic["keywords"] if k not in ql][:6]
        return query if not extra else f"{query} {' '.join(extra)}"

    def _bm25_scores(self, tokens: list[str]) -> dict[str, float]:
        k1, b = config.BM25_K1, config.BM25_B
        avg = self.store.avg_len or 1.0
        scores: dict[str, float] = {}
        for tok in tokens:
            idf = self.store.idf(tok)
            if idf == 0:
                continue
            for cid, tf in self.store.index.get(tok, {}).items():
                dl = self.store.doc_len.get(cid, 0) or 1
                denom = tf + k1 * (1 - b + b * dl / avg)
                scores[cid] = scores.get(cid, 0.0) + idf * tf * (k1 + 1) / denom
        return scores

    def _vector_scores(self, query: str) -> dict[str, float]:
        if not self._embedder:
            return {}
        try:
            qv = self._embedder.encode([query])[0]
            vecs, cids = self._embed_chunks()
            sims = _cosine(qv, vecs)
            return {cid: float(s) for cid, s in zip(cids, sims)}
        except Exception:
            return {}

    def _embed_chunks(self):
        """Encode all chunks once, cached in memory."""
        if not hasattr(self, "_vec_cache"):
            texts = [c.text for c in self.store.chunks]
            ids = [c.id for c in self.store.chunks]
            vecs = self._embedder.encode(texts)
            self._vec_cache = (vecs, ids)
        return self._vec_cache


# ---------------------------------------------------------------------------
# Optional sentence-transformers support
# ---------------------------------------------------------------------------

def _load_embedder():
    if config.EMBED_ENABLED == "off":
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(config.EMBED_MODEL)
        return model
    except Exception:
        return None


def _cosine(qv, matrix) -> list[float]:
    q_norm = float(math.sqrt(sum(x * x for x in qv))) or 1.0
    out = []
    for row in matrix:
        dot = sum(a * b for a, b in zip(qv, row))
        r_norm = math.sqrt(sum(x * x for x in row)) or 1.0
        out.append(dot / (q_norm * r_norm))
    return out
