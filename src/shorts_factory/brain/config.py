"""Central configuration for the faceless-shorts brain.

Everything lives here so the rest of the library has a single source of truth.
No API keys are required: the brain is fully local (pre-built chunks -> search).
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BRAIN_DATA_DIR", PROJECT_DIR / "data"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K_DEFAULT = 8
BM25_K1 = 1.5
BM25_B = 0.75

# Optional embeddings (requires: pip install sentence-transformers)
# When available, scores = EMBED_WEIGHT * vector + (1 - EMBED_WEIGHT) * bm25
EMBED_ENABLED = os.environ.get("BRAIN_EMBED_ENABLED", "auto")  # auto|on|off
EMBED_WEIGHT = 0.35
EMBED_MODEL = os.environ.get("BRAIN_EMBED_MODEL", "all-MiniLM-L6-v2")
