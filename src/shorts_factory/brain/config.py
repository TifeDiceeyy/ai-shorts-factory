"""Central configuration for the faceless-shorts brain.

Everything lives here so the rest of the library has a single source of truth.
No API keys are required: the brain is fully local (PDF -> chunks -> search).
An optional OpenAI-compatible LLM can be attached for polished scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BRAIN_DATA_DIR", PROJECT_DIR / "data"))

# Default source PDFs (used by build_brain.py and Brain.build when no pdfs given)
DEFAULT_PDFS = [
    Path(
        os.environ.get(
            "BRAIN_PDF_KNOWLEDGE",
            r"C:\Users\owner\Downloads\The Knowledge by Lewis Dartnell PDF.pdf",
        )
    ),
    Path(
        os.environ.get(
            "BRAIN_PDF_INVENT",
            r"C:\Users\owner\Downloads\How to Invent Everything_ A Survival Guide for the Stranded Time Traveler - PDF Room.pdf",
        )
    ),
]

# Default OCR'd text sources (scanned PDFs that have no text layer).
# Produce these with:  python ocr_pdf.py "book.pdf" -o sources/book.txt
DEFAULT_TXTS = [
    Path(
        os.environ.get(
            "BRAIN_TXT_BOOK",
            str(PROJECT_DIR / "sources" / "the_book_ultimate_guide.txt"),
        )
    ),
]

# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 1800          # target characters per chunk
CHUNK_OVERLAP = 250        # characters carried over between chunks
MIN_CHUNK_CHARS = 40       # drop only truly empty/blank pages

# Header/footer filtering: a line that appears on more than this fraction of
# pages is treated as boilerplate (running heads, page numbers, watermarks).
BOILERPLATE_LINE_RATIO = 0.25

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

# ---------------------------------------------------------------------------
# Optional LLM (OpenAI-compatible chat completions; Ollama works too)
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ.get("BRAIN_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("BRAIN_LLM_API_KEY", "")
LLM_MODEL = os.environ.get("BRAIN_LLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = int(os.environ.get("BRAIN_LLM_TIMEOUT", "90"))

# ---------------------------------------------------------------------------
# Script engine
# ---------------------------------------------------------------------------

SHORTS_DURATION_SECONDS = 45
WORDS_PER_SECOND = 2.5          # spoken pace for Shorts
MAX_BEATS = 6
