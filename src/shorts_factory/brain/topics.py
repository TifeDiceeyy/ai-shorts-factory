"""Topic inventory: the conceptual map the idea generator draws from.

Three sources are merged:
  1. curated seed topics (core rebuilding-civilisation knowledge areas)
  2. PDF bookmarks/outline entries (chapter titles)
  3. heading-like lines detected in the text
"""
from __future__ import annotations

import re
from collections import Counter

from . import config
from .extract import Book, extract_outline

# Curated knowledge areas, with keywords used to boost retrieval and idea
# generation. These are the "hooks" a faceless Shorts channel lives on.
SEED_TOPICS = [
    {"name": "electricity & the power grid", "keywords": ["electricity", "grid", "power", "generator", "turbine", "voltage", "current", "transformer"]},
    {"name": "clean water & sanitation", "keywords": ["water", "drinking", "sanitation", "sewage", "cholera", "filter", "purify", "well"]},
    {"name": "agriculture & food production", "keywords": ["agriculture", "crop", "farm", "soil", "fertiliser", "harvest", "grain", "irrigation", "food"]},
    {"name": "metallurgy & metalworking", "keywords": ["metal", "iron", "steel", "copper", "bronze", "smelt", "forge", "ore", "furnace", "alloy"]},
    {"name": "medicine & public health", "keywords": ["medicine", "disease", "infection", "antibiotic", "vaccine", "surgery", "hygiene", "germ", "health"]},
    {"name": "chemistry & materials", "keywords": ["chemistry", "acid", "alkali", "soap", "glass", "ceramic", "lime", "cement", "plastic", "chemical"]},
    {"name": "energy & fuel", "keywords": ["energy", "fuel", "coal", "oil", "wood", "charcoal", "steam", "battery", "solar", "heat"]},
    {"name": "tools & machines", "keywords": ["tool", "machine", "lathe", "wheel", "lever", "gear", "engine", "pump", "motor"]},
    {"name": "communication & information", "keywords": ["communication", "radio", "signal", "printing", "paper", "writing", "telegraph", "internet", "message"]},
    {"name": "transportation", "keywords": ["transport", "road", "bridge", "ship", "sail", "cart", "railway", "horse", "vehicle"]},
    {"name": "shelter & construction", "keywords": ["shelter", "house", "building", "brick", "wood", "roof", "construction", "concrete"]},
    {"name": "timekeeping & navigation", "keywords": ["clock", "time", "calendar", "navigation", "compass", "latitude", "longitude", "map", "star"]},
    {"name": "refrigeration & food preservation", "keywords": ["refrigerat", "freeze", "preserve", "salt", "smoke", "canning", "cold", "spoilage"]},
    {"name": "scientific method & measurement", "keywords": ["scientific", "experiment", "measure", "unit", "observation", "hypothesis", "test", "knowledge"]},
    {"name": "glass & optics", "keywords": ["glass", "lens", "microscope", "telescope", "optics", "mirror", "light"]},
    {"name": "textiles & clothing", "keywords": ["textile", "cloth", "wool", "cotton", "spinning", "weaving", "clothing", "fiber"]},
]

_HEADING_RE = re.compile(
    r"^(chapter\s+\d+|part\s+\d+|section\s+\d+|[ivxlcdm]+\.\s*\d*|appendix|introduction|conclusion|"
    r"how to|the art of|a (short|brief) history|rebuilding|reinventing)",
    re.IGNORECASE,
)

_NUMBER_PREFIX_RE = re.compile(r"^\d+(\.\d+)*\s*[:\-.]?\s*")
_JUNK_TOPIC_RE = re.compile(
    r"quick brown fox|fc3000|user-serviceable|repair guide|table \d+|figure \d+|"
    r"chapter \d+|section \d+",
    re.IGNORECASE,
)


def _clean_topic_title(raw: str) -> str:
    """Turn a raw outline/heading entry into a usable topic name."""
    t = (raw or "").strip()
    if not t:
        return ""
    if _JUNK_TOPIC_RE.search(t):
        return ""
    t = _NUMBER_PREFIX_RE.sub("", t).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) > 80:
        return ""
    return t


def seed_topics() -> list[dict]:
    return [dict(t) for t in SEED_TOPICS]


def topics_from_outline(pdfs: list) -> list[dict]:
    out = []
    seen = set()
    for path in pdfs:
        for title in extract_outline(path):
            clean = _clean_topic_title(title)
            key = clean.lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({"name": clean, "keywords": [], "source": "outline"})
    return out


def topics_from_headings(books: list[Book]) -> list[dict]:
    """Heading-like lines that occur a handful of times (not boilerplate)."""
    counter = Counter()
    for book in books:
        for page in book.pages:
            for line in page.splitlines():
                line = line.strip()
                if 4 <= len(line) <= 90 and _HEADING_RE.match(line):
                    counter[line] += 1
    out = []
    for line, count in counter.items():
        if 1 <= count <= 40:  # keep real headings, not page furniture
            clean = _clean_topic_title(line)
            if clean:
                out.append({"name": clean, "keywords": [], "source": "heading"})
    return out


def build_topics(pdfs: list, books: list[Book]) -> list[dict]:
    topics = seed_topics()
    topics += topics_from_outline(pdfs)
    topics += topics_from_headings(books)
    # Dedupe by name
    seen: set[str] = set()
    unique = []
    for t in topics:
        key = t["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        t["name"] = t["name"].strip()
        unique.append(t)
    return unique


def topics_for_idea(topic_name: str) -> dict | None:
    """Find the closest seed topic to a free-text idea/topic."""
    t = (topic_name or "").strip().lower()
    if not t:
        return None
    best, best_score = None, 0
    for topic in SEED_TOPICS:
        score = 0
        name = topic["name"].lower()
        if name in t or t in name:
            score += 3
        for kw in topic["keywords"]:
            if kw in t:
                score += 1
        if score > best_score:
            best, best_score = topic, score
    return dict(best) if best else None
