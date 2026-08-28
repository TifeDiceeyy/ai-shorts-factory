"""Curated topic inventory used to expand retrieval queries.

Only the seed list + lookup survive here — the outline/heading-mining
machinery that used to merge in PDF-derived topics only ever ran from
Brain.build() (removed 2026-08-28, dead: nothing in the real pipeline
calls it, and it depended on the also-removed extract.py)."""
from __future__ import annotations

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
