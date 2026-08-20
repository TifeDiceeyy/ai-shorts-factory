"""Runtime-editable topic registry.

Used to hardcode two separate, hand-edited sources of truth: topics.py's
TOPIC_QUERIES/TOPIC_KEYWORDS (Phase 1 retrieval config) and safety.py's
GREEN_TOPICS/YELLOW_TOPICS/YELLOW_CAUTION (safety classification). Both are
now one JSON file so the Telegram bot's new-topic flow can register a topic
at runtime, visible immediately without a restart.

Safety invariant (CLAUDE.md §5, unchanged): this module never decides safety
for a topic — it only stores what a green/yellow classification has already
decided. A topic absent from the registry is not proven safe; safety.py's
classify_topic() still fails closed to RED for anything not found here.
register_topic() refuses to persist a "red" entry for the same reason: red
topics must stay unregistered so they keep failing closed forever, not get
"remembered" as refused (which would itself be a subtle way to encode them).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data" / "topic_registry.json"

# Seeds the registry file on first run with exactly what topics.py +
# safety.py hardcoded before this refactor — migrating existing behavior,
# not changing it. Entries with no retrieval config yet (queries/keywords
# empty) mirror topics that safety.py classified but topics.py never had
# Phase 1 queries for.
_SEED_REGISTRY: dict[str, dict] = {
    "soap": {
        "queries": [
            "how is soap made saponification chemistry",
            "history of soap making wood ash lye tallow",
        ],
        "keywords": ["soap", "saponification", "lye", "tallow", "fat", "alkali", "glycerol"],
        "safety_class": "yellow",
        "caution": (
            "Caution: lye (sodium/potassium hydroxide) is caustic. It burns skin "
            "and eyes on contact and releases irritating fumes when dissolved. "
            "Handle only with eye protection, gloves, and ventilation."
        ),
    },
    "roman concrete": {
        "queries": [
            "how was roman concrete made ingredients volcanic ash",
            "why is roman concrete more durable than modern concrete",
        ],
        "keywords": ["concrete", "roman", "pozzolan", "pozzolana", "lime", "volcanic", "mortar", "cement"],
        "safety_class": "green",
        "caution": None,
    },
    "apple cider vinegar": {
        "queries": [
            "apple cider vinegar fermentation process acetic acid bacteria",
            "history and chemistry of vinegar production reliable source",
        ],
        "keywords": ["vinegar", "apple", "fermentation", "acetic", "bacteria", "ethanol"],
        "safety_class": "yellow",
        "caution": "Caution: fermentation can be contaminated. Follow tested food-safety guidance.",
    },
    "charcoal": {
        "queries": [
            "charcoal production pyrolysis wood chemistry historical",
            "charcoal carbon monoxide safety authoritative",
        ],
        "keywords": ["charcoal", "pyrolysis", "wood", "carbon", "oxygen", "kiln"],
        "safety_class": "yellow",
        "caution": "Caution: charcoal production creates fire and carbon-monoxide hazards. Never attempt it indoors.",
    },
    "pottery": {
        "queries": [
            "pottery clay firing process archaeology",
            "ceramic firing clay transformation chemistry",
        ],
        "keywords": ["pottery", "clay", "ceramic", "firing", "kiln", "temper"],
        "safety_class": "green",
        "caution": None,
    },
    "rope": {
        "queries": [
            "traditional rope making plant fibers history",
            "rope construction twist fibers tensile strength",
        ],
        "keywords": ["rope", "fiber", "twist", "cordage", "strand", "tensile"],
        "safety_class": "green",
        "caution": None,
    },
    "water filtration": {
        "queries": [
            "water filtration removes particles does not disinfect CDC",
            "slow sand filtration history water treatment",
        ],
        "keywords": ["filter", "filtration", "water", "sand", "pathogen", "turbidity"],
        "safety_class": "green",
        "caution": None,
    },
    "basic compass": {
        "queries": [
            "magnetic compass needle earth magnetic field explanation",
            "history of magnetic compass navigation",
        ],
        "keywords": ["compass", "magnetic", "needle", "north", "navigation", "field"],
        "safety_class": "green",
        "caution": None,
    },
    "food preservation": {
        "queries": [
            "food preservation drying salting fermentation science",
            "food preservation safety USDA historical methods",
        ],
        "keywords": ["preservation", "food", "drying", "salting", "fermentation", "spoilage"],
        "safety_class": "yellow",
        "caution": "Caution: unsafe preservation can cause severe foodborne illness. Follow current public-health guidance.",
    },
    "simple mechanical water pump": {
        "queries": [
            "history hand water pump piston check valve mechanics",
            "positive displacement hand pump operation engineering",
        ],
        "keywords": ["pump", "water", "piston", "valve", "suction", "cylinder"],
        "safety_class": "yellow",
        "caution": "Caution: pumped water is not necessarily safe to drink; test and treat it appropriately.",
    },
    # Safety-classified aliases/extras that safety.py recognized but
    # topics.py never had Phase 1 retrieval config for — no queries/keywords,
    # so retrieval still refuses them exactly as before this refactor.
    "concrete": {"queries": [], "keywords": [], "safety_class": "green", "caution": None},
    "compass": {"queries": [], "keywords": [], "safety_class": "green", "caution": None},
    "crop rotation": {"queries": [], "keywords": [], "safety_class": "green", "caution": None},
    "furnaces": {
        "queries": [], "keywords": [], "safety_class": "yellow",
        "caution": "Caution: extreme heat and combustion gases can kill. Use expert-designed equipment and ventilation.",
    },
    "furnace": {
        "queries": [], "keywords": [], "safety_class": "yellow",
        "caution": "Caution: extreme heat and combustion gases can kill. Use expert-designed equipment and ventilation.",
    },
    "electricity": {
        "queries": [], "keywords": [], "safety_class": "yellow",
        "caution": "Caution: electrical work can cause shock, fire, or death. Use qualified guidance and proper protection.",
    },
    "water pump": {
        "queries": [], "keywords": [], "safety_class": "yellow",
        "caution": "Caution: pumped water is not necessarily safe to drink; test and treat it appropriately.",
    },
}


def normalize_topic(topic: str) -> str:
    return " ".join(topic.strip().lower().split())


def load_registry() -> dict[str, dict]:
    """Reads straight from disk every call — deliberately not cached, so a
    topic registered by one process (e.g. the Telegram bot) is visible to
    the next call in the same process, not just after a restart."""
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(_SEED_REGISTRY, indent=2), encoding="utf-8")
        return dict(_SEED_REGISTRY)
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def get_topic(topic: str) -> dict | None:
    return load_registry().get(normalize_topic(topic))


def register_topic(
    topic: str,
    queries: list[str],
    keywords: list[str],
    safety_class: str,
    caution: str | None = None,
) -> None:
    """Persist a new green/yellow topic. Refuses "red" — see module docstring."""
    if safety_class not in ("green", "yellow"):
        raise ValueError(
            f"refusing to register topic with safety_class={safety_class!r}; "
            "only 'green' or 'yellow' may be registered — red topics must stay unregistered"
        )
    name = normalize_topic(topic)
    registry = load_registry()
    registry[name] = {
        "queries": list(queries),
        "keywords": list(keywords),
        "safety_class": safety_class,
        "caution": caution,
    }
    tmp_path = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    tmp_path.replace(REGISTRY_PATH)
