"""Per-topic search queries and keywords for Phase 1 retrieval.

CLAUDE.md's Phase 1 gate names soap and roman concrete specifically ("Retrieve
cited passages for soap & concrete") — both are configured here so the gate
can be proven for real against both, once a real SEARCH_PROVIDER is connected.
"""

TOPIC_QUERIES: dict[str, list[str]] = {
    "soap": [
        "how is soap made saponification chemistry",
        "history of soap making wood ash lye tallow",
    ],
    "roman concrete": [
        "how was roman concrete made ingredients volcanic ash",
        "why is roman concrete more durable than modern concrete",
    ],
    "apple cider vinegar": [
        "apple cider vinegar fermentation process acetic acid bacteria",
        "history and chemistry of vinegar production reliable source",
    ],
    "charcoal": [
        "charcoal production pyrolysis wood chemistry historical",
        "charcoal carbon monoxide safety authoritative",
    ],
    "pottery": [
        "pottery clay firing process archaeology",
        "ceramic firing clay transformation chemistry",
    ],
    "rope": [
        "traditional rope making plant fibers history",
        "rope construction twist fibers tensile strength",
    ],
    "water filtration": [
        "water filtration removes particles does not disinfect CDC",
        "slow sand filtration history water treatment",
    ],
    "basic compass": [
        "magnetic compass needle earth magnetic field explanation",
        "history of magnetic compass navigation",
    ],
    "food preservation": [
        "food preservation drying salting fermentation science",
        "food preservation safety USDA historical methods",
    ],
    "simple mechanical water pump": [
        "history hand water pump piston check valve mechanics",
        "positive displacement hand pump operation engineering",
    ],
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "soap": ["soap", "saponification", "lye", "tallow", "fat", "alkali", "glycerol"],
    "roman concrete": ["concrete", "roman", "pozzolan", "pozzolana", "lime", "volcanic", "mortar", "cement"],
    "apple cider vinegar": ["vinegar", "apple", "fermentation", "acetic", "bacteria", "ethanol"],
    "charcoal": ["charcoal", "pyrolysis", "wood", "carbon", "oxygen", "kiln"],
    "pottery": ["pottery", "clay", "ceramic", "firing", "kiln", "temper"],
    "rope": ["rope", "fiber", "twist", "cordage", "strand", "tensile"],
    "water filtration": ["filter", "filtration", "water", "sand", "pathogen", "turbidity"],
    "basic compass": ["compass", "magnetic", "needle", "north", "navigation", "field"],
    "food preservation": ["preservation", "food", "drying", "salting", "fermentation", "spoilage"],
    "simple mechanical water pump": ["pump", "water", "piston", "valve", "suction", "cylinder"],
}
