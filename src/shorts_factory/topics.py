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
}

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "soap": ["soap", "saponification", "lye", "tallow", "fat", "alkali", "glycerol"],
    "roman concrete": ["concrete", "roman", "pozzolan", "pozzolana", "lime", "volcanic", "mortar", "cement"],
}
