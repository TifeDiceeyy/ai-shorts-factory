"""Idea generator: turns topics + templates into Shorts video ideas.

Every idea is grounded in the brain: it carries a knowledge_score computed
from how much book material the retriever can find for it. That score is the
first (pre-publish) signal for the learning loop.
"""
from __future__ import annotations

import re

from .retrieval import Retriever
from .topics import SEED_TOPICS, topics_for_idea

TEMPLATES = [
    ("what_if_disappears", "What happens if {topic} disappears tomorrow?"),
    ("rebuild_from_scratch", "How to rebuild {topic} from scratch"),
    ("first_24_hours", "The first 24 hours without {topic}"),
    ("what_dies_first", "Which technologies die first when {topic} fails?"),
    ("survival_plan", "You wake up in a world without {topic} — here's your survival plan"),
    ("before_modern", "What did people use before {topic}?"),
    ("invention_saved", "{topic}: the forgotten invention that built civilization"),
    ("apocalypse_guide", "The apocalypse guide to {topic}"),
]


def _clean_topic(name: str) -> str:
    return re.sub(r"\s*&\s*", " and ", name.strip())


class IdeaGenerator:
    def __init__(self, retriever: Retriever, topics: list[dict] | None = None):
        self.retriever = retriever
        self.topics = topics or [dict(t) for t in SEED_TOPICS]

    def generate(self, seed: str | None = None, n: int = 10, top_k: int = 5) -> list[dict]:
        """Generate ``n`` video ideas. If ``seed`` is given, it is expanded;
        otherwise ideas are drawn across the topic inventory."""
        ideas: list[dict] = []
        if seed:
            for tpl_id, tpl in TEMPLATES:
                title = tpl.format(topic=_clean_topic(seed))
                ideas.append(self._make_idea(title, seed, tpl_id, top_k))
        else:
            # Rotate through topics and templates, taking the most
            # knowledge-dense combinations first.
            for topic in self.topics:
                for tpl_id, tpl in TEMPLATES[:4]:  # strongest hooks first
                    title = tpl.format(topic=_clean_topic(topic["name"]))
                    ideas.append(self._make_idea(title, topic["name"], tpl_id, top_k))
        ideas.sort(key=lambda i: i["knowledge_score"], reverse=True)
        return ideas[:n]

    # ------------------------------------------------------------------
    def _make_idea(self, title: str, topic: str, template_id: str, top_k: int) -> dict:
        results = self.retriever.search(topic, top_k=top_k)
        score = sum(r.score for r in results)
        matched = sorted({t for r in results for t in r.matched_terms})
        # A rough "how much unique book material backs this idea" score.
        coverage = len({r.chunk.id for r in results})
        knowledge_score = round(score * (1 + 0.2 * coverage), 3)
        return {
            "title": title,
            "topic": topic,
            "angle": template_id,
            "knowledge_score": knowledge_score,
            "source_hits": coverage,
            "matched_terms": matched[:10],
        }

    def best(self, seed: str | None = None, n: int = 10) -> list[dict]:
        return self.generate(seed=seed, n=n)
