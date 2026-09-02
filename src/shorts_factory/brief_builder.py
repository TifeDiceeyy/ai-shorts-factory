"""Phase 2: build a brief from Phase 1's citation store instead of a
hand-written JSON file. Output is the same shape schema_validate.py already
validates (brief.schema.json) — StubLLMProvider.generate_script() doesn't
need to change at all to consume it; this is the one piece that changes
upstream of it.

Only VERIFIED citations at or above min_confidence are eligible. This is
where Phase 1's verification actually pays off: an unverified or fabricated
claim can't reach a script just because it was extracted — it has to have
cleared the independent-corroboration bar first.
"""
from __future__ import annotations

from typing import Any

DEFAULT_MIN_CONFIDENCE = 0.5
MIN_CLAIMS_FOR_BRIEF = 4
# Raised 6 -> 15 on 2026-09-01 (user: videos should be busy, no slow
# moments). One scene per claim, so this IS the scene count. 15 scenes
# across the 40-50s window is a cut roughly every 3s, matching the
# reference short's pacing. The citation stores comfortably support it:
# roman_concrete has 63 verified claims, soap 66, concrete 50. A topic
# with fewer simply yields fewer scenes — MIN_CLAIMS_FOR_BRIEF still
# guards the floor.
MAX_CLAIMS_FOR_BRIEF = 15


class InsufficientVerifiedClaims(Exception):
    def __init__(self, topic: str, verified_count: int, needed: int):
        self.topic = topic
        self.verified_count = verified_count
        self.needed = needed
        super().__init__(
            f"topic {topic!r} has only {verified_count} verified claim(s) at or above "
            f"the confidence bar — need at least {needed} to build a brief. Run more "
            "retrieval queries or lower the confidence bar deliberately (not silently)."
        )


def build_brief_from_citations(
    topic: str,
    citation_store: dict[str, Any],
    safety_class: str,
    caution: str | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    idea: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """idea, if given, is a {concept, angle, chosen_hook, payoff} dict that
    steers HOW the script frames the (still citation-bound) facts below —
    never a source of facts itself. No caller currently populates it (the
    interactive idea-selection step was removed 2026-08-28); idea=None is
    the fully-supported default and the param is kept for that framing hook."""
    verified = [
        c for c in citation_store.get("citations", [])
        if c.get("verified") and c.get("confidence", 0) >= min_confidence
    ]
    if len(verified) < MIN_CLAIMS_FOR_BRIEF:
        raise InsufficientVerifiedClaims(topic, len(verified), MIN_CLAIMS_FOR_BRIEF)

    verified.sort(key=lambda c: c.get("confidence", 0), reverse=True)
    selected = verified[:MAX_CLAIMS_FOR_BRIEF]

    claims = []
    for i, c in enumerate(selected, start=1):
        source_desc = "; ".join(
            f"{s.get('title') or s.get('domain')} ({s.get('url')})" for s in c.get("sources", [])
        )
        claims.append({
            "id": f"claim-{i:02d}",
            "claim": c["claim_text"],
            "source": source_desc or "source unavailable",
        })

    brief: dict[str, Any] = {"topic": topic, "safety_class": safety_class, "claims": claims}
    if caution:
        brief["caution"] = caution
    if idea:
        if idea.get("concept"):
            brief["concept"] = idea["concept"]
        if idea.get("angle"):
            brief["angle"] = idea["angle"]
        if idea.get("hooks"):
            brief["chosen_hook"] = idea["hooks"][0]["text"]
        if idea.get("payoff"):
            brief["payoff"] = idea["payoff"]
    return brief
