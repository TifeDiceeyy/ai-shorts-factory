"""Phase 7: scheduling / graduated autonomy — deliberately inert right now.

CLAUDE.md §4/§5: autonomy only after ~30-50 human-reviewed videos, and
red/health/safety topics (which in this system's classifier means anything
that isn't GREEN — see safety.py) stay human-gated *permanently*, regardless
of review count.

This module is structurally incapable of running anything right now: the
experiment ledger is empty (zero published videos exist — Phase 5 upload
was never live-run, only built and gated), so human_reviewed_count() is 0
and is_autonomy_eligible() returns False for every topic unconditionally.
That's not a flag someone could accidentally flip — it's a real count read
from real state, and the gate reads the same way it will once videos exist.
"""
from __future__ import annotations

from .experiment_ledger import human_reviewed_count
from .safety import SafetyClass, classify_topic

MIN_HUMAN_REVIEWED_VIDEOS = 30


def is_autonomy_eligible(topic: str) -> tuple[bool, str]:
    reviewed = human_reviewed_count()
    if reviewed < MIN_HUMAN_REVIEWED_VIDEOS:
        return False, f"only {reviewed} human-reviewed video(s) so far, need >= {MIN_HUMAN_REVIEWED_VIDEOS}"

    safety_class = classify_topic(topic)
    if safety_class != SafetyClass.GREEN:
        return False, (
            f"topic {topic!r} is classified {safety_class.value!r}, not green — "
            "yellow/red topics stay human-gated permanently (CLAUDE.md §5), "
            "no review count changes this"
        )

    return True, "eligible"


class AutonomyNotEligible(Exception):
    pass


def attempt_autonomous_run(topic: str):
    """The only entrypoint that would ever trigger an unattended run. Refuses
    before touching the pipeline, ideation, or publish modules at all if not
    eligible — there's nothing downstream of this check to accidentally reach."""
    eligible, reason = is_autonomy_eligible(topic)
    if not eligible:
        raise AutonomyNotEligible(f"refusing autonomous run for {topic!r}: {reason}")

    # Deliberately unimplemented beyond this point. Reaching graduated
    # autonomy is a real product decision for later — when it happens, it
    # plugs in here (ideation.generate_ideas -> pipeline.run_pipeline ->
    # dashboard review still required per CLAUDE.md, since "human-approved
    # publish only" applies to publish, not to render/queue generation).
    raise NotImplementedError(
        "autonomy eligibility check passed, but the autonomous run itself is "
        "not implemented — this is intentionally unreachable until there's a "
        "real product decision about what 'autonomous' means here, made with "
        "30+ reviewed videos of real data to inform it"
    )
