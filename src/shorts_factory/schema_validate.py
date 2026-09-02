"""JSON Schema validation plus the cross-document checks a formal schema
can't express on its own: every scene's source_claim_id must resolve to a
real claim in the brief, and total scripted duration must land in the
40-50s window CLAUDE.md specifies for a Short.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

SCRIPT_MIN_TOTAL_SECONDS = 40.0
SCRIPT_MAX_TOTAL_SECONDS = 50.0
# The accepted window is the requested length +-10%. The two constants above
# remain the DEFAULT window (i.e. a 45s target), so a brief that asks for no
# particular length validates exactly as it always has.
SCRIPT_DURATION_TOLERANCE = 0.10


def script_duration_window(brief: dict[str, Any] | None = None) -> tuple[float, float]:
    """The (min, max) total-duration window a script must land in.

    Derived from the brief's own target_seconds when it has one. Without
    this, asking for a 30s or 60s video produced a script that the validator
    then rejected against the fixed 45s window — and the caller silently
    fell back to the deterministic stub script, so the requested length
    would have appeared to do nothing at all.
    """
    target = (brief or {}).get("target_seconds")
    if not isinstance(target, (int, float)) or target <= 0:
        return SCRIPT_MIN_TOTAL_SECONDS, SCRIPT_MAX_TOTAL_SECONDS
    target = float(min(180.0, max(15.0, target)))
    return target * (1 - SCRIPT_DURATION_TOLERANCE), target * (1 + SCRIPT_DURATION_TOLERANCE)


class ValidationError(Exception):
    pass


def _load_schema(name: str) -> dict[str, Any]:
    with open(SCHEMAS_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_brief(brief: dict[str, Any]) -> None:
    schema = _load_schema("brief.schema.json")
    try:
        jsonschema.validate(instance=brief, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"brief failed schema validation: {e.message}") from e

    ids = [c["id"] for c in brief["claims"]]
    if len(ids) != len(set(ids)):
        raise ValidationError("brief has duplicate claim ids")


def validate_script_shape(script: dict[str, Any]) -> None:
    """JSON-Schema-only check — shape, types, per-field constraints."""
    schema = _load_schema("script.schema.json")
    try:
        jsonschema.validate(instance=script, schema=schema)
    except jsonschema.ValidationError as e:
        raise ValidationError(f"script failed schema validation: {e.message}") from e


def validate_script_against_brief(script: dict[str, Any], brief: dict[str, Any]) -> None:
    """Full validation: shape + citation integrity + duration window.
    This is the function the pipeline actually calls."""
    validate_script_shape(script)

    known_claim_ids = {c["id"] for c in brief["claims"]}
    for i, scene in enumerate(script["scenes"]):
        cid = scene["source_claim_id"]
        if cid not in known_claim_ids:
            raise ValidationError(
                f"scene {i} cites source_claim_id={cid!r}, which does not exist "
                f"in the brief (known ids: {sorted(known_claim_ids)}) — missing/invalid citation"
            )

    total = sum(s["duration"] for s in script["scenes"])
    min_total, max_total = script_duration_window(brief)
    if not (min_total <= total <= max_total):
        raise ValidationError(
            f"total scripted duration {total:.2f}s is outside the "
            f"{min_total:.1f}-{max_total:.1f}s window"
        )
