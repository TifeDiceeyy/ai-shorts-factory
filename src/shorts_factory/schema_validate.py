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
    if not (SCRIPT_MIN_TOTAL_SECONDS <= total <= SCRIPT_MAX_TOTAL_SECONDS):
        raise ValidationError(
            f"total scripted duration {total:.2f}s is outside the "
            f"{SCRIPT_MIN_TOTAL_SECONDS}-{SCRIPT_MAX_TOTAL_SECONDS}s window"
        )
