"""JSON Schema validation plus the cross-document checks a formal schema
can't express on its own: every scene's source_claim_id must resolve to a
real claim in the brief, and total scripted duration must land in the
40-90s window (configurable via env).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jsonschema

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

SCRIPT_MIN_TOTAL_SECONDS = float(os.environ.get("SCRIPT_MIN_TOTAL_SECONDS", "40"))
SCRIPT_MAX_TOTAL_SECONDS = float(os.environ.get("SCRIPT_MAX_TOTAL_SECONDS", "90"))
DEFAULT_STICKER_TARGET_MIN = int(os.environ.get("STICKER_TARGET_MIN", "12"))
DEFAULT_STICKER_TARGET_MAX = int(os.environ.get("STICKER_TARGET_MAX", "15"))


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


def _image_sticker_count(script: dict[str, Any]) -> int:
    return sum(
        1
        for scene in script.get("scenes", [])
        for sticker in scene.get("stickers") or []
        if not sticker.get("is_label")
    )


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

    sticker_count = _image_sticker_count(script)
    if sticker_count:
        if not (DEFAULT_STICKER_TARGET_MIN <= sticker_count <= DEFAULT_STICKER_TARGET_MAX):
            raise ValidationError(
                f"script declares {sticker_count} image stickers total; expected "
                f"{DEFAULT_STICKER_TARGET_MIN}-{DEFAULT_STICKER_TARGET_MAX}"
            )
        for i, scene in enumerate(script["scenes"]):
            stickers = [s for s in (scene.get("stickers") or []) if not s.get("is_label")]
            duration = scene["duration"]
            for j, sticker in enumerate(stickers):
                appear_at = sticker.get("appear_at")
                if appear_at is None or appear_at < 0 or appear_at >= duration:
                    raise ValidationError(
                        f"scene {i} sticker {j} appear_at={appear_at!r} is outside "
                        f"scene duration {duration:.2f}s"
                    )
