import copy
import json
from pathlib import Path

import pytest
from shorts_factory.schema_validate import ValidationError, validate_brief, validate_script_against_brief

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIEF = json.loads((REPO_ROOT / "data" / "soap" / "soap.brief.json").read_text())


def _valid_script():
    # Each scene must be <=15s (per-scene schema cap) while the total lands
    # in the 40-50s window (cross-document check) — four scenes at 11s each.
    return {
        "topic": "soap",
        "language": "English",
        "visual_style": "illustrated realism",
        "scenes": [
            {
                "narration": "Soap forms via saponification.",
                "caption": "Soap forms via saponification.",
                "duration": 11.0,
                "visual_prompt": "a workshop scene",
                "source_claim_id": "claim-01",
            },
            {
                "narration": "Lye came from wood ash.",
                "caption": "Lye came from wood ash.",
                "duration": 11.0,
                "visual_prompt": "wood ash and water",
                "source_claim_id": "claim-02",
            },
            {
                "narration": "Tallow was a common fat.",
                "caption": "Tallow was a common fat.",
                "duration": 11.0,
                "visual_prompt": "rendered fat in a pot",
                "source_claim_id": "claim-03",
            },
            {
                "narration": "Lye is caustic; handle with care.",
                "caption": "Lye is caustic; handle with care.",
                "duration": 12.0,
                "visual_prompt": "protective gloves and goggles",
                "source_claim_id": "claim-06",
            },
        ],
    }


def test_valid_brief_passes():
    validate_brief(BRIEF)  # must not raise


def test_valid_script_passes():
    validate_script_against_brief(_valid_script(), BRIEF)  # must not raise


def test_invalid_script_missing_field_rejected():
    bad = _valid_script()
    del bad["scenes"][0]["visual_prompt"]
    with pytest.raises(ValidationError):
        validate_script_against_brief(bad, BRIEF)


def test_invalid_script_wrong_type_rejected():
    bad = _valid_script()
    bad["scenes"][0]["duration"] = "eight seconds"  # should be a number
    with pytest.raises(ValidationError):
        validate_script_against_brief(bad, BRIEF)


def test_script_caption_over_max_length_rejected():
    bad = _valid_script()
    bad["scenes"][0]["caption"] = "x" * 200
    with pytest.raises(ValidationError):
        validate_script_against_brief(bad, BRIEF)


def test_missing_citation_rejected():
    """A scene citing a claim id that doesn't exist in the brief must be
    rejected — this is the cross-document check a bare JSON Schema can't do."""
    bad = _valid_script()
    bad["scenes"][0]["source_claim_id"] = "claim-99"
    with pytest.raises(ValidationError, match="missing/invalid citation"):
        validate_script_against_brief(bad, BRIEF)


def test_script_duration_outside_window_rejected():
    bad = _valid_script()
    bad["scenes"][0]["duration"] = 1.0
    bad["scenes"][1]["duration"] = 1.0  # total way under 40s
    with pytest.raises(ValidationError, match="outside the"):
        validate_script_against_brief(bad, BRIEF)


def test_script_props_nullable_and_mascot_id_validates():
    script = _valid_script()
    script["mascot_id"] = "mascot_4"
    script["scenes"][0]["props"] = None
    script["scenes"][1]["props"] = "wooden paddle"
    script["scenes"][0]["scene_type"] = "mascot_reaction"
    script["scenes"][0]["layout"] = "centered"
    script["scenes"][0]["fx"] = "sparks"
    validate_script_against_brief(script, BRIEF)  # must not raise

