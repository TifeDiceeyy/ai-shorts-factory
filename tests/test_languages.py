"""Output language, across narration AND burned-in captions.

Two things were wrong before this existed (found 2026-09-02):

1. `language` was accepted by generate_script and then never used — it was
   stamped onto the output JSON as metadata while the prompt said nothing,
   so asking for Russian produced English narration LABELLED Russian.
2. Nothing checked whether the caption font could draw the requested
   script. Captions are burned into the frame, so an unrenderable language
   yields a finished, paid-for video whose text is empty boxes.
"""
import pytest
from PIL import ImageFont

from shorts_factory.captions import get_font
from shorts_factory.languages import (
    DEFAULT_LANGUAGE,
    SUPPORTED_LANGUAGES,
    UNSUPPORTED_LANGUAGES,
    by_code,
    is_supported,
    unsupported_reason,
)
from shorts_factory.providers.llm import _script_prompt

BRIEF = {"topic": "roman concrete", "claims": [{"id": "c-01", "claim": "x", "source": "y"}]}

# One representative string per offered language, in its own script.
SAMPLES = {
    "English": "ANCIENT ROMAN CONCRETE", "Spanish": "HORMIGÓN ROMANO",
    "Portuguese": "CONCRETO ROMANO", "French": "BÉTON ROMAIN",
    "German": "RÖMISCHER BETON", "Italian": "CALCESTRUZZO ROMANO",
    "Polish": "STAROŻYTNY BETON", "Turkish": "ANTİK ROMA BETONU",
    "Russian": "ДРЕВНИЙ РИМСКИЙ БЕТОН", "Ukrainian": "ДАВНІЙ РИМСЬКИЙ БЕТОН",
    "Greek": "ΑΡΧΑΙΟ ΡΩΜΑΪΚΟ ΣΚΥΡΟΔΕΜΑ",
}


def _tofu_count(font: ImageFont.FreeTypeFont, text: str) -> int:
    """Characters the font has no glyph for.

    A missing glyph renders as .notdef — the same mask for every absent
    character — so comparing against a character the font certainly lacks
    identifies them.
    """
    notdef = font.getmask("￿").getbbox()
    return sum(1 for ch in text if ch.strip() and font.getmask(ch).getbbox() == notdef)


def test_every_offered_language_actually_renders():
    """The whole basis of the offered list. If this fails, we are about to
    sell someone a video full of empty boxes."""
    font = get_font(64)
    for language in SUPPORTED_LANGUAGES:
        sample = SAMPLES[language.name]
        assert _tofu_count(font, sample) == 0, (
            f"{language.name} is offered but renders tofu: {sample!r}"
        )


def test_the_refused_languages_really_are_unrenderable():
    """Guards against over-refusing: if a font gains coverage, this fails
    and tells us the list can grow."""
    font = get_font(64)
    unrenderable = {
        "Arabic": "الخرسانة", "Hindi": "कंक्रीट", "Japanese": "古代ローマ",
        "Korean": "고대 로마", "Chinese": "古罗马", "Thai": "คอนกรีต",
    }
    for name, sample in unrenderable.items():
        assert name in UNSUPPORTED_LANGUAGES
        assert _tofu_count(font, sample) > 0, (
            f"{name} now renders — it can be moved into SUPPORTED_LANGUAGES"
        )


def test_a_non_english_language_is_an_actual_instruction():
    """Not metadata. The surrounding prompt is entirely in English, so
    without an explicit instruction the model answers in English."""
    russian = _script_prompt(BRIEF, "Russian", "flat")
    assert "CRITICAL LANGUAGE REQUIREMENT" in russian
    assert russian.count("Russian") >= 3, "the requirement must be unambiguous"
    # English needs no such instruction — it is the prompt's own language.
    assert "CRITICAL LANGUAGE REQUIREMENT" not in _script_prompt(BRIEF, "English", "flat")
    for alias in ("english", "en", "EN-US"):
        assert "CRITICAL LANGUAGE REQUIREMENT" not in _script_prompt(BRIEF, alias, "flat")


def test_field_names_stay_english_so_the_schema_still_validates():
    """Translating the JSON keys would fail schema validation and throw away
    a paid script."""
    prompt = _script_prompt(BRIEF, "Greek", "flat")
    assert "JSON field NAMES in English" in prompt


def test_lookup_and_refusal():
    assert by_code("ru").name == "Russian"
    assert by_code("RU").name == "Russian", "codes are case-insensitive"
    assert by_code("jp") is None
    assert is_supported("russian") and is_supported("Russian")
    assert not is_supported("Japanese")
    assert DEFAULT_LANGUAGE == "English" and is_supported(DEFAULT_LANGUAGE)

    reason = unsupported_reason("Japanese")
    assert "Japanese" in reason and "font" in reason.lower()
    assert "Russian" in reason, "the refusal must list what IS available"


def test_language_codes_are_unique():
    codes = [lang.code for lang in SUPPORTED_LANGUAGES]
    assert len(codes) == len(set(codes)), codes
