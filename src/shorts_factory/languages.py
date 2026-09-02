"""Which output languages this pipeline can actually deliver.

A language is only offered when the CAPTION FONT can render it. Measured
2026-09-02 against the real caption font: Latin, Cyrillic and Greek scripts
render correctly, while Arabic, Hindi, Japanese, Korean, Chinese, Thai and
Vietnamese come back as tofu boxes — every glyph a hollow rectangle.

That is not a small cosmetic problem: captions are burned into the frame, so
an unsupported language yields a finished, paid-for video whose text is
unreadable. Refusing up front is the only honest option until the caption
renderer gains a font that covers those scripts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str
    code: str
    label: str


# Ordered by likely usefulness, not alphabetically — the picker shows them
# in this order.
SUPPORTED_LANGUAGES: tuple[Language, ...] = (
    Language("English", "en", "🇬🇧 English"),
    Language("Spanish", "es", "🇪🇸 Español"),
    Language("Portuguese", "pt", "🇧🇷 Português"),
    Language("French", "fr", "🇫🇷 Français"),
    Language("German", "de", "🇩🇪 Deutsch"),
    Language("Italian", "it", "🇮🇹 Italiano"),
    Language("Polish", "pl", "🇵🇱 Polski"),
    Language("Turkish", "tr", "🇹🇷 Türkçe"),
    Language("Russian", "ru", "🇷🇺 Русский"),
    Language("Ukrainian", "uk", "🇺🇦 Українська"),
    Language("Greek", "el", "🇬🇷 Ελληνικά"),
)

# Named explicitly so the refusal can say WHY, and so a future font change
# has an obvious list to revisit.
UNSUPPORTED_LANGUAGES: tuple[str, ...] = (
    "Arabic", "Hindi", "Japanese", "Korean", "Chinese", "Thai", "Vietnamese",
)

DEFAULT_LANGUAGE = "English"


def by_code(code: str) -> Language | None:
    wanted = (code or "").strip().lower()
    return next((lang for lang in SUPPORTED_LANGUAGES if lang.code == wanted), None)


def is_supported(name: str) -> bool:
    wanted = (name or "").strip().lower()
    return any(lang.name.lower() == wanted for lang in SUPPORTED_LANGUAGES)


def unsupported_reason(name: str) -> str:
    """Why a language is refused, in terms the requester can act on."""
    return (
        f"{name} can't be produced yet: the caption font has no glyphs for its "
        f"script, so every burned-in caption would render as empty boxes. "
        f"Supported: {', '.join(lang.name for lang in SUPPORTED_LANGUAGES)}."
    )
