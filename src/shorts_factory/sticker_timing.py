"""Narration-synced sticker appearance timing — word-level sync."""
from __future__ import annotations

import re
from typing import Any

from .assembly import narration_word_timings

_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "for", "with", "is", "are", "was",
    "flat", "cartoon", "sticker", "isolated", "pure", "solid", "white", "background", "outline",
    "black", "no", "text", "floor", "shadow", "single", "object", "only", "character", "expressive",
    "face", "work", "apron", "mascot", "detail", "process", "effect", "ambient", "tool", "workshop",
})

_MIN_KEYWORD_LEN = 3
_MIN_STAGGER_S = 0.10


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _subject_from_visual_prompt(prompt: str) -> str:
    match = re.search(r"sticker of (.+?) on pure", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return prompt


def _sticker_keywords(sticker: dict[str, Any], props: list[str], prop_index: int | None) -> list[str]:
    keywords: list[str] = []

    for field in ("trigger_words", "label"):
        value = sticker.get(field)
        if isinstance(value, list):
            keywords.extend(str(v) for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            keywords.extend(re.split(r"[\s,/]+", value))

    if prop_index is not None and prop_index < len(props):
        keywords.extend(re.split(r"[\s,/]+", props[prop_index]))

    subject = _subject_from_visual_prompt(sticker.get("visual_prompt", ""))
    keywords.extend(re.split(r"[\s,/]+", subject))

    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in keywords:
        token = _normalize_token(raw)
        if len(token) < _MIN_KEYWORD_LEN or token in _STOP_WORDS:
            continue
        if token not in seen:
            seen.add(token)
            cleaned.append(token)
    return cleaned


def _phrase_start(words: list[str], timings: list[tuple[str, float, float]], phrase_words: list[str]) -> float | None:
    if not phrase_words:
        return None
    norm_phrase = [_normalize_token(w) for w in phrase_words]
    norm_words = [_normalize_token(w) for w in words]
    for index in range(len(words) - len(phrase_words) + 1):
        if norm_words[index : index + len(phrase_words)] == norm_phrase:
            return timings[index][1]
    return None


def _keyword_matches(spoken_norm: str, keyword: str) -> bool:
    if spoken_norm == keyword:
        return True
    if len(keyword) >= 4 and keyword in spoken_norm:
        return True
    if len(spoken_norm) >= 4 and spoken_norm in keyword:
        return True
    return False


def _word_start_for_keywords(narration: str, duration: float, keywords: list[str]) -> float | None:
    """Return the timestamp when any keyword is first spoken in the narration."""
    if not keywords:
        return None

    words = narration.split()
    timings = narration_word_timings(narration, duration)
    if not words or not timings:
        return None

    ordered = sorted(set(keywords), key=len, reverse=True)
    for keyword in ordered:
        if " " in keyword:
            phrase_words = keyword.split()
            found = _phrase_start(words, timings, phrase_words)
            if found is not None:
                return found

        for word, start, _end in timings:
            spoken = _normalize_token(word)
            if _keyword_matches(spoken, keyword):
                return start

    return None


def _stagger_times(count: int, duration: float, *, start: float = 0.35, end_pad: float = 0.85) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [0.0]
    span = max(0.5, duration * end_pad - start)
    step = span / (count - 1)
    return [round(start + step * i, 2) for i in range(count)]


def _clamp_appear(t: float, duration: float) -> float:
    return min(max(0.0, t), max(0.0, duration - 0.05))


def _enforce_monotonic_appears(stickers: list[dict[str, Any]], duration: float) -> None:
    """Keep progressive build order — later stickers never pop before earlier ones."""
    movable = [s for s in stickers if not s.get("uses_hero")]
    movable.sort(key=lambda s: float(s.get("appear_at", 0.0)))
    last = -_MIN_STAGGER_S
    for sticker in movable:
        t = float(sticker.get("appear_at", 0.0))
        if t <= last:
            t = last + _MIN_STAGGER_S
        sticker["appear_at"] = _clamp_appear(t, duration)
        last = float(sticker["appear_at"])


def sync_sticker_appear_times(scenes: list[dict[str, Any]], durations: list[float]) -> None:
    """Rewrite sticker appear_at from measured TTS duration and spoken-word timing.

    Each sticker enters when the narration first speaks its subject noun (e.g.
    "stone" / "limestone" / "volcanic ash") — not on a fixed delay afterward.
    """
    for scene, duration in zip(scenes, durations):
        stickers = scene.get("stickers") or []
        if not stickers:
            continue

        scene_type = scene.get("scene_type", "mascot_reaction")
        props = [p.strip() for p in (scene.get("props") or "").split(",") if p.strip()]
        narration = scene["narration"]

        image_stickers = [s for s in stickers if not s.get("is_label")]
        label_stickers = [s for s in stickers if s.get("is_label")]

        prop_cursor = 0
        for sticker in image_stickers:
            if sticker.get("uses_hero"):
                sticker["appear_at"] = 0.0
                continue

            prop_index = prop_cursor if scene_type == "ingredient_grid" else None
            if scene_type == "ingredient_grid":
                prop_cursor += 1

            keywords = _sticker_keywords(sticker, props, prop_index)
            matched = _word_start_for_keywords(narration, duration, keywords)
            if matched is None and scene_type != "ingredient_grid":
                # Also try matching words from scene props/fx/action against narration.
                for blob in (scene.get("fx"), scene.get("action")):
                    if blob:
                        matched = _word_start_for_keywords(narration, duration, re.split(r"[\s,/]+", str(blob)))
                        if matched is not None:
                            break

            if matched is not None:
                sticker["appear_at"] = _clamp_appear(matched, duration)
            else:
                fallback = _stagger_times(len(image_stickers), duration)
                idx = image_stickers.index(sticker)
                sticker["appear_at"] = _clamp_appear(fallback[min(idx, len(fallback) - 1)], duration)

        _enforce_monotonic_appears(image_stickers, duration)

        label_parent: dict[str, float] = {}
        for sticker in image_stickers:
            if sticker.get("label"):
                label_parent[sticker["id"]] = float(sticker["appear_at"])

        for sticker in label_stickers:
            parent_id = sticker.get("parent_id")
            if parent_id and parent_id in label_parent:
                sticker["appear_at"] = label_parent[parent_id]
            elif image_stickers:
                sticker["appear_at"] = float(image_stickers[-1]["appear_at"])
