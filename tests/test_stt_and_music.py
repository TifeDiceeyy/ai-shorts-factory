"""Caption alignment from real audio, and the per-story music bed.

Both were identified as genuine gaps against the generation spec
(2026-09-02): captions were timed by weighting each word's LENGTH rather
than by listening, and there was no music generator at all — only mixing
for a track nobody supplied.

Both are optional by construction. A stub provider, an unset model, or a
failed call must leave the render exactly as it was, because a slightly
mistimed caption or a missing bed is worth far more than a lost paid video.
"""
import pytest

from shorts_factory.assembly import narration_caption_cues
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.music import build_mood_prompt, get_music_provider
from shorts_factory.providers.stt import (
    WordTiming,
    get_stt_provider,
    parse_word_timings,
)

NARRATION = "Roman concrete still stands today"


def _timings():
    return [
        WordTiming("Roman", 0.20, 0.50), WordTiming("concrete", 0.55, 1.10),
        WordTiming("still", 1.20, 1.40), WordTiming("stands", 1.50, 1.90),
        WordTiming("today", 2.10, 2.80),
    ]


def test_real_word_timings_drive_the_captions():
    """The whole point: a caption appears when the word is actually said."""
    cues = narration_caption_cues(NARRATION, 3.0, word_timings=_timings())
    assert len(cues) == len(NARRATION.split())
    # Second word starts when the audio says it does, not where a
    # length-weighted guess put it.
    assert cues[1].start == pytest.approx(0.55, abs=0.01)
    assert cues[0].start == 0.0, "first caption holds from the top of the scene"
    assert cues[-1].end >= 3.0, "last caption runs to the end of the scene"


def test_cues_stay_contiguous_across_speech_gaps():
    """STT reports the silence between words as gaps. Honouring them
    literally would blank the caption between every word, which flickers —
    each word is held until the next one starts."""
    cues = narration_caption_cues(NARRATION, 3.0, word_timings=_timings())
    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end == pytest.approx(later.start, abs=1e-6)


def test_a_mismatched_alignment_falls_back_to_the_estimate():
    """A partial alignment is WORSE than the estimate — captions would drift
    against the voice rather than merely being slightly off."""
    estimated = narration_caption_cues(NARRATION, 3.0)
    for broken in (
        _timings()[:3],                                    # too few words
        _timings() + [WordTiming("extra", 2.9, 3.0)],      # too many
        list(reversed(_timings())),                        # out of order
        [],
        None,
    ):
        assert narration_caption_cues(NARRATION, 3.0, word_timings=broken) == estimated


def test_parse_drops_entries_that_would_break_a_cue():
    """Non-word events and zero/negative spans would produce a caption that
    never displays."""
    parsed = parse_word_timings({"words": [
        {"text": "one", "start": 0.0, "end": 0.4, "type": "word"},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing"},
        {"text": "bad", "start": 1.0, "end": 1.0, "type": "word"},
        {"text": "rev", "start": 2.0, "end": 1.5, "type": "word"},
        {"text": "two", "start": 2.0, "end": 2.4, "type": "word"},
    ]})
    assert [w.text for w in parsed] == ["one", "two"]


def test_stub_providers_are_inert():
    """Neither feature may change a render until it is configured."""
    tracker = CostTracker(budget_cap_usd=1.0)
    assert get_stt_provider("stub").align(__import__("pathlib").Path("x.wav"), tracker) == []
    assert get_music_provider("stub").generate_bed("mood", __import__("pathlib").Path("x.wav"), tracker) is None
    assert tracker.total_spent_usd == 0.0


def test_music_prompt_is_instrumental_and_names_the_topic():
    """The bed sits UNDER narration at -26dB: vocals or a strong lead line
    fight the voice instead of supporting it. It also has to suit the story,
    which is why it is generated per topic rather than cached once."""
    prompt = build_mood_prompt("roman concrete")
    assert "roman concrete" in prompt.lower()
    assert "instrumental" in prompt.lower()
    assert "no vocals" in prompt.lower()
    assert build_mood_prompt("soap making") != prompt, "the bed must vary by story"


def test_unknown_providers_are_refused_not_silently_stubbed():
    for factory in (get_stt_provider, get_music_provider):
        with pytest.raises(NotImplementedError):
            factory("madeup")
