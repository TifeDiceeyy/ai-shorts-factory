"""Regression test for the duration-drift bug found in review: assembly must
track ACTUAL audio duration, not the script's nominal `duration` field. The
stub TTS provider hides this because it generates audio at exactly the
requested length by construction, so this test uses a fake provider that
deliberately produces audio LONGER than scripted, simulating what a real
voice will do, and proves the final render follows the real audio."""
from shorts_factory.assembly import (
    NARRATION_SPEED_FACTOR,
    assemble,
    probe_duration,
    solid_color_frame,
    synthesize_scenes,
)
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.tts import TTSProvider

FIXTURE_SCENES = [
    {
        "narration": "First scene narration, scripted short.",
        "caption": "First scene caption.",
        "duration": 2.0,  # nominal/estimated
        "visual_prompt": "a plain workshop table",
        "source_claim_id": "claim-01",
    },
    {
        "narration": "Second scene narration, scripted short too.",
        "caption": "Second scene caption.",
        "duration": 2.0,  # nominal/estimated
        "visual_prompt": "a plain workshop table, different angle",
        "source_claim_id": "claim-02",
    },
]

# Deliberately different from the scripted 2.0s each, simulating a real voice
# that ran long — this is exactly the drift a real TTS provider can produce.
ACTUAL_SECONDS = {0: 3.5, 1: 2.7}


class FakeVariableTTSProvider(TTSProvider):
    """Produces real (sine-tone) audio at a length that deliberately does NOT
    match scene['duration'], to prove the pipeline reconciles against actual
    audio rather than trusting the script's estimate."""

    name = "fake-variable"

    def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
        import subprocess
        cost_tracker.check_budget(f"fake.synthesize[{scene_index}]", 0.0)
        actual = ACTUAL_SECONDS[scene_index]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={actual}:sample_rate=48000",
                "-ac", "1", str(out_path),
            ],
            check=True,
        )
        cost_tracker.record("fake-variable", f"fake.synthesize[{scene_index}]", 0.0, 0.0, is_stub=False)
        return out_path


def test_video_duration_follows_actual_audio_not_scripted_estimate(tmp_path):
    tracker = CostTracker(budget_cap_usd=2.00)
    provider = FakeVariableTTSProvider()

    scene_audio = synthesize_scenes(provider, FIXTURE_SCENES, tmp_path / "audio", tracker)

    # The measured durations must reflect the FAKE provider's real output
    # (also passed through the pipeline's own narration speed-up, same as
    # any real TTS clip — see assembly.NARRATION_SPEED_FACTOR), not the
    # script's nominal 2.0s guess.
    assert scene_audio[0].duration != 2.0
    assert abs(scene_audio[0].duration - ACTUAL_SECONDS[0] / NARRATION_SPEED_FACTOR) < 0.05
    assert abs(scene_audio[1].duration - ACTUAL_SECONDS[1] / NARRATION_SPEED_FACTOR) < 0.05
    assert scene_audio[0].scripted_duration == 2.0

    out_mp4 = tmp_path / "out.mp4"
    assemble(
        scenes=FIXTURE_SCENES,
        frame_source=lambda i, scene: solid_color_frame(i),
        audio=scene_audio,
        workdir=tmp_path / "work",
        out_mp4=out_mp4,
    )

    expected_total = (ACTUAL_SECONDS[0] + ACTUAL_SECONDS[1]) / NARRATION_SPEED_FACTOR
    scripted_total = sum(s["duration"] for s in FIXTURE_SCENES)
    actual_rendered = probe_duration(out_mp4)

    # The critical assertion: the render matches the REAL audio total, not
    # the script's nominal total (which would be 4.0s, badly wrong here).
    assert abs(actual_rendered - expected_total) < 0.15, (
        f"rendered duration {actual_rendered:.2f}s should track actual audio "
        f"total {expected_total:.2f}s, not the scripted nominal total {scripted_total:.2f}s"
    )
    assert abs(actual_rendered - scripted_total) > 0.5, (
        "fixture is only meaningful if scripted and actual totals clearly differ"
    )


def test_word_budget_accounts_for_per_scene_audio_overhead():
    """The word budget must subtract fixed per-scene overhead before
    converting a scene's duration into words.

    Measured by least squares over 11 real generated videos (2026-09-01):
    total audio = 0.3219s/word + 0.426s/scene. That second term is the
    head/tail silence each separately-synthesized scene carries, and it
    scales with scene COUNT, not word count. Budgeting words as
    duration * rate with no overhead term put the first 15-scene video at
    50.6s against a 50s ceiling — 15 x 0.43s = 6.4s unbudgeted, essentially
    the whole overshoot. At 6 scenes the same omission was only 2.6s, which
    is why it survived until the scene count trebled.
    """
    from shorts_factory.providers.llm import (
        NARRATION_WORDS_PER_SECOND,
        SCENE_AUDIO_OVERHEAD_SECONDS,
        TARGET_TOTAL_SECONDS,
    )

    for n_scenes in (4, 6, 10, 15):
        avg_duration = TARGET_TOTAL_SECONDS / n_scenes
        speaking = max(0.6, avg_duration - SCENE_AUDIO_OVERHEAD_SECONDS)
        words = max(5, round(speaking * NARRATION_WORDS_PER_SECOND))
        predicted = n_scenes * words / NARRATION_WORDS_PER_SECOND + n_scenes * SCENE_AUDIO_OVERHEAD_SECONDS
        assert 40.0 <= predicted <= 50.0, (
            f"{n_scenes} scenes x {words} words predicts {predicted:.1f}s, outside the 40-50s window"
        )
