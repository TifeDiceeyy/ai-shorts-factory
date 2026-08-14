"""Regression test for the duration-drift bug found in review: assembly must
track ACTUAL audio duration, not the script's nominal `duration` field. The
stub TTS provider hides this because it generates audio at exactly the
requested length by construction, so this test uses a fake provider that
deliberately produces audio LONGER than scripted, simulating what a real
voice will do, and proves the final render follows the real audio."""
from shorts_factory.assembly import assemble, probe_duration, solid_color_frame, synthesize_scenes
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

    # The measured durations must reflect the FAKE provider's real output,
    # not the script's nominal 2.0s guess.
    assert scene_audio[0].duration != 2.0
    assert abs(scene_audio[0].duration - ACTUAL_SECONDS[0]) < 0.05
    assert abs(scene_audio[1].duration - ACTUAL_SECONDS[1]) < 0.05
    assert scene_audio[0].scripted_duration == 2.0

    out_mp4 = tmp_path / "out.mp4"
    assemble(
        scenes=FIXTURE_SCENES,
        frame_source=lambda i, scene: solid_color_frame(i),
        audio=scene_audio,
        workdir=tmp_path / "work",
        out_mp4=out_mp4,
    )

    expected_total = ACTUAL_SECONDS[0] + ACTUAL_SECONDS[1]
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
