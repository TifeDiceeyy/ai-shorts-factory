"""Generation-speed fix, 2026-08-29: assembly.synthesize_scenes() used to
render each scene's narration one at a time (a plain for-loop over real
network TTS calls) — real wall time for a multi-scene video was dominated
by this. Now renders concurrently (bounded by SYNTHESIZE_SCENES_MAX_WORKERS),
accepting either a single shared provider instance (used by every existing
caller/test, unaffected) or a zero-arg factory callable that returns a
fresh provider per call — required for a real fal-backed provider, since
fal_client's synchronous client is not thread-safe."""
import time

from shorts_factory.assembly import SceneAudio, synthesize_scenes
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.tts import StubTTSProvider, TTSProvider

FIXTURE_SCENES = [
    {"narration": "Scene zero narration.", "caption": "Scene zero.", "duration": 2.0, "source_claim_id": "c0"},
    {"narration": "Scene one narration.", "caption": "Scene one.", "duration": 2.0, "source_claim_id": "c1"},
    {"narration": "Scene two narration.", "caption": "Scene two.", "duration": 2.0, "source_claim_id": "c2"},
]


class SlowestFirstTTSProvider(TTSProvider):
    """Scene 0 deliberately takes the longest — if results were collected in
    COMPLETION order instead of being placed back at their own scene index,
    this would silently scramble which audio belongs to which scene."""

    name = "slowest-first"
    DELAYS = {0: 0.35, 1: 0.05, 2: 0.15}

    def synthesize_scene(self, scene, scene_index, out_path, cost_tracker):
        import subprocess
        time.sleep(self.DELAYS[scene_index])
        cost_tracker.check_budget(f"slow.synthesize[{scene_index}]", 0.0)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={scene_index + 1}:sample_rate=48000",
                "-ac", "1", str(out_path),
            ],
            check=True,
        )
        cost_tracker.record("slowest-first", f"slow.synthesize[{scene_index}]", 0.0, 0.0, is_stub=False)
        return out_path


def test_results_stay_ordered_by_scene_index_regardless_of_completion_order(tmp_path):
    tracker = CostTracker(budget_cap_usd=2.00)
    provider = SlowestFirstTTSProvider()

    result = synthesize_scenes(provider, FIXTURE_SCENES, tmp_path / "audio", tracker)

    assert [a.path.name for a in result] == ["scene_00.wav", "scene_01.wav", "scene_02.wav"]
    # Each scene's own real (pre-speed-up) duration was scene_index+1 seconds
    # — confirms scene 2's audio didn't end up swapped into scene 0's slot
    # or vice versa despite scene 0 finishing last.
    assert result[0].duration < result[1].duration < result[2].duration


def test_factory_callable_is_invoked_once_per_scene(tmp_path):
    tracker = CostTracker(budget_cap_usd=2.00)
    calls: list[int] = []

    def factory():
        calls.append(1)
        return StubTTSProvider()

    result = synthesize_scenes(factory, FIXTURE_SCENES, tmp_path / "audio", tracker)

    assert len(result) == len(FIXTURE_SCENES)
    assert len(calls) == len(FIXTURE_SCENES)


def test_plain_provider_instance_still_works_unchanged(tmp_path):
    tracker = CostTracker(budget_cap_usd=2.00)
    result = synthesize_scenes(StubTTSProvider(), FIXTURE_SCENES, tmp_path / "audio", tracker)
    assert len(result) == len(FIXTURE_SCENES)
    assert all(isinstance(a, SceneAudio) for a in result)


def test_empty_scene_list_returns_empty_without_erroring(tmp_path):
    tracker = CostTracker(budget_cap_usd=2.00)
    assert synthesize_scenes(StubTTSProvider(), [], tmp_path / "audio", tracker) == []
