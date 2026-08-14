"""Assembly must be byte-stable across runs given identical local inputs
(CLAUDE.md skeleton acceptance criterion). Uses a small 2-scene fixture
rather than the full soap script, so the test stays fast and self-contained."""
import hashlib
from pathlib import Path

from shorts_factory.assembly import assemble, solid_color_frame, synthesize_scenes
from shorts_factory.cost_tracker import CostTracker
from shorts_factory.providers.tts import StubTTSProvider

FIXTURE_SCENES = [
    {
        "narration": "First scene narration.",
        "caption": "First scene caption.",
        "duration": 2.0,
        "visual_prompt": "a plain workshop table",
        "source_claim_id": "claim-01",
    },
    {
        "narration": "Second scene narration.",
        "caption": "Second scene caption.",
        "duration": 2.0,
        "visual_prompt": "a plain workshop table, different angle",
        "source_claim_id": "claim-02",
    },
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_once(tmp_path: Path, name: str) -> Path:
    workdir = tmp_path / name
    out_mp4 = tmp_path / f"{name}.mp4"
    tracker = CostTracker(budget_cap_usd=2.00)
    tts = StubTTSProvider()
    scene_audio = synthesize_scenes(tts, FIXTURE_SCENES, workdir / "audio", tracker)
    assemble(
        scenes=FIXTURE_SCENES,
        frame_source=lambda i, scene: solid_color_frame(i),
        audio=scene_audio,
        workdir=workdir,
        out_mp4=out_mp4,
    )
    return out_mp4


def test_assembly_is_byte_stable_across_runs(tmp_path):
    out1 = _run_once(tmp_path, "run1")
    out2 = _run_once(tmp_path, "run2")

    hash1 = _sha256(out1)
    hash2 = _sha256(out2)

    assert hash1 == hash2, (
        "Assembly produced different output bytes from identical local inputs. "
        "If this starts failing, the nondeterministic boundary (e.g. encoder "
        "threading, embedded timestamps) must be found and either fixed or "
        "documented here — not silently ignored."
    )
