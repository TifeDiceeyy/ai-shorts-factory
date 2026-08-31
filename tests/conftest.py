import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import imageio_ffmpeg
    ffmpeg_dir = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    scripts_dir = str(Path(sys.prefix) / "Scripts")
    if scripts_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

# Real-provider env vars that must never leak into a test run. config.py's
# load_settings() reads these straight from os.environ every call — once the
# project has a real .env (real fal.ai / Tavily credentials), an unguarded
# test that reaches run_pipeline()/retrieval.main() would
# make real, paid network calls just because that .env exists on this
# machine. Real-provider code paths are exercised via injected fakes
# (see test_real_providers.py's FakeFalClient), never via the ambient
# environment — this fixture is what makes that guarantee hold for every
# test, not just the ones that remember to monkeypatch it themselves.
_REAL_PROVIDER_ENV_VARS = [
    "LLM_PROVIDER", "TTS_PROVIDER", "IMAGE_PROVIDER", "VIDEO_PROVIDER", "SEARCH_PROVIDER",
    "FAL_KEY", "SEARCH_API_KEY", "TAVILY_API_KEY",
    # Added 2026-08-30: real gap surfaced the moment YOUTUBE_CLIENT_SECRETS_FILE
    # was actually set for the first time (previously always blank, so this
    # leak had never manifested as a test failure before). Settings.
    # youtube_configured just checks this string's truthiness — an unguarded
    # test asserting YouTubeNotConfigured silently saw the real, now-non-empty
    # value from this machine's own .env and failed downstream instead
    # (test_publish.py::test_publish_refuses_when_youtube_not_configured).
    "YOUTUBE_CLIENT_SECRETS_FILE", "YOUTUBE_TOKEN_FILE",
]


@pytest.fixture(autouse=True)
def _no_real_providers_by_default(monkeypatch):
    for name in _REAL_PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _isolated_custom_mascot_registry(tmp_path, monkeypatch):
    """mascots.select_mascot_for_story()/generate_custom_mascot() read/write
    data/custom_mascots.json — a real, non-tmp_path location — and are
    called from inside run_pipeline() itself whenever a test doesn't pass an
    explicit mascot_id. Without this, any such test would read (and
    potentially write) the real project file, exactly the class of test/
    production collision artifacts_root was added to prevent for
    artifacts/<topic>/ (see pipeline.run_pipeline's own docstring)."""
    from shorts_factory import mascots
    monkeypatch.setattr(mascots, "CUSTOM_MASCOT_REGISTRY_PATH", tmp_path / "custom_mascots.json")
