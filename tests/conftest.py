import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Real-provider env vars that must never leak into a test run. config.py's
# load_settings() reads these straight from os.environ every call — once the
# project has a real .env (real fal.ai / Tavily credentials), an unguarded
# test that reaches run_pipeline()/retrieval.main()/generate_ideas() would
# make real, paid network calls just because that .env exists on this
# machine. Real-provider code paths are exercised via injected fakes
# (see test_real_providers.py's FakeFalClient), never via the ambient
# environment — this fixture is what makes that guarantee hold for every
# test, not just the ones that remember to monkeypatch it themselves.
_REAL_PROVIDER_ENV_VARS = [
    "LLM_PROVIDER", "TTS_PROVIDER", "IMAGE_PROVIDER", "VIDEO_PROVIDER", "SEARCH_PROVIDER",
    "FAL_KEY", "SEARCH_API_KEY", "TAVILY_API_KEY",
]


@pytest.fixture(autouse=True)
def _no_real_providers_by_default(monkeypatch):
    for name in _REAL_PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
