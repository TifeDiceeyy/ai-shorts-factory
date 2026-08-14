import json

import pytest

from shorts_factory.cost_tracker import BudgetExceeded, CostTracker
from shorts_factory.providers.llm import AnthropicLLMProvider
from shorts_factory.providers.image import FalImageProvider


class Response:
    def __init__(self, payload=None, content=b"image"):
        self._payload = payload or {}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def sample_script():
    return {
        "topic": "soap", "language": "English", "visual_style": "illustrated",
        "scenes": [{
            "narration": "A factual line.", "caption": "A factual line", "duration": 8,
            "visual_prompt": "illustration", "source_claim_id": "claim-01",
            "camera": "static", "sfx": None,
        }],
    }


def test_anthropic_provider_parses_json_and_records_cost(monkeypatch):
    script = sample_script()
    monkeypatch.setattr("shorts_factory.providers.llm.requests.post", lambda *a, **k: Response({
        "content": [{"type": "text", "text": json.dumps(script)}]
    }))
    tracker = CostTracker(1)
    provider = AnthropicLLMProvider("key", "model", 0.25)
    assert provider.generate_script({"topic": "soap", "claims": []}, "English", "style", tracker) == script
    assert tracker.total_spent_usd == 0.25


def test_anthropic_budget_refuses_before_http(monkeypatch):
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
    monkeypatch.setattr("shorts_factory.providers.llm.requests.post", forbidden)
    provider = AnthropicLLMProvider("key", "model", 0.25)
    with pytest.raises(BudgetExceeded):
        provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(0.1))
    assert called is False


def test_fal_budget_refuses_before_http(monkeypatch, tmp_path):
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
    monkeypatch.setattr("shorts_factory.providers.image.requests.post", forbidden)
    provider = FalImageProvider("key", "fal-ai/model", 0.2)
    with pytest.raises(BudgetExceeded):
        provider.generate_scene_image({"visual_prompt": "safe scene"}, 0, tmp_path / "x.png", CostTracker(0.1))
    assert called is False
