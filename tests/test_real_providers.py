import json

import pytest

from shorts_factory.cost_tracker import BudgetExceeded, CostTracker
from shorts_factory.providers.fal import FalGateway
from shorts_factory.providers.image import FalImageProvider
from shorts_factory.providers.llm import FalLLMProvider


def sample_script():
    return {
        "topic": "soap", "language": "English", "visual_style": "illustrated",
        "scenes": [{
            "narration": "A factual line.", "caption": "A factual line", "duration": 8,
            "visual_prompt": "illustration", "source_claim_id": "claim-01",
            "camera": "static", "sfx": None,
        }],
    }


class FakeFalClient:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def subscribe(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.result


def gateway(result):
    return FalGateway("", client=FakeFalClient(result))


def test_fal_llm_parses_json_and_records_real_response_cost():
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    tracker = CostTracker(1)
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    assert provider.generate_script({"topic": "soap", "claims": []}, "English", "style", tracker) == script
    assert tracker.total_spent_usd == 0.017
    assert fal.client.calls[0][0] == "openrouter/router"


def test_fal_llm_budget_refuses_before_gateway_call():
    fal = gateway({"output": "{}"})
    provider = FalLLMProvider(fal, "model", 0.25)
    with pytest.raises(BudgetExceeded):
        provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(0.1))
    assert fal.client.calls == []


def test_fal_image_budget_refuses_before_gateway_call(tmp_path):
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    provider = FalImageProvider(fal, "fal-ai/model", 0.2)
    with pytest.raises(BudgetExceeded):
        provider.generate_scene_image({"visual_prompt": "safe scene"}, 0, tmp_path / "x.png", CostTracker(0.1))
    assert fal.client.calls == []
