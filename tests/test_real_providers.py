import io
import json
from pathlib import Path

import pytest
from PIL import Image

from shorts_factory.cost_tracker import BudgetExceeded, CostTracker
from shorts_factory.providers.fal import FalGateway
from shorts_factory.providers.image import FalImageProvider
from shorts_factory.providers.llm import FalLLMProvider
from shorts_factory.providers.video import FalVideoProvider


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
        self.uploaded = []

    def subscribe(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.result

    def upload_file(self, path):
        self.uploaded.append(path)
        return f"https://fake.fal.media/uploaded/{Path(path).name}"


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


def test_fal_llm_records_cost_even_if_json_is_malformed():
    fal = gateway({"output": "NOT VALID JSON", "usage": {"cost": 0.017}})
    tracker = CostTracker(1)
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    with pytest.raises(Exception):
        provider.generate_script({"topic": "soap", "claims": []}, "English", "style", tracker)
    # Even though json decoding failed, the money was spent and must be in the ledger!
    assert tracker.total_spent_usd == 0.017


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


def test_fal_image_forces_illustration_style_by_default(tmp_path):
    """Regression test: Recraft-v3's "style" param defaults to
    "realistic_image" when omitted (confirmed live against fal.ai's docs
    2026-08-17) — a real generation produced a photorealistic sports-mascot
    photo instead of our illustrated house style before this was fixed.
    The style param and the visual_style text must both always be present,
    regardless of what the script-generation LLM wrote into visual_prompt."""
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    provider = FalImageProvider(fal, "fal-ai/recraft-v3", 0.04, visual_style="a distinctive hand-drawn mascot style")

    provider.generate_scene_image(
        {"visual_prompt": "Mascot pointing at a beaker"}, 0, tmp_path / "out.png", CostTracker(1)
    )

    args = fal.client.calls[0][1]["arguments"]
    assert args["style"] == "digital_illustration/hand_drawn_outline"
    assert "a distinctive hand-drawn mascot style" in args["prompt"]


def test_fal_image_prompt_is_truncated_under_recrafts_1000_char_limit(tmp_path):
    """Regression test: Recraft-v3 hard-rejects (422) any prompt over 1000
    characters — hit for real when a detailed per-scene character
    description plus the full house style text together exceeded it."""
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    long_style = "x" * 800
    provider = FalImageProvider(fal, "fal-ai/recraft-v3", 0.04, visual_style=long_style)

    provider.generate_scene_image(
        {"visual_prompt": "y" * 400}, 0, tmp_path / "out.png", CostTracker(1)
    )

    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert len(prompt) <= 990


def test_fal_image_imagen3_uses_9_16_aspect_ratio(tmp_path):
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    provider = FalImageProvider(fal, "fal-ai/imagen3", 0.04, visual_style="3D sticker on pure white")

    provider.generate_scene_image(
        {"visual_prompt": "Dwarf mascot pointing up at limestone"}, 0, tmp_path / "out.png", CostTracker(1)
    )

    args = fal.client.calls[0][1]["arguments"]
    assert args["aspect_ratio"] == "9:16"
    assert "style" not in args
    assert "3D sticker on pure white" in args["prompt"]


def test_fal_llm_script_prompt_carries_the_chosen_idea():
    """A human's /plan idea pick must actually steer the real LLM's script,
    not just get logged — the prompt sent to fal.ai must say so."""
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    brief = {
        "topic": "soap", "claims": [],
        "concept": "The lost science behind soap",
        "angle": "a myth-busting historical explainer",
        "chosen_hook": "Everyone assumes soap needs a factory. Here's why that's wrong.",
        "payoff": "Viewer walks away knowing the real steps behind soap.",
    }
    provider.generate_script(brief, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "myth-busting historical explainer" in prompt
    assert "Everyone assumes soap needs a factory" in prompt


def test_fal_llm_script_prompt_tells_llm_the_exact_average_duration_to_hit():
    """Regression test: two real runs in a row undershot the 40-50s window
    (31.5s and 35.5s for 6 scenes) with only a vague '40-50 seconds' /
    'duration (3-9.5 seconds)' instruction — the LLM wasn't doing the
    arithmetic. The prompt must now spell out scene count and average
    per-scene duration explicitly."""
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    brief = {"topic": "soap", "claims": [{"id": f"claim-{i:02d}", "claim": "x", "source": "y"} for i in range(6)]}

    provider.generate_script(brief, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "6 claims" in prompt
    assert "6 scenes" in prompt
    assert "7.5 seconds per scene" in prompt  # 45.0 / 6


def test_fal_llm_script_prompt_has_no_idea_instruction_when_none_given():
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "human already picked" not in prompt


def test_fal_llm_propose_ideas_returns_parsed_ideas_and_records_real_cost():
    payload = {
        "ideas": [
            {
                "concept": "What if soap disappeared tomorrow?",
                "angle": "speculative what-if",
                "hooks": ["Hook one", "Hook two", "Hook three", "Hook four", "Hook five"],
                "payoff": "Viewer understands hygiene collapse risk.",
                "series": "what-if-collapse",
            }
        ]
    }
    fal = gateway({"output": json.dumps(payload), "usage": {"cost": 0.02}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    tracker = CostTracker(1)
    ideas = provider.propose_ideas("soap", 1, tracker)
    assert ideas == payload["ideas"]
    assert tracker.total_spent_usd == 0.02


def test_fal_llm_propose_ideas_rejects_malformed_response():
    fal = gateway({"output": json.dumps({"ideas": [{"concept": "x"}]}), "usage": {"cost": 0.02}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    with pytest.raises(ValueError):
        provider.propose_ideas("soap", 1, CostTracker(1))


def test_fal_video_uploads_hero_image_and_animates_it(tmp_path):
    hero_path = tmp_path / "hero.png"
    hero_path.write_bytes(b"fake png bytes")
    fal = gateway({"video": {"url": "https://example.test/clip.mp4"}})
    fal.download = lambda url: b"fake mp4 bytes"
    provider = FalVideoProvider(fal, "fal-ai/minimax/hailuo-02/standard/image-to-video", 0.045)
    tracker = CostTracker(1)

    out = provider.generate_scene_video(
        {"visual_prompt": "character gestures at a bar of soap"}, hero_path, 0, tmp_path / "clip.mp4", tracker
    )

    assert out.read_bytes() == b"fake mp4 bytes"
    assert fal.client.uploaded == [hero_path]
    args = fal.client.calls[0][1]["arguments"]
    assert args["image_url"] == "https://fake.fal.media/uploaded/hero.png"
    assert args["prompt"] == "character gestures at a bar of soap"
    assert args["duration"] == "6"
    # 6 seconds at $0.045/s, flat — Hailuo has no variable usage.cost field
    assert tracker.total_spent_usd == pytest.approx(0.27)


def test_fal_video_kling_formats_aspect_ratio_and_duration(tmp_path):
    hero_path = tmp_path / "hero.png"
    hero_path.write_bytes(b"fake png bytes")
    fal = gateway({"video": {"url": "https://example.test/kling.mp4"}})
    fal.download = lambda url: b"fake kling mp4 bytes"
    provider = FalVideoProvider(fal, "fal-ai/kling-video/v1.5/pro/image-to-video", 0.05)
    tracker = CostTracker(1)

    out = provider.generate_scene_video(
        {"visual_prompt": "dwarf mascot smiles and points staff up"}, hero_path, 0, tmp_path / "clip.mp4", tracker
    )

    assert out.read_bytes() == b"fake kling mp4 bytes"
    args = fal.client.calls[0][1]["arguments"]
    assert args["aspect_ratio"] == "9:16"
    assert args["duration"] == "5"
    assert args["prompt"] == "dwarf mascot smiles and points staff up"
    # Kling is 5s at $0.05/s = $0.25
    assert provider.cost == pytest.approx(0.25)
    assert tracker.total_spent_usd == pytest.approx(0.25)


def test_fal_video_url_extraction_fallback_and_error(tmp_path):
    hero_path = tmp_path / "hero.png"
    hero_path.write_bytes(b"fake png bytes")
    
    # Matches third path: output.url
    fal_fallback = gateway({"output": {"url": "https://example.test/fallback.mp4"}})
    fal_fallback.download = lambda url: b"fallback bytes"
    provider_fallback = FalVideoProvider(fal_fallback, "fal-ai/kling-video/v1.5/pro/image-to-video", 0.05)
    out = provider_fallback.generate_scene_video({"visual_prompt": "x"}, hero_path, 0, tmp_path / "f.mp4", CostTracker(1))
    assert out.read_bytes() == b"fallback bytes"

    # Matches none -> raises KeyError with attempted paths
    fal_bad = gateway({"unrecognized_key": "some_value"})
    provider_bad = FalVideoProvider(fal_bad, "fal-ai/kling-video/v1.5/pro/image-to-video", 0.05)
    with pytest.raises(KeyError, match="Attempted paths:"):
        provider_bad.generate_scene_video({"visual_prompt": "x"}, hero_path, 0, tmp_path / "b.mp4", CostTracker(1))


def test_fal_video_budget_refuses_before_gateway_call(tmp_path):
    hero_path = tmp_path / "hero.png"
    hero_path.write_bytes(b"fake png bytes")
    fal = gateway({"video": {"url": "https://example.test/clip.mp4"}})
    provider = FalVideoProvider(fal, "fal-ai/minimax/hailuo-02/standard/image-to-video", 10.0)  # way over any cap
    with pytest.raises(BudgetExceeded):
        provider.generate_scene_video({"visual_prompt": "x"}, hero_path, 0, tmp_path / "clip.mp4", CostTracker(0.1))
    assert fal.client.calls == []
    assert fal.client.uploaded == []
