import io
import json
from pathlib import Path

import pytest
from PIL import Image

from shorts_factory.cost_tracker import BudgetExceeded, CostTracker
from shorts_factory.providers.fal import FalGateway
from shorts_factory.providers.image import FalImageProvider
from shorts_factory.providers.llm import FalLLMProvider, LLMResponseFormatError
from shorts_factory.providers.video import FalVideoProvider, NONVERBAL_CONTINUOUS_MOTION


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
    assert provider.generate_script(
        {"topic": "soap", "claims": []}, "English", script["visual_style"], tracker
    ) == script
    assert tracker.total_spent_usd == 0.017
    assert fal.client.calls[0][0] == "openrouter/router"


def test_fal_llm_normalizes_harmless_script_schema_drift():
    script = sample_script()
    script["reasoning"] = "This key is not part of the script contract."
    script["topic"] = "a model typo"
    script["language"] = "a model typo"
    script["visual_style"] = "a model typo"
    script["scenes"][0]["notes"] = "also not part of the contract"
    script["scenes"][0]["caption"] = "word " * 30
    script["scenes"][0]["duration"] = "8"
    fal = gateway({"output": json.dumps({"script": script}), "usage": {"cost": 0.017}})
    tracker = CostTracker(1)
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    brief = {"topic": "soap", "claims": []}

    result = provider.generate_script(brief, "English", "house style", tracker)

    assert result["topic"] == "soap"
    assert result["language"] == "English"
    assert result["visual_style"] == "house style"
    assert "reasoning" not in result
    assert "notes" not in result["scenes"][0]
    assert len(result["scenes"][0]["caption"]) <= 90
    assert result["scenes"][0]["duration"] == 8.0


def test_fal_llm_does_not_invent_missing_required_scene_fields():
    script = sample_script()
    del script["scenes"][0]["source_claim_id"]
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)

    result = provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))

    assert "source_claim_id" not in result["scenes"][0]


def test_fal_llm_records_cost_even_if_json_is_malformed():
    fal = gateway({"output": "NOT VALID JSON", "usage": {"cost": 0.017}})
    tracker = CostTracker(1)
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    with pytest.raises(Exception):
        provider.generate_script({"topic": "soap", "claims": []}, "English", "style", tracker)
    # Even though json decoding failed, the money was spent and must be in the ledger!
    assert tracker.total_spent_usd == 0.017


def test_fal_llm_repairs_only_trailing_comma_json_drift():
    script = sample_script()
    malformed = json.dumps(script).replace('"sfx": null}', '"sfx": null,}')
    fal = gateway({"output": malformed, "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)

    result = provider.generate_script(
        {"topic": "soap", "claims": []}, "English", script["visual_style"], CostTracker(1)
    )

    assert result == script


def test_fal_llm_rejects_non_trailing_comma_json_corruption():
    fal = gateway({"output": '{"topic": unquoted}', "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)

    with pytest.raises(LLMResponseFormatError, match="not valid JSON"):
        provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))


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


def test_fal_image_nano_banana_uses_9_16_aspect_ratio(tmp_path):
    """fal-ai/imagen3 is confirmed deprecated (2026-08-20) and fal-ai/imagen4
    doesn't exist under any tested path — nano-banana is the live
    replacement, same aspect_ratio-only schema as the imagen family."""
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    provider = FalImageProvider(fal, "fal-ai/nano-banana", 0.04, visual_style="3D sticker on pure white")

    provider.generate_scene_image(
        {"visual_prompt": "Dwarf mascot pointing up at limestone"}, 0, tmp_path / "out.png", CostTracker(1)
    )

    args = fal.client.calls[0][1]["arguments"]
    assert args["aspect_ratio"] == "9:16"
    assert "style" not in args
    assert "3D sticker on pure white" in args["prompt"]


def test_fal_image_with_reference_dispatches_to_nano_banana_edit_endpoint(tmp_path):
    """Mascot-type scenes must be generated FROM the hero image (image-to-
    image editing), not pure text-to-image, so the character stays
    recognizable while pose/composition can still vary per scene — see
    pipeline._scene_base_image_path. Confirmed live against fal.ai's docs
    (2026-08-21): fal-ai/nano-banana/edit takes required `prompt` +
    `image_urls` (list of reference image URLs)."""
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    provider = FalImageProvider(fal, "fal-ai/nano-banana", 0.04)

    reference_path = tmp_path / "hero_mascot_4.png"
    Image.new("RGB", (4, 4)).save(reference_path)

    provider.generate_scene_image(
        {"visual_prompt": "Mascot pointing at a beaker"}, 0, tmp_path / "out.png", CostTracker(1),
        reference_image_path=reference_path,
    )

    endpoint, kwargs = fal.client.calls[0]
    assert endpoint == "fal-ai/nano-banana/edit"
    assert kwargs["arguments"]["image_urls"] == [f"https://fake.fal.media/uploaded/{reference_path.name}"]
    assert reference_path in fal.client.uploaded


def test_fal_image_without_reference_uses_base_endpoint_not_edit(tmp_path):
    """ingredient_grid/process_action scenes have no character reference —
    must stay on the plain text-to-image endpoint, not silently switch to
    /edit with an empty/missing image_urls."""
    small_png = io.BytesIO()
    Image.new("RGB", (4, 4)).save(small_png, format="PNG")
    fal = gateway({"images": [{"url": "https://example.test/image.png"}]})
    fal.download = lambda url: small_png.getvalue()
    provider = FalImageProvider(fal, "fal-ai/nano-banana", 0.04)

    provider.generate_scene_image(
        {"visual_prompt": "Grid of raw ingredients"}, 0, tmp_path / "out.png", CostTracker(1)
    )

    endpoint, kwargs = fal.client.calls[0]
    assert endpoint == "fal-ai/nano-banana"
    assert "image_urls" not in kwargs["arguments"]
    assert fal.client.uploaded == []


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


def test_fal_llm_script_prompt_says_scene_one_is_mascot_first():
    """User feedback 2026-08-28: scene 1 should center the mascot alone
    (empty space is fine), not open on a crowded ingredient dump — and
    later scenes should introduce props one at a time as they're named,
    not all bundled into the opening frame. The script-writing prompt sent
    to the real LLM must say so explicitly, pipeline-wide (every topic),
    not as a one-off tweak to a single generated script."""
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "scene 1 should center on the mascot alone" in prompt
    assert "introduce props/ingredients one at a time" in prompt
    assert "rather than energetic hopping, bobbing, or repeated bounce" in prompt


def test_fal_llm_script_prompt_says_narrated_objects_must_be_shown():
    """User request 2026-08-29: when a scene's own narration names an
    object, that object should be visible in the scene's image — fixed
    upstream at the script-prompt step (props/action, the same field that
    already drives the real image-generation call) rather than a separate
    late-insertion system. Must be pipeline-wide, every topic."""
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "must also appear in that scene's props (or action) field" in prompt


def test_fal_llm_script_prompt_has_no_idea_instruction_when_none_given():
    script = sample_script()
    fal = gateway({"output": json.dumps(script), "usage": {"cost": 0.017}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    provider.generate_script({"topic": "soap", "claims": []}, "English", "style", CostTracker(1))
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "human already picked" not in prompt


def test_fal_llm_design_mascot_returns_parsed_design_and_records_real_cost():
    design = {
        "name": "Mascot: Deep-Sea Diver",
        "short_desc": "A friendly deep-sea diver in a vintage brass diving helmet.",
        "hero_prompt": "Full-body 3D CGI cartoon diver mascot, fully clothed, white background.",
        "visual_style": "High-end 3D CGI cartoon render, stark white background, no text.",
        "motion_instruction": "Describe the diver's actions and emotions per scene.",
        "scene_role_template": "Hook/Discovery/Process/Challenge/Payoff arc.",
        "keywords": ["diving", "submarine", "ocean", "deep", "sea", "pressure"],
    }
    fal = gateway({"output": json.dumps(design), "usage": {"cost": 0.02}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    tracker = CostTracker(1)
    result = provider.design_mascot("deep sea diving", None, tracker)
    assert result == design
    assert tracker.total_spent_usd == 0.02
    prompt = fal.client.calls[0][1]["arguments"]["prompt"]
    assert "deep sea diving" in prompt
    assert "fully clothed" in prompt.lower() or "shirtless" in prompt.lower()


def test_fal_llm_design_mascot_rejects_malformed_response():
    fal = gateway({"output": json.dumps({"name": "incomplete"}), "usage": {"cost": 0.02}})
    provider = FalLLMProvider(fal, "google/gemini-2.5-flash", 0.05)
    with pytest.raises(ValueError, match="missing keys"):
        provider.design_mascot("deep sea diving", None, CostTracker(1))


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
    assert args["prompt"].startswith("character gestures at a bar of soap")
    assert NONVERBAL_CONTINUOUS_MOTION in args["prompt"]
    assert args["duration"] == "6"
    # 6 seconds at $0.045/s, flat — Hailuo has no variable usage.cost field
    assert tracker.total_spent_usd == pytest.approx(0.27)


def test_fal_video_uses_motion_prompt_not_raw_visual_prompt(tmp_path):
    """Regression test: FalVideoProvider used to always animate with the
    scene's raw visual_prompt field, even though the actual base image was
    built from a different, reconstructed prompt
    (pipeline.get_scene_image_prompt/mascot.build_scene_prompt discards
    visual_prompt whenever any structured field is present). Animating with
    text describing a different shot than what's in the frame risked Kling/
    Hailuo producing motion inconsistent with the actual image. The caller
    now passes a caller-provided motion_prompt (see pipeline.
    get_scene_motion_prompt — a dedicated motion prompt, not the raw
    visual_prompt and not the still-image composition prompt either), which
    must take priority — this test only checks that whatever motion_prompt
    is given wins over the raw visual_prompt field, independent of how the
    caller constructs it."""
    hero_path = tmp_path / "hero.png"
    hero_path.write_bytes(b"fake png bytes")
    fal = gateway({"video": {"url": "https://example.test/clip.mp4"}})
    fal.download = lambda url: b"fake mp4 bytes"
    provider = FalVideoProvider(fal, "fal-ai/minimax/hailuo-02/standard/image-to-video", 0.045)
    tracker = CostTracker(1)

    provider.generate_scene_video(
        {"visual_prompt": "a completely different, stale description"},
        hero_path, 0, tmp_path / "clip.mp4", tracker,
        motion_prompt="small mascot in the bottom-left corner pointing up at a floating soap bar",
    )

    args = fal.client.calls[0][1]["arguments"]
    assert args["prompt"].startswith("small mascot in the bottom-left corner pointing up at a floating soap bar")
    assert "stale description" not in args["prompt"]
    assert "must not speak or lip-sync" in args["prompt"]
    assert "must never go completely static" in args["prompt"]


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
    assert args["prompt"].startswith("dwarf mascot smiles and points staff up")
    assert "no talking mouth shapes" in args["prompt"]
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
