"""Regression test for the gap found in review: budget_cap_is_stub was
recorded but never enforced, so a real (non-stub) provider could silently
run against the $2.00 stub default instead of an explicitly-approved budget."""
import pytest
from shorts_factory.config import BudgetApprovalRequired, ProviderConfig, Settings, require_budget_approval_if_paid


def _settings(*, llm="stub", tts="stub", image="stub", search="stub", budget_is_stub=True) -> Settings:
    return Settings(
        book_file="",
        output_language="English",
        visual_style="test style",
        budget_cap_usd=2.00,
        budget_cap_is_stub=budget_is_stub,
        music_sfx_source="",
        llm=ProviderConfig(kind="llm", provider=llm, model_or_voice=""),
        tts=ProviderConfig(kind="tts", provider=tts, model_or_voice=""),
        image=ProviderConfig(kind="image", provider=image, model_or_voice=""),
        search=ProviderConfig(kind="search", provider=search, model_or_voice=""),
        search_api_key="",
        anthropic_api_key="",
        openai_api_key="",
        elevenlabs_api_key="",
        fal_key="",
        llm_cost_per_script_usd=0,
        tts_cost_per_1k_chars_usd=0,
        image_cost_per_image_usd=0,
        youtube_client_secrets_file="",
        youtube_token_file="",
        telegram_bot_token="",
        telegram_allowed_user_ids=(),
    )


def test_all_stub_never_requires_approval_even_with_stub_budget():
    require_budget_approval_if_paid(_settings(budget_is_stub=True))  # must not raise


def test_real_llm_with_unset_budget_is_blocked():
    with pytest.raises(BudgetApprovalRequired):
        require_budget_approval_if_paid(_settings(llm="anthropic", budget_is_stub=True))


def test_real_tts_with_unset_budget_is_blocked():
    with pytest.raises(BudgetApprovalRequired):
        require_budget_approval_if_paid(_settings(tts="elevenlabs", budget_is_stub=True))


def test_real_image_with_unset_budget_is_blocked():
    with pytest.raises(BudgetApprovalRequired):
        require_budget_approval_if_paid(_settings(image="fal", budget_is_stub=True))


def test_real_search_with_unset_budget_is_blocked():
    with pytest.raises(BudgetApprovalRequired):
        require_budget_approval_if_paid(_settings(search="tavily", budget_is_stub=True))


def test_real_provider_with_explicit_budget_is_allowed():
    require_budget_approval_if_paid(_settings(llm="anthropic", budget_is_stub=False))  # must not raise


def test_blocked_error_names_the_real_providers():
    with pytest.raises(BudgetApprovalRequired) as exc_info:
        require_budget_approval_if_paid(_settings(llm="anthropic", image="fal", budget_is_stub=True))
    assert "LLM" in exc_info.value.real_providers
    assert "IMAGE" in exc_info.value.real_providers
    assert "TTS" not in exc_info.value.real_providers
