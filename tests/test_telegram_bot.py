import json

import pytest

from shorts_factory.dashboard import review_state
from shorts_factory.telegram_bot import TelegramController, TelegramNotConfigured, create_dispatcher


def make_artifacts(tmp_path):
    topic = tmp_path / "soap"
    topic.mkdir()
    (topic / "soap.mp4").write_bytes(b"video")
    (topic / "verification-report.json").write_text(json.dumps({"overall_pass": True}))
    return topic


def test_controller_requires_allowlist():
    with pytest.raises(TelegramNotConfigured):
        TelegramController(())


def test_unauthorized_user_cannot_approve(tmp_path):
    topic = make_artifacts(tmp_path)
    controller = TelegramController((1,), tmp_path)
    assert controller.authorized(2) is False
    assert review_state.load(topic).status == "pending"


def test_authorized_approval_is_explicit_and_does_not_publish(tmp_path):
    topic = make_artifacts(tmp_path)
    controller = TelegramController((1,), tmp_path)
    reply = controller.approve("soap", 1)
    assert review_state.load(topic).status == "approved"
    assert "not been published" in reply


def test_topic_path_traversal_is_rejected(tmp_path):
    controller = TelegramController((1,), tmp_path)
    with pytest.raises(ValueError):
        controller.topic_dir("../../etc")


def test_aiogram_dispatcher_registers_message_handler(tmp_path):
    make_artifacts(tmp_path)
    dispatcher = create_dispatcher(TelegramController((1,), tmp_path))
    assert "message" in dispatcher.resolve_used_update_types()


def test_aiogram_dispatcher_registers_callback_query_handler(tmp_path):
    # The planning flow's Confirm/Cancel and idea-pick buttons are delivered
    # as callback_query updates, not message updates — without this, aiogram
    # never asks Telegram to deliver them and every button silently no-ops.
    make_artifacts(tmp_path)
    dispatcher = create_dispatcher(TelegramController((1,), tmp_path))
    assert "callback_query" in dispatcher.resolve_used_update_types()
