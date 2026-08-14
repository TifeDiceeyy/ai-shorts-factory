import json

import pytest

from shorts_factory.dashboard import review_state
from shorts_factory.telegram_bot import TelegramBot, TelegramNotConfigured


def update(user_id, text):
    return {"message": {"chat": {"id": 99}, "from": {"id": user_id}, "text": text}}


def make_artifacts(tmp_path):
    topic = tmp_path / "soap"
    topic.mkdir()
    (topic / "soap.mp4").write_bytes(b"video")
    (topic / "verification-report.json").write_text(json.dumps({"overall_pass": True}))
    return topic


def test_bot_requires_allowlist():
    with pytest.raises(TelegramNotConfigured):
        TelegramBot("token", ())


def test_unauthorized_user_cannot_approve(tmp_path, monkeypatch):
    topic = make_artifacts(tmp_path)
    bot = TelegramBot("token", (1,), tmp_path)
    sent = []
    monkeypatch.setattr(bot, "send_message", lambda chat, text: sent.append(text))
    bot.handle_update(update(2, "/approve soap"))
    assert review_state.load(topic).status == "pending"
    assert sent == ["Unauthorized."]


def test_authorized_approval_is_explicit_and_does_not_publish(tmp_path, monkeypatch):
    topic = make_artifacts(tmp_path)
    bot = TelegramBot("token", (1,), tmp_path)
    sent = []
    monkeypatch.setattr(bot, "send_message", lambda chat, text: sent.append(text))
    bot.handle_update(update(1, "/approve soap"))
    assert review_state.load(topic).status == "approved"
    assert "not been published" in sent[-1]


def test_topic_path_traversal_is_rejected(tmp_path):
    bot = TelegramBot("token", (1,), tmp_path)
    with pytest.raises(ValueError):
        bot._topic_dir("../../etc")
