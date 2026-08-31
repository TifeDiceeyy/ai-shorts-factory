"""Real bug found 2026-08-29 while setting up YouTube OAuth for real:
config._env(name, default) only fell back to `default` when the env var was
entirely ABSENT from the environment — a .env line present but explicitly
blank (e.g. `YOUTUBE_TOKEN_FILE=`) returned "" instead of the documented
default, which crashed get_credentials() with a confusing IsADirectoryError
(Path("") resolves to the current directory)."""
from shorts_factory.config import _env


def test_env_falls_back_to_default_when_variable_is_entirely_absent(monkeypatch):
    monkeypatch.delenv("SHORTS_FACTORY_TEST_VAR", raising=False)
    assert _env("SHORTS_FACTORY_TEST_VAR", "the-default") == "the-default"


def test_env_falls_back_to_default_when_variable_is_present_but_blank(monkeypatch):
    """The actual bug: a .env line like `FOO=` sets os.environ["FOO"] = ""
    — present, not absent — so the old os.environ.get(name, default) never
    reached the default at all."""
    monkeypatch.setenv("SHORTS_FACTORY_TEST_VAR", "")
    assert _env("SHORTS_FACTORY_TEST_VAR", "the-default") == "the-default"


def test_env_falls_back_to_default_when_variable_is_only_whitespace(monkeypatch):
    monkeypatch.setenv("SHORTS_FACTORY_TEST_VAR", "   ")
    assert _env("SHORTS_FACTORY_TEST_VAR", "the-default") == "the-default"


def test_env_returns_the_real_value_when_actually_set(monkeypatch):
    monkeypatch.setenv("SHORTS_FACTORY_TEST_VAR", "real-value")
    assert _env("SHORTS_FACTORY_TEST_VAR", "the-default") == "real-value"


def test_env_with_no_default_still_returns_empty_string_not_none(monkeypatch):
    monkeypatch.delenv("SHORTS_FACTORY_TEST_VAR", raising=False)
    assert _env("SHORTS_FACTORY_TEST_VAR") == ""
