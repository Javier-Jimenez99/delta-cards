"""Tests for centralized Settings and environment configuration."""

import pytest
from delta_cards.settings import Settings, get_settings
from delta_cards.database.connection import get_db_url


def test_default_settings():
    """Verify default values when no environment variables are set."""
    settings = Settings(
        _env_file=None,
        database_url="sqlite:///data/delta_cards.db",
        discord_webhook_url=None,
        discord_alerts_enabled=True,
        alert_min_saving_percentage=None,
        alert_min_saving_absolute=None,
        alert_condition_operator="or",
    )
    assert settings.database_url == "sqlite:///data/delta_cards.db"
    assert settings.discord_webhook_url is None
    assert settings.discord_alerts_enabled is True
    assert settings.alert_min_saving_percentage is None
    assert settings.alert_min_saving_absolute is None
    assert settings.alert_condition_operator == "or"
    assert settings.debug is False


def test_settings_from_environment(monkeypatch):
    """Verify settings correctly load from environment variables."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/testdb")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/abc")
    monkeypatch.setenv("DISCORD_ALERTS_ENABLED", "false")
    monkeypatch.setenv("ALERT_MIN_SAVING_PERCENTAGE", "25.5")
    monkeypatch.setenv("ALERT_MIN_SAVING_ABSOLUTE", "15.0")
    monkeypatch.setenv("ALERT_CONDITION_OPERATOR", "and")
    monkeypatch.setenv("DEBUG", "true")

    get_settings.cache_clear()
    settings = Settings(_env_file=None)

    assert settings.database_url == "postgresql+psycopg://user:pass@localhost:5432/testdb"
    assert settings.discord_webhook_url == "https://discord.com/api/webhooks/123/abc"
    assert settings.discord_alerts_enabled is False
    assert settings.alert_min_saving_percentage == 25.5
    assert settings.alert_min_saving_absolute == 15.0
    assert settings.alert_condition_operator == "and"
    assert settings.debug is True


def test_settings_load_from_env_file(tmp_path):
    """Verify that Settings can load alert thresholds from a .env file."""
    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "ALERT_MIN_SAVING_PERCENTAGE=30.0\n"
        "ALERT_MIN_SAVING_ABSOLUTE=20.0\n"
        "ALERT_CONDITION_OPERATOR=and\n"
        "DISCORD_ALERTS_ENABLED=true\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=str(env_file))
    assert settings.alert_min_saving_percentage == 30.0
    assert settings.alert_min_saving_absolute == 20.0
    assert settings.alert_condition_operator == "and"
    assert settings.discord_alerts_enabled is True


def test_invalid_condition_operator():
    """Verify that an invalid condition operator raises a validation error."""
    with pytest.raises(ValueError, match="alert_condition_operator must be either 'or' or 'and'"):
        Settings(_env_file=None, alert_condition_operator="xor")


def test_get_db_url_uses_settings(monkeypatch):
    """Verify get_db_url() accesses settings rather than raw os.getenv."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://dbuser:secret@dbhost:5432/proddb")
    get_settings.cache_clear()

    url = get_db_url()
    assert url == "postgresql+psycopg://dbuser:secret@dbhost:5432/proddb"
