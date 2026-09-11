"""Centralized configuration management for delta-cards using Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="sqlite:///data/delta_cards.db",
        description="Database connection URL",
    )

    # Discord Webhook & Alerts
    discord_webhook_url: Optional[str] = Field(
        default=None,
        description="Discord incoming webhook URL to post alerts to",
    )
    discord_alerts_enabled: bool = Field(
        default=True,
        description="Master switch to enable or disable sending Discord alerts",
    )

    # Alert Trigger Conditions
    alert_min_saving_percentage: Optional[float] = Field(
        default=None,
        description="Minimum saving percentage threshold (0-100) to trigger an alert",
    )
    alert_min_saving_absolute: Optional[float] = Field(
        default=None,
        description="Minimum absolute savings in currency (e.g. EUR) to trigger an alert",
    )
    alert_condition_operator: Literal["or", "and"] = Field(
        default="or",
        description="Logical operator when both percentage and absolute conditions are set ('or' / 'and')",
    )

    # Debug Mode: Controls alerting on previously saved database offers
    debug: bool = Field(
        default=False,
        description="If true, send alerts with saved offers. If false, do not send alerts for existing deals.",
    )

    @field_validator("alert_condition_operator", mode="before")
    @classmethod
    def normalize_operator(cls, v: Optional[str]) -> str:
        if v is None:
            return "or"
        val = str(v).strip().lower()
        if val not in ("or", "and"):
            raise ValueError(f"alert_condition_operator must be either 'or' or 'and', got '{v}'")
        return val


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of application settings."""
    return Settings()
