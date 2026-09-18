"""Centralized configuration management for delta-cards using Pydantic Settings."""

from functools import lru_cache
from typing import Literal, Optional
from pydantic import AliasChoices, Field, field_validator, model_validator
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

    # Discord Webhooks & Alerts
    pokemon_cards_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "POKEMON_CARDS_WEBHOOK",
            "pokemon_cards_webhook",
            "DISCORD_WEBHOOK_URL",
            "discord_webhook_url",
        ),
        description="Discord incoming webhook URL to post Pokémon single card alerts to",
    )
    pokemon_booster_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "POKEMON_BOOSTER_WEBHOOK",
            "pokemon_booster_webhook",
        ),
        description="Discord incoming webhook URL to post Pokémon booster and blister alerts to",
    )
    pokemon_booster_box_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "POKEMON_BOOSTER_BOX_WEBHOOK",
            "pokemon_booster_box_webhook",
        ),
        description="Discord incoming webhook URL to post Pokémon booster box, ETB, and sealed box alerts to",
    )
    magic_cards_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "MAGIC_CARDS_WEBHOOK",
            "magic_cards_webhook",
        ),
        description="Discord incoming webhook URL to post Magic: The Gathering single card alerts to",
    )
    magic_booster_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "MAGIC_BOOSTER_WEBHOOK",
            "magic_booster_webhook",
        ),
        description="Discord incoming webhook URL to post Magic: The Gathering booster and blister alerts to",
    )
    magic_booster_box_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "MAGIC_BOOSTER_BOX_WEBHOOK",
            "magic_booster_box_webhook",
        ),
        description="Discord incoming webhook URL to post Magic: The Gathering booster box, display, bundle, and sealed box alerts to",
    )
    magic_sealed_webhook: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "MAGIC_SEALED_WEBHOOK",
            "magic_sealed_webhook",
        ),
        description="Legacy alias for Magic sealed product webhook",
    )
    discord_webhook_url: Optional[str] = Field(
        default=None,
        description="Resolved active Discord incoming webhook URL to post alerts to",
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
    alert_wild_offer_emoji: str = Field(
        default="<:pokeball_light:1548052080357412995>",
        description="Custom Discord emoji or unicode tag preceding wild offer message (e.g. <:pokeball_light:123456789>)",
    )
    time_between_offers: float = Field(
        default=0.0,
        validation_alias=AliasChoices(
            "time_between_offers",
            "alert_time_between_offers",
            "delay_between_offers",
            "alert_delay_between_offers",
        ),
        description="Delay in seconds between sending offer alerts to Discord to prevent flooding (default: 0)",
    )

    # Alert Service / Worker Settings
    alert_dispatch_interval_minutes: Optional[int] = Field(
        default=None,
        description="Interval in minutes between periodic alert dispatches (default: 30 minutes)",
    )
    alert_dispatch_interval_seconds: int = Field(
        default=30,
        description="Interval in seconds between worker queue checks when no deals are pending (default: 30s)",
    )
    alert_templates_dir: str = Field(
        default="config/templates/alerts",
        validation_alias=AliasChoices(
            "ALERT_TEMPLATES_DIR",
            "alert_templates_dir",
        ),
        description="Path to directory containing alert message templates per channel",
    )

    # Debug Mode: Controls alerting on previously saved database offers
    debug: bool = Field(
        default=False,
        description="If true, send alerts with saved offers. If false, do not send alerts for existing deals.",
    )

    # LLM False-Positive Verification Settings
    llm_verifier_enabled: bool = Field(
        default=True,
        description="Enable LLM false positive verification before dispatching alerts.",
    )
    llm_verifier_model_path: str = Field(
        default="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        description="Path to GGUF model for LLM verification.",
    )
    llm_verifier_min_confidence: float = Field(
        default=70.0,
        description="Minimum LLM confidence percentage (0-100) required to dispatch an alert.",
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

    @model_validator(mode="after")
    def resolve_webhooks(self) -> "Settings":
        if not self.magic_booster_box_webhook and self.magic_sealed_webhook:
            self.magic_booster_box_webhook = self.magic_sealed_webhook
        elif not self.magic_sealed_webhook and self.magic_booster_box_webhook:
            self.magic_sealed_webhook = self.magic_booster_box_webhook

        if not self.discord_webhook_url:
            self.discord_webhook_url = (
                self.magic_cards_webhook
                or self.magic_booster_box_webhook
                or self.pokemon_cards_webhook
                or self.pokemon_booster_box_webhook
                or self.pokemon_booster_webhook
            )
        return self

    def get_webhook_url(
        self,
        game: str = "pokemon",
        product_type: Optional[str] = None,
    ) -> Optional[str]:
        """Returns the appropriate Discord webhook URL for the given game and product category."""
        if str(game).lower() in ("magic", "mtg"):
            pt = str(product_type or "").lower().strip()
            if pt in ("booster_pack", "booster", "blister", "pack", "sobre"):
                return (
                    self.magic_booster_webhook
                    or self.magic_sealed_webhook
                    or self.magic_cards_webhook
                    or self.discord_webhook_url
                )
            if pt in (
                "booster_box",
                "display",
                "etb",
                "elite_trainer_box",
                "caja",
                "box",
                "tin",
                "box_set",
                "deck",
                "bundle",
                "booster_bundle",
                "non_single",
                "sealed",
            ):
                return (
                    self.magic_booster_box_webhook
                    or self.magic_sealed_webhook
                    or self.magic_cards_webhook
                    or self.discord_webhook_url
                )
            return self.magic_cards_webhook or self.discord_webhook_url or self.pokemon_cards_webhook

        pt = str(product_type or "").lower().strip()

        # Booster / Blister pack alerts
        if pt in ("booster_pack", "booster", "blister", "pack", "sobre"):
            return (
                self.pokemon_booster_webhook
                or self.pokemon_cards_webhook
                or self.discord_webhook_url
            )

        # Booster Box / Display / ETB / Sealed box alerts
        if pt in (
            "booster_box",
            "display",
            "etb",
            "elite_trainer_box",
            "caja",
            "box",
            "tin",
            "box_set",
            "deck",
            "bundle",
            "booster_bundle",
        ):
            return (
                self.pokemon_booster_box_webhook
                or self.pokemon_cards_webhook
                or self.discord_webhook_url
            )

        # Default / Single cards
        return (
            self.pokemon_cards_webhook
            or self.discord_webhook_url
            or self.pokemon_booster_box_webhook
            or self.pokemon_booster_webhook
            or self.magic_cards_webhook
        )

    @property
    def alert_interval(self) -> int:
        """Returns the effective interval in seconds between alert dispatches."""
        if self.alert_dispatch_interval_minutes is not None:
            return self.alert_dispatch_interval_minutes * 60
        return self.alert_dispatch_interval_seconds


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton instance of application settings."""
    return Settings()
