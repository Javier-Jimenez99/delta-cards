"""Discord alerting module for delta-cards."""

from .service import (
    AlertConditionResult,
    DiscordAlertService,
    build_discord_embed,
    evaluate_alert_condition,
    send_discord_webhook,
)

__all__ = [
    "AlertConditionResult",
    "DiscordAlertService",
    "build_discord_embed",
    "evaluate_alert_condition",
    "send_discord_webhook",
]
