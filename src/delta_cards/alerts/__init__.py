"""Discord alerting module for delta-cards."""

from .service import (
    DEFAULT_OFFER_ALERT_MESSAGE,
    DEFAULT_BOT_NAME_POKEMON,
    DEFAULT_BOT_NAME_MAGIC,
    get_bot_username,
    AlertConditionResult,
    DiscordAlertService,
    AlertWorkerService,
    build_discord_embed,
    evaluate_alert_condition,
    format_product_type,
    send_discord_webhook,
    is_magic_offer,
    extract_alert_header,
)
from .templates import (
    AlertTemplateManager,
    render_alert_text,
    resolve_channel_key,
)

__all__ = [
    "DEFAULT_OFFER_ALERT_MESSAGE",
    "DEFAULT_BOT_NAME_POKEMON",
    "DEFAULT_BOT_NAME_MAGIC",
    "get_bot_username",
    "AlertConditionResult",
    "DiscordAlertService",
    "AlertWorkerService",
    "build_discord_embed",
    "evaluate_alert_condition",
    "format_product_type",
    "send_discord_webhook",
    "is_magic_offer",
    "extract_alert_header",
    "AlertTemplateManager",
    "render_alert_text",
    "resolve_channel_key",
]
