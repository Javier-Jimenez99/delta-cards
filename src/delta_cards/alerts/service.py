"""Discord alerting service and condition evaluation for card deals."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
from pydantic import BaseModel

from ..models import OfferListing
from ..settings import Settings, get_settings

logger = logging.getLogger("delta_cards.alerts")


class AlertConditionResult(BaseModel):
    """Result of evaluating an offer against alert conditions."""
    is_triggered: bool
    savings_absolute: float
    savings_percentage: float
    offer_price: float
    market_price: float
    reason: str


def evaluate_alert_condition(
    offer_price: float,
    market_price: float,
    settings: Optional[Settings] = None,
) -> AlertConditionResult:
    """Evaluates whether an offer meets saving thresholds to trigger an alert.

    Args:
        offer_price: The listing price found on the marketplace (e.g. Wallapop).
        market_price: The reference market price (e.g. Cardmarket low price).
        settings: Optional Settings instance. If None, uses global get_settings().

    Returns:
        AlertConditionResult with evaluation metrics and trigger status.
    """
    if settings is None:
        settings = get_settings()

    if market_price is None or market_price <= 0 or offer_price < 0:
        return AlertConditionResult(
            is_triggered=False,
            savings_absolute=0.0,
            savings_percentage=0.0,
            offer_price=offer_price or 0.0,
            market_price=market_price or 0.0,
            reason="Invalid or missing market price",
        )

    savings_absolute = round(market_price - offer_price, 2)
    if savings_absolute <= 0:
        return AlertConditionResult(
            is_triggered=False,
            savings_absolute=savings_absolute,
            savings_percentage=0.0,
            offer_price=offer_price,
            market_price=market_price,
            reason=f"No discount (offer price {offer_price:.2f} >= market price {market_price:.2f})",
        )

    savings_percentage = round((savings_absolute / market_price) * 100.0, 2)

    pct_threshold = settings.alert_min_saving_percentage
    abs_threshold = settings.alert_min_saving_absolute
    operator = settings.alert_condition_operator.lower()

    # If neither threshold is configured, any positive saving triggers
    if pct_threshold is None and abs_threshold is None:
        return AlertConditionResult(
            is_triggered=True,
            savings_absolute=savings_absolute,
            savings_percentage=savings_percentage,
            offer_price=offer_price,
            market_price=market_price,
            reason="Triggered: positive saving (no minimum threshold configured)",
        )

    pct_met = (savings_percentage >= pct_threshold) if pct_threshold is not None else None
    abs_met = (savings_absolute >= abs_threshold) if abs_threshold is not None else None

    if pct_met is not None and abs_met is not None:
        if operator == "and":
            triggered = pct_met and abs_met
            reason_details = (
                f"pct >= {pct_threshold}% ({pct_met}) AND abs >= {abs_threshold}€ ({abs_met})"
            )
        else:  # default: "or"
            triggered = pct_met or abs_met
            reason_details = (
                f"pct >= {pct_threshold}% ({pct_met}) OR abs >= {abs_threshold}€ ({abs_met})"
            )
    elif pct_met is not None:
        triggered = pct_met
        reason_details = f"pct >= {pct_threshold}% ({pct_met})"
    else:  # abs_met is not None
        triggered = abs_met
        reason_details = f"abs >= {abs_threshold}€ ({abs_met})"

    status_str = "Triggered" if triggered else "Below threshold"
    reason = (
        f"{status_str}: {savings_absolute:.2f}€ savings ({savings_percentage:.1f}%), "
        f"condition: {reason_details}"
    )

    return AlertConditionResult(
        is_triggered=triggered,
        savings_absolute=savings_absolute,
        savings_percentage=savings_percentage,
        offer_price=offer_price,
        market_price=market_price,
        reason=reason,
    )


def build_discord_embed(
    offer: OfferListing,
    result: AlertConditionResult,
) -> Dict[str, Any]:
    """Formats a deal offer into a Discord embed object."""
    fields = [
        {"name": "💶 Wallapop Price", "value": f"**{offer.price:.2f} €**", "inline": True},
        {"name": "📈 Market Price", "value": f"{result.market_price:.2f} €", "inline": True},
        {
            "name": "💰 Savings",
            "value": f"**-{result.savings_absolute:.2f} €** ({result.savings_percentage:.1f}%)",
            "inline": True,
        },
    ]

    if offer.product_type:
        fields.append({"name": "📦 Type", "value": offer.product_type, "inline": True})
    if offer.location:
        fields.append({"name": "📍 Location", "value": offer.location, "inline": True})
    if offer.cardmarket_url:
        fields.append(
            {"name": "🔗 Cardmarket", "value": f"[View Product]({offer.cardmarket_url})", "inline": True}
        )

    embed: Dict[str, Any] = {
        "title": f"🔥 Deal: {offer.title[:200]}",
        "url": offer.link,
        "color": 0x2ECC71,  # Emerald Green
        "fields": fields,
        "footer": {"text": "Delta Cards Scraper Alert"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if offer.image_url:
        embed["thumbnail"] = {"url": offer.image_url}

    return embed


def send_discord_webhook(
    webhook_url: str,
    embeds: List[Dict[str, Any]],
    content: Optional[str] = None,
    timeout: float = 10.0,
) -> bool:
    """Sends one or more embeds to a Discord incoming webhook URL.

    Batches embeds into chunks of 10 (Discord API maximum per request).

    Args:
        webhook_url: The Discord webhook endpoint URL.
        embeds: List of embed dictionaries to send.
        content: Optional top-level message text.
        timeout: HTTP request timeout in seconds.

    Returns:
        True if all batches were delivered successfully, False otherwise.
    """
    if not webhook_url or not webhook_url.strip():
        logger.debug("Discord webhook URL is empty; skipping webhook delivery.")
        return False

    if not embeds:
        return True

    success = True
    chunk_size = 10  # Discord API limit per webhook execution

    for i in range(0, len(embeds), chunk_size):
        chunk = embeds[i : i + chunk_size]
        payload: Dict[str, Any] = {
            "username": "Delta Cards Alert",
            "embeds": chunk,
        }
        if content and i == 0:
            payload["content"] = content

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(webhook_url, json=payload)
                if response.status_code in (200, 204):
                    logger.info(f"Successfully sent Discord alert with {len(chunk)} item(s).")
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "unknown")
                    logger.warning(
                        f"Discord rate limited request (HTTP 429). Retry-After: {retry_after}s"
                    )
                    success = False
                else:
                    logger.warning(
                        f"Discord webhook failed with HTTP {response.status_code}: {response.text}"
                    )
                    success = False
        except Exception as exc:
            logger.warning(f"Error dispatching Discord webhook alert: {exc}")
            success = False

    return success


class DiscordAlertService:
    """Coordinates evaluating offers and dispatching Discord notifications."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def process_and_alert_offers(
        self,
        offers: List[OfferListing],
        reference_prices: Optional[Dict[str, float]] = None,
    ) -> int:
        """Evaluates scraped offers, identifies qualifying deals, and sends Discord alerts.

        Args:
            offers: Scraped OfferListing items.
            reference_prices: Optional mapping from offer.id or product_id to market price.
                              If not provided, falls back to offer.cardmarket_low_price.

        Returns:
            Number of alert notifications successfully triggered and dispatched.
        """
        if not self.settings.discord_alerts_enabled:
            logger.info("Discord alerts are disabled in settings (DISCORD_ALERTS_ENABLED=false).")
            return 0

        webhook_url = self.settings.discord_webhook_url
        if not webhook_url:
            logger.debug(
                "No DISCORD_WEBHOOK_URL configured in settings. Skipping Discord alerts."
            )
            return 0

        matching_embeds = []
        for offer in offers:
            market_price = None
            if reference_prices and offer.id in reference_prices:
                market_price = reference_prices[offer.id]
            elif reference_prices and offer.cardmarket_product_id in reference_prices:
                market_price = reference_prices[offer.cardmarket_product_id]
            elif offer.cardmarket_low_price is not None:
                market_price = offer.cardmarket_low_price

            if market_price is None:
                continue

            cond = evaluate_alert_condition(
                offer_price=offer.price,
                market_price=market_price,
                settings=self.settings,
            )

            if cond.is_triggered:
                logger.info(
                    f"Deal detected for '{offer.title[:35]}': {cond.reason}"
                )
                embed = build_discord_embed(offer, cond)
                matching_embeds.append(embed)

        if not matching_embeds:
            logger.debug("No offers met the configured Discord alert conditions.")
            return 0

        logger.info(f"Dispatching {len(matching_embeds)} Discord alert(s)...")
        header = f"🚨 **Delta Cards Scraper Alert**: Found {len(matching_embeds)} deal(s) matching criteria!"
        ok = send_discord_webhook(
            webhook_url=webhook_url,
            embeds=matching_embeds,
            content=header,
        )
        return len(matching_embeds) if ok else 0
