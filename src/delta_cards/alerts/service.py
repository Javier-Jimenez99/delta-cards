"""Discord alerting service and condition evaluation for card deals."""

import logging
from pathlib import Path
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
from pydantic import BaseModel

from ..models import OfferListing, is_card_offer
from ..settings import Settings, get_settings
from .templates import render_alert_text, resolve_channel_key

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

    pct_threshold = (
        abs(settings.alert_min_saving_percentage)
        if settings.alert_min_saving_percentage is not None
        else None
    )
    abs_threshold = (
        abs(settings.alert_min_saving_absolute)
        if settings.alert_min_saving_absolute is not None
        else None
    )
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


def format_product_type(product_type: Optional[str]) -> str:
    """Translates raw product type identifier to a user-friendly category label."""
    if not product_type:
        return "Otro"
    pt = str(product_type).lower().strip()
    mapping = {
        "single_card": "Carta",
        "card": "Carta",
        "carta": "Carta",
        "booster_pack": "Paquete",
        "pack": "Paquete",
        "sobre": "Paquete",
        "booster_box": "Caja",
        "box": "Caja",
        "caja": "Caja",
        "etb": "Caja (ETB)",
        "elite_trainer_box": "Caja (ETB)",
        "deck": "Mazo",
        "mazo": "Mazo",
        "lote": "Lote",
        "lot": "Lote",
    }
    return mapping.get(pt, pt.replace("_", " ").title())


def is_magic_offer(offer: Any) -> bool:
    """Returns True if the offer corresponds to Magic: The Gathering."""
    if not offer:
        return False
    g = getattr(offer, "game", None)
    if g:
        if str(g).lower() == "magic":
            return True
        if str(g).lower() == "pokemon":
            return False
    sp = str(getattr(offer, "search_profile_id", "") or "").lower()
    title = str(getattr(offer, "title", "") or "").lower()
    desc = str(getattr(offer, "description", "") or "").lower()
    cm_url = str(getattr(offer, "cardmarket_url", "") or "").lower()
    if "/magic/" in cm_url:
        return True
    if "magic" in sp or "mtg" in sp:
        return True
    if any(k in title for k in ("magic", "mtg", "scryfall")) or any(k in desc for k in ("magic the gathering", "mtg")):
        return True
    return False


def extract_alert_header(rendered_text: Optional[str]) -> str:
    """Extracts the header line from rendered alert text to display outside the green embed."""
    if not rendered_text:
        return DEFAULT_OFFER_ALERT_MESSAGE
    lines = [l.strip() for l in rendered_text.strip().split("\n") if l.strip()]
    field_pattern = re.compile(r"^(.*?)\*\*(.*?):\*\*\s*(.*)$")
    header_lines = []
    for line in lines:
        if re.match(r"^\*{0,2}\[?Oferta:.*", line) or field_pattern.match(line):
            break
        header_lines.append(line)
    return "\n".join(header_lines) if header_lines else DEFAULT_OFFER_ALERT_MESSAGE


def build_discord_embed(
    offer: OfferListing,
    result: AlertConditionResult,
    llm_verification: Optional[Any] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Formats a deal offer into a Discord embed object.

    If `description` is provided, parses fields into 3-column inline format
    inside the Discord embed's green block, leaving the header to be rendered outside.
    Otherwise, builds the default embed structure with fields.
    """
    if description:
        # Determine platform URL
        is_vinted = (getattr(offer, "platform", None) == "vinted") or ("vinted" in (offer.link or "").lower())
        if is_vinted:
            offer_url = offer.link or ""
            platform_name = "Vinted"
            platform_emoji = "<:vinted:1549420019518013470>"
            offer_link_text = "Oferta en Vinted"
        else:
            offer_url = (offer.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
            platform_name = "Wallapop"
            platform_emoji = "<:wallapop:1548103689477038080>"
            offer_link_text = "Oferta en Wallapop"

        lines = [l.strip() for l in description.strip().split("\n") if l.strip()]
        header_lines = []
        parsed_fields = []
        field_pattern = re.compile(r"^(.*?)\*\*(.*?):\*\*\s*(.*)$")

        for idx, line in enumerate(lines):
            # First line is typically the header (e.g. wild offer greeting)
            if idx == 0 and not field_pattern.match(line) and not line.startswith("**[Oferta:"):
                header_lines.append(line)
                continue
            # Skip title line from template since it is rendered via embed["title"] and embed["url"]
            if re.match(r"^\*{0,2}\[?Oferta:.*", line):
                continue
            m = field_pattern.match(line)
            if m:
                prefix, name, val = m.groups()
                full_name = f"{prefix.strip()} {name.strip()}".strip()
                val_clean = val.strip()
                if not val_clean.startswith("**") and not val_clean.startswith("["):
                    val_clean = f"**{val_clean}**"
                parsed_fields.append({"name": full_name, "value": val_clean, "inline": True})
            else:
                header_lines.append(line)

        # Fallback fields if no structured fields were parsed from template
        if not parsed_fields:
            cardmarket_url = offer.cardmarket_url
            if not cardmarket_url and offer.cardmarket_product_id:
                game_category = "Magic" if is_magic_offer(offer) else "Pokemon"
                cardmarket_url = f"https://www.cardmarket.com/es/{game_category}/Products?idProduct={offer.cardmarket_product_id}"

            avg_price = getattr(offer, "cardmarket_avg_price", None)
            if avg_price is None:
                avg_price = result.market_price if result else getattr(offer, "cardmarket_low_price", None)
            low_price = getattr(offer, "cardmarket_low_price", None)

            avg_str = f"{avg_price:.2f} €" if avg_price is not None else "N/A"
            low_str = f"{low_price:.2f} €" if low_price is not None else "N/A"
            type_label = format_product_type(offer.product_type)
            location_str = offer.location or "N/A"

            links = [f"[{offer_link_text}]({offer_url})"]
            if cardmarket_url:
                links.append(f"[Producto en Cardmarket]({cardmarket_url})")
            links_str = " • ".join(links)

            parsed_fields = [
                {"name": f"{platform_emoji} {platform_name}", "value": f"**{offer.price:.2f} €**", "inline": True},
                {"name": "<:cardmarket:1548103777750487056> Precio medio", "value": f"**{avg_str}**", "inline": True},
                {"name": "<:cardmarket:1548103777750487056> Precio mínimo", "value": f"**{low_str}**", "inline": True},
                {"name": "📦 Tipo", "value": f"**{type_label}**", "inline": True},
                {"name": "📍 Ubicación", "value": f"**{location_str}**", "inline": True},
                {"name": "🔗 Enlaces", "value": links_str, "inline": True},
            ]

        if llm_verification:
            score = getattr(llm_verification, "confidence_score", None)
            if score is None and isinstance(llm_verification, dict):
                score = llm_verification.get("confidence_score")
            confirmed = getattr(llm_verification, "confirmed_count", None)
            if confirmed is None and isinstance(llm_verification, dict):
                confirmed = llm_verification.get("confirmed_count")
            total = getattr(llm_verification, "total_questions", 20)
            if total is None and isinstance(llm_verification, dict):
                total = llm_verification.get("total_questions", 20)

            if confirmed is not None and total and total > 0:
                pass_pct = (confirmed / total) * 100.0
            elif score is not None:
                pass_pct = float(score)
            else:
                pass_pct = 0.0

            if pass_pct >= 67.0:
                value_msg = f"**{round(pass_pct)}%** ✅"
            elif pass_pct <= 33.0:
                value_msg = f"**{round(pass_pct)}%** ❌"
            else:
                value_msg = f"**{round(pass_pct)}%** ⚠️"

            parsed_fields.append({
                "name": "🤖 Verificación IA",
                "value": value_msg,
                "inline": True,
            })

        embed: Dict[str, Any] = {
            "title": f"Oferta: {offer.title[:200]}",
            "url": offer_url,
            "color": 0x2ECC71,  # Emerald Green
            "fields": parsed_fields,
            "footer": {"text": "Alerta de ofertas Delta Cards"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if offer.image_url:
            embed["thumbnail"] = {"url": offer.image_url}

        return embed

    # Determine platform
    is_vinted = (getattr(offer, "platform", None) == "vinted") or ("vinted" in (offer.link or "").lower())

    if is_vinted:
        offer_url = offer.link or ""
        platform_name = "Vinted"
        platform_emoji = "<:vinted:1549420019518013470>"
        offer_link_text = "Oferta en Vinted"
    else:
        # Ensure Wallapop link is normalized
        wallapop_url = (offer.link or "").replace("https://wallapop.com/", "https://es.wallapop.com/")
        offer_url = wallapop_url
        platform_name = "Wallapop"
        platform_emoji = "<:wallapop:1548103689477038080>"
        offer_link_text = "Oferta en Wallapop"

    # Resolve Cardmarket URL (either explicitly set or built from product ID)
    cardmarket_url = offer.cardmarket_url
    if not cardmarket_url and offer.cardmarket_product_id:
        game_category = "Magic" if is_magic_offer(offer) else "Pokemon"
        cardmarket_url = f"https://www.cardmarket.com/es/{game_category}/Products?idProduct={offer.cardmarket_product_id}"

    # Average market price ("Precio medio") from TCGDex
    avg_price = getattr(offer, "cardmarket_avg_price", None)
    if avg_price is None:
        avg_price = result.market_price if result else getattr(offer, "cardmarket_low_price", None)

    low_price = getattr(offer, "cardmarket_low_price", None)

    avg_str = f"{avg_price:.2f} €" if avg_price is not None else "N/A"
    low_str = f"{low_price:.2f} €" if low_price is not None else "N/A"

    # Translated product category (Carta, Paquete, Caja, etc.)
    type_label = format_product_type(offer.product_type)

    location_str = offer.location or "N/A"

    fields = [
        {"name": f"{platform_emoji} {platform_name}", "value": f"**{offer.price:.2f} €**", "inline": True},
        {"name": "<:cardmarket:1548103777750487056> Precio medio", "value": f"**{avg_str}**", "inline": True},
        {"name": "<:cardmarket:1548103777750487056> Precio mínimo", "value": f"**{low_str}**", "inline": True},
        {"name": "📦 Tipo", "value": f"**{type_label}**", "inline": True},
        {"name": "📍 Ubicación", "value": f"**{location_str}**", "inline": True},
    ]

    # Enlaces a la oferta y a la página de Cardmarket
    links = [f"[{offer_link_text}]({offer_url})"]
    if cardmarket_url:
        links.append(f"[Producto en Cardmarket]({cardmarket_url})")
    links_str = " • ".join(links)

    # LLM False-Positive Verification Field
    if llm_verification:
        score = getattr(llm_verification, "confidence_score", None)
        if score is None and isinstance(llm_verification, dict):
            score = llm_verification.get("confidence_score")
        confirmed = getattr(llm_verification, "confirmed_count", None)
        if confirmed is None and isinstance(llm_verification, dict):
            confirmed = llm_verification.get("confirmed_count")
        total = getattr(llm_verification, "total_questions", 20)
        if total is None and isinstance(llm_verification, dict):
            total = llm_verification.get("total_questions", 20)
        is_ver = getattr(llm_verification, "is_verified", None)
        if is_ver is None and isinstance(llm_verification, dict):
            is_ver = llm_verification.get("is_verified")

        # Determine check pass percentage
        if confirmed is not None and total and total > 0:
            pass_pct = (confirmed / total) * 100.0
        elif score is not None:
            pass_pct = float(score)
        else:
            pass_pct = 0.0

        # Hard veto overrides confidence
        if pass_pct >= 67.0:
            value_msg = f"**{round(pass_pct)}%** ✅"
        elif pass_pct <= 33.0:
            value_msg = f"**{round(pass_pct)}%** ❌"
        else:
            value_msg = f"**{round(pass_pct)}%** ⚠️"

        fields.append({
            "name": "🤖 Verificación IA",
            "value": value_msg,
            "inline": True,
        })
    else:
        fields.append({
            "name": "🔗 Enlaces",
            "value": links_str,
            "inline": True,
        })

    embed: Dict[str, Any] = {
        "title": f"Oferta: {offer.title[:200]}",
        "url": offer_url,
        "description": links_str,
        "color": 0x2ECC71,  # Emerald Green
        "fields": fields,
        "footer": {"text": "Alerta de ofertas Delta Cards"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if offer.image_url:
        embed["thumbnail"] = {"url": offer.image_url}

    return embed


DEFAULT_OFFER_ALERT_MESSAGE: str = "<:pokeball_light:1548052080357412995> ¡Ha aparecido una oferta salvaje!"
DEFAULT_MAGIC_OFFER_ALERT_MESSAGE: str = "🔮 ¡Ha aparecido una oferta salvaje de Magic!"

DEFAULT_BOT_NAME_POKEMON: str = "El Meowth de la suerte"
DEFAULT_BOT_NAME_MAGIC: str = "Ragavan Buscatesoros"


def get_bot_username(game: Optional[str] = None) -> str:
    """Returns the bot username based on game ('magic' -> 'Ragavan Buscatesoros', other -> 'El Meowth de la suerte')."""
    if game and str(game).lower() in ("magic", "mtg"):
        return DEFAULT_BOT_NAME_MAGIC
    return DEFAULT_BOT_NAME_POKEMON


def send_discord_webhook(
    webhook_url: str,
    embeds: List[Dict[str, Any]],
    content: Optional[Union[str, List[str]]] = DEFAULT_OFFER_ALERT_MESSAGE,
    timeout: float = 10.0,
    delay_between_offers: float = 0.0,
    username: Optional[Union[str, List[str]]] = None,
) -> bool:
    """Sends each offer embed as a separate notification message to Discord with alert text.

    Args:
        webhook_url: The Discord webhook endpoint URL.
        embeds: List of embed dictionaries to send.
        content: Message text (or list of texts per embed) displayed before each offer embed.
        timeout: HTTP request timeout in seconds.
        delay_between_offers: Delay in seconds to pause between consecutive offer alerts (default: 0.0).
        username: Bot username displayed in Discord (defaults to 'Delta Cards Alert' if not provided).

    Returns:
        True if all embeds were delivered successfully, False otherwise.
    """
    if not webhook_url or not webhook_url.strip():
        logger.debug("Discord webhook URL is empty; skipping webhook delivery.")
        return False

    if not embeds:
        return True

    success = True

    for idx, embed in enumerate(embeds):
        if idx > 0 and delay_between_offers > 0:
            logger.info(
                f"Pausing {delay_between_offers}s before sending next offer alert to prevent flooding..."
            )
            time.sleep(delay_between_offers)

        if isinstance(username, list):
            bot_user = username[idx] if idx < len(username) else username[0]
        elif isinstance(username, str) and username.strip():
            bot_user = username
        else:
            bot_user = "Delta Cards Alert"

        payload: Dict[str, Any] = {
            "username": bot_user,
            "embeds": [embed],
        }
        if isinstance(content, list):
            item_msg = content[idx] if idx < len(content) else None
        else:
            item_msg = content

        if item_msg:
            payload["content"] = item_msg

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(webhook_url, json=payload)
                if response.status_code in (200, 204):
                    logger.info("Successfully sent Discord alert for offer.")
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

    def __init__(
        self,
        settings: Optional[Settings] = None,
        verifier: Optional[Any] = None,
        webhook_url: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.verifier = verifier
        self.webhook_url = webhook_url

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

        has_any_webhook = bool(
            self.webhook_url
            or self.settings.discord_webhook_url
            or self.settings.pokemon_cards_webhook
            or self.settings.pokemon_booster_webhook
            or self.settings.pokemon_booster_box_webhook
            or self.settings.magic_cards_webhook
            or self.settings.magic_sealed_webhook
        )
        if not has_any_webhook:
            logger.debug(
                "No Discord webhooks configured in settings. Skipping Discord alerts."
            )
            return 0

        matching_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        for offer in offers:
            ref_entry = None
            if reference_prices and offer.id in reference_prices:
                ref_entry = reference_prices[offer.id]
            elif reference_prices and offer.cardmarket_product_id in reference_prices:
                ref_entry = reference_prices[offer.cardmarket_product_id]

            avg_price = getattr(offer, "cardmarket_avg_price", None)
            low_price = getattr(offer, "cardmarket_low_price", None)
            trend_price = getattr(offer, "cardmarket_trend_price", None)

            if isinstance(ref_entry, (int, float)):
                market_price = float(ref_entry)
            elif isinstance(ref_entry, dict):
                market_price = float(
                    ref_entry.get("avg_price")
                    or ref_entry.get("low_price")
                    or ref_entry.get("trend_price")
                    or 0.0
                )
            elif avg_price is not None and avg_price > 0:
                market_price = float(avg_price)
            elif low_price is not None and low_price > 0:
                market_price = float(low_price)
            elif trend_price is not None and trend_price > 0:
                market_price = float(trend_price)
            else:
                market_price = 0.0

            cond = evaluate_alert_condition(
                offer_price=offer.price,
                market_price=market_price,
                settings=self.settings,
            )

            if cond.is_triggered:
                # LLM False-Positive Verification before dispatching alerts
                llm_res = None
                offer_is_magic = (getattr(offer, "game", None) == "magic") or is_magic_offer(offer)
                has_verification_data = bool(
                    self.verifier is not None
                    or getattr(offer, "_query_ocr", None)
                    or getattr(offer, "image_url", None)
                    or getattr(offer, "tcgdex_id", None)
                )
                if self.settings.llm_verifier_enabled and has_verification_data and not offer_is_magic:
                    try:
                        if self.verifier is None:
                            from ..recognition.llm_verifier import get_llm_verifier
                            self.verifier = get_llm_verifier(
                                model_path=self.settings.llm_verifier_model_path,
                                min_confidence=self.settings.llm_verifier_min_confidence,
                            )
                        llm_res = self.verifier.verify_match(offer)
                        if not llm_res.is_verified:
                            logger.info(
                                f"Alert suppressed for '{offer.title[:35]}' by LLM Verifier "
                                f"(confidence {llm_res.confidence_score}% < {self.settings.llm_verifier_min_confidence}%): "
                                f"{llm_res.verdict_summary}"
                            )
                            continue
                    except Exception as err:
                        logger.warning(f"LLM verification error: {err}. Proceeding with alert.")

                logger.info(
                    f"Deal detected for '{offer.title[:35]}': {cond.reason}"
                )
                embed = build_discord_embed(offer, cond, llm_verification=llm_res)
                matching_items.append((offer, embed, cond, llm_res))

        if not matching_items:
            logger.debug("No offers met the configured Discord alert conditions.")
            return 0

        # Group matching items by game and product category
        pokemon_card_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        pokemon_booster_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        pokemon_box_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        magic_card_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        magic_booster_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []
        magic_box_items: List[Tuple[OfferListing, Dict[str, Any], AlertConditionResult, Any]] = []

        for off, emb, cond, llm_res in matching_items:
            ch_key = resolve_channel_key(
                game=getattr(off, "game", None) or ("magic" if is_magic_offer(off) else "pokemon"),
                product_type=getattr(off, "product_type", None),
            )
            if (getattr(off, "game", None) == "magic") or is_magic_offer(off):
                if ch_key == "magic_boosters":
                    magic_booster_items.append((off, emb, cond, llm_res))
                elif ch_key in ("magic_boxes", "magic_sealed"):
                    magic_box_items.append((off, emb, cond, llm_res))
                else:
                    magic_card_items.append((off, emb, cond, llm_res))
            else:
                if ch_key == "pokemon_boosters":
                    pokemon_booster_items.append((off, emb, cond, llm_res))
                elif ch_key == "pokemon_boxes":
                    pokemon_box_items.append((off, emb, cond, llm_res))
                else:
                    pokemon_card_items.append((off, emb, cond, llm_res))

        templates_dir = Path(self.settings.alert_templates_dir) if hasattr(self.settings, "alert_templates_dir") else None
        total_sent = 0

        def _dispatch_category(items, wh_url, channel_key, label) -> int:
            if not items:
                return 0
            if not wh_url:
                logger.warning(f"No webhook configured for {label}. Skipping alerts.")
                return 0
            logger.info(f"Dispatching {len(items)} {label} alert(s)...")
            embeds = []
            content_list = []
            for off, _, c, llm in items:
                msg_text = render_alert_text(
                    offer=off,
                    result=c,
                    channel_key=channel_key,
                    llm_verification=llm,
                    emoji=self.settings.alert_wild_offer_emoji,
                    templates_dir=templates_dir,
                )
                header = extract_alert_header(msg_text)
                emb = build_discord_embed(
                    offer=off,
                    result=c,
                    llm_verification=llm,
                    description=msg_text,
                )
                embeds.append(emb)
                content_list.append(header)

            is_magic = "magic" in channel_key.lower()
            bot_username = self.settings.get_bot_username("magic" if is_magic else "pokemon")

            if send_discord_webhook(
                webhook_url=wh_url,
                embeds=embeds,
                content=content_list,
                username=bot_username,
                delay_between_offers=self.settings.time_between_offers,
            ):
                return len(items)
            return 0

        # 1. Pokémon Single Cards
        card_wh = (
            self.webhook_url
            or self.settings.pokemon_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(pokemon_card_items, card_wh, "pokemon_cards", "Pokémon Single Card")

        # 2. Pokémon Booster / Blister Packs
        booster_wh = (
            self.webhook_url
            or self.settings.pokemon_booster_webhook
            or self.settings.pokemon_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(pokemon_booster_items, booster_wh, "pokemon_boosters", "Pokémon Booster Pack")

        # 3. Pokémon Booster Boxes / Displays / ETBs
        box_wh = (
            self.webhook_url
            or self.settings.pokemon_booster_box_webhook
            or self.settings.pokemon_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(pokemon_box_items, box_wh, "pokemon_boxes", "Pokémon Booster Box")

        # 4. Magic Cards
        mag_card_wh = (
            self.webhook_url
            or self.settings.magic_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(magic_card_items, mag_card_wh, "magic_cards", "Magic Card")

        # 5. Magic Booster / Blister Packs
        mag_booster_wh = (
            self.webhook_url
            or self.settings.magic_booster_webhook
            or self.settings.magic_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(magic_booster_items, mag_booster_wh, "magic_boosters", "Magic Booster Pack")

        # 6. Magic Booster Boxes / Displays / Bundles
        mag_box_wh = (
            self.webhook_url
            or self.settings.magic_booster_box_webhook
            or self.settings.magic_sealed_webhook
            or self.settings.magic_cards_webhook
            or self.settings.discord_webhook_url
        )
        total_sent += _dispatch_category(magic_box_items, mag_box_wh, "magic_boxes", "Magic Booster Box")

        return total_sent


class AlertWorkerService:
    """Independent worker service that periodically dispatches 1 verified deal alert to Discord."""

    def __init__(
        self,
        engine: Optional[Any] = None,
        settings: Optional[Settings] = None,
        verifier: Optional[Any] = None,
        webhook_url: Optional[str] = None,
    ):
        from ..database.connection import get_engine
        self.engine = engine or get_engine()
        self.settings = settings or get_settings()
        self.verifier = verifier
        self.webhook_url = webhook_url

    def dispatch_one_pending_alert(self) -> Optional[Any]:
        """Finds and evaluates pending offers one by one until 1 is verified and sent to Discord.

        1. Retrieves offers with alert_status == 'pending'.
        2. For each offer:
           - Verifies pricing condition (Wallapop < Cardmarket).
           - Executes the LLM Verifier right before sending.
           - If verified: sends Discord alert webhook, marks alert_status = 'sent', and returns.
           - If rejected by LLM: marks alert_status = 'rejected' and advances to the next offer.
        3. Returns the PlatformOffers model that was alerted, or None if no alert was dispatched.
        """
        if not self.settings.discord_alerts_enabled:
            logger.info("Discord alerts are disabled in settings (DISCORD_ALERTS_ENABLED=false).")
            return None

        from ..database.models import PlatformOffers, CardmarketPrice, CardmarketProduct
        from sqlmodel import Session, select, and_

        with Session(self.engine) as session:
            stmt = (
                select(PlatformOffers, CardmarketPrice, CardmarketProduct)
                .outerjoin(
                    CardmarketPrice,
                    and_(
                        PlatformOffers.cardmarket_product_id == CardmarketPrice.cardmarket_product_id,
                        PlatformOffers.language == CardmarketPrice.language,
                    ),
                )
                .outerjoin(
                    CardmarketProduct,
                    PlatformOffers.cardmarket_product_id == CardmarketProduct.id,
                )
                .where(PlatformOffers.alert_status == "pending")
                .order_by(PlatformOffers.scraped_at.desc())
            )
            pending_candidates = session.exec(stmt).all()

        if not pending_candidates:
            logger.debug("No pending offers with alert_status='pending' found in database.")
            return None

        logger.info(f"Found {len(pending_candidates)} pending offer(s) to evaluate for dispatch.")

        for offer, price_row, product_row in pending_candidates:
            # Re-check market price
            avg_p = price_row.avg_price if price_row else None
            low_p = price_row.min_price_all if price_row else None
            ref_market_price = avg_p if avg_p is not None else low_p

            if ref_market_price is None or ref_market_price <= 0:
                logger.debug(f"Offer {offer.id} missing reference price. Setting alert_status='none'.")
                with Session(self.engine) as s:
                    db_o = s.exec(select(PlatformOffers).where(PlatformOffers.id == offer.id)).first()
                    if db_o:
                        db_o.alert_status = "none"
                        db_o.updated_at = datetime.now(timezone.utc)
                        s.add(db_o)
                        s.commit()
                continue

            cond = evaluate_alert_condition(
                offer_price=offer.price,
                market_price=ref_market_price,
                settings=self.settings,
            )

            if not cond.is_triggered:
                logger.debug(f"Offer {offer.id} ('{offer.title[:30]}') does not trigger alert. Setting alert_status='none'.")
                with Session(self.engine) as s:
                    db_o = s.exec(select(PlatformOffers).where(PlatformOffers.id == offer.id)).first()
                    if db_o:
                        db_o.alert_status = "none"
                        db_o.updated_at = datetime.now(timezone.utc)
                        s.add(db_o)
                        s.commit()
                continue

            # Convert to OfferListing for embed and LLM verification
            game_category = "Magic" if is_magic_offer(offer) else "Pokemon"
            cm_url = (
                product_row.cardmarket_url if product_row
                else (f"https://www.cardmarket.com/{offer.language or 'es'}/{game_category}/Products?idProduct={offer.cardmarket_product_id}" if offer.cardmarket_product_id else None)
            )
            offer_listing = OfferListing(
                id=offer.id,
                title=offer.title,
                price=offer.price,
                currency=offer.currency,
                product_type=offer.product_type,
                location=offer.location,
                country=offer.country,
                description=offer.description,
                link=offer.link,
                image_url=offer.image_url,
                is_reserved=offer.is_reserved,
                shipping_available=offer.shipping_available,
                seller_user_id=offer.seller_user_id,
                seller_link=offer.seller_link,
                published_date=offer.published_date,
                search_profile_id=offer.search_profile_id,
                cardmarket_product_id=offer.cardmarket_product_id,
                cardmarket_url=cm_url,
                language=offer.language,
                cardmarket_low_price=low_p,
                cardmarket_avg_price=avg_p,
                cardmarket_trend_price=price_row.trend_price if price_row else None,
                cardmarket_price_spain=price_row.min_price_spain if price_row else None,
                cardmarket_price_all=ref_market_price,
                tcgdex_id=offer.tcgdex_id,
                assigned_state=offer.assigned_state or "assigned",
                platform=getattr(offer, "platform", "wallapop") or "wallapop",
                game=(getattr(offer, "game", None) or ("magic" if is_magic_offer(offer) else "pokemon")).lower(),
                alert_status="pending",
            )

            offer_game = offer_listing.game
            is_magic = (offer_game == "magic")

            # LLM False-Positive Verification exclusively executed here before dispatch (for Pokémon single cards)
            llm_res = None
            if (
                self.settings.llm_verifier_enabled
                and not is_magic
                and is_card_offer(offer.product_type, title=offer.title)
            ):
                try:
                    if self.verifier is None:
                        from ..recognition.llm_verifier import get_llm_verifier
                        self.verifier = get_llm_verifier(
                            model_path=self.settings.llm_verifier_model_path,
                            min_confidence=self.settings.llm_verifier_min_confidence,
                        )
                    llm_res = self.verifier.verify_match(offer_listing)
                    if not llm_res.is_verified:
                        logger.info(
                            f"Offer '{offer.title[:35]}' rejected by LLM Verifier "
                            f"(score {llm_res.confidence_score}% < {self.settings.llm_verifier_min_confidence}%): "
                            f"{llm_res.verdict_summary}. Marking as 'rejected' and trying next offer."
                        )
                        with Session(self.engine) as s:
                            db_o = s.exec(select(PlatformOffers).where(PlatformOffers.id == offer.id)).first()
                            if db_o:
                                db_o.alert_status = "rejected"
                                db_o.updated_at = datetime.now(timezone.utc)
                                s.add(db_o)
                                s.commit()
                        continue
                except Exception as err:
                    logger.warning(f"LLM verification error: {err}. Proceeding with alert.")

            # Render alert text and construct Discord Embed (inside the green block)
            ch_key = resolve_channel_key(
                game=getattr(offer_listing, "game", None) or ("magic" if is_magic else "pokemon"),
                product_type=getattr(offer_listing, "product_type", None),
            )
            templates_dir = Path(self.settings.alert_templates_dir) if hasattr(self.settings, "alert_templates_dir") else None
            offer_msg = render_alert_text(
                offer=offer_listing,
                result=cond,
                channel_key=ch_key,
                llm_verification=llm_res,
                emoji=self.settings.alert_wild_offer_emoji,
                templates_dir=templates_dir,
            )
            embed = build_discord_embed(
                offer_listing,
                cond,
                llm_verification=llm_res,
                description=offer_msg,
            )

            webhook_url = self.webhook_url
            if not webhook_url:
                if is_magic:
                    webhook_url = self.settings.get_webhook_url("magic", offer.product_type)
                else:
                    webhook_url = self.settings.get_webhook_url("pokemon", offer.product_type)

            if not webhook_url:
                logger.debug(
                    f"No Discord webhook configured for offer '{offer.title[:30]}' ({offer.product_type}). Skipping alert."
                )
                return None

            header = extract_alert_header(offer_msg)
            bot_username = self.settings.get_bot_username(offer_game)
            ok = send_discord_webhook(
                webhook_url=webhook_url,
                embeds=[embed],
                content=header,
                username=bot_username,
                delay_between_offers=self.settings.time_between_offers,
            )

            if ok:
                with Session(self.engine) as s:
                    db_o = s.exec(select(PlatformOffers).where(PlatformOffers.id == offer.id)).first()
                    if db_o:
                        db_o.alert_status = "sent"
                        db_o.updated_at = datetime.now(timezone.utc)
                        s.add(db_o)
                        s.commit()
                logger.info(f"Successfully dispatched 1 alert for offer '{offer.title[:35]}' (ID: {offer.id}).")
                return offer
            else:
                logger.warning(f"Failed to dispatch webhook for offer {offer.id}. Trying next candidate.")

        logger.info("Checked all pending offers in this cycle; none were dispatched.")
        return None

    def run_loop(
        self,
        interval_seconds: Optional[int] = None,
        once: bool = False,
    ) -> None:
        """Runs the alert dispatcher loop periodically (e.g. every 30 minutes)."""
        import time

        interval = interval_seconds if interval_seconds is not None else self.settings.alert_interval
        logger.info(f"Starting AlertWorkerService loop (interval: {interval}s, once: {once}).")

        while True:
            try:
                sent_offer = self.dispatch_one_pending_alert()
                if sent_offer:
                    logger.info(f"Dispatched deal: {sent_offer.title[:40]} ({sent_offer.price:.2f}€)")
                    if once:
                        break
                    # If an offer was dispatched, check for the next pending deal after delay
                    delay = self.settings.time_between_offers
                    if delay > 0:
                        logger.info(f"Waiting {delay}s before checking next pending deal...")
                        time.sleep(delay)
                    continue
                else:
                    logger.info("No pending deal dispatched in this cycle.")
            except Exception as exc:
                logger.error(f"Error in AlertWorkerService cycle: {exc}", exc_info=True)

            if once:
                logger.info("Exiting AlertWorkerService (--once specified).")
                break

            # When queue is empty, poll every 30s (or custom interval if specified on CLI)
            poll_interval = interval_seconds if interval_seconds is not None else min(interval, 30)
            logger.info(f"No pending deals. Alert worker sleeping for {poll_interval}s before next check (Ctrl+C to stop)...")
            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                logger.info("Alert worker stopped by user (KeyboardInterrupt).")
                break
