"""Discord alerting service and condition evaluation for card deals."""

import logging
import time
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


def build_discord_embed(
    offer: OfferListing,
    result: AlertConditionResult,
    llm_verification: Optional[Any] = None,
) -> Dict[str, Any]:
    """Formats a deal offer into a Discord embed object.

    Structure:
    Deal: [title]
    Wallapop price - Market Price of average seller - Market price of low price seller
    Type (Carta, Paquete, Caja, etc.) - Location
    Links to wallapop offer and to cardmarket product page.
    """
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
        cardmarket_url = f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={offer.cardmarket_product_id}"

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


def send_discord_webhook(
    webhook_url: str,
    embeds: List[Dict[str, Any]],
    content: Optional[str] = DEFAULT_OFFER_ALERT_MESSAGE,
    timeout: float = 10.0,
    delay_between_offers: float = 0.0,
) -> bool:
    """Sends each offer embed as a separate notification message to Discord with alert text.

    Args:
        webhook_url: The Discord webhook endpoint URL.
        embeds: List of embed dictionaries to send.
        content: Message text displayed before each offer embed.
        timeout: HTTP request timeout in seconds.
        delay_between_offers: Delay in seconds to pause between consecutive offer alerts (default: 0.0).

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

        payload: Dict[str, Any] = {
            "username": "Delta Cards Alert",
            "embeds": [embed],
        }
        if content:
            payload["content"] = content

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

    def __init__(self, settings: Optional[Settings] = None, verifier: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.verifier = verifier

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
            ref_entry = None
            if reference_prices and offer.id in reference_prices:
                ref_entry = reference_prices[offer.id]
            elif reference_prices and offer.cardmarket_product_id in reference_prices:
                ref_entry = reference_prices[offer.cardmarket_product_id]

            avg_price = getattr(offer, "cardmarket_avg_price", None)
            low_price = getattr(offer, "cardmarket_low_price", None)

            if isinstance(ref_entry, dict):
                if avg_price is None:
                    avg_price = ref_entry.get("avg") or ref_entry.get("all")
                if low_price is None:
                    low_price = ref_entry.get("low") or ref_entry.get("all")
            elif isinstance(ref_entry, (int, float)):
                if avg_price is None:
                    avg_price = float(ref_entry)

            if avg_price is None:
                avg_price = low_price

            try:
                offer.cardmarket_avg_price = avg_price
                if low_price is not None:
                    offer.cardmarket_low_price = low_price
            except Exception:
                try:
                    object.__setattr__(offer, "cardmarket_avg_price", avg_price)
                    if low_price is not None:
                        object.__setattr__(offer, "cardmarket_low_price", low_price)
                except Exception:
                    pass

            market_price = avg_price if avg_price is not None else low_price
            if market_price is None:
                continue

            cond = evaluate_alert_condition(
                offer_price=offer.price,
                market_price=market_price,
                settings=self.settings,
            )

            if cond.is_triggered:
                # LLM False-Positive Verification before dispatching alerts
                llm_res = None
                has_verification_data = bool(
                    self.verifier is not None
                    or getattr(offer, "_query_ocr", None)
                    or getattr(offer, "image_url", None)
                    or getattr(offer, "tcgdex_id", None)
                )
                if self.settings.llm_verifier_enabled and has_verification_data:
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
                matching_embeds.append(embed)

        if not matching_embeds:
            logger.debug("No offers met the configured Discord alert conditions.")
            return 0

        logger.info(f"Dispatching {len(matching_embeds)} Discord alert(s)...")
        emoji = (self.settings.alert_wild_offer_emoji or "").strip()
        offer_msg = f"{emoji} ¡Ha aparecido una oferta salvaje!" if emoji else DEFAULT_OFFER_ALERT_MESSAGE
        ok = send_discord_webhook(
            webhook_url=webhook_url,
            embeds=matching_embeds,
            content=offer_msg,
            delay_between_offers=self.settings.time_between_offers,
        )
        return len(matching_embeds) if ok else 0


class AlertWorkerService:
    """Independent worker service that periodically dispatches 1 verified deal alert to Discord."""

    def __init__(
        self,
        engine: Optional[Any] = None,
        settings: Optional[Settings] = None,
        verifier: Optional[Any] = None,
    ):
        from ..database.connection import get_engine
        self.engine = engine or get_engine()
        self.settings = settings or get_settings()
        self.verifier = verifier

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

        webhook_url = self.settings.discord_webhook_url
        if not webhook_url:
            logger.debug("No DISCORD_WEBHOOK_URL configured in settings. Skipping Discord alerts.")
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
            cm_url = (
                product_row.cardmarket_url if product_row
                else (f"https://www.cardmarket.com/{offer.language or 'es'}/Pokemon/Products?idProduct={offer.cardmarket_product_id}" if offer.cardmarket_product_id else None)
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
                alert_status="pending",
            )

            # LLM False-Positive Verification exclusively executed here before dispatch
            llm_res = None
            if self.settings.llm_verifier_enabled:
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

            # Construct Discord Embed and Dispatch
            embed = build_discord_embed(offer_listing, cond, llm_verification=llm_res)
            emoji = (self.settings.alert_wild_offer_emoji or "").strip()
            offer_msg = f"{emoji} ¡Ha aparecido una oferta salvaje!" if emoji else DEFAULT_OFFER_ALERT_MESSAGE
            ok = send_discord_webhook(
                webhook_url=webhook_url,
                embeds=[embed],
                content=offer_msg,
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
