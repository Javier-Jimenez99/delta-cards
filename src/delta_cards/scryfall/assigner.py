"""Offer assignment service to enrich Magic: The Gathering card offers using Scryfall API."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..database.connection import get_engine
from ..database.models import PlatformOffers
from ..database.repository import upsert_cardmarket_product, upsert_cardmarket_price
from ..models import OfferListing, is_card_offer
from .client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
    _default_client,
)

logger = logging.getLogger(__name__)

# Regex pattern to match set code (3-4 alphanumeric) and collector number (e.g. MH3 138, LTR #001)
SET_NUMBER_PATTERN = re.compile(
    r"\b([a-zA-Z0-9]{3,5})\s*(?:#|nº|no\.?|num\.?)?\s*(\d{1,4}[a-zA-Z]?)\b",
    re.IGNORECASE,
)

# Strip out boilerplate marketplace terms from Magic titles to extract the card name
CLEANUP_TITLE_TERMS = [
    r"\bcarta\s+magic\s+the\s+gathering\b",
    r"\bcartas\s+magic\s+the\s+gathering\b",
    r"\bmagic\s+the\s+gathering\b",
    r"\bcarta\s+magic\b",
    r"\bcartas\s+magic\b",
    r"\bmagic\s+carta\b",
    r"\bmtg\s+carta\b",
    r"\bcarta\s+mtg\b",
    r"\bmtg\s+single\b",
    r"\bmagic\b",
    r"\bmtg\b",
    r"\bthe\s+gathering\b",
    r"\bfoil\b",
    r"\bholo\b",
    r"\betched\b",
    r"\btextured\b",
    r"\bshowcase\b",
    r"\bborderless\b",
    r"\bextended\s+art\b",
    r"\bretro\s+frame\b",
    r"\bretro\b",
    r"\bsecret\s+lair\b",
    r"\bplaneswalker\b",
    r"\bcommander\b",
    r"\bedh\b",
    r"\bmodern\b",
    r"\blegacy\b",
    r"\bstandard\b",
    r"\bvintage\b",
    r"\bnear\s+mint\b",
    r"\bnm\b",
    r"\bmint\b",
    r"\bex\b",
    r"\bexcellent\b",
    r"\bgd\b",
    r"\blp\b",
    r"\bspanish\b",
    r"\bespañol\b",
    r"\bingles\b",
    r"\benglish\b",
    r"\ben\s+castellano\b",
    r"\ben\s+ingles\b",
    r"\bprecintada?\b",
    r"\bnueva?\b",
    r"\boriginal\b",
    r"\bvendo\b",
    r"\bcompro\b",
    r"\bperfecto\s+estado\b",
]


def extract_magic_identifiers(title: str, description: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extracts candidate set code, collector number, and clean card name from listing text.

    Returns:
        Tuple of (set_code, collector_number, candidate_card_name)
    """
    combined = f"{title} {description or ''}".strip()

    set_code: Optional[str] = None
    collector_number: Optional[str] = None

    # Check for set code + number pattern
    match = SET_NUMBER_PATTERN.search(title)
    if not match and description:
        match = SET_NUMBER_PATTERN.search(description)

    if match:
        set_code = match.group(1).lower()
        collector_number = match.group(2).lower()

    # Extract clean card name from title
    cleaned = title
    for pat in CLEANUP_TITLE_TERMS:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # Remove extra punctuation and multiple spaces
    cleaned = re.sub(r"[#\(\)\[\]\{\}\/\\\"\'\|•·\-_]", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()

    candidate_name = cleaned if len(cleaned) >= 2 else None
    return set_code, collector_number, candidate_name


def resolve_scryfall_card(
    title: str,
    description: Optional[str] = None,
    client: Optional[ScryfallClient] = None,
    language: str = "es",
) -> Optional[Dict[str, Any]]:
    """Resolves a marketplace listing to a Scryfall card and Cardmarket pricing."""
    cli = client or _default_client
    set_code, collector_num, candidate_name = extract_magic_identifiers(title, description)

    card_data: Optional[Dict[str, Any]] = None

    # Strategy 1: Look up by set code and collector number
    if set_code and collector_num:
        try:
            card_data = cli.get_card_by_set_number(set_code, collector_num, lang=language)
            if not card_data:
                card_data = cli.get_card_by_set_number(set_code, collector_num)
        except Exception as err:
            logger.debug(f"Error checking Scryfall set/number ({set_code}/{collector_num}): {err}")

    # Strategy 2: Fuzzy name search using cleaned card name
    if not card_data and candidate_name:
        try:
            card_data = cli.get_card_by_name(candidate_name, set_code=set_code)
        except Exception as err:
            logger.debug(f"Error checking Scryfall name '{candidate_name}': {err}")

    # Strategy 3: Fallback query search if candidate has multiple tokens
    if not card_data and candidate_name and len(candidate_name.split()) >= 2:
        try:
            results = cli.search_cards(candidate_name)
            if results:
                card_data = results[0]
        except Exception as err:
            logger.debug(f"Error in Scryfall search fallback for '{candidate_name}': {err}")

    if not card_data:
        return None

    return extract_scryfall_cardmarket_prices(card_data, language=language)


def _apply_scryfall_result_to_offer(offer: Any, pricing_info: Dict[str, Any]) -> None:
    """Applies Scryfall pricing and metadata to an OfferListing or PlatformOffers model."""
    if not pricing_info:
        return

    cm_id = pricing_info.get("idProduct")
    cm_url = pricing_info.get("url")
    prices = pricing_info.get("prices") or {}

    raw = pricing_info.get("raw") or {}
    scryfall_id = raw.get("id")

    if scryfall_id:
        if hasattr(offer, "tcgdex_id"):
            offer.tcgdex_id = scryfall_id
        if hasattr(offer, "scryfall_id"):
            offer.scryfall_id = scryfall_id

    if cm_id:
        if hasattr(offer, "cardmarket_product_id"):
            offer.cardmarket_product_id = cm_id
    if cm_url:
        if hasattr(offer, "cardmarket_url"):
            offer.cardmarket_url = cm_url
        object.__setattr__(offer, "_temp_cm_url", cm_url)

    low = prices.get("low")
    avg = prices.get("avg")
    trend = prices.get("trend")

    if low is not None:
        if hasattr(offer, "cardmarket_low_price"):
            offer.cardmarket_low_price = low
        object.__setattr__(offer, "_temp_cm_low_price", low)
    if avg is not None:
        if hasattr(offer, "cardmarket_avg_price"):
            offer.cardmarket_avg_price = avg
        object.__setattr__(offer, "_temp_cm_avg_price", avg)
    if trend is not None:
        if hasattr(offer, "cardmarket_trend_price"):
            offer.cardmarket_trend_price = trend
        object.__setattr__(offer, "_temp_cm_trend_price", trend)

    if pricing_info.get("image_url") and not getattr(offer, "image_url", None):
        offer.image_url = pricing_info["image_url"]


def assign_scryfall_to_offers(
    offers: List[OfferListing],
    client: Optional[ScryfallClient] = None,
    progress_callback: Optional[Any] = None,
) -> List[OfferListing]:
    """Enriches a list of OfferListing objects with Scryfall ID and Cardmarket Product ID."""
    if not offers:
        return []

    cli = client or _default_client
    card_offers = [
        o for o in offers
        if is_card_offer(o.product_type, title=o.title, description=o.description)
        and not getattr(o, "cardmarket_product_id", None)
    ]

    total = len(card_offers)
    for idx, offer in enumerate(card_offers, 1):
        try:
            res = resolve_scryfall_card(
                title=offer.title,
                description=offer.description,
                client=cli,
                language=offer.language or "es",
            )
            if res:
                _apply_scryfall_result_to_offer(offer, res)
        except Exception as exc:
            logger.debug(f"Failed to assign Scryfall for offer '{offer.title}': {exc}")

        if progress_callback:
            progress_callback(idx, total, offer)

    return offers


def enrich_magic_offers_in_db(
    engine: Optional[Engine] = None,
    limit: Optional[int] = None,
    client: Optional[ScryfallClient] = None,
) -> int:
    """Enriches existing Magic card offers stored in the PlatformOffers database table

    Finds listings with assigned_state == 'pending', matches them with Scryfall,
    saves CardmarketProduct and CardmarketPrice, and marks deal candidates.
    """
    eng = engine or get_engine()
    cli = client or _default_client

    with Session(eng) as session:
        statement = select(PlatformOffers).where(
            PlatformOffers.assigned_state == "pending"
        )
        if limit is not None:
            statement = statement.limit(limit)

        offers_to_enrich = session.exec(statement).all()
        if not offers_to_enrich:
            logger.info("No unassigned Magic card offers found in database.")
            return 0

        valid_offers: List[PlatformOffers] = []
        for db_o in offers_to_enrich:
            if is_card_offer(db_o.product_type, title=db_o.title, description=db_o.description):
                valid_offers.append(db_o)
            else:
                db_o.assigned_state = "not_assigned"
                session.add(db_o)

        if not valid_offers:
            session.commit()
            return 0

        updated_count = 0
        for db_o in valid_offers:
            try:
                res = resolve_scryfall_card(
                    title=db_o.title,
                    description=db_o.description,
                    client=cli,
                    language=db_o.language or "es",
                )
                if res and res.get("idProduct"):
                    _apply_scryfall_result_to_offer(db_o, res)
                    cm_id = res["idProduct"]
                    cm_url = res.get("url") or build_magic_cardmarket_url(cm_id, language=db_o.language or "es")

                    upsert_cardmarket_product(
                        product_id=cm_id,
                        cardmarket_url=cm_url,
                        session=session,
                    )

                    prices = res.get("prices") or {}
                    low_p = prices.get("low")
                    avg_p = prices.get("avg")
                    trend_p = prices.get("trend")

                    if low_p is not None or avg_p is not None or trend_p is not None:
                        upsert_cardmarket_price(
                            product_id=cm_id,
                            language=db_o.language or "es",
                            min_price_all=low_p,
                            avg_price=avg_p,
                            trend_price=trend_p,
                            session=session,
                        )

                    ref_price = avg_p if avg_p is not None else low_p
                    if ref_price is not None and ref_price > 0 and db_o.price < ref_price:
                        if getattr(db_o, "alert_status", None) != "sent":
                            db_o.alert_status = "pending"
                    else:
                        if getattr(db_o, "alert_status", None) != "sent":
                            db_o.alert_status = "none"

                    db_o.assigned_state = "assigned"
                    updated_count += 1
                else:
                    db_o.assigned_state = "not_assigned"
                    if getattr(db_o, "alert_status", None) != "sent":
                        db_o.alert_status = "none"

                session.add(db_o)
            except Exception as exc:
                logger.warning(f"Error enriching offer {db_o.id} ({db_o.title[:30]}): {exc}")

        session.commit()
        return updated_count
