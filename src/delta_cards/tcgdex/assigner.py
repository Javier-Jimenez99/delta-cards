"""Offer assignment service to enrich card offers with TCGDex and Cardmarket IDs."""

import logging
from typing import List, Optional
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..database.connection import get_engine
from ..database.models import WallapopOffers
from ..models import OfferListing
from .matcher import MatchResult, TCGDexMatcher

logger = logging.getLogger(__name__)

# Products that MUST NOT have TCGDex IDs assigned
EXCLUDED_PRODUCT_TYPES = {
    "booster_box",
    "booster_pack",
    "deck",
    "etb",
    "elite_trainer_box",
}


def is_card_offer(product_type: Optional[str]) -> bool:
    """Verifies whether an offer represents a single card listing.

    Strictly excludes booster boxes, booster packs, displays, full decks, and ETBs.
    """
    if not product_type:
        return False
    pt = product_type.lower().strip()
    if pt in EXCLUDED_PRODUCT_TYPES:
        return False
    if pt in ("single_card", "card"):
        return True
    # If product_type contains hints of sealed packs or boxes, exclude it
    if any(sealed_term in pt for sealed_term in ["box", "pack", "deck", "etb", "sobre", "caja", "display"]):
        return False
    return False


def assign_tcgdex_to_offers(
    offers: List[OfferListing],
    matcher: Optional[TCGDexMatcher] = None,
) -> List[OfferListing]:
    """Enriches a list of OfferListing objects with TCGDex ID and Cardmarket Product ID.

    Applies ONLY to card offers (`single_card`). Booster boxes, booster packs, etc.,
    are strictly skipped.

    Args:
        offers: List of scraped OfferListing instances.
        matcher: Optional pre-configured TCGDexMatcher instance.

    Returns:
        List[OfferListing]: Enriched offers (modified in-place and returned).
    """
    if not offers:
        return []

    card_matcher = matcher or TCGDexMatcher()

    for offer in offers:
        # Check product type rule
        if not is_card_offer(offer.product_type):
            continue

        # If already populated (e.g. manually set or previous pass), keep it
        if offer.tcgdex_id and offer.cardmarket_product_id:
            continue

        try:
            match_res = card_matcher.match(
                title=offer.title,
                description=offer.description,
            )
            if match_res.tcgdex_id:
                offer.tcgdex_id = match_res.tcgdex_id
                offer.cardmarket_product_id = match_res.cardmarket_product_id
                logger.debug(
                    f"Assigned '{offer.title[:30]}' -> TCGDex: {offer.tcgdex_id}, "
                    f"Cardmarket: {offer.cardmarket_product_id} (Score: {match_res.confidence_score:.1f})"
                )
        except Exception as exc:
            logger.warning(f"Error matching offer '{offer.id}' ({offer.title[:30]}): {exc}")

    return offers


def enrich_card_offers_in_db(
    engine: Optional[Engine] = None,
    matcher: Optional[TCGDexMatcher] = None,
    limit: Optional[int] = None,
) -> int:
    """Enriches existing card offers stored in the WallapopOffers database table

    that currently lack a TCGDex ID.

    Args:
        engine: Optional SQLAlchemy Engine.
        matcher: Optional TCGDexMatcher instance.
        limit: Optional maximum number of rows to process.

    Returns:
        int: Number of records successfully matched and updated.
    """
    eng = engine or get_engine()
    card_matcher = matcher or TCGDexMatcher()
    updated_count = 0

    with Session(eng) as session:
        # Query single_card records with null tcgdex_id
        statement = select(WallapopOffers).where(
            WallapopOffers.product_type == "single_card",
            WallapopOffers.tcgdex_id.is_(None),  # type: ignore
        )
        if limit is not None:
            statement = statement.limit(limit)

        offers_to_enrich = session.exec(statement).all()
        if not offers_to_enrich:
            logger.info("No unassigned card offers found in database.")
            return 0

        for db_offer in offers_to_enrich:
            if not is_card_offer(db_offer.product_type):
                continue

            try:
                res = card_matcher.match(
                    title=db_offer.title,
                    description=db_offer.description,
                )
                if res.tcgdex_id:
                    db_offer.tcgdex_id = res.tcgdex_id
                    db_offer.cardmarket_product_id = res.cardmarket_product_id
                    session.add(db_offer)
                    updated_count += 1
            except Exception as exc:
                logger.warning(f"Error enriching DB offer '{db_offer.id}': {exc}")

        session.commit()

    return updated_count
