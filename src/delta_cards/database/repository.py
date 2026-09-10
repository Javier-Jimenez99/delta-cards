"""Database repository functions for saving and querying Wallapop offers."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..models import OfferListing
from .connection import get_engine, get_session
from .models import WallapopOffers

logger = logging.getLogger(__name__)


def offer_to_db_model(offer: OfferListing, now: Optional[datetime] = None) -> WallapopOffers:
    """Converts an OfferListing into a WallapopOffers database model instance."""
    now_utc = now or datetime.now(timezone.utc)
    data = offer.model_dump()
    data["scraped_at"] = now_utc
    data["updated_at"] = now_utc
    return WallapopOffers(**data)


def save_offers_to_db(
    offers: List[OfferListing],
    db_url: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> int:
    """Upserts a list of scraped OfferListing objects into the WallapopOffers table.

    Handles duplicates by updating existing records with refreshed price, reserved status,
    and updated timestamp.

    Args:
        offers: List of OfferListing instances to persist.
        db_url: Optional database connection URL override.
        engine: Optional pre-configured SQLAlchemy engine.

    Returns:
        int: Number of offers saved/updated.
    """
    if not offers:
        return 0

    eng = engine or get_engine(db_url)
    now = datetime.now(timezone.utc)
    dialect_name = eng.dialect.name

    with Session(eng) as session:
        if dialect_name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            records = [offer_to_db_model(o, now=now).model_dump() for o in offers]
            stmt = pg_insert(WallapopOffers).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=[WallapopOffers.id],
                set_={
                    "title": stmt.excluded.title,
                    "price": stmt.excluded.price,
                    "is_reserved": stmt.excluded.is_reserved,
                    "shipping_available": stmt.excluded.shipping_available,
                    "description": stmt.excluded.description,
                    "location": stmt.excluded.location,
                    "image_url": stmt.excluded.image_url,
                    "seller_link": stmt.excluded.seller_link,
                    "seller_user_id": stmt.excluded.seller_user_id,
                    "cardmarket_product_id": stmt.excluded.cardmarket_product_id,
                    "tcgdex_id": stmt.excluded.tcgdex_id,
                    "updated_at": now,
                },
            )
            session.exec(stmt)
            session.commit()
            return len(records)

        elif dialect_name == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert

            records = [offer_to_db_model(o, now=now).model_dump() for o in offers]
            stmt = sqlite_insert(WallapopOffers).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=[WallapopOffers.id],
                set_={
                    "title": stmt.excluded.title,
                    "price": stmt.excluded.price,
                    "is_reserved": stmt.excluded.is_reserved,
                    "shipping_available": stmt.excluded.shipping_available,
                    "description": stmt.excluded.description,
                    "location": stmt.excluded.location,
                    "image_url": stmt.excluded.image_url,
                    "seller_link": stmt.excluded.seller_link,
                    "seller_user_id": stmt.excluded.seller_user_id,
                    "cardmarket_product_id": stmt.excluded.cardmarket_product_id,
                    "tcgdex_id": stmt.excluded.tcgdex_id,
                    "updated_at": now,
                },
            )
            session.exec(stmt)
            session.commit()
            return len(records)

        else:
            # Fallback using session.merge for any other dialect
            for offer in offers:
                item = offer_to_db_model(offer, now=now)
                session.merge(item)
            session.commit()
            return len(offers)


def get_all_offers(engine: Optional[Engine] = None) -> List[WallapopOffers]:
    """Retrieves all offers currently stored in the database."""
    eng = engine or get_engine()
    with Session(eng) as session:
        statement = select(WallapopOffers)
        results = session.exec(statement).all()
        return list(results)


def update_offer_tcgdex(
    offer_id: str,
    tcgdex_id: str,
    cardmarket_product_id: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> bool:
    """Updates the tcgdex_id and cardmarket_product_id for an existing offer in the database.

    Args:
        offer_id: ID of the Wallapop offer.
        tcgdex_id: TCGDex identifier.
        cardmarket_product_id: Cardmarket product ID.
        engine: Optional SQLAlchemy engine.

    Returns:
        bool: True if offer was found and updated, False otherwise.
    """
    eng = engine or get_engine()
    now = datetime.now(timezone.utc)
    with Session(eng) as session:
        statement = select(WallapopOffers).where(WallapopOffers.id == offer_id)
        offer = session.exec(statement).first()
        if not offer:
            return False
        offer.tcgdex_id = tcgdex_id
        offer.cardmarket_product_id = cardmarket_product_id
        offer.updated_at = now
        session.add(offer)
        session.commit()
        return True


def get_unassigned_card_offers(
    engine: Optional[Engine] = None,
    limit: Optional[int] = None,
) -> List[WallapopOffers]:
    """Retrieves card offers (product_type == 'single_card') that do not have a TCGDex ID."""
    eng = engine or get_engine()
    with Session(eng) as session:
        statement = select(WallapopOffers).where(
            WallapopOffers.product_type == "single_card",
            WallapopOffers.tcgdex_id.is_(None),  # type: ignore
        )
        if limit is not None:
            statement = statement.limit(limit)
        return list(session.exec(statement).all())

