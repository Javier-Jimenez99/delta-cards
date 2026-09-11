"""Database repository functions for saving and querying Wallapop offers, Cardmarket products, and prices."""

import logging
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import func
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..models import OfferListing
from .connection import get_engine, get_session
from .models import WallapopOffers, CardmarketProduct, CardmarketPrice

logger = logging.getLogger(__name__)


def offer_to_db_model(offer: OfferListing, now: Optional[datetime] = None) -> WallapopOffers:
    """Converts an OfferListing into a WallapopOffers database model instance."""
    now_utc = now or datetime.now(timezone.utc)
    valid_fields = set(WallapopOffers.model_fields.keys())
    data = {k: v for k, v in offer.model_dump().items() if k in valid_fields}
    data["scraped_at"] = now_utc
    data["updated_at"] = now_utc
    return WallapopOffers(**data)


def upsert_cardmarket_product(
    product_id: str,
    cardmarket_url: str,
    engine: Optional[Engine] = None,
    session: Optional[Session] = None,
) -> CardmarketProduct:
    """Inserts or updates a CardmarketProduct entity in the database."""
    pid = str(product_id)
    now = datetime.now(timezone.utc)

    def _do_upsert(s: Session) -> CardmarketProduct:
        prod = s.exec(select(CardmarketProduct).where(CardmarketProduct.id == pid)).first()
        if not prod:
            prod = CardmarketProduct(
                id=pid,
                cardmarket_url=cardmarket_url,
                scraped_at=now,
                updated_at=now,
            )
            s.add(prod)
        else:
            prod.cardmarket_url = cardmarket_url
            prod.updated_at = now
            s.add(prod)
        return prod

    if session is not None:
        return _do_upsert(session)

    eng = engine or get_engine()
    with Session(eng) as s:
        prod = _do_upsert(s)
        s.commit()
        s.refresh(prod)
        return prod


def upsert_cardmarket_price(
    product_id: str,
    language: str,
    min_price_all: Optional[float] = None,
    min_price_spain: Optional[float] = None,
    engine: Optional[Engine] = None,
    session: Optional[Session] = None,
) -> CardmarketPrice:
    """Inserts or updates a CardmarketPrice entry for a specific product and language."""
    pid = str(product_id)
    now = datetime.now(timezone.utc)

    def _do_upsert(s: Session) -> CardmarketPrice:
        # Guarantee parent CardmarketProduct exists
        prod = s.exec(select(CardmarketProduct).where(CardmarketProduct.id == pid)).first()
        if not prod:
            prod = CardmarketProduct(
                id=pid,
                cardmarket_url=f"https://www.cardmarket.com/{language}/Pokemon/Products?idProduct={pid}",
                scraped_at=now,
                updated_at=now,
            )
            s.add(prod)
            s.flush()

        price_rec = s.exec(
            select(CardmarketPrice).where(
                CardmarketPrice.cardmarket_product_id == pid,
                CardmarketPrice.language == language,
            )
        ).first()
        if not price_rec:
            price_rec = CardmarketPrice(
                cardmarket_product_id=pid,
                language=language,
                min_price_all=min_price_all,
                min_price_spain=min_price_spain,
                scraped_at=now,
                updated_at=now,
            )
            s.add(price_rec)
        else:
            if min_price_all is not None:
                price_rec.min_price_all = min_price_all
            if min_price_spain is not None:
                price_rec.min_price_spain = min_price_spain
            price_rec.updated_at = now
            s.add(price_rec)
        return price_rec

    if session is not None:
        return _do_upsert(session)

    eng = engine or get_engine()
    with Session(eng) as s:
        rec = _do_upsert(s)
        s.commit()
        s.refresh(rec)
        return rec


def save_offers_to_db(
    offers: List[OfferListing],
    db_url: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> int:
    """Upserts a list of scraped OfferListing objects into the WallapopOffers table.

    Also ensures associated CardmarketProduct entities are recorded.

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
        # Ensure referenced CardmarketProduct records exist
        for o in offers:
            if o.cardmarket_product_id:
                url = (
                    o.cardmarket_url
                    or f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={o.cardmarket_product_id}"
                )
                upsert_cardmarket_product(
                    product_id=o.cardmarket_product_id,
                    cardmarket_url=url,
                    session=session,
                )
                if getattr(o, "cardmarket_low_price", None) is not None:
                    upsert_cardmarket_price(
                        product_id=o.cardmarket_product_id,
                        language=o.language or "es",
                        min_price_all=o.cardmarket_low_price,
                        session=session,
                    )
        session.flush()

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
                    "cardmarket_product_id": func.coalesce(stmt.excluded.cardmarket_product_id, WallapopOffers.cardmarket_product_id),
                    "language": func.coalesce(stmt.excluded.language, WallapopOffers.language),
                    "tcgdex_id": func.coalesce(stmt.excluded.tcgdex_id, WallapopOffers.tcgdex_id),
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
                    "cardmarket_product_id": func.coalesce(stmt.excluded.cardmarket_product_id, WallapopOffers.cardmarket_product_id),
                    "language": func.coalesce(stmt.excluded.language, WallapopOffers.language),
                    "tcgdex_id": func.coalesce(stmt.excluded.tcgdex_id, WallapopOffers.tcgdex_id),
                    "updated_at": now,
                },
            )
            session.exec(stmt)
            session.commit()
            return len(records)

        else:
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
    cardmarket_url: Optional[str] = None,
    language: Optional[str] = None,
    engine: Optional[Engine] = None,
) -> bool:
    """Updates tcgdex_id, cardmarket_product_id, and language for an offer.

    Also ensures CardmarketProduct exists if cardmarket_product_id is provided.

    Args:
        offer_id: ID of the Wallapop offer.
        tcgdex_id: TCGDex identifier.
        cardmarket_product_id: Cardmarket product ID.
        cardmarket_url: Direct Cardmarket product URL.
        language: Card language code for the listing (e.g. 'es').
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

        if cardmarket_product_id:
            url = (
                cardmarket_url
                or f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={cardmarket_product_id}"
            )
            upsert_cardmarket_product(
                product_id=cardmarket_product_id,
                cardmarket_url=url,
                session=session,
            )
            session.flush()

        offer.tcgdex_id = tcgdex_id
        offer.cardmarket_product_id = cardmarket_product_id
        if language is not None:
            offer.language = language
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
        results = session.exec(statement).all()
        return list(results)
