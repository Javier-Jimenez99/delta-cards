"""Database repository functions for saving and querying Wallapop offers, Cardmarket products, and prices."""

import logging
from datetime import datetime, timezone
from typing import Any, List, Optional
from sqlalchemy import func, case
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from ..models import OfferListing
from .connection import get_engine, get_session, init_db
from .models import WallapopOffers, CardmarketProduct, CardmarketPrice, CardmarketOffers

logger = logging.getLogger(__name__)


def offer_to_db_model(offer: OfferListing, now: Optional[datetime] = None) -> WallapopOffers:
    """Converts an OfferListing into a WallapopOffers database model instance."""
    now_utc = now or datetime.now(timezone.utc)
    valid_fields = set(WallapopOffers.model_fields.keys())
    data = {k: v for k, v in offer.model_dump().items() if k in valid_fields}
    data["scraped_at"] = now_utc
    data["updated_at"] = now_utc
    if not data.get("platform"):
        data["platform"] = getattr(offer, "platform", "wallapop") or "wallapop"
    if not data.get("assigned_state") or data["assigned_state"] == "pending":
        if data.get("cardmarket_product_id"):
            data["assigned_state"] = "assigned"
        else:
            data["assigned_state"] = "pending"
    if not data.get("alert_status"):
        data["alert_status"] = "none"
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
    avg_price: Optional[float] = None,
    trend_price: Optional[float] = None,
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
                avg_price=avg_price,
                trend_price=trend_price,
                scraped_at=now,
                updated_at=now,
            )
            s.add(price_rec)
        else:
            if min_price_all is not None:
                price_rec.min_price_all = min_price_all
            if min_price_spain is not None:
                price_rec.min_price_spain = min_price_spain
            if avg_price is not None:
                price_rec.avg_price = avg_price
            if trend_price is not None:
                price_rec.trend_price = trend_price
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


def upsert_cardmarket_offers(
    offers: List[Any],
    product_id: Optional[str] = None,
    engine: Optional[Engine] = None,
    session: Optional[Session] = None,
) -> int:
    """Inserts or updates a list of Cardmarket offer records into the CardmarketOffers table.

    Args:
        offers: List of CardmarketOffer dataclass/dict objects.
        product_id: Optional fallback product ID if not set on each offer.
        engine: Optional database engine.
        session: Optional existing Session.

    Returns:
        int: Number of offers saved or updated.
    """
    if not offers:
        return 0

    now = datetime.now(timezone.utc)

    def _do_upsert(s: Session) -> int:
        count = 0
        for off in offers:
            data = off if isinstance(off, dict) else (off.__dict__ if hasattr(off, "__dict__") else {})
            pid = data.get("product_id") or product_id or data.get("cardmarket_product_id")
            if not pid:
                continue

            # Ensure parent CardmarketProduct exists
            prod = s.exec(select(CardmarketProduct).where(CardmarketProduct.id == str(pid))).first()
            if not prod:
                prod = CardmarketProduct(
                    id=str(pid),
                    cardmarket_url=data.get("offer_url") or f"https://www.cardmarket.com/es/Pokemon/Products?idProduct={pid}",
                    scraped_at=now,
                    updated_at=now,
                )
                s.add(prod)
                s.flush()

            # Unique identifier for the offer
            oid = str(data.get("id") or f"{pid}_{data.get('seller_name')}_{data.get('price')}_{data.get('condition')}")

            existing = s.exec(select(CardmarketOffers).where(CardmarketOffers.id == oid)).first()
            if not existing:
                rec = CardmarketOffers(
                    id=oid,
                    cardmarket_product_id=str(pid),
                    seller_name=data.get("seller_name") or "Unknown",
                    seller_country=data.get("seller_country"),
                    seller_type=data.get("seller_type"),
                    seller_rating=data.get("seller_rating"),
                    seller_sales_count=data.get("seller_sales_count"),
                    price=float(data.get("price", 0.0)),
                    currency=data.get("currency", "EUR"),
                    condition=data.get("condition"),
                    language=data.get("language") or "es",
                    is_foil=bool(data.get("is_foil", False)),
                    is_signed=bool(data.get("is_signed", False)),
                    is_playset=bool(data.get("is_playset", False)),
                    is_first_edition=bool(data.get("is_first_edition", False)),
                    is_reverse_holo=bool(data.get("is_reverse_holo", False)),
                    is_altered=bool(data.get("is_altered", False)),
                    item_count=int(data.get("item_count") or data.get("quantity") or 1),
                    comment=data.get("comment"),
                    offer_url=data.get("offer_url"),
                    scraped_at=now,
                    updated_at=now,
                )
                s.add(rec)
            else:
                existing.price = float(data.get("price", existing.price))
                existing.condition = data.get("condition") or existing.condition
                existing.language = data.get("language") or existing.language
                existing.item_count = int(data.get("item_count") or data.get("quantity") or existing.item_count)
                existing.comment = data.get("comment") or existing.comment
                existing.updated_at = now
                s.add(existing)
            count += 1
        return count

    if session is not None:
        return _do_upsert(session)

    eng = engine or get_engine()
    with Session(eng) as s:
        c = _do_upsert(s)
        s.commit()
        return c


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
    init_db(eng)
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
                low_p = getattr(o, "cardmarket_low_price", None)
                avg_p = getattr(o, "cardmarket_avg_price", None)
                trend_p = getattr(o, "cardmarket_trend_price", None)
                if low_p is not None or avg_p is not None or trend_p is not None:
                    upsert_cardmarket_price(
                        product_id=o.cardmarket_product_id,
                        language=o.language or "es",
                        min_price_all=low_p,
                        avg_price=avg_p,
                        trend_price=trend_p,
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
                    "platform": func.coalesce(stmt.excluded.platform, WallapopOffers.platform),
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
                    "assigned_state": case(
                        (stmt.excluded.assigned_state == "assigned", "assigned"),
                        (WallapopOffers.assigned_state.is_not(None), WallapopOffers.assigned_state),
                        else_=stmt.excluded.assigned_state,
                    ),
                    "alert_status": case(
                        (stmt.excluded.alert_status.in_(["pending", "sent", "rejected"]), stmt.excluded.alert_status),
                        (WallapopOffers.alert_status.is_not(None), WallapopOffers.alert_status),
                        else_=stmt.excluded.alert_status,
                    ),
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
                    "platform": func.coalesce(stmt.excluded.platform, WallapopOffers.platform),
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
                    "assigned_state": case(
                        (stmt.excluded.assigned_state == "assigned", "assigned"),
                        (WallapopOffers.assigned_state.is_not(None), WallapopOffers.assigned_state),
                        else_=stmt.excluded.assigned_state,
                    ),
                    "alert_status": case(
                        (stmt.excluded.alert_status.in_(["pending", "sent", "rejected"]), stmt.excluded.alert_status),
                        (WallapopOffers.alert_status.is_not(None), WallapopOffers.alert_status),
                        else_=stmt.excluded.alert_status,
                    ),
                    "updated_at": now,
                },
            )
            session.exec(stmt)
            session.commit()
            return len(records)

        else:
            for offer in offers:
                item = offer_to_db_model(offer, now=now)
                existing = session.exec(select(WallapopOffers).where(WallapopOffers.id == item.id)).first()
                if existing and existing.assigned_state in ("assigned", "not_assigned") and item.assigned_state == "pending":
                    item.assigned_state = existing.assigned_state
                if existing and existing.alert_status in ("sent", "pending", "rejected") and item.alert_status == "none":
                    item.alert_status = existing.alert_status
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


def get_pending_alert_offers(
    engine: Optional[Engine] = None,
    limit: Optional[int] = None,
) -> List[WallapopOffers]:
    """Retrieves offers with alert_status == 'pending' waiting to be evaluated and dispatched.

    Orders them by scraped_at descending (newest offers first).
    """
    eng = engine or get_engine()
    with Session(eng) as session:
        statement = (
            select(WallapopOffers)
            .where(WallapopOffers.alert_status == "pending")
            .order_by(WallapopOffers.scraped_at.desc())
        )
        if limit is not None:
            statement = statement.limit(limit)
        results = session.exec(statement).all()
        return list(results)


def update_offer_alert_status(
    offer_id: str,
    status: str,
    engine: Optional[Engine] = None,
) -> bool:
    """Updates the alert_status of an offer.

    Args:
        offer_id: ID of the Wallapop offer.
        status: New alert status ('none', 'pending', 'sent', 'rejected').
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
        offer.alert_status = status
        offer.updated_at = now
        session.add(offer)
        session.commit()
        return True
