"""Database package for delta-cards."""

from .models import WallapopOffers, CardmarketProduct, CardmarketPrice, CardmarketOffers
from .connection import get_db_url, get_engine, get_session, init_db
from .repository import (
    save_offers_to_db,
    get_all_offers,
    update_offer_tcgdex,
    get_unassigned_card_offers,
    upsert_cardmarket_product,
    upsert_cardmarket_price,
    upsert_cardmarket_offers,
    get_pending_alert_offers,
    update_offer_alert_status,
)

__all__ = [
    "WallapopOffers",
    "CardmarketProduct",
    "CardmarketPrice",
    "CardmarketOffers",
    "get_db_url",
    "get_engine",
    "get_session",
    "init_db",
    "save_offers_to_db",
    "get_all_offers",
    "update_offer_tcgdex",
    "get_unassigned_card_offers",
    "upsert_cardmarket_product",
    "upsert_cardmarket_price",
    "upsert_cardmarket_offers",
    "get_pending_alert_offers",
    "update_offer_alert_status",
]

