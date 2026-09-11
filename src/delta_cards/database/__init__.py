"""Database package for delta-cards."""

from .models import WallapopOffers, CardmarketProduct, CardmarketPrice
from .connection import get_db_url, get_engine, get_session, init_db
from .repository import (
    save_offers_to_db,
    get_all_offers,
    update_offer_tcgdex,
    get_unassigned_card_offers,
    upsert_cardmarket_product,
    upsert_cardmarket_price,
)

__all__ = [
    "WallapopOffers",
    "CardmarketProduct",
    "CardmarketPrice",
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
]

