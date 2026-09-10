"""Unit tests for delta_cards database models, connection, and repository."""

import pytest
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, select, Session

from delta_cards.database.models import WallapopOffers
from delta_cards.database.connection import normalize_db_url, init_db
from delta_cards.database.repository import (
    offer_to_db_model,
    save_offers_to_db,
    get_all_offers,
)
from delta_cards.models import OfferListing


def test_wallapop_offers_is_pydantic_model():
    """Verifies that WallapopOffers inherits from Pydantic BaseModel and has correct table name."""
    assert issubclass(WallapopOffers, BaseModel)
    assert issubclass(WallapopOffers, SQLModel)
    assert WallapopOffers.__tablename__ == "WallapopOffers"


def test_wallapop_offers_validation_and_dump():
    """Tests Pydantic validation, default values, and serialization."""
    offer = WallapopOffers(
        id="item_999",
        title="Pokemon Booster Box 151",
        price=180.0,
        currency="EUR",
        product_type="booster_box",
        link="https://wallapop.com/item/item_999",
    )

    assert offer.id == "item_999"
    assert offer.price == 180.0
    assert offer.currency == "EUR"
    assert offer.is_reserved is False
    assert offer.cardmarket_product_id is None
    assert offer.tcgdex_id is None
    assert offer.scraped_at is not None
    assert offer.updated_at is not None

    dumped = offer.model_dump()
    assert dumped["id"] == "item_999"
    assert dumped["product_type"] == "booster_box"

    reconstructed = WallapopOffers.model_validate(dumped)
    assert reconstructed.id == offer.id

    # Test with Cardmarket and TCGDex IDs populated
    enriched = WallapopOffers(
        id="item_1000",
        title="Charizard Base Set",
        price=350.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/item_1000",
        cardmarket_product_id="cm_prod_12345",
        tcgdex_id="base1-4",
    )
    assert enriched.cardmarket_product_id == "cm_prod_12345"
    assert enriched.tcgdex_id == "base1-4"
    assert enriched.model_dump()["cardmarket_product_id"] == "cm_prod_12345"
    assert enriched.model_dump()["tcgdex_id"] == "base1-4"



def test_normalize_db_url():
    """Verifies that standard postgres URLs are rewritten to use psycopg (v3)."""
    assert normalize_db_url("postgres://user:pass@localhost:5432/db") == (
        "postgresql+psycopg://user:pass@localhost:5432/db"
    )
    assert normalize_db_url("postgresql://user:pass@localhost:5432/db") == (
        "postgresql+psycopg://user:pass@localhost:5432/db"
    )
    assert normalize_db_url("sqlite:///data.db") == "sqlite:///data.db"


def test_database_init_and_upsert():
    """Tests table creation and idempotent upserts in an in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    listing_1 = OfferListing(
        id="offer_1",
        title="Pokemon Base Set Charizard",
        price=350.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/offer_1",
        is_reserved=False,
    )
    listing_2 = OfferListing(
        id="offer_2",
        title="Pokemon 151 Booster Pack",
        price=8.0,
        currency="EUR",
        product_type="booster_pack",
        link="https://wallapop.com/item/offer_2",
        is_reserved=False,
    )

    # First insert
    saved_count = save_offers_to_db([listing_1, listing_2], engine=engine)
    assert saved_count == 2

    records = get_all_offers(engine=engine)
    assert len(records) == 2
    r_map = {r.id: r for r in records}
    assert r_map["offer_1"].price == 350.0
    assert r_map["offer_1"].is_reserved is False

    # Second insert with price change, reservation update, and enriched IDs (Upsert test)
    listing_1_updated = OfferListing(
        id="offer_1",
        title="Pokemon Base Set Charizard (Reserved)",
        price=300.0,  # Price drop
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/offer_1",
        is_reserved=True,  # Now reserved
        cardmarket_product_id="cm_123",
        tcgdex_id="base1-4",
    )

    saved_count_2 = save_offers_to_db([listing_1_updated], engine=engine)
    assert saved_count_2 == 1

    # Total records should still be 2 (no duplicate row for offer_1)
    records_after = get_all_offers(engine=engine)
    assert len(records_after) == 2

    r_map_after = {r.id: r for r in records_after}
    assert r_map_after["offer_1"].price == 300.0
    assert r_map_after["offer_1"].title == "Pokemon Base Set Charizard (Reserved)"
    assert r_map_after["offer_1"].is_reserved is True
    assert r_map_after["offer_1"].cardmarket_product_id == "cm_123"
    assert r_map_after["offer_1"].tcgdex_id == "base1-4"

