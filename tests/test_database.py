"""Unit tests for delta_cards database models, connection, and repository."""

import pytest
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, select, Session

from delta_cards.database.models import WallapopOffers, CardmarketProduct, CardmarketPrice
from delta_cards.database.connection import normalize_db_url, init_db
from delta_cards.database.repository import (
    offer_to_db_model,
    save_offers_to_db,
    get_all_offers,
    upsert_cardmarket_product,
    upsert_cardmarket_price,
)
from delta_cards.models import OfferListing


def test_models_inheritance_and_table_names():
    """Verifies that database models inherit from Pydantic BaseModel and SQLModel."""
    assert issubclass(WallapopOffers, BaseModel)
    assert issubclass(WallapopOffers, SQLModel)
    assert WallapopOffers.__tablename__ == "WallapopOffers"

    assert issubclass(CardmarketProduct, BaseModel)
    assert issubclass(CardmarketProduct, SQLModel)
    assert CardmarketProduct.__tablename__ == "CardmarketProduct"

    assert issubclass(CardmarketPrice, BaseModel)
    assert issubclass(CardmarketPrice, SQLModel)
    assert CardmarketPrice.__tablename__ == "CardmarketPrice"


def test_cardmarket_product_and_price_models():
    """Tests CardmarketProduct and CardmarketPrice creation and validation."""
    prod = CardmarketProduct(
        id="483559",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559",
    )
    assert prod.id == "483559"
    assert "idProduct=483559" in prod.cardmarket_url
    assert prod.scraped_at is not None
    assert prod.updated_at is not None

    price = CardmarketPrice(
        cardmarket_product_id="483559",
        language="es",
        min_price_all=0.08,
        min_price_spain=0.15,
    )
    assert price.cardmarket_product_id == "483559"
    assert price.language == "es"
    assert price.min_price_all == 0.08
    assert price.min_price_spain == 0.15


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
    assert offer.language == "es"
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
        language="es",
        tcgdex_id="base1-4",
    )
    assert enriched.cardmarket_product_id == "cm_prod_12345"
    assert enriched.language == "es"
    assert enriched.tcgdex_id == "base1-4"
    assert enriched.model_dump()["cardmarket_product_id"] == "cm_prod_12345"
    assert enriched.model_dump()["language"] == "es"
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
    """Tests table creation, CardmarketProduct creation, and idempotent upserts."""
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
        product_type="booster_box",
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
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=cm_123",
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

    # Verify that CardmarketProduct was created in database
    with Session(engine) as session:
        prod = session.exec(select(CardmarketProduct).where(CardmarketProduct.id == "cm_123")).first()
        assert prod is not None
        assert prod.id == "cm_123"
        assert "idProduct=cm_123" in prod.cardmarket_url


def test_cardmarket_repository_helpers_and_relations():
    """Tests upsert_cardmarket_product, upsert_cardmarket_price, and foreign key relations."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    # 1. Upsert product
    prod = upsert_cardmarket_product(
        product_id="483559",
        cardmarket_url="https://www.cardmarket.com/es/Pokemon/Products?idProduct=483559",
        engine=engine,
    )
    assert prod.id == "483559"

    # 2. Upsert prices for Spanish and English
    price_es = upsert_cardmarket_price(
        product_id="483559",
        language="es",
        min_price_all=0.08,
        min_price_spain=0.12,
        engine=engine,
    )
    assert price_es.id is not None
    assert price_es.language == "es"
    assert price_es.min_price_all == 0.08
    assert price_es.min_price_spain == 0.12

    price_en = upsert_cardmarket_price(
        product_id="483559",
        language="en",
        min_price_all=0.02,
        min_price_spain=0.20,
        engine=engine,
    )
    assert price_en.id != price_es.id

    # 3. Create offer with language
    offer = OfferListing(
        id="offer_furret_1",
        title="Furret Oscuridad Incandescente Español",
        price=1.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/furret_1",
        cardmarket_product_id="483559",
        language="es",
    )
    save_offers_to_db([offer], engine=engine)

    with Session(engine) as session:
        db_offer = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_furret_1")).first()
        assert db_offer is not None
        assert db_offer.cardmarket_product_id == "483559"
        assert db_offer.language == "es"

        # Check relationships from offer side
        assert db_offer.cardmarket_product.id == "483559"
        assert db_offer.cardmarket_price is not None
        assert db_offer.cardmarket_price.language == "es"
        assert db_offer.cardmarket_price.min_price_spain == 0.12

        # Check relationship from price side
        price_db = session.exec(select(CardmarketPrice).where(CardmarketPrice.id == price_es.id)).first()
        assert len(price_db.offers) == 1
        assert price_db.offers[0].id == "offer_furret_1"


def test_offers_decoupled_price_lifecycle():
    """Verifies that an offer saved before any CardmarketPrice exists automatically

    links to the price as soon as CardmarketPrice is scraped and saved, without
    needing to update WallapopOffers.
    """
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    # 1. Offer is discovered on Wallapop, matched to product ID 9999, language 'es'
    offer = OfferListing(
        id="offer_pikachu",
        title="Pikachu Illustrator Español",
        price=50.0,
        product_type="single_card",
        link="https://wallapop.com/item/pikachu",
        cardmarket_product_id="9999",
        language="es",
    )
    save_offers_to_db([offer], engine=engine)

    with Session(engine) as session:
        db_offer = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_pikachu")).first()
        assert db_offer.cardmarket_product.id == "9999"
        # No price scraped yet
        assert db_offer.cardmarket_price is None

    # 2. Later, price scraper runs for product 9999 in Spanish
    upsert_cardmarket_price(
        product_id="9999",
        language="es",
        min_price_all=45.0,
        min_price_spain=48.0,
        engine=engine,
    )

    # 3. Verify offer now immediately resolves the price without any update to WallapopOffers
    with Session(engine) as session:
        db_offer = session.exec(select(WallapopOffers).where(WallapopOffers.id == "offer_pikachu")).first()
        assert db_offer.cardmarket_price is not None
        assert db_offer.cardmarket_price.min_price_all == 45.0
        assert db_offer.cardmarket_price.min_price_spain == 48.0

