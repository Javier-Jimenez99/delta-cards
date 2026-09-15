"""Tests for Vinted scraper module, models, and database persistence."""

import pytest
from delta_cards.models import OfferListing, SearchConfig, SearchProfile, VintedSettings
from delta_cards.vinted import VintedSearchClient, VintedScraper, raw_to_offer_listing
from delta_cards.database.models import WallapopOffers
from delta_cards.database.repository import offer_to_db_model, save_offers_to_db
from delta_cards.cli import build_parser
from sqlmodel import SQLModel, create_engine, Session, select


def test_raw_to_offer_listing_vinted():
    """Verify raw Vinted dict conversion to OfferListing model."""
    raw = {
        "id": "987654321",
        "title": "Mewtwo GX 78/73",
        "price": 25.5,
        "currency": "EUR",
        "link": "https://www.vinted.es/items/987654321-mewtwo-gx",
        "image_url": "https://images1.vinted.net/mewtwo.webp",
    }
    offer = raw_to_offer_listing(raw, profile_id="single_card", product_type="single_card")

    assert offer.id == "vinted_987654321"
    assert offer.title == "Mewtwo GX 78/73"
    assert offer.price == 25.5
    assert offer.platform == "vinted"
    assert offer.product_type == "single_card"
    assert offer.assigned_state == "pending"
    assert offer.alert_status == "none"


def test_parse_catalog_html():
    """Verify HTML parsing from Vinted catalog box."""
    sample_html = """
    <div class="ItemBox-module" data-testid="product-item-id-10011904127">
      <div class="web_ui__Image">
        <img src="https://images1.vinted.net/item.webp"
             alt="Umbreon GX, Marca: Pokémon, Estado: Nuevo sin etiquetas, 15.00 €, 16.45 €" />
      </div>
      <a href="/items/10011904127-umbreon-gx?referrer=catalog" title="Umbreon GX"></a>
      <div>15,00 €</div>
    </div>
    """
    client = VintedSearchClient()
    items = client._parse_catalog_html(sample_html)

    assert len(items) == 1
    item = items[0]
    assert item["id"] == "10011904127"
    assert item["title"] == "Umbreon GX"
    assert item["price"] == 15.0
    assert item["link"] == "https://www.vinted.es/items/10011904127-umbreon-gx"
    assert item["image_url"] == "https://images1.vinted.net/item.webp"


def test_db_persistence_vinted_platform():
    """Verify OfferListing with platform='vinted' is persisted correctly in DB."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    offer = OfferListing(
        id="vinted_555666",
        title="Pikachu VMAX",
        price=35.0,
        product_type="single_card",
        platform="vinted",
        link="https://www.vinted.es/items/555666-pikachu-vmax",
        image_url="https://images1.vinted.net/pika.webp",
    )

    db_item = offer_to_db_model(offer)
    assert db_item.platform == "vinted"
    assert db_item.id == "vinted_555666"

    saved = save_offers_to_db([offer], engine=engine)
    assert saved == 1

    with Session(engine) as session:
        fetched = session.exec(select(WallapopOffers).where(WallapopOffers.id == "vinted_555666")).first()
        assert fetched is not None
        assert fetched.platform == "vinted"
        assert fetched.price == 35.0
        assert fetched.assigned_state == "pending"


def test_cli_parser_platform_argument():
    """Verify CLI parser supports --platform and --test-discord-vinted."""
    parser = build_parser()

    # Default platform is wallapop
    args = parser.parse_args(["scrape"])
    assert args.platform == "wallapop"

    # Specify vinted
    args = parser.parse_args(["scrape", "--platform", "vinted"])
    assert args.platform == "vinted"

    # Specify all
    args = parser.parse_args(["scrape", "--platform", "all"])
    assert args.platform == "all"

    # Test discord vinted flag
    args = parser.parse_args(["--test-discord-vinted"])
    assert args.test_discord_vinted is True
