"""Tests for Scryfall client, Magic cardmarket pricing extraction, and Magic assigner."""

from unittest.mock import patch, MagicMock
import pytest
from sqlmodel import Session, select

from delta_cards.database.connection import get_engine
from delta_cards.database.models import PlatformOffers, CardmarketProduct, CardmarketPrice
from delta_cards.models import OfferListing
from delta_cards.scryfall.client import (
    ScryfallClient,
    build_magic_cardmarket_url,
    extract_scryfall_cardmarket_prices,
    fetch_scryfall_card_pricing,
)
from delta_cards.scryfall.assigner import (
    extract_magic_identifiers,
    resolve_scryfall_card,
    assign_scryfall_to_offers,
    enrich_magic_offers_in_db,
)


def test_build_magic_cardmarket_url():
    """Verify Magic Cardmarket URL construction."""
    assert build_magic_cardmarket_url("771076", "es") == "https://www.cardmarket.com/es/Magic/Products?idProduct=771076"
    assert build_magic_cardmarket_url("771076", "en") == "https://www.cardmarket.com/en/Magic/Products?idProduct=771076"
    assert build_magic_cardmarket_url(None) is None


def test_extract_scryfall_cardmarket_prices():
    """Verify extracting EUR prices and cardmarket product ID from Scryfall card dict."""
    sample_scryfall = {
        "id": "7673784e-db4b-43a1-8d55-1bb9fc1e284f",
        "name": "Lightning Bolt",
        "set": "msc",
        "set_name": "Marvel Super Heroes Commander",
        "collector_number": "806",
        "cardmarket_id": 892161,
        "prices": {
            "usd": "0.99",
            "usd_foil": "4.41",
            "eur": "2.03",
            "eur_foil": "1.86",
        },
        "purchase_uris": {
            "cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=892161&referrer=scryfall",
        },
        "image_uris": {
            "normal": "https://cards.scryfall.io/normal/front/7/6/7673784e-db4b-43a1-8d55-1bb9fc1e284f.jpg",
        },
    }

    result = extract_scryfall_cardmarket_prices(sample_scryfall, language="es")
    assert result["idProduct"] == "892161"
    assert result["url"] == "https://www.cardmarket.com/en/Magic/Products?idProduct=892161&referrer=scryfall"
    assert result["prices"]["eur"] == 2.03
    assert result["prices"]["eur_foil"] == 1.86
    assert result["prices"]["avg"] == 2.03
    assert result["prices"]["low"] == 2.03
    assert result["name"] == "Lightning Bolt"
    assert result["set"] == "msc"
    assert result["collector_number"] == "806"
    assert result["image_url"] == "https://cards.scryfall.io/normal/front/7/6/7673784e-db4b-43a1-8d55-1bb9fc1e284f.jpg"


def test_extract_magic_identifiers():
    """Verify extraction of set codes, numbers, and clean card names from marketplace titles."""
    # Pattern: Set code + collector number
    set_code, num, name = extract_magic_identifiers("Carta Magic The Gathering Ragavan, Nimble Pilferer MH2 138 Foil")
    assert set_code == "mh2"
    assert num == "138"
    assert "ragavan" in name.lower()

    # Pattern: Simple card name with marketplace fluff
    set_code2, num2, name2 = extract_magic_identifiers("Carta MTG Sheoldred, the Apocalypse Near Mint Spanish")
    assert "sheoldred" in name2.lower()
    assert "apocalypse" in name2.lower()


def test_resolve_scryfall_card_mock():
    """Verify resolve_scryfall_card delegates to ScryfallClient."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_set_number.return_value = {
        "id": "mock-uuid-1",
        "name": "The One Ring",
        "set": "ltr",
        "collector_number": "269",
        "cardmarket_id": 718300,
        "prices": {"eur": "95.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=718300"},
    }

    res = resolve_scryfall_card("The One Ring LTR 269", client=mock_client)
    assert res is not None
    assert res["idProduct"] == "718300"
    assert res["prices"]["avg"] == 95.0
    mock_client.get_card_by_set_number.assert_called_once_with("ltr", "269", lang="es")


def test_assign_scryfall_to_offers():
    """Verify assign_scryfall_to_offers populates OfferListing attributes."""
    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "id": "mock-uuid-2",
        "name": "Sheoldred, the Apocalypse",
        "set": "dmu",
        "collector_number": "107",
        "cardmarket_id": 673200,
        "prices": {"eur": "75.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/en/Magic/Products?idProduct=673200"},
    }

    offers = [
        OfferListing(
            id="offer-1",
            title="Carta Magic Sheoldred, the Apocalypse",
            price=25.0,
            product_type="single_card",
            platform="wallapop",
            link="https://es.wallapop.com/item/1",
        ),
        OfferListing(
            id="offer-2",
            title="Lote 500 cartas Magic bulk",
            price=15.0,
            product_type="booster_box",  # non-single card
            platform="wallapop",
            link="https://es.wallapop.com/item/2",
        ),
    ]

    assigned = assign_scryfall_to_offers(offers, client=mock_client)
    assert assigned[0].cardmarket_product_id == "673200"
    assert assigned[0].cardmarket_avg_price == 75.0
    # Non-card offer should remain unassigned
    assert getattr(assigned[1], "cardmarket_product_id", None) is None


def test_enrich_magic_offers_in_db():
    """Verify database enrichment updates PlatformOffers, CardmarketProduct, and detects deals."""
    engine = get_engine("sqlite:///:memory:")
    from delta_cards.database import init_db
    init_db(engine)

    mock_client = MagicMock(spec=ScryfallClient)
    mock_client.get_card_by_name.return_value = {
        "id": "uuid-sheoldred",
        "name": "Sheoldred, the Apocalypse",
        "set": "dmu",
        "collector_number": "107",
        "cardmarket_id": 673200,
        "prices": {"eur": "80.00"},
        "purchase_uris": {"cardmarket": "https://www.cardmarket.com/es/Magic/Products?idProduct=673200"},
    }

    with Session(engine) as session:
        # Deal candidate (price 30 < market 80)
        session.add(
            PlatformOffers(
                id="deal_offer",
                title="Sheoldred, the Apocalypse",
                price=30.0,
                product_type="single_card",
                platform="wallapop",
                game="magic",
                link="https://es.wallapop.com/item/deal",
                assigned_state="pending",
                alert_status="none",
            )
        )
        session.commit()

    updated = enrich_magic_offers_in_db(engine=engine, client=mock_client)
    assert updated == 1

    with Session(engine) as session:
        db_offer = session.exec(select(PlatformOffers).where(PlatformOffers.id == "deal_offer")).first()
        assert db_offer.assigned_state == "assigned"
        assert db_offer.cardmarket_product_id == "673200"
        assert db_offer.alert_status == "pending"  # Deal detected!

        cm_prod = session.exec(select(CardmarketProduct).where(CardmarketProduct.id == "673200")).first()
        assert cm_prod is not None
        assert "Magic" in cm_prod.cardmarket_url

        cm_price = session.exec(select(CardmarketPrice).where(CardmarketPrice.cardmarket_product_id == "673200")).first()
        assert cm_price is not None
        assert cm_price.avg_price == 80.0
