"""Unit tests for TCGDex pricing integration, extraction, database persistence, and alerts."""

import pytest
from sqlmodel import Session, create_engine, select

from delta_cards.alerts.service import (
    AlertConditionResult,
    DiscordAlertService,
    build_discord_embed,
    evaluate_alert_condition,
)
from delta_cards.database.connection import init_db
from delta_cards.database.models import CardmarketPrice, CardmarketProduct, WallapopOffers
from delta_cards.database.repository import (
    save_offers_to_db,
    upsert_cardmarket_price,
    upsert_cardmarket_product,
)
from delta_cards.models import OfferListing
from delta_cards.settings import Settings
from delta_cards.tcgdex.assigner import (
    _apply_card_result_to_offer,
    enrich_card_offers_in_db,
    extract_cardmarket_prices,
)


def test_tcgdex_price_extraction():
    """Verify that extract_cardmarket_prices accurately parses avg, low, and trend prices from TCGDex card JSON."""
    # Standard card pricing
    sample_data = {
        "pricing": {
            "cardmarket": {
                "updated": "2026-09-11T12:14:32.691Z",
                "unit": "EUR",
                "idProduct": 733620,
                "avg": 0.15,
                "low": 0.02,
                "trend": 0.12,
            }
        }
    }
    prices = extract_cardmarket_prices(sample_data)
    assert prices["avg"] == 0.15
    assert prices["low"] == 0.02
    assert prices["trend"] == 0.12

    # Holo card fallback where avg is only in avg-holo
    sample_holo = {
        "pricing": {
            "cardmarket": {
                "avg-holo": 2.50,
                "low-holo": 1.10,
                "trend-holo": 2.30,
            }
        }
    }
    prices_holo = extract_cardmarket_prices(sample_holo)
    assert prices_holo["avg"] == 2.50
    assert prices_holo["low"] == 1.10
    assert prices_holo["trend"] == 2.30

    # Variants detailed fallback
    sample_variants = {
        "variants_detailed": [
            {
                "pricing": {
                    "cardmarket": {
                        "avg": 15.0,
                        "low": 10.0,
                        "trend": 14.5,
                    }
                }
            }
        ]
    }
    prices_var = extract_cardmarket_prices(sample_variants)
    assert prices_var["avg"] == 15.0
    assert prices_var["low"] == 10.0
    assert prices_var["trend"] == 14.5

    # Missing or empty pricing
    empty_prices = extract_cardmarket_prices({})
    assert empty_prices["avg"] is None
    assert empty_prices["low"] is None
    assert empty_prices["trend"] is None


def test_apply_card_result_to_offer_tcgdex_prices():
    """Verify _apply_card_result_to_offer attaches TCGDex avg, low, and trend prices to OfferListing."""
    offer = OfferListing(
        id="offer_001",
        title="Pikachu 025/165",
        price=1.0,
        currency="EUR",
        product_type="single_card",
        link="https://wallapop.com/item/001",
    )

    pipeline_res = {
        "status": "success",
        "verified_card": {"id": "sv03.5-025", "name": "Pikachu"},
        "cardmarket": {
            "idProduct": "733620",
            "url": "https://www.cardmarket.com/es/Pokemon/Products?idProduct=733620",
            "prices": {"avg": 0.50, "low": 0.10, "trend": 0.45},
        },
        "extracted_metadata": {"language": "es"},
    }

    _apply_card_result_to_offer(offer, pipeline_res)

    assert offer.tcgdex_id == "sv03.5-025"
    assert offer.cardmarket_product_id == "733620"
    assert offer.cardmarket_avg_price == 0.50
    assert offer.cardmarket_low_price == 0.10
    assert offer.cardmarket_trend_price == 0.45
    assert getattr(offer, "_temp_cm_avg_price") == 0.50


def test_upsert_cardmarket_price_database():
    """Verify upsert_cardmarket_price persists avg_price, trend_price, and min_price_all."""
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)

    upsert_cardmarket_price(
        product_id="733620",
        language="es",
        min_price_all=0.10,
        avg_price=0.50,
        trend_price=0.45,
        engine=engine,
    )

    with Session(engine) as session:
        price_rec = session.exec(
            select(CardmarketPrice).where(CardmarketPrice.cardmarket_product_id == "733620")
        ).first()
        assert price_rec is not None
        assert price_rec.avg_price == 0.50
        assert price_rec.min_price_all == 0.10
        assert price_rec.trend_price == 0.45


def test_discord_alert_uses_precio_medio():
    """Verify that Discord embed displays '📊 Precio medio' and evaluates savings against average price."""
    offer = OfferListing(
        id="offer_deal_1",
        title="Gengar VMAX Fusion Strike",
        price=15.0,
        currency="EUR",
        product_type="single_card",
        link="https://es.wallapop.com/item/gengar-vmax",
        cardmarket_product_id="574000",
        cardmarket_avg_price=30.0,
        cardmarket_low_price=20.0,
    )

    # Offer: 15€, Market Avg: 30€ -> Savings: 15€ (50%)
    cond = evaluate_alert_condition(
        offer_price=offer.price,
        market_price=offer.cardmarket_avg_price,
        settings=Settings(_env_file=None, alert_min_saving_percentage=20.0),
    )
    assert cond.is_triggered is True
    assert cond.savings_absolute == 15.0
    assert cond.savings_percentage == 50.0

    embed = build_discord_embed(offer, cond)
    fields = {f["name"]: f["value"] for f in embed["fields"]}

    # Ensure Row 1 has Wallapop, Precio medio, and Precio mínimo with custom emotes
    assert "<:wallapop:1548103689477038080> Wallapop" in fields
    assert "15.00 €" in fields["<:wallapop:1548103689477038080> Wallapop"]

    assert "<:cardmarket:1548103777750487056> Precio medio" in fields
    assert "30.00 €" in fields["<:cardmarket:1548103777750487056> Precio medio"]

    assert "<:cardmarket:1548103777750487056> Precio mínimo" in fields
    assert "20.00 €" in fields["<:cardmarket:1548103777750487056> Precio mínimo"]

    # Ensure Row 2 has Ahorro, Tipo, Ubicación
    assert "💰 Ahorro" in fields
    assert "15.00 €" in fields["💰 Ahorro"]

    assert "📦 Tipo" in fields
    assert "📍 Ubicación" in fields

    # Ensure Spanish and European flags are removed
    assert "🇪🇸 ESP" not in fields
    assert "🇪🇺 UE" not in fields
    assert "🇪🇸" not in str(embed)
    assert "🇪🇺" not in str(embed)

    # Ensure Row 3 links have no emotes in link text
    links_val = fields["🔗 Enlaces"]
    assert "[Oferta en Wallapop]" in links_val
    assert "[Producto en Cardmarket]" in links_val
    assert "🛒" not in links_val
    assert "📈" not in links_val
