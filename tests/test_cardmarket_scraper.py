"""Unit tests for the Cardmarket scraper, proxy rotation, TLS impersonation, and Playwright fallback."""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from sqlmodel import Session, select

from delta_cards.cardmarket.constants import CARDMARKET_LANGUAGE_MAP, SPAIN_SELLER_COUNTRY_ID
from delta_cards.cardmarket.proxy_rotator import ProxyRotator
from delta_cards.cardmarket.playwright_fetcher import PlaywrightFetcher, PlaywrightFetchError
from delta_cards.cardmarket.scraper import (
    CardmarketScraper,
    CardmarketPriceResult,
    CardmarketOffer,
    build_filtered_url,
    parse_minimum_price,
    parse_offers,
    resolve_language_id,
)
from delta_cards.database.connection import get_engine, init_db
from delta_cards.database.models import CardmarketOffers, CardmarketProduct
from delta_cards.database.repository import upsert_cardmarket_offers


def test_resolve_language_id():
    """Verifies that language codes and names map to correct Cardmarket numeric IDs."""
    assert resolve_language_id("es") == 4
    assert resolve_language_id("spanish") == 4
    assert resolve_language_id("en") == 1
    assert resolve_language_id("english") == 1
    assert resolve_language_id("de") == 3
    assert resolve_language_id("it") == 5
    assert resolve_language_id(4) == 4
    assert resolve_language_id("4") == 4
    assert resolve_language_id("unsupported_lang") == 4


def test_build_filtered_url_from_product_id():
    """Tests URL construction from product ID with language and sellerCountry filters."""
    url_all = build_filtered_url("483559", language="es", spanish_sellers_only=False)
    assert "https://www.cardmarket.com/es/Pokemon/Products?" in url_all
    assert "idProduct=483559" in url_all
    assert "language=4" in url_all
    assert "sellerCountry" not in url_all

    url_spain = build_filtered_url("483559", language="es", spanish_sellers_only=True)
    assert "idProduct=483559" in url_spain
    assert "language=4" in url_spain
    assert f"sellerCountry={SPAIN_SELLER_COUNTRY_ID}" in url_spain
    assert "sellerCountry=10" in url_spain

    url_en = build_filtered_url("483559", language="en", spanish_sellers_only=False)
    assert "language=1" in url_en


def test_build_filtered_url_from_canonical_url():
    """Tests URL construction from full existing Cardmarket URL."""
    canonical = "https://www.cardmarket.com/en/Pokemon/Products/Singles/Darkness-Ablaze/Furret"
    filtered = build_filtered_url(canonical, language="es", spanish_sellers_only=True)
    assert "language=4" in filtered
    assert "sellerCountry=10" in filtered
    assert "/Products/Singles/Darkness-Ablaze/Furret" in filtered


def test_parse_minimum_price_from_article_rows():
    """Verifies that parse_minimum_price extracts the lowest price from article listings."""
    mock_html = """
    <html>
        <body>
            <div class="table-body">
                <div class="article-row row">
                    <div class="col-seller">Seller A (Germany)</div>
                    <div class="col-price text-right">
                        <span class="color-primary font-weight-bold">0,15 €</span>
                    </div>
                </div>
                <div class="article-row row">
                    <div class="col-seller">Seller B (Spain)</div>
                    <div class="col-price text-right">
                        <span class="color-primary font-weight-bold">0,08 €</span>
                    </div>
                </div>
                <div class="article-row row">
                    <div class="col-seller">Seller C (France)</div>
                    <div class="col-price text-right">
                        <span class="color-primary font-weight-bold">1,25 €</span>
                    </div>
                </div>
            </div>
        </body>
    </html>
    """
    min_price = parse_minimum_price(mock_html)
    assert min_price == 0.08


def test_parse_minimum_price_fallback_and_empty():
    """Verifies that parser handles alternative elements and returns None when empty."""
    mock_html_simple = """
    <div>
        <span class="price">12,50 €</span>
        <span class="price">9,99 €</span>
    </div>
    """
    assert parse_minimum_price(mock_html_simple) == 9.99

    mock_empty = """
    <div class="alert alert-info">
        No articles found matching your criteria.
    </div>
    """
    assert parse_minimum_price(mock_empty) is None
    assert parse_minimum_price("") is None


# ==============================================================================
# ProxyRotator Tests
# ==============================================================================

def test_proxy_rotator_round_robin_and_random():
    """Tests ProxyRotator initialization and rotation strategies."""
    proxies = ["http://p1.com:8080", "http://p2.com:8080", "http://p3.com:8080"]
    rotator = ProxyRotator(proxies=proxies, strategy="round_robin")

    assert rotator.total_count == 3
    assert rotator.active_count == 3
    assert rotator.has_proxies() is True

    # Round robin ordering
    assert rotator.get_proxy() == "http://p1.com:8080"
    assert rotator.get_proxy() == "http://p2.com:8080"
    assert rotator.get_proxy() == "http://p3.com:8080"
    assert rotator.get_proxy() == "http://p1.com:8080"

    # String input
    rotator_str = ProxyRotator(proxies="http://a:1, http://b:2")
    assert rotator_str.total_count == 2

    # Random strategy
    rotator_rnd = ProxyRotator(proxies=proxies, strategy="random")
    assert rotator_rnd.get_proxy() in proxies


def test_proxy_rotator_cooldown_and_health():
    """Tests failure reporting, temporary cooldown, and recovery upon success."""
    p1 = "http://p1.com:8080"
    p2 = "http://p2.com:8080"
    rotator = ProxyRotator(proxies=[p1, p2], max_failures=2, cooldown_seconds=60)

    # 1 failure on p1
    rotator.report_failure(p1)
    assert rotator.active_count == 2

    # 2nd failure on p1 -> triggers cooldown
    rotator.report_failure(p1)
    assert rotator.active_count == 1

    # Pool should now only return p2
    assert rotator.get_proxy() == p2
    assert rotator.get_proxy() == p2

    # Reporting success on p1 resets cooldown
    rotator.report_success(p1)
    assert rotator.active_count == 2


def test_proxy_rotator_to_playwright_proxy():
    """Tests converting standard proxy URLs to Playwright proxy dictionaries."""
    assert ProxyRotator.to_playwright_proxy(None) is None
    assert ProxyRotator.to_playwright_proxy("") is None

    p_simple = "http://proxy.example.com:8080"
    res1 = ProxyRotator.to_playwright_proxy(p_simple)
    assert res1 == {"server": "http://proxy.example.com:8080"}

    p_auth = "http://admin:secret123@192.168.1.50:3128"
    res2 = ProxyRotator.to_playwright_proxy(p_auth)
    assert res2 == {
        "server": "http://192.168.1.50:3128",
        "username": "admin",
        "password": "secret123",
    }


# ==============================================================================
# Specific Offers Parsing Tests
# ==============================================================================

def test_parse_offers_comprehensive():
    """Tests extracting structured CardmarketOffer objects from article HTML."""
    mock_html = """
    <div class="articles-table">
        <div class="article-row row" id="articleRow1001" data-article-id="1001">
            <div class="col-seller">
                <span class="flag-es icon-flag" title="Item location: Spain"></span>
                <span class="seller-name"><a href="/en/Pokemon/Users/PokeTraderES">PokeTraderES</a></span>
                <span class="badge badge-primary">Powerseller</span>
                <span class="sales">1.500 sales</span>
            </div>
            <div class="col-attributes">
                <span class="badge badge-success font-weight-bold">NM</span>
                <span class="icon-lang icon-flag" title="Spanish"></span>
                <span class="icon-foil" title="Foil"></span>
                <span class="icon-first-ed" title="1st Edition"></span>
            </div>
            <div class="col-price">
                <span class="color-primary font-weight-bold">14,99 €</span>
            </div>
            <div class="col-availability item-count">
                <span>3</span>
            </div>
            <div class="seller-comment">
                Mint card packed straight into sleeve and toploader!
            </div>
        </div>

        <div class="article-row row" id="articleRow1002" data-article-id="1002">
            <div class="col-seller">
                <span class="flag-de icon-flag" title="Germany"></span>
                <span class="seller-name">BerlinCards</span>
                <span class="badge">Private</span>
            </div>
            <div class="col-attributes">
                <span class="badge">EX</span>
                <span class="icon-lang" title="English"></span>
                <span class="icon-signed" title="Signed"></span>
            </div>
            <div class="col-price">
                <span class="color-primary">9,50 €</span>
            </div>
            <div class="col-availability">
                <span>1</span>
            </div>
        </div>
    </div>
    """
    offers = parse_offers(mock_html, product_id="483559")

    assert len(offers) == 2

    # Check offer 1
    o1 = offers[0]
    assert o1.id == "1001"
    assert o1.product_id == "483559"
    assert o1.seller_name == "PokeTraderES"
    assert o1.seller_country == "Spain"
    assert o1.seller_type == "Powerseller"
    assert o1.seller_sales_count == 1500
    assert o1.condition == "NM"
    assert o1.language == "es"
    assert o1.is_foil is True
    assert o1.is_first_edition is True
    assert o1.is_signed is False
    assert o1.price == 14.99
    assert o1.item_count == 3
    assert "sleeve and toploader" in (o1.comment or "")

    # Check offer 2
    o2 = offers[1]
    assert o2.id == "1002"
    assert o2.seller_name == "BerlinCards"
    assert o2.seller_country == "Germany"
    assert o2.seller_type == "Private"
    assert o2.condition == "EX"
    assert o2.language == "en"
    assert o2.is_foil is False
    assert o2.is_signed is True
    assert o2.price == 9.50
    assert o2.item_count == 1


def test_parse_offers_empty():
    """Tests that empty or no-articles HTML returns an empty list."""
    assert parse_offers("") == []
    mock_no_art = "<div class='alert alert-info'>No articles found</div>"
    assert parse_offers(mock_no_art) == []


# ==============================================================================
# Scraper & Fallback Flow Tests
# ==============================================================================

@pytest.mark.anyio
async def test_cardmarket_scraper_scrape_product_offers():
    """Tests scrape_product_offers using a mocked fetch_html."""
    scraper = CardmarketScraper()

    mock_html = """
    <div class="article-row" id="articleRow555" data-article-id="555">
        <div class="col-seller"><a href="/Users/CharizardFan">CharizardFan</a></div>
        <div class="col-attributes">
            <span class="badge">NM</span>
            <span class="icon-lang" title="Spanish"></span>
        </div>
        <div class="col-price">
            <span class="color-primary">45,00 €</span>
        </div>
    </div>
    """
    scraper.fetch_html = AsyncMock(return_value=mock_html)

    offers = await scraper.scrape_product_offers("483559", language="es", max_offers=10)
    assert len(offers) == 1
    assert offers[0].id == "555"
    assert offers[0].price == 45.0
    assert offers[0].condition == "NM"
    assert offers[0].seller_name == "CharizardFan"


@pytest.mark.anyio
async def test_cardmarket_scraper_curl_cffi_and_playwright_fallback():
    """Verifies that CardmarketScraper falls back to Playwright when curl_cffi is challenged."""
    scraper = CardmarketScraper(use_playwright_fallback=True)

    # Mock curl_cffi raising Cloudflare challenge
    scraper._fetch_with_curl_cffi = AsyncMock(side_effect=RuntimeError("Cloudflare challenge detected"))

    # Mock Playwright returning rendered page
    expected_html = "<div class='article-row'><span class='color-primary'>10,00 €</span></div>"
    scraper.playwright_fetcher.fetch_html = AsyncMock(return_value=expected_html)

    html = await scraper.fetch_html("https://www.cardmarket.com/es/Pokemon/Products/Singles/Base-Set/Charizard")
    assert html == expected_html
    scraper.playwright_fetcher.fetch_html.assert_called_once()


@pytest.mark.anyio
async def test_cardmarket_scraper_scrape_product_prices():
    """Tests scrape_product_prices extracting both overall and Spanish minimum prices."""
    scraper = CardmarketScraper()

    async def mock_fetch_html(url: str) -> str:
        if "sellerCountry=10" in url:
            return """
            <div class="article-row">
                <span class="color-primary font-weight-bold">0,25 €</span>
            </div>
            """
        else:
            return """
            <div class="article-row">
                <span class="color-primary font-weight-bold">0,08 €</span>
            </div>
            """

    scraper.fetch_html = mock_fetch_html

    result = await scraper.scrape_product_prices("483559", language="es")

    assert isinstance(result, CardmarketPriceResult)
    assert result.cardmarket_product_id == "483559"
    assert result.language == "es"
    assert result.language_id == 4
    assert result.min_price_all == 0.08
    assert result.min_price_spain == 0.25
    assert "sellerCountry=10" not in result.url_all
    assert "sellerCountry=10" in result.url_spain


# ==============================================================================
# Database Persistence Tests
# ==============================================================================

def test_database_upsert_cardmarket_offers():
    """Verifies that upsert_cardmarket_offers properly records offers and parent product."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)

    offers = [
        CardmarketOffer(
            id="art_101",
            product_id="483559",
            seller_name="AshKetchum",
            seller_country="Spain",
            seller_type="Private",
            price=25.50,
            condition="NM",
            language="es",
            is_foil=True,
            item_count=2,
            comment="Pack fresh",
        ),
        CardmarketOffer(
            id="art_102",
            product_id="483559",
            seller_name="GaryOak",
            seller_country="Germany",
            seller_type="Professional",
            price=22.00,
            condition="EX",
            language="en",
            is_foil=False,
            item_count=1,
        ),
    ]

    saved_count = upsert_cardmarket_offers(offers, product_id="483559", engine=engine)
    assert saved_count == 2

    with Session(engine) as session:
        # Verify parent CardmarketProduct exists
        prod = session.exec(select(CardmarketProduct).where(CardmarketProduct.id == "483559")).first()
        assert prod is not None

        # Verify saved offers
        db_offers = session.exec(
            select(CardmarketOffers).where(CardmarketOffers.cardmarket_product_id == "483559")
        ).all()
        assert len(db_offers) == 2

        o1 = next(o for o in db_offers if o.id == "art_101")
        assert o1.seller_name == "AshKetchum"
        assert o1.price == 25.50
        assert o1.is_foil is True
        assert o1.item_count == 2
        assert o1.comment == "Pack fresh"

    # Test update existing offer with new price
    updated_offer = CardmarketOffer(
        id="art_101",
        product_id="483559",
        seller_name="AshKetchum",
        price=23.50,  # Lowered price
        condition="NM",
        language="es",
        item_count=1,
    )
    upsert_cardmarket_offers([updated_offer], product_id="483559", engine=engine)

    with Session(engine) as session:
        o1_updated = session.exec(select(CardmarketOffers).where(CardmarketOffers.id == "art_101")).first()
        assert o1_updated is not None
        assert o1_updated.price == 23.50
        assert o1_updated.item_count == 1
