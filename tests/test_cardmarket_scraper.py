"""Unit tests for the Cardmarket scraper, URL building, language mapping, and price extraction."""

import pytest
from unittest.mock import AsyncMock

from delta_cards.cardmarket.constants import CARDMARKET_LANGUAGE_MAP, SPAIN_SELLER_COUNTRY_ID
from delta_cards.cardmarket.scraper import (
    CardmarketScraper,
    CardmarketPriceResult,
    build_filtered_url,
    parse_minimum_price,
    resolve_language_id,
)


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
    # Unknown defaults to Spanish (4)
    assert resolve_language_id("unsupported_lang") == 4


def test_build_filtered_url_from_product_id():
    """Tests URL construction from product ID with language and sellerCountry filters."""
    # Spanish language, all sellers
    url_all = build_filtered_url("483559", language="es", spanish_sellers_only=False)
    assert "https://www.cardmarket.com/es/Pokemon/Products?" in url_all
    assert "idProduct=483559" in url_all
    assert "language=4" in url_all
    assert "sellerCountry" not in url_all

    # Spanish language, Spanish sellers only (sellerCountry=10)
    url_spain = build_filtered_url("483559", language="es", spanish_sellers_only=True)
    assert "idProduct=483559" in url_spain
    assert "language=4" in url_spain
    assert f"sellerCountry={SPAIN_SELLER_COUNTRY_ID}" in url_spain
    assert "sellerCountry=10" in url_spain

    # English language
    url_en = build_filtered_url("483559", language="en", spanish_sellers_only=False)
    assert "language=1" in url_en


def test_build_filtered_url_from_canonical_url():
    """Tests URL construction from full existing Cardmarket URL."""
    canonical = "https://www.cardmarket.com/en/Pokemon/Products/Singles/Darkness-Ablaze/Furret"

    # Filter by Spanish language and Spanish sellers
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
    # Alternative format with spans
    mock_html_simple = """
    <div>
        <span class="price">12,50 €</span>
        <span class="price">9,99 €</span>
    </div>
    """
    assert parse_minimum_price(mock_html_simple) == 9.99

    # Empty listings page
    mock_empty = """
    <div class="alert alert-info">
        No articles found matching your criteria.
    </div>
    """
    assert parse_minimum_price(mock_empty) is None
    assert parse_minimum_price("") is None


@pytest.mark.anyio
async def test_cardmarket_scraper_scrape_product_prices():
    """Tests scrape_product_prices extracting both overall and Spanish minimum prices."""
    scraper = CardmarketScraper()

    # Mock fetch_html to return different prices for overall vs Spanish sellers
    async def mock_fetch_html(url: str) -> str:
        if "sellerCountry=10" in url:
            # Spanish seller page: minimum is 0.25 €
            return """
            <div class="article-row">
                <span class="color-primary font-weight-bold">0,25 €</span>
            </div>
            """
        else:
            # Overall sellers page: minimum is 0.08 € (from Germany, etc.)
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
