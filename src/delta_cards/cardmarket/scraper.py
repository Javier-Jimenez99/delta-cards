"""Cardmarket price scraper module for extracting minimum product prices by language and country."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from urllib.parse import urlencode, parse_qs, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup

from .constants import (
    CARDMARKET_BASE_URL,
    CARDMARKET_LANGUAGE_MAP,
    SPAIN_SELLER_COUNTRY_ID,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


@dataclass
class CardmarketPriceResult:
    """Stores the scraped price result for a card in a specific language."""

    cardmarket_product_id: str
    language: str
    language_id: int
    min_price_all: Optional[float]
    min_price_spain: Optional[float]
    url_all: str
    url_spain: str


def resolve_language_id(language: Union[str, int]) -> int:
    """Resolves a language string (e.g. 'es', 'en') or int to a Cardmarket numeric ID."""
    if isinstance(language, int):
        return language
    lang_str = str(language).lower().strip()
    if lang_str.isdigit():
        return int(lang_str)
    return CARDMARKET_LANGUAGE_MAP.get(lang_str, 4)  # Default to Spanish (4)


def build_filtered_url(
    product_id_or_url: str,
    language: Union[str, int] = "es",
    spanish_sellers_only: bool = False,
    interface_lang: str = "es",
) -> str:
    """Builds a Cardmarket product URL with language and country query filters.

    Args:
        product_id_or_url: Either a numeric product ID ('483559') or a full Cardmarket URL.
        language: Language code ('es', 'en') or numeric ID (4, 1).
        spanish_sellers_only: If True, appends sellerCountry=10.
        interface_lang: Language path segment for direct ID navigation (default 'es').

    Returns:
        The filtered Cardmarket URL.
    """
    lang_id = resolve_language_id(language)

    if product_id_or_url.startswith("http://") or product_id_or_url.startswith("https://"):
        parts = urlsplit(product_id_or_url)
        query = parse_qs(parts.query)
        query["language"] = [str(lang_id)]
        if spanish_sellers_only:
            query["sellerCountry"] = [str(SPAIN_SELLER_COUNTRY_ID)]
        else:
            query.pop("sellerCountry", None)
        new_query = urlencode(query, doseq=True)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))

    pid = str(product_id_or_url).strip()
    params = {"idProduct": pid, "language": str(lang_id)}
    if spanish_sellers_only:
        params["sellerCountry"] = str(SPAIN_SELLER_COUNTRY_ID)

    return f"{CARDMARKET_BASE_URL}/{interface_lang}/Pokemon/Products?{urlencode(params)}"


def parse_minimum_price(html_content: str) -> Optional[float]:
    """Parses HTML content from a Cardmarket product page and extracts the lowest price.

    Args:
        html_content: Raw HTML text of the page.

    Returns:
        The minimum price as float, or None if no listings/prices found.
    """
    if not html_content:
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Check for empty listings indicator
    empty_msgs = soup.find_all(string=re.compile(r"(no articles found|no hay artículos|keine artikel)", re.I))
    if empty_msgs and not soup.find(class_=re.compile(r"article-row|table-body")):
        return None

    def _to_float(raw: str) -> Optional[float]:
        raw = raw.strip()
        if "," in raw and "." in raw:
            raw = raw.replace(".", "").replace(",", ".")
        elif "," in raw:
            raw = raw.replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            return None

    prices = []

    # 1. Look for articles table rows
    article_rows = soup.find_all(class_=re.compile(r"article-row|articleTable|row.*has-action"))
    if not article_rows:
        table_body = soup.find(class_=re.compile(r"table-body|articles-table"))
        if table_body:
            article_rows = table_body.find_all(class_=re.compile(r"row"))

    for row in article_rows:
        # Price is typically located in a dedicated price column or bold primary span
        price_elem = row.find(class_=re.compile(r"col-price|price-container|color-primary|font-weight-bold"))
        text_to_search = price_elem.get_text() if price_elem else row.get_text()
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*€|€\s*(\d+(?:[.,]\d+)?)", text_to_search)
        if match:
            val_str = match.group(1) or match.group(2)
            val = _to_float(val_str)
            if val is not None:
                prices.append(val)

    # 2. Fallback: Search all price-like elements in the document
    if not prices:
        for span in soup.find_all(["span", "div", "td"], class_=re.compile(r"price|color-primary")):
            text = span.get_text().strip()
            match = re.search(r"^(\d+(?:[.,]\d+)?)\s*€$", text)
            if match:
                val = _to_float(match.group(1))
                if val is not None:
                    prices.append(val)

    if prices:
        return min(prices)

    return None


class CardmarketScraper:
    """Scraper client for retrieving minimum card prices from Cardmarket."""

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 15.0,
    ):
        self.headers = headers or DEFAULT_HEADERS
        self.timeout = timeout

    async def fetch_html(self, url: str) -> str:
        """Fetches page HTML with standard browser headers."""
        async with httpx.AsyncClient(
            headers=self.headers,
            timeout=self.timeout,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def scrape_product_prices(
        self,
        product_id_or_url: str,
        language: Union[str, int] = "es",
    ) -> CardmarketPriceResult:
        """Scrapes both overall minimum price and Spanish seller minimum price for a product.

        Args:
            product_id_or_url: Cardmarket Product ID (e.g. '483559') or product URL.
            language: Language code (e.g. 'es', 'en') or numeric ID.

        Returns:
            CardmarketPriceResult with min_price_all and min_price_spain.
        """
        lang_id = resolve_language_id(language)
        lang_code = "es" if lang_id == 4 else ("en" if lang_id == 1 else str(language))

        # Extract product ID if full URL passed
        pid = str(product_id_or_url)
        if "idProduct=" in pid:
            match = re.search(r"idProduct=(\d+)", pid)
            if match:
                pid = match.group(1)

        url_all = build_filtered_url(
            product_id_or_url=product_id_or_url,
            language=language,
            spanish_sellers_only=False,
        )
        url_spain = build_filtered_url(
            product_id_or_url=product_id_or_url,
            language=language,
            spanish_sellers_only=True,
        )

        min_all: Optional[float] = None
        min_spain: Optional[float] = None

        def _handle_scrape_error(msg: str, exc: Exception) -> None:
            is_403 = False
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403:
                is_403 = True
            elif "403" in str(exc):
                is_403 = True

            if is_403:
                if not getattr(self, "_warned_cloudflare", False):
                    logger.warning(
                        "Cardmarket web requests blocked by Cloudflare (HTTP 403). "
                        "Direct web price scraping skipped; using TCGDex reference prices."
                    )
                    self._warned_cloudflare = True
                logger.debug("%s: %s", msg, exc)
            else:
                logger.warning("%s: %s", msg, exc)

        try:
            html_all = await self.fetch_html(url_all)
            min_all = parse_minimum_price(html_all)
        except Exception as exc:
            _handle_scrape_error(f"Failed to fetch overall prices for {pid} ({url_all})", exc)

        try:
            html_spain = await self.fetch_html(url_spain)
            min_spain = parse_minimum_price(html_spain)
        except Exception as exc:
            _handle_scrape_error(f"Failed to fetch Spanish seller prices for {pid} ({url_spain})", exc)

        return CardmarketPriceResult(
            cardmarket_product_id=pid,
            language=lang_code,
            language_id=lang_id,
            min_price_all=min_all,
            min_price_spain=min_spain,
            url_all=url_all,
            url_spain=url_spain,
        )
