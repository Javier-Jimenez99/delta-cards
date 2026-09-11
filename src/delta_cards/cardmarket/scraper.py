"""Cardmarket price and specific offers scraper module with anti-bot bypass mechanisms.

Supports proxy rotation, TLS fingerprint impersonation using curl_cffi,
and browser simulation fallback using Playwright for Cloudflare challenges.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, parse_qs, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from .constants import (
    CARDMARKET_BASE_URL,
    CARDMARKET_LANGUAGE_MAP,
    SPAIN_SELLER_COUNTRY_ID,
)
from .proxy_rotator import ProxyRotator
from .playwright_fetcher import PlaywrightFetcher, PlaywrightFetchError
from ..settings import get_settings

logger = logging.getLogger(__name__)

CLOUDFLARE_BLOCK_INDICATORS = [
    "just a moment...",
    "attention required!",
    "cf-turnstile",
    "ddos protection by cloudflare",
    "checking your browser",
]

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# Country code and flag map
COUNTRY_NAME_MAP = {
    "es": "Spain",
    "spain": "Spain",
    "espana": "Spain",
    "de": "Germany",
    "germany": "Germany",
    "fr": "France",
    "france": "France",
    "it": "Italy",
    "italy": "Italy",
    "uk": "United Kingdom",
    "gb": "United Kingdom",
    "us": "United States",
    "at": "Austria",
    "be": "Belgium",
    "nl": "Netherlands",
    "pt": "Portugal",
    "ch": "Switzerland",
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


@dataclass
class CardmarketOffer:
    """Represents an individual seller offer listing on Cardmarket."""

    id: Optional[str] = None
    product_id: Optional[str] = None
    seller_name: str = "Unknown"
    seller_country: Optional[str] = None
    seller_type: Optional[str] = None
    seller_rating: Optional[str] = None
    seller_sales_count: Optional[int] = None
    price: float = 0.0
    currency: str = "EUR"
    condition: Optional[str] = None
    language: Optional[str] = None
    is_foil: bool = False
    is_signed: bool = False
    is_playset: bool = False
    is_first_edition: bool = False
    is_reverse_holo: bool = False
    is_altered: bool = False
    item_count: int = 1
    comment: Optional[str] = None
    offer_url: Optional[str] = None


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


def _to_float(raw: str) -> Optional[float]:
    """Helper to convert European and standard decimal strings to float."""
    raw = raw.strip()
    if "," in raw and "." in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


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

    prices = []

    # 1. Look for articles table rows
    article_rows = soup.find_all(class_=re.compile(r"article-row|articleTable|row.*has-action"))
    if not article_rows:
        table_body = soup.find(class_=re.compile(r"table-body|articles-table"))
        if table_body:
            article_rows = table_body.find_all(class_=re.compile(r"row"))

    for row in article_rows:
        price_elem = row.find(class_=re.compile(r"col-price|price-container|color-primary", re.I))
        text_to_search = price_elem.get_text() if price_elem else ""
        if "€" not in text_to_search:
            euro_node = row.find(string=re.compile(r"€"))
            text_to_search = euro_node.parent.get_text() if (euro_node and euro_node.parent) else row.get_text()

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


def parse_offers(html_content: str, product_id: Optional[str] = None) -> List[CardmarketOffer]:
    """Parses individual Cardmarket seller offers from HTML content.

    Args:
        html_content: Raw HTML content of Cardmarket product page.
        product_id: Optional Cardmarket product ID to associate with offers.

    Returns:
        List of parsed CardmarketOffer objects.
    """
    if not html_content:
        return []

    soup = BeautifulSoup(html_content, "html.parser")

    # Check for empty listings indicator
    empty_msgs = soup.find_all(string=re.compile(r"(no articles found|no hay artículos|keine artikel)", re.I))
    if empty_msgs and not soup.find(class_=re.compile(r"article-row|table-body")):
        return []

    # Find article rows
    article_rows = soup.find_all(class_=re.compile(r"article-row|articleTable|row.*has-action"))
    if not article_rows:
        table_body = soup.find(class_=re.compile(r"table-body|articles-table"))
        if table_body:
            article_rows = table_body.find_all(class_=re.compile(r"row"))

    offers: List[CardmarketOffer] = []

    for row in article_rows:
        # 1. Article ID
        article_id = row.get("data-article-id") or row.get("id")
        if article_id and "articleRow" in article_id:
            article_id = article_id.replace("articleRow", "")
        if not article_id:
            btn = row.find(["button", "a", "input"], attrs={"name": re.compile(r"articleId", re.I)})
            if btn:
                article_id = btn.get("value") or btn.get("data-id")

        # 2. Price
        price_elem = row.find(class_=re.compile(r"col-price|price-container|color-primary", re.I))
        price_text = price_elem.get_text() if price_elem else ""
        if "€" not in price_text:
            euro_node = row.find(string=re.compile(r"€"))
            price_text = euro_node.parent.get_text() if (euro_node and euro_node.parent) else row.get_text()

        match_price = re.search(r"(\d+(?:[.,]\d+)?)\s*€|€\s*(\d+(?:[.,]\d+)?)", price_text)
        if not match_price:
            continue
        price_val = _to_float(match_price.group(1) or match_price.group(2))
        if price_val is None:
            continue

        # 3. Seller Information
        seller_name = "Unknown"
        seller_country = None
        seller_type = None
        seller_rating = None
        seller_sales = None

        seller_col = row.find(class_=re.compile(r"col-seller|seller", re.I))
        if seller_col:
            name_link = seller_col.find("a", href=re.compile(r"/Users/"))
            if name_link:
                seller_name = name_link.get_text(strip=True)
            else:
                seller_span = seller_col.find(class_=re.compile(r"seller-name|d-flex", re.I))
                if seller_span:
                    seller_name = seller_span.get_text(strip=True)
                else:
                    raw_text = seller_col.get_text(strip=True)
                    if raw_text:
                        seller_name = raw_text.split()[0]

            # Country from flag icon or title
            flag_elem = seller_col.find(attrs={"class": re.compile(r"flag-|icon-flag", re.I)})
            if flag_elem:
                classes = flag_elem.get("class", [])
                for cls_name in classes:
                    flag_match = re.search(r"flag-([a-z]{2})", cls_name, re.I)
                    if flag_match:
                        code = flag_match.group(1).lower()
                        seller_country = COUNTRY_NAME_MAP.get(code, code.upper())
                        break

            if not seller_country:
                title_elem = seller_col.find(attrs={"title": re.compile(r"location|ubicación|spain|españa|germany", re.I)})
                if title_elem:
                    t_val = title_elem.get("title", "")
                    for k, v in COUNTRY_NAME_MAP.items():
                        if k in t_val.lower():
                            seller_country = v
                            break

            # Seller type (Private, Professional, Powerseller)
            type_badge = seller_col.find(class_=re.compile(r"badge|seller-type", re.I))
            if type_badge:
                tb_text = type_badge.get_text(strip=True).lower()
                if "power" in tb_text:
                    seller_type = "Powerseller"
                elif "pro" in tb_text:
                    seller_type = "Professional"
                elif "priv" in tb_text:
                    seller_type = "Private"
                else:
                    seller_type = type_badge.get_text(strip=True)

            # Rating / Sales count
            sales_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:sales|ventas)", seller_col.get_text(), re.I)
            if sales_match:
                try:
                    seller_sales = int(sales_match.group(1).replace(".", "").replace(",", ""))
                except ValueError:
                    pass

        # 4. Condition & Attributes
        condition = None
        card_lang = None
        is_foil = False
        is_signed = False
        is_playset = False
        is_first_ed = False
        is_reverse_holo = False
        is_altered = False

        attr_col = row.find(class_=re.compile(r"col-attributes|col-condition|product-attributes", re.I))
        search_scope = attr_col if attr_col else row

        # Condition badge (MT, NM, EX, GD, LP, PL, PO)
        cond_badge = search_scope.find(class_=re.compile(r"badge.*condition|badge", re.I))
        if cond_badge:
            c_text = cond_badge.get_text(strip=True).upper()
            if c_text in ("MT", "NM", "EX", "GD", "LP", "PL", "PO"):
                condition = c_text
        if not condition:
            cond_match = re.search(r"\b(MT|NM|EX|GD|LP|PL|PO|Near Mint|Mint|Excellent|Light Played)\b", search_scope.get_text(), re.I)
            if cond_match:
                condition = cond_match.group(1).upper()

        # Language from flag or title in attributes
        lang_flag = search_scope.find(attrs={"class": re.compile(r"icon-lang|icon-flag", re.I)})
        if lang_flag:
            l_title = (lang_flag.get("title") or lang_flag.get("aria-label") or "").lower()
            for l_code, l_name in [("es", "spanish"), ("en", "english"), ("de", "german"), ("fr", "french"), ("it", "italian"), ("ja", "japanese")]:
                if l_code in l_title or l_name in l_title:
                    card_lang = l_code
                    break

        # Attribute icons (foil, signed, playset, 1st ed)
        is_foil = bool(search_scope.find(attrs={"class": re.compile(r"icon-foil|badge-foil|is-foil", re.I)}) or
                       search_scope.find(attrs={"title": re.compile(r"foil|holo", re.I)}))
        is_signed = bool(search_scope.find(attrs={"class": re.compile(r"icon-signed", re.I)}) or
                         search_scope.find(attrs={"title": re.compile(r"signed|firmad", re.I)}))
        is_playset = bool(search_scope.find(attrs={"class": re.compile(r"icon-playset", re.I)}) or
                          search_scope.find(attrs={"title": re.compile(r"playset|4x", re.I)}))
        is_first_ed = bool(search_scope.find(attrs={"class": re.compile(r"icon-first-ed|icon-firsted", re.I)}) or
                           search_scope.find(attrs={"title": re.compile(r"first edition|1st edition|1ª edición", re.I)}))
        is_reverse_holo = bool(search_scope.find(attrs={"title": re.compile(r"reverse", re.I)}))
        is_altered = bool(search_scope.find(attrs={"title": re.compile(r"altered|alterad", re.I)}))

        # 5. Quantity / Item Count
        item_count = 1
        qty_elem = row.find(class_=re.compile(r"item-count|col-availability|col-amount|count", re.I))
        if qty_elem:
            qty_match = re.search(r"\b(\d+)\b", qty_elem.get_text())
            if qty_match:
                item_count = int(qty_match.group(1))

        # 6. Seller Comment
        comment = None
        comment_elem = row.find(class_=re.compile(r"seller-comment|article-comment|comment", re.I))
        if comment_elem:
            comment = comment_elem.get_text(strip=True) or None

        offer = CardmarketOffer(
            id=article_id or f"{product_id or 'prod'}_{seller_name}_{price_val}_{condition}",
            product_id=product_id,
            seller_name=seller_name,
            seller_country=seller_country,
            seller_type=seller_type,
            seller_rating=seller_rating,
            seller_sales_count=seller_sales,
            price=price_val,
            currency="EUR",
            condition=condition,
            language=card_lang or "es",
            is_foil=is_foil,
            is_signed=is_signed,
            is_playset=is_playset,
            is_first_edition=is_first_ed,
            is_reverse_holo=is_reverse_holo,
            is_altered=is_altered,
            item_count=item_count,
            comment=comment,
        )
        offers.append(offer)

    return offers


class CardmarketScraper:
    """Cardmarket scraper with proxy rotation, curl_cffi TLS impersonation, and Playwright fallback."""

    def __init__(
        self,
        proxies: Optional[Union[str, List[str]]] = None,
        proxies_file: Optional[Union[str, Any]] = None,
        proxy_rotator: Optional[ProxyRotator] = None,
        impersonate: Optional[str] = None,
        impersonate_profiles: Optional[List[str]] = None,
        use_playwright_fallback: Optional[bool] = None,
        playwright_headless: Optional[bool] = None,
        timeout: float = 20.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        settings = get_settings()

        # Initialize proxy rotator
        if proxy_rotator is not None:
            self.proxy_rotator = proxy_rotator
        else:
            p_list = proxies or settings.cardmarket_proxies
            p_file = proxies_file or settings.cardmarket_proxies_file
            self.proxy_rotator = ProxyRotator(proxies=p_list, proxies_file=p_file)

        # TLS Impersonation profiles
        self.default_impersonate = impersonate or settings.cardmarket_impersonate or "safari180"
        self.impersonate_profiles = impersonate_profiles or [
            self.default_impersonate,
            "safari170",
            "chrome124",
            "chrome131",
        ]
        # Keep unique order
        seen = set()
        self.impersonate_profiles = [x for x in self.impersonate_profiles if not (x in seen or seen.add(x))]

        # Playwright fallback settings
        self.use_playwright_fallback = (
            use_playwright_fallback
            if use_playwright_fallback is not None
            else settings.cardmarket_use_playwright_fallback
        )
        self.playwright_headless = (
            playwright_headless
            if playwright_headless is not None
            else settings.cardmarket_playwright_headless
        )
        self.playwright_timeout = settings.cardmarket_playwright_timeout or 30.0

        self.timeout = timeout
        self.headers = headers or DEFAULT_HEADERS
        self._warned_cloudflare = False
        self._playwright_fetcher: Optional[PlaywrightFetcher] = None

    @property
    def playwright_fetcher(self) -> PlaywrightFetcher:
        """Lazy-loaded Playwright fetcher instance."""
        if self._playwright_fetcher is None:
            self._playwright_fetcher = PlaywrightFetcher(
                headless=self.playwright_headless,
                timeout=self.playwright_timeout,
            )
        return self._playwright_fetcher

    def _is_cloudflare_challenge(self, status_code: int, text: str) -> bool:
        """Determines whether a response represents a Cloudflare bot protection challenge."""
        if status_code in (403, 503):
            return True
        text_lower = text.lower()
        return any(ind in text_lower for ind in CLOUDFLARE_BLOCK_INDICATORS)

    async def _fetch_with_curl_cffi(
        self,
        url: str,
        proxy: Optional[str] = None,
        impersonate: str = "safari180",
    ) -> str:
        """Fetches page HTML using curl_cffi with specified TLS impersonation and proxy."""
        from curl_cffi.requests import AsyncSession

        proxies_arg = {"all": proxy} if proxy else None
        async with AsyncSession(impersonate=impersonate, proxies=proxies_arg) as session:
            resp = await session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
            )
            if self._is_cloudflare_challenge(resp.status_code, resp.text):
                raise RuntimeError(
                    f"Cloudflare challenge detected (HTTP {resp.status_code}, profile={impersonate})"
                )
            if resp.status_code >= 400:
                resp.raise_for_status()
            return resp.text

    async def fetch_html(self, url: str) -> str:
        """Fetches page HTML using rotating proxies and TLS impersonation, with Playwright fallback.

        Args:
            url: Target URL to retrieve.

        Returns:
            The raw HTML content.

        Raises:
            RuntimeError: If all scraping methods and fallbacks are blocked.
        """
        last_error: Optional[Exception] = None

        # 1. Try with curl_cffi and proxy rotation
        if self.proxy_rotator.has_proxies():
            # Try across available active proxies
            attempts = min(3, self.proxy_rotator.active_count or 1)
            for _ in range(attempts):
                proxy = self.proxy_rotator.get_proxy()
                for imp in self.impersonate_profiles[:2]:
                    try:
                        logger.debug("Attempting curl_cffi fetch with proxy %s and profile %s", proxy, imp)
                        html = await self._fetch_with_curl_cffi(url, proxy=proxy, impersonate=imp)
                        self.proxy_rotator.report_success(proxy)
                        return html
                    except Exception as exc:
                        last_error = exc
                        self.proxy_rotator.report_failure(proxy, exc)
                        break

        # 2. Try direct connection with curl_cffi impersonations
        for imp in self.impersonate_profiles:
            try:
                logger.debug("Attempting direct curl_cffi fetch with profile %s", imp)
                html = await self._fetch_with_curl_cffi(url, proxy=None, impersonate=imp)
                return html
            except Exception as exc:
                last_error = exc

        # 3. Fallback to Playwright browser simulation
        if self.use_playwright_fallback:
            logger.info("curl_cffi requests blocked by Cloudflare; attempting Playwright browser simulation...")
            active_proxy = self.proxy_rotator.get_proxy() if self.proxy_rotator.has_proxies() else None
            try:
                html = await self.playwright_fetcher.fetch_html(url, proxy=active_proxy)
                if active_proxy:
                    self.proxy_rotator.report_success(active_proxy)
                return html
            except Exception as pw_err:
                logger.debug("Playwright browser simulation failed: %s", pw_err)
                if active_proxy:
                    self.proxy_rotator.report_failure(active_proxy, pw_err)
                last_error = pw_err

        # If everything failed
        if not self._warned_cloudflare:
            logger.warning(
                "Cardmarket web requests blocked by Cloudflare. Direct scraping failed: %s",
                last_error,
            )
            self._warned_cloudflare = True
        raise RuntimeError(f"Cardmarket scraping failed: {last_error}")

    async def scrape_product_offers(
        self,
        product_id_or_url: str,
        language: Union[str, int] = "es",
        spanish_sellers_only: bool = False,
        max_offers: Optional[int] = None,
    ) -> List[CardmarketOffer]:
        """Scrapes specific seller offers for a Cardmarket product.

        Args:
            product_id_or_url: Cardmarket Product ID (e.g. '483559') or product URL.
            language: Language code (e.g. 'es', 'en') or numeric ID.
            spanish_sellers_only: If True, filters to sellers located in Spain.
            max_offers: Optional limit on the number of offers returned.

        Returns:
            List of CardmarketOffer instances.
        """
        pid = str(product_id_or_url)
        if "idProduct=" in pid:
            match = re.search(r"idProduct=(\d+)", pid)
            if match:
                pid = match.group(1)

        url = build_filtered_url(
            product_id_or_url=product_id_or_url,
            language=language,
            spanish_sellers_only=spanish_sellers_only,
        )

        html = await self.fetch_html(url)
        offers = parse_offers(html, product_id=pid)

        if max_offers and len(offers) > max_offers:
            offers = offers[:max_offers]

        return offers

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
            if not self._warned_cloudflare:
                logger.warning(
                    "Cardmarket web requests blocked by Cloudflare (%s). "
                    "Direct web price scraping skipped; using TCGDex reference prices.",
                    exc,
                )
                self._warned_cloudflare = True
            logger.debug("%s: %s", msg, exc)

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
