"""HTTP client for querying Vinted card offers."""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import httpx

from ..models import VintedSettings

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class VintedSearchClient:
    """Client for querying the Vinted public catalog using HTML parsing."""

    def __init__(self, settings: Optional[VintedSettings] = None):
        self.settings = settings or VintedSettings()
        self.domain = self.settings.domain or "vinted.es"
        self.delay_sec = max(0.5, self.settings.delay_between_requests_ms / 1000.0)
        self.headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": f"https://www.{self.domain}/",
        }

    async def search_offers(
        self,
        query: str,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        max_total_items: int = 50,
        order_by: str = "newest_first",
    ) -> List[Dict[str, Any]]:
        """Searches Vinted catalog for listings matching query and price criteria.

        Args:
            query: Search query string (e.g. 'pokemon carta').
            min_price: Optional minimum price filter in EUR.
            max_price: Optional maximum price filter in EUR.
            max_total_items: Maximum items to collect.
            order_by: Sorting order (e.g. 'newest_first').

        Returns:
            List of raw item dictionaries.
        """
        results: List[Dict[str, Any]] = []
        page = 1
        seen_ids = set()

        # Build base URL and query parameters
        base_url = f"https://www.{self.domain}/catalog"

        async with httpx.AsyncClient(headers=self.headers, timeout=15.0, follow_redirects=True) as client:
            while len(results) < max_total_items:
                params: Dict[str, Any] = {
                    "search_text": query,
                    "order": order_by or "newest_first",
                    "page": page,
                }
                if min_price is not None:
                    params["price_from"] = min_price
                if max_price is not None:
                    params["price_to"] = max_price

                logger.info(
                    "Querying Vinted (page %d): query='%s' (min=%s, max=%s)...",
                    page,
                    query,
                    min_price,
                    max_price,
                )

                try:
                    resp = await client.get(base_url, params=params)
                    if resp.status_code != 200:
                        logger.warning(
                            "Vinted search returned HTTP %d for query '%s'",
                            resp.status_code,
                            query,
                        )
                        break

                    page_items = self._parse_catalog_html(resp.text)
                    if not page_items:
                        logger.debug("No items parsed on Vinted page %d for '%s'", page, query)
                        break

                    new_items_count = 0
                    for item in page_items:
                        item_id = item.get("id")
                        if item_id and item_id not in seen_ids:
                            seen_ids.add(item_id)
                            results.append(item)
                            new_items_count += 1
                            if len(results) >= max_total_items:
                                break

                    if new_items_count == 0:
                        break

                    page += 1
                    await asyncio.sleep(self.delay_sec)

                except Exception as exc:
                    logger.error("Error during Vinted request for '%s': %s", query, exc)
                    break

        return results

    def _parse_catalog_html(self, html_content: str) -> List[Dict[str, Any]]:
        """Parses Vinted catalog HTML and extracts structured item dictionaries."""
        items: List[Dict[str, Any]] = []
        soup = BeautifulSoup(html_content, "html.parser")

        product_boxes = soup.find_all(attrs={"data-testid": re.compile(r"^product-item-id-\d+$")})
        for box in product_boxes:
            test_id = box.get("data-testid", "")
            id_match = re.search(r"product-item-id-(\d+)", test_id)
            if not id_match:
                continue
            item_id = id_match.group(1)

            # Extract image & alt
            img = box.find("img")
            img_url = img.get("src") if img else None
            alt_text = img.get("alt", "") if img else ""

            # Extract link
            a_tag = box.find("a", href=re.compile(r"^/items/"))
            href = a_tag.get("href", "") if a_tag else ""
            clean_href = href.split("?")[0] if href else ""
            link = f"https://www.{self.domain}{clean_href}" if clean_href else ""

            # Extract title: from alt text or title attribute
            title_text = alt_text
            for token in [", Marca:", ", Estado:", ", Brand:", ", Condition:", ", Brand :", ", Condition :"]:
                if token in title_text:
                    title_text = title_text.split(token)[0]
                    break
            title_text = title_text.strip()
            if not title_text and a_tag and a_tag.get("title"):
                title_text = a_tag.get("title", "").strip()

            # Extract price
            price_val: Optional[float] = None
            for string_val in box.stripped_strings:
                cleaned = string_val.replace("\xa0", "").replace(" ", "").replace(",", ".")
                m = re.match(r"^(\d+(?:\.\d+)?)(?:€|EUR)?$", cleaned)
                if m:
                    try:
                        price_val = float(m.group(1))
                        break
                    except ValueError:
                        pass

            if price_val is None and alt_text:
                price_matches = re.findall(r"(\d+(?:[.,]\d+)?)\s*€", alt_text)
                if price_matches:
                    try:
                        price_val = float(price_matches[0].replace(",", "."))
                    except ValueError:
                        pass

            if item_id and title_text and price_val is not None:
                items.append({
                    "id": item_id,
                    "title": title_text,
                    "price": price_val,
                    "currency": "EUR",
                    "link": link,
                    "image_url": img_url,
                    "description": title_text,
                    "is_reserved": False,
                })

        return items
