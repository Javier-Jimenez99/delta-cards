"""WallaPy client wrapper for Wallapop searches."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from wallapy import WallaPyClient
from ..models import WallapopSettings

logger = logging.getLogger(__name__)


class CardSearchClient:
    """Wrapper around WallaPyClient tailored for trading cards offers."""

    def __init__(self, settings: WallapopSettings):
        self.settings = settings
        self.delay_ms = settings.delay_between_requests_ms
        self.client = WallaPyClient(
            latitude=settings.latitude,
            longitude=settings.longitude,
            delay_between_requests=settings.delay_between_requests_ms,
        )

    async def search_offers(
        self,
        product_name: str,
        keywords: Optional[List[str]] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        excluded_keywords: Optional[List[str]] = None,
        max_total_items: int = 50,
        order_by: str = "newest",
    ) -> List[Dict[str, Any]]:
        """Executes a search for offers using WallaPyClient.

        Args:
            product_name: Primary search query.
            keywords: Additional keywords.
            min_price: Minimum price filter.
            max_price: Maximum price filter.
            excluded_keywords: Keywords to exclude.
            max_total_items: Limit of items to retrieve.
            order_by: Sorting criteria ('newest', 'price_low_to_high', etc.).

        Returns:
            List[Dict[str, Any]]: List of raw item dictionaries.
        """
        logger.info(
            "Searching Wallapop query='%s' (min_price=%s, max_price=%s, limit=%d)...",
            product_name,
            min_price,
            max_price,
            max_total_items,
        )

        try:
            results = await self.client.check_wallapop(
                product_name=product_name,
                keywords=keywords if keywords else None,
                min_price=min_price,
                max_price=max_price,
                excluded_keywords=excluded_keywords if excluded_keywords else None,
                max_total_items=max_total_items,
                order_by=order_by,
                deep_search=False,
            )
            return results or []
        except Exception as exc:
            logger.error("Error during Wallapop search for '%s': %s", product_name, exc)
            return []
