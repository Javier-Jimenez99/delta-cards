"""Wallapop scraper orchestrator for card offers."""

import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from .client import CardSearchClient
from .filters import is_offer_valid
from ..config_loader import load_config
from ..models import SearchConfig, OfferListing

logger = logging.getLogger(__name__)


def extract_user_id(seller_link: Optional[str]) -> Optional[str]:
    """Extracts seller user ID from a Wallapop user URL if present."""
    if not seller_link:
        return None
    try:
        parts = urlparse(seller_link).path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "user":
            return parts[1]
    except Exception:
        pass
    return None


def raw_to_offer_listing(
    raw: Dict[str, Any],
    profile_id: str,
    product_type: str,
) -> OfferListing:
    """Converts a raw WallaPy result dictionary into a clean OfferListing model."""
    creation_date = raw.get("creation_date_utc")
    published_str = None
    if isinstance(creation_date, datetime):
        published_str = creation_date.isoformat()
    elif creation_date:
        published_str = str(creation_date)

    seller_link = raw.get("seller_link")
    user_id = raw.get("user_id") or extract_user_id(seller_link)

    img = raw.get("main_image")
    if not img and raw.get("all_images"):
        images = raw.get("all_images")
        if isinstance(images, list) and images:
            img = images[0]

    return OfferListing(
        id=str(raw.get("id")),
        title=str(raw.get("title") or "").strip(),
        price=float(raw.get("price") or 0.0),
        currency=str(raw.get("currency") or "EUR"),
        product_type=product_type,
        location=raw.get("location"),
        country=raw.get("state"),
        description=raw.get("description"),
        link=str(raw.get("link") or ""),
        image_url=img,
        is_reserved=bool(raw.get("is_reserved", False)),
        shipping_available=raw.get("shipping_available"),
        seller_user_id=user_id,
        seller_link=seller_link,
        published_date=published_str,
        search_profile_id=profile_id,
        cardmarket_product_id=raw.get("cardmarket_product_id"),
        tcgdex_id=raw.get("tcgdex_id"),
    )



class CardOffersScraper:
    """Scrapes card offers from Wallapop according to a SearchConfig."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.client = CardSearchClient(config.wallapop_settings)

    async def run(
        self,
        progress_callback: Optional[Any] = None,
    ) -> List[OfferListing]:
        """Runs the search pipeline across all configured profiles.

        Args:
            progress_callback: Optional callable(event_name: str, data: dict) for UI progress.
        """
        all_offers: Dict[str, OfferListing] = {}
        total_profiles = len(self.config.search_profiles)

        for p_idx, profile in enumerate(self.config.search_profiles, 1):
            queries = profile.queries
            logger.info(
                "Executing search profile: '%s' (%s) with %d query variation(s)...",
                profile.name,
                profile.id,
                len(queries),
            )
            if progress_callback:
                progress_callback(
                    "profile_start",
                    {
                        "profile": profile,
                        "profile_name": profile.name,
                        "profile_id": profile.id,
                        "index": p_idx,
                        "total": total_profiles,
                        "num_queries": len(queries),
                    },
                )

            profile_new_valid = 0
            profile_total_retrieved = 0

            for q_idx, q in enumerate(queries, 1):
                logger.info(
                    "  [%d/%d] Searching query='%s'...",
                    q_idx,
                    len(queries),
                    q,
                )
                if progress_callback:
                    progress_callback(
                        "query_start",
                        {
                            "query": q,
                            "index": q_idx,
                            "total": len(queries),
                            "profile_name": profile.name,
                        },
                    )

                raw_items = await self.client.search_offers(
                    product_name=q,
                    keywords=profile.keywords,
                    min_price=profile.min_price,
                    max_price=profile.max_price,
                    excluded_keywords=profile.excluded_keywords,
                    max_total_items=profile.max_items,
                    order_by=self.config.wallapop_settings.order_by,
                )
                profile_total_retrieved += len(raw_items)
                query_new_valid = 0

                for item in raw_items:
                    item_id = str(item.get("id"))
                    if not item_id or item_id in all_offers:
                        continue

                    valid, reason = is_offer_valid(item, profile)
                    if not valid:
                        logger.debug("Filtered out item %s: %s", item_id, reason)
                        continue

                    offer = raw_to_offer_listing(
                        raw=item,
                        profile_id=profile.id,
                        product_type=profile.product_type,
                    )
                    all_offers[item_id] = offer
                    profile_new_valid += 1
                    query_new_valid += 1

                if progress_callback:
                    progress_callback(
                        "query_done",
                        {
                            "query": q,
                            "index": q_idx,
                            "total": len(queries),
                            "retrieved": len(raw_items),
                            "kept": query_new_valid,
                            "profile_total_kept": profile_new_valid,
                            "total_offers": len(all_offers),
                        },
                    )

                delay_sec = self.config.wallapop_settings.delay_between_requests_ms / 1000.0
                if delay_sec > 0 and (q_idx < len(queries)):
                    await asyncio.sleep(delay_sec)

            logger.info(
                "Profile '%s': retrieved %d total items across %d query variation(s), added %d unique valid offers.",
                profile.name,
                profile_total_retrieved,
                len(queries),
                profile_new_valid,
            )
            if progress_callback:
                progress_callback(
                    "profile_done",
                    {
                        "profile": profile,
                        "profile_name": profile.name,
                        "retrieved_total": profile_total_retrieved,
                        "total_kept": profile_new_valid,
                        "total_offers": len(all_offers),
                    },
                )

            delay_sec = self.config.wallapop_settings.delay_between_requests_ms / 1000.0
            if delay_sec > 0:
                await asyncio.sleep(delay_sec)

        offer_list = list(all_offers.values())
        logger.info("Total unique valid offers collected: %d", len(offer_list))
        return offer_list

    def save_results(self, offers: List[OfferListing]) -> Dict[str, Path]:
        """Saves scraped offers to JSON and CSV formats as configured."""
        out_settings = self.config.output_settings
        out_dir = Path(out_settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}
        prefix = out_settings.filename_prefix

        # Save JSON
        if out_settings.save_json:
            json_path = out_dir / f"{prefix}.json"
            data = [offer.model_dump() for offer in offers]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved_files["json"] = json_path
            logger.info("Saved %d offers to JSON: %s", len(offers), json_path)

        # Save CSV
        if out_settings.save_csv:
            csv_path = out_dir / f"{prefix}.csv"
            fieldnames = [
                "id",
                "title",
                "price",
                "currency",
                "product_type",
                "location",
                "country",
                "link",
                "shipping_available",
                "is_reserved",
                "published_date",
                "seller_user_id",
                "seller_link",
                "image_url",
                "cardmarket_product_id",
                "tcgdex_id",
            ]

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for offer in offers:
                    writer.writerow(offer.model_dump())
            saved_files["csv"] = csv_path
            logger.info("Saved %d offers to CSV: %s", len(offers), csv_path)

        return saved_files


def run_scraper(config_path: str) -> List[OfferListing]:
    """Synchronous helper entry point to execute the scraper."""
    config = load_config(config_path)
    scraper = CardOffersScraper(config)
    offers = asyncio.run(scraper.run())
    scraper.save_results(offers)
    return offers
