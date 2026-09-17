"""Vinted scraper orchestrator for card offers."""

import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import VintedSearchClient
from ..models import SearchConfig, SearchProfile, OfferListing
from ..wallapop.filters import is_offer_valid

logger = logging.getLogger(__name__)


def raw_to_offer_listing(
    raw: Dict[str, Any],
    profile_id: str,
    product_type: str,
    game: str = "pokemon",
) -> OfferListing:
    """Converts a raw Vinted dictionary into a clean OfferListing model."""
    item_id = str(raw.get("id"))
    # Prefix with vinted_ to distinguish and prevent ID collisions with Wallapop
    prefixed_id = f"vinted_{item_id}" if not item_id.startswith("vinted_") else item_id

    return OfferListing(
        id=prefixed_id,
        title=str(raw.get("title") or "").strip(),
        price=float(raw.get("price") or 0.0),
        currency=str(raw.get("currency") or "EUR"),
        product_type=product_type,
        platform="vinted",
        game=game,
        location=raw.get("location"),
        country=raw.get("country"),
        description=raw.get("description"),
        link=str(raw.get("link") or "").strip(),
        image_url=raw.get("image_url"),
        is_reserved=bool(raw.get("is_reserved", False)),
        shipping_available=raw.get("shipping_available", True),
        seller_user_id=raw.get("seller_user_id"),
        seller_link=raw.get("seller_link"),
        published_date=raw.get("published_date"),
        search_profile_id=profile_id,
        cardmarket_product_id=None,
        cardmarket_url=None,
        tcgdex_id=None,
        assigned_state="pending",
        alert_status="none",
    )


class VintedScraper:
    """Scrapes card offers from Vinted according to a SearchConfig."""

    def __init__(self, config: SearchConfig):
        self.config = config
        self.client = VintedSearchClient(config.vinted_settings)

    async def run(
        self,
        progress_callback: Optional[Any] = None,
        on_offer: Optional[Any] = None,
        on_batch: Optional[Any] = None,
    ) -> List[OfferListing]:
        """Runs the search pipeline across all configured profiles for Vinted.

        Args:
            progress_callback: Optional callable(event_name: str, data: dict) for UI progress.
            on_offer: Optional callable(offer: OfferListing) called as each valid unique offer is found.
            on_batch: Optional callable(offers: List[OfferListing]) called with newly found valid offers.

        Returns:
            List[OfferListing]: All unique validated listings collected.
        """
        all_offers: Dict[str, OfferListing] = {}
        total_profiles = len(self.config.search_profiles)

        for p_idx, profile in enumerate(self.config.search_profiles, 1):
            if not profile.enabled:
                continue

            queries = profile.queries
            logger.info(
                "Executing Vinted search profile: '%s' (%s) with %d query variation(s)...",
                profile.name,
                profile.id,
                len(queries),
            )
            if progress_callback:
                progress_callback(
                    "profile_start",
                    {
                        "profile": profile,
                        "profile_name": f"{profile.name} (Vinted)",
                        "profile_id": profile.id,
                        "index": p_idx,
                        "total": total_profiles,
                        "num_queries": len(queries),
                        "platform": "vinted",
                    },
                )

            profile_new_valid = 0
            profile_total_retrieved = 0

            for q_idx, query in enumerate(queries, 1):
                if progress_callback:
                    progress_callback(
                        "query_start",
                        {
                            "query": query,
                            "index": q_idx,
                            "total": len(queries),
                            "profile_name": profile.name,
                            "platform": "vinted",
                        },
                    )

                raw_items = await self.client.search_offers(
                    query=query,
                    min_price=profile.min_price,
                    max_price=profile.max_price,
                    max_total_items=profile.max_items,
                    order_by=self.config.vinted_settings.order_by,
                )
                profile_total_retrieved += len(raw_items)

                query_new_valid = 0
                batch_valid_offers: List[OfferListing] = []
                for item in raw_items:
                    valid, reason = is_offer_valid(item, profile)
                    if not valid:
                        logger.debug(
                            "Vinted item '%s' (ID %s) disqualified: %s",
                            item.get("title"),
                            item.get("id"),
                            reason,
                        )
                        continue

                    offer = raw_to_offer_listing(
                        raw=item,
                        profile_id=profile.id,
                        product_type=profile.product_type,
                        game=getattr(self.config, "game", "pokemon") or "pokemon",
                    )

                    if offer.id not in all_offers:
                        all_offers[offer.id] = offer
                        batch_valid_offers.append(offer)
                        query_new_valid += 1
                        profile_new_valid += 1
                        if on_offer:
                            on_offer(offer)

                if on_batch and batch_valid_offers:
                    on_batch(batch_valid_offers)

                if progress_callback:
                    progress_callback(
                        "query_done",
                        {
                            "query": query,
                            "index": q_idx,
                            "total": len(queries),
                            "retrieved": len(raw_items),
                            "kept": query_new_valid,
                            "profile_total_kept": profile_new_valid,
                            "total_offers": len(all_offers),
                            "platform": "vinted",
                        },
                    )

            if progress_callback:
                progress_callback(
                    "profile_done",
                    {
                        "profile": profile,
                        "profile_name": f"{profile.name} (Vinted)",
                        "retrieved_total": profile_total_retrieved,
                        "total_kept": profile_new_valid,
                        "total_offers": len(all_offers),
                        "platform": "vinted",
                    },
                )

        return list(all_offers.values())

    def save_results(self, offers: List[OfferListing]) -> Dict[str, Path]:
        """Saves scraped offers to JSON and CSV formats as configured."""
        out_settings = self.config.output_settings
        out_dir = Path(out_settings.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_files: Dict[str, Path] = {}
        prefix = f"{out_settings.filename_prefix}_vinted"

        if out_settings.save_json:
            json_path = out_dir / f"{prefix}.json"
            data = [offer.model_dump() for offer in offers]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            saved_files["json"] = json_path
            logger.info("Saved %d Vinted offers to JSON: %s", len(offers), json_path)

        if out_settings.save_csv:
            csv_path = out_dir / f"{prefix}.csv"
            fieldnames = [
                "id", "platform", "title", "price", "currency", "product_type",
                "link", "image_url", "search_profile_id", "assigned_state", "alert_status"
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for offer in offers:
                    writer.writerow(offer.model_dump())
            saved_files["csv"] = csv_path
            logger.info("Saved %d Vinted offers to CSV: %s", len(offers), csv_path)

        return saved_files
