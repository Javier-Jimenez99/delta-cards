"""Daily Pokémon TCG Catalog Synchronization Service.

Automatically checks TCGdex for newly released sets and cards,
incrementally downloading card art, computing embeddings, and caching OCR
by delegating to the unified index builder.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from tcgdexsdk import TCGdex

from ..recognition.index_builder import build_index
from ..recognition.vector_index import USearchCardIndex

logger = logging.getLogger(__name__)


@dataclass
class SyncSummary:
    """Summary of a catalog synchronization run."""

    updated: bool = False
    new_cards_added: int = 0
    new_sets_detected: int = 0
    total_cards_indexed: int = 0
    elapsed_seconds: float = 0.0
    sets_updated: List[str] = field(default_factory=list)


class CatalogSyncService:
    """Synchronizes local offline USearch index with the latest TCGdex catalog."""

    def __init__(
        self,
        index_path: str = "data/pokemon_cards.usearch",
        cache_dir: str = "data/images_cache",
        language: str = "en",
        concurrency: int = 16,
    ):
        self.index_path = index_path
        self.meta_path = index_path.rsplit(".", 1)[0] + "_meta.sqlite"
        self.cache_dir = cache_dir
        self.language = language
        self.concurrency = concurrency

    async def check_for_new_cards(self) -> List[dict]:
        """Fast pre-check to detect if remote TCGdex catalog has cards not in local index.

        Returns:
            List of dictionaries describing sets with missing cards.
        """
        tcg = TCGdex(self.language)
        card_index = USearchCardIndex(
            index_path=self.index_path,
            meta_path=self.meta_path,
            view_only=True,
        )

        all_sets = await tcg.set.list()
        sets_with_new_cards = []

        for s in all_sets:
            set_id = s.id
            card_count = getattr(s, "cardCount", None)
            total_expected = card_count.total if card_count and hasattr(card_count, "total") else 0

            cursor = card_index._db_conn.execute(
                """
                SELECT 
                    (SELECT COUNT(*) FROM card_metadata WHERE set_id = ?) +
                    (SELECT COUNT(*) FROM unindexable_cards WHERE tcgdex_id LIKE ? || '-%')
                """,
                (set_id, set_id),
            )
            local_total = cursor.fetchone()[0]

            if local_total < total_expected or local_total == 0:
                sets_with_new_cards.append(
                    {
                        "set_id": set_id,
                        "set_name": getattr(s, "name", set_id),
                        "local_count": local_total,
                        "total_expected": total_expected,
                    }
                )

        return sets_with_new_cards

    async def sync(self) -> SyncSummary:
        """Runs an incremental sync against TCGdex.

        Delegates to the unified build_index pipeline for missing sets/cards.
        """
        t_start = time.perf_counter()
        logger.info(f"Starting daily catalog sync (lang='{self.language}')...")

        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        card_index = USearchCardIndex(
            index_path=self.index_path,
            meta_path=self.meta_path,
            view_only=True,
        )
        before_count = len(card_index)
        sets_to_check = await self.check_for_new_cards()

        if not sets_to_check:
            elapsed = time.perf_counter() - t_start
            logger.info(f"Catalog is completely up to date! (0 new cards, checked in {elapsed:.2f}s).")
            return SyncSummary(
                updated=False,
                new_cards_added=0,
                new_sets_detected=0,
                total_cards_indexed=before_count,
                elapsed_seconds=round(elapsed, 2),
            )

        set_ids = [s["set_id"] for s in sets_to_check]
        set_names = [s["set_name"] for s in sets_to_check]
        logger.info(f"Found {len(set_ids)} sets requiring sync: {set_ids[:5]}...")

        # Delegate downloading, embedding, and indexing to unified build_index
        await build_index(
            sets_filter=set_ids,
            index_path=self.index_path,
            cache_dir=self.cache_dir,
            concurrency=self.concurrency,
            lang=self.language,
        )

        card_index_updated = USearchCardIndex(
            index_path=self.index_path,
            meta_path=self.meta_path,
            view_only=True,
        )
        after_count = len(card_index_updated)
        new_cards_added = max(0, after_count - before_count)
        elapsed = time.perf_counter() - t_start

        logger.info(
            f"Sync complete! Added {new_cards_added} new cards across {len(set_names)} sets in {elapsed:.2f}s."
        )

        return SyncSummary(
            updated=new_cards_added > 0,
            new_cards_added=new_cards_added,
            new_sets_detected=len(set_names),
            total_cards_indexed=after_count,
            elapsed_seconds=round(elapsed, 2),
            sets_updated=set_names,
        )

    async def run_daemon(self, interval_hours: float = 24.0) -> None:
        """Runs continuous daily synchronization loop."""
        interval_seconds = interval_hours * 3600
        logger.info(f"Starting CatalogSync daemon. Sync interval: every {interval_hours} hours.")

        while True:
            try:
                summary = await self.sync()
                if summary.updated:
                    logger.info(f"[Daemon Sync] Added {summary.new_cards_added} new cards.")
                else:
                    logger.info("[Daemon Sync] Up-to-date. No new cards detected.")
            except Exception as e:
                logger.error(f"[Daemon Sync Error] {e}", exc_info=True)

            logger.info(f"Sleeping for {interval_hours} hours until next sync cycle...")
            await asyncio.sleep(interval_seconds)
