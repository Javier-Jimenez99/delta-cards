"""Cardmarket price guide loader and indexer for Magic: The Gathering.

Loads official Cardmarket price guide JSON catalogs (e.g. data/catalogs/magic/price_guide.json)
into an in-memory index for fast O(1) lookups of low, avg, and trend prices by product ID.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class CardmarketPriceGuide:
    """In-memory indexer for Cardmarket price guide catalogs."""

    _instances: Dict[str, "CardmarketPriceGuide"] = {}

    def __init__(self, game: str = "magic", catalog_path: Optional[Union[str, Path]] = None):
        self.game = game.lower().strip()
        if catalog_path is not None:
            self.catalog_path = Path(catalog_path)
        else:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            candidates = [
                project_root / f"data/catalogs/{self.game}/price_guide.json",
                project_root / f"data/catalogs/{self.game}/price_guide_{self.game}.json",
                project_root / f"data/catalogs/price_guide_{self.game}.json",
            ]
            self.catalog_path = next((c for c in candidates if c.exists()), candidates[0])

        self._prices: Dict[int, Dict[str, Optional[float]]] = {}
        self._loaded: bool = False

    @classmethod
    def get_instance(
        cls, game: str = "magic", catalog_path: Optional[Union[str, Path]] = None
    ) -> "CardmarketPriceGuide":
        """Returns singleton instance for the specified game."""
        g = "magic" if str(game).lower().strip() in ("magic", "mtg") else "pokemon"
        if catalog_path is not None:
            path_obj = Path(catalog_path)
            if g not in cls._instances or cls._instances[g].catalog_path != path_obj:
                cls._instances[g] = cls(game=g, catalog_path=path_obj)
        elif g not in cls._instances:
            cls._instances[g] = cls(game=g)
        return cls._instances[g]

    def reload(self) -> None:
        """Forces reloading the price guide catalog from disk."""
        self._loaded = False
        self._prices.clear()
        self._ensure_loaded()

    @classmethod
    def reload_all(cls) -> None:
        """Forces reloading all cached instances."""
        for inst in list(cls._instances.values()):
            inst.reload()

    def _ensure_loaded(self) -> None:
        """Lazily reads and indexes the price guide JSON if not already loaded."""
        if self._loaded:
            return
        self._loaded = True

        if not self.catalog_path.exists():
            logger.warning("Cardmarket price guide not found at %s", self.catalog_path)
            return

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            guide_list = data.get("priceGuides") if isinstance(data, dict) else data
            if isinstance(guide_list, list):
                for item in guide_list:
                    if not isinstance(item, dict):
                        continue
                    pid = item.get("idProduct")
                    if pid is None:
                        continue
                    try:
                        int_pid = int(pid)
                        self._prices[int_pid] = {
                            "avg": item.get("avg"),
                            "low": item.get("low"),
                            "trend": item.get("trend"),
                            "avg_foil": item.get("avg-foil"),
                            "low_foil": item.get("low-foil"),
                            "trend_foil": item.get("trend-foil"),
                        }
                    except (ValueError, TypeError):
                        continue

            logger.info(
                "CardmarketPriceGuide (%s) loaded %d product prices from %s",
                self.game,
                len(self._prices),
                self.catalog_path,
            )
        except Exception as exc:
            logger.error("Failed to load Cardmarket price guide from %s: %s", self.catalog_path, exc)

    def get_prices(
        self, id_product: Union[int, str, None], is_foil: Optional[bool] = None
    ) -> Optional[Dict[str, Optional[float]]]:
        """Looks up low, avg, and trend prices for a given Cardmarket product ID.

        Args:
            id_product: Cardmarket product ID (integer or string representation).
            is_foil: True if foil version requested, False if regular, None if unspecified.

        Returns:
            Dict with 'low', 'avg', 'trend' float values (or None if missing),
            or None if the product ID is not found in the price guide.
        """
        if id_product is None:
            return None

        self._ensure_loaded()

        try:
            pid = int(id_product)
        except (ValueError, TypeError):
            return None

        entry = self._prices.get(pid)
        if not entry:
            return None

        if is_foil is True:
            low = entry.get("low_foil") if entry.get("low_foil") is not None else entry.get("low")
            avg = entry.get("avg_foil") if entry.get("avg_foil") is not None else entry.get("avg")
            trend = entry.get("trend_foil") if entry.get("trend_foil") is not None else entry.get("trend")
        else:
            low = entry.get("low") if entry.get("low") is not None else entry.get("low_foil")
            avg = entry.get("avg") if entry.get("avg") is not None else entry.get("avg_foil")
            trend = entry.get("trend") if entry.get("trend") is not None else entry.get("trend_foil")

        return {
            "low": low,
            "avg": avg,
            "trend": trend,
        }
