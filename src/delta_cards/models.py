"""Data models for card offers search configuration and scraped listings."""

import re
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Products that MUST NOT have TCGDex IDs assigned
EXCLUDED_PRODUCT_TYPES = {
    "booster_box",
    "booster_pack",
    "deck",
    "etb",
    "elite_trainer_box",
}

LOT_PATTERNS = [
    r"\blote\s+(\d+|de)\b",
    r"\blotto\s+(\d+|di)\b",
    r"\bbulk\b",
    r"\bmistery\s+box\b",
    r"\bmystery\s+box\b",
    r"\bbooster\s+box\b",
    r"\bultra\s+premium\s+collection\b",
    r"\bupc\b",
    r"\bgift\s+box\b",
    r"\b\d+\s+carte\b",
    r"^lote\b",
    r"^lotto\b",
    r"^bulk\b",
    r"^mazo\b",
]


def is_card_offer(
    product_type: Optional[str],
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> bool:
    """Verifies whether an offer represents a single card listing.

    Strictly excludes booster boxes, booster packs, displays, full decks, ETBs,
    and bulk lots / mystery boxes.
    """
    if not product_type:
        return False
    pt = product_type.lower().strip()
    if pt in EXCLUDED_PRODUCT_TYPES:
        return False
    if any(sealed_term in pt for sealed_term in ["box", "pack", "deck", "etb", "sobre", "caja", "display"]):
        return False
    if pt not in ("single_card", "card"):
        return False

    # Check title for clear non-single-card listings (lots, bulk, mystery boxes, decks)
    if title:
        t_clean = title.strip().lower()
        if any(re.search(pat, t_clean) for pat in LOT_PATTERNS):
            return False

    return True


class WallapopSettings(BaseModel):
    """General connection and geolocation settings for Wallapop."""
    latitude: float = 40.4168
    longitude: float = -3.7038
    location_name: Optional[str] = "Madrid, Spain"
    order_by: str = "newest"
    delay_between_requests_ms: int = 1500


class SearchProfile(BaseModel):
    """Configuration for a product search profile supporting multiple query variations."""
    id: str
    name: str
    product_type: str
    query: Union[str, List[str]]
    keywords: List[str] = Field(default_factory=list)
    excluded_keywords: List[str] = Field(default_factory=list)
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_items: int = 50
    enabled: bool = True

    @property
    def queries(self) -> List[str]:
        """Returns a normalized list of query strings."""
        if isinstance(self.query, list):
            return [q.strip() for q in self.query if q and q.strip()]
        if isinstance(self.query, str) and self.query.strip():
            return [self.query.strip()]
        return []


class OutputSettings(BaseModel):
    """Output configuration for saving scraped offers."""
    output_dir: str = "data/output"
    save_json: bool = True
    save_csv: bool = True
    save_db: bool = True
    filename_prefix: str = "pokemon_offers"


class SearchConfig(BaseModel):
    """Root configuration object loaded from the JSON file."""
    name: str = "Card Search Config"
    description: Optional[str] = None
    wallapop_settings: WallapopSettings = Field(default_factory=WallapopSettings)
    search_profiles: List[SearchProfile] = Field(default_factory=list)
    output_settings: OutputSettings = Field(default_factory=OutputSettings)


class OfferListing(BaseModel):
    """Normalized data model for a single offer listing found on a marketplace."""
    id: str
    title: str
    price: float
    currency: str = "EUR"
    product_type: str
    location: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None
    link: str
    image_url: Optional[str] = None
    is_reserved: bool = False
    shipping_available: Optional[bool] = None
    seller_user_id: Optional[str] = None
    seller_link: Optional[str] = None
    published_date: Optional[str] = None
    search_profile_id: Optional[str] = None
    cardmarket_product_id: Optional[str] = None
    language: Optional[str] = "es"
    cardmarket_url: Optional[str] = None
    cardmarket_low_price: Optional[float] = None
    cardmarket_avg_price: Optional[float] = None
    cardmarket_trend_price: Optional[float] = None
    cardmarket_price_spain: Optional[float] = None
    cardmarket_price_all: Optional[float] = None
    tcgdex_id: Optional[str] = None
    assigned_state: str = "pending"
    alert_status: str = "none"

