"""Database models for delta-cards using SQLModel (Pydantic + SQLAlchemy)."""

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


class WallapopOffers(SQLModel, table=True):
    """PostgreSQL table model storing scraped Wallapop offer listings."""

    __tablename__ = "WallapopOffers"

    id: str = Field(primary_key=True, index=True, description="Wallapop unique item ID")
    title: str = Field(index=True, description="Title of the listing")
    price: float = Field(index=True, description="Price in currency units")
    currency: str = Field(default="EUR", description="Currency code (e.g. EUR)")
    product_type: str = Field(index=True, description="Product category (booster_box, booster_pack, etc.)")
    location: Optional[str] = Field(default=None, description="City/locality of seller")
    country: Optional[str] = Field(default=None, description="Country or state code")
    description: Optional[str] = Field(default=None, description="Full item description")
    link: str = Field(description="Direct URL to listing")
    image_url: Optional[str] = Field(default=None, description="URL of main product image")
    is_reserved: bool = Field(default=False, description="Whether the item is currently reserved")
    shipping_available: Optional[bool] = Field(default=None, description="Whether shipping is enabled")
    seller_user_id: Optional[str] = Field(default=None, index=True, description="Seller profile ID")
    seller_link: Optional[str] = Field(default=None, description="URL to seller profile")
    published_date: Optional[str] = Field(default=None, description="Listing publication date ISO string")
    search_profile_id: Optional[str] = Field(default=None, index=True, description="Search profile used")
    cardmarket_product_id: Optional[str] = Field(
        default=None, index=True, description="Card Market product ID"
    )
    tcgdex_id: Optional[str] = Field(
        default=None, index=True, description="TCGDex card/product ID"
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was first scraped",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was last updated",
    )

