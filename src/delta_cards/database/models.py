"""Database models for delta-cards using SQLModel (Pydantic + SQLAlchemy)."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import foreign
from sqlmodel import SQLModel, Field, Relationship


class CardmarketProduct(SQLModel, table=True):
    """Stores Cardmarket product entity metadata and canonical URL."""

    __tablename__ = "CardmarketProduct"

    id: str = Field(
        primary_key=True,
        index=True,
        description="Cardmarket unique product ID (e.g. '483559')",
    )
    cardmarket_url: str = Field(
        description="Direct or canonical Cardmarket product URL"
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the product was first recorded",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the product was last updated",
    )

    # Relationships
    prices: List["CardmarketPrice"] = Relationship(back_populates="product")
    offers: List["WallapopOffers"] = Relationship(back_populates="cardmarket_product")


class CardmarketPrice(SQLModel, table=True):
    """Stores Cardmarket price for a product in a specific language."""

    __tablename__ = "CardmarketPrice"
    __table_args__ = (
        UniqueConstraint(
            "cardmarket_product_id",
            "language",
            name="uq_cardmarket_product_language",
        ),
    )

    id: Optional[int] = Field(
        default=None,
        primary_key=True,
        description="Internal price record surrogate ID",
    )
    cardmarket_product_id: str = Field(
        foreign_key="CardmarketProduct.id",
        index=True,
        description="Reference to the Cardmarket product ID",
    )
    language: str = Field(
        index=True,
        description="Language code or identifier (e.g. 'es', 'en')",
    )
    min_price_all: Optional[float] = Field(
        default=None,
        description="Minimum price among all sellers for this language",
    )
    min_price_spain: Optional[float] = Field(
        default=None,
        description="Minimum price from Spanish sellers (sellerCountry=10) for this language",
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the price was first recorded",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the price was last updated",
    )

    # Relationships
    product: Optional[CardmarketProduct] = Relationship(back_populates="prices")
    offers: List["WallapopOffers"] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "and_(CardmarketPrice.cardmarket_product_id == foreign(WallapopOffers.cardmarket_product_id), CardmarketPrice.language == foreign(WallapopOffers.language))",
            "viewonly": True,
        }
    )


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
        default=None,
        foreign_key="CardmarketProduct.id",
        index=True,
        description="Cardmarket product ID foreign key",
    )
    language: Optional[str] = Field(
        default="es",
        index=True,
        description="Card language code for the listing (e.g. 'es', 'en')",
    )
    tcgdex_id: Optional[str] = Field(
        default=None, index=True, description="TCGDex card/product ID"
    )
    assigned_state: str = Field(
        default="pending",
        index=True,
        description="Cardmarket assignment state: 'pending', 'assigned', 'not_assigned'",
    )
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was first scraped",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was last updated",
    )

    # Relationships
    cardmarket_product: Optional[CardmarketProduct] = Relationship(back_populates="offers")
    cardmarket_price: Optional[CardmarketPrice] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "and_(foreign(WallapopOffers.cardmarket_product_id) == CardmarketPrice.cardmarket_product_id, foreign(WallapopOffers.language) == CardmarketPrice.language)",
            "viewonly": True,
        }
    )
