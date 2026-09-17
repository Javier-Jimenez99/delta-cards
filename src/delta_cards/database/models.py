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
    offers: List["PlatformOffers"] = Relationship(back_populates="cardmarket_product")
    specific_offers: List["CardmarketOffers"] = Relationship(back_populates="product")


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
    avg_price: Optional[float] = Field(
        default=None,
        description="Average price (Precio medio) from TCGDex",
    )
    trend_price: Optional[float] = Field(
        default=None,
        description="Trend price from TCGDex",
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
    offers: List["PlatformOffers"] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "and_(CardmarketPrice.cardmarket_product_id == foreign(PlatformOffers.cardmarket_product_id), CardmarketPrice.language == foreign(PlatformOffers.language))",
            "viewonly": True,
        }
    )


class PlatformOffers(SQLModel, table=True):
    """Database table model storing scraped marketplace offer listings."""

    __tablename__ = "PlatformOffers"

    id: str = Field(primary_key=True, index=True, description="Marketplace unique item ID")
    title: str = Field(index=True, description="Title of the listing")
    price: float = Field(index=True, description="Price in currency units")
    currency: str = Field(default="EUR", description="Currency code (e.g. EUR)")
    product_type: str = Field(index=True, description="Product category (booster_box, booster_pack, etc.)")
    platform: str = Field(
        default="wallapop",
        index=True,
        description="Marketplace platform ('wallapop', 'vinted')",
    )
    game: str = Field(
        default="pokemon",
        index=True,
        description="Game identifier ('pokemon', 'magic')",
    )
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
    alert_status: str = Field(
        default="none",
        index=True,
        description="Alert status: 'none', 'pending', 'sent', 'rejected'",
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
            "primaryjoin": "and_(foreign(PlatformOffers.cardmarket_product_id) == CardmarketPrice.cardmarket_product_id, foreign(PlatformOffers.language) == CardmarketPrice.language)",
            "viewonly": True,
        }
    )


# Alias for backward compatibility
WallapopOffers = PlatformOffers


class CardmarketOffers(SQLModel, table=True):
    """Stores individual Cardmarket seller offers for a product."""

    __tablename__ = "CardmarketOffers"

    id: str = Field(
        primary_key=True,
        index=True,
        description="Cardmarket unique article/offer ID (or synthetic composite id)",
    )
    cardmarket_product_id: str = Field(
        foreign_key="CardmarketProduct.id",
        index=True,
        description="Reference to the Cardmarket product ID",
    )
    seller_name: str = Field(index=True, description="Username of the seller")
    seller_country: Optional[str] = Field(default=None, index=True, description="Seller location or country code")
    seller_type: Optional[str] = Field(default=None, description="Seller type (Private, Professional, Powerseller)")
    seller_rating: Optional[str] = Field(default=None, description="Seller evaluation rating")
    seller_sales_count: Optional[int] = Field(default=None, description="Total sales count of the seller")
    price: float = Field(index=True, description="Price in currency units")
    currency: str = Field(default="EUR", description="Currency code (e.g. EUR)")
    condition: Optional[str] = Field(default=None, index=True, description="Condition code (MT, NM, EX, GD, LP, PL, PO)")
    language: Optional[str] = Field(default="es", index=True, description="Card language code (e.g. es, en, de)")
    is_foil: bool = Field(default=False, description="Whether the card is foil/holo")
    is_signed: bool = Field(default=False, description="Whether the card is signed")
    is_playset: bool = Field(default=False, description="Whether this offer is for a 4x playset")
    is_first_edition: bool = Field(default=False, description="Whether the card is first edition")
    is_reverse_holo: bool = Field(default=False, description="Whether the card is reverse holo")
    is_altered: bool = Field(default=False, description="Whether the card is altered")
    item_count: int = Field(default=1, description="Available quantity in this offer")
    comment: Optional[str] = Field(default=None, description="Seller comment or notes on the item")
    offer_url: Optional[str] = Field(default=None, description="Direct URL to product or article")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was scraped",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the offer was last updated",
    )

    # Relationships
    product: Optional[CardmarketProduct] = Relationship(back_populates="specific_offers")
