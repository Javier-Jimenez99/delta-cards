"""rename WallapopOffers table to PlatformOffers

Revision ID: 0006_rename_wallapop_offers_to_platform_offers
Revises: 0005_add_platform_column
Create Date: 2026-09-15 21:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0006_rename_wallapop_offers_to_platform_offers"
down_revision: Union[str, None] = "0005_add_platform_column"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table("WallapopOffers", "PlatformOffers")
    indexed_columns = [
        "id",
        "title",
        "price",
        "product_type",
        "seller_user_id",
        "search_profile_id",
        "cardmarket_product_id",
        "language",
        "tcgdex_id",
        "assigned_state",
        "alert_status",
        "platform",
    ]
    for col in indexed_columns:
        try:
            op.create_index(f"ix_PlatformOffers_{col}", "PlatformOffers", [col], unique=False)
        except Exception:
            pass


def downgrade() -> None:
    op.rename_table("PlatformOffers", "WallapopOffers")
