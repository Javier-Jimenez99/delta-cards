"""create WallapopOffers table

Revision ID: 0001_create_wallapop_offers
Revises: 
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0001_create_wallapop_offers"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "WallapopOffers",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("title", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="EUR"),
        sa.Column("product_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("location", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("country", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("description", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("link", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("image_url", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("is_reserved", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("shipping_available", sa.Boolean(), nullable=True),
        sa.Column("seller_user_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("seller_link", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("published_date", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("search_profile_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("cardmarket_product_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("tcgdex_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_WallapopOffers_id"), "WallapopOffers", ["id"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_title"), "WallapopOffers", ["title"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_price"), "WallapopOffers", ["price"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_product_type"), "WallapopOffers", ["product_type"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_seller_user_id"), "WallapopOffers", ["seller_user_id"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_search_profile_id"), "WallapopOffers", ["search_profile_id"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_cardmarket_product_id"), "WallapopOffers", ["cardmarket_product_id"], unique=False)
    op.create_index(op.f("ix_WallapopOffers_tcgdex_id"), "WallapopOffers", ["tcgdex_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_WallapopOffers_tcgdex_id"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_cardmarket_product_id"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_search_profile_id"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_seller_user_id"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_product_type"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_price"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_title"), table_name="WallapopOffers")
    op.drop_index(op.f("ix_WallapopOffers_id"), table_name="WallapopOffers")
    op.drop_table("WallapopOffers")
