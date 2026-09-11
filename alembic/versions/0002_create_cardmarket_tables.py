"""create CardmarketProduct and CardmarketPrice tables and link to WallapopOffers

Revision ID: 0002_create_cardmarket_tables
Revises: 0001_create_wallapop_offers
Create Date: 2026-09-11 02:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0002_create_cardmarket_tables"
down_revision: Union[str, None] = "0001_create_wallapop_offers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create CardmarketProduct table
    op.create_table(
        "CardmarketProduct",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("cardmarket_url", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_CardmarketProduct_id"), "CardmarketProduct", ["id"], unique=False)

    # 2. Create CardmarketPrice table
    op.create_table(
        "CardmarketPrice",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("cardmarket_product_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("language", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("min_price_all", sa.Float(), nullable=True),
        sa.Column("min_price_spain", sa.Float(), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["cardmarket_product_id"], ["CardmarketProduct.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("cardmarket_product_id", "language", name="uq_cardmarket_product_language"),
    )
    op.create_index(op.f("ix_CardmarketPrice_cardmarket_product_id"), "CardmarketPrice", ["cardmarket_product_id"], unique=False)
    op.create_index(op.f("ix_CardmarketPrice_language"), "CardmarketPrice", ["language"], unique=False)

    # 3. Add language to WallapopOffers (default 'es')
    op.add_column("WallapopOffers", sa.Column("language", sqlmodel.sql.sqltypes.AutoString(), nullable=True, server_default="es"))
    op.create_index(op.f("ix_WallapopOffers_language"), "WallapopOffers", ["language"], unique=False)
    op.create_foreign_key(
        "fk_WallapopOffers_cardmarket_product_id",
        "WallapopOffers",
        "CardmarketProduct",
        ["cardmarket_product_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_WallapopOffers_cardmarket_product_id", "WallapopOffers", type_="foreignkey")
    op.drop_index(op.f("ix_WallapopOffers_language"), table_name="WallapopOffers")
    op.drop_column("WallapopOffers", "language")
    op.drop_index(op.f("ix_CardmarketPrice_language"), table_name="CardmarketPrice")
    op.drop_index(op.f("ix_CardmarketPrice_cardmarket_product_id"), table_name="CardmarketPrice")
    op.drop_table("CardmarketPrice")
    op.drop_index(op.f("ix_CardmarketProduct_id"), table_name="CardmarketProduct")
    op.drop_table("CardmarketProduct")
