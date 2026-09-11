"""create CardmarketOffers table

Revision ID: 0003_create_cardmarket_offers
Revises: 0002_create_cardmarket_tables
Create Date: 2026-09-11 22:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0003_create_cardmarket_offers"
down_revision: Union[str, None] = "0002_create_cardmarket_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "CardmarketOffers",
        sa.Column("id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("cardmarket_product_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("seller_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("seller_country", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("seller_type", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("seller_rating", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("seller_sales_count", sa.Integer(), nullable=True),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("currency", sqlmodel.sql.sqltypes.AutoString(), nullable=False, server_default="EUR"),
        sa.Column("condition", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("language", sqlmodel.sql.sqltypes.AutoString(), nullable=True, server_default="es"),
        sa.Column("is_foil", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_signed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_playset", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_first_edition", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_reverse_holo", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("is_altered", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("comment", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("offer_url", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["cardmarket_product_id"], ["CardmarketProduct.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_CardmarketOffers_id"), "CardmarketOffers", ["id"], unique=False)
    op.create_index(op.f("ix_CardmarketOffers_cardmarket_product_id"), "CardmarketOffers", ["cardmarket_product_id"], unique=False)
    op.create_index(op.f("ix_CardmarketOffers_seller_name"), "CardmarketOffers", ["seller_name"], unique=False)
    op.create_index(op.f("ix_CardmarketOffers_price"), "CardmarketOffers", ["price"], unique=False)
    op.create_index(op.f("ix_CardmarketOffers_condition"), "CardmarketOffers", ["condition"], unique=False)
    op.create_index(op.f("ix_CardmarketOffers_language"), "CardmarketOffers", ["language"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_CardmarketOffers_language"), table_name="CardmarketOffers")
    op.drop_index(op.f("ix_CardmarketOffers_condition"), table_name="CardmarketOffers")
    op.drop_index(op.f("ix_CardmarketOffers_price"), table_name="CardmarketOffers")
    op.drop_index(op.f("ix_CardmarketOffers_seller_name"), table_name="CardmarketOffers")
    op.drop_index(op.f("ix_CardmarketOffers_cardmarket_product_id"), table_name="CardmarketOffers")
    op.drop_index(op.f("ix_CardmarketOffers_id"), table_name="CardmarketOffers")
    op.drop_table("CardmarketOffers")
