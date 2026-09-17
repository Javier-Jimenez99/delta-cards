"""add game column to PlatformOffers

Revision ID: 0007_add_game_column
Revises: 0006_rename_wallapop_offers_to_platform_offers
Create Date: 2026-09-17 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0007_add_game_column"
down_revision: Union[str, None] = "0006_rename_wallapop_offers_to_platform_offers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("PlatformOffers") as batch_op:
        batch_op.add_column(
            sa.Column(
                "game",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="pokemon",
            )
        )
        batch_op.create_index(
            batch_op.f("ix_PlatformOffers_game"), ["game"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("PlatformOffers") as batch_op:
        batch_op.drop_index(batch_op.f("ix_PlatformOffers_game"))
        batch_op.drop_column("game")
