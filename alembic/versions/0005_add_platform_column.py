"""add platform column to WallapopOffers

Revision ID: 0005_add_platform_column
Revises: 0004_add_alert_status
Create Date: 2026-09-15 16:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0005_add_platform_column"
down_revision: Union[str, None] = "0004_add_alert_status"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("WallapopOffers") as batch_op:
        batch_op.add_column(
            sa.Column(
                "platform",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="wallapop",
            )
        )
        batch_op.create_index(
            batch_op.f("ix_WallapopOffers_platform"), ["platform"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("WallapopOffers") as batch_op:
        batch_op.drop_index(batch_op.f("ix_WallapopOffers_platform"))
        batch_op.drop_column("platform")
