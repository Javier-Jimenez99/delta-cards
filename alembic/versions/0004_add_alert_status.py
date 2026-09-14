"""add alert_status to WallapopOffers

Revision ID: 0004_add_alert_status
Revises: 0003_create_cardmarket_offers
Create Date: 2026-09-14 23:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# revision identifiers, used by Alembic.
revision: str = "0004_add_alert_status"
down_revision: Union[str, None] = "0003_create_cardmarket_offers"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("WallapopOffers") as batch_op:
        batch_op.add_column(
            sa.Column(
                "alert_status",
                sqlmodel.sql.sqltypes.AutoString(),
                nullable=False,
                server_default="none",
            )
        )
        batch_op.create_index(
            batch_op.f("ix_WallapopOffers_alert_status"), ["alert_status"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("WallapopOffers") as batch_op:
        batch_op.drop_index(batch_op.f("ix_WallapopOffers_alert_status"))
        batch_op.drop_column("alert_status")
