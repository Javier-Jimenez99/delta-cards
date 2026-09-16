"""Database connection and session management for delta-cards."""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.engine import Engine
from ..settings import get_settings

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/delta_cards"


def normalize_db_url(url: str) -> str:
    """Ensures standard PostgreSQL URLs use the psycopg (v3) driver dialect."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_db_url(db_url: Optional[str] = None) -> str:
    """Resolves database URL from argument or centralized application settings."""
    raw_url = db_url or get_settings().database_url or DEFAULT_DATABASE_URL
    return normalize_db_url(raw_url)


from pathlib import Path
from sqlalchemy import event


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Creates a SQLAlchemy engine configured for the resolved database URL."""
    url = get_db_url(db_url)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        connect_args["timeout"] = 30.0
        if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
            db_path = Path(url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url, echo=echo, connect_args=connect_args)

    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

    return engine


@contextmanager
def get_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager yielding a transactional database Session."""
    eng = engine or get_engine()
    with Session(eng) as session:
        yield session


import logging
from sqlalchemy import event, inspect, text

logger = logging.getLogger(__name__)


def init_db(engine: Optional[Engine] = None) -> None:
    """Initializes database schema directly by creating all tables defined in SQLModel metadata.

    Also automatically verifies table migrations (e.g. WallapopOffers -> PlatformOffers)
    and newly introduced columns to existing tables if missing.
    """
    eng = engine or get_engine()

    try:
        insp = inspect(eng)
        table_names = insp.get_table_names()
        if "WallapopOffers" in table_names and "PlatformOffers" not in table_names:
            logger.info("Migrating table 'WallapopOffers' to 'PlatformOffers'...")
            with eng.begin() as conn:
                conn.execute(text('ALTER TABLE "WallapopOffers" RENAME TO "PlatformOffers"'))
                # Ensure indexes exist on PlatformOffers
                for col in ["id", "title", "price", "product_type", "seller_user_id", "search_profile_id", "cardmarket_product_id", "language", "tcgdex_id", "assigned_state", "alert_status", "platform"]:
                    try:
                        conn.execute(text(f'CREATE INDEX IF NOT EXISTS "ix_PlatformOffers_{col}" ON "PlatformOffers" ({col})'))
                    except Exception:
                        pass
    except Exception as exc:
        logger.warning(f"Could not check/migrate WallapopOffers table to PlatformOffers: {exc}")

    SQLModel.metadata.create_all(eng)

    try:
        insp = inspect(eng)
        table_names = insp.get_table_names()
        target_table = "PlatformOffers" if "PlatformOffers" in table_names else ("WallapopOffers" if "WallapopOffers" in table_names else None)
        if target_table:
            existing_cols = {c["name"] for c in insp.get_columns(target_table)}
            with eng.begin() as conn:
                if "platform" not in existing_cols:
                    conn.execute(text(f'ALTER TABLE "{target_table}" ADD COLUMN platform VARCHAR DEFAULT \'wallapop\''))
                    conn.execute(text(f'UPDATE "{target_table}" SET platform = \'wallapop\' WHERE platform IS NULL OR platform = \'\''))
                    try:
                        conn.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{target_table}_platform ON "{target_table}" (platform)'))
                    except Exception:
                        pass
                if "alert_status" not in existing_cols:
                    conn.execute(text(f'ALTER TABLE "{target_table}" ADD COLUMN alert_status VARCHAR DEFAULT \'none\''))
                    try:
                        conn.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{target_table}_alert_status ON "{target_table}" (alert_status)'))
                    except Exception:
                        pass
                if "assigned_state" not in existing_cols:
                    conn.execute(text(f'ALTER TABLE "{target_table}" ADD COLUMN assigned_state VARCHAR DEFAULT \'pending\''))
                    try:
                        conn.execute(text(f'CREATE INDEX IF NOT EXISTS ix_{target_table}_assigned_state ON "{target_table}" (assigned_state)'))
                    except Exception:
                        pass

        if "CardmarketPrice" in table_names:
            cp_cols = {c["name"] for c in insp.get_columns("CardmarketPrice")}
            with eng.begin() as conn:
                if "avg_price" not in cp_cols:
                    conn.execute(text('ALTER TABLE "CardmarketPrice" ADD COLUMN avg_price FLOAT'))
                if "trend_price" not in cp_cols:
                    conn.execute(text('ALTER TABLE "CardmarketPrice" ADD COLUMN trend_price FLOAT'))
    except Exception as exc:
        logger.warning(f"Could not verify/migrate columns in init_db: {exc}")

