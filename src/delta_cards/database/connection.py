"""Database connection and session management for delta-cards."""

import os
from contextlib import contextmanager
from typing import Generator, Optional
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine
from sqlalchemy.engine import Engine

# Load environment variables from .env if present
load_dotenv()

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/delta_cards"


def normalize_db_url(url: str) -> str:
    """Ensures standard PostgreSQL URLs use the psycopg (v3) driver dialect."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_db_url(db_url: Optional[str] = None) -> str:
    """Resolves database URL from argument, environment, or default."""
    raw_url = db_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    return normalize_db_url(raw_url)


def get_engine(db_url: Optional[str] = None, echo: bool = False) -> Engine:
    """Creates a SQLAlchemy engine configured for the resolved database URL."""
    url = get_db_url(db_url)
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=echo, connect_args=connect_args)


@contextmanager
def get_session(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """Context manager yielding a transactional database Session."""
    eng = engine or get_engine()
    with Session(eng) as session:
        yield session


def init_db(engine: Optional[Engine] = None) -> None:
    """Initializes database schema directly by creating all tables defined in SQLModel metadata."""
    eng = engine or get_engine()
    SQLModel.metadata.create_all(eng)
