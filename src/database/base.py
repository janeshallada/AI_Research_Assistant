"""
SQLAlchemy engine/session bootstrap. Uses SQLite by default (file configured
via settings.sqlite_db_path). Swapping to PostgreSQL only requires changing
the connection URL — no other code changes are needed since all queries go
through the ORM.
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config.settings import settings

os.makedirs(os.path.dirname(settings.sqlite_db_path) or ".", exist_ok=True)

DATABASE_URL = f"sqlite:///{settings.sqlite_db_path}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create all tables. Called once on application startup."""
    from src.database import models  # noqa: F401  (ensures models are registered)
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """Context-manager version for use outside request handlers (e.g. background tasks)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
