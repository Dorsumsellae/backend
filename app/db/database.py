"""Connexion et session SQLAlchemy vers la base applicative."""

from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Base declarative commune a tous les modeles."""


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Moteur SQLAlchemy (cree paresseusement, ne se connecte pas a l'import).

    `pool_pre_ping` recycle les connexions mortes (ex. redemarrage de Postgres).
    """
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def session_scope() -> Session:
    """Session transactionnelle : commit si succes, rollback si exception, close au final."""
    session = _session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """Cree les tables manquantes. Appele au demarrage (best-effort)."""
    from app.db import models  # noqa: F401 — enregistre les modeles sur Base

    Base.metadata.create_all(bind=get_engine())
