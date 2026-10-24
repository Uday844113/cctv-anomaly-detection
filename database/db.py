"""Database bootstrap."""
from __future__ import annotations

import logging
from pathlib import Path

from config.settings import settings
from database.models import create_db_engine, get_session_factory as _make_session_factory, init_db

logger = logging.getLogger(__name__)

_engine = None
_session_factory = None


def get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first use."""
    global _engine
    if _engine is None:
        db_url = settings.database_url
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
            rel = db_url[len("sqlite:///"):]
            if rel.startswith("./"):
                rel = rel[2:]
            db_path = (settings.data_path / rel).resolve()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        _engine = create_db_engine(db_url)
        init_db(_engine)
        logger.info("Database engine initialized: %s", db_url)
    return _engine


def get_session_factory():
    """Return the singleton session factory, creating it on first use."""
    global _session_factory
    if _session_factory is None:
        _session_factory = _make_session_factory(get_engine())
    return _session_factory


def reset_engine() -> None:
    """Test helper: drop and recreate all tables, reset singletons."""
    global _engine, _session_factory
    if _engine is not None:
        from database.models import Base
        Base.metadata.drop_all(_engine)
        Base.metadata.create_all(_engine)
    _session_factory = None