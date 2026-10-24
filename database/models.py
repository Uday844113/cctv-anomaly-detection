"""SQLite database models for event storage."""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class EventRecord(Base):
    """A confirmed anomaly event."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, unique=True, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    video_time = Column(Float, nullable=True, index=True)
    session_id = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=False)
    source_id = Column(String, nullable=False, index=True)
    track_id = Column(Integer, nullable=True, index=True)
    bbox_x1 = Column(Float, nullable=True)
    bbox_y1 = Column(Float, nullable=True)
    bbox_x2 = Column(Float, nullable=True)
    bbox_y2 = Column(Float, nullable=True)
    confirmation_state = Column(String, default="confirmed")
    snapshot_path = Column(Text, nullable=True)
    incident_clip_path = Column(Text, nullable=True)
    evidence = Column(Text, nullable=True)  # JSON string
    model_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StreamSession(Base):
    """A live or recorded processing session."""

    __tablename__ = "stream_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, nullable=False, index=True)
    source_type = Column(String, nullable=False)
    source_id = Column(String, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    frames_processed = Column(Integer, default=0)
    events_detected = Column(Integer, default=0)
    status = Column(String, default="running")


class SystemConfig(Base):
    """Key-value store for runtime configuration."""

    __tablename__ = "system_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, unique=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_db_engine(database_url: str):
    """Create a SQLAlchemy engine for the given SQLite URL."""
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    else:
        connect_args = {}
    return create_engine(database_url, connect_args=connect_args)


def get_session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(engine) -> None:
    """Create all tables."""
    Base.metadata.create_all(engine)