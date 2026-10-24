"""Database access repository."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from database.models import EventRecord, StreamSession, SystemConfig

logger = logging.getLogger(__name__)


class EventRepository:
    """CRUD operations for events."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def add(self, event: EventRecord) -> EventRecord:
        with self._session_factory() as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def get_by_id(self, event_id: str) -> Optional[EventRecord]:
        with self._session_factory() as session:
            stmt = select(EventRecord).where(EventRecord.event_id == event_id)
            return session.scalar(stmt)

    def list(self, *, event_type: Optional[str] = None,
              source_id: Optional[str] = None,
              session_id: Optional[str] = None,
              since: Optional[datetime] = None,
              until: Optional[datetime] = None,
              limit: int = 100) -> list[EventRecord]:
        with self._session_factory() as session:
            stmt = select(EventRecord)
            if event_type:
                stmt = stmt.where(EventRecord.event_type == event_type)
            if source_id:
                stmt = stmt.where(EventRecord.source_id == source_id)
            if session_id:
                stmt = stmt.where(EventRecord.session_id == session_id)
            if since:
                stmt = stmt.where(EventRecord.timestamp >= since)
            if until:
                stmt = stmt.where(EventRecord.timestamp <= until)
            stmt = stmt.order_by(desc(EventRecord.timestamp)).limit(limit)
            return list(session.scalars(stmt))

    def count(self, *, event_type: Optional[str] = None,
              source_id: Optional[str] = None) -> int:
        with self._session_factory() as session:
            stmt = select(func.count()).select_from(EventRecord)
            if event_type:
                stmt = stmt.where(EventRecord.event_type == event_type)
            if source_id:
                stmt = stmt.where(EventRecord.source_id == source_id)
            return int(session.scalar(stmt))

    def count_by_type(self) -> dict[str, int]:
        with self._session_factory() as session:
            rows = session.execute(
                select(EventRecord.event_type, func.count())
                .group_by(EventRecord.event_type)
            ).all()
            return {r[0]: int(r[1]) for r in rows}

    def delete_older_than(self, days: int) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self._session_factory() as session:
            stmt = select(EventRecord).where(EventRecord.timestamp < cutoff)
            deleted = 0
            for ev in session.scalars(stmt):
                session.delete(ev)
                deleted += 1
            session.commit()
            return deleted


class StreamSessionRepository:
    """CRUD for stream sessions."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def create(self, session_id: str, source_type: str, source_id: str) -> StreamSession:
        rec = StreamSession(
            session_id=session_id,
            source_type=source_type,
            source_id=source_id,
            status="running",
        )
        with self._session_factory() as session:
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec

    def update(self, session_id: str, **kwargs) -> None:
        with self._session_factory() as session:
            rec = session.get(StreamSession, session_id)
            if rec is None:
                return
            for k, v in kwargs.items():
                setattr(rec, k, v)
            session.commit()

    def end_session(self, session_id: str, frames_processed: int, events_detected: int) -> None:
        self.update(session_id, ended_at=datetime.utcnow(),
                    frames_processed=frames_processed,
                    events_detected=events_detected, status="completed")


class ConfigRepository:
    """Key-value config store."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._session_factory() as session:
            rec = session.get(SystemConfig, key)
            return rec.value if rec else default

    def set(self, key: str, value: str) -> None:
        with self._session_factory() as session:
            rec = session.get(SystemConfig, key)
            if rec is None:
                rec = SystemConfig(key=key, value=value)
                session.add(rec)
            else:
                rec.value = value
            session.commit()

    def all(self) -> dict[str, str]:
        with self._session_factory() as session:
            rows = session.scalars(select(SystemConfig)).all()
            return {r.key: r.value for r in rows}