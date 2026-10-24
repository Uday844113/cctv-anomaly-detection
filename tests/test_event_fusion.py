"""Tests for event fusion and database."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from detection.event_fusion import EventFusion, FusionConfig
from detection.person_detector import PersonDetector
from models.model_manager import ModelManager
from database.db import get_session_factory, reset_engine
from database.repository import EventRepository
from database.models import EventRecord
from datetime import datetime


class _StubDetector:
    def detect_and_track(self, frame, ts):
        return []

    @property
    def active_tracks(self):
        return {}


def test_fusion_no_tracks_no_events():
    mm = ModelManager(Path(tempfile.mkdtemp()))
    fusion = EventFusion(person_detector=_StubDetector(), model_manager=mm)
    fusion._detectors = {}
    from video.video_source import FramePacket
    import numpy as np
    pkt = FramePacket(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_index=1,
                     source_id="t", timestamp=1.0)
    events = fusion.process_frame(pkt)
    assert events == []


def test_database_insert_and_query(tmp_path, monkeypatch):
    from database import db as dbmod
    monkeypatch.setattr(dbmod, "_engine", None)
    monkeypatch.setattr(dbmod, "_session_factory", None)
    monkeypatch.setattr(dbmod.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(dbmod.settings, "data_dir", str(tmp_path))
    engine = dbmod.get_engine()
    sf = dbmod.get_session_factory()
    repo = EventRepository(sf)
    rec = EventRecord(
        event_id="ev1", event_type="fire", timestamp=datetime.utcnow(),
        confidence=0.9, source_id="cam1",
    )
    repo.add(rec)
    fetched = repo.get_by_id("ev1")
    assert fetched is not None
    assert fetched.event_type == "fire"
    assert repo.count() == 1
    by_type = repo.count_by_type()
    assert by_type.get("fire") == 1


def test_database_reset(tmp_path, monkeypatch):
    from database import db as dbmod
    monkeypatch.setattr(dbmod, "_engine", None)
    monkeypatch.setattr(dbmod, "_session_factory", None)
    monkeypatch.setattr(dbmod.settings, "database_url", f"sqlite:///{tmp_path}/r.db")
    monkeypatch.setattr(dbmod.settings, "data_dir", str(tmp_path))
    engine = dbmod.get_engine()
    sf = dbmod.get_session_factory()
    repo = EventRepository(sf)
    repo.add(EventRecord(event_id="x", event_type="fall",
                         timestamp=datetime.utcnow(), confidence=0.5, source_id="s"))
    assert repo.count() == 1
    dbmod.reset_engine()
    # After reset, the engine is recreated and tables are empty
    sf2 = dbmod.get_session_factory()
    repo2 = EventRepository(sf2)
    assert repo2.count() == 0