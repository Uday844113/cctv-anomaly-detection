"""Business logic services."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import settings
from database.db import get_session_factory
from database.models import EventRecord
from database.repository import EventRepository, StreamSessionRepository
from detection._base import EventCandidate
from detection.event_fusion import EventFusion, FusionConfig
from models.model_manager import ModelManager
from recording.incident_recorder import IncidentRecorder
from recording.report_exporter import ReportExporter
from video.frame_processor import FrameProcessor, ProcessingStats
from video.video_source import VideoSource, open_video_source

logger = logging.getLogger(__name__)


class PipelineService:
    """Orchestrates a single video processing pipeline."""

    def __init__(self, source: VideoSource, *,
                 model_manager: ModelManager,
                 on_event=None) -> None:
        self.source = source
        self.model_manager = model_manager
        self._proxy = PersonDetectorProxy(model_manager)
        self.fusion = EventFusion(
            person_detector=self._proxy,
            model_manager=model_manager,
        )
        self.recorder = IncidentRecorder(
            settings.data_path / "incidents",
            pre_event_frames=int(settings.pre_event_seconds * (settings.target_fps or 15)),
            post_event_frames=int(settings.post_event_seconds * (settings.target_fps or 15)),
            fps=settings.target_fps or 15.0,
        )
        self.exporter = ReportExporter(settings.data_path / "incidents")
        self.repo = EventRepository(get_session_factory())
        self.stream_repo = StreamSessionRepository(get_session_factory())
        self.session_id = str(uuid.uuid4())[:8]
        self.fusion.set_on_event(self._on_event)
        self.processor = FrameProcessor(
            source,
            frame_skip=settings.frame_skip,
            on_frame=self._on_frame,
        )
        self._stats = ProcessingStats()
        self._events: deque[EventCandidate] = deque(maxlen=100)
        self._started = time.monotonic()
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()

    def _on_frame(self, packet) -> None:
        # Push raw frame to recorder buffer
        try:
            self.recorder.push_frame(packet.frame, packet.timestamp)
        except Exception:
            pass
        # Cache latest frame for dashboard display
        try:
            with self._latest_frame_lock:
                self._latest_frame = packet.frame
        except Exception:
            pass
        # Run inference
        new_events = self.fusion.process_frame(packet)
        for ev in new_events:
            self._events.append(ev)

    def _on_event(self, event: EventCandidate) -> None:
        # Record incident
        self.recorder.on_event(event)
        # Persist to DB with real UTC timestamp and active session id
        ev_id = f"{event.event_type}_{event.frame_index}_{uuid.uuid4().hex[:6]}"
        rec = EventRecord(
            event_id=ev_id,
            event_type=event.event_type,
            timestamp=_dt_from_ts(time.time()),
            video_time=float(event.timestamp),
            session_id=self.session_id,
            confidence=event.confidence,
            source_id=event.source_id,
            track_id=event.track_id,
            bbox_x1=event.bbox[0] if event.bbox else None,
            bbox_y1=event.bbox[1] if event.bbox else None,
            bbox_x2=event.bbox[2] if event.bbox else None,
            bbox_y2=event.bbox[3] if event.bbox else None,
            confirmation_state="confirmed",
            snapshot_path=event.evidence.get("snapshot_path"),
            incident_clip_path=event.evidence.get("incident_clip_path"),
            evidence=_json_safe(event.evidence),
            model_version="heuristic+v1",
        )
        try:
            self.repo.add(rec)
        except Exception as exc:
            logger.error("Failed to persist event: %s", exc)

    def start(self) -> None:
        if not self.source.open():
            raise RuntimeError(f"Failed to open source: {self.source.source_id}")
        self.stream_repo.create(self.session_id, type(self.source).__name__, self.source.source_id)
        # Pre-load the person-detection model BEFORE starting the processor
        # thread.  The first on_frame callback triggers lazy model loading,
        # which takes ~8s on CPU; doing it here keeps the processor thread
        # responsive and prevents the bounded queue from filling with frames
        # that are then dropped while inference is blocked.
        try:
            self._proxy.detector.is_available()
        except Exception as exc:
            logger.warning("Person model pre-load failed: %s", exc)
        self.processor.start()

    def stop(self) -> None:
        self.processor.stop()
        self.source.close()
        self.recorder.shutdown()
        frames = self.processor.stats.frames_processed
        events = len(self._events)
        self.stream_repo.end_session(self.session_id, frames, events)

    @property
    def stats(self) -> ProcessingStats:
        return self.processor.stats

    @property
    def recent_events(self) -> list[EventCandidate]:
        return list(self._events)

    @property
    def active_tracks(self) -> dict:
        return self.fusion.active_tracks

    @property
    def latest_frame(self):
        """Return the most recent frame (BGR ndarray) or None."""
        with self._latest_frame_lock:
            return self._latest_frame


class PersonDetectorProxy:
    """Thin wrapper so EventFusion can be constructed without a real model.

    Delegates to a lazily-created PersonDetector on first use.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        from detection.person_detector import PersonDetector
        self._model_manager = model_manager
        self._detector: Optional[PersonDetector] = None

    @property
    def detector(self):
        if self._detector is None:
            from detection.person_detector import PersonDetector
            self._detector = PersonDetector(self._model_manager)
        return self._detector

    def detect_and_track(self, frame, timestamp):
        return self.detector.detect_and_track(frame, timestamp)

    @property
    def active_tracks(self):
        if self._detector is None:
            return {}
        return self._detector.active_tracks


class ServiceContainer:
    """Singleton-ish container for shared services."""

    def __init__(self) -> None:
        self.model_manager = ModelManager(settings.models_path)
        self.repo = EventRepository(get_session_factory())
        self.stream_repo = StreamSessionRepository(get_session_factory())
        self.exporter = ReportExporter(settings.data_path / "incidents")
        self._pipelines: dict[str, PipelineService] = {}
        self._active: Optional[str] = None

    def start_stream(self, source_type: str, source_value: str, **kwargs) -> str:
        source = open_video_source(source_type, source_value, **kwargs)
        pipe = PipelineService(source, model_manager=self.model_manager)
        pipe.start()
        key = str(uuid.uuid4())[:8]
        self._pipelines[key] = pipe
        self._active = key
        return key

    def stop_stream(self, key: Optional[str] = None) -> None:
        key = key or self._active
        if not key or key not in self._pipelines:
            return
        pipe = self._pipelines.pop(key)
        try:
            pipe.stop()
        except Exception:
            logger.exception("Error stopping pipeline")
        if self._active == key:
            self._active = None

    def get_pipeline(self, key: Optional[str] = None) -> Optional[PipelineService]:
        key = key or self._active
        return self._pipelines.get(key) if key else None

    def list_events(self, **kwargs) -> list[EventRecord]:
        return self.repo.list(**kwargs)

    def get_event(self, event_id: str) -> Optional[EventRecord]:
        return self.repo.get_by_id(event_id)

    def export_events(self, fmt: str = "csv", **kwargs) -> Path:
        events = self.list_events(**kwargs)
        if fmt == "json":
            return self.exporter.export_json(events)
        return self.exporter.export_csv(events)

    def shutdown(self) -> None:
        for key in list(self._pipelines):
            self.stop_stream(key)
        self.model_manager.clear()


def _dt_from_ts(ts: float):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _json_safe(d: dict) -> str:
    import json
    return json.dumps(d, default=str)