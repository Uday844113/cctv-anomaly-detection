"""Pydantic schemas for the REST API."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    fight = "fight"
    fire = "fire"
    smoke = "smoke"
    fall = "fall"
    crowd_violence = "crowd_violence"
    abnormal_running = "abnormal_running"


class EventResponse(BaseModel):
    event_id: str
    event_type: str
    timestamp: datetime
    confidence: float
    source_id: str
    track_id: Optional[int] = None
    bbox: Optional[list[float]] = None
    confirmation_state: str = "confirmed"
    snapshot_path: Optional[str] = None
    incident_clip_path: Optional[str] = None
    evidence: dict = Field(default_factory=dict)
    model_version: Optional[str] = None


class EventListResponse(BaseModel):
    total: int
    events: list[EventResponse]


class StreamStartRequest(BaseModel):
    source_type: str  # file | webcam | rtsp
    source_value: str
    target_fps: Optional[float] = None
    frame_skip: Optional[int] = None
    thresholds: Optional[dict] = None


class StreamStatusResponse(BaseModel):
    running: bool
    source_id: str
    source_type: str
    frames_read: int = 0
    frames_processed: int = 0
    processing_fps: float = 0.0
    active_tracks: int = 0
    last_error: str = ""


class ModelStatusResponse(BaseModel):
    person_model_loaded: bool = False
    fire_smoke_model_loaded: bool = False
    fall_model_loaded: bool = False
    fight_model_loaded: bool = False
    backend: str = "ultralytics"
    models_dir: str = ""
    available: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    models_ready: bool
    database_ok: bool
    uptime_seconds: float


class ConfigResponse(BaseModel):
    config: dict


class ConfigUpdateRequest(BaseModel):
    config: dict


class MetricsResponse(BaseModel):
    cpu_percent: float = 0.0
    processing_fps: float = 0.0
    inference_time_ms: float = 0.0
    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    memory_mb: float = 0.0
    active_tracks: int = 0
    recent_events: int = 0