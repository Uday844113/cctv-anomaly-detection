"""Centralized configuration for the CCTV anomaly detection system."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_TRUE = {"1", "true", "yes", "y", "on", "t"}
_FALSE = {"0", "false", "no", "n", "off", "f", ""}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    # Unknown strings default to False rather than raising, so that
    # environment variables like DEBUG=release do not crash the app.
    return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "CCTV Anomaly Detection"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Paths ---
    models_dir: str = ""
    data_dir: str = ""
    logs_dir: str = ""
    database_url: str = "sqlite:///./data/cctv_events.db"

    # --- Inference ---
    inference_width: int = 320
    inference_height: int = 320
    frame_skip: int = 1
    target_fps: int = 15
    person_conf: float = 0.25
    fire_smoke_conf: float = 0.25
    fall_conf: float = 0.6
    fight_conf: float = 0.6
    crowd_violence_conf: float = 0.6
    running_conf: float = 0.6

    # --- Model paths ---
    person_model: str = "yolo11n.pt"
    fire_smoke_model: str = "runs/detect/fire_smoke/weights/best.pt"
    fall_model: str = ""
    fight_model: str = ""

    # --- Camera / RTSP ---
    rtsp_timeout: int = 10
    camera_reconnect_delay: int = 3
    max_fps_per_stream: int = 30

    # --- Recording ---
    pre_event_seconds: int = 5
    post_event_seconds: int = 10
    incident_buffer_frames: int = 300
    retention_days: int = 30

    # --- Security ---
    secret_key: str = "change-me-in-production"
    api_token: str = ""
    allowed_hosts: str = "127.0.0.1,localhost"

    @field_validator("debug", mode="before")
    @classmethod
    def _bool_debug(cls, v: Any) -> bool:
        return _coerce_bool(v)

    @field_validator("frame_skip", mode="before")
    @classmethod
    def _int_frame_skip(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"frame_skip must be an integer, got {v!r}") from e

    @field_validator("port", mode="before")
    @classmethod
    def _int_port(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"port must be an integer, got {v!r}") from e

    @field_validator("target_fps", mode="before")
    @classmethod
    def _float_target_fps(cls, v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"target_fps must be a number, got {v!r}") from e

    # --- Computed paths ---
    @property
    def models_path(self) -> Path:
        p = Path(self.models_dir) if self.models_dir else PROJECT_ROOT / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir) if self.data_dir else PROJECT_ROOT / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def logs_path(self) -> Path:
        p = Path(self.logs_dir) if self.logs_dir else PROJECT_ROOT / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def database_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            rel = self.database_url[len("sqlite:///"):]
            if rel.startswith("./"):
                rel = rel[2:]
            return (PROJECT_ROOT / rel).resolve()
        return PROJECT_ROOT / "data" / "cctv_events.db"


settings = Settings()


def get_thresholds_path() -> Path:
    return PROJECT_ROOT / "config" / "event_thresholds.yaml"