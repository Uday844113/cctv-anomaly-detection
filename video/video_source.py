"""Abstract video source interface.

All video inputs (file, webcam, RTSP) implement the same protocol so the
downstream pipeline is source-agnostic. Each source yields frames through a
generator and exposes metadata about its configuration.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FramePacket:
    """A single processed frame plus metadata."""

    frame: np.ndarray
    frame_index: int
    source_id: str
    timestamp: float
    source_fps: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.frame.shape[0])

    @property
    def width(self) -> int:
        return int(self.frame.shape[1])


class VideoSource(ABC):
    """Base class for all video sources."""

    def __init__(self, source_id: str, *, target_fps: Optional[float] = None) -> None:
        self.source_id = source_id
        self.target_fps = target_fps
        self._min_interval = (1.0 / target_fps) if target_fps else 0.0
        self._last_yield = 0.0
        self._is_open = False

    @abstractmethod
    def open(self) -> bool:
        """Open the source. Return True on success."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @abstractmethod
    def read(self) -> Optional[FramePacket]:
        """Read the next frame. Return None at end of stream / on error."""

    def __iter__(self) -> Iterator[FramePacket]:
        """Iterate frames with optional FPS throttling."""
        while True:
            if self._min_interval > 0:
                now = time.monotonic()
                wait = self._min_interval - (now - self._last_yield)
                if wait > 0:
                    time.sleep(wait)
                self._last_yield = time.monotonic()
            packet = self.read()
            if packet is None:
                break
            yield packet

    @property
    def is_open(self) -> bool:
        return self._is_open

    def __enter__(self) -> "VideoSource":
        if not self.open():
            raise RuntimeError(f"Failed to open source: {self.source_id}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_video_source(source_type: str, source_value: str, **kwargs) -> VideoSource:
    """Factory that returns the appropriate VideoSource subclass.

    Only kwargs accepted by the target source are forwarded; extras such as
    ``frame_skip`` (handled by FrameProcessor) are silently dropped.
    """
    source_type = (source_type or "").lower()
    if source_type == "file":
        from video.file_source import FileSource
        allowed = {"target_fps", "loop", "max_frames"}
        return FileSource(source_value, **{k: v for k, v in kwargs.items() if k in allowed})
    if source_type == "webcam":
        from video.webcam_source import WebcamSource
        allowed = {"target_fps", "max_frames"}
        return WebcamSource(int(source_value) if source_value.isdigit() else source_value,
                            **{k: v for k, v in kwargs.items() if k in allowed})
    if source_type == "rtsp":
        from video.rtsp_source import RTSPSource
        allowed = {"target_fps", "timeout", "max_frames"}
        return RTSPSource(source_value, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"Unknown source type: {source_type}")