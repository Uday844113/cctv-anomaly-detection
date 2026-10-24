"""RTSP / IP camera source."""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional, Union

import cv2
import numpy as np

from video.video_source import FramePacket, VideoSource

logger = logging.getLogger(__name__)


def _redact_url(url: str) -> str:
    """Return URL with password redacted for safe logging."""
    return re.sub(r"(://[^:/]+:)[^@]+(@)", r"\1***\2", url or "")


class RTSPSource(VideoSource):
    """Reads frames from an RTSP URL with reconnect support."""

    def __init__(self, url: str, *, target_fps: Optional[float] = None,
                 timeout: int = 10,
                 reconnect_delay: float = 3.0,
                 max_reconnect_attempts: int = 5,
                 transport: str = "tcp") -> None:
        super().__init__(source_id=f"rtsp:{_redact_url(url)}", target_fps=target_fps)
        self.url = url
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.transport = transport
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._source_fps: Optional[float] = None
        self._attempt = 0

    def open(self) -> bool:
        env = os.environ.copy()
        # Lower GStreamer/FFmpeg latency for live streams
        env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}"
        try:
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        except Exception as exc:
            logger.error("RTSP open failed for %s: %s", _redact_url(self.url), exc)
            return False
        if not self._cap.isOpened():
            logger.error("RTSP stream unavailable: %s", _redact_url(self.url))
            return False
        self._source_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_index = 0
        self._attempt = 0
        self._is_open = True
        logger.info("Opened RTSP stream (redacted): %s", _redact_url(self.url))
        return True

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_open = False
        logger.info("Closed RTSP stream: %s", _redact_url(self.url))

    @property
    def source_fps(self) -> Optional[float]:
        return self._source_fps

    def _try_reconnect(self) -> bool:
        if self._attempt >= self.max_reconnect_attempts:
            logger.error("RTSP reconnect limit reached for %s", _redact_url(self.url))
            return False
        self._attempt += 1
        logger.warning("RTSP reconnect (%d/%d): %s",
                       self._attempt, self.max_reconnect_attempts, _redact_url(self.url))
        self.close()
        time.sleep(self.reconnect_delay)
        return self.open()

    def read(self) -> Optional[FramePacket]:
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("RTSP read failed, attempting reconnect")
            if not self._try_reconnect():
                return None
            ret, frame = self._cap.read()
            if not ret or frame is None:
                return None
        self._frame_index += 1
        return FramePacket(
            frame=frame,
            frame_index=self._frame_index,
            source_id=self.source_id,
            timestamp=time.time(),
            source_fps=self._source_fps,
            metadata={"width": frame.shape[1], "height": frame.shape[0]},
        )