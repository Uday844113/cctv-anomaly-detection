"""Local webcam source (built-in or USB)."""
from __future__ import annotations

import logging
from typing import Optional, Union

import cv2
import numpy as np

from video.video_source import FramePacket, VideoSource

logger = logging.getLogger(__name__)


class WebcamSource(VideoSource):
    """Reads frames from a local webcam via OpenCV."""

    def __init__(self, camera_index: Union[int, str] = 0, *,
                 target_fps: Optional[float] = None,
                 width: Optional[int] = None,
                 height: Optional[int] = None,
                 reconnect_delay: float = 3.0,
                 max_reconnect_attempts: int = 5) -> None:
        super().__init__(source_id=f"webcam:{camera_index}", target_fps=target_fps)
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._source_fps: Optional[float] = None
        self._attempt = 0

    def open(self) -> bool:
        idx = self.camera_index
        try:
            self._cap = cv2.VideoCapture(idx)
        except Exception as exc:
            logger.error("Webcam open failed (index=%s): %s", idx, exc)
            return False
        if not self._cap.isOpened():
            logger.error("Webcam not available (index=%s)", idx)
            return False
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        self._source_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._frame_index = 0
        self._attempt = 0
        self._is_open = True
        logger.info("Opened webcam %s @ %.2f fps", idx, self._source_fps or 0)
        return True

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_open = False
        logger.info("Closed webcam %s", self.camera_index)

    @property
    def source_fps(self) -> Optional[float]:
        return self._source_fps

    def _try_reconnect(self) -> bool:
        if self._attempt >= self.max_reconnect_attempts:
            logger.error("Webcam reconnect limit reached for %s", self.camera_index)
            return False
        self._attempt += 1
        logger.warning("Attempting webcam reconnect (%d/%d) for %s",
                       self._attempt, self.max_reconnect_attempts, self.camera_index)
        self.close()
        import time as _time
        _time.sleep(self.reconnect_delay)
        return self.open()

    def read(self) -> Optional[FramePacket]:
        if self._cap is None or not self._cap.isOpened():
            if not self._try_reconnect():
                return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Webcam read failed, attempting reconnect")
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
            timestamp=self._frame_index / (self._source_fps or 30.0),
            source_fps=self._source_fps,
            metadata={"width": frame.shape[1], "height": frame.shape[0]},
        )