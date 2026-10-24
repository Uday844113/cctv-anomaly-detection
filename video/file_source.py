"""Local video file source."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from video.video_source import FramePacket, VideoSource

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


class FileSource(VideoSource):
    """Reads frames from a local video file using OpenCV."""

    def __init__(self, path: str, *, target_fps: Optional[float] = None,
                 loop: bool = False, max_frames: Optional[int] = None) -> None:
        super().__init__(source_id=f"file:{path}", target_fps=target_fps)
        self.path = Path(path)
        self.loop = loop
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0
        self._source_fps: Optional[float] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._error_count = 0

    def open(self) -> bool:
        if not self.path.exists():
            logger.error("Video file not found: %s", self.path)
            return False
        if self.path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.error("Unsupported file extension: %s", self.path.suffix)
            return False
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            logger.error("OpenCV could not open video: %s", self.path)
            return False
        self._source_fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_index = 0
        self._is_open = True
        logger.info("Opened file source: %s (%dx%d @ %.2f fps)",
                    self.path, self._width, self._height, self._source_fps)
        return True

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("Closed file source: %s", self.path)

    @property
    def source_fps(self) -> Optional[float]:
        return self._source_fps

    @property
    def width(self) -> Optional[int]:
        return self._width

    @property
    def height(self) -> Optional[int]:
        return self._height

    @property
    def frame_count(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Optional[FramePacket]:
        if self._cap is None or not self._cap.isOpened():
            return None
        if self.max_frames is not None and self._frame_index >= self.max_frames:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_index = 0
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    return None
            else:
                return None
        self._frame_index += 1
        self._error_count = 0
        return FramePacket(
            frame=frame,
            frame_index=self._frame_index,
            source_id=self.source_id,
            timestamp=self._frame_index / (self._source_fps or 30.0),
            source_fps=self._source_fps,
            metadata={"width": frame.shape[1], "height": frame.shape[0]},
        )

    def seek(self, frame_index: int) -> None:
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
            self._frame_index = max(0, frame_index)