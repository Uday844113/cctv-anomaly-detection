"""Person fall detector.

Analyzes bounding-box aspect ratio, vertical displacement, and motion
over a short temporal window. Distinguishes falling from sitting/bending/
lying down by requiring a rapid downward displacement followed by a
sustained low-height posture.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from detection._base import BaseEventDetector, EventCandidate
from detection.person_detector import Track
from video.video_source import FramePacket

logger = logging.getLogger(__name__)


class FallDetector(BaseEventDetector):
    """Detects when a person falls to the ground."""

    def __init__(self, *,
                 confidence_threshold: float = 0.6,
                 confirmation_seconds: float = 1.0,
                 cooldown_seconds: float = 8.0,
                 max_aspect_ratio: float = 1.6,
                 min_height_drop_ratio: float = 0.30,
                 fall_speed_px_per_s: float = 120.0,
                 min_track_frames: int = 3) -> None:
        super().__init__(
            event_type="fall",
            confidence_threshold=confidence_threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.max_aspect_ratio = max_aspect_ratio
        self.min_height_drop_ratio = min_height_drop_ratio
        self.fall_speed_px_per_s = fall_speed_px_per_s
        self.min_track_frames = min_track_frames
        # Per-track history: (timestamp, center_y, height)
        self._history: dict[int, deque] = {}

    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        tracks: list[Track] = context.get("tracks", [])
        candidates: list[EventCandidate] = []
        now = packet.timestamp
        for t in tracks:
            hist = self._history.setdefault(t.track_id, deque(maxlen=15))
            hist.append((now, t.center[1], t.height))
            if len(hist) < self.min_track_frames:
                continue
            # Look for a downward displacement over the recent window
            oldest = hist[0]
            recent = hist[-1]
            dt = recent[0] - oldest[0]
            if dt <= 0:
                continue
            dy = recent[1] - oldest[1]  # positive = moving down in image coords
            speed = dy / dt
            height_drop = (oldest[2] - recent[2]) / max(1.0, oldest[2])
            # A fall: fast downward motion + height reduction + wide aspect
            is_fast = speed > self.fall_speed_px_per_s
            is_dropping = height_drop > self.min_height_drop_ratio
            is_wide = t.aspect_ratio > self.max_aspect_ratio
            score = 0.0
            if is_fast:
                score += 0.45
            if is_dropping:
                score += 0.30
            if is_wide:
                score += 0.25
            if score >= self.confidence_threshold:
                candidates.append(EventCandidate(
                    event_type="fall",
                    confidence=min(1.0, score),
                    timestamp=now,
                    frame_index=packet.frame_index,
                    source_id=packet.source_id,
                    track_id=t.track_id,
                    bbox=t.bbox,
                    evidence={
                        "detector": "heuristic",
                        "speed_px_s": round(speed, 1),
                        "height_drop_ratio": round(height_drop, 3),
                        "aspect_ratio": round(t.aspect_ratio, 3),
                        "dt_s": round(dt, 3),
                    },
                ))
        return candidates

    def clear_history(self) -> None:
        super().clear_history()
        self._history.clear()