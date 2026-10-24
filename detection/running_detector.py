"""Abnormal running detector.

Estimates image-plane speed per track and flags sustained fast movement.
Image-plane speed is NOT physical speed without camera calibration, so the
threshold is configurable and conservative. Uses temporal persistence to
avoid transient spikes.
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


class RunningDetector(BaseEventDetector):
    """Detects unusually fast running behavior."""

    def __init__(self, *,
                 confidence_threshold: float = 0.55,
                 confirmation_seconds: float = 1.5,
                 cooldown_seconds: float = 3.0,
                 speed_threshold_px_per_s: float = 90.0,
                 min_persistence_s: float = 1.0,
                 min_track_frames: int = 5) -> None:
        super().__init__(
            event_type="abnormal_running",
            confidence_threshold=confidence_threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.speed_threshold_px_per_s = speed_threshold_px_per_s
        self.min_persistence_s = min_persistence_s
        self.min_track_frames = min_track_frames
        self._history: dict[int, deque] = {}

    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        tracks: list[Track] = context.get("tracks", [])
        candidates: list[EventCandidate] = []
        now = packet.timestamp
        for t in tracks:
            speed = float(np.hypot(*t.velocity))
            hist = self._history.setdefault(t.track_id, deque(maxlen=30))
            hist.append((now, speed))
            if len(hist) < self.min_track_frames:
                continue
            # Check persistence: how long has speed been above threshold?
            above_start: Optional[float] = None
            for ts, sp in hist:
                if sp >= self.speed_threshold_px_per_s:
                    if above_start is None:
                        above_start = ts
                else:
                    above_start = None
            persistence = (now - above_start) if above_start is not None else 0.0
            if persistence < self.min_persistence_s:
                continue
            # Score: ratio of speed to threshold, capped
            ratio = speed / max(1.0, self.speed_threshold_px_per_s)
            score = min(1.0, 0.4 + 0.6 * min(1.0, ratio - 1.0))
            if score >= self.confidence_threshold:
                candidates.append(EventCandidate(
                    event_type="abnormal_running",
                    confidence=score,
                    timestamp=now,
                    frame_index=packet.frame_index,
                    source_id=packet.source_id,
                    track_id=t.track_id,
                    bbox=t.bbox,
                    evidence={
                        "detector": "heuristic",
                        "speed_px_s": round(speed, 1),
                        "speed_threshold_px_s": self.speed_threshold_px_per_s,
                        "persistence_s": round(persistence, 2),
                        "note": "image-plane speed, not physical speed",
                    },
                ))
        return candidates

    def clear_history(self) -> None:
        super().clear_history()
        self._history.clear()