"""Base classes for event detectors.

Each detector:
  - Takes a FramePacket and returns candidate events with confidence.
  - Maintains its own temporal confirmation state.
  - Has configurable thresholds and cooldowns.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from video.video_source import FramePacket

logger = logging.getLogger(__name__)


class EventState(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


@dataclass
class EventCandidate:
    """A single candidate event detected in a frame."""

    event_type: str
    confidence: float
    timestamp: float
    frame_index: int
    source_id: str
    bbox: Optional[list[float]] = None
    track_id: Optional[int] = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "source_id": self.source_id,
            "bbox": self.bbox,
            "track_id": self.track_id,
            "evidence": self.evidence,
        }


class BaseEventDetector(ABC):
    """Abstract base for all event detectors."""

    def __init__(self, event_type: str, *,
                 confidence_threshold: float = 0.6,
                 confirmation_seconds: float = 1.5,
                 cooldown_seconds: float = 5.0) -> None:
        self.event_type = event_type
        self.confidence_threshold = confidence_threshold
        self.confirmation_seconds = confirmation_seconds
        self.cooldown_seconds = cooldown_seconds
        self._pending: dict[str, list[EventCandidate]] = {}
        self._confirmed: list[EventCandidate] = []
        self._last_fired: dict[str, float] = {}

    @abstractmethod
    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        """Return candidate events for a single frame."""

    def process(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        """Run detect() and apply temporal confirmation + cooldown."""
        candidates = self.detect(packet, context)
        confirmed: list[EventCandidate] = []
        now = packet.timestamp
        for cand in candidates:
            if cand.confidence < self.confidence_threshold:
                continue
            key = self._dedup_key(cand)
            # Cooldown check
            last = self._last_fired.get(key, -1e9)
            if now - last < self.cooldown_seconds:
                continue
            # Temporal confirmation
            bucket = self._pending.setdefault(key, [])
            bucket.append(cand)
            # Drop old candidates outside the confirmation window
            cutoff = now - self.confirmation_seconds
            bucket[:] = [c for c in bucket if c.timestamp >= cutoff]
            if len(bucket) >= 2:
                # Average confidence of the sustained candidates
                avg_conf = float(np.mean([c.confidence for c in bucket]))
                confirmed_cand = EventCandidate(
                    event_type=cand.event_type,
                    confidence=avg_conf,
                    timestamp=now,
                    frame_index=packet.frame_index,
                    source_id=packet.source_id,
                    bbox=cand.bbox,
                    track_id=cand.track_id,
                    evidence={**cand.evidence, "confirmation_frames": len(bucket)},
                )
                self._confirmed.append(confirmed_cand)
                confirmed.append(confirmed_cand)
                self._last_fired[key] = now
                bucket.clear()
        # Cap confirmed list
        if len(self._confirmed) > 100:
            self._confirmed = self._confirmed[-100:]
        return confirmed

    def _dedup_key(self, cand: EventCandidate) -> str:
        """Key used for cooldown + confirmation bucketing."""
        if cand.track_id is not None:
            return f"{cand.event_type}:track:{cand.track_id}"
        if cand.bbox is not None:
            cx = (cand.bbox[0] + cand.bbox[2]) / 2
            cy = (cand.bbox[1] + cand.bbox[3]) / 2
            cell = f"{int(cx // 50)}:{int(cy // 50)}"
            return f"{cand.event_type}:cell:{cell}"
        return f"{cand.event_type}:global"

    @property
    def confirmed_events(self) -> list[EventCandidate]:
        return list(self._confirmed)

    def clear_history(self) -> None:
        self._pending.clear()
        self._confirmed.clear()
        self._last_fired.clear()

    def update_thresholds(self, *, confidence_threshold: Optional[float] = None,
                          confirmation_seconds: Optional[float] = None,
                          cooldown_seconds: Optional[float] = None) -> None:
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        if confirmation_seconds is not None:
            self.confirmation_seconds = confirmation_seconds
        if cooldown_seconds is not None:
            self.cooldown_seconds = cooldown_seconds