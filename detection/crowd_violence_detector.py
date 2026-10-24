"""Crowd violence detector.

Distinguishes violent crowds from ordinary crowds using multi-person
context: number of persons, density, fraction of persons with high
acceleration, and pairwise proximity. Requires evidence from multiple
persons, so a single-person event will never trigger it.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from detection._base import BaseEventDetector, EventCandidate
from detection.person_detector import Track
from video.video_source import FramePacket

logger = logging.getLogger(__name__)


class CrowdViolenceDetector(BaseEventDetector):
    """Detects aggressive multi-person crowd behavior."""

    def __init__(self, *,
                 confidence_threshold: float = 0.6,
                 confirmation_seconds: float = 2.0,
                 cooldown_seconds: float = 5.0,
                 min_persons: int = 4,
                 aggression_motion_ratio: float = 0.4,
                 density_ratio: float = 0.15) -> None:
        super().__init__(
            event_type="crowd_violence",
            confidence_threshold=confidence_threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.min_persons = min_persons
        self.aggression_motion_ratio = aggression_motion_ratio
        self.density_ratio = density_ratio

    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        tracks: list[Track] = context.get("tracks", [])
        candidates: list[EventCandidate] = []
        if len(tracks) < self.min_persons:
            return candidates
        h, w = packet.frame.shape[:2]
        frame_area = max(1.0, float(w * h))
        # Density: fraction of frame covered by person bounding boxes
        total_area = 0.0
        high_motion = 0
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            total_area += max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if t.accel > 150:
                high_motion += 1
        density = total_area / frame_area
        motion_ratio = high_motion / len(tracks)
        # Score
        score = 0.0
        if density > self.density_ratio:
            score += 0.35
        if motion_ratio > self.aggression_motion_ratio:
            score += 0.45
        if len(tracks) >= self.min_persons * 2:
            score += 0.20
        if score >= self.confidence_threshold:
            candidates.append(EventCandidate(
                event_type="crowd_violence",
                confidence=min(1.0, score),
                timestamp=packet.timestamp,
                frame_index=packet.frame_index,
                source_id=packet.source_id,
                evidence={
                    "detector": "heuristic",
                    "num_persons": len(tracks),
                    "density_ratio": round(density, 3),
                    "motion_ratio": round(motion_ratio, 3),
                    "high_motion_count": high_motion,
                },
            ))
        return candidates