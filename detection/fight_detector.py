"""Fight detector.

A fight is characterized by two or more persons in close proximity with
sudden mutual motion (high relative acceleration) sustained over a short
window. This is a heuristic feature-based detector; an optional trained
video classifier can replace it when weights are available.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from detection._base import BaseEventDetector, EventCandidate
from detection.person_detector import Track
from video.video_source import FramePacket

logger = logging.getLogger(__name__)


class FightDetector(BaseEventDetector):
    """Detects physical altercations between people."""

    def __init__(self, *,
                 confidence_threshold: float = 0.6,
                 confirmation_seconds: float = 1.5,
                 cooldown_seconds: float = 5.0,
                 proximity_ratio: float = 0.6,
                 min_overlap_duration_s: float = 0.5,
                 model_name: str = "",
                 model_manager: Any = None,
                 imgsz: int = 640,
                 device: str = "cpu") -> None:
        super().__init__(
            event_type="fight",
            confidence_threshold=confidence_threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.proximity_ratio = proximity_ratio
        self.min_overlap_duration_s = min_overlap_duration_s
        self.model_name = model_name
        self.model_manager = model_manager
        self.imgsz = imgsz
        self.device = device
        self._model: Any = None
        self._pair_first_seen: dict[tuple[int, int], float] = {}

    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        tracks: list[Track] = context.get("tracks", [])
        candidates: list[EventCandidate] = []
        if len(tracks) < 2:
            return candidates
        # Consider pairs of tracks
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                ta, tb = tracks[i], tracks[j]
                iou = self._bbox_iou(ta.bbox, tb.bbox)
                proximity = self._proximity(ta, tb)
                # Motion: relative velocity magnitude
                rel_vel = self._relative_velocity(ta, tb)
                rel_accel = abs(ta.accel) + abs(tb.accel)
                # Heuristic score — tightened to avoid false positives
                score = 0.0
                # Require actual body overlap OR very close proximity
                if iou > 0.15 or proximity < 0.35:
                    score += 0.45
                # Require significant relative motion
                if rel_vel > 100:
                    score += 0.30
                # Require high mutual acceleration (sudden directional change)
                if rel_accel > 350:
                    score += 0.25
                if score >= self.confidence_threshold:
                    pair = tuple(sorted([ta.track_id, tb.track_id]))
                    if pair not in self._pair_first_seen:
                        self._pair_first_seen[pair] = packet.timestamp
                    duration = packet.timestamp - self._pair_first_seen[pair]
                    if duration >= self.min_overlap_duration_s:
                        candidates.append(EventCandidate(
                            event_type="fight",
                            confidence=min(1.0, score),
                            timestamp=packet.timestamp,
                            frame_index=packet.frame_index,
                            source_id=packet.source_id,
                            track_id=ta.track_id,
                            evidence={
                                "detector": "heuristic",
                                "pair": list(pair),
                                "iou": round(iou, 3),
                                "proximity": round(proximity, 3),
                                "rel_vel": round(rel_vel, 1),
                                "rel_accel": round(rel_accel, 1),
                                "overlap_s": round(duration, 2),
                            },
                        ))
        return candidates

    @staticmethod
    def _bbox_iou(a: list[float], b: list[float]) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        area_a = max(1e-6, (a[2]-a[0]) * (a[3]-a[1]))
        area_b = max(1e-6, (b[2]-b[0]) * (b[3]-b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _proximity(ta: Track, tb: Track) -> float:
        ca = np.array(ta.center); cb = np.array(tb.center)
        dist = float(np.linalg.norm(ca - cb))
        ha = max(1.0, ta.height); hb = max(1.0, tb.height)
        return dist / max(1.0, (ha + hb) / 2.0)

    @staticmethod
    def _relative_velocity(ta: Track, tb: Track) -> float:
        va = np.array(ta.velocity); vb = np.array(tb.velocity)
        return float(np.linalg.norm(va - vb))