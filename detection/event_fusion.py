"""Event fusion: combines all detectors into a unified event stream.

Each frame, detectors produce candidate events. The fusion layer:
  - Aggregates candidates across detectors.
  - Applies global deduplication and cooldown.
  - Emits confirmed events with merged evidence.
  - Maintains a rolling event history.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from detection._base import EventCandidate
from detection.crowd_violence_detector import CrowdViolenceDetector
from detection.fall_detector import FallDetector
from detection.fight_detector import FightDetector
from detection.fire_smoke_detector import FireSmokeDetector
from detection.person_detector import PersonDetector, Track
from detection.running_detector import RunningDetector
from video.video_source import FramePacket

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Runtime configuration for the fusion layer."""

    enable_fight: bool = True
    enable_fire: bool = True
    enable_smoke: bool = True
    enable_fall: bool = True
    enable_crowd_violence: bool = True
    enable_abnormal_running: bool = True
    global_cooldown: float = 2.0  # min seconds between any two confirmed events


class EventFusion:
    """Combines per-detector candidates into a single event stream."""

    def __init__(self, person_detector: PersonDetector,
                 model_manager,
                 config: Optional[FusionConfig] = None,
                 thresholds_path: Optional[str] = None) -> None:
        self.person_detector = person_detector
        self.model_manager = model_manager
        self.config = config or FusionConfig()
        self.thresholds_path = thresholds_path
        self._detectors: dict[str, object] = {}
        self._last_event_time: float = -1e9
        self._events: deque[EventCandidate] = deque(maxlen=500)
        self._lock = threading.Lock()
        self._on_event: Optional[Callable[[EventCandidate], None]] = None
        self._init_detectors()

    def _init_detectors(self) -> None:
        from config.settings import settings as s
        self._detectors["fight"] = FightDetector(
            confidence_threshold=s.fight_conf,
            confirmation_seconds=1.5, cooldown_seconds=5.0,
            model_manager=self.model_manager,
        )
        self._detectors["fire_smoke"] = FireSmokeDetector(
            model_manager=self.model_manager,
            model_name=s.fire_smoke_model,
            conf=s.fire_smoke_conf,
            confirmation_seconds=2.0, cooldown_seconds=10.0,
        )
        self._detectors["fall"] = FallDetector(
            confidence_threshold=s.fall_conf,
            confirmation_seconds=1.0, cooldown_seconds=8.0,
        )
        self._detectors["crowd_violence"] = CrowdViolenceDetector(
            confidence_threshold=s.crowd_violence_conf,
            confirmation_seconds=2.0, cooldown_seconds=5.0,
        )
        self._detectors["abnormal_running"] = RunningDetector(
            confidence_threshold=s.running_conf,
            confirmation_seconds=1.5, cooldown_seconds=3.0,
        )

    def set_on_event(self, callback: Callable[[EventCandidate], None]) -> None:
        self._on_event = callback

    def process_frame(self, packet: FramePacket) -> list[EventCandidate]:
        """Run all enabled detectors on a frame and return confirmed events."""
        tracks = self.person_detector.detect_and_track(packet.frame, packet.timestamp)
        context = {"tracks": tracks, "frame": packet.frame, "packet": packet}
        new_events: list[EventCandidate] = []
        with self._lock:
            for dtype, detector in self._detectors.items():
                if dtype == "fight" and not self.config.enable_fight:
                    continue
                if dtype == "fire_smoke" and not (self.config.enable_fire or self.config.enable_smoke):
                    continue
                if dtype == "fall" and not self.config.enable_fall:
                    continue
                if dtype == "crowd_violence" and not self.config.enable_crowd_violence:
                    continue
                if dtype == "abnormal_running" and not self.config.enable_abnormal_running:
                    continue
                try:
                    confirmed = detector.process(packet, context)
                except Exception as exc:
                    logger.exception("Detector %s failed: %s", dtype, exc)
                    continue
                for ev in confirmed:
                    if packet.timestamp - self._last_event_time < self.config.global_cooldown:
                        continue
                    self._last_event_time = packet.timestamp
                    self._events.append(ev)
                    new_events.append(ev)
                    if self._on_event is not None:
                        try:
                            self._on_event(ev)
                        except Exception:
                            logger.exception("on_event callback failed")
        return new_events

    @property
    def recent_events(self) -> list[EventCandidate]:
        with self._lock:
            return list(self._events)

    @property
    def active_tracks(self) -> dict[int, Track]:
        return self.person_detector.active_tracks

    def update_thresholds(self, thresholds: dict) -> None:
        """Update detector thresholds from a config dict.

        Accepts both nested format:
            {'fight': {'confidence_threshold': 0.7, ...}}
        and flat format from the dashboard:
            {'fight': 0.7, 'fire': 0.65, 'smoke': 0.65, ...}
        """
        mapping = {
            "fight": ("fight", "confidence_threshold", "confirmation_seconds", "cooldown_seconds"),
            "fire_smoke": ("fire_smoke", "confidence_threshold", "confirmation_seconds", "cooldown_seconds"),
            "fall": ("fall", "confidence_threshold", "confirmation_seconds", "cooldown_seconds"),
            "crowd_violence": ("crowd_violence", "confidence_threshold", "confirmation_seconds", "cooldown_seconds"),
            "abnormal_running": ("abnormal_running", "confidence_threshold", "confirmation_seconds", "cooldown_seconds"),
        }
        for key, detector in self._detectors.items():
            if key not in mapping:
                continue
            _, conf_attr, conf_s_attr, cooldown_attr = mapping[key]
            raw = thresholds.get(key)
            if raw is None:
                continue
            # Accept flat float values from the dashboard
            if isinstance(raw, (int, float)):
                setattr(detector, conf_attr, float(raw))
                continue
            if not isinstance(raw, dict):
                continue
            if conf_attr in raw:
                setattr(detector, conf_attr, float(raw[conf_attr]))
            if conf_s_attr in raw:
                setattr(detector, conf_s_attr, float(raw[conf_s_attr]))
            if cooldown_attr in raw:
                setattr(detector, cooldown_attr, float(raw[cooldown_attr]))
            # fire_smoke: also accept separate fire/smoke keys
            if key == "fire_smoke":
                fire_t = thresholds.get("fire")
                smoke_t = thresholds.get("smoke")
                if isinstance(fire_t, (int, float)) or isinstance(smoke_t, (int, float)):
                    current = getattr(detector, conf_attr, 0.6)
                    vals = [v for v in [fire_t, smoke_t] if isinstance(v, (int, float))]
                    if vals:
                        setattr(detector, conf_attr, max([current] + vals))

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._last_event_time = -1e9
        for detector in self._detectors.values():
            try:
                detector.clear_history()
            except Exception:
                pass