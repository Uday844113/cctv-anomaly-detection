"""Fire and smoke detector.

Uses a dedicated pretrained fire/smoke detection model when available.
Falls back to a color-temperature + temporal heuristic when the model is
missing. The heuristic is explicitly labeled as non-trained.

Strategy:
  - Trained path: YOLO model on fire/smoke classes.
  - Heuristic path: HSV color analysis (bright orange/red for fire, grey/
    low-saturation plume for smoke) with temporal confirmation to reject
    lighting reflections and fog.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

from detection._base import BaseEventDetector, EventCandidate
from models.model_manager import ModelManager
from video.video_source import FramePacket

logger = logging.getLogger(__name__)


class FireSmokeDetector(BaseEventDetector):
    """Detects fire and smoke events."""

    def __init__(self, model_manager: ModelManager, *,
                 model_name: str = "",
                 conf: float = 0.25,
                 imgsz: int = 640,
                 device: str = "cpu",
                 fire_confidence_threshold: float = 0.6,
                 smoke_confidence_threshold: float = 0.6,
                 confirmation_seconds: float = 2.0,
                 cooldown_seconds: float = 10.0,
                 use_heuristic_fallback: bool = True) -> None:
        super().__init__(
            event_type="fire_smoke",
            confidence_threshold=fire_confidence_threshold,
            confirmation_seconds=confirmation_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        self.model_manager = model_manager
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.fire_confidence_threshold = fire_confidence_threshold
        self.smoke_confidence_threshold = smoke_confidence_threshold
        self.use_heuristic_fallback = use_heuristic_fallback
        self._model: Any = None
        self._model_available: Optional[bool] = None

    def _ensure_model(self) -> Optional[Any]:
        if self._model is not None:
            return self._model
        if not self.model_name:
            self._model_available = False
            return None
        # Reject missing local model files immediately.  If the path is a
        # relative local file that does not exist, ultralytics would attempt
        # to download it from GitHub, blocking for ~30s on a network timeout
        # and stalling the entire processing loop.
        from pathlib import Path
        candidate = Path(self.model_name)
        if not candidate.is_absolute() and not candidate.exists():
            # Also check inside the models directory
            alt = self.model_manager.models_dir / self.model_name
            if not alt.exists():
                logger.warning("Fire/smoke model file not found locally: %s "
                               "(also checked %s) — heuristic fallback only",
                               self.model_name, alt)
                self._model_available = False
                return None
        try:
            self._model = self.model_manager.get_yolo(self.model_name)
            self._model_available = True
        except Exception as exc:
            logger.warning("Fire/smoke model unavailable: %s", exc)
            self._model_available = False
        return self._model

    @property
    def model_available(self) -> bool:
        if self._model_available is None:
            self._ensure_model()
        return bool(self._model_available)

    def detect(self, packet: FramePacket, context: dict) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        model = self._ensure_model()
        if model is not None:
            candidates.extend(self._detect_with_model(packet))
        elif self.use_heuristic_fallback:
            candidates.extend(self._detect_heuristic(packet))
        return candidates

    def _detect_with_model(self, packet: FramePacket) -> list[EventCandidate]:
        candidates: list[EventCandidate] = []
        try:
            results = self._model.predict(
                packet.frame, conf=self.conf, imgsz=self.imgsz,
                device=self.device, verbose=False,
            )
        except Exception as exc:
            logger.error("Fire/smoke inference failed: %s", exc)
            return candidates
        if not results:
            return candidates
        res = results[0]
        boxes = getattr(res, "boxes", None)
        names = getattr(res, "names", {}) or {}
        if boxes is None:
            return candidates
        try:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception:
            return candidates
        for cid, box, cf in zip(cls_ids, xyxy, confs):
            label = names.get(int(cid), str(cid)).lower()
            x1, y1, x2, y2 = [float(v) for v in box]
            if "fire" in label or "flame" in label:
                candidates.append(EventCandidate(
                    event_type="fire", confidence=float(cf),
                    timestamp=packet.timestamp, frame_index=packet.frame_index,
                    source_id=packet.source_id, bbox=[x1, y1, x2, y2],
                    evidence={"label": label, "detector": "trained_model"},
                ))
            elif "smoke" in label:
                candidates.append(EventCandidate(
                    event_type="smoke", confidence=float(cf),
                    timestamp=packet.timestamp, frame_index=packet.frame_index,
                    source_id=packet.source_id, bbox=[x1, y1, x2, y2],
                    evidence={"label": label, "detector": "trained_model"},
                ))
        return candidates

    # ------------------------------------------------------------------
    # Heuristic fallback (explicitly NOT a trained model)
    # ------------------------------------------------------------------
    def _detect_heuristic(self, packet: FramePacket) -> list[EventCandidate]:
        frame = packet.frame
        # Downscale for cheap color analysis — full-resolution HSV + contour
        # extraction on 1080p frames adds ~3s per frame on CPU.
        scale = 0.5
        small = cv2.resize(frame, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        candidates: list[EventCandidate] = []

        # Fire: bright orange/red with high value and high saturation.
        # Tighten the masks to reject warm skin tones and interior lighting.
        fire_mask = (
            cv2.inRange(hsv, np.array([0, 170, 170]), np.array([25, 255, 255])) |
            cv2.inRange(hsv, np.array([165, 170, 170]), np.array([180, 255, 255]))
        )
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fire_contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = small.shape[:2]
        for cnt in fire_contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            # Reject small scattered blobs — fire is a sustained bright region
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 10 or bh < 10:
                continue
            # Map back to original frame coordinates
            ox, oy = x / scale, y / scale
            ow, oh = bw / scale, bh / scale
            conf = min(1.0, 0.4 + 0.6 * (area / max(1.0, float(h * w))))
            candidates.append(EventCandidate(
                event_type="fire", confidence=conf,
                timestamp=packet.timestamp, frame_index=packet.frame_index,
                source_id=packet.source_id,
                bbox=[float(ox), float(oy), float(ox + ow), float(oy + oh)],
                evidence={"detector": "heuristic", "area_px": float(area)},
            ))

        # Smoke: low-saturation, mid-to-high value grey plumes.
        # Reject motion blur by requiring a large, diffuse region.
        smoke_mask = cv2.inRange(hsv, np.array([0, 0, 80]), np.array([180, 45, 230]))
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in smoke_contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < 20 or bh < 20:
                continue
            ox, oy = x / scale, y / scale
            ow, oh = bw / scale, bh / scale
            conf = min(1.0, 0.3 + 0.6 * (area / max(1.0, float(h * w))))
            candidates.append(EventCandidate(
                event_type="smoke", confidence=conf,
                timestamp=packet.timestamp, frame_index=packet.frame_index,
                source_id=packet.source_id,
                bbox=[float(ox), float(oy), float(ox + ow), float(oy + oh)],
                evidence={"detector": "heuristic", "area_px": float(area)},
            ))
        return candidates