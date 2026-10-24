"""Person detection and tracking.

Uses Ultralytics YOLO with the built-in ByteTrack tracker to assign stable
track IDs. Tracks maintain position, velocity, trajectory history, and
bounding-box aspect ratio for downstream event detectors.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0  # COCO person class


@dataclass
class Track:
    """A tracked person."""

    track_id: int
    bbox: list[float]  # [x1, y1, x2, y2] in pixel coords
    confidence: float
    center: tuple[float, float]
    velocity: tuple[float, float]  # px per second in image plane
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    last_seen: float = 0.0
    first_seen: float = 0.0
    frames_since_update: int = 0
    aspect_ratio: float = 1.0
    height: float = 0.0
    accel: float = 0.0  # magnitude of velocity change (px/s^2)

    def update(self, bbox: list[float], confidence: float, timestamp: float,
               frame_shape: tuple[int, int, int]) -> None:
        prev_center = self.center
        prev_vel = self.velocity
        prev_mag = float(np.hypot(*prev_vel))
        self.bbox = bbox
        self.confidence = confidence
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.center = (cx, cy)
        self.aspect_ratio = max(0.1, (x2 - x1) / max(1.0, (y2 - y1)))
        self.height = max(1.0, (y2 - y1))
        dt = max(1e-6, timestamp - self.last_seen)
        vx = (cx - prev_center[0]) / dt
        vy = (cy - prev_center[1]) / dt
        self.velocity = (vx, vy)
        new_mag = float(np.hypot(vx, vy))
        self.accel = abs(new_mag - prev_mag) / dt
        self.trajectory.append((cx, cy, timestamp))
        if len(self.trajectory) > 60:
            self.trajectory.pop(0)
        self.last_seen = timestamp
        self.frames_since_update = 0

    def predict(self, timestamp: float) -> tuple[float, float]:
        """Predict center at timestamp using constant-velocity model."""
        cx, cy = self.center
        vx, vy = self.velocity
        dt = max(0.0, timestamp - self.last_seen)
        return (cx + vx * dt, cy + vy * dt)


class PersonDetector:
    """Detects people and tracks them with ByteTrack."""

    def __init__(self, model_manager: ModelManager, *,
                 model_name: str = "yolo11n.pt",
                 conf: float = 0.25,
                 imgsz: int = 640,
                 device: str = "cpu",
                 max_age: int = 30,
                 track_buffer: int = 30,
                 track_high_thresh: float = 0.25,
                 track_low_thresh: float = 0.1) -> None:
        self.model_manager = model_manager
        self.model_name = model_name
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.max_age = max_age
        self.track_buffer = track_buffer
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self._model: Any = None
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last_frame_time: Optional[float] = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            self._model = self.model_manager.get_yolo(self.model_name)
        return self._model

    def _resolve_tracker(self) -> str:
        """Return an absolute path to a local ByteTrack tracker config.

        Ultralytics resolves a relative tracker name against the current
        working directory. If the file is missing it attempts to download
        it from GitHub, which blocks for ~30s on a network timeout and
        stalls the whole processing loop. We therefore copy the bundled
        config into the project models directory and use an absolute path.
        """
        import os
        import shutil
        from pathlib import Path

        # Bundled config shipped with ultralytics
        bundled = Path(__file__).resolve().parent.parent
        candidates = [
            Path("F:/EMS(CNV)/cctv-anomaly-detection/models/bytetrack.yaml"),
            Path(__file__).resolve().parent.parent / "models" / "bytetrack.yaml",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # Fall back to the ultralytics package location
        try:
            import ultralytics
            pkg_dir = Path(ultralytics.__file__).resolve().parent
            src = pkg_dir / "cfg" / "trackers" / "bytetrack.yaml"
            if src.exists():
                dst = Path("F:/EMS(CNV)/cctv-anomaly-detection/models/bytetrack.yaml")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                return str(dst)
        except Exception:
            pass
        # Last resort: let ultralytics try (may download)
        return "bytetrack.yaml"

    def is_available(self) -> bool:
        """Return True if the person model can be loaded."""
        try:
            self._ensure_model()
            return True
        except Exception as exc:
            logger.warning("Person model not available: %s", exc)
            return False

    def detect_and_track(self, frame: np.ndarray, timestamp: float) -> list[Track]:
        """Run detection + tracking and return updated Track objects.

        Uses ``model.predict()`` for detection and a lightweight in-process
        IoU matcher for tracking.  The built-in ByteTrack tracker shipped with
        ultralytics adds ~10x latency on CPU (6.5s vs 674ms per frame in
        benchmarks), which stalls the whole processing loop.  Our IoU matcher
        is deterministic, has no network dependencies, and keeps track IDs
        stable across frames.
        """
        if frame is None or frame.size == 0:
            return []
        model = self._ensure_model()
        detections = self._detect(frame)
        matched = self._match(detections, timestamp, frame.shape)
        # Age out stale tracks
        stale = [tid for tid, t in self._tracks.items()
                 if timestamp - t.last_seen > self.max_age * (1 / 30.0)]
        for tid in stale:
            del self._tracks[tid]
        self._last_frame_time = timestamp
        return matched

    def _detect(self, frame: np.ndarray) -> list[tuple[list[float], float]]:
        """Run YOLO predict and return [(bbox, conf), ...] for person class."""
        results = self._model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        out: list[tuple[list[float], float]] = []
        if not results:
            return out
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return out
        try:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
        except Exception:
            return out
        for cid, box, cf in zip(cls_ids, xyxy, confs):
            if int(cid) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            out.append(([x1, y1, x2, y2], float(cf)))
        return out

    def _match(self, detections: list[tuple[list[float], float]],
               timestamp: float, frame_shape) -> list[Track]:
        """Greedy IoU matching between detections and existing tracks."""
        matched: list[Track] = []
        if not detections:
            return matched
        track_ids = list(self._tracks.keys())
        iou_matrix: dict[int, dict[int, float]] = {}
        for tid in track_ids:
            iou_matrix[tid] = {}
            for di, (dbbox, _) in enumerate(detections):
                iou_matrix[tid][di] = self._iou(self._tracks[tid].bbox, dbbox)
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        # Greedy: highest IoU first
        pairs = sorted(
            ((iou_matrix[tid][di], tid, di)
             for tid in track_ids for di in range(len(detections))),
            key=lambda x: x[0], reverse=True
        )
        for iou, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            if iou < 0.3:
                continue
            dbbox, conf = detections[di]
            self._tracks[tid].update(dbbox, conf, timestamp, frame_shape)
            matched.append(self._tracks[tid])
            used_tracks.add(tid)
            used_dets.add(di)
        # Remaining detections become new tracks
        for di, (dbbox, conf) in enumerate(detections):
            if di in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            x1, y1, x2, y2 = dbbox
            t = Track(
                track_id=tid,
                bbox=dbbox,
                confidence=conf,
                center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                velocity=(0.0, 0.0),
                aspect_ratio=max(0.1, (x2 - x1) / max(1.0, (y2 - y1))),
                height=max(1.0, (y2 - y1)),
                last_seen=timestamp,
                first_seen=timestamp,
            )
            self._tracks[tid] = t
            matched.append(t)
        return matched

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @property
    def active_tracks(self) -> dict[int, Track]:
        return dict(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1