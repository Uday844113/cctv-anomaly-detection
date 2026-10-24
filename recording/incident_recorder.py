"""Incident recording: snapshots and clip export."""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detection._base import EventCandidate

logger = logging.getLogger(__name__)


class IncidentRecorder:
    """Records pre/post event buffers and exports incident clips."""

    def __init__(self, output_dir: Path, *,
                 pre_event_frames: int = 150,
                 post_event_frames: int = 300,
                 fps: float = 15.0) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames
        self.fps = fps
        self._buffer: deque = deque(maxlen=max(pre_event_frames + post_event_frames, 1))
        self._lock = threading.Lock()
        self._recording: Optional[dict] = None
        self._post_frames_written = 0

    def push_frame(self, frame: np.ndarray, timestamp: float) -> None:
        with self._lock:
            self._buffer.append((frame, timestamp))
            if self._recording is not None:
                self._recording["frames"].append((frame, timestamp))
                self._post_frames_written += 1
                if self._post_frames_written >= self.post_event_frames:
                    self._finalize_recording()

    def on_event(self, event: EventCandidate) -> None:
        """Start a recording for a confirmed event."""
        with self._lock:
            if self._recording is not None:
                logger.info("Overlapping incident for %s", event.event_type)
            frames = list(self._buffer)
            self._recording = {
                "event": event,
                "frames": frames,
                "post_frames_written": 0,
            }
            self._post_frames_written = 0
            logger.info("Started incident recording for %s", event.event_type)

    def _finalize_recording(self) -> None:
        if self._recording is None:
            return
        rec = self._recording
        self._recording = None
        self._post_frames_written = 0
        event = rec["event"]
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{event.event_type}_{ts}_{event.frame_index}"
        # Snapshot (middle frame)
        if rec["frames"]:
            mid = rec["frames"][len(rec["frames"]) // 2]
            snap_path = self.output_dir / f"{base}_snapshot.jpg"
            cv2.imwrite(str(snap_path), mid[0])
            event.evidence["snapshot_path"] = str(snap_path)
        # Clip
        clip_path = self.output_dir / f"{base}.mp4"
        self._write_clip(rec["frames"], clip_path)
        event.evidence["incident_clip_path"] = str(clip_path)

    def _write_clip(self, frames: list, path: Path) -> None:
        if not frames:
            return None
        h, w = frames[0][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        try:
            for f, _ in frames:
                writer.write(f)
        finally:
            writer.release()
        logger.info("Wrote incident clip: %s (%d frames)", path, len(frames))

    def flush(self) -> None:
        """Force-finalize any pending recording."""
        with self._lock:
            if self._recording is not None:
                self._post_frames_written = self.post_event_frames
                self._finalize_recording()

    def shutdown(self) -> None:
        self.flush()
        self._buffer.clear()