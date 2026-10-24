"""Frame processor: orchestrates per-frame inference and event detection.

Architecture: a reader thread pulls frames from the VideoSource and pushes
them into a bounded queue; a separate processor thread drains the queue and
runs the (potentially slow) on_frame callback.  This decoupling keeps the
capture side responsive even when inference falls behind.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from video.video_source import FramePacket, VideoSource

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Live processing statistics."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    inference_time_ms: float = 0.0
    processing_fps: float = 0.0
    cpu_percent: float = 0.0
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "processing_fps": round(self.processing_fps, 2),
            "cpu_percent": round(self.cpu_percent, 1),
            "last_error": self.last_error,
        }


class FrameProcessor:
    """Consumes frames from a VideoSource and dispatches them to a callback.

    Uses two daemon threads:
      - reader:      reads from the source, pushes packets onto a bounded queue
      - processor:   drains the queue and calls on_frame (YOLO + detectors)

    If the processor cannot keep up, the bounded queue applies backpressure:
    excess frames are dropped rather than growing memory unboundedly.
    """

    def __init__(self, source: VideoSource, *,
                 frame_skip: int = 1,
                 max_queue: int = 8,
                 max_process_ms: float = 2000.0,
                 on_frame: Optional[Callable[[FramePacket], None]] = None,
                 on_stats: Optional[Callable[[ProcessingStats], None]] = None) -> None:
        self.source = source
        self.frame_skip = max(1, int(frame_skip))
        self.max_queue = max_queue
        self.max_process_ms = max(0.0, float(max_process_ms))
        self.on_frame = on_frame
        self.on_stats = on_stats
        self._queue: "queue.Queue[Optional[FramePacket]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._processor_thread: Optional[threading.Thread] = None
        self.stats = ProcessingStats()
        self._fps_window: deque[float] = deque(maxlen=30)
        self._lock = threading.Lock()
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None

    def start(self) -> None:
        if self._reader_thread is not None and self._reader_thread.is_alive():
            return
        self._stop.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="frame-reader")
        self._processor_thread = threading.Thread(
            target=self._process_loop, daemon=True, name="frame-processor")
        self._reader_thread.start()
        self._processor_thread.start()
        logger.info("Frame processor started for %s", self.source.source_id)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        for t in (self._reader_thread, self._processor_thread):
            if t is not None:
                t.join(timeout=5.0)
        self._reader_thread = None
        self._processor_thread = None
        logger.info("Frame processor stopped for %s", self.source.source_id)

    def _read_loop(self) -> None:
        """Read frames from the source and push them onto the bounded queue."""
        logger.debug("Reader thread started for %s", self.source.source_id)
        put_count = 0
        full_count = 0
        try:
            for packet in self.source:
                if self._stop.is_set():
                    break
                self.stats.frames_read += 1
                if packet.frame_index % self.frame_skip != 0:
                    self.stats.frames_dropped += 1
                    continue
                try:
                    self._queue.put_nowait(packet)
                    put_count += 1
                except queue.Full:
                    self.stats.frames_dropped += 1
                    full_count += 1
        except Exception as exc:
            logger.exception("Frame read error: %s", exc)
            self.stats.last_error = str(exc)
        finally:
            logger.debug("Reader exiting: reads=%d puts=%d full=%d queue_size=%d",
                         self.stats.frames_read, put_count, full_count, self._queue.qsize())
            # Signal end of stream to the processor
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def _process_loop(self) -> None:
        """Drain the queue and run the on_frame callback for each packet."""
        logger.debug("Processor thread started for %s", self.source.source_id)
        get_count = 0
        empty_count = 0
        try:
            while not self._stop.is_set():
                try:
                    packet = self._queue.get(timeout=1.0)
                    get_count += 1
                except queue.Empty:
                    empty_count += 1
                    continue
                if packet is None:
                    logger.debug("Processor received end-of-stream sentinel")
                    break
                try:
                    self._process_packet(packet)
                except Exception as exc:
                    logger.exception("Frame processing error: %s", exc)
                    self.stats.last_error = str(exc)
        except Exception as exc:
            logger.exception("Processor thread crashed: %s", exc)
            self.stats.last_error = f"processor crash: {exc}"
        finally:
            logger.debug("Processor exiting: gets=%d empties=%d processed=%d",
                         get_count, empty_count, self.stats.frames_processed)
            self._report_stats()
            logger.debug("Processor thread exiting for %s", self.source.source_id)

    def _process_packet(self, packet: FramePacket) -> None:
        t0 = time.perf_counter()
        if self.on_frame is not None:
            self.on_frame(packet)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.stats.frames_processed += 1
            self.stats.inference_time_ms = (
                0.9 * self.stats.inference_time_ms + 0.1 * elapsed_ms
            )
            now = time.monotonic()
            self._fps_window.append(now)
            if len(self._fps_window) >= 2:
                dt = now - self._fps_window[0]
                self.stats.processing_fps = (
                    (len(self._fps_window) - 1) / dt if dt > 0 else 0.0
                )
            if self._proc is not None:
                try:
                    self.stats.cpu_percent = self._proc.cpu_percent(interval=None)
                except Exception:
                    pass
        if self.on_stats is not None:
            self.on_stats(self.stats)

    def _report_stats(self) -> None:
        if self.on_stats is not None:
            self.on_stats(self.stats)

    def read_processed(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """Block for the next processed packet (for synchronous pipelines)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None