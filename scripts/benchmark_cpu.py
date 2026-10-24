"""CPU benchmark for the detection pipeline.

Usage:
    python scripts/benchmark_cpu.py --video data/videos/sample.mp4 [--frames 100]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import numpy as np

from config.settings import settings
from models.model_manager import ModelManager
from detection.person_detector import PersonDetector
from video.file_source import FileSource
from video.frame_processor import FrameProcessor, ProcessingStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark(video_path: str, *, frames: int = 100, frame_skip: int = 2,
              imgsz: int = 640) -> dict:
    src = FileSource(video_path, target_fps=30.0, max_frames=frames)
    if not src.open():
        raise RuntimeError(f"Cannot open {video_path}")
    mm = ModelManager(settings.models_path)
    detector = PersonDetector(mm, conf=0.25, imgsz=imgsz)
    stats = ProcessingStats()
    t0 = time.perf_counter()
    processed = 0
    for packet in src:
        if packet.frame_index % frame_skip != 0:
            continue
        detector.detect_and_track(packet.frame, packet.timestamp)
        processed += 1
        stats.frames_processed += 1
        if processed >= frames:
            break
    elapsed = time.perf_counter() - t0
    src.close()
    fps = processed / elapsed if elapsed > 0 else 0.0
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    return {
        "video": video_path,
        "frames_processed": processed,
        "elapsed_s": round(elapsed, 3),
        "fps": round(fps, 2),
        "imgsz": imgsz,
        "frame_skip": frame_skip,
        "memory_mb": round(mem, 1),
        "cpu_percent": psutil.cpu_percent(interval=None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    result = benchmark(args.video, frames=args.frames, frame_skip=args.frame_skip, imgsz=args.imgsz)
    print("\n=== CPU Benchmark ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()