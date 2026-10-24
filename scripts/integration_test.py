"""Integration test: run the full pipeline on a synthetic video with a
person-like moving rectangle. Verifies person detection + tracking +
event detectors end-to-end.

Usage:
    python scripts/integration_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from config.settings import settings
from models.model_manager import ModelManager
from detection.person_detector import PersonDetector
from detection.event_fusion import EventFusion
from video.file_source import FileSource


def make_test_video(path, frames=60, fps=10.0):
    """Create a video with a person-shaped blob moving across the frame."""
    h, w = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
        x = 50 + (i * 6) % (w - 120)
        y = 100 + int(30 * np.sin(i * 0.3))
        cv2.rectangle(frame, (x, y + 40), (x + 60, y + 200), (60, 60, 60), -1)
        cv2.circle(frame, (x + 30, y + 25), 22, (200, 180, 160), -1)
        writer.write(frame)
    writer.release()
    print("Created " + str(path) + " (" + str(frames) + " frames)")


def main():
    video_path = ROOT / "data" / "videos" / "integration_test.mp4"
    make_test_video(video_path)

    mm = ModelManager(settings.models_path)
    detector = PersonDetector(mm, conf=0.05, imgsz=320)

    class _Stub:
        def detect_and_track(self, frame, ts):
            return detector.detect_and_track(frame, ts)

        @property
        def active_tracks(self):
            return detector.active_tracks

    fusion = EventFusion(person_detector=_Stub(), model_manager=mm)

    src = FileSource(str(video_path), target_fps=10.0)
    assert src.open(), "FileSource open failed"
    print("Video: " + str(src.width) + "x" + str(src.height) +
          " @ " + str(src.source_fps) + " fps, " + str(src.frame_count) + " frames")

    total_tracks = set()
    total_events = 0
    frames_processed = 0
    for packet in src:
        events = fusion.process_frame(packet)
        frames_processed += 1
        for tid in detector.active_tracks:
            total_tracks.add(tid)
        total_events += len(events)
        if frames_processed <= 3:
            print("  frame " + str(frames_processed) +
                  " tracks=" + str(len(detector.active_tracks)) +
                  " events=" + str(len(events)))

    src.close()
    print("Frames processed: " + str(frames_processed))
    print("Unique track IDs: " + str(sorted(total_tracks)))
    print("Total events: " + str(total_events))
    assert len(total_tracks) > 0, "No persons tracked"
    print("INTEGRATION TEST PASSED")


if __name__ == "__main__":
    main()