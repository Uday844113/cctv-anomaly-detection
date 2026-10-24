"""Smoke test: create a synthetic sample video and exercise FileSource.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from video.file_source import FileSource
from video.video_source import open_video_source


def main() -> None:
    out = ROOT / "data" / "videos" / "smoke_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = 120, 160
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 10.0, (w, h))
    for i in range(20):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = (i * 8) % (w - 30)
        cv2.rectangle(frame, (x, 40), (x + 30, 100), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    print(f"Created {out} ({out.stat().st_size} bytes)")

    src = FileSource(str(out))
    assert src.open(), "FileSource.open() failed"
    pkt = src.read()
    assert pkt is not None, "First read returned None"
    print(f"First frame shape: {pkt.frame.shape}, source_fps={src.source_fps}")

    count = 0
    for _ in src:
        count += 1
    print(f"Total frames read: {count}")
    src.close()
    assert not src.is_open, "Source did not close"

    # Test factory
    factory_src = open_video_source("file", str(out))
    assert factory_src.open(), "Factory open failed"
    factory_src.close()

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()