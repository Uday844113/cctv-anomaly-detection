"""Download YOLO person detector weights from HuggingFace mirrors.

GitHub release assets are unreachable from this network, so this script
tries several HuggingFace mirrors before giving up.

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEST = ROOT / "models" / "yolo11n.pt"

CANDIDATES = [
    "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt",
    "https://huggingface.co/Ultralytics/YOLO11n/resolve/main/yolo11n.pt",
    "https://huggingface.co/neulab/YOLO11/resolve/main/yolo11n.pt",
    "https://huggingface.co/neulab/YOLO11n/resolve/main/yolo11n.pt",
]


def _progress(total):
    mb = total / 1024.0 / 1024.0
    msg = "\r  %.1f MB" % mb
    sys.stdout.write(msg)
    sys.stdout.flush()


def download(url, dest, timeout=120):
    print("Trying " + url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                _progress(total)
        print("\n  OK: " + str(dest) + " (" + str(total) + " bytes)")
        return True
    except Exception as exc:
        print("\n  FAIL: " + type(exc).__name__ + ": " + str(exc))
        return False


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size > 1000000:
        print("Already present: " + str(DEST) + " (" + str(DEST.stat().st_size) + " bytes)")
        return 0

    for url in CANDIDATES:
        if download(url, DEST):
            return 0

    print("\nAll mirrors failed. Manual install options:")
    print("  1. Copy yolo11n.pt into " + str(DEST))
    print("  2. Use a different network and run: python scripts/download_models.py")
    print("  3. The system runs in heuristic mode without this weight.")
    return 1


if __name__ == "__main__":
    sys.exit(main())