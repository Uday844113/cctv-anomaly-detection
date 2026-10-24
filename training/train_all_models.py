"""Train YOLO models for fire/smoke and fall detection.

Usage:
    python training/train_all_models.py [--epochs 15] [--imgsz 320]
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASETS = {
    "fire_smoke": ROOT / "data" / "datasets" / "fire_smoke",
    "fall": ROOT / "data" / "datasets" / "fall",
}


def run_train(name: str, ds: Path, *, epochs: int, imgsz: int, batch: int) -> int:
    cmd = [
        sys.executable, str(ROOT / "training" / "train_object_detector.py"),
        "--dataset", str(ds),
        "--epochs", str(epochs),
        "--imgsz", str(imgsz),
        "--batch", str(batch),
        "--device", "cpu",
        "--workers", "0",
        "--project", str(ROOT / "runs" / "detect"),
        "--name", name,
    ]
    logger.info("Starting: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--models", nargs="+", default=list(DATASETS.keys()))
    args = ap.parse_args()

    rc = 0
    for name in args.models:
        ds = DATASETS[name]
        if not ds.exists():
            logger.warning("Dataset not found: %s", ds)
            continue
        rc = run_train(name, ds, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
        if rc != 0:
            logger.error("Training failed for %s", name)
    logger.info("All training done (last rc=%s)", rc)


if __name__ == "__main__":
    main()