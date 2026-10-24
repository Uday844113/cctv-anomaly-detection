"""Train a YOLO object detector for fire/smoke/person/fall.

Usage:
    python training/train_object_detector.py --dataset data/datasets/fire_smoke \\
        --epochs 25 --imgsz 640 --batch 8
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def train(dataset: str, *, epochs: int = 25, imgsz: int = 640, batch: int = 8,
           device: str = "cpu", workers: int = 2, resume: Optional[str] = None,
           project: str = "runs/detect", name: str = "train") -> str:
    ds = Path(dataset)
    yaml = ds / "dataset.yaml"
    if not yaml.exists():
        yaml = ds / "data.yaml"
    if not yaml.exists():
        raise FileNotFoundError(f"dataset.yaml/data.yaml not found in {ds}")
    # Prefer the local yolo11n.pt to avoid network timeouts
    base_model = str(ROOT / "models" / "yolo11n.pt")
    if not Path(base_model).exists():
        base_model = "yolo11n.pt"
    model = YOLO(resume or base_model)
    logger.info("Starting training: %s epochs, imgsz=%d, batch=%d, device=%s",
                epochs, imgsz, batch, device)
    results = model.train(
        data=str(yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True,
    )
    out_dir = Path(project) / name
    logger.info("Training complete. Saved to %s", out_dir)
    # Save metrics summary
    metrics = {
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "output_dir": str(out_dir),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(metrics, indent=2))
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="train")
    args = ap.parse_args()
    train(args.dataset, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
          device=args.device, workers=args.workers, resume=args.resume,
          project=args.project, name=args.name)


if __name__ == "__main__":
    main()