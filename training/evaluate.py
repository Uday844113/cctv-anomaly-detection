"""Evaluate trained models with video-level metrics.

Usage:
    python training/evaluate.py --dataset data/datasets/fire_smoke \\
        --weights runs/detect/train/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_object_detector(weights: str, dataset: str, *, imgsz: int = 640,
                             device: str = "cpu") -> dict:
    from ultralytics import YOLO
    ds = Path(dataset)
    yaml = ds / "dataset.yaml"
    if not yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found in {ds}")
    model = YOLO(weights)
    metrics = model.val(data=str(yaml), imgsz=imgsz, device=device, verbose=False)
    box = getattr(metrics, "box", None)
    result = {
        "task": "object_detection",
        "weights": weights,
        "map50_95": float(getattr(box, "map", 0.0)) if box else 0.0,
        "map50": float(getattr(box, "map50", 0.0)) if box else 0.0,
        "precision": float(getattr(box, "mp", 0.0)) if box else 0.0,
        "recall": float(getattr(box, "mr", 0.0)) if box else 0.0,
    }
    logger.info("Object detection metrics: %s", result)
    return result


def evaluate_behavior(model_path: str, dataset: str, *, imgsz: int = 224) -> dict:
    """Evaluate the behavior classifier with per-class accuracy + confusion matrix."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from training.train_behavior_model import build_model, load_clip_frames

    ds = Path(dataset)
    class_names = sorted(d.name for d in ds.iterdir() if d.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class dirs in {ds}")

    class ClipDataset(Dataset):
        def __init__(self, root):
            self.samples = []
            self.class_to_idx = {n: i for i, n in enumerate(class_names)}
            for cn in class_names:
                for clip in (root / cn).iterdir():
                    if clip.is_dir():
                        self.samples.append((clip, self.class_to_idx[cn]))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            clip, label = self.samples[i]
            arr = load_clip_frames(clip, imgsz)
            if arr.shape[0] == 0:
                arr = np.zeros((16, 3, imgsz, imgsz), dtype=np.float32)
            return torch.from_numpy(arr).float(), label

    model = build_model(len(class_names), imgsz)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    loader = DataLoader(ClipDataset(ds), batch_size=4, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    cm = np.zeros((len(class_names), len(class_names)), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            preds = out.argmax(dim=1).numpy()
            for p, t in zip(preds, y):
                cm[t, p] += 1
                total += 1
                if p == t:
                    correct += 1
    per_class = {}
    for i, n in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[n] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(cm[i, :].sum())}
    result = {
        "task": "behavior_classification",
        "accuracy": round(correct / max(1, total), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }
    logger.info("Behavior metrics: %s", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["object_detection", "behavior"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()
    if args.task == "object_detection":
        result = evaluate_object_detector(args.weights, args.dataset, imgsz=args.imgsz)
    else:
        result = evaluate_behavior(args.weights, args.dataset, imgsz=args.imgsz)
    Path(args.out).write_text(json.dumps(result, indent=2))
    logger.info("Metrics written to %s", args.out)


if __name__ == "__main__":
    main()