"""Train a lightweight behavior classifier (fight / fall / running).

Uses a 3D-CNN or ConvNet over short clip stacks. CPU-friendly defaults:
small image size, small batch, few epochs. This is a development fallback;
heuristic detectors in detection/ are used when weights are absent.

Usage:
    python training/train_behavior_model.py --dataset data/datasets/behavior \\
        --epochs 10 --imgsz 224 --batch 4
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_clip_frames(clip_dir: Path, imgsz: int = 224) -> np.ndarray:
    import cv2
    frames = []
    for p in sorted(clip_dir.glob("*.jpg")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (imgsz, imgsz))
        frames.append(img.transpose(2, 0, 1))  # CHW
    if not frames:
        return np.zeros((0, 3, imgsz, imgsz), dtype=np.float32)
    arr = np.stack(frames).astype(np.float32) / 255.0
    return arr


def build_model(num_classes: int, imgsz: int = 224, clip_len: int = 16):
    """Build a tiny 3D-CNN with PyTorch (CPU)."""
    import torch
    import torch.nn as nn

    class BehaviorNet(nn.Module):
        def __init__(self, num_classes: int, imgsz: int, clip_len: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1, stride=(1, 2, 2)),
                nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1, stride=(1, 2, 2)),
                nn.ReLU(), nn.MaxPool3d(kernel_size=(1, 2, 2)),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
            )
            self.classifier = nn.Linear(32, num_classes)

        def forward(self, x):
            # x: (B, C, T, H, W) -> Conv3d expects (B, C, T, H, W)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    return BehaviorNet(num_classes, imgsz, clip_len)


def train(dataset: str, *, epochs: int = 10, imgsz: int = 224, batch: int = 4,
          device: str = "cpu", lr: float = 1e-3) -> str:
    import torch
    from torch.utils.data import Dataset, DataLoader

    ds = Path(dataset)
    class_dirs = [d for d in ds.iterdir() if d.is_dir()]
    class_names = sorted(d.name for d in class_dirs)
    if not class_dirs:
        raise FileNotFoundError(f"No class dirs in {ds}")

    class ClipDataset(Dataset):
        def __init__(self, root: Path, class_names, imgsz):
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
            # Pad/trim to fixed clip length
            if arr.shape[0] == 0:
                arr = np.zeros((16, 3, imgsz, imgsz), dtype=np.float32)
            return torch.from_numpy(arr).float(), label

    dataset_obj = ClipDataset(ds, class_names, imgsz)
    loader = DataLoader(dataset_obj, batch_size=batch, shuffle=True, num_workers=0)
    model = build_model(len(class_names), imgsz)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, torch.tensor(y, dtype=torch.long))
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        logger.info("Epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(1, len(loader)))
    out_dir = Path("runs/behavior")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "behavior_model.pt")
    (out_dir / "class_names.json").write_text(json.dumps(class_names, indent=2))
    logger.info("Saved behavior model to %s", out_dir)
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args.dataset, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
          device=args.device, lr=args.lr)


if __name__ == "__main__":
    main()