"""Prepare datasets for training.

This script documents the dataset layout and provides download instructions
for legitimate, publicly available sources. It does NOT download data
automatically - see the README for exact setup steps.

Dataset layout expected under data/datasets/<name>/:
    images/    - raw images or video frames
    labels/    - YOLO-format .txt annotations (one per image)
    dataset.yaml - class definitions

Annotation guidelines (per category):
    fire       - bounding box around visible flame
    smoke      - bounding box around visible smoke plume
    fall        - bounding box around the falling person
    fight       - bounding box around each person in altercation
    crowd_violence - bounding box around each person in violent crowd
    abnormal_running - bounding box around the running person
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def create_dataset_structure(name: str, root: Path) -> Path:
    ds = root / name
    for sub in ["images", "labels", "videos"]:
        (ds / sub).mkdir(parents=True, exist_ok=True)
    return ds


def write_dataset_yaml(path: Path, names: dict[int, str]) -> None:
    lines = [f"nc: {len(names)}", f"names: {names}"]
    path.write_text("\n".join(lines) + "\n")


def extract_frames(video_path: Path, out_dir: Path, *, fps: float = 5.0,
                   max_frames: Optional[int] = None) -> int:
    """Extract frames from a video at a target FPS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, int(round(src_fps / fps)))
    count = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            out = out_dir / f"{video_path.stem}_{count:06d}.jpg"
            cv2.imwrite(str(out), frame)
            count += 1
            if max_frames and count >= max_frames:
                break
        idx += 1
    cap.release()
    return count


def split_dataset(images_dir: Path, *, train_ratio: float = 0.8,
                  seed: int = 42) -> tuple[list, list]:
    """Video-level split: group frames by source video to prevent leakage."""
    rng = np.random.default_rng(seed)
    videos = sorted({p.name.rsplit("_", 1)[0] for p in images_dir.glob("*.jpg")})
    rng.shuffle(videos)
    n_train = int(len(videos) * train_ratio)
    train_videos = set(videos[:n_train])
    val_videos = set(videos[n_train:])
    train = [p for p in images_dir.glob("*.jpg") if p.name.rsplit("_", 1)[0] in train_videos]
    val = [p for p in images_dir.glob("*.jpg") if p.name.rsplit("_", 1)[0] in val_videos]
    return train, val


def class_distribution(labels_dir: Path, names: dict) -> dict:
    counts = {n: 0 for n in names.values()}
    for lbl in labels_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            counts[names.get(cid, str(cid))] = counts.get(names.get(cid, str(cid)), 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare anomaly detection datasets")
    ap.add_argument("--name", default="custom")
    ap.add_argument("--root", default=str(Path("data/datasets")))
    ap.add_argument("--extract", help="Extract frames from a video")
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()

    root = Path(args.root)
    ds = create_dataset_structure(args.name, root)
    print(f"Dataset dir: {ds}")

    if args.extract:
        n = extract_frames(Path(args.extract), ds / "images", fps=args.fps)
        print(f"Extracted {n} frames")

    # Write a starter YAML
    names = {0: "person", 1: "fire", 2: "smoke", 3: "fall", 4: "fight", 5: "crowd_violence", 6: "abnormal_running"}
    write_dataset_yaml(ds / "dataset.yaml", names)
    print("Wrote dataset.yaml")

    print("\nNext steps:")
    print("1. Add annotated images/labels following the guidelines in this docstring.")
    print("2. Run: python training/train_object_detector.py --dataset", ds)


if __name__ == "__main__":
    main()