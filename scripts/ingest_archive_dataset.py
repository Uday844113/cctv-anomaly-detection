"""Ingest the fall detection and aggressive-pose datasets into the project.

Usage:
    python scripts/ingest_archive_dataset.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"F:\EMS(CNV)")
FIRE_SMOKE_SRC = ROOT / "archive" / "data"
FALL_SRC = ROOT / "fall  detection"
FIGHT_SRC = ROOT / "fight" / "Aggressive_Poses_Dataset"

DST_ROOT = Path(r"F:\EMS(CNV)\cctv-anomaly-detection\data\datasets")


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ingest_fire_smoke() -> None:
    dst = DST_ROOT / "fire_smoke"
    _copy_tree(FIRE_SMOKE_SRC, dst)
    shutil.copy(ROOT / "archive" / "data.yaml", dst / "data.yaml")
    n_train = len(list((dst / "train" / "images").glob("*.jpg")))
    n_test = len(list((dst / "test" / "images").glob("*.jpg")))
    print(f"fire_smoke: train={n_train} test={n_test}")


def ingest_fall() -> None:
    dst = DST_ROOT / "fall"
    _copy_tree(FALL_SRC, dst)
    n_train = len(list((dst / "train" / "images").glob("*.jpg")))
    n_val = len(list((dst / "valid" / "images").glob("*.jpg")))
    n_test = len(list((dst / "test" / "images").glob("*.jpg")))
    print(f"fall: train={n_train} val={n_val} test={n_test}")
    # Fix the YAML to use absolute paths
    yaml = dst / "data.yaml"
    yaml.write_text(
        f"train: {dst / 'train' / 'images'}\n"
        f"val: {dst / 'valid' / 'images'}\n"
        f"test: {dst / 'test' / 'images'}\n"
        f"nc: 2\n"
        f"names: ['Fall Detected', 'NoFall Detected']\n"
    )
    print(f"Wrote {yaml}")


def ingest_fight() -> None:
    dst = DST_ROOT / "fight"
    _copy_tree(FIGHT_SRC, dst)
    n = len(list((dst / "images").glob("*.jpg")))
    print(f"fight: images={n}")
    yaml = dst / "data.yaml"
    yaml.write_text(
        f"train: {dst / 'images'}\n"
        f"val: {dst / 'images'}\n"
        f"nc: 1\n"
        f"names: ['person']\n"
        f"kpt_shape: [17, 3]\n"
    )
    print(f"Wrote {yaml}")


def main() -> None:
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    ingest_fire_smoke()
    ingest_fall()
    ingest_fight()
    print("\nAll datasets ingested.")


if __name__ == "__main__":
    main()