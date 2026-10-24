"""Split fire_smoke dataset into train/val/test using video-level grouping."""
from __future__ import annotations

import random
import shutil
from pathlib import Path

DS = Path("F:/EMS(CNV)/cctv-anomaly-detection/data/datasets/fire_smoke")
SRC = DS / "train"
DST = DS

random.seed(42)

# Group label files by their source video prefix (strip trailing _NNNNNN)
def stem(p: Path) -> str:
    return p.name.rsplit("_", 1)[0]

labels = sorted((SRC / "labels").glob("*.txt"))
videos = sorted({stem(p) for p in labels})
random.shuffle(videos)

n_train = int(len(videos) * 0.8)
n_val = int(len(videos) * 0.1)
train_set = set(videos[:n_train])
val_set = set(videos[n_train:n_train + n_val])
test_set = set(videos[n_train + n_val:])

print(f"Videos: {len(videos)} total, {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

for split, vset in [("train", train_set), ("valid", val_set), ("test", test_set)]:
    for sub in ["images", "labels"]:
        (DST / split / sub).mkdir(parents=True, exist_ok=True)

# Move files
for lbl in labels:
    v = stem(lbl)
    if v in train_set:
        continue  # already in train
    if v in val_set:
        split = "valid"
    elif v in test_set:
        split = "test"
    else:
        continue
    # Move label
    dst_lbl = DST / split / "labels" / lbl.name
    shutil.move(str(lbl), str(dst_lbl))
    # Move matching image
    img_name = lbl.name.replace(".txt", ".jpg")
    img = SRC / "images" / img_name
    if img.exists():
        dst_img = DST / split / "images" / img.name
        shutil.move(str(img), str(dst_img))

print("Done splitting.")
for split in ["train", "valid", "test"]:
    imgs = (DST / split / "images").glob("*.jpg")
    lbls = (DST / split / "labels").glob("*.txt")
    print(f"  {split}: {len(list(imgs))} images, {len(list(lbls))} labels")