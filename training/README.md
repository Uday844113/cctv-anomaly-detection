# Training

This directory contains scripts to prepare datasets and train models.

## Dataset Preparation

Run the dataset prep helper:
```bash
python training/prepare_dataset.py --name fire_smoke --root data/datasets
```

Expected layout under `data/datasets/<name>/`:
```
images/    # raw frames or images
labels/    # YOLO-format .txt annotations
videos/    # source videos (optional)
dataset.yaml
```

### Annotation Guidelines

For each category, annotate a bounding box around the target:

- **fire**: tight box around visible flame.
- **smoke**: box around the visible smoke plume.
- **fall**: box around the falling person.
- **fight**: box around each person involved.
- **crowd_violence**: box around each person in the violent crowd.
- **abnormal_running**: box around the running person.

Use one `.txt` per image with lines:
```
<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
```

### Dataset Sources

Use publicly available, legally usable datasets. Documented sources include:
- **Fire**: [Fire Detection Dataset](https://github.com/deepak112/Fire-Detection-Dataset) (public).
- **Smoke**: [Smoke Detection Dataset](https://github.com/Xtra-PCI/SMOKE-DETECTION) (public).
- **Fighting**: [Violence Detection in Videos](https://github.com/mendelev/flw) or [UCF-Crime](https://github.com/zhongwen/UCF-Crime) (research use).
- **Fall**: [UR Fall Detection Dataset](https://github.com/zhongwen/URFD) (public).
- **Crowd violence**: [Crowd Violence Dataset](https://github.com/zhongwen/UCF-Crime) (research use).

Always verify licensing and attribution requirements before use.

## Training Object Detector

```bash
python training/train_object_detector.py \
    --dataset data/datasets/fire_smoke \
    --epochs 25 --imgsz 640 --batch 8 --device cpu
```

CPU-friendly defaults: `--imgsz 640 --batch 8 --epochs 25 --workers 2`.

## Training Behavior Classifier

```bash
python training/train_behavior_model.py \
    --dataset data/datasets/behavior \
    --epochs 10 --imgsz 224 --batch 4
```

## Evaluation

```bash
python training/evaluate.py --task object_detection \
    --weights runs/detect/train/weights/best.pt \
    --dataset data/datasets/fire_smoke
```

## Video-Level Split

Frames from the same source video are never split across train and validation
sets (see `training/prepare_dataset.py::split_dataset`).