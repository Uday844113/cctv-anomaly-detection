# CCTV Anomaly Detection - Models
This directory holds pretrained model weights and the lazy-loading manager.

## Layout
- `model_manager.py` - Lazy loader with CPU backend selection.
- `weights/` - Downloaded checkpoints (NOT committed to version control).

## Supported weights
| Purpose            | Default name     | Source                              | Notes                                   |
|--------------------|------------------|-------------------------------------|-----------------------------------------|
| Person detection   | yolo11n.pt       | ultralytics/YOLO11                  | Lightweight CPU-friendly.               |
| Fire/smoke         | (custom)         | See training/prepare_dataset.py     | Not in COCO; needs dedicated weights.   |
| Fall classifier    | (custom)         | training/train_behavior_model.py    | Optional; heuristic fallback if absent.|
| Fight classifier   | (custom)         | training/train_behavior_model.py    | Optional; heuristic fallback if absent.|

## Download
Run:  python scripts/download_models.py

Never commit `.pt`, `.onnx`, or `.pth` files. They are in `.gitignore`.