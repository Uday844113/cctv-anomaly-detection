# CCTV Behavior & Anomaly Detection System

An AI-powered surveillance system that processes CCTV footage and live
webcam/RTSP streams to detect suspicious or dangerous events in near
real time. Built for **CPU-only** Windows 10/11 (and Linux).

## Detected Events

| Event            | Approach                                  |
|------------------|-------------------------------------------|
| Fight            | Pairwise proximity + relative motion heuristic |
| Fire             | Dedicated model or HSV color heuristic    |
| Smoke            | Dedicated model or HSV heuristic          |
| Person Falling   | Aspect ratio + vertical displacement      |
| Crowd Violence   | Multi-person density + motion ratio       |
| Abnormal Running | Image-plane speed with persistence        |

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # Linux

# 2. Install dependencies (CPU-only)
pip install -r requirements-cpu.txt
# PyTorch CPU wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Download person detection model
python scripts/download_models.py

# 4. Run the API backend
python main.py
# or: uvicorn backend.api:app --host 127.0.0.1 --port 8000

# 5. Run the dashboard
streamlit run dashboard/app.py
```

## Project Layout

```
cctv-anomaly-detection/
├── app.py                 # Streamlit launcher
├── main.py                # FastAPI launcher
├── requirements.txt       # Base requirements
├── requirements-cpu.txt   # CPU-verified pinned requirements
├── .env.example           # Environment template
├── .gitignore
├── README.md
├── config/                # settings.py + event_thresholds.yaml
├── models/                # model_manager.py + weights (not committed)
├── detection/             # person, fire/smoke, fight, fall, running, crowd, fusion
├── tracking/              # person_tracker.py (ByteTrack via YOLO)
├── video/                 # file_source, webcam_source, rtsp_source, frame_processor
├── backend/               # api.py, schemas.py, services.py
├── database/              # db.py, models.py, repository.py
├── dashboard/             # app.py + components.py
├── recording/             # incident_recorder.py, report_exporter.py
├── training/              # prepare_dataset, train_object/behavior, evaluate
├── scripts/               # check_environment, download_models, benchmark_cpu, install_deps
├── tests/                 # pytest suite
├── data/                  # videos, datasets, incidents (git-ignored)
└── logs/
```

## API Endpoints

| Method | Path                 | Description                |
|--------|----------------------|----------------------------|
| GET    | /health              | Service health             |
| GET    | /models/status       | Model loading state        |
| GET    | /events              | List confirmed events      |
| GET    | /events/{event_id}   | Single event               |
| POST   | /video/upload        | Upload a video file        |
| POST   | /streams/start       | Start a stream             |
| POST   | /streams/stop        | Stop a stream              |
| GET    | /streams/status      | Current stream status      |
| GET    | /metrics             | FPS, CPU, memory           |
| GET    | /config              | Current configuration      |
| PUT    | /config              | Update configuration       |
| GET    | /reports/export      | CSV/JSON report            |

OpenAPI docs are available at `/docs` when the API is running.

## Model Requirements

- **Person detection**: `yolo11n.pt` (auto-downloaded by `scripts/download_models.py`).
- **Fire/smoke**: Not included in COCO. A dedicated model must be trained
  (see `training/prepare_dataset.py`). Until then, the system uses an
  explicit HSV-based heuristic that is labeled as non-trained.
- **Fall/Fight/Crowd/Running**: Heuristic feature-based detectors are the
  default. Optional trained classifiers can be added via
  `training/train_behavior_model.py`.

## CPU Performance

The system is designed for CPU. Optimizations include:
- Configurable inference resolution (`INFERENCE_WIDTH/HEIGHT`).
- Frame skipping (`FRAME_SKIP`).
- Bounded queues with backpressure.
- Model caching (no reload per frame).
- Thread-safe capture and cleanup.

Expected performance on a typical laptop CPU: **3-10 FPS** at 640px with
`yolo11n.pt` and frame skip 2. Actual numbers vary by hardware. Run
`python scripts/benchmark_cpu.py --video data/videos/sample.mp4` to measure.

## Testing

```bash
pytest tests/ -v
```

Tests use mocked models and synthetic frames; no webcam or downloaded
weights are required for unit tests.

## Training

See `training/README.md` for dataset preparation, annotation guidelines,
and training commands.

## Deployment

See `DEPLOYMENT.md` for Windows setup, Docker, and production guidance.

## Disclaimer

This system provides **decision-support signals**, not proof of wrongdoing.
Image-plane speed is not physical speed without camera calibration.
False alarms are expected; thresholds should be tuned per environment.
No face recognition or identity matching is performed.