# Deployment Guide

## Windows 10/11 (CPU-only)

### 1. Python and virtual environment
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install dependencies
```bash
pip install -r requirements-cpu.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If the PyTorch CPU index is unreachable, fall back to the default PyPI build:
```bash
pip install torch torchvision
```

### 3. Download models
```bash
python scripts/download_models.py
```

### 4. Run the backend
```bash
python main.py
# or
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

OpenAPI docs: http://127.0.0.1:8000/docs

### 5. Run the dashboard
```bash
streamlit run dashboard/app.py
```

### 6. Test
```bash
pytest tests/ -v
```

### 7. Benchmark CPU
```bash
python scripts/benchmark_cpu.py --video data/videos/sample.mp4 --frames 100
```

## Webcam

```bash
python -c "from video.webcam_source import WebcamSource; s=WebcamSource(0); print('open:', s.open())"
```

Camera index 0 is the built-in webcam; 1+ are USB webcams. If a camera is
unavailable, the source reports an error and the pipeline stays idle.

## RTSP

```bash
python -c "from video.rtsp_source import RTSPSource; s=RTSPSource('rtsp://user:pass@10.0.0.55/stream'); print('open:', s.open())"
```

Credentials are never logged (URLs are redacted). Set `RTSP_TIMEOUT` and
`CAMERA_RECONNECT_DELAY` in `.env` to tune reconnection behavior.

## Docker

```bash
docker compose up --build
```

See `docker-compose.yml` and `Dockerfile`. The image is CPU-only and installs
PyTorch from the official CPU index.

## Linux

Same steps as Windows, using the bash shell:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-cpu.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Production Deployment

1. Set `DEBUG=false`, `HOST=0.0.0.0`, and a strong `SECRET_KEY`.
2. Enable authentication via `API_TOKEN` for the PUT /config endpoint.
3. Place the database on persistent storage.
4. Set retention policy (`RETENTION_DAYS`) and schedule periodic cleanup.
5. Use a reverse proxy (nginx) with TLS termination.
6. Keep model weights outside the Git repository (see `.gitignore`).

## Troubleshooting

| Issue                                   | Fix                                              |
|-----------------------------------------|--------------------------------------------------|
| `ModuleNotFoundError: No module 'cv2'`  | `pip install opencv-python`                      |
| `torch` build fails on Python 3.13      | Use prebuilt wheels: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| OpenCV can't open webcam                | Check camera index, permissions, and drivers     |
| RTSP stream unavailable                 | Verify URL, network, and `rtsp_transport=tcp`    |
| Out of memory                          | Reduce `INFERENCE_WIDTH/HEIGHT` and `FRAME_SKIP` |
| Missing model weights                  | Run `python scripts/download_models.py`          |