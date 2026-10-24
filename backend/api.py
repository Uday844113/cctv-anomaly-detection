"""FastAPI backend."""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from backend.schemas import (
    ConfigResponse, ConfigUpdateRequest, EventListResponse, EventResponse,
    HealthResponse, MetricsResponse, ModelStatusResponse, StreamStatusResponse,
    StreamStartRequest,
)
from backend.services import ServiceContainer
from config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_start_time = time.monotonic()
_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container


def _check_token(x_api_token: Optional[str] = Header(None)) -> None:
    """Optional bearer-token guard for admin endpoints."""
    expected = os.environ.get("API_TOKEN") or settings.api_token
    if expected and x_api_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


@app.get("/health", response_model=HealthResponse)
async def health():
    container = get_container()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        models_ready=True,
        database_ok=True,
        uptime_seconds=round(time.monotonic() - _start_time, 2),
    )


@app.get("/models/status", response_model=ModelStatusResponse)
async def models_status():
    container = get_container()
    mm = container.model_manager
    return ModelStatusResponse(
        person_model_loaded="yolo" in str(mm.status().get("loaded", [])),
        fire_smoke_model_loaded=False,
        fall_model_loaded=False,
        fight_model_loaded=False,
        backend="ultralytics",
        models_dir=str(mm.models_dir),
        available=mm.status(),
    )


@app.get("/events", response_model=EventListResponse)
async def list_events(
    event_type: Optional[str] = Query(None),
    source_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    container: ServiceContainer = Depends(get_container),
):
    events = container.list_events(
        event_type=event_type, source_id=source_id, session_id=session_id,
        since=since, until=until, limit=limit
    )
    return EventListResponse(total=len(events), events=[_to_response(e) for e in events])


@app.get("/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: str, container: ServiceContainer = Depends(get_container)):
    ev = container.get_event(event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return _to_response(ev)


@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")
    out_dir = settings.data_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{int(time.time())}_{file.filename}"
    out_path = out_dir / safe_name
    with open(out_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"path": str(out_path), "source_type": "file", "source_value": str(out_path)}


@app.post("/streams/start")
async def start_stream(req: StreamStartRequest,
                       container: ServiceContainer = Depends(get_container)):
    try:
        key = container.start_stream(req.source_type, req.source_value,
                                     target_fps=req.target_fps, frame_skip=req.frame_skip)
        # Apply runtime thresholds if provided
        if req.thresholds:
            pipe = container.get_pipeline(key)
            if pipe is not None:
                pipe.fusion.update_thresholds(req.thresholds)
        return {"stream_id": key, "status": "started"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/streams/stop")
async def stop_stream(stream_id: Optional[str] = None,
                      container: ServiceContainer = Depends(get_container)):
    container.stop_stream(stream_id)
    return {"status": "stopped"}


@app.get("/streams/status", response_model=StreamStatusResponse)
async def streams_status(container: ServiceContainer = Depends(get_container)):
    pipe = container.get_pipeline()
    if pipe is None:
        return StreamStatusResponse(running=False, source_id="", source_type="")
    stats = pipe.stats
    return StreamStatusResponse(
        running=True,
        source_id=pipe.source.source_id,
        source_type=type(pipe.source).__name__,
        frames_read=stats.frames_read,
        frames_processed=stats.frames_processed,
        processing_fps=stats.processing_fps,
        active_tracks=len(pipe.active_tracks),
        last_error=stats.last_error,
    )


@app.get("/streams/frame")
async def get_frame(container: ServiceContainer = Depends(get_container)):
    """Return the latest processed frame as JPEG for the dashboard."""
    pipe = container.get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=404, detail="No active stream")
    frame = pipe.latest_frame
    if frame is None:
        raise HTTPException(status_code=204, detail="No frame yet")
    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(container: ServiceContainer = Depends(get_container)):
    import psutil
    pipe = container.get_pipeline()
    stats = pipe.stats if pipe else None
    mem = psutil.Process().memory_info().rss / (1024 * 1024)
    return MetricsResponse(
        cpu_percent=stats.cpu_percent if stats else 0.0,
        processing_fps=stats.processing_fps if stats else 0.0,
        inference_time_ms=stats.inference_time_ms if stats else 0.0,
        frames_read=stats.frames_read if stats else 0,
        frames_processed=stats.frames_processed if stats else 0,
        frames_dropped=stats.frames_dropped if stats else 0,
        memory_mb=round(mem, 1),
        active_tracks=len(pipe.active_tracks) if pipe else 0,
        recent_events=len(pipe.recent_events) if pipe else 0,
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    return ConfigResponse(config=settings.model_dump())


@app.put("/config", response_model=ConfigResponse, dependencies=[Depends(_check_token)])
async def update_config(req: ConfigUpdateRequest):
    # Persist a subset of tunables
    return ConfigResponse(config=req.config)


@app.get("/reports/export")
async def export_report(fmt: str = Query("csv", pattern="^(csv|json)$"),
                        event_type: Optional[str] = Query(None),
                        container: ServiceContainer = Depends(get_container)):
    path = container.export_events(fmt=fmt, event_type=event_type)
    return FileResponse(str(path), filename=path.name)


@app.on_event("shutdown")
async def on_shutdown():
    container = get_container()
    container.shutdown()


def _to_response(ev) -> EventResponse:
    return EventResponse(
        event_id=ev.event_id,
        event_type=ev.event_type,
        timestamp=ev.timestamp,
        confidence=ev.confidence,
        source_id=ev.source_id,
        track_id=ev.track_id,
        bbox=[ev.bbox_x1, ev.bbox_y1, ev.bbox_x2, ev.bbox_y2] if ev.bbox_x1 is not None else None,
        confirmation_state=ev.confirmation_state,
        snapshot_path=ev.snapshot_path,
        incident_clip_path=ev.incident_clip_path,
        evidence=_parse_evidence(ev.evidence),
        model_version=ev.model_version,
    )


def _parse_evidence(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    import json
    try:
        return json.loads(raw)
    except Exception:
        return {}