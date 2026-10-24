"""Streamlit dashboard components."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

from config.settings import settings

logger = logging.getLogger(__name__)


def _save_upload(uploaded) -> str:
    """Save an uploaded file's bytes to the project data directory."""
    out_dir = settings.data_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{int(time.time())}_{uploaded.name}"
    out_path = out_dir / safe_name
    out_path.write_bytes(uploaded.getvalue())
    return str(out_path)


def status_badge(label: str, ok: bool) -> str:
    color = "green" if ok else "red"
    return f"<span style='color:{color};font-weight:bold'>{'●' if ok else '●'}</span> {label}"


def metric_card(title: str, value, delta=None) -> str:
    return f"""<div style="border:1px solid #333;border-radius:6px;padding:8px;margin:4px">
<b>{title}</b><br><span style="font-size:1.2em">{value}</span>
{f'<br><span style="color:{"green" if (delta or 0)>=0 else "red"}">{delta}</span>' if delta is not None else ''}
</div>"""


def event_table(events: list) -> None:
    if not events:
        st.info("No events yet")
        return
    rows = []
    for ev in events:
        d = ev if isinstance(ev, dict) else ev.to_dict()
        rows.append({
            "Time": d.get("timestamp", ""),
            "Type": d.get("event_type", ""),
            "Conf": d.get("confidence", ""),
            "Track": d.get("track_id", ""),
            "Source": d.get("source_id", ""),
        })
    st.dataframe(rows, width="stretch", hide_index=False)


def threshold_editor() -> dict:
    """Render threshold controls and return current values."""
    st.subheader("Event thresholds")
    cols = st.columns(3)
    keys = ["fight", "fire", "smoke", "fall", "crowd_violence", "abnormal_running"]
    out = {}
    for i, k in enumerate(keys):
        with cols[i % 3]:
            out[k] = st.slider(k, 0.0, 1.0, 0.6, 0.05, key=f"thr_{k}")
    return out


def source_selector() -> tuple[str, str]:
    st.subheader("Video source")
    source_type = st.selectbox("Source type", ["file", "webcam", "rtsp"], index=0)
    if source_type == "file":
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            value = _save_upload(uploaded)
            st.success(f"Saved upload to {value}")
        else:
            value = ""
        st.caption("Or use a local file path:")
        value = st.text_input("File path", value=value or "")
    elif source_type == "webcam":
        value = str(st.selectbox("Camera index", [0, 1, 2, 3], index=0))
    else:
        value = st.text_input("RTSP URL", "")
        st.caption("Leave password redacted in logs; never share the URL.")
    return source_type, value


def draw_overlays(frame: np.ndarray, tracks: dict, events: list) -> np.ndarray:
    """Draw bounding boxes, track IDs, and event labels on a frame."""
    import cv2
    out = frame.copy()
    for tid, t in tracks.items():
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"ID:{tid}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    for ev in events:
        bbox = ev.get("bbox") if isinstance(ev, dict) else ev.bbox
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        etype = ev.get("event_type", "event") if isinstance(ev, dict) else ev.event_type
        conf = ev.get("confidence", 0) if isinstance(ev, dict) else ev.confidence
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"{etype}:{conf:.2f}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out
