"""Tests for incident recording and report export."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from recording.incident_recorder import IncidentRecorder
from recording.report_exporter import ReportExporter
from detection._base import EventCandidate


def test_incident_recorder_snapshot(tmp_path):
    rec = IncidentRecorder(tmp_path, pre_event_frames=5, post_event_frames=5, fps=10.0)
    for i in range(10):
        rec.push_frame(np.zeros((10, 10, 3), dtype=np.uint8), float(i))
    ev = EventCandidate(event_type="fire", confidence=0.9, timestamp=5.0,
                        frame_index=5, source_id="cam")
    rec.on_event(ev)
    # Push post frames
    for i in range(10, 15):
        rec.push_frame(np.zeros((10, 10, 3), dtype=np.uint8), float(i))
    rec.flush()
    files = list(tmp_path.iterdir())
    assert any("fire" in f.name for f in files)
    assert any("snapshot" in f.name for f in files)


def test_report_exporter_csv(tmp_path):
    exp = ReportExporter(tmp_path)
    events = [EventCandidate(event_type="fire", confidence=0.9, timestamp=1.0,
                             frame_index=1, source_id="cam")]
    path = exp.export_csv(events)
    assert path.exists()
    content = path.read_text()
    assert "fire" in content
    assert "confidence" in content


def test_report_exporter_json(tmp_path):
    exp = ReportExporter(tmp_path)
    events = [EventCandidate(event_type="fall", confidence=0.8, timestamp=2.0,
                             frame_index=2, source_id="cam")]
    path = exp.export_json(events)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data[0]["event_type"] == "fall"