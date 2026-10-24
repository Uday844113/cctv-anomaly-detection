"""Tests for the FastAPI backend."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend import api
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings_mod.settings, "database_url", f"sqlite:///{tmp_path}/test.db")
    monkeypatch.setattr(api, "_container", None)
    monkeypatch.setattr(api, "_start_time", time.monotonic())

    fake_container = MagicMock()
    fake_container.list_events.return_value = []
    fake_container.get_event.return_value = None
    fake_container.start_stream.return_value = "fake-key"
    fake_container.get_pipeline.return_value = None
    fake_container.export_events.return_value = tmp_path / "report.csv"
    fake_container.model_manager.status.return_value = {"loaded": [], "backends": {}}

    monkeypatch.setattr(api, "get_container", lambda: fake_container)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_status(client):
    r = client.get("/models/status")
    assert r.status_code == 200
    assert "backend" in r.json()


def test_list_events_empty(client):
    r = client.get("/events")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_get_event_not_found(client):
    r = client.get("/events/nonexistent")
    assert r.status_code == 404


def test_config_get(client):
    r = client.get("/config")
    assert r.status_code == 200
    assert "config" in r.json()


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "processing_fps" in r.json()