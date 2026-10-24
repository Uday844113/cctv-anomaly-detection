"""Tests for configuration loading and validation."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config.settings import Settings, settings
from config import settings as settings_mod


def test_settings_defaults():
    s = Settings()
    assert s.app_name == "CCTV Anomaly Detection"
    assert s.frame_skip >= 1
    assert 0.0 <= s.fall_conf <= 1.0


def test_thresholds_yaml_loads():
    p = Path(settings_mod.PROJECT_ROOT) / "config" / "event_thresholds.yaml"
    data = yaml.safe_load(p.read_text())
    assert "thresholds" in data
    assert "fight" in data["thresholds"]
    assert "cooldown" in data
    assert "confirmation" in data


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FRAME_SKIP", "5")
    s = Settings()
    assert s.frame_skip == 5


def test_settings_invalid_conf():
    # frame_skip must be an integer; a non-numeric string should fail
    with pytest.raises(Exception):
        Settings(frame_skip="not-a-number")