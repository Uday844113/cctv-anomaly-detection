"""Tests for video sources."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from video.file_source import FileSource, SUPPORTED_EXTENSIONS
from video.webcam_source import WebcamSource
from video.rtsp_source import RTSPSource, _redact_url
from video.video_source import open_video_source, VideoSource


def _make_video(path: Path, frames: int = 10, fps: float = 10.0):
    h, w = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), i % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_file_source_open_close(tmp_path):
    p = tmp_path / "test.mp4"
    _make_video(p)
    src = FileSource(str(p))
    assert src.open() is True
    pkt = src.read()
    assert pkt is not None
    assert pkt.frame.shape == (64, 64, 3)
    src.close()
    assert src.is_open is False


def test_file_source_missing_file(tmp_path):
    src = FileSource(str(tmp_path / "nope.mp4"))
    assert src.open() is False


def test_file_source_unsupported_extension(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("nope")
    src = FileSource(str(p))
    assert src.open() is False


def test_file_source_loop(tmp_path):
    p = tmp_path / "loop.mp4"
    _make_video(p, frames=5)
    src = FileSource(str(p), loop=True)
    assert src.open()
    for _ in range(12):
        pkt = src.read()
        assert pkt is not None
    src.close()


def test_file_source_max_frames(tmp_path):
    p = tmp_path / "max.mp4"
    _make_video(p, frames=20)
    src = FileSource(str(p), max_frames=5)
    assert src.open()
    count = 0
    for pkt in src:
        count += 1
    assert count == 5
    src.close()


def test_webcam_invalid_index():
    src = WebcamSource(camera_index=9999)
    assert src.open() is False


def test_redact_url():
    assert _redact_url("rtsp://user:password@10.0.0.1/stream") == "rtsp://user:***@10.0.0.1/stream"
    assert _redact_url("rtsp://10.0.0.1/stream") == "rtsp://10.0.0.1/stream"


def test_open_video_source_factory(tmp_path):
    p = tmp_path / "factory.mp4"
    _make_video(p)
    src = open_video_source("file", str(p))
    assert isinstance(src, FileSource)
    assert src.open()
    src.close()


def test_open_video_source_unknown():
    with pytest.raises(ValueError):
        open_video_source("unknown", "x")


def test_file_source_reads_real_frames(tmp_path):
    """A real MP4 written by OpenCV must yield multiple readable frames."""
    p = tmp_path / "real.mp4"
    _make_video(p, frames=15, fps=25.0)
    src = FileSource(str(p))
    assert src.open() is True
    assert src.width == 64
    assert src.height == 64
    assert abs(src.source_fps - 25.0) < 1.0
    count = 0
    for pkt in src:
        count += 1
        assert pkt.frame.shape == (64, 64, 3)
        assert pkt.source_fps is not None
    assert count == 15
    src.close()


def test_file_source_path_with_spaces(tmp_path):
    """Windows paths with spaces must open correctly."""
    p = tmp_path / "my video 2024.mp4"
    _make_video(p)
    src = FileSource(str(p))
    assert src.open() is True
    pkt = src.read()
    assert pkt is not None
    src.close()


def test_file_source_absolute_path(tmp_path):
    p = tmp_path / "abs.mp4"
    _make_video(p)
    src = FileSource(str(p.resolve()))
    assert src.open() is True
    src.close()


def test_open_video_source_with_spaces(tmp_path):
    p = tmp_path / "path with spaces.mp4"
    _make_video(p)
    src = open_video_source("file", str(p))
    assert src.open()
    src.close()


def test_dashboard_source_selector_saves_upload(tmp_path, monkeypatch):
    """Simulate the dashboard saving uploaded bytes to a safe directory."""
    from dashboard import components
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "data_dir", str(tmp_path))

    video_bytes = _make_video_bytes(tmp_path / "seed.mp4")

    class FakeUpload:
        name = "clip.mp4"
        def getvalue(self):
            return video_bytes

    saved_path = components._save_upload(FakeUpload())
    assert Path(saved_path).exists()
    assert Path(saved_path).suffix == ".mp4"

    src = open_video_source("file", saved_path)
    assert src.open()
    pkt = src.read()
    assert pkt is not None
    assert pkt.frame.shape == (64, 64, 3)
    src.close()


def test_dashboard_source_selector_missing_path(tmp_path, monkeypatch):
    """A non-existent local path must fail to open."""
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "data_dir", str(tmp_path))

    missing = str(tmp_path / "does_not_exist.mp4")
    src = open_video_source("file", missing)
    assert src.open() is False


def test_dashboard_source_selector_unsupported_extension(tmp_path, monkeypatch):
    from config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "data_dir", str(tmp_path))

    bad = tmp_path / "virus.txt"
    bad.write_text("nope")
    src = open_video_source("file", str(bad))
    assert src.open() is False


def _make_video_bytes(path: Path, frames: int = 5, fps: float = 10.0) -> bytes:
    import io
    h, w = 64, 64
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), i % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()
    return path.read_bytes()