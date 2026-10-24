"""Tests for event detectors and fusion (mocked models)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from detection._base import EventCandidate, BaseEventDetector
from detection.fall_detector import FallDetector
from detection.fight_detector import FightDetector
from detection.running_detector import RunningDetector
from detection.crowd_violence_detector import CrowdViolenceDetector
from detection.person_detector import Track
from video.video_source import FramePacket


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _packet(frame=None, idx=1, ts=1.0):
    return FramePacket(frame=frame or _frame(), frame_index=idx,
                      source_id="test", timestamp=ts)


def _track(tid=1, bbox=(100, 100, 150, 200), vel=(0, 0), accel=0.0):
    t = Track(track_id=tid, bbox=list(bbox), confidence=0.9,
              center=((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2),
              velocity=vel, aspect_ratio=(bbox[2]-bbox[0])/max(1, (bbox[3]-bbox[1])),
              height=bbox[3]-bbox[1], last_seen=1.0, first_seen=0.0, accel=accel)
    return t


def test_fight_detector_requires_two_tracks():
    det = FightDetector()
    pkt = _packet()
    cands = det.detect(pkt, {"tracks": [_track(tid=1)]})
    assert cands == []


def test_fight_detector_proximity_and_motion():
    det = FightDetector(confidence_threshold=0.5, min_overlap_duration_s=0.0)
    t1 = _track(tid=1, bbox=(100, 100, 140, 200), vel=(80, 0), accel=300)
    t2 = _track(tid=2, bbox=(110, 100, 150, 200), vel=(-80, 0), accel=300)
    pkt = _packet(ts=2.0)
    cands = det.detect(pkt, {"tracks": [t1, t2]})
    assert len(cands) >= 1
    assert cands[0].event_type == "fight"


def test_fall_detector_fast_downward():
    det = FallDetector(confidence_threshold=0.5, min_track_frames=3,
                       fall_speed_px_per_s=50)
    # Build history: person drops from y=100 to y=300 quickly
    hist = []
    for i, (ts, cy, h) in enumerate([(0.0, 100, 150), (0.5, 150, 140), (1.0, 250, 100), (1.5, 300, 80)]):
        t = _track(tid=1, bbox=(100, cy - h/2, 140, cy + h/2), vel=(0, 200), accel=0)
        hist.append((t, ts, cy, h))
    # Manually inject history
    det._history[1] = __import__("collections").deque(
        [(ts, cy, h) for _, ts, cy, h in hist], maxlen=15)
    pkt = _packet(ts=1.5)
    cands = det.detect(pkt, {"tracks": [hist[-1][0]]})
    assert any(c.event_type == "fall" for c in cands)


def test_running_detector_persistence():
    det = RunningDetector(confidence_threshold=0.4, speed_threshold_px_per_s=50,
                          min_persistence_s=0.0, min_track_frames=3)
    t = _track(tid=1, vel=(120, 0), accel=0)
    for ts in [0.0, 0.5, 1.0]:
        det._history.setdefault(1, __import__("collections").deque(maxlen=30)).append((ts, 120.0))
    pkt = _packet(ts=1.0)
    cands = det.detect(pkt, {"tracks": [t]})
    assert any(c.event_type == "abnormal_running" for c in cands)


def test_crowd_violence_requires_min_persons():
    det = CrowdViolenceDetector(min_persons=4, confidence_threshold=0.5)
    tracks = [_track(tid=i, bbox=(i*10, 100, i*10+40, 200), accel=200) for i in range(3)]
    pkt = _packet()
    cands = det.detect(pkt, {"tracks": tracks, "frame": _frame()})
    assert cands == []


def test_crowd_violence_detects_with_motion():
    det = CrowdViolenceDetector(min_persons=4, confidence_threshold=0.5,
                                aggression_motion_ratio=0.3, density_ratio=0.05)
    tracks = [_track(tid=i, bbox=(i*30, 100, i*30+40, 200), accel=200) for i in range(5)]
    pkt = _packet()
    cands = det.detect(pkt, {"tracks": tracks, "frame": _frame()})
    assert len(cands) == 1
    assert cands[0].event_type == "crowd_violence"


def test_event_candidate_to_dict():
    ev = EventCandidate(event_type="fire", confidence=0.8, timestamp=1.0,
                        frame_index=5, source_id="cam", bbox=[1, 2, 3, 4])
    d = ev.to_dict()
    assert d["event_type"] == "fire"
    assert d["confidence"] == 0.8


def test_base_detector_cooldown():
    det = FallDetector(confidence_threshold=0.0, cooldown_seconds=10.0,
                       confirmation_seconds=0.5, min_track_frames=2,
                       fall_speed_px_per_s=1, min_height_drop_ratio=0.0,
                       max_aspect_ratio=10.0)
    t = _track(tid=1, bbox=(100, 300, 140, 320), vel=(0, 0))
    pkt1 = _packet(ts=1.0)
    det._history[1] = __import__("collections").deque(
        [(0.0, 100, 200), (0.5, 120, 190)], maxlen=15)
    first = det.process(pkt1, {"tracks": [t]})
    # Need 2 sustained candidates within the confirmation window to confirm
    pkt2 = _packet(ts=1.2, idx=2)
    second = det.process(pkt2, {"tracks": [t]})
    assert len(first) + len(second) >= 1
    # Cooldown should suppress a third immediately after
    third = det.process(_packet(ts=1.21, idx=3), {"tracks": [t]})
    assert third == []