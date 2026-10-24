"""Test EventFusion.process_frame directly with a real frame."""
import logging, sys, time, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.insert(0, "F:/EMS(CNV)/cctv-anomaly-detection")

import cv2
import numpy as np
from config.settings import settings
from models.model_manager import ModelManager
from detection.person_detector import PersonDetector
from detection.event_fusion import EventFusion
from video.video_source import FramePacket

mm = ModelManager(settings.models_path)
det = PersonDetector(mm)
print("pre-loading model...")
t0 = time.perf_counter()
det._ensure_model()
print(f"model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

cap = cv2.VideoCapture("F:/EMS(CNV)/cctv-anomaly-detection/data/videos/1792669156_demo2.mp4")
ret, frame = cap.read()
print(f"frame: {frame.shape if ret else None}")

fusion = EventFusion(person_detector=det, model_manager=mm)
packet = FramePacket(frame=frame, frame_index=1, source_id="test",
                     timestamp=1.0, source_fps=25.0)

print("calling fusion.process_frame...")
t0 = time.perf_counter()
try:
    events = fusion.process_frame(packet)
    print(f"process_frame returned {len(events)} events in {(time.perf_counter()-t0)*1000:.0f}ms")
    print("active_tracks:", len(det.active_tracks))
except Exception as e:
    print(f"process_frame FAILED in {(time.perf_counter()-t0)*1000:.0f}ms: {e}")
    traceback.print_exc()