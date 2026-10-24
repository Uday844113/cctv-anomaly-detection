"""Test each step of fusion.process_frame separately."""
import logging, sys, time, traceback
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.insert(0, "F:/EMS(CNV)/cctv-anomaly-detection")

import cv2
from config.settings import settings
from models.model_manager import ModelManager
from detection.person_detector import PersonDetector
from detection.event_fusion import EventFusion
from detection.fight_detector import FightDetector
from detection.fall_detector import FallDetector
from detection.crowd_violence_detector import CrowdViolenceDetector
from detection.running_detector import RunningDetector
from detection.fire_smoke_detector import FireSmokeDetector
from video.video_source import FramePacket

mm = ModelManager(settings.models_path)
det = PersonDetector(mm)
print("pre-loading model...")
t0 = time.perf_counter()
det._ensure_model()
print(f"model loaded in {(time.perf_counter()-t0)*1000:.0f}ms")

cap = cv2.VideoCapture("F:/EMS(CNV)/cctv-anomaly-detection/data/videos/1792669156_demo2.mp4")
ret, frame = cap.read()
print(f"frame: {frame.shape}")

packet = FramePacket(frame=frame, frame_index=1, source_id="test",
                     timestamp=1.0, source_fps=25.0)

# Test each detector individually
print("\n--- PersonDetector.detect_and_track ---")
t0 = time.perf_counter()
tracks = det.detect_and_track(frame, 1.0)
print(f"  {len(tracks)} tracks in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- FireSmokeDetector ---")
fsd = FireSmokeDetector(model_manager=mm, model_name="runs/detect/fire_smoke/weights/best.pt")
t0 = time.perf_counter()
evs = fsd.detect(packet, {"tracks": tracks, "frame": frame})
print(f"  {len(evs)} events in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- FightDetector ---")
fd = FightDetector()
t0 = time.perf_counter()
evs = fd.detect(packet, {"tracks": tracks, "frame": frame})
print(f"  {len(evs)} events in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- FallDetector ---")
falld = FallDetector()
t0 = time.perf_counter()
evs = falld.detect(packet, {"tracks": tracks, "frame": frame})
print(f"  {len(evs)} events in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- CrowdViolenceDetector ---")
cvd = CrowdViolenceDetector()
t0 = time.perf_counter()
evs = cvd.detect(packet, {"tracks": tracks, "frame": frame})
print(f"  {len(evs)} events in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- RunningDetector ---")
rd = RunningDetector()
t0 = time.perf_counter()
evs = rd.detect(packet, {"tracks": tracks, "frame": frame})
print(f"  {len(evs)} events in {(time.perf_counter()-t0)*1000:.0f}ms")

print("\n--- Full fusion.process_frame ---")
fusion = EventFusion(person_detector=det, model_manager=mm)
t0 = time.perf_counter()
events = fusion.process_frame(packet)
print(f"  {len(events)} events in {(time.perf_counter()-t0)*1000:.0f}ms")