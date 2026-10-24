import time
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("F:/EMS(CNV)/cctv-anomaly-detection/models/yolo11n.pt")
model.to("cpu")

cap = cv2.VideoCapture("F:/EMS(CNV)/cctv-anomaly-detection/data/videos/1792668913_demo2.mp4")
ret, frame = cap.read()
print(f"Frame shape: {frame.shape}")

for i in range(3):
    t0 = time.perf_counter()
    model.track(frame, conf=0.25, imgsz=640, device="cpu", persist=True, verbose=False, tracker="bytetrack.yaml")
    print(f"  full-frame inference {i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")

# Resize to 640 width maintaining aspect ratio
h, w = frame.shape[:2]
scale = 640 / w
resized = cv2.resize(frame, (640, max(1, int(h*scale))))
print(f"Resized shape: {resized.shape}")
for i in range(3):
    t0 = time.perf_counter()
    model.track(resized, conf=0.25, imgsz=640, device="cpu", persist=True, verbose=False, tracker="bytetrack.yaml")
    print(f"  resized inference {i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")

# Half resolution
half = cv2.resize(frame, (w//2, h//2))
print(f"Half shape: {half.shape}")
for i in range(3):
    t0 = time.perf_counter()
    model.track(half, conf=0.25, imgsz=640, device="cpu", persist=True, verbose=False, tracker="bytetrack.yaml")
    print(f"  half inference {i+1}: {(time.perf_counter()-t0)*1000:.1f}ms")