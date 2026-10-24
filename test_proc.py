"""Standalone test: verify FrameProcessor reads + processes frames."""
import logging, sys, time
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.insert(0, "F:/EMS(CNV)/cctv-anomaly-detection")

from video.file_source import FileSource
from video.frame_processor import FrameProcessor

src = FileSource("F:/EMS(CNV)/cctv-anomaly-detection/data/videos/1792669156_demo2.mp4")
print("opened:", src.open(), "fps:", src.source_fps, "frames:", src.frame_count)

count = {"n": 0}
def on_frame(packet):
    count["n"] += 1
    if count["n"] <= 3:
        print(f"  on_frame #{count['n']} frame_index={packet.frame_index} shape={packet.frame.shape}")

proc = FrameProcessor(source=src, frame_skip=1, on_frame=on_frame)
proc.start()
time.sleep(15)
print("stats:", proc.stats.to_dict())
proc.stop()
src.close()
print("callback count:", count["n"])