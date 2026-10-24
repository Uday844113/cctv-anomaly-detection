"""Test PipelineService directly."""
import logging, sys, time
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.insert(0, "F:/EMS(CNV)/cctv-anomaly-detection")

from config.settings import settings
from database.db import get_session_factory, reset_engine
from models.model_manager import ModelManager
from backend.services import PipelineService
from video.file_source import FileSource

# Fresh DB
reset_engine()

src = FileSource("F:/EMS(CNV)/cctv-anomaly-detection/data/videos/1792669156_demo2.mp4")
mm = ModelManager(settings.models_path)
svc = PipelineService(src, model_manager=mm)
svc.start()
print("started, session_id:", svc.session_id)

for i in range(6):
    time.sleep(2)
    s = svc.stats
    print(f"  [{i}] read={s.frames_read} processed={s.frames_processed} "
          f"fps={s.processing_fps} tracks={len(svc.active_tracks)} err={s.last_error!r}")

print("events:", len(svc.recent_events))
svc.stop()
src.close()
print("final:", svc.stats.to_dict())