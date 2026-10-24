import requests, time, sys

API = "http://127.0.0.1:8000"

r = requests.get(f"{API}/health", timeout=10)
print("1. health:", r.status_code, r.json().get("status"))

path = r"F:\EMS(CNV)\cctv-anomaly-detection\data\videos\1792669156_demo2.mp4"
r = requests.post(f"{API}/streams/start", json={
    "source_type": "file",
    "source_value": path,
    "thresholds": {"fight": 0.7, "fire": 0.65, "smoke": 0.65, "fall": 0.7,
                   "crowd_violence": 0.7, "abnormal_running": 0.7}
}, timeout=30)
print("2. start:", r.status_code, r.json())
if r.status_code >= 400:
    print("FAILED TO START:", r.text)
    sys.exit(1)
key = r.json().get("stream_id")

for i in range(8):
    time.sleep(2)
    s = requests.get(f"{API}/streams/status", timeout=5).json()
    print(f"3. status[{i}]: running={s.get('running')} read={s.get('frames_read')} "
          f"processed={s.get('frames_processed')} fps={s.get('processing_fps')} "
          f"tracks={s.get('active_tracks')} err={s.get('last_error')}")
    if s.get("frames_processed", 0) > 0:
        break

ev = requests.get(f"{API}/events", params={"limit": 10, "session_id": key}, timeout=5).json()
print("4. events:", ev.get("total"), [(e.get("event_type"), e.get("timestamp")) for e in ev.get("events", [])])

fr = requests.get(f"{API}/streams/frame", timeout=5)
print("5. frame:", fr.status_code, "bytes:", len(fr.content) if fr.content else 0)

r = requests.post(f"{API}/streams/stop", json={"stream_id": key}, timeout=10)
print("6. stop:", r.status_code, r.json())