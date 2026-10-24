import requests
r = requests.post("http://127.0.0.1:8000/streams/stop", json={}, timeout=10)
print(r.status_code, r.json())