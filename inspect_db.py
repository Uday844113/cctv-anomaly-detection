import sqlite3

conn = sqlite3.connect('F:/EMS(CNV)/cctv-anomaly-detection/data/data/cctv_events.db')
c = conn.cursor()
print('=== Tables ===')
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
for r in c.fetchall():
    print(r[0])
print()
print('=== Events by type ===')
c.execute('SELECT event_type, COUNT(*) FROM events GROUP BY event_type')
for r in c.fetchall():
    print(r)
print()
print('=== Events by session ===')
c.execute('SELECT session_id, event_type, COUNT(*) FROM events GROUP BY session_id, event_type ORDER BY session_id')
for r in c.fetchall():
    print(r)
print()
print('=== Sample smoke events ===')
c.execute('SELECT event_id, event_type, confidence, session_id, source_id, track_id, datetime(timestamp) FROM events WHERE event_type="smoke" LIMIT 10')
for r in c.fetchall():
    print(r)
print()
print('=== All smoke events ===')
c.execute('SELECT event_id, confidence, session_id, source_id, track_id, datetime(timestamp) FROM events WHERE event_type="smoke"')
for r in c.fetchall():
    print(r)
conn.close()