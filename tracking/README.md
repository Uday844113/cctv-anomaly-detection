# CCTV Anomaly Detection - Tracking
This directory holds person tracking logic.

## Layout
- `person_tracker.py` - ByteTrack-based tracker with CPU-friendly settings.

## Notes
- ByteTrack is bundled with `ultralytics` since v8.0. We use the built-in
  `ByteTrack` tracker through YOLO's `.track()` API to avoid an extra dependency.
- Track state (position, velocity, trajectory) is maintained per-track ID.