"""Report and CSV export."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


class ReportExporter:
    """Exports event reports to CSV and JSON."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, events: Iterable, path: Optional[Path] = None) -> Path:
        path = path or (self.output_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        rows = []
        for ev in events:
            d = ev if isinstance(ev, dict) else ev.to_dict()
            rows.append({
                "event_id": d.get("event_id", ""),
                "event_type": d.get("event_type", ""),
                "timestamp": d.get("timestamp", ""),
                "confidence": d.get("confidence", ""),
                "source_id": d.get("source_id", ""),
                "track_id": d.get("track_id", ""),
                "bbox": d.get("bbox", ""),
                "evidence": json.dumps(d.get("evidence", {})),
            })
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["event_id"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Exported CSV: %s", path)
        return path

    def export_json(self, events: Iterable, path: Optional[Path] = None) -> Path:
        path = path or (self.output_dir / f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        data = []
        for ev in events:
            d = ev if isinstance(ev, dict) else ev.to_dict()
            data.append(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported JSON: %s", path)
        return path