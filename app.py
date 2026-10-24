"""Convenience launcher for the Streamlit dashboard."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    os.execvp(
        sys.executable,
        [sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
         "--server.address", "127.0.0.1", "--server.port", "8501"],
    )