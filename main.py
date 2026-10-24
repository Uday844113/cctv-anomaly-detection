"""Entry point for the FastAPI backend."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from config.settings import settings

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )