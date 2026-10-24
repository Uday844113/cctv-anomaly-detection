"""Install CPU-only dependencies in the correct order.

Usage:
    python scripts/install_deps.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CORE_PACKAGES = [
    "numpy", "opencv-python", "scipy", "scikit-learn",
    "pydantic", "pydantic-settings", "pyyaml", "pillow",
    "psutil", "python-multipart", "python-dotenv",
    "sqlalchemy", "alembic",
    "fastapi", "uvicorn[standard]", "streamlit", "httpx", "av",
    "pytest", "pytest-asyncio", "pytest-cov",
    "ultralytics",
]

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def run(cmd: list[str], **kwargs) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT), **kwargs)
    return proc.returncode


def main() -> int:
    py = sys.executable
    rc = run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if rc:
        return rc

    print("\n--- Installing core packages ---")
    rc = run([py, "-m", "pip", "install"] + CORE_PACKAGES)
    if rc:
        print("Core install failed; continuing with torch anyway")

    print("\n--- Installing PyTorch CPU wheels ---")
    rc = run([py, "-m", "pip", "install", "torch", "torchvision",
              "--index-url", TORCH_INDEX])
    if rc:
        print("PyTorch CPU install failed; trying default index")
        rc = run([py, "-m", "pip", "install", "torch", "torchvision"])
    return rc


if __name__ == "__main__":
    sys.exit(main())