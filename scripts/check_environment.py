"""Environment and dependency checker."""
from __future__ import annotations

import importlib
import platform
import sys
from typing import Optional

console = None
try:
    from rich.console import Console
    console = Console()
except Exception:
    console = None


def _print(msg: str) -> None:
    if console is not None:
        console.print(msg)
    else:
        print(msg)


def check_python() -> bool:
    v = sys.version_info
    ok = v >= (3, 11)
    _print(f"[{'OK' if ok else 'FAIL'}] Python {platform.python_version()} (need >=3.11)")
    return ok


def check_import(name: str, label: Optional[str] = None) -> bool:
    label = label or name
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "unknown")
        _print(f"[OK] {label} {ver}")
        return True
    except Exception as exc:
        _print(f"[FAIL] {label} not installed ({exc})")
        return False


def main() -> None:
    _print(f"Platform: {platform.platform()}")
    check_python()
    for mod in ["numpy", "cv2", "scipy", "sklearn", "pydantic", "yaml",
                "PIL", "psutil", "sqlalchemy", "alembic", "fastapi", "uvicorn",
                "streamlit", "requests", "pytest", "av"]:
        check_import(mod)
    try:
        import torch
        _print(f"[OK] torch {torch.__version__} (cuda={torch.cuda.is_available()})")
    except Exception as exc:
        _print(f"[FAIL] torch not installed ({exc})")
    try:
        from ultralytics import YOLO
        _print("[OK] ultralytics")
    except Exception as exc:
        _print(f"[FAIL] ultralytics ({exc})")


if __name__ == "__main__":
    main()