# CCTV Anomaly Detection - Model Manager
"""Lazy-loading model manager.

Models are NOT loaded at application startup. They are loaded on first use
and cached. Missing weights are reported explicitly rather than silently
substituted with random predictions.

Supported backends:
  - ultralytics.YOLO  (default, CPU via PyTorch)
  - onnxruntime       (optional, when an ONNX checkpoint is available)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lazy loading and caching of detection models."""

    def __init__(self, models_dir: Path) -> None:
        self._models_dir = Path(models_dir)
        self._cache: dict[str, Any] = {}
        self._backends: dict[str, str] = {}

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def _resolve_path(self, model_name: str) -> Path:
        """Resolve a model name to a local path, downloading if needed."""
        if not model_name:
            raise ValueError("Empty model name")
        path = Path(model_name)
        if path.is_absolute() and path.exists():
            return path
        local = self._models_dir / model_name
        if local.exists():
            return local
        # Allow ultralytics to download to its cache dir
        return Path(model_name)

    def get_yolo(self, model_name: str, *, task: str = "detect") -> Any:
        """Load (or return cached) a YOLO model.

        Raises:
            RuntimeError if ultralytics is not installed.
            FileNotFoundError if the model cannot be resolved.
        """
        cache_key = f"yolo:{model_name}:{task}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment check
            raise RuntimeError(
                "ultralytics is not installed. Install with: pip install -r requirements-cpu.txt"
            ) from exc

        resolved = self._resolve_path(model_name)
        logger.info("Loading YOLO model: %s (resolved: %s)", model_name, resolved)
        try:
            if resolved.suffix == ".onnx":
                model = YOLO(str(resolved), task=task)
            else:
                model = YOLO(str(resolved))
            # Force CPU device
            model.to("cpu")
            self._cache[cache_key] = model
            self._backends[cache_key] = "ultralytics"
            return model
        except Exception as exc:
            logger.error("Failed to load model %s: %s", model_name, exc)
            raise

    def get_onnx(self, model_path: str) -> Any:
        """Load an ONNX model via onnxruntime (optional backend)."""
        cache_key = f"onnx:{model_path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            ) from exc
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(model_path), providers=providers)
        self._cache[cache_key] = session
        self._backends[cache_key] = "onnx"
        return session

    def is_available(self, model_name: str) -> bool:
        """Return True if the named model can be resolved locally or via cache."""
        try:
            self._resolve_path(model_name)
            return True
        except Exception:
            return False

    def clear(self) -> None:
        """Free all cached models."""
        for model in self._cache.values():
            try:
                del model
            except Exception:
                pass
        self._cache.clear()
        self._backends.clear()
        logger.info("Model cache cleared")

    def status(self) -> dict[str, Any]:
        """Return a status report of loaded models."""
        return {
            "loaded": list(self._cache.keys()),
            "backends": dict(self._backends),
            "models_dir": str(self._models_dir),
        }