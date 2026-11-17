"""Lazy, bounded and thread-safe inference for the supplied TransUNet checkpoint."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "TransUNet_R50_ViT_B_16_all_best_model.pth"
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_MODEL = None
_MODEL_KEY = None
_RUNTIME = {}
_LOAD_ERROR = None
NORMALIZATIONS = ("unit", "imagenet")


def _checkpoint_path() -> Path:
    return Path(os.environ.get("TRANSUNET_CHECKPOINT", str(_DEFAULT_CHECKPOINT))).expanduser().resolve()


def get_model_status() -> dict:
    """Inventory metadata without importing torch or reading checkpoint tensors.

    ``available`` means checkpoint/dependency presence, not verified quality.
    ``checkpoint_verified`` becomes true only after a successful strict load.
    CUDA availability is null until torch has been imported by an inference call.
    """
    path = _checkpoint_path()
    exists = path.is_file()
    installed = importlib.util.find_spec("torch") is not None
    try:
        version = importlib.metadata.version("torch") if installed else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    stat = path.stat() if exists else None
    loaded = _MODEL is not None and _MODEL_KEY is not None and _MODEL_KEY[:3] == (
        str(path), stat.st_mtime_ns if stat else None, stat.st_size if stat else None)
    torch_module = sys.modules.get("torch")
    # sys.modules exposes a module before its import has finished in another thread.
    cuda_probe = getattr(getattr(torch_module, "cuda", None), "is_available", None)
    cuda_available = bool(cuda_probe()) if callable(cuda_probe) else None
    return {
        "id": "transunet", "architecture": "R50-ViT-B_16", "task": "binary segmentation",
        "checkpoint": path.name, "checkpoint_path": str(path),
        "checkpoint_bytes": stat.st_size if stat else None,
        "checkpoint_present": exists, "torch_installed": installed,
        "available": bool(exists and installed), "loaded": loaded,
        "checkpoint_verified": loaded,
        "input_size": [_MODEL.image_size, _MODEL.image_size] if loaded else None,
        "expected_input_size": [224, 224],
        "normalization_options": list(NORMALIZATIONS),
        "normalization_verified": False,
        "preprocessing_note": "Original training preprocessing was not supplied; choose unit or imagenet explicitly.",
        "runtime": {"torch_version": version, "cuda_available": cuda_available,
                    "requested_device": os.environ.get("TRANSUNET_DEVICE", "auto"), **(_RUNTIME if loaded else {})},
        "load_error": _LOAD_ERROR,
    }


def _validate_normalization(normalization: str) -> None:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {', '.join(NORMALIZATIONS)}")


def preprocess_image(image: Image.Image, image_size: int, normalization: str = "unit") -> torch.Tensor:
    """Convert PIL input to float32 RGB NCHW, bilinear-resized to model resolution."""
    _validate_normalization(normalization)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    import torch

    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized, dtype=np.float32) / np.float32(255.0)
    if normalization == "imagenet":
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).unsqueeze(0)


def probabilities_from_logits(logits: torch.Tensor, original_size: tuple[int, int]) -> np.ndarray:
    """Apply sigmoid then bilinear-restore probabilities; size is PIL (width,height)."""
    import torch
    from torch.nn import functional as F

    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("Expected binary logits of shape (1, 1, height, width)")
    if not torch.isfinite(logits).all().item():
        raise ValueError("Model logits must be finite")
    probabilities = torch.sigmoid(logits.float())
    width, height = original_size
    probabilities = F.interpolate(probabilities, size=(height, width), mode="bilinear", align_corners=False)
    return probabilities[0, 0].clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32, copy=False)


def _get_model():
    global _MODEL, _MODEL_KEY, _RUNTIME, _LOAD_ERROR
    with _LOAD_LOCK:
        path = _checkpoint_path()
        if not path.is_file():
            raise FileNotFoundError(f"TransUNet checkpoint not found: {path}")
        import torch
        from .transunet import load_transunet_state_dict

        requested_device = os.environ.get("TRANSUNET_DEVICE", "auto").lower()
        if requested_device not in ("auto", "cpu", "cuda"):
            raise ValueError("TRANSUNET_DEVICE must be auto, cpu or cuda")
        device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; set TRANSUNET_DEVICE=cpu or install a CUDA-enabled torch build")
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size, device)
        if _MODEL is not None and key == _MODEL_KEY:
            return _MODEL, device
        try:
            cpu_threads = max(1, min(8, int(os.environ.get("TRANSUNET_CPU_THREADS", "4")), os.cpu_count() or 1))
            torch.set_num_threads(cpu_threads)
            started = time.perf_counter()
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = load_transunet_state_dict(state).to(device).eval()
            _RUNTIME = {"device": device, "cpu_threads": cpu_threads,
                        "load_seconds": round(time.perf_counter() - started, 4),
                        "parameter_count": sum(parameter.numel() for parameter in model.parameters())}
            _MODEL, _MODEL_KEY, _LOAD_ERROR = model, key, None
        except Exception as error:
            _LOAD_ERROR = f"{type(error).__name__}: {error}"
            raise
        return _MODEL, device


def predict_transunet(image: Image.Image, normalization: str = "unit") -> np.ndarray:
    """Return float32 lesion probabilities in the ORIGINAL image's H x W shape.

    This runs the supplied weights, with one cached model per process. The
    training normalization is unknown; unit and imagenet are explicit options.
    Requests share an inference lock to bound simultaneous CPU/GPU memory use.
    """
    _validate_normalization(normalization)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    import torch

    with _INFERENCE_LOCK:
        model, device = _get_model()
        tensor = preprocess_image(image, model.image_size, normalization).to(device)
        with torch.inference_mode():
            logits = model(tensor)
            return probabilities_from_logits(logits, image.size)
