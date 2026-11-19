"""Portable supervised pixel classifier; inference reads JSON, never pickle."""

from functools import lru_cache
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / 'results/pixel_classifier/model.json'
MODEL_FORMAT = 'cs406-pixel-logistic-v1'
FEATURE_NAMES = ('red', 'green', 'blue', 'lab_l', 'lab_a', 'lab_b',
                 'hsv_saturation', 'gray', 'gray_mean_7', 'gray_std_7')
FEATURE_SCALING = {
    'color_and_gray': 'OpenCV uint8 channels / 127.5 - 1',
    'gray_mean_7': '7x7 mean of gray/255, then *2-1; BORDER_REFLECT_101',
    'gray_std_7': 'sqrt(max(E[(gray/255)^2]-E[gray/255]^2,0))*2; 7x7 BORDER_REFLECT_101',
}


def pixel_features(image: Image.Image, input_size: int = 256) -> np.ndarray:
    """Ten fixed features, HxWx10 float32; no learned preprocessing or labels."""
    if not isinstance(input_size, int) or input_size < 2:
        raise ValueError('input_size must be an integer of at least 2.')
    rgb = np.asarray(image.convert('RGB').resize((input_size, input_size), Image.Resampling.BILINEAR))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray, -1, (7, 7), borderType=cv2.BORDER_REFLECT_101)
    square_mean = cv2.boxFilter(gray * gray, -1, (7, 7), borderType=cv2.BORDER_REFLECT_101)
    std = np.sqrt(np.maximum(square_mean - mean * mean, 0))
    return np.concatenate((rgb.astype(np.float32) / 127.5 - 1,
                           lab.astype(np.float32) / 127.5 - 1,
                           (saturation.astype(np.float32) / 127.5 - 1)[..., None],
                           (gray * 2 - 1)[..., None], (mean * 2 - 1)[..., None],
                           (std * 2)[..., None]), axis=-1).astype(np.float32)


def probabilities_from_features(features: np.ndarray, coefficients, intercept: float) -> np.ndarray:
    """Stable sigmoid, preserving feature-map shape and float32 output."""
    logits = np.sum(features * np.asarray(coefficients, dtype=np.float32), axis=-1) + np.float32(intercept)
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))).astype(np.float32)


@lru_cache(maxsize=4)
def _load_model(path_string: str, modified_ns: int, size: int) -> dict:
    del modified_ns
    if size > 1024 * 1024:
        raise ValueError('Pixel model JSON is unexpectedly large.')
    model = json.loads(Path(path_string).read_text(encoding='utf-8'))
    if model.get('format') != MODEL_FORMAT or model.get('features') != list(FEATURE_NAMES):
        raise ValueError('Unsupported pixel model format or feature definition.')
    if model.get('scaling') != FEATURE_SCALING:
        raise ValueError('Unsupported pixel feature scaling.')
    coefficients = np.asarray(model.get('coefficients'), dtype=np.float64)
    intercept = model.get('intercept')
    threshold = model.get('threshold')
    input_size = model.get('input_size')
    if (coefficients.shape != (len(FEATURE_NAMES),) or not np.isfinite(coefficients).all()
            or not isinstance(intercept, (int, float)) or not np.isfinite(intercept)
            or not isinstance(threshold, (int, float)) or not 0 < threshold < 1
            or not isinstance(input_size, int) or not 2 <= input_size <= 2048):
        raise ValueError('Pixel model contains invalid coefficients or dimensions.')
    return model


def load_pixel_model(model_path: Path | None = None) -> dict:
    """Lazy load with schema validation; a replaced file invalidates the cache."""
    path = Path(model_path or os.environ.get('PIXEL_MODEL_PATH') or DEFAULT_MODEL_PATH).resolve()
    stat = path.stat()
    return _load_model(str(path), stat.st_mtime_ns, stat.st_size)


def predict_pixel_model(image: Image.Image, model_path: Path | None = None) -> np.ndarray:
    """Return HxW foreground probabilities in the original image dimensions."""
    model = load_pixel_model(model_path)
    features = pixel_features(image, model['input_size'])
    probability = probabilities_from_features(features, model['coefficients'], model['intercept'])
    if image.size != (model['input_size'], model['input_size']):
        probability = np.asarray(Image.fromarray(probability).resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32)
    return probability


def pixel_model_status(model_path: Path | None = None) -> dict:
    """Describe model availability and recorded training provenance without sklearn."""
    path = Path(model_path or os.environ.get('PIXEL_MODEL_PATH') or DEFAULT_MODEL_PATH)
    status = {'name': 'Supervised pixel logistic regression', 'model_found': path.is_file(),
              'checkpoint_found': path.is_file(), 'inference_ready': False,
              'model_name': path.name, 'error': None}
    if not path.is_file():
        return status
    try:
        model = load_pixel_model(path)
        status.update(inference_ready=True, threshold=model['threshold'], input_size=model['input_size'],
                      features=model['features'], training_origin=model.get('training_origin', {}),
                      trained_origin='Trained locally from supplied train split only; validation selected threshold.')
    except (OSError, ValueError, TypeError) as error:
        status['error'] = f'{type(error).__name__}: {error}'
    return status
