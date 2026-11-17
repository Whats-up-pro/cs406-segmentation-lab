"""Deterministic image-only redness segmentation, with no trained parameters."""

import cv2
import numpy as np
from PIL import Image


BASELINE_PROTOCOL = {
    'name': 'lab_a_otsu_morphology_v1',
    'input': 'RGB image only; no annotation or checkpoint access',
    'color_space': 'OpenCV uint8 CIELAB a channel (neutral=128)',
    'smoothing': '5x5 Gaussian blur, sigma=0',
    'threshold': 'a > max(per-image Otsu threshold, 130); empty if a range < 3',
    'morphology': '5x5 ellipse opening once then closing twice',
    'small_components': 'remove 8-connected components below max(16, 0.001 * image pixels)',
    'output': 'float32 binary score map, 0 or 1; not calibrated probabilities',
    'training': 'none; fixed rules, no ground-truth parameter tuning',
}


def baseline_probability(image: Image.Image) -> np.ndarray:
    """Return a finite float32 HxW binary score map at original resolution.

    Higher LAB a values indicate redness. Otsu separates image-specific color
    groups; a fixed neutral floor avoids calling gray background a lesion.
    This deliberately simple classical comparator does not identify a disease.
    """
    rgb = np.asarray(image.convert('RGB'))
    redness = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 1]
    redness = cv2.GaussianBlur(redness, (5, 5), 0)
    if int(redness.max()) - int(redness.min()) < 3:
        return np.zeros(redness.shape, dtype=np.float32)
    threshold, _ = cv2.threshold(redness, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = (redness > max(130.0, threshold)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    accepted = np.zeros(count, dtype=np.float32)
    if count > 1:
        accepted[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(16, mask.size * 0.001)
    return accepted[labels]
