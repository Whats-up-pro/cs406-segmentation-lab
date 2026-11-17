"""Foreground segmentation scores with explicit empty-mask behavior."""

import numpy as np


METRIC_NAMES = ('dice', 'iou', 'pixel_accuracy', 'precision', 'recall')


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Score equal-sized, nonempty 2-D masks; nonzero pixels are foreground.

    Threshold probability maps before calling. Both masks empty yields 1 for
    every score. If just one mask is empty all foreground scores are 0; pixel
    accuracy still measures actual pixel agreement. Nonfinite values, shape
    mismatches, empty arrays, and extra dimensions are errors.
    """
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or not prediction.size:
        raise ValueError('Masks must be nonempty 2-D arrays of the same shape.')
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError('Masks must contain finite values.')
    predicted, actual = prediction.astype(bool), target.astype(bool)
    intersection = int(np.count_nonzero(predicted & actual))
    predicted_count, actual_count = int(predicted.sum()), int(actual.sum())
    union = predicted_count + actual_count - intersection
    if union == 0:
        return dict.fromkeys(METRIC_NAMES, 1.0)
    return {
        'dice': 2.0 * intersection / (predicted_count + actual_count),
        'iou': intersection / union,
        'pixel_accuracy': float(np.mean(predicted == actual)),
        'precision': intersection / predicted_count if predicted_count else 0.0,
        'recall': intersection / actual_count if actual_count else 0.0,
    }
