import numpy as np
import pytest

from backend.metrics import segmentation_metrics


def test_counts_overlap_instead_of_averaging_pixels():
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    assert segmentation_metrics(prediction, target) == {
        'dice': 0.5, 'iou': 1 / 3, 'pixel_accuracy': 0.5,
        'precision': 0.5, 'recall': 0.5,
    }


def test_both_empty_masks_have_perfect_agreement():
    mask = np.zeros((3, 4), dtype=bool)
    assert set(segmentation_metrics(mask, mask).values()) == {1.0}


@pytest.mark.parametrize('prediction,target', [
    (np.zeros((2, 2)), np.ones((2, 2))),
    (np.ones((2, 2)), np.zeros((2, 2))),
])
def test_one_empty_mask_has_zero_foreground_scores(prediction, target):
    assert set(segmentation_metrics(prediction, target).values()) == {0.0}


@pytest.mark.parametrize('prediction,target', [
    (np.zeros((2, 3)), np.zeros((3, 2))),
    (np.zeros((2, 2, 1)), np.zeros((2, 2, 1))),
    (np.zeros((0, 2)), np.zeros((0, 2))),
    (np.array([[np.nan]]), np.ones((1, 1))),
])
def test_rejects_ambiguous_or_invalid_masks(prediction, target):
    with pytest.raises(ValueError):
        segmentation_metrics(prediction, target)


def test_nonzero_label_values_are_foreground():
    assert segmentation_metrics(np.array([[0, 255]]), np.array([[0, 1]]))['dice'] == 1
