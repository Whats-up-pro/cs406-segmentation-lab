import numpy as np
import pytest
from PIL import Image

from backend.baseline import baseline_probability


@pytest.mark.parametrize('mode', ['RGB', 'RGBA', 'L'])
def test_returns_original_shape_finite_float32_scores(mode):
    image = Image.new(mode, (73, 41))
    score = baseline_probability(image)
    assert score.shape == (41, 73)
    assert score.dtype == np.float32
    assert np.isfinite(score).all()
    assert np.all((0 <= score) & (score <= 1))


def test_uniform_neutral_image_has_no_foreground():
    assert not baseline_probability(Image.new('RGB', (64, 64), '#b0b0b0')).any()


def test_detects_red_region_from_image_without_reference_mask():
    pixels = np.full((96, 96, 3), 180, dtype=np.uint8)
    pixels[25:71, 25:71] = [220, 80, 90]
    image = Image.fromarray(pixels)
    first = baseline_probability(image)
    assert first[35:60, 35:60].mean() > 0.95
    assert first[:15, :15].mean() < 0.05
    np.testing.assert_array_equal(first, baseline_probability(image))


def test_evaluation_discovers_variable_annotations_and_uses_strict_majority(tmp_path):
    from scripts.audit_data import discover_samples
    from scripts.evaluate import load_reference

    (tmp_path / 'imagesTest').mkdir()
    (tmp_path / 'labelsTest').mkdir()
    Image.new('RGB', (2, 2)).save(tmp_path / 'imagesTest/IMG_ENC00001_00001.jpg')
    first = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    second = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    for suffix, array in [(2, first), (3, second)]:
        Image.fromarray(array).save(tmp_path / f'labelsTest/IMG_ENC00001_00001_mask_ann{suffix}.tiff')
    samples = discover_samples(tmp_path, 'test')
    assert len(samples) == 1
    assert len(samples[0]['masks']) == 2
    consensus, annotations = load_reference(samples[0], (2, 2))
    np.testing.assert_array_equal(consensus, [[True, False], [False, False]])
    assert len(annotations) == 2
