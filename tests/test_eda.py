from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts.eda import collect_eda


def make_sample(root: Path, split: str, sample_id: str, mask: np.ndarray):
    suffix = {'train': 'Train', 'valid': 'Valid', 'test': 'Test'}[split]
    image_dir, mask_dir = root / f'images{suffix}', root / f'labels{suffix}'
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    Image.new('RGB', (mask.shape[1], mask.shape[0]), '#909090').save(image_dir / f'{sample_id}.png')
    Image.fromarray(mask).save(mask_dir / f'{sample_id}.tiff')


def test_native_pixel_coverage_and_empty_masks_are_measured(tmp_path):
    make_sample(tmp_path, 'train', 'IMG_ENC00001_00001', np.array([[0, 1], [0, 0]], dtype=np.uint8))
    make_sample(tmp_path, 'valid', 'IMG_ENC00002_00001', np.zeros((2, 2), dtype=np.uint8))
    result = collect_eda(tmp_path, thumbnail_size=8)
    annotations = result['per_annotation']
    assert annotations[0]['foreground_pixels'] == 1
    assert annotations[0]['foreground_fraction'] == 0.25
    assert annotations[0]['unique_values'] == '0|1'
    assert annotations[0]['binary_encoding'] is True
    assert result['summary']['splits']['valid']['empty_masks'] == 1
    assert result['summary']['totals']['images_decoded'] == 2
    assert result['summary']['totals']['masks_decoded'] == 2


def test_unexpected_mask_values_are_reported_without_silently_becoming_valid(tmp_path):
    make_sample(tmp_path, 'train', 'IMG_ENC00001_00001', np.array([[0, 2], [0, 2]], dtype=np.uint8))
    result = collect_eda(tmp_path)
    assert result['summary']['splits']['train']['nonbinary_masks'] == 1
    assert result['per_annotation'][0]['unique_values'] == '0|2'
    assert result['per_annotation'][0]['foreground_fraction'] == 0.5


def test_truncated_pixel_data_is_detected_even_when_header_opens(tmp_path):
    make_sample(tmp_path, 'train', 'IMG_ENC00001_00001', np.zeros((64, 64), dtype=np.uint8))
    path = tmp_path / 'imagesTrain/IMG_ENC00001_00001.png'
    Image.fromarray(np.random.default_rng(12).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
    encoded = path.read_bytes()
    path.write_bytes(encoded[:len(encoded) // 2])
    with Image.open(path) as header_only:
        assert header_only.size == (64, 64)
    result = collect_eda(tmp_path)
    assert result['summary']['totals']['corrupt_images'] == 1
    assert result['summary']['totals']['images_decoded'] == 0
    assert result['summary']['totals']['masks_decoded'] == 1
    assert result['summary']['splits']['train']['mask_dimension_mismatches'] == 0


def test_corrupt_mask_is_counted_separately_from_successful_image(tmp_path):
    make_sample(tmp_path, 'test', 'IMG_ENC00001_00001', np.zeros((2, 2), dtype=np.uint8))
    (tmp_path / 'labelsTest/IMG_ENC00001_00001.tiff').write_bytes(b'broken pixels')
    result = collect_eda(tmp_path)
    assert result['summary']['totals']['images_decoded'] == 1
    assert result['summary']['totals']['corrupt_masks'] == 1
    assert result['summary']['totals']['masks_decoded'] == 0


def test_pairwise_agreement_uses_all_discovered_annotations(tmp_path):
    make_sample(tmp_path, 'test', 'IMG_ENC00001_00001', np.ones((2, 2), dtype=np.uint8))
    label_dir = tmp_path / 'labelsTest'
    (label_dir / 'IMG_ENC00001_00001.tiff').rename(label_dir / 'IMG_ENC00001_00001_mask_ann1.tiff')
    Image.fromarray(np.array([[1, 0], [1, 0]], dtype=np.uint8)).save(label_dir / 'IMG_ENC00001_00001_mask_ann3.tiff')
    result = collect_eda(tmp_path, thumbnail_size=4)
    image = result['per_image'][0]
    assert image['annotations_decoded'] == 2
    assert image['annotation_coverage_range'] == 0.5
    assert image['pairwise_dice_thumbnail_mean'] == pytest.approx(2 / 3)
    assert image['consensus_fraction_thumbnail'] == 0.5


def test_refuses_a_path_outside_the_data_root(tmp_path):
    from scripts.eda import checked_path

    with pytest.raises(ValueError, match='outside'):
        checked_path(tmp_path.parent / 'unrelated.png', tmp_path.resolve())
