"""Archive integrity checks use real tiny ZIP/image fixtures, without network."""

import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts.prepare_external import prepare_external


def encoded_image(array, format='PNG'):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=format)
    return stream.getvalue()


def archives(tmp_path, image_entries=None, mask_entries=None):
    image = encoded_image(np.full((3, 5, 3), 100, dtype=np.uint8), 'JPEG')
    mask = encoded_image(np.array([[0, 0, 255, 255, 0]] * 3, dtype=np.uint8))
    paths = [tmp_path / 'images.zip', tmp_path / 'masks.zip']
    entries = [image_entries if image_entries is not None else [('input/ISIC_0000001.jpg', image)],
               mask_entries if mask_entries is not None else [('gold/ISIC_0000001_segmentation.png', mask)]]
    for path, members in zip(paths, entries):
        with zipfile.ZipFile(path, 'w') as archive:
            for name, content in members:
                archive.writestr(name, content)
    return paths


def prepare_fixture(tmp_path, images=None, masks=None):
    first, second = archives(tmp_path, images, masks)
    return prepare_external(first, second, tmp_path / 'prepared', expected_count=1, verify_official_sizes=False)


def test_preparation_preserves_archives_and_records_hashes_and_binary_pairs(tmp_path):
    first, second = archives(tmp_path)
    original = [p.read_bytes() for p in (first, second)]
    report = prepare_external(first, second, tmp_path / 'prepared', expected_count=1, verify_official_sizes=False)
    assert [p.read_bytes() for p in (first, second)] == original
    assert report['counts'] == {'images': 1, 'masks': 1}
    assert report['license']['id'] == 'CC0-1.0'
    assert report['archives']['images']['sha256'] == hashlib.sha256(original[0]).hexdigest()
    assert report['samples'][0]['width'] == 5
    assert report['samples'][0]['height'] == 3
    prepared = tmp_path / 'prepared'
    assert (prepared / 'imagesTest/ISIC_0000001.jpg').is_file()
    with zipfile.ZipFile(first) as archive:
        assert (prepared / 'imagesTest/ISIC_0000001.jpg').read_bytes() == archive.read('input/ISIC_0000001.jpg')
    with zipfile.ZipFile(second) as archive:
        assert (prepared / 'labelsTest/ISIC_0000001.png').read_bytes() == archive.read('gold/ISIC_0000001_segmentation.png')
    with Image.open(prepared / 'labelsTest/ISIC_0000001.png') as mask:
        assert np.asarray(mask).tolist() == [[0, 0, 255, 255, 0]] * 3
    saved = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    assert saved['samples'] == report['samples']
    assert saved['official_release_identity_verified'] is False


@pytest.mark.parametrize('bad_name', ['../escape.jpg', '/absolute.jpg', 'C:/escape.jpg', r'..\escape.jpg'])
def test_traversal_is_rejected_before_any_prepared_output(tmp_path, bad_name):
    with pytest.raises(ValueError, match='Unsafe archive'):
        prepare_fixture(tmp_path, images=[(bad_name, b'bad')])
    assert not (tmp_path / 'prepared/imagesTest').exists()
    assert not (tmp_path / 'escape.jpg').exists()


def test_flattened_duplicate_image_names_are_rejected(tmp_path):
    content = encoded_image(np.zeros((3, 5, 3), dtype=np.uint8), 'JPEG')
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_fixture(tmp_path, images=[('a/ISIC_0000001.jpg', content), ('b/ISIC_0000001.jpg', content)])


def test_missing_matching_mask_is_rejected_before_any_prepared_output(tmp_path):
    mask = encoded_image(np.zeros((3, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match='pair'):
        prepare_fixture(tmp_path, masks=[('gold/ISIC_0000002_segmentation.png', mask)])
    assert not (tmp_path / 'prepared/manifest.json').exists()


@pytest.mark.parametrize('mask', [np.ones((3, 5), dtype=np.uint8),
                                 np.zeros((4, 5), dtype=np.uint8),
                                 np.zeros((3, 5, 3), dtype=np.uint8)])
def test_nonbinary_or_wrong_shape_mask_is_rejected(tmp_path, mask):
    with pytest.raises(ValueError, match='mask'):
        prepare_fixture(tmp_path, masks=[('gold/ISIC_0000001_segmentation.png', encoded_image(mask))])
    assert not (tmp_path / 'prepared/manifest.json').exists()


def test_corrupt_image_payload_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='image'):
        prepare_fixture(tmp_path, images=[('input/ISIC_0000001.jpg', b'broken JPEG')])


def test_default_requires_official_hundred_image_release(tmp_path):
    first, second = archives(tmp_path)
    with pytest.raises(ValueError, match='100|size'):
        prepare_external(first, second, tmp_path / 'prepared')


def test_existing_prepared_output_cannot_be_overwritten(tmp_path):
    first, second = archives(tmp_path)
    target = tmp_path / 'prepared'
    prepare_external(first, second, target, expected_count=1, verify_official_sizes=False)
    with pytest.raises(FileExistsError):
        prepare_external(first, second, target, expected_count=1, verify_official_sizes=False)
