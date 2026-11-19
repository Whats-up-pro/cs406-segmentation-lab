"""Behavior checks for the read-only image leakage candidate audit."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import pytest

from scripts.audit_leakage import audit_leakage, write_audit


def put_image(root: Path, split: str, name: str, image: Image.Image) -> Path:
    directory = root / {'train': 'imagesTrain', 'valid': 'imagesValid', 'test': 'imagesTest'}[split]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{name}.png'
    image.save(path)
    return path


def pattern(seed=406):
    values = np.random.default_rng(seed).integers(30, 210, (12, 12, 3), dtype=np.uint8)
    return Image.fromarray(values).resize((96, 96), Image.Resampling.BICUBIC)


def test_exact_cross_split_bytes_and_encounters_are_separate_evidence(tmp_path):
    image = pattern()
    source = put_image(tmp_path, 'train', 'IMG_ENC00001_00001', image)
    target = put_image(tmp_path, 'valid', 'IMG_ENC00002_00001', image)
    target.write_bytes(source.read_bytes())
    put_image(tmp_path, 'test', 'IMG_ENC00001_00002', pattern(9))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.png')}
    report = audit_leakage(tmp_path)
    assert report['counts']['cross_split_exact_groups'] == 1
    assert report['cross_split_encounter_overlap']['train__test'] == ['ENC00001']
    assert report['cross_split_encounter_overlap']['train__valid'] == []
    assert report['exact_duplicate_groups'][0]['cross_split'] is True
    assert report['counts']['images'] == 3
    assert {p: p.read_bytes() for p in before} == before


def test_transformed_copy_is_a_candidate_but_independent_pattern_is_not(tmp_path):
    image = pattern()
    put_image(tmp_path, 'train', 'IMG_ENC00001_00001', image)
    altered = ImageEnhance.Brightness(image.resize((151, 143), Image.Resampling.BICUBIC)).enhance(1.08)
    put_image(tmp_path, 'valid', 'IMG_ENC00002_00001', altered)
    put_image(tmp_path, 'test', 'IMG_ENC00003_00001', pattern(97))
    report = audit_leakage(tmp_path, phash_threshold=6, dhash_threshold=8)
    assert report['counts']['cross_split_exact_groups'] == 0
    assert len(report['cross_split_candidates']) == 1
    pair = report['cross_split_candidates'][0]
    assert {pair['left_split'], pair['right_split']} == {'train', 'valid'}
    assert pair['review_status'] == 'pending_visual_review'
    assert pair['phash_distance'] <= 6
    assert pair['dhash_distance'] <= 8


def test_corrupt_images_are_reported_without_skipping_the_inventory(tmp_path):
    path = put_image(tmp_path, 'test', 'unrecognized', pattern())
    path.write_bytes(b'not an image')
    report = audit_leakage(tmp_path)
    assert report['counts']['images'] == 1
    assert report['counts']['decoded_images'] == 0
    assert report['counts']['unrecognized_encounters'] == 1
    assert len(report['decode_errors']) == 1
    assert len(report['inventory'][0]['sha256']) == 64


def test_audit_exports_candidate_evidence_without_writing_in_the_data(tmp_path):
    data = tmp_path / 'data'
    image = pattern()
    put_image(data, 'train', 'IMG_ENC00001_00001', image)
    put_image(data, 'test', 'IMG_ENC00002_00001', ImageEnhance.Brightness(image).enhance(1.05))
    report = audit_leakage(data)
    output = tmp_path / 'results'
    write_audit(report, output, contact_sheet_limit=2)
    saved = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert len(saved['cross_split_candidates']) == 1
    assert (output / 'cross_split_candidates.csv').is_file()
    assert (output / 'inventory.csv').is_file()
    assert (output / 'candidate_contact_sheet.jpg').is_file()
    with pytest.raises(ValueError, match='data root'):
        write_audit(report, data / 'results')
    assert not (data / 'results').exists()


@pytest.mark.parametrize('threshold', [-1, 65])
def test_invalid_hamming_threshold_is_rejected(tmp_path, threshold):
    with pytest.raises(ValueError, match='threshold'):
        audit_leakage(tmp_path, phash_threshold=threshold)
