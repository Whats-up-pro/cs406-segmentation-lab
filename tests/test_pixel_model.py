"""Behavioral checks for the supervised pixel experiment and portable inference."""

import argparse
import builtins
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def dataset(root: Path):
    """Small color-separable data; test includes an even-annotator tie."""
    for suffix in ('Train', 'Valid', 'Test'):
        (root / f'images{suffix}').mkdir(parents=True)
        (root / f'labels{suffix}').mkdir()
        for index in range(2 if suffix == 'Train' else 1):
            rgb = np.zeros((16, 20, 3), dtype=np.uint8)
            rgb[:, :10] = [240, 30, 30]
            rgb[:, 10:] = [30, 190, 40]
            mask = np.zeros((16, 20), dtype=np.uint8)
            mask[:, :10] = 1
            stem = f'{suffix.lower()}_{index}'
            Image.fromarray(rgb).save(root / f'images{suffix}' / f'{stem}.png')
            Image.fromarray(mask).save(root / f'labels{suffix}' / f'{stem}{"_mask_ann2" if suffix == "Test" else ""}.tif')
            if suffix == 'Test':
                Image.fromarray(1 - mask).save(root / 'labelsTest' / f'{stem}_mask_ann7.tif')


def arguments(root, output):
    return argparse.Namespace(data_dir=root, output=output, epochs=2,
                              pixels_per_image=256, seed=406, input_size=32,
                              alpha=1e-4)


def test_training_learns_from_train_and_defers_test_until_selection(tmp_path, monkeypatch):
    # Fails if optimization sees test, skips samples, ignores duplicates, or
    # portable inference silently uses a heuristic instead of learned weights.
    from backend.pixel_model import predict_pixel_model, pixel_model_status
    import scripts.train_pixel_model as experiment

    root, output = tmp_path / 'data', tmp_path / 'run'
    dataset(root)
    original_path_open, original_open, original_iterdir = Path.open, builtins.open, Path.iterdir

    def assert_test_is_held_out(path):
        if isinstance(path, (str, Path)) and {'imagesTest', 'labelsTest'} & set(Path(path).parts):
            assert (output / 'selection.json').is_file()
            assert (output / 'model.json').is_file()

    def guarded_path_open(path, *args, **kwargs):
        assert_test_is_held_out(path)
        return original_path_open(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        assert_test_is_held_out(path)
        return original_open(path, *args, **kwargs)

    def guarded_iterdir(path):
        assert_test_is_held_out(path)
        return original_iterdir(path)

    # Cover directory discovery, hash reads and Pillow pixel reads directly.
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(Path, 'iterdir', guarded_iterdir)
    manifest = experiment.train(arguments(root, output))
    assert manifest['status'] == 'complete'
    assert manifest['splits']['train']['count'] == 2
    assert [epoch['train_images'] for epoch in manifest['history']] == [2, 2]
    assert [epoch['train_pixels'] for epoch in manifest['history']] == [512, 512]
    assert manifest['duplicates']['train_groups'][0]['ids'] == ['train_0', 'train_1']
    assert manifest['duplicates']['train_effective_image_weight'] == 1.0
    assert manifest['test']['images'] == 1
    model = json.loads((output / 'model.json').read_text())
    assert model['training_origin']['optimizer_splits'] == ['train']
    assert model['threshold'] in [.3, .4, .5, .6, .7]
    assert pixel_model_status(output / 'model.json')['inference_ready'] is True
    image = Image.open(root / 'imagesTest/test_0.png')
    probabilities = predict_pixel_model(image, output / 'model.json')
    assert probabilities.dtype == np.float32
    assert probabilities.shape == (16, 20)
    assert probabilities[:, :8].mean() > probabilities[:, 12:].mean() + .4
    # The two opposite test annotations tie everywhere; strict majority is empty.
    import csv
    with (output / 'per_image.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert float(rows[0]['target_fraction']) == 0.0


def test_test_labels_cannot_change_model_or_selected_threshold(tmp_path):
    # Fails if test data participate in fitting or threshold selection.
    from scripts.train_pixel_model import train

    root = tmp_path / 'data'
    dataset(root)
    first, second = tmp_path / 'first', tmp_path / 'second'
    train(arguments(root, first))
    for path in (root / 'labelsTest').iterdir():
        Image.fromarray(np.ones((16, 20), dtype=np.uint8)).save(path)
    train(arguments(root, second))
    one = json.loads((first / 'model.json').read_text())
    two = json.loads((second / 'model.json').read_text())
    assert one['coefficients'] == two['coefficients']
    assert one['intercept'] == two['intercept']
    assert one['threshold'] == two['threshold']


def test_rejects_mismatched_training_mask_and_output_inside_raw_data(tmp_path):
    from scripts.train_pixel_model import train

    root = tmp_path / 'data'
    dataset(root)
    with pytest.raises(ValueError, match='outside'):
        train(arguments(root, root / 'run'))
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / 'labelsTrain/train_0.tif')
    with pytest.raises(ValueError, match='dimensions'):
        train(arguments(root, tmp_path / 'run'))


def test_portable_model_rejects_nonfinite_coefficients(tmp_path):
    from backend.pixel_model import FEATURE_NAMES, FEATURE_SCALING, predict_pixel_model

    path = tmp_path / 'broken.json'
    path.write_text(json.dumps({'format': 'cs406-pixel-logistic-v1',
                               'features': list(FEATURE_NAMES), 'scaling': FEATURE_SCALING,
                               'coefficients': [float('nan')] + [0.0] * 9,
                               'intercept': 0, 'threshold': .5, 'input_size': 32}))
    with pytest.raises(ValueError):
        predict_pixel_model(Image.new('RGB', (10, 10)), path)
