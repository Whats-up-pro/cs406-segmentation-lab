"""Offline fixtures prove freeze enforcement and common scoring behavior."""
from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')

from backend.compact_model import CompactUNet, save_compact_checkpoint
from scripts.evaluate_completion import aggregate_rows, boundary_f1, evaluate, load_reference, main, score_masks, verify_frozen_runs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def frozen_run(tmp_path):
    """A clearly synthetic random-weight fixture, never a reported experiment."""
    run = tmp_path / 'runs' / 'fixture_s406'
    run.mkdir(parents=True)
    model = CompactUNet(base_channels=4)
    for parameter in model.parameters():
        parameter.data.zero_()
    save_compact_checkpoint(model, run / 'best.pth', input_size=32, metadata={'seed': 406, 'epoch': 1})
    counts = {'train': 1, 'valid': 1}
    sources = {'backend/compact_model.py': digest(Path('backend/compact_model.py'))}
    config = {'status': 'frozen', 'optimizer_splits': ['train'], 'selection_split': 'valid',
              'test_accessed': False, 'counts': counts, 'data_fingerprint': 'a' * 64,
              'source_sha256': sources, 'seed': 406, 'epochs': 1, 'best_epoch': 1,
              'base_channels': 4, 'image_size': 32, 'cache_dir': str(tmp_path / 'missing-original-cache')}
    history = [{'epoch': 1, 'train_samples': 1, 'valid_samples': 1, 'train_batches': 1,
                'valid_batches': 1, 'updated_parameters': 1, 'valid_dice': .25}]
    write_json(run / 'config.json', config)
    write_json(run / 'history.json', history)
    frozen = {'format': 'compact-selection-v1', 'status': 'frozen', 'model': 'compact_unet',
              'optimizer_splits': ['train'], 'selection_split': 'valid', 'test_accessed': False,
              'threshold': .6, 'valid_dice': .25, 'threshold_search': [{'threshold': .6, 'valid_dice': .25}],
              'checkpoint': 'best.pth', 'checkpoint_sha256': digest(run / 'best.pth'),
              'config_sha256': digest(run / 'config.json'), 'history_sha256': digest(run / 'history.json'),
              'cache_fingerprint': 'a' * 64, 'evaluation_size': 256, 'counts': counts,
              'seed': 406, 'epochs': 1, 'best_epoch': 1, 'source_sha256': sources}
    write_json(run / 'frozen-selection.json', frozen)
    return run


def arguments(tmp_path):
    return Namespace(data_dir=tmp_path / 'data', runs_dir=tmp_path / 'runs', output=tmp_path / 'evaluation',
                     device='cpu', reference_checkpoint=None, pixel_model=None, split='test', cache_dir=None)


def sample(tmp_path):
    data = tmp_path / 'data'
    (data / 'imagesTest').mkdir(parents=True)
    (data / 'labelsTest').mkdir()
    sample_id = 'IMG_ENC00001_00001'
    label = np.zeros((15, 23), dtype=np.uint8)
    label[4:10, 6:13] = 255
    Image.fromarray(np.repeat(label[..., None], 3, axis=2)).save(data / f'imagesTest/{sample_id}.png')
    for annotator in (1, 3):
        Image.fromarray(label).save(data / f'labelsTest/{sample_id}_mask_ann{annotator}.tiff')
    return sample_id


def forbid_test_indexing(monkeypatch):
    original_iterdir, original_open = Path.iterdir, Image.open

    def checked_iterdir(path):
        assert path.name not in {'imagesTest', 'labelsTest'}, 'Test data indexed before freeze verification'
        return original_iterdir(path)

    def checked_open(path, *args, **kwargs):
        assert 'imagesTest' not in str(path) and 'labelsTest' not in str(path), 'Test pixels opened before verification'
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'iterdir', checked_iterdir)
    monkeypatch.setattr(Image, 'open', checked_open)


def test_missing_freeze_is_rejected_before_any_test_index_or_open(tmp_path, monkeypatch):
    (tmp_path / 'runs').mkdir()
    sample(tmp_path)
    forbid_test_indexing(monkeypatch)
    with pytest.raises(ValueError, match='frozen'):
        evaluate(arguments(tmp_path))
    assert not (tmp_path / 'evaluation').exists()


def test_tampered_checkpoint_is_rejected_before_test_access(tmp_path, monkeypatch):
    run = frozen_run(tmp_path)
    with (run / 'best.pth').open('ab') as stream:
        stream.write(b'tamper')
    sample(tmp_path)
    forbid_test_indexing(monkeypatch)
    with pytest.raises(ValueError, match='checkpoint.*hash|hash.*checkpoint'):
        evaluate(arguments(tmp_path))
    assert not (tmp_path / 'evaluation').exists()


@pytest.mark.parametrize('field,value', [('selection_split', 'test'), ('optimizer_splits', ['train', 'test']),
                                        ('counts', {'train': 2, 'valid': 1}), ('cache_fingerprint', 'b' * 64)])
def test_inconsistent_training_provenance_is_rejected(tmp_path, field, value):
    run = frozen_run(tmp_path)
    frozen = json.loads((run / 'frozen-selection.json').read_text())
    frozen[field] = value
    write_json(run / 'frozen-selection.json', frozen)
    with pytest.raises(ValueError):
        verify_frozen_runs(tmp_path / 'runs')


def test_portable_frozen_run_without_original_cache_is_validated(tmp_path):
    frozen_run(tmp_path)
    runs = verify_frozen_runs(tmp_path / 'runs')
    assert len(runs) == 1
    assert runs[0]['cache_verification']['performed'] is False
    assert runs[0]['threshold'] == .6


def test_available_relocated_cache_cannot_disagree_with_frozen_fingerprint(tmp_path):
    frozen_run(tmp_path)
    cache = tmp_path / 'relocated-cache'
    cache.mkdir()
    write_json(cache / 'manifest.json', {'fingerprint': 'b' * 64, 'counts': {'train': 1, 'valid': 1}})
    with pytest.raises(ValueError, match='cache|fingerprint'):
        verify_frozen_runs(tmp_path / 'runs', cache_dir=cache)


def test_historical_source_snapshot_may_differ_from_current_code_without_invalidating_weights(tmp_path):
    run = frozen_run(tmp_path)
    config = json.loads((run / 'config.json').read_text())
    frozen = json.loads((run / 'frozen-selection.json').read_text())
    config['source_sha256'] = frozen['source_sha256'] = {'backend/compact_model.py': 'b' * 64}
    write_json(run / 'config.json', config)
    frozen['config_sha256'] = digest(run / 'config.json')
    write_json(run / 'frozen-selection.json', frozen)
    verified = verify_frozen_runs(run.parent)[0]
    comparison = verified['historical_source_comparison']['backend/compact_model.py']
    assert comparison['matches_current'] is False
    assert comparison['trained_sha256'] == 'b' * 64
    assert comparison['current_sha256'] == digest(Path('backend/compact_model.py'))


def cache_recipe_fixture(tmp_path, version, declared_recipe=None, cache_recipe=None):
    run = frozen_run(tmp_path)
    legacy, modern = 'native-rgb-to-model-v1', 'rgb-bilinear-256-then-model-v2'
    recipe = declared_recipe or (legacy if version == 1 else modern)
    cache = tmp_path / 'relocated-cache'
    cache.mkdir()
    samples = {'train': [{'id': 'fixture_train'}], 'valid': [{'id': 'fixture_valid'}]}
    payload = {'size': 32, 'samples': samples}
    manifest = {'format': f'compact-training-cache-v{version}', 'image_size': 32, 'samples': samples,
                'counts': {'train': 1, 'valid': 1}, 'test_accessed': False}
    if version == 2:
        manifest['preprocessing'] = cache_recipe or modern
        payload['preprocessing'] = manifest['preprocessing']
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    manifest['fingerprint'] = fingerprint
    write_json(cache / 'manifest.json', manifest)
    config = json.loads((run / 'config.json').read_text())
    frozen = json.loads((run / 'frozen-selection.json').read_text())
    config.update(data_fingerprint=fingerprint, input_preprocessing=recipe)
    write_json(run / 'config.json', config)
    frozen.update(cache_fingerprint=fingerprint, input_preprocessing=recipe, config_sha256=digest(run / 'config.json'))
    write_json(run / 'frozen-selection.json', frozen)
    return run, cache


@pytest.mark.parametrize('version', [1, 2])
def test_cache_fingerprint_accepts_its_declared_v1_or_v2_recipe(tmp_path, version):
    run, cache = cache_recipe_fixture(tmp_path, version)
    verified = verify_frozen_runs(run.parent, cache_dir=cache)[0]
    assert verified['cache_verification']['performed'] is True
    assert verified['cache_verification']['format'] == f'compact-training-cache-v{version}'
    assert verified['input_preprocessing'] == ('native-rgb-to-model-v1' if version == 1 else 'rgb-bilinear-256-then-model-v2')


def test_v2_run_cannot_be_corroborated_with_a_v1_cache_recipe(tmp_path):
    run, cache = cache_recipe_fixture(tmp_path, 1, declared_recipe='rgb-bilinear-256-then-model-v2')
    with pytest.raises(ValueError, match='preprocessing|recipe'):
        verify_frozen_runs(run.parent, cache_dir=cache)


def test_v2_cache_cannot_declare_the_legacy_recipe(tmp_path):
    run, cache = cache_recipe_fixture(tmp_path, 2, cache_recipe='native-rgb-to-model-v1')
    with pytest.raises(ValueError, match='preprocessing|recipe'):
        verify_frozen_runs(run.parent, cache_dir=cache)


def test_config_and_frozen_preprocessing_must_agree_even_without_cache(tmp_path):
    run = frozen_run(tmp_path)
    frozen = json.loads((run / 'frozen-selection.json').read_text())
    frozen['input_preprocessing'] = 'rgb-bilinear-256-then-model-v2'
    write_json(run / 'frozen-selection.json', frozen)
    with pytest.raises(ValueError, match='preprocessing|recipe'):
        verify_frozen_runs(run.parent)


def test_macro_and_micro_are_distinct_and_empty_rules_match_existing_metrics():
    first = score_masks(np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool))
    second = score_masks(np.zeros((3, 4), dtype=bool), np.ones((3, 4), dtype=bool))
    summary = aggregate_rows([first, second])
    assert summary['macro']['mean']['dice'] == .5
    assert summary['micro']['dice'] == 0
    assert summary['micro']['pixel_accuracy'] == .25
    assert summary['confusion'] == {'TP': 0, 'FP': 0, 'FN': 12, 'TN': 4}
    assert first['precision'] == first['recall'] == first['boundary_f1'] == 1
    assert second['precision'] == second['recall'] == second['boundary_f1'] == 0
    assert aggregate_rows([])['macro'] is None


def test_reference_uses_all_even_numbered_annotations_and_ties_are_background(tmp_path):
    sample_id = sample(tmp_path)
    data = tmp_path / 'data'
    paths = sorted((data / 'labelsTest').iterdir())
    Image.fromarray(np.ones((15, 23), dtype=np.uint8)).save(paths[0])
    Image.fromarray(np.zeros((15, 23), dtype=np.uint8)).save(paths[1])
    target, annotations = load_reference({'id': sample_id, 'image': data / f'imagesTest/{sample_id}.png', 'masks': paths}, (256, 256))
    assert len(annotations) == 2
    assert target.shape == (256, 256)
    assert not target.any()
    assert annotations[0].all() and not annotations[1].any()


def test_boundary_tolerance_uses_euclidean_distance_including_empty_masks():
    first = np.zeros((11, 11), dtype=bool)
    second = first.copy()
    assert boundary_f1(first, second) == 1
    first[4, 4] = True
    assert boundary_f1(first, second) == 0
    second[4, 6] = True
    assert boundary_f1(first, second, tolerance=2) == 1
    second[:] = False
    second[6, 6] = True
    assert boundary_f1(first, second, tolerance=2) == 0


def test_fixture_evaluation_writes_real_metrics_annotations_and_binary_predictions(tmp_path):
    frozen_run(tmp_path)
    sample_id = sample(tmp_path)
    report = evaluate(arguments(tmp_path))
    assert report['failures'] == []
    assert set(report['methods']) == {'baseline', 'fixture_s406'}
    assert report['methods']['fixture_s406']['sample_count'] == 1
    assert report['methods']['fixture_s406']['per_annotation']['sample_count'] == 2
    with (tmp_path / 'evaluation/per_image.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    compact = next(row for row in rows if row['method'] == 'fixture_s406')
    assert compact['encounter'] == 'ENC00001'
    assert int(compact['TP']) == int(compact['FP']) == 0
    with Image.open(tmp_path / f'evaluation/masks/fixture_s406/{sample_id}.png') as prediction:
        assert prediction.size == (256, 256)
        assert set(np.unique(prediction)) == {0}
    with pytest.raises(FileExistsError):
        evaluate(arguments(tmp_path))


def test_external_manifest_preserves_original_validation_split_and_stress_role(tmp_path):
    frozen_run(tmp_path)
    sample(tmp_path)
    write_json(tmp_path / 'data/manifest.json', {'dataset': 'ISIC 2018 Task 1 validation',
                                               'source_split': 'validation', 'role': 'External dermoscopy domain-shift stress test',
                                               'counts': {'images': 1, 'masks': 2}, 'license': {'id': 'CC0-1.0'}})
    report = evaluate(arguments(tmp_path))
    assert report['dataset']['source_split'] == 'validation'
    assert report['dataset']['role'] == 'external_stress_test'
    assert report['dataset']['manifest_sha256'] == digest(tmp_path / 'data/manifest.json')


def test_optional_pixel_model_keeps_its_json_path_when_sample_paths_are_indexed(tmp_path):
    from backend.pixel_model import FEATURE_NAMES, FEATURE_SCALING, MODEL_FORMAT

    frozen_run(tmp_path)
    sample_id = sample(tmp_path)
    model_path = tmp_path / 'pixel.json'
    write_json(model_path, {'format': MODEL_FORMAT, 'features': list(FEATURE_NAMES), 'scaling': FEATURE_SCALING,
                            'input_size': 256, 'coefficients': [0.] * 10, 'intercept': 0., 'threshold': .6})
    args = arguments(tmp_path)
    args.pixel_model = model_path
    report = evaluate(args)
    assert report['failures'] == []
    assert report['methods']['pixel_logistic']['sample_count'] == 1
    with Image.open(tmp_path / f'evaluation/masks/pixel_logistic/{sample_id}.png') as prediction:
        assert not np.asarray(prediction).any()


def test_cli_reports_the_result_directory_in_its_output_field(tmp_path, capsys):
    frozen_run(tmp_path)
    sample(tmp_path)
    code = main(['--data-dir', str(tmp_path / 'data'), '--runs-dir', str(tmp_path / 'runs'),
                 '--output', str(tmp_path / 'evaluation'), '--device', 'cpu'])
    printed = capsys.readouterr().out
    report = json.loads(printed[printed.index('{'):])
    assert code == 0
    assert report['output'] == str((tmp_path / 'evaluation').resolve())


def test_multi_image_evaluation_reads_each_checkpoint_once_with_more_models_than_cache_slots(tmp_path, monkeypatch):
    from backend.compact_model import _load_checkpoint

    run = frozen_run(tmp_path)
    for index in range(3):
        shutil.copytree(run, run.parent / f'fixture_variant_{index}')
    sample_id = sample(tmp_path)
    for encounter in (2, 3):
        new_id = f'IMG_ENC{encounter:05d}_00001'
        shutil.copyfile(tmp_path / f'data/imagesTest/{sample_id}.png', tmp_path / f'data/imagesTest/{new_id}.png')
        shutil.copyfile(tmp_path / f'data/labelsTest/{sample_id}_mask_ann1.tiff', tmp_path / f'data/labelsTest/{new_id}.tiff')
    _load_checkpoint.cache_clear()
    original_load = torch.load
    loaded_files = []

    def record_actual_checkpoint_read(path, *args, **kwargs):
        loaded_files.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, 'load', record_actual_checkpoint_read)
    report = evaluate(arguments(tmp_path))
    assert report['failures'] == []
    assert report['counts']['per_image_rows'] == 15
    assert len(loaded_files) == 4


def test_real_one_epoch_training_artifacts_and_cache_are_accepted_without_training_imports_in_evaluator(tmp_path):
    from scripts.train_compact import train

    source_id = sample(tmp_path)
    for split, encounter in [('Train', 2), ('Valid', 3)]:
        (tmp_path / f'data/images{split}').mkdir()
        (tmp_path / f'data/labels{split}').mkdir()
        sample_id = f'IMG_ENC{encounter:05d}_00001'
        shutil.copyfile(tmp_path / f'data/imagesTest/{source_id}.png', tmp_path / f'data/images{split}/{sample_id}.png')
        shutil.copyfile(tmp_path / f'data/labelsTest/{source_id}_mask_ann1.tiff', tmp_path / f'data/labels{split}/{sample_id}.tiff')
    train(Namespace(data_dir=tmp_path / 'data', cache_dir=tmp_path / 'cache', output=tmp_path / 'runs/one_epoch_fixture',
                    image_size=32, base_channels=4, epochs=1, batch_size=1, learning_rate=.001,
                    weight_decay=.0001, dice_weight=1., augment=False, seed=406, cpu_threads=2, device='cpu'))
    report = evaluate(arguments(tmp_path))
    assert report['failures'] == []
    assert report['methods']['one_epoch_fixture']['provenance']['cache_verification']['performed'] is True
    assert report['methods']['one_epoch_fixture']['provenance']['cache_verification']['fingerprint_reproduced_from_source_hash_inventory'] is True
    assert report['methods']['one_epoch_fixture']['provenance']['cache_verification']['format'] == 'compact-training-cache-v2'
    assert report['methods']['one_epoch_fixture']['provenance']['input_preprocessing'] == 'rgb-bilinear-256-then-model-v2'
