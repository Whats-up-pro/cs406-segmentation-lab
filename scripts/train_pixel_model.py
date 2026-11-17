"""Train a small, auditable supervised segmenter, then select on valid and evaluate test.

python -m scripts.train_pixel_model --data-dir data --output results/pixel_classifier
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version
import sklearn
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

from backend.metrics import METRIC_NAMES, segmentation_metrics
from backend.pixel_model import (FEATURE_NAMES, FEATURE_SCALING, MODEL_FORMAT,
                                 pixel_features, probabilities_from_features)
from scripts.audit_data import PROJECT_ROOT, discover_samples, file_sha256
from scripts.evaluate import macro_summary, write_csv


THRESHOLDS = (.3, .4, .5, .6, .7)
IMAGE_FIELDS = ['id', 'split', 'method', 'width', 'height', 'evaluation_width',
                'evaluation_height', 'annotations', 'target_fraction',
                'prediction_fraction', *METRIC_NAMES, 'seconds']


def _write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _load_sample(sample, size, single=False):
    """Decode image and every raw binary mask, validating original dimensions."""
    with Image.open(sample['image']) as source:
        source.load()
        image = source.convert('RGB')
    if not sample['masks'] or (single and (len(sample['masks']) != 1 or sample['masks'][0].stem != sample['id'])):
        raise ValueError(f"{sample['id']}: train/valid require one exact-stem label; no annotation may be missing.")
    masks = []
    for path in sample['masks']:
        with Image.open(path) as mask:
            mask.load()
            if mask.size != image.size:
                raise ValueError(f'{path.name}: mask dimensions do not match image dimensions.')
            values = np.asarray(mask)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f'{path.name}: mask must be finite and 2-D.')
            unique = set(np.unique(values).tolist())
            if not (unique <= {0, 1} or unique <= {0, 255}):
                raise ValueError(f'{path.name}: mask must have binary values 0/1 or 0/255.')
            resized = Image.fromarray((values != 0).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)
            masks.append(np.asarray(resized, dtype=bool))
    consensus = np.stack(masks).sum(axis=0) > len(masks) / 2
    return image, consensus, masks


def _log_loss(probability, target):
    p = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log1p(-p)))


def _inventory(samples, data_dir):
    return [{'id': sample['id'], 'image': sample['image'].relative_to(data_dir).as_posix(),
             'image_sha256': file_sha256(sample['image']),
             'masks': [{'path': path.relative_to(data_dir).as_posix(), 'sha256': file_sha256(path)}
                       for path in sample['masks']]} for sample in samples]


def _validate_output(args):
    data_dir, output = args.data_dir.resolve(), args.output.resolve()
    if output.is_relative_to(data_dir):
        raise ValueError('Output must be outside raw data directory.')
    if args.epochs < 1 or not 2 <= args.input_size <= 2048 or not 1 <= args.pixels_per_image <= args.input_size ** 2:
        raise ValueError('Positive epochs, input_size>=2, and pixels_per_image<=input_size squared are required.')
    if args.seed < 0 or not np.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError('Nonnegative seed and finite positive alpha are required.')
    for name in ('manifest.json', 'model.json', 'selection.json', 'training_log.json', 'summary.json', 'per_image.csv'):
        if (output / name).exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment artifact: {output / name}')
    output.mkdir(parents=True, exist_ok=True)
    return data_dir, output


def _validate_epoch(classifier, samples, size, threshold=.5):
    losses, scores = [], []
    for sample in samples:
        image, target, _ = _load_sample(sample, size, single=True)
        p = probabilities_from_features(pixel_features(image, size), classifier.coef_[0], classifier.intercept_[0])
        losses.append(_log_loss(p, target))
        scores.append(segmentation_metrics(p >= threshold, target)['dice'])
    return float(np.mean(losses)), float(np.mean(scores))


def _select_threshold(classifier, samples, size):
    scores = {threshold: [] for threshold in THRESHOLDS}
    for sample in samples:
        image, target, _ = _load_sample(sample, size, single=True)
        p = probabilities_from_features(pixel_features(image, size), classifier.coef_[0], classifier.intercept_[0])
        for threshold in THRESHOLDS:
            scores[threshold].append(segmentation_metrics(p >= threshold, target)['dice'])
    rows = [{'threshold': threshold, 'valid_images': len(samples), 'valid_mean_dice': float(np.mean(values))}
            for threshold, values in scores.items()]
    best = max(rows, key=lambda row: row['valid_mean_dice'])
    return best['threshold'], rows


def _evaluate_test(classifier, samples, size, threshold):
    image_rows, annotation_rows = [], []
    for index, sample in enumerate(samples, 1):
        started = time.perf_counter()
        image, target, annotations = _load_sample(sample, size)
        p = probabilities_from_features(pixel_features(image, size), classifier.coef_[0], classifier.intercept_[0])
        prediction = p >= threshold
        image_rows.append({'id': sample['id'], 'split': 'test', 'method': 'pixel_logistic',
                           'width': image.width, 'height': image.height,
                           'evaluation_width': size, 'evaluation_height': size,
                           'annotations': len(annotations), 'target_fraction': float(target.mean()),
                           'prediction_fraction': float(prediction.mean()),
                           **segmentation_metrics(prediction, target), 'seconds': time.perf_counter() - started})
        for path, annotation in zip(sample['masks'], annotations):
            annotation_rows.append({'id': sample['id'], 'split': 'test', 'method': 'pixel_logistic',
                                    'annotation': path.name, **segmentation_metrics(prediction, annotation)})
        if index % 50 == 0 or index == len(samples):
            print(f'Test evaluated {index}/{len(samples)} after frozen validation threshold={threshold}', flush=True)
    return image_rows, annotation_rows


def train(args) -> dict:
    """Run every training image each epoch; test discovery starts after selection."""
    data_dir, output = _validate_output(args)
    started = time.perf_counter()
    samples = {split: discover_samples(data_dir, split) for split in ('train', 'valid')}
    if any(not items for items in samples.values()):
        raise ValueError('Nonempty train and valid splits are required.')
    ids = {split: [sample['id'] for sample in items] for split, items in samples.items()}
    if any(len(value) != len(set(value)) for value in ids.values()) or set(ids['train']) & set(ids['valid']):
        raise ValueError('Duplicate image IDs within split or overlapping train/valid IDs.')
    print(f'Hashing train={len(samples["train"])} and valid={len(samples["valid"])} image/label provenance; test unopened.', flush=True)
    inventories = {split: _inventory(items, data_dir) for split, items in samples.items()}
    groups = defaultdict(list)
    for row in inventories['train']:
        groups[row['image_sha256']].append(row['id'])
    group_size = {sample_id: len(group) for group in groups.values() for sample_id in group}
    classifier = SGDClassifier(loss='log_loss', alpha=args.alpha, random_state=args.seed,
                               average=True, shuffle=False)
    source_files = ('backend/pixel_model.py', 'scripts/train_pixel_model.py', 'backend/metrics.py', 'scripts/audit_data.py')
    manifest = {
        'experiment': 'Supervised pixel logistic regression trained locally from scratch',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
        'parameters': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'data_root': str(data_dir),
        'splits': {split: {'count': len(items), 'ids': ids[split], 'files': inventories[split]}
                   for split, items in samples.items()},
        'duplicates': {'algorithm': 'SHA256 of original train image file bytes only; labels unchanged',
                       'weight': 'Every sampled pixel receives 1/train image hash group size',
                       'train_groups': [{'sha256': digest, 'ids': group} for digest, group in groups.items() if len(group) > 1],
                       'train_effective_image_weight': float(len(groups))},
        'protocol': {'optimizer': 'SGDClassifier(loss=log_loss, learning_rate=optimal, average=True, shuffle=False)',
                     'classes': [0, 1], 'optimizer_splits': ['train'],
                     'epochs': 'Fixed number of full passes; final epoch retained, no epoch selection',
                     'sampling': 'Seeded shuffled image order per epoch; uniform pixels without replacement per image; no foreground balancing',
                     'rng': 'numpy.default_rng(seed), sequential image permutation and pixel choice per epoch',
                     'features': list(FEATURE_NAMES), 'scaling': FEATURE_SCALING,
                     'input_size': args.input_size, 'image_resize': 'RGB bilinear to fixed square',
                     'mask_resize': 'Validate raw binary mask, nonzero to binary, nearest-neighbor',
                     'threshold_candidates': list(THRESHOLDS),
                     'selection': 'Highest mean per-image Dice on valid only; first (lowest) threshold wins exact ties',
                     'test_reference': 'Strict majority of ALL discovered annotations; ties are background',
                     'test_access': 'Test files discovered only after selection.json and model.json are saved',
                     'averaging': 'Equal weight per test image; population std; per-annotation summary separately',
                     'training_metrics': 'Online post-update log loss and Dice@0.5 on each sampled training image; descriptive, not final full-train evaluation',
                     'threads': 'One OpenCV and BLAS thread'},
        'source_sha256': {name: file_sha256(PROJECT_ROOT / name) for name in source_files},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
                        'pillow': pillow_version, 'scikit_learn': sklearn.__version__, 'platform': platform.platform()},
        'history': [], 'test': None,
        'limitations': ['Linear color/texture pixel model has no learned object shape or spatial context beyond a 7x7 gray window.',
                        'Raw-file hashes detect identical files only; re-encoded copies may remain.',
                        'Original split and all annotations are preserved; validation/test duplicates retain ordinary macro weights.',
                        'Dataset source, license and target diagnosis remain unspecified; this experiment establishes its own training provenance only.'],
    }
    _write_json(output / 'manifest.json', manifest)
    cv2.setNumThreads(1)
    rng = np.random.default_rng(args.seed)
    try:
        with threadpool_limits(limits=1):
            for epoch in range(1, args.epochs + 1):
                epoch_start, losses, dices, train_pixels = time.perf_counter(), [], [], 0
                for position, sample_index in enumerate(rng.permutation(len(samples['train'])), 1):
                    sample = samples['train'][int(sample_index)]
                    image, target, _ = _load_sample(sample, args.input_size, single=True)
                    features = pixel_features(image, args.input_size).reshape(-1, len(FEATURE_NAMES))
                    selected = rng.choice(len(features), args.pixels_per_image, replace=False)
                    x, y = np.ascontiguousarray(features[selected]), target.ravel()[selected].astype(np.int32)
                    weights = np.full(len(y), 1.0 / group_size[sample['id']], dtype=np.float32)
                    classifier.partial_fit(x, y, classes=np.array([0, 1]), sample_weight=weights)
                    probability = probabilities_from_features(x, classifier.coef_[0], classifier.intercept_[0])
                    losses.append(_log_loss(probability, y))
                    dices.append(segmentation_metrics((probability >= .5)[None, :], y[None, :])['dice'])
                    train_pixels += len(y)
                    if position % 250 == 0 or position == len(samples['train']):
                        print(f'Epoch {epoch}/{args.epochs}: trained {position}/{len(samples["train"])} images; {time.perf_counter() - epoch_start:.1f}s', flush=True)
                valid_loss, valid_dice = _validate_epoch(classifier, samples['valid'], args.input_size)
                row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'train_dice': float(np.mean(dices)),
                       'train_images': len(samples['train']), 'train_pixels': train_pixels,
                       'valid_images': len(samples['valid']), 'valid_loss': valid_loss,
                       'valid_dice_at_0_5': valid_dice, 'seconds': time.perf_counter() - epoch_start}
                manifest['history'].append(row)
                _write_json(output / 'training_log.json', manifest['history'])
                _write_json(output / 'manifest.json', manifest)
                print(json.dumps(row), flush=True)
            threshold, threshold_rows = _select_threshold(classifier, samples['valid'], args.input_size)
            selection = {'split': 'valid', 'images': len(samples['valid']), 'selected_threshold': threshold,
                         'criterion': 'macro image Dice; lowest threshold wins exact ties', 'candidates': threshold_rows,
                         'completed_utc': datetime.now(timezone.utc).isoformat()}
            write_csv(output / 'thresholds.csv', threshold_rows, ['threshold', 'valid_images', 'valid_mean_dice'])
            _write_json(output / 'selection.json', selection)
            model = {'format': MODEL_FORMAT, 'input_size': args.input_size, 'features': list(FEATURE_NAMES),
                     'scaling': FEATURE_SCALING, 'coefficients': classifier.coef_[0].astype(float).tolist(),
                     'intercept': float(classifier.intercept_[0]), 'threshold': threshold,
                     'training_origin': {'optimizer_splits': ['train'], 'threshold_split': 'valid',
                                         'train_images': len(samples['train']), 'validation_images': len(samples['valid']),
                                         'epochs': args.epochs, 'pixels_per_image_per_epoch': args.pixels_per_image,
                                         'seed': args.seed, 'from_scratch': True, 'source_sha256': manifest['source_sha256']}}
            _write_json(output / 'model.json', model)
            manifest['model_sha256'] = file_sha256(output / 'model.json')
            manifest['selection'] = selection
            print(f'Validation selection frozen: threshold={threshold}; model saved. Opening test now.', flush=True)
            # Do not move discovery or any access to test before the selection above.
            test_samples = discover_samples(data_dir, 'test')
            if not test_samples:
                raise ValueError('Nonempty test split required for the held-out evaluation.')
            test_inventory = _inventory(test_samples, data_dir)
            manifest['splits']['test'] = {'count': len(test_samples), 'ids': [sample['id'] for sample in test_samples], 'files': test_inventory}
            per_image, per_annotation = _evaluate_test(classifier, test_samples, args.input_size, threshold)
        write_csv(output / 'per_image.csv', per_image, IMAGE_FIELDS)
        write_csv(output / 'per_annotation.csv', per_annotation, ['id', 'split', 'method', 'annotation', *METRIC_NAMES])
        summary = {'method': 'pixel_logistic', 'images': len(per_image), 'annotations': len(per_annotation),
                   'threshold': threshold, 'consensus': macro_summary(per_image),
                   'per_annotation': macro_summary(per_annotation)}
        _write_json(output / 'summary.json', summary)
        write_csv(output / 'summary.csv', [{'method': 'pixel_logistic', 'split': 'test', 'images': len(per_image),
                  'annotations': len(per_annotation), 'threshold': threshold, **summary['consensus']['mean']}],
                  ['method', 'split', 'images', 'annotations', 'threshold', *METRIC_NAMES])
        manifest['test'], manifest['status'] = summary, 'complete'
    except Exception as error:
        manifest['status'], manifest['error'] = 'failed', f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['wall_seconds'] = time.perf_counter() - started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        _write_json(output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'model_sha256': manifest['model_sha256'], 'test': manifest['test']}, indent=2), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/pixel_classifier')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--pixels-per-image', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=406)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=1e-4)
    train(parser.parse_args(argv))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
