"""Score frozen compact runs and references using a common 256-square protocol.

Freeze/config/history/checkpoint validation precedes any data-set discovery.
This module imports inference code only, never training or optimization code.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from backend.baseline import BASELINE_PROTOCOL, baseline_probability
from backend.metrics import METRIC_NAMES, segmentation_metrics
from scripts.audit_data import PROJECT_ROOT, discover_samples, file_sha256
from scripts.evaluate import load_reference, write_csv

SIZE = (256, 256)
BOUNDARY_FIELDS = ('pred_boundary_pixels', 'target_boundary_pixels', 'pred_boundary_matched', 'target_boundary_matched')
SCORES = (*METRIC_NAMES, 'boundary_f1')
COUNTS = ('TP', 'FP', 'FN', 'TN')
LEGACY_INPUT_PREPROCESSING = 'native-rgb-to-model-v1'
INPUT_PREPROCESSING_V2 = 'rgb-bilinear-256-then-model-v2'


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read manifest {path}: {exc}') from exc


def _sha_string(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value) is not None


def verify_frozen_runs(runs_dir: Path, cache_dir: Path | None = None) -> list[dict]:
    """Validate portable historical artifacts, without touching evaluation data.

    Current code changes are reported, rather than invalidating a historical
    source snapshot. A relocated cache manifest can corroborate the recorded
    source dataset fingerprint; its original absolute path is not required.
    """
    runs_dir = Path(runs_dir).resolve()
    if not runs_dir.is_dir():
        raise ValueError(f'No frozen runs directory: {runs_dir}')
    manifests = sorted(path / 'frozen-selection.json' for path in runs_dir.iterdir()
                       if path.is_dir() and (path / 'frozen-selection.json').is_file())
    if not manifests:
        raise ValueError('No frozen-selection.json files found; test access is prohibited.')
    result = []
    for manifest_path in manifests:
        run = manifest_path.parent.resolve()
        if not run.is_relative_to(runs_dir) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run.name):
            raise ValueError('Unsafe frozen run path or method ID.')
        if run.name in {'baseline', 'transunet', 'pixel_logistic'}:
            raise ValueError('Frozen run name collides with a reference method ID.')
        frozen = _read_json(manifest_path)
        if frozen.get('format') != 'compact-selection-v1' or frozen.get('status') != 'frozen' or frozen.get('model') != 'compact_unet':
            raise ValueError(f'Invalid frozen selection status/format: {run.name}')
        if (frozen.get('optimizer_splits') != ['train'] or frozen.get('selection_split') != 'valid'
                or frozen.get('test_accessed') is not False or frozen.get('evaluation_size') != 256):
            raise ValueError(f'Invalid frozen training/selection split policy: {run.name}')
        threshold = frozen.get('threshold')
        if not isinstance(threshold, (float, int)) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError(f'Invalid frozen threshold: {run.name}')
        checkpoint_name = frozen.get('checkpoint')
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name or ':' in checkpoint_name:
            raise ValueError('Checkpoint must be a local run filename.')
        checkpoint = (run / checkpoint_name).resolve()
        for kind, path in [('checkpoint', checkpoint), ('config', run / 'config.json'), ('history', run / 'history.json')]:
            expected = frozen.get(f'{kind}_sha256')
            if not path.resolve().is_relative_to(run) or not path.is_file() or not _sha_string(expected) or file_sha256(path) != expected:
                raise ValueError(f'Frozen {kind} hash mismatch: {run.name}')
        config, history = _read_json(run / 'config.json'), _read_json(run / 'history.json')
        if (config.get('status') != 'frozen' or config.get('optimizer_splits') != ['train']
                or config.get('selection_split') != 'valid' or config.get('test_accessed') is not False):
            raise ValueError(f'Invalid training config split policy: {run.name}')
        counts = frozen.get('counts')
        if (not isinstance(counts, dict) or set(counts) != {'train', 'valid'}
                or any(type(counts[key]) is not int or counts[key] < 1 for key in counts)
                or config.get('counts') != counts):
            raise ValueError(f'Inconsistent frozen training counts: {run.name}')
        fingerprint = frozen.get('cache_fingerprint')
        if not _sha_string(fingerprint) or config.get('data_fingerprint') != fingerprint:
            raise ValueError(f'Inconsistent frozen source data fingerprint: {run.name}')
        input_preprocessing = frozen.get('input_preprocessing', LEGACY_INPUT_PREPROCESSING)
        if (input_preprocessing not in {LEGACY_INPUT_PREPROCESSING, INPUT_PREPROCESSING_V2}
                or config.get('input_preprocessing', LEGACY_INPUT_PREPROCESSING) != input_preprocessing):
            raise ValueError(f'Inconsistent config/frozen RGB preprocessing recipe: {run.name}')
        sources = frozen.get('source_sha256')
        if not isinstance(sources, dict) or not sources or config.get('source_sha256') != sources:
            raise ValueError(f'Inconsistent historical training source hashes: {run.name}')
        current_sources = {}
        for name, digest in sources.items():
            source = (PROJECT_ROOT / name).resolve()
            if not _sha_string(digest) or not source.is_relative_to(PROJECT_ROOT.resolve()):
                raise ValueError('Invalid historical source hash or path.')
            current = file_sha256(source) if source.is_file() else None
            current_sources[name] = {'trained_sha256': digest, 'current_sha256': current, 'matches_current': current == digest}
        for key in ('seed', 'epochs', 'best_epoch'):
            if type(frozen.get(key)) is not int or frozen[key] != config.get(key):
                raise ValueError(f'Inconsistent frozen {key}: {run.name}')
        epochs, best_epoch = frozen['epochs'], frozen['best_epoch']
        if epochs < 1 or not 1 <= best_epoch <= epochs or not isinstance(history, list) or len(history) != epochs:
            raise ValueError(f'Incomplete frozen training history: {run.name}')
        for index, row in enumerate(history, 1):
            if (row.get('epoch') != index or row.get('train_samples') != counts['train']
                    or row.get('valid_samples') != counts['valid'] or row.get('updated_parameters', 0) < 1
                    or not isinstance(row.get('valid_dice'), (int, float))
                    or not math.isfinite(row['valid_dice']) or not 0 <= row['valid_dice'] <= 1):
                raise ValueError(f'Invalid frozen history coverage: {run.name}')
        if max(range(len(history)), key=lambda index: history[index]['valid_dice']) + 1 != best_epoch:
            raise ValueError(f'Frozen best epoch does not match earliest maximum validation Dice: {run.name}')
        search = frozen.get('threshold_search')
        if not isinstance(search, list) or not search:
            raise ValueError(f'Missing validation threshold search: {run.name}')
        for row in search:
            if (not isinstance(row.get('threshold'), (int, float)) or not 0 < row['threshold'] < 1
                    or not isinstance(row.get('valid_dice'), (int, float)) or not math.isfinite(row['valid_dice'])
                    or not 0 <= row['valid_dice'] <= 1):
                raise ValueError('Invalid validation threshold search values.')
        selected = max(search, key=lambda row: row['valid_dice'])
        if selected['threshold'] != threshold or selected['valid_dice'] != frozen.get('valid_dice'):
            raise ValueError(f'Threshold selection is inconsistent with recorded validation search: {run.name}')
        candidates = [Path(cache_dir) / 'manifest.json'] if cache_dir is not None else []
        if cache_dir is None:
            if isinstance(config.get('cache_dir'), str):
                candidates.append(Path(config['cache_dir']) / 'manifest.json')
            candidates.extend(runs_dir.parent / directory / 'manifest.json' for directory in ('compact-cache', 'cache'))
        available = next((path for path in candidates if path.is_file()), None)
        if cache_dir is not None and available is None:
            raise ValueError('Explicit relocated cache manifest is missing.')
        cache_verification = {'performed': False, 'reason': 'Original/relocated cache manifest unavailable; config/frozen fingerprint and counts agree.'}
        if available is not None:
            cache = _read_json(available)
            if cache.get('fingerprint') != fingerprint or cache.get('counts') != counts:
                raise ValueError(f'Cache fingerprint/count mismatch: {run.name}')
            samples = cache.get('samples')
            if (not isinstance(samples, dict) or set(samples) != {'train', 'valid'}
                    or any(len(samples[split]) != counts[split] for split in counts)
                    or cache.get('image_size') != config.get('image_size') or cache.get('test_accessed') is not False):
                raise ValueError('Cache manifest coverage or split policy is invalid.')
            cache_format = cache.get('format')
            fingerprint_payload = {'size': cache['image_size'], 'samples': samples}
            if cache_format == 'compact-training-cache-v2':
                if cache.get('preprocessing') != INPUT_PREPROCESSING_V2 or input_preprocessing != INPUT_PREPROCESSING_V2:
                    raise ValueError('V2 cache preprocessing recipe must match the V2 config/frozen recipe.')
                fingerprint_payload['preprocessing'] = cache['preprocessing']
            elif cache_format == 'compact-training-cache-v1':
                if (cache.get('preprocessing', LEGACY_INPUT_PREPROCESSING) != LEGACY_INPUT_PREPROCESSING
                        or input_preprocessing != LEGACY_INPUT_PREPROCESSING):
                    raise ValueError('V1 cache preprocessing recipe must match the legacy config/frozen recipe.')
            else:
                raise ValueError('Unsupported cache format/preprocessing recipe.')
            reproduced = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
            if reproduced != fingerprint:
                raise ValueError('Stored source samples do not reproduce the cache fingerprint.')
            cache_verification = {'performed': True, 'manifest': str(available.resolve()), 'sha256': file_sha256(available),
                                  'format': cache_format, 'preprocessing': input_preprocessing,
                                  'fingerprint_reproduced_from_source_hash_inventory': True,
                                  'raw_training_files_rehashed': False}
        result.append({'method': run.name, 'run_dir': str(run), 'checkpoint': str(checkpoint), 'threshold': threshold,
                       'input_preprocessing': input_preprocessing,
                       'frozen': frozen, 'config': config, 'frozen_manifest_sha256': file_sha256(manifest_path),
                       'cache_verification': cache_verification, 'historical_source_comparison': current_sources})
    return result


def _boundary_counts(prediction, target, tolerance=2) -> dict:
    prediction, target = np.asarray(prediction), np.asarray(target)
    if (prediction.shape != target.shape or prediction.ndim != 2 or not prediction.size
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('Boundary masks must be finite, nonempty, matching 2-D arrays.')
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Boundary tolerance must be finite and nonnegative.')
    boundaries = []
    for values in (prediction, target):
        binary = values.astype(bool).astype(np.uint8)
        eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        boundaries.append((binary != 0) & (eroded == 0))
    predicted, actual = boundaries
    counts = {'pred_boundary_pixels': int(predicted.sum()), 'target_boundary_pixels': int(actual.sum()),
              'pred_boundary_matched': 0, 'target_boundary_matched': 0}
    if predicted.any() and actual.any():
        to_actual = cv2.distanceTransform((~actual).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_predicted = cv2.distanceTransform((~predicted).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        counts['pred_boundary_matched'] = int(np.count_nonzero(to_actual[predicted] <= tolerance))
        counts['target_boundary_matched'] = int(np.count_nonzero(to_predicted[actual] <= tolerance))
    return counts


def _boundary_score(counts: dict) -> float:
    predicted, actual = counts['pred_boundary_pixels'], counts['target_boundary_pixels']
    if not predicted and not actual:
        return 1.0
    if not predicted or not actual:
        return 0.0
    precision = counts['pred_boundary_matched'] / predicted
    recall = counts['target_boundary_matched'] / actual
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def boundary_f1(prediction, target, tolerance=2) -> float:
    return _boundary_score(_boundary_counts(prediction, target, tolerance))


def score_masks(prediction, target) -> dict:
    metrics = segmentation_metrics(prediction, target)
    predicted, actual = np.asarray(prediction).astype(bool), np.asarray(target).astype(bool)
    counts = {'TP': int(np.count_nonzero(predicted & actual)), 'FP': int(np.count_nonzero(predicted & ~actual)),
              'FN': int(np.count_nonzero(~predicted & actual)), 'TN': int(np.count_nonzero(~predicted & ~actual))}
    boundary = _boundary_counts(predicted, actual)
    return {**counts, **metrics, **boundary, 'boundary_f1': _boundary_score(boundary)}


def aggregate_rows(rows: list[dict]) -> dict:
    totals = {key: sum(row[key] for row in rows) for key in COUNTS}
    if not rows:
        return {'sample_count': 0, 'macro': None, 'micro': None, 'confusion': totals}
    values = {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in SCORES}
    macro = {'mean': {key: float(value.mean()) for key, value in values.items()},
             'std_population': {key: float(value.std(ddof=0)) for key, value in values.items()},
             'median': {key: float(np.median(value)) for key, value in values.items()}}
    tp, fp, fn, tn = (totals[key] for key in COUNTS)
    if tp + fp + fn == 0:
        micro = dict.fromkeys(METRIC_NAMES, 1.0)
    else:
        micro = {'dice': 2 * tp / (2 * tp + fp + fn), 'iou': tp / (tp + fp + fn),
                 'pixel_accuracy': (tp + tn) / (tp + fp + fn + tn),
                 'precision': tp / (tp + fp) if tp + fp else 0.0,
                 'recall': tp / (tp + fn) if tp + fn else 0.0}
    boundary = {key: sum(row[key] for row in rows) for key in BOUNDARY_FIELDS}
    micro['boundary_f1'] = _boundary_score(boundary)
    return {'sample_count': len(rows), 'macro': macro, 'micro': micro, 'confusion': totals,
            'boundary_totals': boundary}


def evaluate(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Evaluation output already exists: {output}')
    if args.split not in {'test', 'valid'} or args.device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Invalid evaluation split or device.')
    started = time.perf_counter()
    runs = verify_frozen_runs(Path(args.runs_dir), getattr(args, 'cache_dir', None))
    import torch
    from backend.compact_model import predict_compact

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    methods = [{'id': 'baseline', 'threshold': .5, 'predict': baseline_probability,
                'provenance': BASELINE_PROTOCOL}]
    for run in runs:
        methods.append({'id': run['method'], 'threshold': run['threshold'],
                        'predict': lambda image, checkpoint=run['checkpoint']: predict_compact(image, checkpoint, device=device),
                        'provenance': run})
    reference_checkpoint, pixel_model = getattr(args, 'reference_checkpoint', None), getattr(args, 'pixel_model', None)
    reference_environment = {}
    if reference_checkpoint is not None:
        checkpoint = Path(reference_checkpoint).resolve()
        if not checkpoint.is_file():
            raise ValueError('Supplied reference checkpoint is missing.')
        from backend.inference import predict_transunet
        reference_environment = {'TRANSUNET_CHECKPOINT': str(checkpoint), 'TRANSUNET_DEVICE': device, 'TRANSUNET_CPU_THREADS': '4'}
        methods.append({'id': 'transunet', 'threshold': .5,
                        'predict': lambda image: predict_transunet(image, normalization='unit'),
                        'provenance': {'checkpoint': str(checkpoint), 'sha256': file_sha256(checkpoint),
                                       'normalization': 'unit', 'threshold': .5,
                                       'training_history': 'Unknown; no claim of unseen local/external test data'}})
    if pixel_model is not None:
        from backend.pixel_model import load_pixel_model, predict_pixel_model
        path = Path(pixel_model).resolve()
        model = load_pixel_model(path)
        methods.append({'id': 'pixel_logistic', 'threshold': model['threshold'],
                        'predict': lambda image, model_path=path: predict_pixel_model(image, model_path),
                        'provenance': {'model': str(path), 'sha256': file_sha256(path),
                                       'training_origin': model.get('training_origin', {}), 'input_size': model['input_size']}})
    verified_at = datetime.now(timezone.utc).isoformat()
    # No data-directory discovery, pixel opening, or source manifest reading is permitted above this line.
    data_dir = Path(args.data_dir).resolve()
    samples = discover_samples(data_dir, args.split)
    if not samples:
        raise ValueError(f'No evaluation images found for {args.split}.')
    if len({sample['id'] for sample in samples}) != len(samples):
        raise ValueError('Duplicate evaluation image IDs would overwrite exported masks.')
    inventory = []
    for sample in samples:
        paths = [sample['image'], *sample['masks']]
        for path in paths:
            if not path.resolve().is_relative_to(data_dir):
                raise ValueError('Evaluation image or annotation lies outside data root.')
        inventory.append({'id': sample['id'], 'image': sample['image'].relative_to(data_dir).as_posix(),
                          'image_sha256': file_sha256(sample['image']),
                          'masks': [{'path': path.relative_to(data_dir).as_posix(), 'sha256': file_sha256(path)} for path in sample['masks']]})
    dataset = {'role': 'previously_inspected_local_benchmark', 'source_split': args.split, 'data_root': str(data_dir),
               'manifest_sha256': None}
    manifest_path = data_dir / 'manifest.json'
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if 'external' in str(manifest.get('role', '')).lower() or manifest.get('dataset') == 'ISIC 2018 Task 1 validation':
            if manifest.get('counts') != {'images': len(samples), 'masks': sum(len(sample['masks']) for sample in samples)}:
                raise ValueError('External manifest counts do not match discovered image/mask pairs.')
            if manifest.get('samples') is not None:
                external_by_id = {row['id']: row for row in manifest['samples']}
                if set(external_by_id) != {row['id'] for row in inventory}:
                    raise ValueError('External manifest sample IDs do not match evaluation samples.')
                for row in inventory:
                    source = external_by_id[row['id']]
                    if (source.get('image_sha256') != row['image_sha256'] or len(row['masks']) != 1
                            or source.get('mask_sha256') != row['masks'][0]['sha256']):
                        raise ValueError('External prepared payload hash mismatch.')
            dataset.update(role='external_stress_test', source_split=manifest.get('source_split'),
                           name=manifest.get('dataset'), license=manifest.get('license'),
                           preparation_manifest=str(manifest_path), manifest_sha256=file_sha256(manifest_path))
    output.mkdir(parents=True)
    for method in methods:
        (output / 'masks' / method['id']).mkdir(parents=True)
    per_image, per_annotation, failures = [], [], []
    prepared = []
    for sample in samples:
        try:
            with Image.open(sample['image']) as source:
                original_size = source.size
                resized = source.convert('RGB').resize(SIZE, Image.Resampling.BILINEAR)
            target, annotations = load_reference(sample, SIZE)
        except (OSError, ValueError) as exc:
            failures.append({'id': sample['id'], 'method': 'reference', 'error': f'{type(exc).__name__}: {exc}'})
            continue
        match = re.fullmatch(r'IMG_(ENC\d+)_\d+', sample['id'])
        prepared.append((sample, resized, original_size, target, annotations, match.group(1) if match else ''))
    old_environment = {key: os.environ.get(key) for key in reference_environment}
    try:
        os.environ.update(reference_environment)
        # Reuse the same prepared inputs and avoid thrashing the inference
        # module's three-model cache when a suite contains six compact runs.
        for method in methods:
            for position, (sample, resized, original_size, target, annotations, encounter) in enumerate(prepared, 1):
                try:
                    prediction_started = time.perf_counter()
                    probability = np.asarray(method['predict'](resized))
                    seconds = time.perf_counter() - prediction_started
                    if (probability.shape != target.shape or not np.isfinite(probability).all()
                            or probability.min() < 0 or probability.max() > 1):
                        raise ValueError('Predictor returned invalid probability dimensions or values.')
                    prediction = probability >= method['threshold']
                    shared = {'method': method['id'], 'id': sample['id'], 'encounter': encounter,
                              'split': args.split, 'width': original_size[0], 'height': original_size[1],
                              'evaluation_width': 256, 'evaluation_height': 256, 'annotations': len(annotations),
                              'threshold': method['threshold'], 'prediction_fraction': float(prediction.mean()), 'seconds': seconds}
                    row = {**shared, 'target_fraction': float(target.mean()), **score_masks(prediction, target)}
                    annotated = [{**shared, 'annotation': path.name, 'target_fraction': float(annotation.mean()),
                                  **score_masks(prediction, annotation)} for path, annotation in zip(sample['masks'], annotations)]
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(output / 'masks' / method['id'] / f'{sample["id"]}.png')
                    per_image.append(row)
                    per_annotation.extend(annotated)
                except Exception as exc:
                    failures.append({'id': sample['id'], 'method': method['id'], 'error': f'{type(exc).__name__}: {exc}'})
                if position % 25 == 0 or position == len(prepared):
                    print(f'Evaluated {method["id"]}: {position}/{len(prepared)} images; {time.perf_counter()-started:.1f}s', flush=True)
    finally:
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    fields = ['method', 'id', 'encounter', 'split', 'width', 'height', 'evaluation_width', 'evaluation_height',
              'annotations', 'threshold', 'target_fraction', 'prediction_fraction', *COUNTS, *SCORES, *BOUNDARY_FIELDS, 'seconds']
    write_csv(output / 'per_image.csv', per_image, fields)
    write_csv(output / 'per_annotation.csv', per_annotation, [*fields, 'annotation'])
    summary = {
        'format': 'cs406-completion-evaluation-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'frozen_verification_completed_utc': verified_at, 'frozen_verified_before_data_discovery': True,
        'dataset': dataset, 'split': args.split, 'available_images': len(samples),
        'requested_ids': [sample['id'] for sample in samples], 'input_inventory': inventory,
        'dataset_fingerprint': hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
        'protocol': {'evaluation_size': 256, 'image_resize': 'RGB, PIL bilinear to 256x256, then predictor-specific internal resize',
                     'compact_preprocessing': {run['method']: run['input_preprocessing'] for run in runs},
                     'execution': 'Prepare shared256RGB/reference arrays once in RAM, then process methods sequentially to reuse checkpoint cache',
                     'mask_resize': 'Each annotation nonzero-to-binary, nearest-neighbor to 256x256, then strict pixel majority; ties background',
                     'thresholds': 'Compact: frozen validation selection; baseline/TransUNet:0.5; pixel logistic:validated model JSON threshold',
                     'macro': 'Equal weight per successfully scored image; per-annotation macro weights each available annotation equally',
                     'micro': 'Pool TP/FP/FN/TN; boundary F1 pools matched/detected boundary counts',
                     'empty_masks': 'Both empty all scores1; only one empty foreground scores0; pixel accuracy actual agreement',
                     'boundary': 'Mask minus 3x3 erosion, zero outside image; symmetric nearest-boundary matching within Euclidean2pixels via precise L2 distance transform',
                     'latency': 'seconds measures predictor call including lazy load on first image; no warm-up; shared across annotations; excludes reference/scoring/export',
                     'export': '0/255 grayscale PNG at256x256; masks/<method>/<id>.png',
                     'training_performed': False, 'parameter_selection_performed': False,
                     'encounter': 'Filename ENC group, not a verified patient identity; blank when unavailable'},
        'methods': {}, 'failures': failures, 'comparison_complete': not failures,
        'counts': {'images': len(samples), 'annotations': sum(len(sample['masks']) for sample in samples),
                   'methods': len(methods), 'per_image_rows': len(per_image), 'per_annotation_rows': len(per_annotation)},
        'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
                        'opencv': cv2.__version__, 'pillow': pillow_version, 'device': device, 'cpu_threads': torch.get_num_threads(),
                        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'platform': platform.platform()},
        'wall_seconds': time.perf_counter() - started,
        'limitations': ['The local test set was historically inspected before current completion work; it is not a pristine untouched holdout.',
                        'Frozen manifests establish recorded artifact consistency, not independent attestation of all historical conduct.',
                        'Supplied TransUNet training history is unknown; neither local nor external unseen-data independence is established for it.',
                        'Filename encounters do not establish patient-disjoint sampling; duplicate images retain ordinary macro weight.',
                        'External dermoscopy boundaries differ from consumer-photo problem regions; this is stress testing, not clinical validation.',
                        'Current source differences are reported without invalidating historical source hash records; missing cache manifests are disclosed.'],
    }
    for method in methods:
        rows = [row for row in per_image if row['method'] == method['id']]
        annotations = [row for row in per_annotation if row['method'] == method['id']]
        summary['methods'][method['id']] = {**aggregate_rows(rows), 'threshold': method['threshold'],
                                           'per_annotation': aggregate_rows(annotations), 'provenance': method['provenance'],
                                           'evaluated_ids': [row['id'] for row in rows],
                                           'annotation_count_distribution': dict(Counter(row['annotations'] for row in rows)),
                                           'mean_prediction_seconds': float(np.mean([row['seconds'] for row in rows])) if rows else None}
    sources = ['scripts/evaluate_completion.py', 'scripts/evaluate.py', 'scripts/audit_data.py',
               'backend/compact_model.py', 'backend/baseline.py', 'backend/metrics.py']
    if reference_checkpoint is not None:
        sources += ['backend/inference.py', 'backend/transunet.py']
    if pixel_model is not None:
        sources += ['backend/pixel_model.py']
    summary['source_sha256'] = {name: file_sha256(PROJECT_ROOT / name) for name in sources}
    summary['output_sha256'] = {name: file_sha256(output / name) for name in ('per_image.csv', 'per_annotation.csv')}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--reference-checkpoint', type=Path)
    parser.add_argument('--pixel-model', type=Path)
    parser.add_argument('--split', choices=['test', 'valid'], default='test')
    parser.add_argument('--cache-dir', type=Path, help='Optional relocated training-cache directory for source fingerprint corroboration.')
    args = parser.parse_args(argv)
    result = evaluate(args)
    print(json.dumps({'output': str(args.output.resolve()), 'counts': result['counts'],
                      'methods': {key: value['macro']['mean'] if value['macro'] else None for key, value in result['methods'].items()},
                      'failure_count': len(result['failures'])}, indent=2))
    return 1 if result['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
