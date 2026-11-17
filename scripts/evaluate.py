"""Reproducible local foreground segmentation evaluation; no training claims.

Example: python -m scripts.evaluate --method baseline --split test
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image, __version__ as pillow_version

from backend.baseline import BASELINE_PROTOCOL, baseline_probability
from backend.metrics import METRIC_NAMES, segmentation_metrics
from scripts.audit_data import PROJECT_ROOT, discover_samples, file_sha256


def load_reference(sample: dict, size: tuple[int, int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Nearest-neighbor resize each binary annotation, then strict majority.

    With 2 or 4 annotators, a tied pixel is background. Consensus is computed
    only from annotations and never influences the predictor.
    """
    if not sample['masks']:
        raise ValueError('Image has no discovered reference annotations.')
    with Image.open(sample['image']) as source:
        original_size = source.size
    annotations = []
    for path in sample['masks']:
        with Image.open(path) as mask:
            if mask.size != original_size:
                raise ValueError(f'Annotation {path.name} does not match original image dimensions.')
            binary = np.asarray(mask)
            if binary.ndim != 2:
                raise ValueError(f'Annotation {path.name} must be a 2-D mask.')
            resized = Image.fromarray((binary != 0).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
            annotations.append(np.asarray(resized, dtype=bool))
    consensus = np.stack(annotations).sum(axis=0) > len(annotations) / 2
    return consensus, annotations


def macro_summary(rows: list[dict]) -> dict:
    if not rows:
        return {'mean': {}, 'std_population': {}, 'median': {}}
    values = {name: np.array([row[name] for row in rows], dtype=np.float64) for name in METRIC_NAMES}
    return {
        'mean': {name: float(arr.mean()) for name, arr in values.items()},
        'std_population': {name: float(arr.std(ddof=0)) for name, arr in values.items()},
        'median': {name: float(np.median(arr)) for name, arr in values.items()},
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> dict:
    methods = ['baseline', 'transunet'] if args.method == 'both' else [args.method]
    samples = discover_samples(args.data_dir, args.split)
    available = len(samples)
    samples = samples[:args.limit] if args.limit is not None else samples
    if not samples:
        raise ValueError(f'No images found for split {args.split}.')
    if args.evaluation_size < 0 or args.evaluation_size == 1:
        raise ValueError('Evaluation size must be 0 (original resolution) or at least 2.')
    args.output.mkdir(parents=True, exist_ok=True)
    predictor, model_status = None, None
    if 'transunet' in methods:
        from backend.inference import get_model_status, predict_transunet
        predictor = predict_transunet
        model_status = get_model_status
    per_image, per_annotation, failures = [], [], []
    started = time.perf_counter()
    for index, sample in enumerate(samples, 1):
        try:
            with Image.open(sample['image']) as source:
                image = source.convert('RGB')
            original_size = image.size
            size = (args.evaluation_size, args.evaluation_size) if args.evaluation_size else original_size
            resized = image.resize(size, Image.Resampling.BILINEAR)
            consensus, annotations = load_reference(sample, size)
        except (OSError, ValueError) as exc:
            failures.append({'id': sample['id'], 'method': 'reference', 'error': str(exc)})
            continue
        for method in methods:
            method_start = time.perf_counter()
            try:
                probabilities = baseline_probability(resized) if method == 'baseline' else predictor(resized, normalization=args.normalization)
                probabilities = np.asarray(probabilities)
                if probabilities.shape != consensus.shape or not np.isfinite(probabilities).all():
                    raise ValueError('Predictor returned invalid probability shape or nonfinite values.')
                if probabilities.min() < 0 or probabilities.max() > 1:
                    raise ValueError('Predictor scores must lie in [0, 1].')
                prediction = probabilities >= args.threshold
                row = {
                    'id': sample['id'], 'split': args.split, 'method': method,
                    'width': original_size[0], 'height': original_size[1],
                    'evaluation_width': size[0], 'evaluation_height': size[1],
                    'annotations': len(annotations),
                    'target_fraction': float(consensus.mean()), 'prediction_fraction': float(prediction.mean()),
                    **segmentation_metrics(prediction, consensus),
                    'seconds': time.perf_counter() - method_start,
                }
                per_image.append(row)
                if args.per_annotation:
                    for path, annotation in zip(sample['masks'], annotations):
                        per_annotation.append({'id': sample['id'], 'split': args.split, 'method': method, 'annotation': path.name, **segmentation_metrics(prediction, annotation)})
            except Exception as exc:
                failures.append({'id': sample['id'], 'method': method, 'error': f'{type(exc).__name__}: {exc}'})
        if index % 25 == 0 or index == len(samples):
            print(f'Evaluated {index}/{len(samples)} images; {time.perf_counter() - started:.1f}s', flush=True)
    fields = ['id', 'split', 'method', 'width', 'height', 'evaluation_width', 'evaluation_height', 'annotations', 'target_fraction', 'prediction_fraction', *METRIC_NAMES, 'seconds']
    write_csv(args.output / 'per_image.csv', per_image, fields)
    if args.per_annotation:
        write_csv(args.output / 'per_annotation.csv', per_annotation, ['id', 'split', 'method', 'annotation', *METRIC_NAMES])
    source_files = ['backend/baseline.py', 'backend/metrics.py', 'scripts/evaluate.py', 'scripts/audit_data.py']
    if 'transunet' in methods:
        source_files.extend(['backend/inference.py', 'backend/transunet.py'])
    manifest = '\n'.join(f"{s['image'].relative_to(args.data_dir).as_posix()}:{s['image'].stat().st_size}:" + ','.join(f'{p.name}:{p.stat().st_size}' for p in s['masks']) for s in samples)
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': args.split, 'available_images': available, 'requested_images': len(samples),
        'limited_run': args.limit is not None and args.limit < available,
        'requested_ids': [sample['id'] for sample in samples],
        'data_root': str(args.data_dir.resolve()), 'manifest_names_sizes_sha256': hashlib.sha256(manifest.encode()).hexdigest(),
        'protocol': {
            'task': 'Binary foreground segmentation against supplied local TIFF annotations',
            'image_order': 'Lexicographic filename order; optional limit selects first N, not a random sample',
            'evaluation_size': args.evaluation_size or 'original',
            'image_resize': 'Bilinear; fixed square if evaluation_size>0 (aspect ratio may change)',
            'mask_resize': 'Nonzero-to-binary, then nearest-neighbor to evaluation dimensions',
            'reference': 'Pixelwise strict majority of ALL discovered annotations; ties are background',
            'annotation_discovery': 'Exact image stem or image_stem_mask_annN, without assuming suffix start or count',
            'threshold': args.threshold, 'comparison': 'probability >= threshold',
            'averaging': 'Macro: each successfully evaluated image has equal weight; population std',
            'empty_masks': 'Both empty: all scores=1. One empty: foreground scores=0; pixel accuracy uses actual agreement.',
            'transunet_normalization': args.normalization if 'transunet' in methods else None,
            'transunet_resize': 'Inference additionally resizes the evaluation RGB image to checkpoint input size; sigmoid probabilities are bilinearly restored to evaluation dimensions.' if 'transunet' in methods else None,
            'per_annotation': args.per_annotation,
            'per_annotation_averaging': 'Each annotation has equal weight; images with more annotations receive more weight.' if args.per_annotation else None,
            'baseline': BASELINE_PROTOCOL if 'baseline' in methods else None,
        },
        'methods': {}, 'failures': failures,
        'wall_seconds': time.perf_counter() - started,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__, 'pillow': pillow_version, 'platform': platform.platform()},
        'source_sha256': {name: file_sha256(PROJECT_ROOT / name) for name in source_files if (PROJECT_ROOT / name).is_file()},
        'limitations': [
            'Dataset source, license, acquisition process and precise target diagnosis were not supplied.',
            'Supplied checkpoint training split, optimization history and original preprocessing are unknown.',
            'No training/validation loss curves or training-time measurements are available or fabricated.',
            'Current subject-disjoint splits do not rule out pretrained checkpoint exposure to test images.',
            'These are local reproducibility measurements, not independently validated clinical performance.',
            'Square resized evaluation does not measure boundary quality at original image resolution.',
            'Consensus suppresses annotator disagreement; per-annotation CSV is available with --per-annotation.',
            'Pixel accuracy can be inflated by background; compare Dice and IoU as foreground metrics.',
            'The baseline returns binary heuristic scores rather than calibrated probabilities.',
        ],
    }
    examples = {}
    for method in methods:
        rows = [row for row in per_image if row['method'] == method]
        annotations = [row for row in per_annotation if row['method'] == method]
        summary['methods'][method] = {
            'sample_count': len(rows), 'evaluated_ids': [row['id'] for row in rows],
            'annotation_count_distribution': dict(sorted(Counter(row['annotations'] for row in rows).items())),
            'macro': macro_summary(rows),
            'per_annotation_macro': macro_summary(annotations) if args.per_annotation else None,
            'mean_prediction_seconds': float(np.mean([row['seconds'] for row in rows])) if rows else None,
        }
        ranked = sorted(rows, key=lambda row: (row['dice'], row['id']))
        keep = lambda row: {key: row[key] for key in ('id', 'dice', 'iou', 'target_fraction', 'prediction_fraction', 'annotations')}
        examples[method] = {'worst_by_dice': [keep(row) for row in ranked[:8]], 'best_by_dice': [keep(row) for row in ranked[-8:][::-1]]}
    if 'transunet' in methods:
        summary['model_status'] = model_status()
        checkpoint = Path(summary['model_status']['checkpoint_path'])
        summary['checkpoint'] = {'file': checkpoint.name, 'bytes': checkpoint.stat().st_size, 'sha256': file_sha256(checkpoint)} if checkpoint.is_file() else {'file': str(checkpoint), 'found': False}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'examples.json').write_text(json.dumps({'selection': 'Deterministic post-hoc ranking by consensus Dice; no parameter tuning from these examples.', 'methods': examples, 'failures': failures}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['baseline', 'transunet', 'both'], default='baseline')
    parser.add_argument('--split', choices=['test', 'valid'], default='test')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--evaluation-size', type=int, default=256, help='Square evaluation resolution; 0 keeps original image size.')
    parser.add_argument('--normalization', choices=['unit', 'imagenet'], default='unit')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--per-annotation', action='store_true')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if not 0 < args.threshold <= 1:
        parser.error('--threshold must be in (0, 1]')
    if args.evaluation_size < 0 or args.evaluation_size == 1:
        parser.error('--evaluation-size must be 0 or at least 2')
    args.output = args.output or PROJECT_ROOT / f'results/{args.method}_{args.split}'
    summary = evaluate(args)
    print(json.dumps({'output': str(args.output), 'methods': {name: {'sample_count': report['sample_count'], 'macro': report['macro']['mean']} for name, report in summary['methods'].items()}, 'failure_count': len(summary['failures'])}, indent=2))
    return 1 if summary['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
