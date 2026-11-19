"""Analyze verified evaluation CSVs without fitting or choosing model parameters."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCORES = ('dice', 'iou', 'precision', 'recall', 'pixel_accuracy', 'boundary_f1')
PRIMARY = 'bce_dice_aug_s406'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _indexed(rows):
    result = {}
    for row in rows:
        if not row.get('id') or row['id'] in result:
            raise ValueError('Duplicate or missing image IDs')
        result[row['id']] = row
    if not result:
        raise ValueError('No paired images')
    return result


def paired_cluster_bootstrap(left, right, metric='dice', iterations=2000, seed=406):
    """Resample encounters uniformly, retaining every image in each sampled group.

    The point estimate and replicate means weight images equally. Filename ENC
    groups are a correlation proxy; they do not attest patient independence.
    Blank groups are individual images. Pairing never depends on CSV row order.
    """
    if type(iterations) is not int or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    a, b = _indexed(left), _indexed(right)
    if set(a) != set(b):
        raise ValueError('Paired methods have different image IDs')
    grouped = defaultdict(list)
    for key in sorted(a):
        group = a[key].get('encounter') or ''
        if group != (b[key].get('encounter') or ''):
            raise ValueError('Paired image encounter mismatch')
        x, y = float(a[key][metric]), float(b[key][metric])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (x, y)):
            raise ValueError('Invalid paired metric value')
        grouped[('encounter', group) if group else ('image', key)].append(x-y)
    totals = np.array([sum(values) for values in grouped.values()], dtype=np.float64)
    counts = np.array([len(values) for values in grouped.values()], dtype=np.int64)
    generator = np.random.default_rng(seed)
    # Bounded chunks avoid allocating iterations x images for large datasets.
    replicates = []
    for start in range(0, iterations, 256):
        draws = generator.integers(len(counts), size=(min(256, iterations-start), len(counts)))
        replicates.extend((totals[draws].sum(axis=1)/counts[draws].sum(axis=1)).tolist())
    return {'delta': float(totals.sum()/counts.sum()),
            'ci95': np.quantile(replicates, [.025, .975]).tolist(),
            'images': len(a), 'clusters': len(grouped), 'iterations': iterations, 'seed': seed,
            'method': 'paired encounter-cluster percentile bootstrap; image-weighted means'}


def load_verified_evaluation(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    csv_path = directory/'per_image.csv'
    if summary.get('format') != 'cs406-completion-evaluation-v1':
        raise ValueError('Unsupported evaluation format')
    if sha256(csv_path) != summary.get('output_sha256', {}).get('per_image.csv'):
        raise ValueError('Evaluation CSV hash mismatch')
    if not summary.get('comparison_complete') or summary.get('failures'):
        raise ValueError('Incomplete evaluation cannot support paired comparisons')
    rows = defaultdict(list)
    with csv_path.open(encoding='utf-8', newline='') as stream:
        for source in csv.DictReader(stream):
            row = dict(source)
            for field in (*SCORES, 'seconds', 'target_fraction'):
                row[field] = float(row[field])
                if not math.isfinite(row[field]) or (field == 'seconds' and row[field] < 0):
                    raise ValueError(f'Invalid {field} in evaluation CSV')
                if field != 'seconds' and not 0 <= row[field] <= 1:
                    raise ValueError(f'Out of range {field} in evaluation CSV')
            for field in ('width', 'height', 'annotations'):
                row[field] = int(row[field])
                if row[field] < 1:
                    raise ValueError(f'Invalid {field} in evaluation CSV')
            rows[row['method']].append(row)
    if set(rows) != set(summary['methods']):
        raise ValueError('CSV and summary method IDs differ')
    ids = None
    for method, values in rows.items():
        index = _indexed(values)
        recorded = summary['methods'][method]
        if (len(index) != summary['available_images'] or recorded['sample_count'] != len(index)
                or set(recorded['evaluated_ids']) != set(index) or (ids is not None and set(index) != ids)):
            raise ValueError('Incomplete or inconsistent evaluation image IDs')
        ids = set(index)
        for field in ('dice', 'iou'):
            if not np.isclose(np.mean([row[field] for row in values]), recorded['macro']['mean'][field], atol=1e-10, rtol=0):
                raise ValueError('Summary macro scores disagree with CSV')
    return summary, dict(rows)


def _method_stats(rows):
    timings = np.array([row['seconds'] for row in rows])
    subgroups = defaultdict(list)
    for row in rows:
        area = 'small_0_to_5pct' if row['target_fraction'] <= .05 else ('medium_5_to_25pct' if row['target_fraction'] <= .25 else 'large_over_25pct')
        subgroups[area].append(row)
    return {'images': len(rows), 'macro': {field: float(np.mean([row[field] for row in rows])) for field in SCORES},
            'latency': {'mean_seconds': float(timings.mean()), 'median_seconds': float(np.median(timings)),
                        'p95_seconds': float(np.quantile(timings, .95)), 'first_call_seconds': float(timings[0]),
                        'subsequent_mean_seconds': float(timings[1:].mean()) if len(timings)>1 else None,
                        'protocol': 'Recorded evaluation calls; first includes lazy load, subsequent calls have no formal warm-up; not a dedicated benchmark'},
            'target_area_subgroups': {name: {'images': len(values), 'dice': float(np.mean([row['dice'] for row in values])),
                                           'iou': float(np.mean([row['iou'] for row in values]))} for name, values in subgroups.items()}}


def _write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(output, methods, primary, runs_dir, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(methods)
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(8, len(names)*1.2), 4.8))
    axis.bar(positions-.18, [methods[key]['macro']['dice'] for key in names], .36, label='Dice', color='#146b69')
    axis.bar(positions+.18, [methods[key]['macro']['iou'] for key in names], .36, label='IoU', color='#c8873b')
    axis.set_xticks(positions, names, rotation=25, ha='right', fontsize=8)
    axis.set_ylim(0, 1); axis.set_ylabel('Image macro score'); axis.legend()
    axis.set_title('Frozen model comparison'); figure.tight_layout()
    figure.savefig(output/'method_comparison.png', dpi=180); plt.close(figure)
    if runs_dir is None:
        return
    histories = []
    for method, recorded in summary['methods'].items():
        frozen = recorded.get('provenance', {}).get('frozen')
        if not frozen:
            continue
        path = Path(runs_dir)/method/'history.json'
        if not path.is_file() or sha256(path) != frozen['history_sha256']:
            raise ValueError(f'Missing or changed training history for plot: {method}')
        histories.append((method, json.loads(path.read_text(encoding='utf-8'))))
    if histories:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for name, history in histories:
            epochs = [row['epoch'] for row in history]
            axes[0].plot(epochs, [row['train_loss'] for row in history], label=name)
            axes[1].plot(epochs, [row['valid_dice'] for row in history], label=name)
        axes[0].set_ylabel('Training loss (different objectives)')
        axes[1].set_ylabel('Validation Dice at threshold 0.5')
        for axis in axes:
            axis.set_xlabel('Epoch'); axis.grid(alpha=.15)
        axes[1].legend(fontsize=7); figure.tight_layout()
        figure.savefig(output/'training_curves.png', dpi=180); plt.close(figure)


def analyze(evaluation_dir, output, primary=PRIMARY, iterations=2000, seed=406, plots=True, runs_dir=None):
    output, evaluation_dir = Path(output), Path(evaluation_dir)
    if output.exists():
        raise FileExistsError(f'Analysis output exists: {output}')
    summary, rows = load_verified_evaluation(evaluation_dir)
    if primary not in rows:
        raise ValueError('Predeclared primary method is absent')
    methods = {key: _method_stats(values) for key, values in rows.items()}
    comparisons = {f'{primary}-minus-{key}': {metric: paired_cluster_bootstrap(rows[primary], values, metric, iterations, seed)
                                            for metric in ('dice', 'iou', 'boundary_f1')}
                   for key, values in rows.items() if key != primary}
    ablation = {key: methods[key]['macro'] for key in ('bce_plain_s406', 'bce_dice_s406', 'bce_aug_s406', PRIMARY) if key in methods}
    seed_methods = [key for key in ('bce_dice_aug_s406', 'bce_dice_aug_s407', 'bce_dice_aug_s408') if key in methods]
    seeds = {'methods': seed_methods, 'runs': len(seed_methods),
             'note': 'Run-to-run variation on the same benchmark; not independent dataset replicates or patient-level confidence.'}
    if len(seed_methods)>1:
        seeds['scores'] = {metric: {'mean': float(np.mean([methods[key]['macro'][metric] for key in seed_methods])),
                                    'sample_std': float(np.std([methods[key]['macro'][metric] for key in seed_methods], ddof=1)),
                                    'values': [methods[key]['macro'][metric] for key in seed_methods]}
                           for metric in ('dice', 'iou', 'boundary_f1')}
    worst = sorted(rows[primary], key=lambda row: (row['dice'], row['id']))[:10]
    errors = [{key: row[key] for key in ('id', 'encounter', 'dice', 'iou', 'boundary_f1', 'target_fraction', 'width', 'height', 'annotations')} for row in worst]
    for error in errors:
        error['mask'] = f'masks/{primary}/{error["id"]}.png'
        error['interpretation'] = 'Ranked error candidate; requires image inspection before assigning a visual cause.'
    result = {'format': 'cs406-completion-analysis-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'primary_method': primary, 'model_selection_performed': False, 'methods': methods,
              'comparisons': comparisons, 'ablation_seed406': ablation, 'seed_variation': seeds,
              'worst_primary_cases': errors, 'source_dataset_fingerprint': summary['dataset_fingerprint'],
              'input_sha256': {'summary.json': sha256(evaluation_dir/'summary.json'), 'per_image.csv': sha256(evaluation_dir/'per_image.csv')},
              'protocol': {'point_estimate': 'Equal weight per image; paired differences on identical image IDs',
                           'uncertainty': '95% percentile CI, encounter-cluster resampling; blank ENC becomes one image per group',
                           'iterations': iterations, 'seed': seed, 'test_usage': 'Descriptive inference and error ranking only; no tuning',
                           'multiplicity': 'Several exploratory comparisons; no familywise correction and no confirmatory significance claim'},
              'limitations': ['Filename encounters are not verified patient IDs.',
                              'The local test was previously inspected; intervals do not remove selection or provenance bias.',
                              'Seed variation uses at most three runs and should not be read as population uncertainty.',
                              'External modality/annotation differences limit direct transfer interpretation.']}
    output.mkdir(parents=True)
    _write_csv(output/'method_summary.csv', [{'method': key, **value['macro'], **{name: value['latency'][name] for name in ('mean_seconds','median_seconds','p95_seconds')}} for key,value in methods.items()])
    _write_csv(output/'paired_comparisons.csv', [{'comparison': key, 'metric': metric, 'delta': value['delta'], 'ci95_low': value['ci95'][0], 'ci95_high': value['ci95'][1], 'images': value['images'], 'clusters': value['clusters']} for key, metrics in comparisons.items() for metric,value in metrics.items()])
    _write_csv(output/'worst_cases.csv', errors)
    if plots:
        _plots(output, methods, primary, runs_dir, summary)
    result['source_sha256'] = sha256(__file__)
    result['output_sha256'] = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--primary', default=PRIMARY)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=406)
    args = parser.parse_args(argv)
    result = analyze(args.evaluation_dir, args.output, args.primary, args.iterations, args.seed, runs_dir=args.runs_dir)
    print(json.dumps({'output': str(args.output.resolve()), 'methods': len(result['methods']), 'primary': result['primary_method']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
