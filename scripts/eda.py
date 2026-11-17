"""Read-only pixel EDA: python -m scripts.eda --data-dir data --output results/eda."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
from itertools import combinations
import json
from pathlib import Path
import platform
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, __version__ as pillow_version

from backend.metrics import segmentation_metrics
from scripts.audit_data import (
    PROJECT_ROOT, SPLIT_FOLDERS, MASK_EXTENSIONS,
    annotation_image_id, discover_samples, file_sha256,
)


def numeric_summary(values) -> dict:
    numbers = np.array([v for v in values if v is not None], dtype=np.float64)
    if not numbers.size:
        return {'count': 0, 'mean': None, 'std_population': None, 'min': None, 'p01': None,
                'p25': None, 'median': None, 'p75': None, 'p99': None, 'max': None}
    quantiles = np.percentile(numbers, [0, 1, 25, 50, 75, 99, 100])
    return {'count': int(numbers.size), 'mean': float(numbers.mean()),
            'std_population': float(numbers.std(ddof=0)),
            **dict(zip(('min', 'p01', 'p25', 'median', 'p75', 'p99', 'max'), map(float, quantiles)))}


def checked_path(path: Path, data_dir: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(data_dir):
        raise ValueError('Refusing dataset path outside the selected data root.')
    return resolved


def analyze_image(path: Path, data_dir: Path, thumbnail_size: int) -> dict:
    """Force complete pixel decoding before measuring an aspect-preserving thumbnail."""
    path = checked_path(path, data_dir)
    with Image.open(path) as source:
        source.load()
        width, height = source.size
        mode = source.mode
        image = source.convert('RGB')
        image.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rgb_mean = rgb.mean(axis=(0, 1))
    return {
        'width': width, 'height': height, 'pixels': width * height,
        'megapixels': width * height / 1_000_000, 'aspect_ratio': width / height,
        'original_mode': mode, 'thumbnail_width': image.width, 'thumbnail_height': image.height,
        'brightness_mean': float(gray.mean()), 'contrast_std': float(gray.std()),
        'sharpness_laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'red_mean': float(rgb_mean[0]), 'green_mean': float(rgb_mean[1]), 'blue_mean': float(rgb_mean[2]),
        'sha256': file_sha256(path),
    }


def analyze_mask(path: Path, data_dir: Path, thumbnail_size: int) -> tuple[dict, np.ndarray]:
    path = checked_path(path, data_dir)
    with Image.open(path) as source:
        source.load()
        array = np.array(source)
        mode = source.mode
    if array.ndim != 2:
        raise ValueError('Reference mask must have two dimensions.')
    values = np.unique(array)
    values_set = set(values.tolist())
    binary_encoding = values_set.issubset({0, 1}) or values_set.issubset({0, 255})
    foreground = array != 0
    foreground_pixels = int(np.count_nonzero(foreground))
    thumbnail = Image.fromarray(foreground.astype(np.uint8)).resize(
        (thumbnail_size, thumbnail_size), Image.Resampling.NEAREST)
    return {
        'width': int(array.shape[1]), 'height': int(array.shape[0]), 'pixels': int(array.size),
        'original_mode': mode, 'dtype': str(array.dtype),
        'unique_values': '|'.join(str(value) for value in values.tolist()),
        'binary_encoding': binary_encoding, 'foreground_pixels': foreground_pixels,
        'foreground_fraction': foreground_pixels / array.size,
        'empty': foreground_pixels == 0, 'full': foreground_pixels == array.size,
        'nonfinite_pixels': int(np.count_nonzero(~np.isfinite(array))),
    }, np.asarray(thumbnail, dtype=bool)


def collect_eda(data_dir: Path, thumbnail_size: int = 256, progress: bool = False) -> dict:
    """Decode every indexed image and every mask, including orphan masks.

    No images, labels, splits, or model settings are changed. Native mask
    coverage is exact. Inter-annotation Dice is explicitly measured on square
    nearest-neighbor thumbnails to bound memory and computation.
    """
    if thumbnail_size < 2:
        raise ValueError('Thumbnail size must be at least 2.')
    data_dir = Path(data_dir).resolve()
    started = time.perf_counter()
    image_rows, annotation_rows, subject_rows, errors = [], [], [], []
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'data_root': str(data_dir),
        'protocol': {
            'selection': 'All files indexed by supported image/mask extensions in all three split folders; no sample limit.',
            'pixel_integrity': 'Pillow load() forces complete pixel decode; header-only validation is insufficient.',
            'foreground': 'Every nonzero native mask pixel; report unexpected encoding separately without repairing it.',
            'valid_binary_encoding': 'Unique values subset of {0,1} or subset of {0,255}.',
            'coverage': 'Exact native-resolution foreground pixel count / native mask pixels.',
            'quality_thumbnail_max_side': thumbnail_size,
            'quality_resize': 'RGB thumbnail, bilinear, original aspect preserved, no upscaling.',
            'brightness': 'Mean OpenCV RGB-to-gray intensity, scale 0..255.',
            'contrast': 'Population standard deviation of grayscale intensity.',
            'sharpness': 'Variance of 3x3-support default OpenCV Laplacian (ksize=1), CV_64F; descriptive proxy, no blur cutoff.',
            'annotation_agreement_size': [thumbnail_size, thumbnail_size],
            'annotation_agreement': 'Mean/min of all unordered annotation-pair foreground Dice after nearest-neighbor square resize.',
            'consensus': 'Strict pixelwise majority; tied pixels are background.',
            'aggregation': 'Per-image statistics weight each image equally; mask statistics weight each annotation equally.',
            'subject_rule': 'Filename prefix IMG_ENC followed by digits before final image suffix; identity unverified.',
            'duplicates': 'SHA-256 of every successfully decoded original image file; exact bytes only.',
            'data_mutations': 'None: no deletion, deduplication, relabeling, augmentation, normalization or split changes.',
        },
        'splits': {},
        'limitations': [
            'Dataset source, license, target diagnosis and annotation process were not supplied.',
            'Filename subjects do not establish verified patient identities.',
            'Metadata disjointness and exact hashes do not rule out re-encoded near duplicates or pretrained test exposure.',
            'EDA is descriptive. No predictive threshold or preprocessing was tuned from test EDA.',
            'Thumbnail brightness/contrast/sharpness are image-level proxies influenced by content, resizing and lighting.',
            'Agreement on 256-square thumbnails (or configured size) is not native-resolution boundary agreement.',
            'Foreground coverage cannot identify disease prevalence or clinical severity.',
        ],
    }
    stat_names = ('width', 'height', 'megapixels', 'aspect_ratio', 'brightness_mean', 'contrast_std',
                  'sharpness_laplacian_variance', 'annotation_coverage_mean', 'annotation_coverage_range',
                  'consensus_fraction_thumbnail', 'pairwise_dice_thumbnail_mean')
    all_hashes = defaultdict(list)
    for split, suffix in SPLIT_FOLDERS.items():
        samples = discover_samples(data_dir, split)
        masks_by_image = defaultdict(list)
        masks_by_thumbnail = defaultdict(list)
        split_images, split_annotations = [], []
        label_dir = data_dir / f'labels{suffix}'
        all_masks = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS) if label_dir.is_dir() else []
        image_ids = {sample['id'] for sample in samples}
        # Every mask is decoded independently, even if the corresponding image is corrupt or absent.
        for index, path in enumerate(all_masks, 1):
            image_id = annotation_image_id(path)
            row = {'split': split, 'id': image_id, 'annotation': path.name,
                   'path': str(path.relative_to(data_dir)), 'orphan': image_id not in image_ids,
                   'status': 'ok', 'error': ''}
            try:
                statistics, thumbnail = analyze_mask(path, data_dir, thumbnail_size)
                row.update(statistics)
                masks_by_image[image_id].append(row)
                masks_by_thumbnail[image_id].append(thumbnail)
            except (OSError, ValueError, RuntimeError) as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                errors.append({'kind': 'mask', 'split': split, 'file': row['path'], 'error': row['error']})
            split_annotations.append(row)
            if progress and (index % 250 == 0 or index == len(all_masks)):
                print(f'{split}: decoded masks {index}/{len(all_masks)}', flush=True)
        subject_counter = Counter()
        for index, sample in enumerate(samples, 1):
            match = re.match(r'^(IMG_ENC\d+)_\d+$', sample['id'])
            subject = match.group(1) if match else ''
            if subject:
                subject_counter[subject] += 1
            row = {'split': split, 'id': sample['id'], 'path': str(sample['image'].relative_to(data_dir)),
                   'subject': subject, 'annotations_discovered': len(sample['masks']),
                   'annotations_decoded': len(masks_by_image[sample['id']]), 'status': 'ok', 'error': ''}
            try:
                row.update(analyze_image(sample['image'], data_dir, thumbnail_size))
                all_hashes[row['sha256']].append({'split': split, 'id': sample['id'], 'path': row['path']})
            except (OSError, ValueError, RuntimeError) as exc:
                row.update(status='error', error=f'{type(exc).__name__}: {exc}')
                errors.append({'kind': 'image', 'split': split, 'file': row['path'], 'error': row['error']})
            labels = masks_by_image[sample['id']]
            for label in labels:
                label['dimension_matches_image'] = (
                    label['width'] == row['width'] and label['height'] == row['height']
                    if row['status'] == 'ok' else None)
            if labels:
                coverages = [label['foreground_fraction'] for label in labels]
                row.update(annotation_coverage_mean=float(np.mean(coverages)),
                           annotation_coverage_min=min(coverages), annotation_coverage_max=max(coverages),
                           annotation_coverage_range=max(coverages) - min(coverages))
                thumbnails = masks_by_thumbnail[sample['id']]
                stack = np.stack(thumbnails)
                row['consensus_fraction_thumbnail'] = float((stack.sum(axis=0) > len(thumbnails) / 2).mean())
                if len(thumbnails) > 1:
                    pair_dice = [segmentation_metrics(a, b)['dice'] for a, b in combinations(thumbnails, 2)]
                    row.update(annotation_pairs=len(pair_dice), pairwise_dice_thumbnail_mean=float(np.mean(pair_dice)),
                               pairwise_dice_thumbnail_min=min(pair_dice))
            split_images.append(row)
            if progress and (index % 250 == 0 or index == len(samples)):
                print(f'{split}: decoded images {index}/{len(samples)}', flush=True)
        good_images = [row for row in split_images if row['status'] == 'ok']
        good_masks = [row for row in split_annotations if row['status'] == 'ok']
        pixel_total = sum(row['pixels'] for row in good_masks)
        summary['splits'][split] = {
            'images': len(split_images), 'images_decoded': len(good_images),
            'corrupt_images': len(split_images) - len(good_images),
            'masks': len(split_annotations), 'masks_decoded': len(good_masks),
            'corrupt_masks': len(split_annotations) - len(good_masks),
            'subjects': len(subject_counter),
            'images_per_subject': numeric_summary(subject_counter.values()),
            'subject_image_counts': dict(sorted(subject_counter.items())),
            'annotations_per_image': dict(sorted(Counter(row['annotations_discovered'] for row in split_images).items())),
            'image_modes': dict(Counter(row['original_mode'] for row in good_images)),
            'mask_modes': dict(Counter(row['original_mode'] for row in good_masks)),
            'mask_value_sets': dict(Counter(row['unique_values'] for row in good_masks)),
            'nonbinary_masks': sum(not row['binary_encoding'] for row in good_masks),
            'nonfinite_mask_pixels': sum(row['nonfinite_pixels'] for row in good_masks),
            'empty_masks': sum(row['empty'] for row in good_masks),
            'full_masks': sum(row['full'] for row in good_masks),
            'orphan_masks': sum(row['orphan'] for row in split_annotations),
            'missing_mask_images': sum(row['annotations_discovered'] == 0 for row in split_images),
            'mask_dimension_mismatches': sum(row.get('dimension_matches_image') is False for row in good_masks if not row['orphan']),
            'foreground_fraction_per_annotation': numeric_summary(row['foreground_fraction'] for row in good_masks),
            'foreground_fraction_pooled_pixels': sum(row['foreground_pixels'] for row in good_masks) / pixel_total if pixel_total else None,
            'native_mask_pixels': pixel_total,
            'native_image_pixels': sum(row['pixels'] for row in good_images),
            'images_with_multiple_annotations': sum(row['annotations_decoded'] > 1 for row in split_images),
            'images_with_pairwise_dice_below_0_5': sum(row.get('pairwise_dice_thumbnail_mean', 1) < 0.5 for row in split_images),
            'statistics': {name: numeric_summary(row.get(name) for row in good_images) for name in stat_names},
        }
        image_rows.extend(split_images)
        annotation_rows.extend(split_annotations)
        subject_rows.extend({'split': split, 'subject': subject, 'image_count': count} for subject, count in sorted(subject_counter.items()))
    duplicate_groups = [{'sha256': digest, 'files': files} for digest, files in all_hashes.items() if len(files) > 1]
    within_groups = [group for group in duplicate_groups if len({f['split'] for f in group['files']}) == 1]
    cross_groups = [group for group in duplicate_groups if len({f['split'] for f in group['files']}) > 1]
    summary['duplicates'] = {
        'within_split_group_count': len(within_groups), 'cross_split_group_count': len(cross_groups),
        'within_split_redundant_files': sum(len(g['files']) - 1 for g in within_groups),
        'within_split_groups_by_split': dict(Counter(g['files'][0]['split'] for g in within_groups)),
        'groups': duplicate_groups,
        'recommendation': 'Review duplicate groups and subject correlations before retraining; retain one copy or control sampling weights inside training only. No files were removed.',
    }
    summary['totals'] = {name: sum(report[name] for report in summary['splits'].values()) for name in (
        'images', 'images_decoded', 'corrupt_images', 'masks', 'masks_decoded', 'corrupt_masks',
        'nonbinary_masks', 'empty_masks', 'full_masks', 'native_mask_pixels', 'native_image_pixels')}
    summary['errors'] = errors
    summary['wall_seconds'] = time.perf_counter() - started
    summary['environment'] = {'python': platform.python_version(), 'numpy': np.__version__,
                              'opencv': cv2.__version__, 'pillow': pillow_version}
    summary['source_sha256'] = {'scripts/eda.py': file_sha256(Path(__file__)),
                                'scripts/audit_data.py': file_sha256(PROJECT_ROOT / 'scripts/audit_data.py')}
    return {'summary': summary, 'per_image': image_rows, 'per_annotation': annotation_rows, 'per_subject': subject_rows}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['split', 'id'])
        writer.writeheader()
        writer.writerows(rows)


def create_overview(result: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'train': '#147d92', 'valid': '#d5982e', 'test': '#9a5b9a'}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    specs = [('megapixels', 'Original image size (megapixels)'), ('aspect_ratio', 'Aspect ratio (width / height)'),
             ('annotation_coverage_mean', 'Native foreground fraction (mean annotation)'),
             ('brightness_mean', 'Thumbnail brightness (0-255)'), ('contrast_std', 'Thumbnail grayscale contrast'),
             ('sharpness_laplacian_variance', 'Thumbnail Laplacian variance (log scale)')]
    for axis, (statistic, title) in zip(axes.flat, specs):
        for split, color in colors.items():
            values = [row[statistic] for row in result['per_image'] if row['split'] == split and row['status'] == 'ok' and statistic in row]
            if statistic == 'sharpness_laplacian_variance':
                values = np.log10(np.array(values) + 1)
            if values.size if isinstance(values, np.ndarray) else len(values):
                axis.hist(values, bins=24, density=True, histtype='step', linewidth=1.8, label=split, color=color)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel('Density')
        axis.grid(axis='y', alpha=0.2)
        if statistic == 'sharpness_laplacian_variance':
            axis.set_xlabel('log10(variance + 1)')
    axes.flat[0].legend(frameon=False)
    figure.suptitle('Dataset EDA: all decoded images; descriptive split distributions', fontsize=14)
    figure.savefig(output / 'overview.png', dpi=150)
    plt.close(figure)


def create_outliers(result: dict, data_dir: Path, output: Path) -> list[dict]:
    good = [row for row in result['per_image'] if row['status'] == 'ok']
    selections, seen = [], set()
    criteria = [('Smallest original image', 'pixels', False), ('Lowest thumbnail contrast', 'contrast_std', False),
                ('Lowest thumbnail sharpness proxy', 'sharpness_laplacian_variance', False),
                ('Darkest thumbnail', 'brightness_mean', False), ('Largest annotation coverage range', 'annotation_coverage_range', True),
                ('Lowest annotation agreement', 'pairwise_dice_thumbnail_mean', False),
                ('Smallest foreground fraction', 'annotation_coverage_mean', False),
                ('Largest foreground fraction', 'annotation_coverage_mean', True)]
    for label, key, reverse in criteria:
        candidates = sorted((row for row in good if key in row), key=lambda row: (row[key], row['split'], row['id']), reverse=reverse)
        for row in candidates:
            identity = (row['split'], row['id'])
            if identity not in seen:
                seen.add(identity)
                selections.append({'criterion': label, 'statistic': key, 'value': row[key], 'split': row['split'], 'id': row['id'], 'path': row['path']})
                break
    empty = [row for row in result['per_annotation'] if row.get('empty')]
    for row in empty[:2]:
        selections.append({'criterion': 'Empty native reference mask', 'statistic': 'foreground_fraction', 'value': 0,
                           'split': row['split'], 'id': row['id'], 'annotation': row['annotation'], 'path': row['path']})
    if not selections:
        return selections
    columns, cell_width, cell_height = 4, 320, 292
    sheet = Image.new('RGB', (columns * cell_width, ((len(selections) + columns - 1) // columns) * cell_height), '#f4f4f4')
    draw = ImageDraw.Draw(sheet)
    for index, selection in enumerate(selections):
        x, y = index % columns * cell_width, index // columns * cell_height
        path = checked_path(data_dir / selection['path'], data_dir)
        with Image.open(path) as source:
            im = source.convert('RGB')
            fitted = ImageOps.contain(im, (cell_width - 16, 220), Image.Resampling.BILINEAR)
        sheet.paste(fitted, (x + (cell_width - fitted.width) // 2, y + 4))
        draw.text((x + 8, y + 228), selection['criterion'], fill='#111111')
        draw.text((x + 8, y + 245), f"{selection['split']} / {selection['id']}", fill='#222222')
        draw.text((x + 8, y + 262), f"value={selection['value']:.6g}", fill='#333333')
    sheet.save(output / 'outliers.png')
    return selections


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/eda')
    parser.add_argument('--thumbnail-size', type=int, default=256)
    args = parser.parse_args(argv)
    args.data_dir, args.output = args.data_dir.resolve(), args.output.resolve()
    if args.thumbnail_size < 2:
        parser.error('--thumbnail-size must be at least 2')
    if args.output == args.data_dir or args.output.is_relative_to(args.data_dir):
        parser.error('--output must be outside the raw dataset directory')
    if not args.data_dir.is_dir():
        parser.error('--data-dir must be an existing directory')
    result = collect_eda(args.data_dir, args.thumbnail_size, progress=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ('per_image', 'per_annotation', 'per_subject'):
        write_csv(args.output / f'{key}.csv', result[key])
    create_overview(result, args.output)
    outliers = create_outliers(result, args.data_dir, args.output)
    (args.output / 'outliers.json').write_text(json.dumps(outliers, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (args.output / 'summary.json').write_text(json.dumps(result['summary'], indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'totals': result['summary']['totals'],
                      'duplicate_groups': result['summary']['duplicates']['within_split_group_count'],
                      'wall_seconds': result['summary']['wall_seconds']}, indent=2))
    return 1 if result['summary']['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
