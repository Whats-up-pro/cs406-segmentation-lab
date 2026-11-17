"""Descriptive paired analysis of SAVED local results; no fitting or test tuning.

Run ``python -m scripts.analyze_results --with-cases`` from Project. Aggregate
analysis reads only CSVs. --with-cases reruns only six explicitly selected
images at the original 256-square evaluation settings to export real overlays.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np

from scripts.audit_data import PROJECT_ROOT, file_sha256


METHODS = ('baseline', 'transunet')
METRICS = ('dice', 'iou', 'pixel_accuracy', 'precision', 'recall')
COLORS = {'baseline': '#64748b', 'transunet': '#0f766e'}
LABELS = {'baseline': 'Baseline', 'transunet': 'TransUNet'}
STRATA = ('[0, 0.05)', '[0.05, 0.20)', '[0.20, 0.50)', '[0.50, 1.00]')
CASE_SELECTION = (
    ('IMG_ENC00764_00002', 'largest paired Dice gain'),
    ('IMG_ENC00238_00003', 'highest TransUNet Dice'),
    ('IMG_ENC00192_00005', 'largest baseline advantage / undercoverage'),
    ('IMG_ENC00084_00003', 'zero TransUNet overlap'),
    ('IMG_ENC00210_00001', 'low Dice with overcoverage'),
    ('IMG_ENC00028_00003', 'lowest annotation agreement'),
)


def foreground_stratum(fraction: float) -> str:
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Foreground fraction must be in [0, 1].')
    return STRATA[int(np.searchsorted([.05, .2, .5], fraction, side='right'))]


def agreement_stratum(value: float | None) -> str:
    return 'unavailable' if value is None else ('high (>=0.50)' if value >= .5 else 'low (<0.50)')


def means(rows: list[dict]) -> dict:
    return {f'{method}_{metric}': float(np.mean([r[f'{method}_{metric}'] for r in rows]))
            for method in METHODS for metric in METRICS if rows and f'{method}_{metric}' in rows[0]}


def paired_cluster_bootstrap(rows: list[dict], n_resamples: int = 2000, seed: int = 20261114) -> dict:
    """Resample entire filename groups; retain method pairs in each draw.

    Image estimand: sum of sampled group score sums / sampled image count.
    Group estimand: mean of sampled within-group means. Unequal group sizes
    make these different estimands, even though both resample the same groups.
    """
    if not rows or n_resamples < 2:
        raise ValueError('Bootstrap needs paired rows and at least two resamples.')
    groups = defaultdict(list)
    for row in rows:
        pair = [row['baseline_dice'], row['transunet_dice']]
        if not row['subject'] or not np.isfinite(pair).all():
            raise ValueError('Bootstrap rows need a group and finite paired Dice.')
        groups[row['subject']].append(pair)
    arrays = [np.array(groups[group], dtype=np.float64) for group in sorted(groups)]
    counts = np.array([len(a) for a in arrays], dtype=np.int64)
    sums = np.array([a.sum(axis=0) for a in arrays])
    group_means = sums / counts[:, None]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(arrays), size=(n_resamples, len(arrays)))
    estimates = {
        'image_macro': (sums.sum(axis=0) / counts.sum(), sums[draws].sum(axis=1) / counts[draws].sum(axis=1)[:, None]),
        'filename_subject_macro': (group_means.mean(axis=0), group_means[draws].mean(axis=1)),
    }
    result = {'groups': len(arrays), 'resamples': n_resamples, 'seed': seed,
              'interval': 'percentile 2.5th and 97.5th; paired whole-group resampling'}
    for name, (estimate, samples) in estimates.items():
        result[name] = {}
        for i, method in enumerate((*METHODS, 'difference')):
            point = estimate[i] if i < 2 else estimate[1] - estimate[0]
            sample = samples[:, i] if i < 2 else samples[:, 1] - samples[:, 0]
            result[name][method] = {'estimate': float(point), 'ci95': np.percentile(sample, [2.5, 97.5]).tolist()}
    return result


def duplicate_sensitivity(rows: list[dict]) -> dict:
    """Give each image hash equal weight, averaging all annotation versions.

    Byte-identical photographs need not have identical annotation masks. The
    hash-balanced estimate retains all masks in each hash's mean; selecting
    lexicographically first is a separate, explicitly arbitrary sensitivity.
    """
    groups = defaultdict(list)
    for row in rows:
        if not row['sha256']:
            raise ValueError('Missing image SHA-256 prevents duplicate analysis.')
        groups[row['sha256']].append(row)
    metric_keys = [key for key in rows[0] if any(key == f'{m}_{n}' for m in METHODS for n in METRICS)]
    hash_rows, first_rows, duplicates = [], [], []
    for digest, values in sorted(groups.items()):
        values = sorted(values, key=lambda r: r['id'])
        hash_rows.append({k: float(np.mean([r[k] for r in values])) for k in metric_keys})
        first_rows.append(values[0])
        if len(values) > 1:
            duplicates.append({'sha256': digest, 'files': len(values), 'ids': [r['id'] for r in values],
                               'target_fraction_range': max(r['target_fraction'] for r in values) - min(r['target_fraction'] for r in values),
                               **hash_rows[-1]})
    return {'unique_hashes': len(groups), 'duplicate_group_count': len(duplicates),
            'redundant_files': len(rows) - len(groups), 'duplicate_groups': duplicates,
            'all_images': means(rows), 'hash_balanced': means(hash_rows),
            'lexicographic_first_per_hash': means(first_rows),
            'policy': 'Primary sensitivity averages all annotation-version scores within image SHA-256, then weights hashes equally; no labels or files removed.'}


def spearman(x, y) -> float | None:
    """Pearson correlation of average tied ranks; no inferential p value."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None

    def ranks(a):
        values, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    a, b = ranks(x), ranks(y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: '|'.join(v) if isinstance(v, list) else v for k, v in r.items()} for r in rows)


def load_paired(comparison: Path, eda: Path) -> list[dict]:
    metadata = {r['id']: r for r in read_csv(eda / 'per_image.csv') if r['split'] == 'test'}
    by_id = defaultdict(dict)
    for row in read_csv(comparison / 'per_image.csv'):
        if row['split'] != 'test' or row['method'] not in METHODS:
            continue
        if row['method'] in by_id[row['id']]:
            raise ValueError('Duplicate method/id metric row.')
        by_id[row['id']][row['method']] = row
    paired = []
    for image_id, methods in sorted(by_id.items()):
        if set(methods) != set(METHODS) or image_id not in metadata:
            raise ValueError(f'Missing method or EDA row: {image_id}')
        meta = metadata[image_id]
        subject = re.fullmatch(r'(IMG_ENC\d+)_\d+', image_id)
        if subject is None or subject.group(1) != meta['subject']:
            raise ValueError(f'Unrecognized filename group: {image_id}')
        base, trans = methods['baseline'], methods['transunet']
        for key in ('target_fraction', 'evaluation_width', 'evaluation_height', 'annotations'):
            if base[key] != trans[key]:
                raise ValueError(f'Methods do not share reference/protocol: {image_id}')
        row = {'id': image_id, 'subject': subject.group(1), 'sha256': meta['sha256'],
               'target_fraction': float(base['target_fraction']), 'annotations': int(base['annotations']),
               'evaluation_width': int(base['evaluation_width']), 'evaluation_height': int(base['evaluation_height'])}
        for method in METHODS:
            for metric in (*METRICS, 'prediction_fraction', 'seconds'):
                row[f'{method}_{metric}'] = float(methods[method][metric])
        for feature in ('brightness_mean', 'contrast_std', 'sharpness_laplacian_variance', 'aspect_ratio', 'megapixels', 'pairwise_dice_thumbnail_mean', 'annotation_coverage_range'):
            row[feature] = float(meta[feature]) if meta[feature] else None
        row['dice_difference'] = row['transunet_dice'] - row['baseline_dice']
        row['foreground_stratum'] = foreground_stratum(row['target_fraction'])
        row['agreement_stratum'] = agreement_stratum(row['pairwise_dice_thumbnail_mean'])
        paired.append(row)
    if not paired:
        raise ValueError('No paired test images.')
    return paired


def stratify(rows, key, labels):
    result = []
    for label in labels:
        subset = [r for r in rows if r[key] == label]
        result.append({'stratum': label, 'images': len(subset), 'filename_groups': len({r['subject'] for r in subset}),
                       'baseline_dice': float(np.mean([r['baseline_dice'] for r in subset])) if subset else None,
                       'transunet_dice': float(np.mean([r['transunet_dice'] for r in subset])) if subset else None,
                       'dice_difference': float(np.mean([r['dice_difference'] for r in subset])) if subset else None})
    return result


def save_figures(output: Path, rows: list[dict], report: dict) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    figures = {}
    source = 'Source: saved comparison_test/per_image.csv + eda/per_image.csv; test n=295. Descriptive local checkpoint analysis.'

    def save(fig, name, caption):
        fig.text(.02, .015, source, ha='left', va='bottom', fontsize=8, color='#475569')
        fig.savefig(output / name, dpi=180, bbox_inches='tight')
        plt.close(fig)
        figures[name] = caption

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=False)
    a, b = np.array([[r['baseline_dice'], r['transunet_dice']] for r in rows]).T
    axes[0].scatter(a, b, s=21, alpha=.63, color=COLORS['transunet'], edgecolors='none')
    axes[0].plot([0, 1], [0, 1], '--', color='#94a3b8', lw=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Baseline Dice', ylabel='TransUNet Dice', title='Paired image scores')
    axes[1].hist(b - a, bins=np.linspace(-1, 1, 26), color=COLORS['transunet'], edgecolor='white')
    axes[1].axvline(0, color='#334155', lw=1)
    axes[1].axvline((b - a).mean(), color='#b45309', lw=2, label=f'Mean difference {(b-a).mean():+.3f}')
    axes[1].set(xlabel='TransUNet Dice - baseline Dice', ylabel='Images', title='Distribution of paired differences')
    axes[1].legend(frameon=False)
    fig.subplots_adjust(bottom=.19, wspace=.3)
    save(fig, 'paired_dice.png', 'Each point is the same image evaluated by both methods; diagonal is equal Dice. Histogram retains negative differences and does not imply universal improvement.')

    fig, ax = plt.subplots(figsize=(9.3, 4.8))
    specs = [('Image macro', 'image_macro'), ('Filename-group macro', 'filename_subject_macro')]
    positions = np.arange(2)
    for index, method in enumerate(METHODS):
        values = [report['cluster_bootstrap'][key][method] for _, key in specs]
        points = np.array([v['estimate'] for v in values])
        intervals = np.array([v['ci95'] for v in values]).T
        ax.bar(positions + (index - .5) * .32, points, width=.3, color=COLORS[method], label=LABELS[method])
        ax.errorbar(positions + (index - .5) * .32, points, yerr=[points - intervals[0], intervals[1] - points], fmt='none', color='#0f172a', capsize=4, lw=1)
        for position, point in zip(positions + (index - .5) * .32, points):
            ax.text(position, point + .065, f'{point:.3f}', ha='center', fontsize=10)
    ax.set(xticks=positions, xticklabels=[s[0] for s in specs], ylim=(0, 1), ylabel='Mean foreground Dice', title='Averaging choice and paired group-bootstrap 95% intervals')
    ax.legend(frameon=False, loc='upper left')
    fig.subplots_adjust(bottom=.17)
    save(fig, 'macro_comparison.png', '2000 seeded whole-filename-group bootstrap draws. Filename groups (103) are proxy clusters, not verified patients. Intervals describe this saved checkpoint comparison; checkpoint test exposure is unknown.')

    for key, filename, title in [('foreground_strata', 'foreground_strata.png', 'Dice by consensus foreground fraction'),
                                  ('agreement_strata', 'agreement_strata.png', 'Dice by annotation agreement')]:
        strata = [s for s in report[key] if s['images']]
        fig, ax = plt.subplots(figsize=(9.3, 4.8))
        positions = np.arange(len(strata))
        for index, method in enumerate(METHODS):
            values = [s[f'{method}_dice'] if s['images'] else 0 for s in strata]
            ax.bar(positions + (index - .5) * .34, values, width=.32, color=COLORS[method], label=LABELS[method])
        ax.set(xticks=positions, xticklabels=[f"{s['stratum']}\nn={s['images']}" for s in strata], ylim=(0, 1), ylabel='Image-macro Dice', title=title)
        ax.legend(frameon=False, loc='upper left')
        fig.subplots_adjust(bottom=.22)
        save(fig, filename, 'Fixed descriptive strata; no tuning. ' + ('Bins are [0,.05), [.05,.20), [.20,.50), [.50,1]. Foreground area is not clinical severity.' if key == 'foreground_strata' else 'Agreement is mean pairwise annotation Dice on 256-square nearest-neighbor masks; high >=0.50, low <0.50. Associations are not causal.'))
    (output / 'figure_captions.json').write_text(json.dumps(figures, indent=2) + '\n', encoding='utf-8')
    return figures


def export_cases(data_dir: Path, output: Path, rows: list[dict]) -> list[dict]:
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from backend.baseline import baseline_probability
    from backend.inference import predict_transunet
    from backend.metrics import segmentation_metrics
    from scripts.audit_data import discover_samples
    from scripts.evaluate import load_reference

    cases_dir = output / 'cases'
    cases_dir.mkdir(exist_ok=True)
    indexed = {s['id']: s for s in discover_samples(data_dir, 'test')}
    saved = {r['id']: r for r in rows}
    results = []
    for image_id, reason in CASE_SELECTION:
        sample, reference_row = indexed[image_id], saved[image_id]
        if (reference_row['evaluation_width'], reference_row['evaluation_height']) != (256, 256):
            raise ValueError('Case export expects original 256-square evaluation.')
        with Image.open(sample['image']) as original:
            image = original.convert('RGB').resize((256, 256), Image.Resampling.BILINEAR)
        target, annotations = load_reference(sample, (256, 256))
        predictions = {'baseline': baseline_probability(image) >= .5,
                       'transunet': predict_transunet(image, normalization='unit') >= .5}
        result = {'id': image_id, 'selection': reason, 'annotation_agreement': reference_row['pairwise_dice_thumbnail_mean'],
                  'target_pixels': int(target.sum()), 'target_fraction': float(target.mean()),
                  'figure': f'cases/{image_id}.png', 'source_photo': f'cases/{image_id}_source.png'}
        image.save(cases_dir / f'{image_id}_source.png')
        Image.fromarray(target.astype('uint8') * 255).save(cases_dir / f'{image_id}_consensus.png')
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 8.8))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Source photo (256 square)')
        axes[0, 1].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Strict-majority mask\nforeground {target.mean():.1%}')
        agreement_map = np.stack(annotations).mean(axis=0)
        votes = axes[0, 2].imshow(agreement_map, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(votes, ax=axes[0, 2], fraction=.046, pad=.03, ticks=[0, .5, 1])
        axes[0, 2].set_title(f'Annotation vote fraction\nmean pairwise Dice {result["annotation_agreement"]:.3f}')
        for index, method in enumerate(METHODS):
            prediction = predictions[method]
            metrics = segmentation_metrics(prediction, target)
            if any(abs(metrics[m] - reference_row[f'{method}_{m}']) > 1e-6 for m in METRICS):
                raise ValueError(f'Recomputed selected case differs from saved metrics: {image_id}/{method}')
            tp, fp, fn, tn = [int(np.count_nonzero(value)) for value in
                              (prediction & target, prediction & ~target, ~prediction & target, ~prediction & ~target)]
            result.update({f'{method}_{m}': value for m, value in metrics.items()})
            result.update({f'{method}_tp': tp, f'{method}_fp': fp, f'{method}_fn': fn, f'{method}_tn': tn,
                           f'{method}_prediction_fraction': float(prediction.mean())})
            Image.fromarray(prediction.astype('uint8') * 255).save(cases_dir / f'{image_id}_{method}.png')
            overlay = np.asarray(image).astype(float) / 255
            for mask, color in [(prediction & target, (.05, .65, .35)), (prediction & ~target, (.95, .3, .1)), (~prediction & target, (.3, .35, .95))]:
                overlay[mask] = .35 * overlay[mask] + .65 * np.array(color)
            axes[1, index].imshow(overlay)
            axes[1, index].set_title(f'{LABELS[method]}: Dice {metrics["dice"]:.3f}, IoU {metrics["iou"]:.3f}\nFP {fp:,} | FN {fn:,}', fontsize=11)
        axes[1, 2].axis('off')
        axes[1, 2].legend(handles=[Patch(facecolor='#0da659', label='True positive'), Patch(facecolor='#f24d1a', label='False positive'), Patch(facecolor='#4d59f2', label='False negative')], loc='upper left', frameon=False)
        axes[1, 2].text(0, .55, 'Errors are relative to the\nstrict-majority reference.\n\nSelected post hoc for explanation;\nnot a representative sample.\n\nInference: 256 RGB -> 224 model\n-> 256 probabilities; threshold .5.', va='top', fontsize=10)
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f'{image_id}\n{reason}', fontsize=14, y=.985)
        fig.text(.03, .015, 'Source: supplied local test photo, TIFF annotations, and selected-case inference; actual masks, no synthetic imagery.', fontsize=8, color='#475569')
        fig.subplots_adjust(left=.035, right=.965, bottom=.08, top=.84, wspace=.20, hspace=.40)
        fig.savefig(cases_dir / f'{image_id}.png', dpi=170, bbox_inches='tight')
        plt.close(fig)
        results.append(result)
        print(f'Exported selected case {image_id}', flush=True)
    return results


def write_narrative(path: Path, report: dict) -> None:
    b = report['cluster_bootstrap']
    d = report['duplicate_sensitivity']
    lines = [
        '# Phân tích sai số và độ nhạy của kết quả', '',
        'Phân tích mô tả từ các bảng CSV đã lưu, không huấn luyện lại, không chọn tham số từ tập test. '
        'Mô hình là checkpoint được cung cấp; lịch sử huấn luyện và khả năng đã tiếp xúc ảnh test chưa được xác minh. '
        'Các số liệu dưới đây không chứng minh hiệu quả lâm sàng hay khả năng khái quát trên dữ liệu độc lập.', '',
        '## Giao thức và đơn vị tổng hợp', '',
        '295 ảnh test, 103 nhóm tên tệp `IMG_ENCxxxxx`; các nhóm chỉ là biến đại diện theo tên, chưa xác minh là bệnh nhân. '
        'Ảnh RGB đổi kích thước bilinear thành 256×256, các mask nhị phân đổi kích thước nearest-neighbor; '
        'tham chiếu là đa số nghiêm ngặt của tất cả annotation (hòa phiếu là nền). '
        'TransUNet dùng normalization `unit`, đầu vào mô hình 224×224 rồi nội suy xác suất về 256×256; ngưỡng 0,5. '
        'Baseline là quy tắc Lab-a/Otsu và hình thái học cố định. Kích thước vuông có thể làm thay đổi tỉ lệ hình học.', '',
        '| Tổng hợp Dice | Baseline | TransUNet | Chênh lệch |', '|---|---:|---:|---:|',
    ]
    for label, key in [('Mỗi ảnh trọng số bằng nhau', 'image_macro'), ('Mỗi nhóm tên tệp trọng số bằng nhau', 'filename_subject_macro')]:
        v = b[key]
        lines.append(f'| {label} | {v["baseline"]["estimate"]:.4f} | {v["transunet"]["estimate"]:.4f} | {v["difference"]["estimate"]:+.4f} |')
    ci = b['image_macro']['difference']['ci95']
    wins = report['paired_dice']
    lines += ['', f'TransUNet có Dice cao hơn trên {wins["transunet_wins"]}/295 ảnh, hòa {wins["ties"]}, thấp hơn {wins["baseline_wins"]}. '
              f'Chênh lệch trung bình {wins["mean_difference"]:+.4f}; khoảng bootstrap 95% [{ci[0]:.4f}, {ci[1]:.4f}]. '
              'Bootstrap percentile dùng 2.000 lần lấy mẫu lại toàn bộ nhóm tên tệp với hoàn lại, seed 20261114; '
              'giữ cặp phương pháp và tất cả ảnh trong mỗi nhóm. Khoảng này chỉ mô tả biến động lấy mẫu theo nhóm trong bộ kết quả hiện có; '
              'không sửa được nguồn checkpoint chưa rõ hay trùng lặp gần.', '',
              '## Độ nhạy với ảnh trùng byte', '',
              f'{d["duplicate_group_count"]} nhóm ảnh test trùng SHA-256 tạo {d["redundant_files"]} bản dư; '
              f'{d["unique_hashes"]} hash duy nhất. Không xóa dữ liệu. '
              'Cùng ảnh byte có thể mang mask khác nhau, vì vậy phân tích chính cho mỗi hash trọng số bằng nhau sau khi lấy trung bình tất cả phiên bản annotation; '
              'không tuyên bố đã khử trùng cả cặp ảnh–nhãn. Lựa chọn bản đầu theo tên chỉ là kiểm tra độ nhạy tùy ý.', '',
              '| Độ nhạy Dice | Baseline | TransUNet |', '|---|---:|---:|']
    for label, key in [('Tất cả ảnh', 'all_images'), ('Trọng số bằng nhau theo hash, giữ mọi nhãn', 'hash_balanced'), ('Một bản đầu theo tên mỗi hash', 'lexicographic_first_per_hash')]:
        lines.append(f'| {label} | {d[key]["baseline_dice"]:.4f} | {d[key]["transunet_dice"]:.4f} |')
    lines += ['', '## Phân tầng mô tả', '',
              'Các ngưỡng diện tích foreground được cố định: [0; 0,05), [0,05; 0,20), [0,20; 0,50), [0,50; 1]. '
              'Diện tích dựa trên mask đồng thuận tại 256×256, không đồng nghĩa mức độ bệnh.', '',
              '| Tỉ lệ foreground | Số ảnh | Baseline Dice | TransUNet Dice |', '|---|---:|---:|---:|']
    for s in report['foreground_strata']:
        lines.append(f'| {s["stratum"]} | {s["images"]} | {s["baseline_dice"]:.4f} | {s["transunet_dice"]:.4f} |')
    lines += ['', 'Đồng thuận annotation là trung bình Dice giữa các cặp mask trên thumbnail vuông 256×256; '
              'nhóm thấp <0,50 và nhóm cao ≥0,50. Điểm với tham chiếu đồng thuận và mức đồng thuận cùng xuất phát từ mask; '
              'chênh lệch giữa nhóm có thể liên quan cấu trúc nhãn, diện tích và nội dung ảnh, không cho phép kết luận nhân quả.', '',
              '| Đồng thuận annotation | Số ảnh | Baseline Dice | TransUNet Dice |', '|---|---:|---:|---:|']
    for s in report['agreement_strata']:
        if s['images']:
            lines.append(f'| {s["stratum"]} | {s["images"]} | {s["baseline_dice"]:.4f} | {s["transunet_dice"]:.4f} |')
    lines += ['', 'Tương quan Spearman trong `feature_associations.csv` chỉ mô tả mối liên hệ theo thứ hạng. '
              'Không có kiểm định nhân quả hay p-value; không điều chỉnh nhiễu hay đa so sánh. '
              'Độ sáng, tương phản, Laplacian trên thumbnail chịu ảnh hưởng nội dung, ánh sáng và phép resize; '
              'không chứng minh rằng ánh sáng hoặc độ nét gây ra sai số.', '', '## Sáu trường hợp cụ thể', '',
              'Lựa chọn hậu nghiệm để giải thích các hành vi khác nhau, không phải mẫu đại diện. '
              'Chỉ sáu ảnh này được chạy lại; mỗi metric được đối chiếu với CSV gốc (sai khác tối đa 1e-6). '
              'TP/FP/FN/TN đếm trên 65.536 pixel theo đúng mask tham chiếu gốc. '
              'Màu xanh lá: TP; cam: FP; xanh lam: FN. “Thừa/thiếu” là so với annotation, không phải chẩn đoán.', '']
    for case in report['selected_cases']:
        lines += [f'### {case["id"]} — {case["selection"]}', '',
                  f'![{case["id"]}](../results/analysis/{case["figure"]})', '',
                  f'Baseline Dice {case["baseline_dice"]:.4f}, FP {case["baseline_fp"]:,}, FN {case["baseline_fn"]:,}; '
                  f'TransUNet Dice {case["transunet_dice"]:.4f}, FP {case["transunet_fp"]:,}, FN {case["transunet_fn"]:,}. '
                  f'Foreground tham chiếu {case["target_fraction"]:.2%}; TransUNet dự đoán {case["transunet_prediction_fraction"]:.2%}. '
                  f'Dice đồng thuận giữa annotation {case["annotation_agreement"]:.4f}.', '']
        if case['id'] == 'IMG_ENC00764_00002':
            text = 'Baseline dự đoán 3.416 pixel foreground nhưng không pixel nào giao với tham chiếu (TP=0), đồng thời bỏ sót cả 169 pixel tham chiếu; TransUNet thu hồi 156/169 pixel với 4 FP và 13 FN. Đây là trường hợp cải thiện lớn nhất, không đại diện mức cải thiện trung bình.'
        elif case['id'] == 'IMG_ENC00238_00003':
            text = 'Hai phương pháp đều có mức trùng khớp cao; TransUNet đạt Dice cao nhất trong tập hiện tại. Sai số ranh giới vẫn còn tại độ phân giải đánh giá.'
        elif case['id'] == 'IMG_ENC00192_00005':
            text = 'TransUNet dự đoán diện tích nhỏ hơn nhiều tham chiếu, FN chiếm ưu thế; baseline tốt hơn rõ rệt. Đây là bằng chứng cụ thể rằng lợi thế trung bình không áp dụng cho mọi ảnh.'
        elif case['id'] == 'IMG_ENC00084_00003':
            text = 'TransUNet có Dice bằng 0: không có giao foreground với tham chiếu. FP/FN tách biệt cho thấy lỗi bỏ sót và vùng dự đoán ngoài tham chiếu, thay vì chỉ dựa vào tổng diện tích.'
        elif case['id'] == 'IMG_ENC00210_00001':
            text = 'TransUNet dự đoán phủ quá rộng so với đồng thuận; FP lớn và Dice thấp. Mức đồng thuận annotation cũng thấp nên tham chiếu có bất định, nhưng quan sát này không xác định nguyên nhân mô hình thất bại.'
        else:
            text = 'Mức đồng thuận annotation thấp nhất tập test. Bản đồ phiếu annotation cho thấy cần xem cả các mask riêng; TransUNet có pixel accuracy 0,9980 nhưng chỉ thu hồi 39/159 pixel foreground đồng thuận. Nền lớn làm accuracy cao trong khi recall thấp. Không thể chọn một annotator là “đúng” từ các dữ liệu hiện có.'
        lines += [text, '']
    lines += ['## Tệp và nguồn tái lập', '',
              '- `results/analysis/summary.json`: giao thức, counts, bootstrap, độ nhạy và danh mục hình.',
              '- `paired_images.csv`, `filename_subjects.csv`: từng cặp ảnh và từng nhóm tên tệp.',
              '- `foreground_strata.csv`, `agreement_strata.csv`, `feature_associations.csv`, `duplicate_groups.csv`: các bảng phụ.',
              '- `selected_cases.csv` và `cases/*`: metric, số pixel lỗi, ảnh nguồn PNG và mask thực.',
              '- `figure_captions.json`: chú thích nguồn và giới hạn của từng biểu đồ.', '',
              'Chạy lại: `.venv/Scripts/python.exe -m scripts.analyze_results --with-cases`. '
              'Đầu vào nguồn: `results/comparison_test/{per_image,per_annotation,summary}.csv/json` và `results/eda/per_image.csv`; '
              'hash nguồn được ghi trong summary. Bảng so sánh gốc và dữ liệu thô không bị thay đổi.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def analyze(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_paired(args.comparison, args.eda)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['subject']].append(row)
    group_rows = [{'subject': subject, 'images': len(values), **means(values),
                   'dice_difference': float(np.mean([r['dice_difference'] for r in values]))}
                  for subject, values in sorted(grouped.items())]
    boot = paired_cluster_bootstrap(rows, args.resamples, args.seed)
    differences = np.array([r['dice_difference'] for r in rows])
    dupes = duplicate_sensitivity(rows)
    feature_rows = []
    for feature in ('target_fraction', 'brightness_mean', 'contrast_std', 'sharpness_laplacian_variance', 'aspect_ratio', 'megapixels', 'pairwise_dice_thumbnail_mean', 'annotation_coverage_range'):
        for outcome in ('baseline_dice', 'transunet_dice', 'dice_difference'):
            valid = [r for r in rows if r[feature] is not None]
            feature_rows.append({'feature': feature, 'outcome': outcome, 'images': len(valid),
                                 'spearman_rho': spearman([r[feature] for r in valid], [r[outcome] for r in valid])})
    original = json.loads((args.comparison / 'summary.json').read_text(encoding='utf-8'))
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': {'scope': 'Saved paired test metrics; no retraining, no parameter tuning, no full-test re-inference.',
                     'reference': original['protocol']['reference'], 'evaluation_size': original['protocol']['evaluation_size'],
                     'normalization': original['protocol']['transunet_normalization'], 'threshold': original['protocol']['threshold'],
                     'cluster_rule': 'Regex (IMG_ENC digits) before final image suffix; proxy groups, not verified patients.',
                     'image_macro': 'Each test image equal weight.',
                     'filename_subject_macro': 'Mean within each filename proxy group, then each group equal weight.',
                     'bootstrap': 'Resample 103 whole filename proxy groups with replacement; retain paired methods and all images per draw. Image macro reweights by sampled group image counts; group macro averages sampled group means.',
                     'agreement': 'Saved EDA mean pairwise annotation Dice on nearest-neighbor 256-square thumbnails; high >=0.5.',
                     'foreground_bins': '[0,.05), [.05,.20), [.20,.50), [.50,1]',
                     'duplicate_policy': dupes['policy'],
                     'limits': ['Unknown checkpoint training history and test exposure; descriptive intervals cannot resolve leakage.',
                                'Filename groups are not verified patient identities; independent-cluster assumption unverified.',
                                'SHA-256 only detects byte-identical photographs, not near duplicates; masks can differ.',
                                'No causal, clinical or independent generalization inference; fixed local square-resized reference.']},
        'counts': {'images': len(rows), 'filename_groups': len(grouped), 'annotations_per_image': sorted({r['annotations'] for r in rows}),
                   'annotation_instances': sum(r['annotations'] for r in rows), 'unique_test_image_sha256': dupes['unique_hashes']},
        'image_macro': means(rows), 'filename_subject_macro': means(group_rows),
        'paired_dice': {'mean_difference': float(differences.mean()), 'median_difference': float(np.median(differences)),
                        'p25_difference': float(np.percentile(differences, 25)), 'p75_difference': float(np.percentile(differences, 75)),
                        'transunet_wins': int(np.count_nonzero(differences > 1e-12)), 'ties': int(np.count_nonzero(abs(differences) <= 1e-12)),
                        'baseline_wins': int(np.count_nonzero(differences < -1e-12)), 'tie_absolute_tolerance': 1e-12},
        'cluster_bootstrap': boot, 'duplicate_sensitivity': dupes,
        'foreground_strata': stratify(rows, 'foreground_stratum', STRATA),
        'agreement_strata': stratify(rows, 'agreement_stratum', ('low (<0.50)', 'high (>=0.50)', 'unavailable')),
        'feature_associations': feature_rows, 'selected_cases': [],
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)).replace('\\', '/'): file_sha256(p) for p in
                          (args.comparison / 'per_image.csv', args.comparison / 'per_annotation.csv', args.comparison / 'summary.json', args.eda / 'per_image.csv', Path(__file__))},
    }
    annotation_rows = read_csv(args.comparison / 'per_annotation.csv')
    report['per_annotation_macro'] = {
        method: {metric: float(np.mean([float(r[metric]) for r in annotation_rows if r['method'] == method])) for metric in METRICS}
        for method in METHODS}
    write_csv(args.output / 'paired_images.csv', rows)
    write_csv(args.output / 'filename_subjects.csv', group_rows)
    write_csv(args.output / 'duplicate_groups.csv', dupes['duplicate_groups'])
    for name, values in [('foreground_strata', report['foreground_strata']), ('agreement_strata', report['agreement_strata']), ('feature_associations', feature_rows)]:
        write_csv(args.output / f'{name}.csv', values)
    report['figures'] = save_figures(args.output, rows, report)
    if args.with_cases:
        report['selected_cases'] = export_cases(args.data_dir, args.output, rows)
    elif (args.output / 'selected_cases.json').is_file():
        report['selected_cases'] = json.loads((args.output / 'selected_cases.json').read_text(encoding='utf-8'))
    if report['selected_cases']:
        (args.output / 'selected_cases.json').write_text(json.dumps(report['selected_cases'], indent=2) + '\n', encoding='utf-8')
        write_csv(args.output / 'selected_cases.csv', report['selected_cases'])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_narrative(PROJECT_ROOT / 'docs/failure-analysis.md', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=PROJECT_ROOT / 'results/comparison_test')
    parser.add_argument('--eda', type=Path, default=PROJECT_ROOT / 'results/eda')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/analysis')
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261114)
    parser.add_argument('--with-cases', action='store_true')
    report = analyze(parser.parse_args(argv))
    print(json.dumps({'counts': report['counts'], 'paired_dice': report['paired_dice'],
                      'image_macro_cluster_bootstrap': report['cluster_bootstrap']['image_macro'],
                      'selected_cases': len(report['selected_cases'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
