"""Audit local extraction, annotation pairings, and split leakage indicators.

Run from Project: python -m scripts.audit_data
Only local files are read; archive payloads are not extracted or executed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FOLDERS = {'train': 'Train', 'valid': 'Valid', 'test': 'Test'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MASK_EXTENSIONS = {'.tif', '.tiff', '.png'}


def annotation_image_id(path: Path) -> str:
    return re.sub(r'_mask_ann\d+$', '', path.stem)


def discover_samples(data_dir: Path, split: str) -> list[dict]:
    """Match exact image stems or every discovered `_mask_annN` suffix."""
    folder = SPLIT_FOLDERS[split]
    image_dir, label_dir = data_dir / f'images{folder}', data_dir / f'labels{folder}'
    masks = defaultdict(list)
    if label_dir.is_dir():
        for path in sorted(label_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS:
                masks[annotation_image_id(path)].append(path)
    samples = []
    if image_dir.is_dir():
        for path in sorted(image_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append({'id': path.stem, 'image': path, 'masks': masks[path.stem]})
    return samples


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(data_dir: Path, archive: Path | None, check_hashes: bool = True) -> dict:
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_root': str(data_dir.resolve()),
        'scope': 'Local file inventory, header dimensions, subject identifiers, extraction sizes, image SHA-256.',
        'provenance': {
            'dataset_source': 'Unknown: archive contains no supplied dataset source or license documentation.',
            'subject_identifier_rule': 'IMG_ENC followed by digits, before the final image-number suffix.',
            'limitations': [
                'No clinical diagnosis, patient identity, or original acquisition provenance is established.',
                'Byte equality detects exact duplicate files only; resized/re-encoded copies and related subjects may remain.',
                'No split overlap does not establish checkpoint training provenance or exclude pretrained test contamination.',
                'Archive verification compares file sizes, not archive CRC or decompressed byte hashes.',
                'Image and mask integrity is checked by headers; this audit does not decode every pixel.',
            ],
        },
        'splits': {},
    }
    subjects, image_hashes = {}, defaultdict(list)
    problems = []
    for split, suffix in SPLIT_FOLDERS.items():
        samples = discover_samples(data_dir, split)
        image_ids = {sample['id'] for sample in samples}
        label_dir = data_dir / f'labels{suffix}'
        mask_paths = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in MASK_EXTENSIONS) if label_dir.is_dir() else []
        dimensions, modes, annotation_counts, subject_counts = Counter(), Counter(), Counter(), Counter()
        mismatched_masks, unreadable, unknown_subject_ids = [], [], []
        for sample in samples:
            annotation_counts[len(sample['masks'])] += 1
            match = re.match(r'^(IMG_ENC\d+)_\d+$', sample['id'])
            if match:
                subject_counts[match.group(1)] += 1
            else:
                unknown_subject_ids.append(sample['id'])
            try:
                with Image.open(sample['image']) as im:
                    size = im.size
                    dimensions[f'{size[0]}x{size[1]}'] += 1
                    modes[im.mode] += 1
                for mask_path in sample['masks']:
                    with Image.open(mask_path) as mask:
                        if mask.size != size:
                            mismatched_masks.append({'id': sample['id'], 'mask': mask_path.name, 'image_size': list(size), 'mask_size': list(mask.size)})
            except (OSError, ValueError) as exc:
                unreadable.append({'id': sample['id'], 'error': str(exc)})
            if check_hashes:
                image_hashes[file_sha256(sample['image'])].append({'split': split, 'id': sample['id'], 'file': sample['image'].name})
        subjects[split] = set(subject_counts)
        missing = [sample['id'] for sample in samples if not sample['masks']]
        orphans = [p.name for p in mask_paths if annotation_image_id(p) not in image_ids]
        duplicate_ids = [key for key, count in Counter(s['id'] for s in samples).items() if count > 1]
        split_result = {
            'images': len(samples), 'masks': len(mask_paths),
            'subjects': len(subject_counts), 'subject_ids': sorted(subject_counts),
            'images_per_subject': dict(sorted(Counter(subject_counts.values()).items())),
            'annotations_per_image': dict(sorted(annotation_counts.items())),
            'image_dimensions': dict(dimensions.most_common()), 'image_modes': dict(modes),
            'missing_masks': missing, 'orphan_masks': orphans,
            'duplicate_image_stems': duplicate_ids, 'mask_dimension_mismatches': mismatched_masks,
            'unreadable_headers': unreadable, 'unrecognized_subject_ids': unknown_subject_ids,
            'image_bytes': sum(s['image'].stat().st_size for s in samples),
            'mask_bytes': sum(p.stat().st_size for p in mask_paths),
        }
        result['splits'][split] = split_result
        if missing or orphans or duplicate_ids or mismatched_masks or unreadable:
            problems.append(f'{split}: missing/orphan/duplicate/mismatched/unreadable data; see split details')
    pairs = [('train', 'valid'), ('train', 'test'), ('valid', 'test')]
    result['cross_split_subject_overlap'] = {
        f'{a}__{b}': sorted(subjects[a] & subjects[b]) for a, b in pairs
    }
    duplicate_groups = [
        {'sha256': digest, 'files': entries} for digest, entries in image_hashes.items()
        if len({entry['split'] for entry in entries}) > 1
    ]
    result['exact_image_duplicates'] = {
        'algorithm': 'SHA-256', 'performed': check_hashes,
        'files_hashed': sum(len(group) for group in image_hashes.values()),
        'cross_split_groups': duplicate_groups,
        'within_split_duplicate_group_count': sum(
            len(entries) > 1 and len({e['split'] for e in entries}) == 1 for entries in image_hashes.values()
        ),
    }
    result['archive_size_verification'] = {'performed': False, 'reason': 'Archive not available.'}
    if archive is not None and archive.is_file():
        missing, mismatched, checked = [], [], 0
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                parts = Path(info.filename.replace('\\', '/')).parts
                # Only the known dataset/checkpoint payload; confine archive paths.
                if not parts or any(part in {'.', '..'} for part in parts) or Path(info.filename).is_absolute():
                    continue
                if not (parts[0].startswith(('images', 'labels')) or info.filename.endswith('.pth')):
                    continue
                checked += 1
                target = data_dir.joinpath(*parts)
                if not target.is_file():
                    missing.append(info.filename)
                elif target.stat().st_size != info.file_size:
                    mismatched.append({'file': info.filename, 'archive_bytes': info.file_size, 'extracted_bytes': target.stat().st_size})
        result['archive_size_verification'] = {
            'performed': True, 'archive': str(archive.resolve()), 'archive_bytes': archive.stat().st_size,
            'payload_files_checked': checked, 'missing_files': missing, 'size_mismatches': mismatched,
            'all_payload_sizes_match': not missing and not mismatched,
        }
        if missing or mismatched:
            problems.append('Archive payload is missing files or has file-size mismatches.')
    result['problems'] = problems
    return result


def main(argv=None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--archive', type=Path, default=PROJECT_ROOT / 'xulyanhungdung.zip')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'results/data_audit.json')
    parser.add_argument('--skip-hashes', action='store_true', help='Skip exact image duplicate checks.')
    args = parser.parse_args(argv)
    report = audit_dataset(args.data_dir, args.archive, not args.skip_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'splits': {k: {'images': v['images'], 'masks': v['masks'], 'subjects': v['subjects']} for k, v in report['splits'].items()}, 'subject_overlap': {k: len(v) for k, v in report['cross_split_subject_overlap'].items()}, 'cross_split_exact_duplicate_groups': len(report['exact_image_duplicates']['cross_split_groups']), 'problems': report['problems']}, indent=2))
    return 1 if report['problems'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
