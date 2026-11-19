"""Read-only SHA-256, encounter-name and perceptual image candidate audit.

Example: python -m scripts.audit_leakage --data-dir ../Project/data
Perceptual proximity nominates pairs for visual review; it never proves identity.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from scripts.audit_data import PROJECT_ROOT, discover_samples, file_sha256


def _perceptual_hashes(image: Image.Image) -> tuple[int, int, float]:
    gray = image.convert('L')
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    bits[0] = False  # Exclude the DC brightness component from comparisons.
    phash = int.from_bytes(np.packbits(bits).tobytes(), 'big')
    gradient = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = int.from_bytes(np.packbits(gradient[:, 1:] > gradient[:, :-1]).tobytes(), 'big')
    return phash, dhash, float(small.std())


def audit_leakage(data_dir: Path, phash_threshold: int = 6, dhash_threshold: int = 8) -> dict:
    """Inspect all supplied images, preserving files and supplied splits."""
    if not all(isinstance(value, int) and 0 <= value <= 64 for value in (phash_threshold, dhash_threshold)):
        raise ValueError('Each Hamming threshold must be an integer in [0, 64].')
    data_dir = Path(data_dir).resolve()
    started = time.perf_counter()
    inventory, errors = [], []
    by_split, by_hash, by_encounter = defaultdict(list), defaultdict(list), defaultdict(list)
    for split in ('train', 'valid', 'test'):
        for sample in discover_samples(data_dir, split):
            path = sample['image'].resolve()
            if not path.is_relative_to(data_dir):
                raise ValueError(f'Image lies outside the data root: {path}')
            match = re.fullmatch(r'IMG_(ENC\d+)_\d+', sample['id'])
            record = {'split': split, 'id': sample['id'], 'path': path.relative_to(data_dir).as_posix(),
                      'encounter': match.group(1) if match else None,
                      'sha256': file_sha256(path), 'bytes': path.stat().st_size,
                      'width': None, 'height': None, 'phash': None, 'dhash': None,
                      'grayscale_std_32': None}
            try:
                with Image.open(path) as image:
                    image.load()
                    record['width'], record['height'] = image.size
                    phash, dhash, deviation = _perceptual_hashes(image)
                record.update(phash=f'{phash:016x}', dhash=f'{dhash:016x}', grayscale_std_32=deviation)
                by_split[split].append(record)
            except (OSError, ValueError) as exc:
                errors.append({'split': split, 'id': sample['id'], 'path': record['path'],
                               'error': f'{type(exc).__name__}: {exc}'})
            inventory.append(record)
            by_hash[record['sha256']].append(record)
            if record['encounter'] is not None:
                by_encounter[record['encounter']].append(record)
    if not inventory:
        raise ValueError(f'No images found under data root {data_dir}.')
    duplicates = [{'sha256': digest, 'cross_split': len({row['split'] for row in rows}) > 1,
                   'files': [{'split': row['split'], 'id': row['id'], 'path': row['path']} for row in rows]}
                  for digest, rows in sorted(by_hash.items()) if len(rows) > 1]
    candidates, pair_count = [], 0
    for left_split, right_split in itertools.combinations(('train', 'valid', 'test'), 2):
        for left in by_split[left_split]:
            left_p, left_d = int(left['phash'], 16), int(left['dhash'], 16)
            for right in by_split[right_split]:
                pair_count += 1
                if left['sha256'] == right['sha256']:
                    continue  # Exact byte identity is stronger, separately reported evidence.
                pd = (left_p ^ int(right['phash'], 16)).bit_count()
                if pd > phash_threshold:
                    continue
                dd = (left_d ^ int(right['dhash'], 16)).bit_count()
                if dd <= dhash_threshold:
                    candidates.append({'left_split': left_split, 'left_id': left['id'], 'left_path': left['path'],
                                       'right_split': right_split, 'right_id': right['id'], 'right_path': right['path'],
                                       'phash_distance': pd, 'dhash_distance': dd,
                                       'low_information': min(left['grayscale_std_32'], right['grayscale_std_32']) < 5,
                                       'review_status': 'pending_visual_review'})
    candidates.sort(key=lambda row: (row['phash_distance'] + row['dhash_distance'], row['left_path'], row['right_path']))
    overlap = {}
    for a, b in itertools.combinations(('train', 'valid', 'test'), 2):
        overlap[f'{a}__{b}'] = sorted(encounter for encounter, rows in by_encounter.items()
                                    if {a, b} <= {row['split'] for row in rows})
    return {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'data_root': str(data_dir),
        'protocol': {'exact': 'SHA-256 of original file bytes; no image or label edits',
                     'encounter_rule': r'Full filename stem IMG_(ENC\d+)_\d+; encounter is not patient identity',
                     'phash': 'Grayscale LANCZOS 32x32; OpenCV DCT; top-left 8x8; median excluding DC; DC bit zero',
                     'dhash': 'Grayscale LANCZOS 9x8; each right pixel strictly greater than left pixel',
                     'phash_bits': 64, 'dhash_bits': 64, 'phash_max_hamming': phash_threshold,
                     'dhash_max_hamming': dhash_threshold,
                     'candidate_rule': 'Different SHA-256 across splits AND pHash distance <= threshold AND dHash distance <= threshold',
                     'low_information_rule': 'Either grayscale 32x32 thumbnail standard deviation < 5; possible hash collision',
                     'comparison_scope': 'All decoded cross-split pairs; exact duplicates separately reported',
                     'labels_read': False, 'files_modified': False},
        'counts': {'images': len(inventory), 'decoded_images': sum(len(rows) for rows in by_split.values()),
                   'images_by_split': dict(Counter(row['split'] for row in inventory)),
                   'unique_sha256': len(by_hash), 'encounter_groups': len(by_encounter),
                   'unrecognized_encounters': sum(row['encounter'] is None for row in inventory),
                   'cross_split_pairs_compared': pair_count,
                   'cross_split_exact_groups': sum(group['cross_split'] for group in duplicates),
                   'within_split_only_exact_groups': sum(not group['cross_split'] for group in duplicates),
                   'cross_split_perceptual_candidates': len(candidates)},
        'cross_split_encounter_overlap': overlap,
        'encounter_groups': [{'encounter': key, 'splits': sorted({row['split'] for row in rows}),
                              'images': [{'split': row['split'], 'id': row['id']} for row in rows]}
                             for key, rows in sorted(by_encounter.items())],
        'exact_duplicate_groups': duplicates, 'cross_split_candidates': candidates,
        'inventory': inventory, 'decode_errors': errors,
        'wall_seconds': time.perf_counter() - started,
        'source_sha256': file_sha256(Path(__file__)),
        'limitations': [
            'Encounter names are provided filename groups; patient independence is not established.',
            'Perceptual proximity is a candidate signal, not proof of duplicate content or clinical identity.',
            'Fixed thresholds can miss crops, rotations and other transformations and can flag unrelated similar images.',
            'No source files, supplied splits or annotations are changed; candidates require documented visual review.',
            'A clean audit cannot establish the supplied checkpoint training history or rule out prior test exposure.',
            'The local test set and results were historically inspected before this completion work; it is not a pristine holdout.',
        ],
    }


def write_audit(report: dict, output: Path, contact_sheet_limit: int = 30) -> None:
    output, data_dir = Path(output).resolve(), Path(report['data_root']).resolve()
    if output.is_relative_to(data_dir):
        raise ValueError('Audit output must be outside the read-only data root.')
    if contact_sheet_limit < 0:
        raise ValueError('Contact sheet limit cannot be negative.')
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows, fields in [
        ('inventory.csv', report['inventory'], ['split', 'id', 'path', 'encounter', 'sha256', 'bytes', 'width', 'height', 'phash', 'dhash', 'grayscale_std_32']),
        ('cross_split_candidates.csv', report['cross_split_candidates'], ['left_split', 'left_id', 'left_path', 'right_split', 'right_id', 'right_path', 'phash_distance', 'dhash_distance', 'low_information', 'review_status']),
    ]:
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    selected = report['cross_split_candidates'][:contact_sheet_limit]
    if selected:
        sheet = Image.new('RGB', (720, 184 * len(selected)), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(selected):
            for column, side in enumerate(('left', 'right')):
                path = (data_dir / row[f'{side}_path']).resolve()
                if not path.is_relative_to(data_dir):
                    raise ValueError('Contact sheet image is outside the data root.')
                with Image.open(path) as image:
                    thumb = ImageOps.contain(image.convert('RGB'), (350, 142))
                sheet.paste(thumb, (column * 360, index * 184 + 36))
                draw.text((column * 360 + 3, index * 184 + 3), f"{row[f'{side}_split']} {row[f'{side}_id']}", fill='black')
            draw.text((3, index * 184 + 20), f"Candidate {index+1}: pHash={row['phash_distance']}, dHash={row['dhash_distance']}; PENDING REVIEW", fill='black')
        sheet.save(output / 'candidate_contact_sheet.jpg', quality=88)
    report['contact_sheet'] = {'candidate_count_shown': len(selected),
                               'selection': 'Lowest sum of Hamming distances, then filenames; no adjudication implied',
                               'file': 'candidate_contact_sheet.jpg' if selected else None}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT.parent / 'Project/data')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT.parent / 'Project/results/completion/leakage')
    parser.add_argument('--phash-threshold', type=int, default=6)
    parser.add_argument('--dhash-threshold', type=int, default=8)
    parser.add_argument('--contact-sheet-limit', type=int, default=30)
    args = parser.parse_args(argv)
    report = audit_leakage(args.data_dir, args.phash_threshold, args.dhash_threshold)
    write_audit(report, args.output, args.contact_sheet_limit)
    print(json.dumps({'output': str(args.output.resolve()), 'counts': report['counts'],
                      'encounter_overlap': report['cross_split_encounter_overlap'], 'decode_errors': report['decode_errors']}, indent=2))
    return 1 if report['decode_errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
