"""Validate and prepare official ISIC2018 Task 1 validation archives.

Download both official URLs below with curl, retaining their original bytes,
then run python -m scripts.prepare_external --images-archive PATH --masks-archive PATH.
No network, training, inference or parameter selection occurs in this helper.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile

import numpy as np
from PIL import Image

from scripts.audit_data import PROJECT_ROOT, file_sha256


SOURCES = {
    'images': {'url': 'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip',
               'bytes': 239231159},
    'masks': {'url': 'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip',
              'bytes': 759706},
}
DATASET_PAGE = 'https://challenge.isic-archive.com/data/#2018'
SPECIFICATION = 'https://challenge.isic-archive.com/landing/2018/45/'
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024


def _members(archive: zipfile.ZipFile, kind: str) -> dict:
    selected, seen, total = {}, set(), 0
    expression = r'(ISIC_\d{7})\.jpg' if kind == 'images' else r'(ISIC_\d{7})_segmentation\.png'
    for info in archive.infolist():
        name = info.filename.replace('\\', '/')
        parts = name.split('/')
        mode = info.external_attr >> 16
        if (not name or name.startswith('/') or any(part in {'.', '..'} for part in parts)
                or any(':' in part or '\x00' in part for part in parts) or stat.S_ISLNK(mode)
                or info.flag_bits & 1):
            raise ValueError(f'Unsafe archive member: {info.filename}')
        if info.is_dir():
            continue
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f'Unsafe archive member size: {info.filename}')
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError('Unsafe archive total expanded size.')
        basename = PurePosixPath(name).name
        normalized = basename.casefold()
        if normalized in seen:
            raise ValueError(f'Duplicate flattened archive member: {basename}')
        seen.add(normalized)
        match = re.fullmatch(expression, basename)
        if match:
            if match.group(1) in selected:
                raise ValueError(f'Duplicate image/mask id: {match.group(1)}')
            selected[match.group(1)] = info
        elif basename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
            raise ValueError(f'Unexpected {kind} image filename: {info.filename}')
        elif basename.lower() not in {'license.txt', 'license', 'readme.txt', 'readme.md', 'attribution.txt'}:
            raise ValueError(f'Unexpected archive payload: {info.filename}')
    return selected


def prepare_external(images_archive: Path, masks_archive: Path, output: Path,
                     expected_count: int = 100, verify_official_sizes: bool = True) -> dict:
    """Preflight both archives completely before writing paired prepared files.

    Overwrite is refused. Nondefault counts/size checking exist for small offline
    tests; the CLI always requires the official 100 pairs and advertised sizes.
    """
    output = Path(output).resolve()
    if expected_count < 1:
        raise ValueError('Expected image count must be positive.')
    targets = [output / name for name in ('imagesTest', 'labelsTest', 'manifest.json')]
    if any(path.exists() for path in targets):
        raise FileExistsError(f'Prepared output already exists under {output}.')
    paths = {'images': Path(images_archive).resolve(), 'masks': Path(masks_archive).resolve()}
    for kind, path in paths.items():
        if path in targets or any(path.is_relative_to(folder) for folder in targets[:2]):
            raise ValueError('Original archive cannot be inside a prepared destination folder.')
        if verify_official_sizes and path.stat().st_size != SOURCES[kind]['bytes']:
            raise ValueError(f'Official {kind} archive size mismatch: expected {SOURCES[kind]["bytes"]} bytes.')
    samples, ignored = [], {}
    try:
        with zipfile.ZipFile(paths['images']) as image_zip, zipfile.ZipFile(paths['masks']) as mask_zip:
            images, masks = _members(image_zip, 'images'), _members(mask_zip, 'masks')
            if len(images) != expected_count or len(masks) != expected_count:
                raise ValueError(f'Expected {expected_count} image/mask pairs, found {len(images)} images and {len(masks)} masks.')
            if images.keys() != masks.keys():
                raise ValueError(f'Image/mask pairing mismatch: missing masks={sorted(images.keys()-masks.keys())}, orphan masks={sorted(masks.keys()-images.keys())}.')
            for sample_id in sorted(images):
                image_bytes, mask_bytes = image_zip.read(images[sample_id]), mask_zip.read(masks[sample_id])
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        size = image.size
                        if image.format != 'JPEG' or image.mode != 'RGB':
                            raise ValueError('Expected an RGB JPEG image.')
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Invalid image {sample_id}: {exc}') from exc
                try:
                    with Image.open(io.BytesIO(mask_bytes)) as mask:
                        mask.load()
                        values = np.asarray(mask)
                        if mask.format != 'PNG' or mask.mode != 'L' or values.ndim != 2 or mask.size != size:
                            raise ValueError('Expected a single-channel PNG mask with matching image dimensions.')
                        unique = np.unique(values).tolist()
                        if not set(unique) <= {0, 255}:
                            raise ValueError(f'Expected mask values 0/255, found {unique}.')
                        foreground_pixels = int(np.count_nonzero(values))
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Invalid mask {sample_id}: {exc}') from exc
                samples.append({'id': sample_id, 'width': size[0], 'height': size[1],
                                'image': f'imagesTest/{sample_id}.jpg', 'mask': f'labelsTest/{sample_id}.png',
                                'source_image_member': images[sample_id].filename, 'source_mask_member': masks[sample_id].filename,
                                'image_sha256': hashlib.sha256(image_bytes).hexdigest(),
                                'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
                                'mask_values': unique, 'foreground_pixels': foreground_pixels,
                                'foreground_fraction': foreground_pixels / (size[0] * size[1])})
            for kind, archive, selected in [('images', image_zip, images), ('masks', mask_zip, masks)]:
                selected_names = {info.filename for info in selected.values()}
                ignored[kind] = [info.filename for info in archive.infolist() if not info.is_dir() and info.filename not in selected_names]
                for name in ignored[kind]:
                    archive.read(name)  # Also check CRC on permitted accompanying notices.
            manifest = {
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'dataset': 'ISIC 2018 Task 1 validation', 'source_split': 'validation',
                'role': 'External dermoscopy domain-shift stress test with frozen parameters; no external fitting or selection',
                'counts': {'images': len(images), 'masks': len(masks)},
                'official_release_identity_verified': False,
                'integrity_scope': 'Advertised archive sizes (when enabled), ZIP CRC, exact pairing, decoded JPEG/PNG, 0/255 masks and shape; hashes recorded locally. No organizer-published SHA-256 was available for identity comparison.',
                'validation': {'required_pairs': expected_count, 'official_sizes_checked': verify_official_sizes},
                'license': {'id': 'CC0-1.0', 'url': 'https://creativecommons.org/publicdomain/zero/1.0/', 'evidence': DATASET_PAGE},
                'specification': SPECIFICATION, 'collection': 'https://api.isic-archive.com/collections/62/',
                'archives': {kind: {'path': str(path), 'url': SOURCES[kind]['url'], 'bytes': path.stat().st_size,
                                     'sha256': file_sha256(path), 'advertised_bytes': SOURCES[kind]['bytes']}
                             for kind, path in paths.items()},
                'preparation': 'Image and mask payload bytes preserved; paths flattened; mask suffix removed for discover_samples pairing; no image resize or label modification',
                'original_archives_modified': False, 'ignored_notice_members': ignored,
                'samples': samples, 'source_sha256': file_sha256(Path(__file__)),
                'limitations': [
                    'Dermoscopic single-primary-lesion boundaries differ from contextual consumer photograph problem-region masks.',
                    'No model, threshold, augmentation or checkpoint is selected using this dataset.',
                    'Unknown supplied checkpoint history prevents claiming its external images were unseen during pretraining.',
                    'This stress evaluation is not a clinical validation or a reproduction of an official ISIC challenge submission.',
                ],
                'citation': 'Codella et al. Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by ISIC. arXiv:1902.03368; Tschandl et al. The HAM10000 dataset. Scientific Data 5, 180161 (2018).',
            }
            output.mkdir(parents=True, exist_ok=True)
            staging = output / f'.preparing-{uuid.uuid4().hex}'
            staging.mkdir()
            try:
                for folder in ('imagesTest', 'labelsTest'):
                    (staging / folder).mkdir()
                for sample in samples:
                    (staging / sample['image']).write_bytes(image_zip.read(images[sample['id']]))
                    (staging / sample['mask']).write_bytes(mask_zip.read(masks[sample['id']]))
                (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
                for name in ('imagesTest', 'labelsTest', 'manifest.json'):
                    (staging / name).rename(output / name)
            finally:
                # This exclusively-created staging directory is confined to output.
                if staging.exists() and staging.resolve().parent == output:
                    shutil.rmtree(staging)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f'Invalid archive: {exc}') from exc
    return manifest


def main(argv=None) -> int:
    default = PROJECT_ROOT.parent / 'Project/data_external/isic2018_validation'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-archive', type=Path, default=default / 'raw/ISIC2018_Task1-2_Validation_Input.zip')
    parser.add_argument('--masks-archive', type=Path, default=default / 'raw/ISIC2018_Task1_Validation_GroundTruth.zip')
    parser.add_argument('--output', type=Path, default=default)
    args = parser.parse_args(argv)
    manifest = prepare_external(args.images_archive, args.masks_archive, args.output)
    print(json.dumps({'manifest': str((args.output / 'manifest.json').resolve()), 'counts': manifest['counts'],
                      'archives': manifest['archives']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
