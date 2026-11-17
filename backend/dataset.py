"""Read-only index and image conversion for the supplied dataset."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import os
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

SPLITS = {'train': 'Train', 'valid': 'Valid', 'test': 'Test'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


@dataclass(frozen=True)
class Sample:
    id: str
    image: Path
    masks: tuple[Path, ...]


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.samples: dict[str, dict[str, Sample]] = {}
        self.mask_counts: dict[str, int] = {}
        for split, suffix in SPLITS.items():
            labels: dict[str, list[Path]] = {}
            mask_files = self._files(f'labels{suffix}', {'.tif', '.tiff', '.png'})
            for f in mask_files:
                stem = re.sub(r'_mask_ann\d+$', '', f.stem)
                labels.setdefault(stem, []).append(f)
            self.mask_counts[split] = len(mask_files)
            self.samples[split] = {
                f.stem: Sample(f.stem, f, tuple(sorted(labels.get(f.stem, []), key=self._annotation_key)))
                for f in self._files(f'images{suffix}', IMAGE_SUFFIXES)
            }

    def _files(self, folder: str, suffixes: set[str]) -> list[Path]:
        directory = self.root / folder
        if not directory.is_dir():
            return []
        return sorted(f for f in directory.iterdir() if f.is_file()
                      and f.suffix.lower() in suffixes and f.resolve().is_relative_to(self.root))

    @staticmethod
    def _annotation_key(path: Path) -> int:
        match = re.search(r'_mask_ann(\d+)$', path.stem)
        return int(match[1]) if match else -1

    def get(self, split: str, sample_id: str) -> Sample:
        try:
            return self.samples[split][sample_id]
        except KeyError:
            raise KeyError('Không tìm thấy ảnh trong bộ dữ liệu.') from None

    def summary(self) -> dict:
        splits = {s: {'images': len(v), 'masks': self.mask_counts[s],
                       'unlabeled': sum(not x.masks for x in v.values())}
                  for s, v in self.samples.items()}
        checkpoint_path = Path(os.environ.get('TRANSUNET_CHECKPOINT', str(self.root / 'TransUNet_R50_ViT_B_16_all_best_model.pth'))).expanduser().resolve()
        checkpoint = checkpoint_path if checkpoint_path.is_file() else None
        return {
            'total_images': sum(s['images'] for s in splits.values()),
            'total_masks': sum(s['masks'] for s in splits.values()),
            'splits': splits,
            'model': {
                'name': 'TransUNet', 'architecture': 'R50 + ViT-B/16',
                'checkpoint_found': checkpoint is not None,
                'checkpoint_name': checkpoint.name if checkpoint else None,
                'checkpoint_bytes': checkpoint.stat().st_size if checkpoint else 0,
                'inference_ready': False,
            },
        }

    def list_samples(self, split: str, query: str, page: int, limit: int) -> dict:
        matches = [s for s in self.samples[split].values() if query.strip().casefold() in s.id.casefold()]
        items = matches[(page - 1) * limit:page * limit]
        return {'items': [{'id': s.id, 'filename': s.image.name, 'mask_count': len(s.masks)} for s in items],
                'total': len(matches), 'page': page, 'limit': limit, 'split': split}

    @staticmethod
    def read_image(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise ValueError('Không đọc được ảnh. Hãy kiểm tra file dữ liệu.') from None

    def read_mask(self, sample: Sample, annotation: int) -> Image.Image:
        if not 0 <= annotation < len(sample.masks):
            raise KeyError('Không tìm thấy bản gán nhãn đã chọn.')
        try:
            with Image.open(sample.masks[annotation]) as mask:
                array = np.asarray(mask.convert('L')) > 0
            with Image.open(sample.image) as image:
                if array.shape != (image.height, image.width):
                    raise ValueError('Kích thước mask không khớp ảnh gốc.')
            return Image.fromarray(array.astype('uint8') * 255)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError('Không đọc được file mask.') from None

    def read_reference(self, sample: Sample, annotation: int | str) -> Image.Image:
        if annotation != 'consensus':
            return self.read_mask(sample, int(annotation))
        if not sample.masks:
            raise KeyError('Ảnh chưa có nhãn tham chiếu để tạo mask đồng thuận.')
        counts = None
        for index in range(len(sample.masks)):
            binary = np.asarray(self.read_mask(sample, index)) > 0
            if counts is None:
                counts = np.zeros(binary.shape, dtype=np.uint16)
            counts += binary
        return Image.fromarray((counts > len(sample.masks) / 2).astype('uint8') * 255)

    def detail(self, split: str, sample_id: str) -> dict:
        sample = self.get(split, sample_id)
        image = self.read_image(sample.image)
        annotations = []
        for index, path in enumerate(sample.masks):
            try:
                mask = np.asarray(self.read_mask(sample, index)) > 0
                coverage = round(float(mask.mean()) * 100, 2)
                error = None
            except ValueError as exc:
                coverage, error = None, str(exc)
            match = re.search(r'_(ann\d+)$', path.stem)
            annotations.append({'index': index, 'name': match[1] if match else 'ground-truth',
                                'filename': path.name, 'coverage': coverage, 'error': error})
        consensus = None
        if len(sample.masks) > 1 and all(a['error'] is None for a in annotations):
            consensus_mask = np.asarray(self.read_reference(sample, 'consensus')) > 0
            consensus = {'index': 'consensus', 'name': 'consensus',
                         'coverage': round(float(consensus_mask.mean()) * 100, 2), 'error': None}
        return {'id': sample.id, 'filename': sample.image.name, 'split': split,
                'width': image.width, 'height': image.height, 'format': sample.image.suffix[1:].upper(),
                'annotations': annotations, 'consensus': consensus}


def encode_image(image: Image.Image, max_size: int, fmt: str = 'PNG') -> bytes:
    image = image.copy()
    if max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.NEAREST if fmt == 'PNG' else Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, format=fmt, **({'quality': 88} if fmt == 'JPEG' else {}))
    return output.getvalue()


def color_mask(binary: Image.Image) -> Image.Image:
    array = np.asarray(binary)
    rgba = np.zeros((*array.shape, 4), dtype=np.uint8)
    rgba[array > 0] = [32, 177, 153, 255]
    return Image.fromarray(rgba)


def color_errors(prediction: np.ndarray, reference: np.ndarray) -> Image.Image:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(reference, dtype=bool)
    if predicted.ndim != 2 or predicted.shape != target.shape:
        raise ValueError('Kích thước dự đoán không khớp mask tham chiếu.')
    rgba = np.zeros((*predicted.shape, 4), dtype=np.uint8)
    rgba[predicted & target] = [32, 177, 153, 255]
    rgba[predicted & ~target] = [239, 109, 86, 255]
    rgba[~predicted & target] = [80, 130, 210, 255]
    return Image.fromarray(rgba)
