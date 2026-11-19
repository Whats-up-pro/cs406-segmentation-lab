"""Local resource paths and compact-model inventory without importing PyTorch."""
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_root(default=None):
    return Path(os.environ.get('SEGMENTATION_DATA_DIR', str(default or ROOT/'data'))).expanduser().resolve()


def results_root(default=None):
    return Path(os.environ.get('SEGMENTATION_RESULTS_DIR', str(default or ROOT/'results'))).expanduser().resolve()


def deliverables_root(default=None):
    return Path(os.environ.get('SEGMENTATION_DELIVERABLES_DIR', str(default or ROOT/'deliverables'))).expanduser().resolve()


def compact_status(root=None):
    path = Path(os.environ.get('COMPACT_CHECKPOINT', str(Path(root or results_root())/'compact/bce_dice_aug_s406/best.pth'))).expanduser().resolve()
    result = {'name':'Compact U-Net', 'checkpoint_found':path.is_file(), 'inference_ready':False,
              'available':False, 'checkpoint_name':path.name, 'error':None, 'threshold':None,
              'checkpoint_path':str(path)}
    if not path.is_file():
        return result
    try:
        frozen = json.loads((path.parent/'frozen-selection.json').read_text(encoding='utf-8'))
        if (frozen.get('status') != 'frozen' or frozen.get('format') != 'compact-selection-v1'
                or frozen.get('model') != 'compact_unet' or frozen.get('checkpoint') != path.name
                or frozen.get('optimizer_splits') != ['train'] or frozen.get('selection_split') != 'valid'
                or frozen.get('test_accessed') is not False):
            raise ValueError('Thí nghiệm chưa có cấu hình đã chốt hợp lệ.')
        threshold = frozen.get('threshold')
        if not isinstance(threshold, (float,int)) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('Ngưỡng validation không hợp lệ.')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024), b''):digest.update(block)
        if digest.hexdigest() != frozen.get('checkpoint_sha256'):
            raise ValueError('SHA-256 trọng số khác cấu hình đã chốt.')
        config_path = path.parent/'config.json'
        config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
        result.update(inference_ready=True, available=True, threshold=threshold, seed=frozen.get('seed'),
                      epochs=frozen.get('epochs'), best_epoch=frozen.get('best_epoch'), counts=frozen.get('counts', {}),
                      checkpoint_bytes=path.stat().st_size, checkpoint_sha256=digest.hexdigest(),
                      input_size=config.get('image_size'), parameter_count=config.get('parameter_count'),
                      training_origin='Khởi tạo ngẫu nhiên; chỉ train cập nhật trọng số, validation chọn checkpoint/ngưỡng.')
    except (OSError, ValueError, TypeError) as error:
        result['error'] = str(error)
    return result
