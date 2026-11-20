"""Full controlled Compact U-Net training; no test directory is indexed or read."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from backend.compact_model import CompactUNet, binary_loss, save_compact_checkpoint, load_compact_checkpoint, resize_compact_input, INPUT_PREPROCESSING
from scripts.train import discover_pairs

ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SIZE = 256
CACHE_ARRAYS = ('train_images.npy', 'train_masks.npy', 'valid_images.npy',
                'valid_masks.npy', 'valid_eval_masks.npy')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def prepare_training_cache(data_dir, cache_dir, image_size=128):
    data_dir, cache_dir = Path(data_dir).resolve(), Path(cache_dir).resolve()
    if not isinstance(image_size, int) or image_size < 8:
        raise ValueError('image_size must be >=8')
    splits = {split: discover_pairs(data_dir, split) for split in ['train', 'valid']}
    if {p.image.stem for p in splits['train']} & {p.image.stem for p in splits['valid']}:
        raise ValueError('Training and validation image IDs overlap')
    samples = {split: [{'id': pair.image.stem, 'image': str(pair.image.relative_to(data_dir)),
                        'mask': str(pair.mask.relative_to(data_dir)), 'image_sha256': sha256(pair.image),
                        'mask_sha256': sha256(pair.mask)} for pair in pairs] for split, pairs in splits.items()}
    fingerprint = hashlib.sha256(json.dumps({'size': image_size, 'samples': samples, 'preprocessing': INPUT_PREPROCESSING}, sort_keys=True).encode()).hexdigest()
    manifest_path = cache_dir / 'manifest.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('fingerprint') != fingerprint:
            raise ValueError('Existing cache differs from source data or requested resolution; choose a fresh cache directory')
        checksums = existing.get('array_sha256')
        if not isinstance(checksums, dict) or set(checksums) != set(CACHE_ARRAYS):
            raise ValueError('Missing cache array hashes; choose a fresh cache directory')
        for name in CACHE_ARRAYS:
            path = cache_dir / name
            if not path.is_file() or sha256(path) != checksums[name]:
                raise ValueError(f'Invalid cache array hash: {name}; choose a fresh cache directory')
        return existing
    cache_dir.mkdir(parents=True, exist_ok=True)
    if any(cache_dir.glob('*.npy')):
        raise FileExistsError('Incomplete cache already exists; choose a fresh cache directory')
    for split, pairs in splits.items():
        images = np.lib.format.open_memmap(cache_dir / f'{split}_images.npy', mode='w+', dtype=np.uint8,
                                          shape=(len(pairs), 3, image_size, image_size))
        masks = np.lib.format.open_memmap(cache_dir / f'{split}_masks.npy', mode='w+', dtype=np.uint8,
                                         shape=(len(pairs), 1, image_size, image_size))
        evaluation_masks = None
        if split == 'valid':
            evaluation_masks = np.lib.format.open_memmap(cache_dir / 'valid_eval_masks.npy', mode='w+', dtype=np.uint8,
                                                        shape=(len(pairs), 1, EVALUATION_SIZE, EVALUATION_SIZE))
        for index, pair in enumerate(pairs):
            with Image.open(pair.image) as source:
                image = source.convert('RGB')
            with Image.open(pair.mask) as source:
                raw = np.asarray(source)
                if source.size != image.size or raw.ndim != 2 or not np.isfinite(raw).all():
                    raise ValueError(f'Invalid label dimensions or values: {pair.mask.name}')
                label = Image.fromarray((raw != 0).astype(np.uint8))
            images[index] = np.asarray(resize_compact_input(image, image_size)).transpose(2, 0, 1)
            masks[index, 0] = np.asarray(label.resize((image_size, image_size), Image.Resampling.NEAREST))
            if evaluation_masks is not None:
                evaluation_masks[index, 0] = np.asarray(label.resize((EVALUATION_SIZE, EVALUATION_SIZE), Image.Resampling.NEAREST))
        images.flush(); masks.flush()
        if evaluation_masks is not None:
            evaluation_masks.flush()
    manifest = {'format': 'compact-training-cache-v2', 'fingerprint': fingerprint, 'image_size': image_size,
                'preprocessing': INPUT_PREPROCESSING,
                'evaluation_size': EVALUATION_SIZE, 'data_dir': str(data_dir), 'samples': samples,
                'counts': {split: len(pairs) for split, pairs in splits.items()}, 'test_accessed': False,
                'array_sha256': {name: sha256(cache_dir / name) for name in CACHE_ARRAYS}}
    write_json(manifest_path, manifest)
    return manifest


class CachedPairs(Dataset):
    def __init__(self, cache_dir, split, augment=False, seed=406, color_jitter=True):
        if split not in {'train', 'valid'}:
            raise ValueError('Training cache supports only train and valid')
        if split == 'valid' and augment:
            raise ValueError('Validation augmentation is prohibited')
        self.images = np.load(Path(cache_dir) / f'{split}_images.npy', mmap_mode='r', allow_pickle=False)
        self.masks = np.load(Path(cache_dir) / f'{split}_masks.npy', mmap_mode='r', allow_pickle=False)
        self.augment, self.color_jitter, self.seed, self.epoch = augment, color_jitter, seed, 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.array(self.images[index], copy=True)).float()/255.0
        target = torch.from_numpy(np.array(self.masks[index], copy=True)).float()
        if self.augment:
            rng = random.Random((self.seed << 32) + self.epoch*1_000_003 + index)
            for dimension in [1, 2]:
                if rng.random() < .5:
                    image, target = image.flip(dimension), target.flip(dimension)
            turns = rng.randrange(4)
            image, target = image.rot90(turns, (1, 2)), target.rot90(turns, (1, 2))
            if self.color_jitter:
                brightness, contrast = rng.uniform(.85, 1.15), rng.uniform(.85, 1.15)
                mean = image.mean()
                image = (((image-mean)*contrast + mean)*brightness).clamp(0, 1)
        return image, target


def hard_dice(probabilities, targets, threshold=.5):
    prediction, truth = probabilities >= threshold, targets.bool()
    intersection = (prediction & truth).sum((1, 2, 3)).float()
    total = prediction.sum((1, 2, 3)) + truth.sum((1, 2, 3))
    return torch.where(total == 0, torch.ones_like(intersection), 2*intersection/total.clamp_min(1))


def run_epoch(model, loader, optimizer=None, dice_weight=1.0, device='cpu'):
    model.train(optimizer is not None)
    loss_sum, dice_sum, samples, batches = 0., 0., 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = binary_loss(logits, targets, dice_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite compact-model loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            count = images.shape[0]
            loss_sum += loss.item()*count
            dice_sum += hard_dice(logits.detach().sigmoid(), targets).sum().item()
            samples += count; batches += 1
    if samples == 0:
        raise ValueError('Epoch did not process any samples')
    return {'loss': loss_sum/samples, 'dice': dice_sum/samples, 'samples': samples, 'batches': batches}


def select_threshold(model, cache_dir, batch_size=16, device='cpu'):
    dataset = CachedPairs(cache_dir, 'valid')
    targets = torch.from_numpy(np.array(np.load(Path(cache_dir) / 'valid_eval_masks.npy', mmap_mode='r'), copy=True))
    outputs = []
    model.eval()
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            scores = model(images.to(device)).sigmoid()
            outputs.append(F.interpolate(scores, size=(EVALUATION_SIZE, EVALUATION_SIZE), mode='bilinear', align_corners=False).cpu())
    probabilities = torch.cat(outputs)
    candidates = [{'threshold': round(float(value), 2), 'valid_dice': hard_dice(probabilities, targets, float(value)).mean().item()}
                  for value in np.linspace(.30, .70, 9)]
    best = max(candidates, key=lambda row: row['valid_dice'])
    return best, candidates


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or not 1 <= args.cpu_threads <= 8:
        raise ValueError('epochs/batch_size must be positive; cpu_threads must be between1and8')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.seed < 0:
        raise ValueError('Invalid learning rate or seed')
    args.output, args.cache_dir, args.data_dir = Path(args.output).resolve(), Path(args.cache_dir).resolve(), Path(args.data_dir).resolve()
    resume = bool(getattr(args, 'resume', False))
    if args.output.exists() and not resume:
        raise FileExistsError(f'Run output already exists: {args.output}')
    if resume and not args.output.is_dir():
        raise FileNotFoundError(f'No existing run to resume: {args.output}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    requested = getattr(args, 'device', 'cpu')
    if requested not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('device must be auto, cpu or cuda')
    device = 'cuda' if requested == 'auto' and torch.cuda.is_available() else ('cpu' if requested == 'auto' else requested)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    # Bilinear interpolation backward on CUDA has no deterministic kernel in
    # some supported PyTorch versions. Record that limitation instead of
    # failing the GPU run or claiming bitwise reproducibility there.
    torch.use_deterministic_algorithms(True, warn_only=(device == 'cuda'))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cache = prepare_training_cache(args.data_dir, args.cache_dir, args.image_size)
    args.output.mkdir(parents=True, exist_ok=resume)
    model = CompactUNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    source_hashes = {name: sha256(ROOT/name) for name in ['backend/compact_model.py', 'scripts/train_compact.py', 'scripts/train.py']}
    config = {**{name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
              'format': 'compact-training-run-v1', 'status': 'training', 'initialization': 'random from seed; no pretrained parameters',
              'optimizer_splits': ['train'], 'selection_split': 'valid', 'test_accessed': False,
              'input_preprocessing': INPUT_PREPROCESSING,
              'counts': cache['counts'], 'data_fingerprint': cache['fingerprint'], 'source_sha256': source_hashes,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'environment': {'python': platform.python_version(), 'torch': str(torch.__version__), 'numpy': np.__version__,
                              'device': device, 'cpu_threads': args.cpu_threads,
                              'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                              'deterministic_algorithms': 'warn_only' if device == 'cuda' else 'strict',
                              'cudnn_deterministic': True},
              'created_utc': datetime.now(timezone.utc).isoformat()}
    history, best_dice, best_epoch, start_epoch = [], -1., 0, 1
    if resume:
        existing = json.loads((args.output/'config.json').read_text(encoding='utf-8'))
        immutable = ['image_size', 'base_channels', 'epochs', 'batch_size', 'learning_rate', 'weight_decay',
                     'dice_weight', 'augment', 'seed', 'data_fingerprint', 'source_sha256']
        if any(existing.get(key) != config.get(key) for key in immutable):
            raise ValueError('Resume configuration, source or data differs from the original protocol')
        config = existing
        if (args.output/'frozen-selection.json').is_file():
            frozen = json.loads((args.output/'frozen-selection.json').read_text(encoding='utf-8'))
            for filename, key in [('best.pth', 'checkpoint_sha256'), ('config.json', 'config_sha256'), ('history.json', 'history_sha256')]:
                if sha256(args.output/filename) != frozen[key]:
                    raise ValueError('Completed frozen run was modified')
            return frozen
        snapshot = torch.load(args.output/'last.pth', map_location='cpu', weights_only=True)
        if snapshot.get('format') != 'compact-resume-v1' or snapshot.get('source_sha256') != source_hashes:
            raise ValueError('Unsupported or incompatible resume snapshot')
        model.load_state_dict(snapshot['state_dict'], strict=True)
        optimizer.load_state_dict(snapshot['optimizer'])
        history = json.loads((args.output/'history.json').read_text(encoding='utf-8'))[:snapshot['epoch']]
        if len(history) != snapshot['epoch']:
            raise ValueError('Resume history does not contain all committed epochs')
        best_dice, best_epoch, start_epoch = snapshot['best_dice'], snapshot['best_epoch'], snapshot['epoch']+1
        torch.save(snapshot['best_payload'], args.output/'best.pth')
    else:
        write_json(args.output / 'config.json', config)
    train_dataset = CachedPairs(args.cache_dir, 'train', args.augment, args.seed)
    valid_dataset = CachedPairs(args.cache_dir, 'valid')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for epoch in range(start_epoch, args.epochs+1):
        started = time.perf_counter(); train_dataset.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        trained = run_epoch(model, train_loader, optimizer, args.dice_weight, device)
        updated = sum(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
        valid = run_epoch(model, valid_loader, None, args.dice_weight, device)
        improved = valid['dice'] > best_dice
        if improved:
            best_dice, best_epoch = valid['dice'], epoch
            save_compact_checkpoint(model, args.output/'best.pth', args.image_size, {'epoch': epoch, 'seed': args.seed, 'valid_dice': best_dice})
        row = {'epoch': epoch, 'train_loss': trained['loss'], 'train_dice': trained['dice'],
               'valid_loss': valid['loss'], 'valid_dice': valid['dice'], 'train_samples': trained['samples'],
               'valid_samples': valid['samples'], 'train_batches': trained['batches'], 'valid_batches': valid['batches'],
               'updated_parameters': updated, 'seconds': time.perf_counter()-started, 'best_so_far': improved}
        history.append(row); write_json(args.output/'history.json', history)
        with (args.output/'history.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        snapshot = {'format': 'compact-resume-v1', 'epoch': epoch, 'best_epoch': best_epoch, 'best_dice': best_dice,
                    'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'source_sha256': source_hashes,
                    'best_payload': torch.load(args.output/'best.pth', map_location='cpu', weights_only=True)}
        temporary = args.output/'last.pth.tmp'
        torch.save(snapshot, temporary); os.replace(temporary, args.output/'last.pth')
        print(json.dumps({'run': args.output.name, **row}), flush=True)
    best_model, _, _ = load_compact_checkpoint(args.output/'best.pth', device)
    selection, candidates = select_threshold(best_model, args.cache_dir, args.batch_size, device)
    config['status'] = 'frozen'; config['best_epoch'] = best_epoch
    write_json(args.output/'config.json', config)
    frozen = {'format': 'compact-selection-v1', 'status': 'frozen', 'model': 'compact_unet', 'best_epoch': best_epoch,
              'optimizer_splits': ['train'], 'selection_split': 'valid', 'test_accessed': False,
              'threshold': selection['threshold'], 'valid_dice': selection['valid_dice'], 'threshold_search': candidates,
              'checkpoint': 'best.pth', 'checkpoint_sha256': sha256(args.output/'best.pth'),
              'config_sha256': sha256(args.output/'config.json'), 'history_sha256': sha256(args.output/'history.json'),
              'cache_fingerprint': cache['fingerprint'], 'evaluation_size': EVALUATION_SIZE,
              'input_preprocessing': INPUT_PREPROCESSING,
              'seed': args.seed, 'epochs': args.epochs, 'counts': cache['counts'], 'source_sha256': source_hashes,
              'frozen_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.output/'frozen-selection.json', frozen)
    return frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'results/compact-cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--base-channels', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--dice-weight', type=float, default=1.)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=406)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--resume', action='store_true', help='Resume only a matching interrupted run, including optimizer state')
    result = train(parser.parse_args(argv))
    print(json.dumps({'status': result['status'], 'best_epoch': result['best_epoch'], 'threshold': result['threshold']}))


if __name__ == '__main__':
    main()
