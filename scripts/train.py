"""Reproducible TransUNet fine-tuning on imagesTrain/Valid and exact-stem TIFFs.

This starts a NEW fine-tuning experiment from supplied weights. The original
checkpoint's training history and normalization were not supplied. No test
split is indexed, opened, scored or used for model selection by this command.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from backend.inference import preprocess_image
from backend.transunet import load_transunet_state_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FIELDS = ["epoch", "train_loss", "train_dice", "valid_loss", "valid_dice",
                  "train_samples", "valid_samples", "train_batches", "valid_batches", "seconds", "best_so_far"]


@dataclass(frozen=True)
class TrainingPair:
    image: Path
    mask: Path


def discover_pairs(data_dir: Path, split: str) -> list[TrainingPair]:
    """Read only the selected train/valid directories; labels match exact stems."""
    if split not in {"train", "valid"}:
        raise ValueError("Fine-tuning supports only train and valid splits")
    suffix = {"train": "Train", "valid": "Valid"}[split]
    image_dir, label_dir = data_dir / f"images{suffix}", data_dir / f"labels{suffix}"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"Missing {split} image/label directories in {data_dir}")
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = {}
    for path in sorted(label_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}:
            if path.stem in labels:
                raise ValueError(f"Ambiguous exact-stem TIFF label: {path.stem}")
            labels[path.stem] = path
    if not images:
        raise ValueError(f"No images found in {image_dir}")
    if len({path.stem for path in images}) != len(images):
        raise ValueError(f"Duplicate image stems in {image_dir}")
    missing = [path.name for path in images if path.stem not in labels]
    if missing:
        raise ValueError(f"Missing exact-stem TIFF labels for {len(missing)} images; first: {missing[0]}")
    return [TrainingPair(path, labels[path.stem]) for path in images]


def augment_pair(image: Image.Image, mask: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
    """Apply the same seeded horizontal/vertical flips and right-angle rotation."""
    transforms = []
    if rng.random() < 0.5:
        transforms.append(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        transforms.append(Image.Transpose.FLIP_TOP_BOTTOM)
    turns = rng.randrange(4)
    if turns:
        transforms.append({1: Image.Transpose.ROTATE_90, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_270}[turns])
    for transform in transforms:
        image, mask = image.transpose(transform), mask.transpose(transform)
    return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, pairs, image_size: int, normalization: str, augment: bool, seed: int):
        self.pairs = pairs
        self.image_size, self.normalization = image_size, normalization
        self.augment, self.seed, self.epoch = augment, seed, 0

    def __len__(self):
        return len(self.pairs)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index):
        pair = self.pairs[index]
        with Image.open(pair.image) as source:
            image = source.convert("RGB")
        with Image.open(pair.mask) as source:
            if source.size != image.size:
                raise ValueError(f"Image/label dimensions differ: {pair.image.name}")
            values = np.asarray(source)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f"Expected a finite 2-D label: {pair.mask.name}")
            mask = Image.fromarray((values != 0).astype(np.uint8) * 255)
        if self.augment:
            rng = random.Random((self.seed << 32) + self.epoch * 1_000_003 + index)
            image, mask = augment_pair(image, mask, rng)
        pixels = preprocess_image(image, self.image_size, self.normalization)[0]
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        target = torch.from_numpy((np.asarray(mask) != 0).astype(np.float32)).unsqueeze(0)
        return pixels, target


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unweighted BCEWithLogits + per-image soft Dice loss (smoothing 1e-6)."""
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("Expected matching N x 1 x H x W binary logits and labels")
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * target).sum(dimensions)
    denominator = probabilities.sum(dimensions) + target.sum(dimensions)
    soft_dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - soft_dice).mean()


def _hard_dice(logits, target):
    prediction = logits.detach() >= 0  # sigmoid(logits) >= 0.5
    truth = target.bool()
    dimensions = (1, 2, 3)
    intersection = (prediction & truth).sum(dimensions).float()
    total = prediction.sum(dimensions) + truth.sum(dimensions)
    return torch.where(total == 0, torch.ones_like(intersection), 2 * intersection / total.clamp_min(1))


def configure_trainable_parameters(model, mode: str) -> int:
    if mode not in {"all", "head"}:
        raise ValueError("Trainable parameters must be all or head")
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "head":
        for parameter in model.segmentation_head.parameters():
            parameter.requires_grad_(True)
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def run_epoch(model, loader, device, optimizer=None, max_batches=None, head_only=False) -> dict:
    """Mean per-image loss/Dice; optimizer presence selects train vs validation."""
    training = optimizer is not None
    model.train(training)
    if training and head_only:
        # Keep frozen dropout and BatchNorm running statistics fixed as well.
        model.eval()
        model.segmentation_head.train()
    loss_sum, dice_sum, samples, batches = 0.0, 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite training/validation loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
                optimizer.step()
            count = images.shape[0]
            loss_sum += loss.item() * count
            dice_sum += _hard_dice(logits, targets).sum().item()
            samples += count
            batches += 1
    if not samples:
        raise ValueError("Epoch did not process any samples")
    return {"loss": loss_sum / samples, "dice": dice_sum / samples, "samples": samples, "batches": batches}


def _same_path(first: Path, second: Path) -> bool:
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def prepare_output(checkpoint: Path, output: Path) -> None:
    if _same_path(checkpoint, output) or _same_path(checkpoint, output / "best_model.pth"):
        raise ValueError("Output would overwrite the input checkpoint; choose a separate run directory")
    if output.exists() and not output.is_dir():
        raise ValueError("--output must be a separate run directory")
    for name in ("config.json", "history.json", "history.csv", "best_model.pth", "best_model.pth.tmp"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing run artifact: {output / name}")
    output.mkdir(parents=True, exist_ok=True)


def save_best_checkpoint(model, destination: Path, source: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if _same_path(destination, source) or _same_path(temporary, source):
        raise ValueError("Saving would overwrite the input checkpoint")
    state = {"base_model." + name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, temporary)
    temporary.replace(destination)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_history(output, history):
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(args) -> dict:
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {checkpoint}")
    prepare_output(checkpoint, output)
    pairs = {split: discover_pairs(args.data_dir.resolve(), split) for split in ("train", "valid")}
    if {pair.image.stem for pair in pairs["train"]} & {pair.image.stem for pair in pairs["valid"]}:
        raise ValueError("Train and validation image IDs overlap")
    if {pair.image.resolve() for pair in pairs["train"]} & {pair.image.resolve() for pair in pairs["valid"]}:
        raise ValueError("Train and validation image paths overlap")
    seed_everything(args.seed)
    torch.set_num_threads(min(args.cpu_threads, os.cpu_count() or 1))
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu")
    config = {
        "experiment": "New fine-tuning from supplied checkpoint; original training history unknown",
        "started_utc": datetime.now(timezone.utc).isoformat(), "status": "initializing",
        "parameters": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "input_checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": _sha256(checkpoint)},
        "splits": {split: {"count": len(items), "pairs": [{"image": str(pair.image.relative_to(args.data_dir.resolve())), "mask": str(pair.mask.relative_to(args.data_dir.resolve()))} for pair in items]} for split, items in pairs.items()},
        "protocol": {"loss": "BCEWithLogits + (1 - mean per-image softDice), smoothing=1e-6",
                     "optimizer": "AdamW, weight_decay=0.01, gradient norm clip=1.0",
                     "best_checkpoint_selection": "Highest macro validation binary Dice at threshold 0.5; first epoch wins ties",
                     "augmentation": "Train only: independent horizontal/vertical flips p=0.5, uniform 0/90/180/270-degree rotation, paired with labels",
                     "mask_preprocessing": "Nonzero to binary, then nearest-neighbor resize to checkpoint input size",
                     "test_split_used": False, "training_normalization_of_input_checkpoint_known": False,
                     "limited_batches": args.max_train_batches is not None or args.max_valid_batches is not None,
                     "data_loader": "Seeded training shuffle, validation filename order, num_workers=0",
                     "trainable": args.trainable,
                     "head_only_mode": "Only segmentation_head weights change; backbone/decoder stay in eval mode" if args.trainable == "head" else None,
                     "deterministic_algorithms": True},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version,
                        "device": device, "cpu_threads": torch.get_num_threads(),
                        "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
                        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "source_sha256": {name: _sha256(PROJECT_ROOT / name) for name in ("scripts/train.py", "backend/transunet.py", "backend/inference.py")},
        "history": [], "best_epoch": None, "best_valid_dice": None, "checkpoint_saved": False,
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    history, best_dice = [], -1.0
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = load_transunet_state_dict(state).to(device)
        del state
        config["model_input_size"] = model.image_size
        config["model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
        config["trainable_parameters"] = configure_trainable_parameters(model, args.trainable)
        datasets = {split: SegmentationDataset(items, model.image_size, args.normalization, split == "train", args.seed)
                    for split, items in pairs.items()}
        generator = torch.Generator().manual_seed(args.seed)
        loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", num_workers=0,
                                    generator=generator if split == "train" else None)
                   for split, dataset in datasets.items()}
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate, weight_decay=0.01)
        config["status"] = "running"
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            datasets["train"].set_epoch(epoch - 1)
            training = run_epoch(model, loaders["train"], device, optimizer, args.max_train_batches, head_only=args.trainable == "head")
            validation = run_epoch(model, loaders["valid"], device, max_batches=args.max_valid_batches)
            improved = validation["dice"] > best_dice
            if improved:
                best_dice = validation["dice"]
                config["best_epoch"], config["best_valid_dice"] = epoch, best_dice
                if not args.no_save:
                    save_best_checkpoint(model, output / "best_model.pth", checkpoint)
                    config["checkpoint_saved"] = True
            row = {"epoch": epoch, "train_loss": training["loss"], "train_dice": training["dice"],
                   "valid_loss": validation["loss"], "valid_dice": validation["dice"],
                   "train_samples": training["samples"], "valid_samples": validation["samples"],
                   "train_batches": training["batches"], "valid_batches": validation["batches"],
                   "seconds": time.perf_counter() - epoch_start, "best_so_far": improved}
            history.append(row)
            _write_history(output, history)
            config["history"] = history
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(row), flush=True)
        config["status"] = "complete"
    except Exception as error:
        config["status"] = "failed"
        config["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        config["wall_seconds"] = time.perf_counter() - started
        config["finished_utc"] = datetime.now(timezone.utc).isoformat()
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "data/TransUNet_R50_ViT_B_16_all_best_model.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/transunet_finetune")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--normalization", choices=("unit", "imagenet"), default="unit")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trainable", choices=("all", "head"), default="all", help="Full fine-tuning, or freeze encoder/decoder and train only the output head")
    parser.add_argument("--seed", type=int, default=406)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--no-save", action="store_true", help="Run real optimization/validation but do not write model weights")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or not 1 <= args.cpu_threads <= 8:
        parser.error("--epochs and --batch-size must be positive; --cpu-threads must be between 1 and 8")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if any(value is not None and value < 1 for value in (args.max_train_batches, args.max_valid_batches)):
        parser.error("Batch limits must be positive")
    config = train(args)
    print(json.dumps({"output": str(args.output.resolve()), "status": config["status"],
                      "best_epoch": config["best_epoch"], "best_valid_dice": config["best_valid_dice"],
                      "checkpoint_saved": config["checkpoint_saved"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
