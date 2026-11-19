"""Compact U-Net trained from scratch; no supplied TransUNet parameters."""
from functools import lru_cache
import math
import os
from pathlib import Path
from threading import RLock

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'compact-unet-v1'
INPUT_PREPROCESSING = 'rgb-bilinear-256-then-model-v2'
LEGACY_PREPROCESSING = 'native-rgb-to-model-v1'
_LOCK = RLock()


def resize_compact_input(image, size, preprocessing=INPUT_PREPROCESSING):
    """One RGB pathway shared by training, validation and inference."""
    rgb = image.convert('RGB')
    if preprocessing == INPUT_PREPROCESSING:
        rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
    elif preprocessing != LEGACY_PREPROCESSING:
        raise ValueError('Unsupported compact RGB preprocessing recipe')
    return rgb.resize((size, size), Image.Resampling.BILINEAR)


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        groups = math.gcd(4, output_channels)
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
        )


class CompactUNet(nn.Module):
    def __init__(self, base_channels=12):
        super().__init__()
        if not isinstance(base_channels, int) or base_channels < 4:
            raise ValueError('base_channels must be an integer >=4')
        self.base_channels = base_channels
        c = base_channels
        self.encoder1, self.encoder2 = ConvBlock(3, c), ConvBlock(c, 2*c)
        self.encoder3, self.bottleneck = ConvBlock(2*c, 4*c), ConvBlock(4*c, 8*c)
        self.decoder3, self.decoder2 = ConvBlock(12*c, 4*c), ConvBlock(6*c, 2*c)
        self.decoder1, self.head = ConvBlock(3*c, c), nn.Conv2d(c, 1, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 8:
            raise ValueError('Expected N x 3 x H x W, with H and W >=8')
        a = self.encoder1(image)
        b = self.encoder2(F.max_pool2d(a, 2))
        c = self.encoder3(F.max_pool2d(b, 2))
        value = self.bottleneck(F.max_pool2d(c, 2))
        for skip, block in [(c, self.decoder3), (b, self.decoder2), (a, self.decoder1)]:
            value = F.interpolate(value, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            value = block(torch.cat([value, skip], dim=1))
        return self.head(value)


def binary_loss(logits, target, dice_weight=1.0):
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected matching N x 1 x H x W logits and binary labels')
    if not math.isfinite(dice_weight) or dice_weight < 0:
        raise ValueError('dice_weight must be finite and non-negative')
    bce = F.binary_cross_entropy_with_logits(logits, target)
    if dice_weight == 0:
        return bce
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum((1, 2, 3))
    total = probabilities.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return bce + dice_weight * (1 - (2*intersection + 1e-6)/(total + 1e-6)).mean()


def save_compact_checkpoint(model, path, input_size=128, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'format': FORMAT, 'base_channels': model.base_channels,
                'input_preprocessing': INPUT_PREPROCESSING,
                'input_size': int(input_size), 'state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
                'metadata': metadata or {}}, temporary)
    os.replace(temporary, path)


@lru_cache(maxsize=3)
def _load_checkpoint(path_string, mtime_ns, size, device):
    payload = torch.load(path_string, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise ValueError('Unsupported compact checkpoint format')
    input_size = payload['input_size']
    if not isinstance(input_size, int) or not 8 <= input_size <= 2048:
        raise ValueError('Invalid compact model input size')
    model = CompactUNet(payload['base_channels'])
    model.load_state_dict(payload['state_dict'], strict=True)
    metadata = {**payload.get('metadata', {}), 'input_preprocessing': payload.get('input_preprocessing', LEGACY_PREPROCESSING)}
    if metadata['input_preprocessing'] not in {INPUT_PREPROCESSING, LEGACY_PREPROCESSING}:
        raise ValueError('Unsupported compact RGB preprocessing recipe')
    return model.to(device).eval(), input_size, metadata


def load_compact_checkpoint(path, device='cpu'):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _load_checkpoint(str(path), stat.st_mtime_ns, stat.st_size, device)


def predict_compact(image, checkpoint=None, device='cpu'):
    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    default = Path(__file__).resolve().parents[1] / 'results/compact/bce_dice_aug_s406/best.pth'
    path = Path(checkpoint or os.environ.get('COMPACT_CHECKPOINT', str(default)))
    with _LOCK:
        model, size, metadata = load_compact_checkpoint(path, device=device)
        pixels = np.asarray(resize_compact_input(image, size, metadata['input_preprocessing']), dtype=np.float32)/255.0
        tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).sigmoid()
            restored = F.interpolate(probabilities, size=(image.height, image.width), mode='bilinear', align_corners=False)
        return restored[0, 0].clamp(0, 1).cpu().numpy().astype(np.float32, copy=False)
