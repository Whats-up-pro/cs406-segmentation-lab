"""Inference-only R50-ViT-B/16 TransUNet, adapted from Beckschen/TransUNet.

Apache-2.0; upstream source, license and modification notice are in
``third_party/TransUNet``. Module names intentionally match the supplied weights.
Training helpers and unrelated model variants were removed.
"""

from collections import OrderedDict
from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


class StdConv2d(nn.Conv2d):
    """The upstream ResNetV2 uses weight-standardized convolutions."""

    def forward(self, x):
        variance, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdim=True, unbiased=False)
        weight = (self.weight - mean) / torch.sqrt(variance + 1e-5)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class PreActBottleneck(nn.Module):
    def __init__(self, cin, cout, cmid, stride=1):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = StdConv2d(cin, cmid, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = StdConv2d(cmid, cmid, 3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = StdConv2d(cmid, cout, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = StdConv2d(cin, cout, 1, stride=stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = self.gn_proj(self.downsample(x)) if hasattr(self, "downsample") else x
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        return self.relu(residual + self.gn3(self.conv3(y)))


class ResNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.root = nn.Sequential(OrderedDict([
            ("conv", StdConv2d(3, 64, 7, stride=2, padding=3, bias=False)),
            ("gn", nn.GroupNorm(32, 64, eps=1e-6)),
            ("relu", nn.ReLU(inplace=True)),
        ]))
        blocks = []
        for index, (count, cin, cout, cmid) in enumerate(((3, 64, 256, 64), (4, 256, 512, 128), (9, 512, 1024, 256))):
            units = [("unit1", PreActBottleneck(cin, cout, cmid, stride=1 if index == 0 else 2))]
            units += [(f"unit{i}", PreActBottleneck(cout, cout, cmid)) for i in range(2, count + 1)]
            blocks.append((f"block{index + 1}", nn.Sequential(OrderedDict(units))))
        self.body = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        input_size = x.shape[-1]
        x = self.root(x)
        features = [x]
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=0)
        for index in range(2):
            x = self.body[index](x)
            right_size = int(input_size / 4 / (index + 1))
            pad = right_size - x.shape[-1]
            # Preserve upstream bottom/right skip padding, without padding x itself.
            features.append(F.pad(x, (0, pad, 0, pad)) if pad else x)
        x = self.body[-1](x)
        return x, features[::-1]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

    def forward(self, x):
        batch, tokens, _ = x.shape
        query, key, value = [projection(x).view(batch, tokens, 12, 64).transpose(1, 2)
                             for projection in (self.query, self.key, self.value)]
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(64)
        attention = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch, tokens, 768)
        return self.proj_dropout(self.out(context))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Attention()

    def forward(self, x):
        x = x + self.attn(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([Block() for _ in range(12)])
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)

    def forward(self, x):
        for block in self.layer:
            x = block(x)
        return self.encoder_norm(x)


class Embeddings(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.hybrid_model = ResNetV2()
        self.patch_embeddings = nn.Conv2d(1024, 768, kernel_size=1, stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, (image_size // 16) ** 2, 768))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x).flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings), features


class Transformer(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.embeddings = Embeddings(image_size)
        self.encoder = Encoder()

    def forward(self, x):
        x, features = self.embeddings(x)
        return self.encoder(x), features


class Conv2dReLU(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cout, skip_channels):
        super().__init__()
        self.conv1 = Conv2dReLU(cin + skip_channels, cout)
        self.conv2 = Conv2dReLU(cout, cout)
        # This is align_corners=True, as in the official decoder.
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_more = Conv2dReLU(768, 512)
        self.blocks = nn.ModuleList([DecoderBlock(cin, cout, skip) for cin, cout, skip in
                                     ((512, 256, 512), (256, 128, 256), (128, 64, 64), (64, 16, 0))])

    def forward(self, x, features):
        batch, tokens, hidden = x.shape
        side = math.isqrt(tokens)
        x = self.conv_more(x.transpose(1, 2).contiguous().view(batch, hidden, side, side))
        for index, block in enumerate(self.blocks):
            x = block(x, features[index] if index < 3 else None)
        return x


class TransUNet(nn.Module):
    """The official hybrid architecture with a single binary segmentation logit."""

    def __init__(self, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.transformer = Transformer(image_size)
        self.decoder = DecoderCup()
        self.segmentation_head = nn.Sequential(nn.Conv2d(16, 1, 3, padding=1), nn.Identity())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected N x 3 x {self.image_size} x {self.image_size} RGB input")
        x, features = self.transformer(x)
        return self.segmentation_head(self.decoder(x, features))


def _unprefix_state_dict(state):
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint must be a nonempty tensor state dictionary")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("Checkpoint must contain only named tensor weights")
    prefixed = [key.startswith("base_model.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Checkpoint has mixed base_model prefixes")
    return {key.removeprefix("base_model."): value for key, value in state.items()}


def infer_checkpoint_config(state):
    """Infer square resolution from positions; validate the supported binary variant."""
    state = _unprefix_state_dict(state)
    positions = state.get("transformer.embeddings.position_embeddings")
    if positions is None or positions.ndim != 3 or positions.shape[0] != 1 or positions.shape[2] != 768:
        raise ValueError("Expected position embeddings with shape (1, square_tokens, 768)")
    tokens = positions.shape[1]
    grid = math.isqrt(tokens)
    if grid * grid != tokens or grid < 2:
        raise ValueError("Position embedding tokens must form a square grid")
    head = state.get("segmentation_head.0.weight")
    if head is None or tuple(head.shape) != (1, 16, 3, 3):
        raise ValueError("Expected one binary segmentation output channel")
    patch = state.get("transformer.embeddings.patch_embeddings.weight")
    if patch is None or tuple(patch.shape) != (768, 1024, 1, 1):
        raise ValueError("Expected R50-ViT-B_16 hybrid patch projection")
    return {"image_size": grid * 16, "position_tokens": tokens, "output_channels": 1}


def load_transunet_state_dict(state):
    """Load every tensor strictly; meta construction avoids a second 421 MB copy."""
    config = infer_checkpoint_config(state)
    with torch.device("meta"):
        model = TransUNet(image_size=config["image_size"])
    model.load_state_dict(_unprefix_state_dict(state), strict=True, assign=True)
    return model.eval()
