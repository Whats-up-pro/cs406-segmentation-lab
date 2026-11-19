from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')


def test_compact_network_handles_non_square_odd_sizes_and_backpropagates():
    from backend.compact_model import CompactUNet

    torch.set_num_threads(2)
    model = CompactUNet(base_channels=4)
    logits = model(torch.rand(2, 3, 33, 47))
    assert logits.shape == (2, 1, 33, 47)
    assert sum(p.numel() for p in model.parameters()) < 100_000
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dice_term_adds_foreground_sensitive_loss_and_finite_gradients():
    from backend.compact_model import binary_loss

    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    labels = torch.zeros_like(logits)
    labels[0, :, 1:3, 2:4] = 1
    bce = binary_loss(logits, labels, dice_weight=0)
    combined = binary_loss(logits, labels, dice_weight=1)
    torch.testing.assert_close(bce, torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
    assert combined > bce
    combined.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_compact_checkpoint_load_restores_original_dimensions_without_labels(tmp_path):
    from backend.compact_model import CompactUNet, predict_compact, save_compact_checkpoint

    torch.manual_seed(406)
    model = CompactUNet(base_channels=4).eval()
    path = tmp_path / 'model.pth'
    save_compact_checkpoint(model, path, input_size=32, metadata={'seed': 406})
    image = Image.new('RGB', (47, 19), '#b58e7b')
    first = predict_compact(image, checkpoint=path)
    second = predict_compact(image, checkpoint=path)
    assert first.shape == (19, 47) and first.dtype == np.float32
    assert np.isfinite(first).all() and 0 <= first.min() <= first.max() <= 1
    np.testing.assert_array_equal(first, second)


def test_compact_predictor_rejects_incompatible_checkpoint(tmp_path):
    from backend.compact_model import predict_compact

    path = tmp_path / 'wrong.pth'
    torch.save({'format': 'unrelated', 'state_dict': {}}, path)
    with pytest.raises(ValueError, match='format'):
        predict_compact(Image.new('RGB', (8, 8)), checkpoint=path)
