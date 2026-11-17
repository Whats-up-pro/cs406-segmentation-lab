"""Training contract tests use tiny tensors, never the 105M-parameter optimizer."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")


def make_split(root, suffix, name="sample"):
    (root / f"images{suffix}").mkdir(parents=True)
    (root / f"labels{suffix}").mkdir(parents=True)
    pixels = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2)).save(root / f"images{suffix}" / f"{name}.png")
    Image.fromarray(pixels).save(root / f"labels{suffix}" / f"{name}.tiff")


def test_pair_discovery_uses_only_requested_train_or_valid_directories(tmp_path, monkeypatch):
    from scripts.train import discover_pairs

    make_split(tmp_path, "Train", "training")
    make_split(tmp_path, "Valid", "validation")
    make_split(tmp_path, "Test", "never_read")
    original = Path.iterdir
    accessed = []

    def checked_iterdir(path):
        accessed.append(path.name)
        assert path.name not in {"imagesTest", "labelsTest"}
        return original(path)

    monkeypatch.setattr(Path, "iterdir", checked_iterdir)
    assert [pair.image.stem for pair in discover_pairs(tmp_path, "train")] == ["training"]
    assert [pair.image.stem for pair in discover_pairs(tmp_path, "valid")] == ["validation"]
    with pytest.raises(ValueError, match="train.*valid"):
        discover_pairs(tmp_path, "test")
    assert set(accessed) == {"imagesTrain", "labelsTrain", "imagesValid", "labelsValid"}


def test_training_pairs_require_exact_stem_tiff(tmp_path):
    from scripts.train import discover_pairs

    make_split(tmp_path, "Train")
    (tmp_path / "labelsTrain" / "sample.tiff").rename(tmp_path / "labelsTrain" / "sample_mask_ann2.tiff")
    with pytest.raises(ValueError, match="exact"):
        discover_pairs(tmp_path, "train")


def test_sample_preprocessing_uses_binary_nearest_labels(tmp_path):
    from scripts.train import SegmentationDataset, discover_pairs

    make_split(tmp_path, "Valid")
    dataset = SegmentationDataset(discover_pairs(tmp_path, "valid"), image_size=4, normalization="unit", augment=False, seed=7)
    image, target = dataset[0]
    assert image.shape == (3, 4, 4)
    assert target.shape == (1, 4, 4)
    assert image.dtype == target.dtype == torch.float32
    assert image.min() >= 0 and image.max() <= 1
    expected = np.repeat(np.repeat([[0, 1], [1, 0]], 2, axis=0), 2, axis=1)
    np.testing.assert_array_equal(target.numpy()[0], expected)


def test_pair_augmentation_is_aligned_and_seeded():
    from scripts.train import augment_pair

    pixels = np.array([[0, 255, 0], [255, 0, 255]], dtype=np.uint8)
    image = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2))
    mask = Image.fromarray(pixels)
    transformed, label = augment_pair(image, mask, random.Random(19))
    repeated_image, repeated_label = augment_pair(image, mask, random.Random(19))
    np.testing.assert_array_equal(np.asarray(transformed)[:, :, 0], np.asarray(label))
    np.testing.assert_array_equal(np.asarray(transformed), np.asarray(repeated_image))
    np.testing.assert_array_equal(np.asarray(label), np.asarray(repeated_label))


def test_dataset_rejects_mismatched_original_label_size(tmp_path):
    from scripts.train import SegmentationDataset, discover_pairs

    make_split(tmp_path, "Valid")
    Image.new("L", (5, 3)).save(tmp_path / "labelsValid" / "sample.tiff")
    dataset = SegmentationDataset(discover_pairs(tmp_path, "valid"), 4, "unit", False, 9)
    with pytest.raises(ValueError, match="dimensions"):
        dataset[0]


def test_bce_softdice_loss_is_finite_and_backpropagates():
    from scripts.train import segmentation_loss

    logits = torch.tensor([[[[-3.0, 2.0], [1.0, -2.0]]]], requires_grad=True)
    target = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    loss = segmentation_loss(logits, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_training_epoch_updates_weights_and_validation_does_not():
    from scripts.train import run_epoch

    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    image = torch.ones(2, 3, 4, 4)
    target = torch.zeros(2, 1, 4, 4)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(image, target), batch_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.weight.detach().clone()
    metrics = run_epoch(model, loader, "cpu", optimizer=optimizer, max_batches=1)
    assert not torch.equal(model.weight, before)
    assert metrics["samples"] == metrics["batches"] == 1
    assert np.isfinite(metrics["loss"]) and 0 <= metrics["dice"] <= 1
    before_validation = model.weight.detach().clone()
    valid = run_epoch(model, loader, "cpu", max_batches=1)
    assert torch.equal(model.weight, before_validation)
    assert valid["samples"] == 1
    assert model.training is False


def test_head_only_training_freezes_backbone_parameters_and_running_statistics():
    from scripts.train import configure_trainable_parameters, run_epoch

    class TinySegmenter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2))
            self.segmentation_head = torch.nn.Conv2d(2, 1, 1)

        def forward(self, image):
            return self.segmentation_head(self.transformer(image))

    model = TinySegmenter()
    configure_trainable_parameters(model, "head")
    assert all(not parameter.requires_grad for parameter in model.transformer.parameters())
    assert all(parameter.requires_grad for parameter in model.segmentation_head.parameters())
    before_mean = model.transformer[1].running_mean.clone()
    before_head = model.segmentation_head.weight.detach().clone()
    loader = [(torch.ones(1, 3, 4, 4), torch.zeros(1, 1, 4, 4))]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    run_epoch(model, loader, "cpu", optimizer=optimizer, head_only=True)
    assert torch.equal(model.transformer[1].running_mean, before_mean)
    assert not torch.equal(model.segmentation_head.weight, before_head)


def test_output_cannot_overwrite_supplied_checkpoint(tmp_path):
    from scripts.train import prepare_output, save_best_checkpoint

    source = tmp_path / "supplied.pth"
    source.write_bytes(b"keep original weights")
    with pytest.raises(ValueError, match="overwrite"):
        prepare_output(source, source)
    with pytest.raises(ValueError, match="overwrite"):
        save_best_checkpoint(torch.nn.Conv2d(3, 1, 1), source, source)
    assert source.read_bytes() == b"keep original weights"


def test_saved_finetuned_checkpoint_keeps_inference_prefix(tmp_path):
    from scripts.train import prepare_output, save_best_checkpoint

    source = tmp_path / "supplied.pth"
    source.write_bytes(b"keep original weights")
    output = tmp_path / "run"
    prepare_output(source, output)
    model = torch.nn.Conv2d(3, 1, 1)
    save_best_checkpoint(model, output / "best_model.pth", source)
    state = torch.load(output / "best_model.pth", map_location="cpu", weights_only=True)
    assert set(state) == {"base_model.weight", "base_model.bias"}
    assert torch.equal(state["base_model.weight"], model.weight)
    assert source.read_bytes() == b"keep original weights"
    with pytest.raises(FileExistsError, match="existing"):
        prepare_output(source, output)
