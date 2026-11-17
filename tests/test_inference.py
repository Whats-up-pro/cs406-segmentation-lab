"""Small deterministic tests; opt in to the supplied 421 MB checkpoint smoke test."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
from PIL import Image
import pytest


def test_inventory_does_not_import_torch_or_load_checkpoint():
    completed = subprocess.run(
        [sys.executable, "-c", "import json, sys; from backend.inference import get_model_status; "
         "s=get_model_status(); print(json.dumps({'status':s, 'torch_imported':'torch' in sys.modules}))"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["torch_imported"] is False
    assert result["status"]["loaded"] is False
    assert result["status"]["architecture"] == "R50-ViT-B_16"
    assert result["status"]["normalization_options"] == ["unit", "imagenet"]


def test_inventory_is_safe_while_another_thread_is_importing_torch(monkeypatch):
    from backend.inference import get_model_status

    spec = importlib.util.find_spec("torch")
    if spec is None:
        pytest.skip("PyTorch is not installed")
    partial_module = ModuleType("torch")
    partial_module.__spec__ = spec
    monkeypatch.setitem(sys.modules, "torch", partial_module)
    assert get_model_status()["runtime"]["cuda_available"] is None


def test_rgb_preprocessing_scales_channels_and_resizes():
    pytest.importorskip("torch")
    from backend.inference import preprocess_image

    tensor = preprocess_image(Image.new("RGB", (23, 17), (255, 128, 0)), 224, "unit")
    assert tuple(tensor.shape) == (1, 3, 224, 224)
    np.testing.assert_allclose(tensor.numpy()[0, :, 100, 100], [1.0, 128 / 255, 0], atol=1e-6)
    assert str(tensor.dtype) == "torch.float32"


def test_grayscale_and_imagenet_preprocessing():
    pytest.importorskip("torch")
    from backend.inference import preprocess_image

    tensor = preprocess_image(Image.new("L", (9, 7), 255), 32, "imagenet")
    expected = (np.ones(3) - np.array([0.485, 0.456, 0.406])) / [0.229, 0.224, 0.225]
    np.testing.assert_allclose(tensor.numpy()[0, :, 0, 0], expected, atol=1e-6)


def test_unknown_normalization_rejected_before_loading():
    from backend.inference import predict_transunet

    with pytest.raises(ValueError, match="normalization"):
        predict_transunet(Image.new("RGB", (3, 5)), normalization="unknown")


def test_binary_logits_restore_original_non_square_dimensions():
    torch = pytest.importorskip("torch")
    from backend.inference import probabilities_from_logits

    result = probabilities_from_logits(torch.zeros(1, 1, 4, 4), (19, 11))
    assert result.shape == (11, 19)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.5, atol=1e-7)


def test_sigmoid_precedes_bilinear_probability_resampling():
    torch = pytest.importorskip("torch")
    from backend.inference import probabilities_from_logits

    logits = torch.tensor([[[[-3.0, 1.0], [-1.0, 3.0]]]])
    result = probabilities_from_logits(logits, (5, 3))
    expected = torch.nn.functional.interpolate(torch.sigmoid(logits), size=(3, 5), mode="bilinear", align_corners=False)
    np.testing.assert_allclose(result, expected.numpy()[0, 0])
    assert 0 <= result.min() <= result.max() <= 1


def test_resampling_extreme_probabilities_stays_in_unit_interval():
    torch = pytest.importorskip("torch")
    from backend.inference import probabilities_from_logits

    result = probabilities_from_logits(torch.full((1, 1, 4, 4), 100.0), (19, 11))
    assert result.min() >= 0
    assert result.max() <= 1


@pytest.mark.parametrize("shape", [(1, 2, 4, 4), (4, 4), (2, 1, 4, 4)])
def test_postprocessing_rejects_non_binary_output(shape):
    torch = pytest.importorskip("torch")
    from backend.inference import probabilities_from_logits

    with pytest.raises(ValueError, match="1, 1"):
        probabilities_from_logits(torch.zeros(*shape), (7, 5))


def test_postprocessing_rejects_nonfinite_logits():
    torch = pytest.importorskip("torch")
    from backend.inference import probabilities_from_logits

    with pytest.raises(ValueError, match="finite"):
        probabilities_from_logits(torch.full((1, 1, 2, 2), float("nan")), (7, 5))


def test_checkpoint_shape_inference_and_strict_loading():
    torch = pytest.importorskip("torch")
    from backend.transunet import TransUNet, infer_checkpoint_config, load_transunet_state_dict

    # Meta tensors exercise every key and shape without allocating 421 MB of weights.
    with torch.device("meta"):
        expected = TransUNet(image_size=224)
    state = {"base_model." + key: value for key, value in expected.state_dict().items()}
    config = infer_checkpoint_config(state)
    assert config["image_size"] == 224
    assert config["position_tokens"] == 196
    assert config["output_channels"] == 1
    loaded = load_transunet_state_dict(state)
    assert len(loaded.state_dict()) == 409
    with pytest.raises(RuntimeError, match="Missing key"):
        load_transunet_state_dict({key: value for key, value in state.items() if not key.endswith("segmentation_head.0.bias")})
    state["base_model.decoder.conv_more.0.weight"] = torch.empty(511, 768, 3, 3, device="meta")
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_transunet_state_dict(state)


def test_position_embeddings_must_form_square_grid():
    torch = pytest.importorskip("torch")
    from backend.transunet import TransUNet, infer_checkpoint_config

    with torch.device("meta"):
        state = TransUNet().state_dict()
    state["transformer.embeddings.position_embeddings"] = torch.empty(1, 195, 768, device="meta")
    with pytest.raises(ValueError, match="square"):
        infer_checkpoint_config(state)


@pytest.mark.skipif(os.environ.get("RUN_MODEL_SMOKE") != "1", reason="Set RUN_MODEL_SMOKE=1 to run the supplied checkpoint")
def test_real_checkpoint_predicts_finite_probabilities():
    pytest.importorskip("torch")
    from backend.inference import get_model_status, predict_transunet

    image_path = next((Path(__file__).resolve().parents[1] / "data" / "imagesValid").glob("*.jpg"))
    with Image.open(image_path) as image:
        result = predict_transunet(image, normalization="unit")
        assert result.shape == (image.height, image.width)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    assert get_model_status()["loaded"] is True
