from argparse import Namespace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip('torch')


def make_pair(root, split, name, offset=0):
    image_dir = root / ('images' + split)
    mask_dir = root / ('labels' + split)
    image_dir.mkdir(exist_ok=True, parents=True)
    mask_dir.mkdir(exist_ok=True, parents=True)
    label = np.zeros((17, 23), dtype=np.uint8)
    label[3:13, 5 + offset:15 + offset] = 1
    pixels = np.repeat((label * 255)[:, :, None], 3, axis=2)
    Image.fromarray(pixels).save(image_dir / (name + '.png'))
    Image.fromarray(label).save(mask_dir / (name + '.tiff'))


def test_cache_only_reads_train_valid_and_uses_binary_nearest_labels(tmp_path, monkeypatch):
    from scripts.train_compact import prepare_training_cache

    for split, name in [('Train', 'a'), ('Valid', 'b'), ('Test', 'never_read')]:
        make_pair(tmp_path / 'data', split, name)
    original = Path.iterdir

    def checked(path):
        assert path.name not in {'imagesTest', 'labelsTest'}
        return original(path)

    monkeypatch.setattr(Path, 'iterdir', checked)
    cache = prepare_training_cache(tmp_path / 'data', tmp_path / 'cache', image_size=32)
    assert cache['counts'] == {'train': 1, 'valid': 1}
    labels = np.load(tmp_path / 'cache' / 'train_masks.npy')
    assert set(np.unique(labels)) == {0, 1}
    assert labels.shape == (1, 1, 32, 32)


def test_training_cache_uses_the_declared_common_rgb_256_then_model_path(tmp_path):
    from scripts.train_compact import prepare_training_cache

    data = tmp_path/'data'
    make_pair(data, 'Train', 'a'); make_pair(data, 'Valid', 'b')
    values = ((np.indices((71, 103)).sum(axis=0) % 2)*255).astype(np.uint8)
    for split, name in [('Train','a'),('Valid','b')]:
        Image.fromarray(np.repeat(values[:,:,None],3,axis=2)).save(data/f'images{split}'/f'{name}.png')
        Image.fromarray((values!=0).astype(np.uint8)).save(data/f'labels{split}'/f'{name}.tiff')
    manifest = prepare_training_cache(data, tmp_path/'cache', image_size=32)
    expected = np.asarray(Image.fromarray(np.repeat(values[:,:,None],3,axis=2)).resize((256,256),Image.Resampling.BILINEAR).resize((32,32),Image.Resampling.BILINEAR)).transpose(2,0,1)
    np.testing.assert_array_equal(np.load(tmp_path/'cache/valid_images.npy')[0],expected)
    assert manifest['preprocessing'] == 'rgb-bilinear-256-then-model-v2'


def test_cached_pair_augmentation_is_seeded_and_spatially_aligned(tmp_path):
    from scripts.train_compact import CachedPairs, prepare_training_cache

    make_pair(tmp_path / 'data', 'Train', 'a')
    make_pair(tmp_path / 'data', 'Valid', 'b')
    prepare_training_cache(tmp_path / 'data', tmp_path / 'cache', image_size=32)
    dataset = CachedPairs(tmp_path / 'cache', 'train', augment=True, seed=406, color_jitter=False)
    image, target = dataset[0]
    repeated = dataset[0]
    assert torch.equal(image, repeated[0]) and torch.equal(target, repeated[1])
    np.testing.assert_array_equal(image.numpy()[0] > .5, target.numpy()[0] > .5)


def test_real_training_freezes_checkpoint_and_validation_threshold_before_any_test_read(tmp_path, monkeypatch):
    from scripts.train_compact import train

    for split, name in [('Train', 'a'), ('Valid', 'b'), ('Test', 'never_read')]:
        make_pair(tmp_path / 'data', split, name)
    original = Path.iterdir

    def checked(path):
        assert path.name not in {'imagesTest', 'labelsTest'}
        return original(path)

    monkeypatch.setattr(Path, 'iterdir', checked)
    args = Namespace(data_dir=tmp_path / 'data', cache_dir=tmp_path / 'cache', output=tmp_path / 'run',
                     image_size=32, base_channels=4, epochs=2, batch_size=1, learning_rate=.01,
                     weight_decay=.0001, dice_weight=1.0, augment=True, seed=406, cpu_threads=2)
    result = train(args)
    history = json.loads((args.output / 'history.json').read_text())
    frozen = json.loads((args.output / 'frozen-selection.json').read_text())
    assert len(history) == 2
    assert all(row['train_samples'] == row['valid_samples'] == 1 for row in history)
    assert all(row['updated_parameters'] for row in history)
    assert result['status'] == frozen['status'] == 'frozen'
    assert frozen['optimizer_splits'] == ['train']
    assert frozen['selection_split'] == 'valid'
    assert frozen['test_accessed'] is False
    assert len(frozen['checkpoint_sha256']) == 64
    assert .30 <= frozen['threshold'] <= .70
    with pytest.raises(FileExistsError):
        train(args)


def test_epoch_validation_does_not_update_parameters():
    from backend.compact_model import CompactUNet
    from scripts.train_compact import run_epoch

    torch.set_num_threads(2)
    model = CompactUNet(base_channels=4)
    loader = [(torch.rand(2, 3, 32, 32), torch.ones(2, 1, 32, 32))]
    before = [p.detach().clone() for p in model.parameters()]
    run_epoch(model, loader, optimizer=None, dice_weight=1)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    run_epoch(model, loader, optimizer=optimizer, dice_weight=1)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_interrupted_training_resumes_with_optimizer_and_matches_uninterrupted_weights(tmp_path, monkeypatch):
    import scripts.train_compact as training

    make_pair(tmp_path/'data', 'Train', 'a')
    make_pair(tmp_path/'data', 'Valid', 'b')
    args = Namespace(data_dir=tmp_path/'data', cache_dir=tmp_path/'cache', output=tmp_path/'interrupted',
                     image_size=32, base_channels=4, epochs=2, batch_size=1, learning_rate=.01,
                     weight_decay=.0001, dice_weight=1., augment=True, seed=406, cpu_threads=2, resume=False)
    original, calls = training.run_epoch, []

    def interrupt_second_training_epoch(*values, **options):
        if len(calls) == 2:
            raise RuntimeError('simulated interrupted worker')
        calls.append(True)
        return original(*values, **options)

    monkeypatch.setattr(training, 'run_epoch', interrupt_second_training_epoch)
    with pytest.raises(RuntimeError, match='interrupted worker'):
        training.train(args)
    assert (args.output/'last.pth').exists()
    monkeypatch.setattr(training, 'run_epoch', original)
    args.resume = True
    resumed = training.train(args)
    resumed_payload = torch.load(args.output/'best.pth', map_location='cpu', weights_only=True)
    assert resumed['status'] == 'frozen'
    assert [row['epoch'] for row in json.loads((args.output/'history.json').read_text())] == [1, 2]
    args.output, args.resume = tmp_path/'uninterrupted', False
    training.train(args)
    reference = torch.load(args.output/'best.pth', map_location='cpu', weights_only=True)
    for key in reference['state_dict']:
        torch.testing.assert_close(resumed_payload['state_dict'][key], reference['state_dict'][key], rtol=0, atol=0)
