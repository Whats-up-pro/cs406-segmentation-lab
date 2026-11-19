from argparse import Namespace
import json

import numpy as np
from PIL import Image
import pytest


def test_completed_suite_run_is_reused_only_when_requested_protocol_matches(tmp_path):
    from scripts.run_compact_experiments import run_suite

    data = tmp_path / 'data'
    for split, name in [('Train', 'a'), ('Valid', 'b')]:
        (data / f'images{split}').mkdir(parents=True)
        (data / f'labels{split}').mkdir(parents=True)
        mask = np.zeros((16, 16), dtype=np.uint8); mask[4:12, 4:12] = 1
        Image.fromarray(np.repeat((mask*255)[:, :, None], 3, axis=2)).save(data/f'images{split}'/f'{name}.png')
        Image.fromarray(mask).save(data/f'labels{split}'/f'{name}.tiff')
    args = Namespace(data_dir=data, cache_dir=tmp_path/'cache', output=tmp_path/'suite', device='cpu',
                     epochs=1, cpu_threads=2, image_size=32, base_channels=4, batch_size=1,
                     cases=['bce_dice_aug_s406'])
    first = run_suite(args)
    path = args.output/'bce_dice_aug_s406'/'best.pth'
    before = path.read_bytes()
    second = run_suite(args)
    assert first['status'] == second['status'] == 'complete'
    assert second['runs'][0]['reused'] is True
    assert path.read_bytes() == before
    args.epochs = 2
    with pytest.raises(ValueError, match='protocol'):
        run_suite(args)


def test_suite_rejects_unfrozen_or_tampered_existing_run_instead_of_claiming_complete(tmp_path):
    from scripts.run_compact_experiments import verify_frozen_run

    path = tmp_path / 'run'; path.mkdir()
    (path/'frozen-selection.json').write_text(json.dumps({'status': 'training'}))
    with pytest.raises(ValueError, match='frozen'):
        verify_frozen_run(path)
