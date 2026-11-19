from base64 import b64decode
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient


def upload():
    output = BytesIO(); Image.new('RGB', (47, 19), '#b58e7b').save(output, format='PNG')
    return output.getvalue()


def frozen_model(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from backend.compact_model import CompactUNet, save_compact_checkpoint

    torch.set_num_threads(2)
    model = CompactUNet(base_channels=4)
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.head.bias.fill_(.6)
    path = tmp_path/'model'/'best.pth'
    save_compact_checkpoint(model, path, input_size=32)
    frozen = {'status':'frozen', 'format':'compact-selection-v1', 'model':'compact_unet',
              'checkpoint':'best.pth', 'checkpoint_sha256':sha256(path.read_bytes()).hexdigest(),
              'threshold':.7, 'seed':406, 'epochs':2, 'best_epoch':1, 'counts':{'train':2,'valid':1},
              'optimizer_splits':['train'], 'selection_split':'valid', 'test_accessed':False}
    (path.parent/'frozen-selection.json').write_text(json.dumps(frozen))
    monkeypatch.setenv('COMPACT_CHECKPOINT', str(path))
    return path


def test_compact_upload_uses_frozen_threshold_and_returns_original_size_without_invented_scores(tmp_path, monkeypatch):
    from backend.main import create_app

    frozen_model(tmp_path, monkeypatch)
    with TestClient(create_app(tmp_path/'data')) as client:
        response = client.post('/api/predict?method=compact_unet', content=upload(), headers={'content-type':'image/png'})
        assert response.status_code == 200
        result = response.json()
        assert result['threshold'] == .7 and result['metrics'] is None
        image = Image.open(BytesIO(b64decode(result['mask'].split(',')[1])))
        assert image.size == (47,19) and np.asarray(image).max() == 0
        override = client.post('/api/predict?method=compact_unet&threshold=0.5&normalization=imagenet', content=upload(), headers={'content-type':'image/png'})
        assert override.status_code == 200
        assert override.json()['normalization'] == 'unit'
        image = Image.open(BytesIO(b64decode(override.json()['mask'].split(',')[1])))
        assert np.asarray(image).min() == 255


def test_compact_unfrozen_or_tampered_checkpoint_is_unavailable(tmp_path, monkeypatch):
    from backend.main import create_app

    path = frozen_model(tmp_path, monkeypatch)
    path.write_bytes(path.read_bytes()+b'corruption')
    with TestClient(create_app(tmp_path/'data')) as client:
        result = client.post('/api/predict?method=compact_unet', content=upload(), headers={'content-type':'image/png'})
        assert result.status_code == 503
        assert 'mask' not in result.json()


def test_data_results_environment_and_nested_result_download_work_together(tmp_path, monkeypatch):
    from backend.main import create_app

    data, results = tmp_path/'data', tmp_path/'results'
    for kind in ['imagesTest','labelsTest']:(data/kind).mkdir(parents=True)
    Image.new('RGB',(3,2)).save(data/'imagesTest'/'case.png')
    Image.new('L',(3,2),1).save(data/'labelsTest'/'case.tiff')
    run = results/'completion'/'evaluation';run.mkdir(parents=True)
    (run/'summary.json').write_text(json.dumps({'format':'cs406-completion-evaluation-v1','dataset':{'role':'external_stress_test'},'methods':{},'available_images':1}))
    monkeypatch.setenv('SEGMENTATION_DATA_DIR', str(data))
    monkeypatch.setenv('SEGMENTATION_RESULTS_DIR', str(results))
    with TestClient(create_app()) as client:
        assert client.get('/api/summary').json()['total_images'] == 1
        assert any(r['file'].endswith('completion/evaluation/summary.json') for r in client.get('/api/experiments').json()['reports'])
        assert client.get('/api/results/completion/evaluation/summary.json?download=true').status_code == 200
        assert client.get('/api/results/..%5Coutside/summary.json').status_code == 404


def test_separate_deliverables_directory_serves_only_named_course_artifacts(tmp_path, monkeypatch):
    from backend.main import create_app

    deliverables = tmp_path/'course_documents'
    deliverables.mkdir()
    (deliverables/'CS406_Presentation.html').write_text('<html>Verified deck</html>')
    (deliverables/'CS406_Report.pdf').write_bytes(b'%PDF-course-report-fixture')
    (deliverables/'private.json').write_text('{"excluded":true}')
    monkeypatch.setenv('SEGMENTATION_DELIVERABLES_DIR', str(deliverables))
    with TestClient(create_app(tmp_path/'data')) as client:
        assert 'Verified deck' in client.get('/presentation').text
        response = client.get('/api/deliverables/CS406_Report.pdf')
        assert response.status_code == 200 and response.content.startswith(b'%PDF-')
        assert 'attachment' in response.headers['content-disposition']
        assert client.get('/api/deliverables/private.json').status_code == 404
        assert client.get('/api/deliverables/..%5Cprivate.json').status_code == 404
