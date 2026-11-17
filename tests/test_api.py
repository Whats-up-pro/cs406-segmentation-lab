from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app


@pytest.fixture
def client(tmp_path):
    for folder in ('imagesTrain', 'labelsTrain', 'imagesValid', 'labelsValid', 'imagesTest', 'labelsTest'):
        (tmp_path / folder).mkdir()
    Image.new('RGB', (4, 2), '#e5b6a2').save(tmp_path / 'imagesTest/IMG_ENC00010_00001.jpg')
    Image.new('RGB', (4, 2), '#bbaabb').save(tmp_path / 'imagesTest/IMG_ENC00011_00001.jpg')
    a = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    b = np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    for annotation, array in [(1, a), (2, b), (3, a)]:
        Image.fromarray(array).save(tmp_path / f'labelsTest/IMG_ENC00010_00001_mask_ann{annotation}.tiff')
    Image.new('RGB', (4, 2)).save(tmp_path / 'imagesTrain/IMG_ENC00002_00001.png')
    Image.fromarray(a).save(tmp_path / 'labelsTrain/IMG_ENC00002_00001.tiff')
    (tmp_path / 'TransUNet_R50_ViT_B_16_all_best_model.pth').write_bytes(b'weights')
    with TestClient(create_app(tmp_path)) as app_client:
        yield app_client


def test_summary_uses_real_inventory_and_no_inference(client):
    data = client.get('/api/summary').json()
    assert data['total_images'] == 3
    assert data['total_masks'] == 4
    assert data['splits']['test']['images'] == 2
    assert data['model']['checkpoint_found'] is True
    assert data['model']['inference_ready'] is False


def test_pixel_prediction_uses_validation_selected_threshold_unless_explicit(client, monkeypatch):
    import backend.pixel_model as pixel_model
    monkeypatch.setattr(pixel_model, 'pixel_model_status', lambda: {'threshold': .3})
    monkeypatch.setattr(pixel_model, 'predict_pixel_model',
                        lambda image: np.full((image.height, image.width), .4, dtype=np.float32))
    url = '/api/predict?method=pixel_logistic&split=test&sample_id=IMG_ENC00010_00001&annotation=consensus'
    selected = client.post(url)
    assert selected.status_code == 200
    assert selected.json()['threshold'] == .3
    assert selected.json()['coverage'] == 100
    assert selected.json()['reference'] == 'consensus'
    explicit = client.post(url + '&threshold=0.5')
    assert explicit.status_code == 200
    assert explicit.json()['coverage'] == 0


def test_search_pagination_and_missing_masks(client):
    data = client.get('/api/samples?split=test&q=enc00010&limit=1').json()
    assert data['total'] == 1
    assert data['items'][0]['id'] == 'IMG_ENC00010_00001'
    assert data['items'][0]['mask_count'] == 3
    page = client.get('/api/samples?split=test&page=2&limit=1').json()
    assert page['items'][0]['id'] == 'IMG_ENC00011_00001'
    assert page['items'][0]['mask_count'] == 0
    assert client.get('/api/samples?split=test&q=absent').json()['items'] == []


def test_discovers_annotations_without_assuming_ann0(client):
    info = client.get('/api/samples/test/IMG_ENC00010_00001').json()
    assert info['width'] == 4 and info['height'] == 2
    assert [a['name'] for a in info['annotations']] == ['ann1', 'ann2', 'ann3']
    assert info['annotations'][0]['index'] == 0
    assert info['annotations'][0]['coverage'] == 25.0


def test_binary_export_keeps_original_resolution_and_annotation(client):
    response = client.get('/api/samples/test/IMG_ENC00010_00001/mask?annotation=1&mode=binary&max_size=0&download=true')
    assert response.status_code == 200
    assert response.headers['content-type'] == 'image/png'
    assert 'attachment' in response.headers['content-disposition']
    im = Image.open(BytesIO(response.content))
    assert im.size == (4, 2)
    assert list(im.getdata()) == [255, 0, 0, 0, 0, 0, 0, 0]


def test_overlay_is_transparent_outside_annotation(client):
    response = client.get('/api/samples/test/IMG_ENC00010_00001/mask?annotation=0')
    im = Image.open(BytesIO(response.content))
    assert im.mode == 'RGBA'
    assert im.getpixel((0, 0))[3] == 0
    assert im.getpixel((1, 0))[3] == 255


def test_preview_preserves_aspect_ratio(client):
    response = client.get('/api/samples/test/IMG_ENC00010_00001/image?max_size=2')
    assert response.status_code == 200
    assert Image.open(BytesIO(response.content)).size == (2, 1)


@pytest.mark.parametrize('url', [
    '/api/samples?split=other',
    '/api/samples?split=test&page=0',
    '/api/samples?split=test&limit=999',
    '/api/samples/test/IMG_ENC00010_00001/mask?annotation=-1',
    '/api/samples/test/IMG_ENC00010_00001/mask?max_size=-1',
    '/api/samples/test/IMG_ENC00010_00001/mask?mode=oops',
])
def test_rejects_invalid_requests(client, url):
    assert client.get(url).status_code == 422


@pytest.mark.parametrize('url', [
    '/api/samples/test/not-found',
    '/api/samples/test/IMG_ENC00011_00001/mask',
    '/api/samples/test/IMG_ENC00010_00001/mask?annotation=9',
    '/api/samples/test/%2E%2E%5Csecret/image',
    '/data/TransUNet_R50_ViT_B_16_all_best_model.pth',
])
def test_only_serves_known_images_and_annotations(client, url):
    assert client.get(url).status_code == 404


def test_missing_dataset_is_a_useful_empty_state(tmp_path):
    with TestClient(create_app(tmp_path / 'missing')) as client:
        data = client.get('/api/summary').json()
        assert data['total_images'] == 0
        assert data['model']['checkpoint_found'] is False
        assert client.get('/api/samples?split=test').json()['items'] == []


def test_mismatched_mask_does_not_get_overlaid(tmp_path):
    (tmp_path / 'imagesTrain').mkdir()
    (tmp_path / 'labelsTrain').mkdir()
    Image.new('RGB', (4, 2)).save(tmp_path / 'imagesTrain/example.png')
    Image.new('L', (2, 2), 1).save(tmp_path / 'labelsTrain/example.tiff')
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/samples/train/example/mask').status_code == 422


def test_corrupt_image_returns_readable_error(tmp_path):
    (tmp_path / 'imagesTest').mkdir()
    (tmp_path / 'imagesTest/broken.jpg').write_bytes(b'not an image')
    with TestClient(create_app(tmp_path)) as client:
        response = client.get('/api/samples/test/broken')
        assert response.status_code == 422
        assert response.json()['detail']


def test_baseline_prediction_returns_real_mask_and_reference_metrics(client):
    response = client.post('/api/predict?method=baseline&split=test&sample_id=IMG_ENC00010_00001')
    assert response.status_code == 200
    data = response.json()
    assert data['method'] == 'baseline'
    assert data['width'] == 4 and data['height'] == 2
    assert data['mask'].startswith('data:image/png;base64,')
    assert data['overlay'].startswith('data:image/png;base64,')
    assert 0 <= data['metrics']['dice'] <= 1
    assert data['reference'].endswith('_mask_ann1.tiff')


def test_uploaded_image_has_no_fabricated_metrics(client):
    output = BytesIO()
    Image.new('RGB', (20, 12), '#8a8a8a').save(output, format='PNG')
    response = client.post('/api/predict?method=baseline', content=output.getvalue(),
                           headers={'content-type': 'image/png'})
    assert response.status_code == 200
    data = response.json()
    assert data['width'] == 20 and data['height'] == 12
    assert data['metrics'] is None and data['reference'] is None
    assert data['coverage'] == 0


def test_malformed_upload_is_rejected(client):
    assert client.post('/api/predict', content=b'not-an-image').status_code == 422


def test_uploaded_phone_photo_uses_exif_display_orientation(client):
    output = BytesIO()
    image = Image.new('RGB', (20, 12), '#8a8a8a')
    exif = image.getexif()
    exif[274] = 6
    image.save(output, format='JPEG', exif=exif)
    response = client.post('/api/predict?method=baseline', content=output.getvalue())
    assert response.status_code == 200
    assert (response.json()['width'], response.json()['height']) == (12, 20)


def test_unrelated_weight_file_is_not_reported_as_active_checkpoint(tmp_path):
    (tmp_path / 'unrelated.pth').write_bytes(b'weights')
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/api/summary').json()['model']['checkpoint_found'] is False


def test_checkpoint_inventory_honors_explicit_model_override(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'chosen.pth'
    checkpoint.write_bytes(b'weights')
    monkeypatch.setenv('TRANSUNET_CHECKPOINT', str(checkpoint))
    with TestClient(create_app(tmp_path / 'data')) as client:
        model = client.get('/api/summary').json()['model']
        assert model['checkpoint_found'] is True
        assert model['checkpoint_name'] == 'chosen.pth'
        assert model['inference_ready'] is False


def test_result_downloads_stay_inside_results_directory(tmp_path, monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    run = tmp_path / 'results/demo'
    run.mkdir(parents=True)
    (run / 'per_image.csv').write_bytes(b'id,dice\nexample,0.5\n')
    (tmp_path / 'summary.json').write_text('{"private": true}', encoding='utf-8')
    with TestClient(create_app(tmp_path / 'data')) as client:
        response = client.get('/api/results/demo/per_image.csv?download=true')
        assert response.status_code == 200
        assert response.text == 'id,dice\nexample,0.5\n'
        assert 'attachment' in response.headers['content-disposition']
        assert client.get('/api/results/..%5C../summary.json').status_code == 404
        assert client.get('/api/results/demo/not-allowed.py').status_code == 404


def test_experiments_separates_eda_and_prioritizes_full_comparison(tmp_path, monkeypatch):
    import backend.main as main
    monkeypatch.setattr(main, 'PROJECT_ROOT', tmp_path)
    for folder in ['baseline_test', 'comparison_test', 'eda']:
        directory = tmp_path / 'results' / folder
        directory.mkdir(parents=True)
        (directory / 'summary.json').write_text('{"name":"' + folder + '","methods":{}}', encoding='utf-8')
    with TestClient(create_app(tmp_path / 'data')) as client:
        response = client.get('/api/experiments').json()
        assert response['reports'][0]['data']['name'] == 'comparison_test'
        assert len(response['reports']) == 2
        assert response['eda']['name'] == 'eda'


def test_consensus_mask_combines_all_discovered_annotations(client):
    response = client.get('/api/samples/test/IMG_ENC00010_00001/mask?annotation=consensus&mode=binary&max_size=0')
    assert response.status_code == 200
    mask = Image.open(BytesIO(response.content))
    assert mask.size == (4, 2)
    assert list(mask.getdata()) == [0, 255, 0, 0, 0, 0, 255, 0]


def test_consensus_tie_stays_background(tmp_path):
    (tmp_path / 'imagesTest').mkdir()
    (tmp_path / 'labelsTest').mkdir()
    Image.new('RGB', (2, 1)).save(tmp_path / 'imagesTest/example.png')
    Image.fromarray(np.array([[1, 1]], dtype=np.uint8)).save(tmp_path / 'labelsTest/example_mask_ann1.tiff')
    Image.fromarray(np.array([[1, 0]], dtype=np.uint8)).save(tmp_path / 'labelsTest/example_mask_ann3.tiff')
    with TestClient(create_app(tmp_path)) as client:
        response = client.get('/api/samples/test/example/mask?annotation=consensus&mode=binary&max_size=0')
        assert response.status_code == 200
        assert list(Image.open(BytesIO(response.content)).getdata()) == [255, 0]


def test_prediction_can_use_consensus_and_returns_measured_error_layer(client):
    from base64 import b64decode
    response = client.post('/api/predict?method=baseline&split=test&sample_id=IMG_ENC00010_00001&annotation=consensus')
    assert response.status_code == 200
    result = response.json()
    assert result['reference'] == 'consensus'
    overlay = Image.open(BytesIO(b64decode(result['errors'].split(',')[1])))
    assert overlay.getpixel((0, 0))[3] == 0
    assert overlay.getpixel((1, 0)) == (80, 130, 210, 255)


def test_error_layer_encodes_true_positive_false_positive_and_false_negative():
    from backend.dataset import color_errors
    predicted = np.array([[1, 1], [0, 0]], dtype=bool)
    target = np.array([[1, 0], [1, 0]], dtype=bool)
    overlay = color_errors(predicted, target)
    assert overlay.getpixel((0, 0)) == (32, 177, 153, 255)
    assert overlay.getpixel((1, 0)) == (239, 109, 86, 255)
    assert overlay.getpixel((0, 1)) == (80, 130, 210, 255)
    assert overlay.getpixel((1, 1))[3] == 0
