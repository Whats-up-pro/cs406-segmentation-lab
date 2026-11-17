"""FastAPI entry point: python -m uvicorn backend.main:app --reload."""

from base64 import b64encode
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageOps
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .dataset import DatasetStore, color_errors, color_mask, encode_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Split = Literal['train', 'valid', 'test']
Method = Literal['baseline', 'transunet', 'pixel_logistic']
Annotation = Annotated[int, Field(ge=0)] | Literal['consensus']


def create_app(data_dir: Path | None = None) -> FastAPI:
    application = FastAPI(title='Segmentation Studio · CS406', version='0.1.0')
    store = DatasetStore(data_dir or PROJECT_ROOT / 'data')
    application.state.dataset = store
    prediction_lock = Lock()

    @application.exception_handler(KeyError)
    async def missing_resource(_request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={'detail': exc.args[0]})

    @application.exception_handler(ValueError)
    async def invalid_image(_request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={'detail': str(exc)})

    @application.get('/api/summary')
    def summary():
        result = store.summary()
        if data_dir is None:
            from .inference import get_model_status
            status = get_model_status()
            result['model']['inference_ready'] = status['checkpoint_verified']
            result['model']['runtime'] = status['runtime']
            result['model']['available'] = status['available']
            from .pixel_model import pixel_model_status
            result['pixel_model'] = pixel_model_status()
        return result

    @application.get('/api/model')
    def model_status():
        from .inference import get_model_status
        return get_model_status()

    @application.get('/api/samples')
    def samples(split: Split = 'test', q: str = Query('', max_length=150),
                page: int = Query(1, ge=1), limit: int = Query(12, ge=1, le=60)):
        return store.list_samples(split, q, page, limit)

    @application.get('/api/samples/{split}/{sample_id}')
    def sample_detail(split: Split, sample_id: str):
        return store.detail(split, sample_id)

    @lru_cache(maxsize=64)
    def preview_bytes(split: str, sample_id: str, max_size: int):
        return encode_image(store.read_image(store.get(split, sample_id).image), max_size, 'JPEG')

    @application.get('/api/samples/{split}/{sample_id}/image')
    def sample_image(split: Split, sample_id: str, max_size: int = Query(1400, ge=0, le=4096)):
        return Response(preview_bytes(split, sample_id, max_size), media_type='image/jpeg',
                        headers={'Cache-Control': 'private, max-age=3600'})

    @application.get('/api/samples/{split}/{sample_id}/mask')
    def sample_mask(split: Split, sample_id: str, annotation: Annotation = 0,
                    mode: Literal['overlay', 'binary'] = 'overlay',
                    max_size: int = Query(1400, ge=0, le=4096), download: bool = False):
        sample = store.get(split, sample_id)
        image = store.read_reference(sample, annotation)
        if mode == 'overlay':
            image = color_mask(image)
        headers = {'Cache-Control': 'private, max-age=3600'}
        if download:
            headers['Content-Disposition'] = f'attachment; filename="{sample.id}_annotation{annotation}.png"'
        return Response(encode_image(image, max_size), media_type='image/png', headers=headers)

    def run_prediction(image: Image.Image, method: str, threshold: float, normalization: str,
                       reference: Image.Image | None, reference_name: str | None):
        if not prediction_lock.acquire(blocking=False):
            raise HTTPException(409, 'Đang xử lý một ảnh khác. Vui lòng thử lại sau.')
        try:
            started = perf_counter()
            if method == 'baseline':
                from .baseline import baseline_probability
                probability = baseline_probability(image)
            elif method == 'transunet':
                from .inference import predict_transunet
                probability = predict_transunet(image, normalization=normalization)
            else:
                from .pixel_model import predict_pixel_model
                probability = predict_pixel_model(image)
            binary = Image.fromarray((probability >= threshold).astype('uint8') * 255)
            metrics = None
            if reference is not None:
                from .metrics import segmentation_metrics
                metrics = segmentation_metrics(np.asarray(binary) > 0, np.asarray(reference) > 0)
            errors = None
            if reference is not None:
                error_image = color_errors(np.asarray(binary) > 0, np.asarray(reference) > 0)
                errors = 'data:image/png;base64,' + b64encode(encode_image(error_image, 1400)).decode()
            return {
                'method': method, 'threshold': threshold, 'normalization': normalization,
                'width': image.width, 'height': image.height,
                'elapsed_ms': round((perf_counter() - started) * 1000),
                'coverage': round(float(np.asarray(binary).mean()) / 255 * 100, 2),
                'overlay': 'data:image/png;base64,' + b64encode(encode_image(color_mask(binary), 1400)).decode(),
                'mask': 'data:image/png;base64,' + b64encode(encode_image(binary, 0)).decode(),
                'metrics': metrics, 'reference': reference_name, 'errors': errors,
            }
        except (ImportError, RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(503, 'Chưa chạy được phương pháp này: ' + str(exc)) from exc
        finally:
            prediction_lock.release()

    @application.post('/api/predict')
    async def predict(request: Request, method: Method = 'baseline',
                      threshold: float | None = Query(None, gt=0, lt=1),
                      normalization: Literal['unit', 'imagenet'] = 'unit',
                      split: Split = 'test', sample_id: str | None = None,
                      annotation: Annotation = 0):
        if threshold is None:
            if method == 'pixel_logistic':
                from .pixel_model import pixel_model_status
                threshold = pixel_model_status().get('threshold') or .5
            else:
                threshold = .5
        reference = None
        reference_name = None
        if sample_id:
            sample = store.get(split, sample_id)
            image = await run_in_threadpool(store.read_image, sample.image)
            if sample.masks:
                reference = await run_in_threadpool(store.read_reference, sample, annotation)
                reference_name = 'consensus' if annotation == 'consensus' else sample.masks[annotation].name
        else:
            chunks = bytearray()
            async for chunk in request.stream():
                chunks.extend(chunk)
                if len(chunks) > 20 * 1024 * 1024:
                    raise HTTPException(413, 'Ảnh vượt quá giới hạn 20 MB.')
            try:
                with Image.open(BytesIO(chunks)) as uploaded:
                    if uploaded.width * uploaded.height > 25_000_000:
                        raise HTTPException(413, 'Ảnh vượt quá giới hạn 25 megapixel.')
                    image = ImageOps.exif_transpose(uploaded).convert('RGB')
            except (OSError, ValueError, Image.DecompressionBombError):
                raise HTTPException(422, 'File tải lên không phải ảnh hợp lệ.') from None
        return await run_in_threadpool(run_prediction, image, method, threshold, normalization, reference, reference_name)

    @application.get('/api/experiments')
    def experiments():
        reports = []
        for path in sorted((PROJECT_ROOT / 'results').glob('**/*summary*.json')):
            if path.parent.name == 'eda':
                continue
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                if data.get('method') == 'pixel_logistic' and 'consensus' in data:
                    data = {**data, 'split': 'test', 'requested_images': data['images'], 'limited_run': False,
                            'failures': [], 'methods': {'pixel_logistic': {'sample_count': data['images'], 'macro': data['consensus']}},
                            'protocol': {'threshold': data['threshold'], 'reference': 'strict majority, ties background',
                                         'evaluation_size': 256, 'training': 'train only, validation-selected threshold'}}
                if isinstance(data.get('methods'), dict):
                    reports.append({'file': str(path.relative_to(PROJECT_ROOT)).replace('\\', '/'), 'data': data})
            except (OSError, ValueError):
                continue
        audit_path = PROJECT_ROOT / 'results/data_audit.json'
        audit = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else None
        eda_path = PROJECT_ROOT / 'results/eda/summary.json'
        eda = json.loads(eda_path.read_text(encoding='utf-8')) if eda_path.exists() else None
        reports.sort(key=lambda r: (r['file'] != 'results/comparison_test/summary.json', r['file']))
        analysis_path = PROJECT_ROOT / 'results/analysis/summary.json'
        analysis = json.loads(analysis_path.read_text(encoding='utf-8')) if analysis_path.exists() else None
        return {'reports': reports, 'audit': audit, 'eda': eda, 'analysis': analysis}

    @application.get('/api/results/{run_name}/{file_name}')
    def result_file(run_name: str, file_name: str, download: bool = False):
        root = (PROJECT_ROOT / 'results').resolve()
        allowed = {'summary.json', 'per_image.csv', 'per_annotation.csv', 'per_subject.csv',
                   'examples.json', 'examples.png', 'overview.png', 'outliers.png', 'outliers.json',
                   'paired_dice.png', 'macro_comparison.png', 'foreground_strata.png', 'agreement_strata.png',
                   'paired_images.csv', 'foreground_strata.csv', 'agreement_strata.csv', 'training_log.json'}
        path = (root / run_name / file_name).resolve()
        if not path.is_relative_to(root) or file_name not in allowed or not path.is_file():
            raise HTTPException(404, 'Không tìm thấy file kết quả.')
        return FileResponse(path, filename=path.name if download else None)

    frontend = PROJECT_ROOT / 'frontend'
    if frontend.is_dir():
        application.mount('/static', StaticFiles(directory=frontend), name='static')

        @application.get('/', include_in_schema=False)
        def index():
            return FileResponse(frontend / 'index.html')

    @application.get('/presentation', include_in_schema=False)
    def presentation():
        path = PROJECT_ROOT / 'deliverables/CS406_Presentation.html'
        if not path.is_file():
            raise HTTPException(404, 'Bộ slide chưa có trong thư mục deliverables.')
        return FileResponse(path)

    @application.get('/api/deliverables/{file_name}')
    def deliverable(file_name: str):
        allowed = {'CS406_Report.docx', 'CS406_Report.pdf', 'CS406_Defense.pptx', 'CS406_Defense.pdf'}
        path = PROJECT_ROOT / 'deliverables' / file_name
        if file_name not in allowed or not path.is_file():
            raise HTTPException(404, 'Không tìm thấy tài liệu yêu cầu.')
        return FileResponse(path, filename=file_name)

    return application


app = create_app()
