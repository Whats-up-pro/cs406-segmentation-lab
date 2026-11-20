"""Opt-in live browser verification: python -m scripts.check_ui [--url URL]."""

import argparse
import asyncio
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright, expect


async def check(url: str, output: Path, presentation=False, initial_split='test'):
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    checks = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 1150})
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(url, wait_until='networkidle')
        if initial_split != 'test':
            await page.locator('#split-select').select_option(initial_split)
            await expect(page.locator('#source-badge')).to_have_text('VALIDATION')
        await expect(page.locator('#original-image')).to_be_visible()
        await expect(page.locator('#app-error')).to_be_hidden()
        await page.wait_for_function('document.querySelector("#mask-image").complete && document.querySelector("#mask-image").naturalWidth > 0')
        assert await page.locator('.sample-card').count() == 6
        await page.screenshot(path=str(output / 'workspace-desktop.png'), full_page=True)
        checks.append('Dataset, original image, reference overlay and thumbnails load')
        initial_id = Path(await page.locator('#image-name').inner_text()).stem
        initial_detail = await (await page.request.get(f'{url.rstrip("/")}/api/samples/{initial_split}/{initial_id}')).json()
        native_size = (initial_detail['width'], initial_detail['height'])

        await page.locator('[data-mode=original]').click()
        await expect(page.locator('#mask-image')).to_be_hidden()
        await page.locator('[data-mode=mask]').click()
        await page.wait_for_function('document.querySelector("#mask-image").complete && document.querySelector("#mask-image").naturalWidth > 0')
        await expect(page.locator('#original-image')).to_have_css('visibility', 'hidden')
        await page.locator('[data-mode=overlay]').click()
        await page.locator('#opacity').evaluate('(el) => {el.value="70";el.dispatchEvent(new Event("input", {bubbles:true}));}')
        await expect(page.locator('#mask-image')).to_have_css('opacity', '0.7')
        annotation = '1' if len(initial_detail['annotations']) > 1 else '0'
        await page.locator('#annotation-select').select_option(annotation)
        await expect(page.locator('#mask-image')).to_have_attribute('src', __import__('re').compile('annotation='+annotation))
        async with page.expect_download() as download_info:
            await page.locator('#download-button').click()
        download = await download_info.value
        await download.save_as(str(output / 'reference-mask.png'))
        with Image.open(output / 'reference-mask.png') as mask:
            assert mask.size == native_size and mask.mode == 'L'
        checks.append('Display modes, opacity, annotation switching and original-size PNG export')

        await page.locator('#predict-button').click()
        await expect(page.locator('#prediction-result')).to_be_visible(timeout=30000)
        await expect(page.locator('#layer-source')).to_have_value('prediction')
        assert await page.locator('#dice-value').inner_text() != '—'
        await page.locator('#method-select').select_option('transunet')
        await page.locator('#predict-button').click()
        await expect(page.locator('#prediction-result')).to_be_visible(timeout=60000)
        await page.wait_for_function('document.querySelector("#mask-image").complete && document.querySelector("#mask-image").naturalWidth > 0')
        await expect(page.locator('#app-error')).to_be_hidden()
        await page.screenshot(path=str(output / 'workspace-prediction.png'), full_page=True)
        checks.append('Real baseline and supplied TransUNet inference with reference metrics')

        selected_reference = 'consensus' if initial_detail.get('consensus') else '0'
        reference_label = 'consensus' if selected_reference == 'consensus' else initial_detail['annotations'][0]['name']
        await page.locator('#annotation-select').select_option(selected_reference)
        await page.locator('#method-select').select_option('pixel_logistic')
        await expect(page.locator('#threshold')).to_have_value('30')
        await page.locator('#predict-button').click()
        await expect(page.locator('#prediction-result')).to_be_visible(timeout=30000)
        assert await page.locator('#precision-value').inner_text() != '—'
        assert await page.locator('#recall-value').inner_text() != '—'
        await page.locator('[data-mode=errors]').click()
        await expect(page.locator('#viewer-legend')).to_contain_text('TP')
        await expect(page.locator('#layer-source')).to_have_value('prediction')
        await page.locator('#layer-source').select_option('reference')
        await expect(page.locator('[data-mode=overlay]')).to_have_attribute('aria-pressed', 'true')
        await expect(page.locator('#mask-image')).to_have_attribute('src', __import__('re').compile('annotation='+selected_reference))
        await expect(page.locator('#viewer-legend')).to_contain_text(reference_label)
        checks.append('Logistic uses validation threshold 0.30; metrics and TP/FP/FN overlay match the selected reference')

        summary = await (await page.request.get(url.rstrip('/')+'/api/summary')).json()
        compact = summary.get('compact_model', {})
        if not compact.get('inference_ready'):
            raise AssertionError('Live compact UI verification requires an actual frozen checkpoint')
        await page.locator('#method-select').select_option('compact_unet')
        await expect(page.locator('#threshold')).to_have_value(str(round(compact['threshold']*100)))
        await expect(page.locator('#normalization-field')).to_be_hidden()
        await page.locator('#predict-button').click()
        await expect(page.locator('#prediction-result')).to_be_visible(timeout=30000)
        await expect(page.locator('#viewer-legend')).to_contain_text('Compact U-Net')
        assert await page.locator('#dice-value').inner_text() != '—'
        async with page.expect_download() as compact_download:
            await page.locator('#download-button').click()
        await (await compact_download.value).save_as(str(output/'compact-mask.png'))
        with Image.open(output/'compact-mask.png') as mask:
            assert mask.mode == 'L' and mask.size == native_size
        await page.screenshot(path=str(output/'compact-prediction.png'), full_page=True)
        checks.append('Actual Compact U-Net inference, frozen validation threshold and original-size mask export')

        await page.locator('#sample-search').fill('this-image-does-not-exist')
        await expect(page.locator('.empty-state')).to_be_visible()
        await page.locator('#sample-search').fill('')
        await expect(page.locator('.sample-card')).to_have_count(6)
        await page.locator('#next-page').click()
        await expect(page.locator('#sample-page')).to_have_text(__import__('re').compile(r'2 / \d+'))
        await page.locator('#split-select').select_option('valid')
        # Changing prediction settings must not cancel an in-flight image selection.
        await page.locator('#method-select').select_option('baseline')
        await expect(page.locator('#source-badge')).to_have_text('VALIDATION', timeout=20000)
        await expect(page.locator('#original-image')).to_be_visible()
        await expect(page.locator('#predict-button')).to_be_enabled()
        checks.append('Search, empty results, pagination, split changes and settings while loading')

        await page.locator('#method-select').select_option('compact_unet')
        payload = BytesIO()
        Image.new('RGB', (120, 80), '#909090').save(payload, format='PNG')
        await page.locator('#upload-input').set_input_files({'name': 'local-preview.png', 'mimeType': 'image/png', 'buffer': payload.getvalue()})
        await expect(page.locator('#image-name')).to_have_text('local-preview.png')
        await expect(page.locator('#download-button')).to_be_disabled()
        await expect(page.locator('#mask-image')).to_be_hidden()
        await page.locator('#predict-button').click()
        await expect(page.locator('#prediction-result')).to_be_visible()
        await expect(page.locator('#dice-value')).to_have_text('—')
        await expect(page.locator('#iou-value')).to_have_text('—')
        await page.locator('#upload-input').set_input_files({'name': 'broken.png', 'mimeType': 'image/png', 'buffer': b'bad-image'})
        await expect(page.locator('#image-name')).to_have_text('local-preview.png')
        await expect(page.locator('#predict-button')).to_be_enabled()
        checks.append('Local image upload clears reference masks; upload inference has no invented metrics; invalid upload preserves viewer')

        for view in ['dataset', 'model', 'experiments']:
            await page.locator(f'nav [data-view={view}]').click()
            await expect(page.locator(f'#view-{view}')).to_be_visible()
            if view == 'dataset':
                await expect(page.locator('#eda-content')).to_contain_text('Khám phá dữ liệu')
                await expect(page.locator('#eda-content .result-figure')).to_be_visible()
                await page.screenshot(path=str(output / 'dataset-desktop.png'), full_page=True)
        await expect(page.locator('#experiments-content table').first).to_be_visible()
        assert '74.6%' in await page.locator('#experiments-content').inner_text()
        assert '34.2%' in await page.locator('#experiments-content').inner_text()
        await page.screenshot(path=str(output / 'experiments-desktop.png'), full_page=True)
        checks.append('Dataset EDA chart, model/experiment navigation, actual comparison metrics and artifacts')

        await page.locator('nav [data-view=workspace]').click()
        await page.locator('#split-select').select_option('test')
        await expect(page.locator('#source-badge')).to_have_text('TEST')
        await page.route('**/mask?**', lambda route: route.abort())
        await page.locator('[data-mode=mask]').click()
        await expect(page.locator('#viewer-legend')).to_contain_text('Không tải được mask')
        await expect(page.locator('#original-image')).to_have_css('visibility', 'visible')
        await expect(page.locator('#stage-label')).to_be_hidden()
        await page.unroute('**/mask?**')
        await page.locator('[data-mode=overlay]').click()
        await expect(page.locator('#toast')).to_be_hidden(timeout=7000)
        checks.append('Failed binary-mask request falls back to original image')

        await page.set_viewport_size({'width': 390, 'height': 844})
        await expect(page.get_by_role('link', name='Bộ dữ liệu', exact=True)).to_be_visible()
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        await page.screenshot(path=str(output / 'workspace-mobile.png'), full_page=True)
        await page.get_by_role('link', name='Bộ dữ liệu', exact=True).click()
        await expect(page.locator('#view-dataset')).to_be_visible()
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        checks.append('390px mobile layout has named navigation and no page overflow')

        for width in [360, 390, 768, 1024, 1440, 1920]:
            await page.set_viewport_size({'width': width, 'height': 1000})
            for view in ['workspace', 'dataset', 'model', 'experiments']:
                await page.locator(f'nav [data-view={view}]').click()
                await expect(page.locator(f'#view-{view}')).to_be_visible()
                assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, view)
        checks.append('All four app views fit six viewport widths from 360 to 1920px')

        if presentation:
            await page.goto(url.rstrip('/') + '/presentation', wait_until='networkidle')
            total = await page.locator('.slide img').count()
            assert total > 1
            await expect(page.locator('#counter')).to_have_text(f'1 / {total}')
            await page.keyboard.press('ArrowRight')
            await expect(page.locator('#counter')).to_have_text(f'2 / {total}')
            await page.keyboard.press('End')
            await expect(page.locator('#counter')).to_have_text(f'{total} / {total}')
            await page.keyboard.press('n')
            await expect(page.locator('#notes')).to_be_visible()
            await page.set_viewport_size({'width': 390, 'height': 844})
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            checks.append('Slide deck keyboard navigation, speaker notes and mobile layout')
        await browser.close()
    assert not errors, errors
    result = {'checks': checks, 'console_errors': errors, 'passed': len(checks),
              'compact_checkpoint_sha256': compact['checkpoint_sha256'],
              'compact_training_epochs': compact['epochs'], 'presentation_checked': presentation,
              'model_inference_split': initial_split}
    (output / 'browser-checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--presentation', action='store_true')
    parser.add_argument('--initial-split', choices=['test','valid'], default='test')
    args = parser.parse_args()
    asyncio.run(check(args.url, args.output, args.presentation, args.initial_split))
