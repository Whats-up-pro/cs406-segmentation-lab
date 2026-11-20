import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def evaluation_fixture(root):
    from scripts.evaluate_completion import score_masks

    data, evaluation = root/'data', root/'evaluation'
    (data/'imagesTest').mkdir(parents=True)
    (data/'labelsTest').mkdir()
    target = np.zeros((256,256),dtype=np.uint8); target[64:192,64:192]=1
    image = data/'imagesTest/case.png'
    mask = data/'labelsTest/case_mask_ann1.tiff'
    Image.new('RGB',(256,256),'#bf9c88').save(image)
    Image.fromarray(target).save(mask)
    methods = {'bce_dice_aug_s406':target, 'baseline':np.zeros_like(target)}
    rows=[]
    for method,prediction in methods.items():
        directory=evaluation/'masks'/method;directory.mkdir(parents=True)
        Image.fromarray(prediction*255).save(directory/'case.png')
        rows.append({'id':'case','method':method,'encounter':'','seconds':.1,'target_fraction':.25,
                     'prediction_fraction':float(prediction.mean()),'width':256,'height':256,'annotations':1,
                     **score_masks(prediction,target)})
    path=evaluation/'per_image.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'format':'cs406-completion-evaluation-v1','comparison_complete':True,'failures':[],
             'available_images':1,'methods':{row['method']:{'sample_count':1,'evaluated_ids':['case'],
                'macro':{'mean':{'dice':row['dice'],'iou':row['iou']}}} for row in rows},
             'input_inventory':[{'id':'case','image':'imagesTest/case.png','image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                                 'masks':[{'path':'labelsTest/case_mask_ann1.tiff','sha256':hashlib.sha256(mask.read_bytes()).hexdigest()}]}],
             'output_sha256':{'per_image.csv':hashlib.sha256(path.read_bytes()).hexdigest()}}
    (evaluation/'summary.json').write_text(json.dumps(summary))
    return data,evaluation


def test_real_saved_masks_form_traceable_case_assets_without_predicting_again(tmp_path):
    from scripts.render_completion_cases import render_cases

    data,evaluation=evaluation_fixture(tmp_path)
    output=tmp_path/'cases'
    result=render_cases(evaluation,data,output)
    assert result['prediction_performed'] is False
    assert len(result['cases'])==1
    case=result['cases'][0]
    assert case['id']=='case' and case['primary_dice']==1 and case['baseline_dice']==0
    assert 'best_foreground_dice' in case['selection_reasons']
    with Image.open(output/case['assets']['errors']) as rendered:
        assert rendered.size==(256,256)
        assert tuple(np.asarray(rendered)[100,100])==(27,158,119)
    assert (output/'case_plate_1.png').is_file()
    assert len(result['source_sha256'])>=2
    with pytest.raises(FileExistsError):render_cases(evaluation,data,output)


def test_stale_prediction_or_source_data_cannot_be_illustrated_as_the_csv_result(tmp_path):
    from scripts.render_completion_cases import render_cases

    data,evaluation=evaluation_fixture(tmp_path)
    path=evaluation/'masks/bce_dice_aug_s406/case.png'
    Image.new('L',(256,256),0).save(path)
    with pytest.raises(ValueError,match='confusion|CSV'):
        render_cases(evaluation,data,tmp_path/'cases')
    assert not (tmp_path/'cases').exists()
    Image.new('RGB',(256,256),'black').save(data/'imagesTest/case.png')
    with pytest.raises(ValueError,match='source|hash'):
        render_cases(evaluation,data,tmp_path/'cases')


def test_self_consistent_csv_hashes_cannot_hide_scores_that_disagree_with_pixels(tmp_path):
    from scripts.render_completion_cases import render_cases

    data,evaluation=evaluation_fixture(tmp_path)
    path=evaluation/'per_image.csv'
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    rows[0]['dice']='0.9'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=json.loads((evaluation/'summary.json').read_text())
    summary['methods']['bce_dice_aug_s406']['macro']['mean']['dice']=.9
    summary['output_sha256']['per_image.csv']=hashlib.sha256(path.read_bytes()).hexdigest()
    (evaluation/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='metric|CSV'):
        render_cases(evaluation,data,tmp_path/'cases')
    assert not (tmp_path/'cases').exists()
