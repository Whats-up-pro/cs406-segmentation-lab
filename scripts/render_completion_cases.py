"""Render traceable error examples from verified saved masks, without inference."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from scripts.analyze_completion import PRIMARY, SCORES, load_verified_evaluation, sha256
from scripts.evaluate import load_reference
from scripts.evaluate_completion import COUNTS, score_masks

COLORS = {'TP':(27,158,119), 'FP':(215,48,39), 'FN':(69,117,180)}


def _read_prediction(path):
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != (256,256):
            raise ValueError('Saved prediction must be a256-square grayscale PNG')
        values=np.asarray(image).copy()
    if not set(np.unique(values)).issubset({0,255}):
        raise ValueError('Saved prediction is not an exported binary mask')
    return values != 0


def _choose(primary, baseline):
    by_baseline={row['id']:row for row in baseline}
    foreground=[row for row in primary if row['target_fraction']>0] or primary
    choices=[('best_foreground_dice',min(foreground,key=lambda r:(-r['dice'],r['id']))),
             ('worst_dice',min(primary,key=lambda r:(r['dice'],r['id']))),
             ('largest_primary_minus_baseline',min(primary,key=lambda r:(-(r['dice']-by_baseline[r['id']]['dice']),r['id']))),
             ('smallest_primary_minus_baseline',min(primary,key=lambda r:(r['dice']-by_baseline[r['id']]['dice'],r['id'])))]
    selected={}
    for reason,row in choices:
        selected.setdefault(row['id'],{'row':row,'reasons':[]})['reasons'].append(reason)
    return selected


def render_cases(evaluation_dir, data_dir, output, primary=PRIMARY):
    evaluation_dir,data_dir,output=Path(evaluation_dir).resolve(),Path(data_dir).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(f'Case output already exists: {output}')
    summary,rows=load_verified_evaluation(evaluation_dir)
    if primary not in rows or 'baseline' not in rows:raise ValueError('Case comparison requires primary and baseline')
    selected=_choose(rows[primary],rows['baseline'])
    inventory={row['id']:row for row in summary['input_inventory']}
    baseline_rows={row['id']:row for row in rows['baseline']}
    prepared=[]
    hashes={'evaluation/summary.json':sha256(evaluation_dir/'summary.json'),
            'evaluation/per_image.csv':sha256(evaluation_dir/'per_image.csv')}
    for case_id,selection in selected.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',case_id):raise ValueError('Unsafe case ID')
        source=inventory[case_id]
        image_path=(data_dir/source['image']).resolve()
        masks=[(data_dir/row['path']).resolve() for row in source['masks']]
        for path,expected in [(image_path,source['image_sha256']),*[(path,row['sha256']) for path,row in zip(masks,source['masks'])]]:
            if not path.is_relative_to(data_dir) or not path.is_file() or sha256(path)!=expected:
                raise ValueError(f'Case source hash mismatch: {case_id}')
            hashes['data/'+path.relative_to(data_dir).as_posix()]=expected
        target,_=load_reference({'image':image_path,'masks':masks},(256,256))
        predictions={}
        for method,row in [(primary,selection['row']),('baseline',baseline_rows[case_id])]:
            path=(evaluation_dir/'masks'/method/f'{case_id}.png').resolve()
            if not path.is_relative_to(evaluation_dir):raise ValueError('Unsafe prediction path')
            prediction=_read_prediction(path)
            actual=score_masks(prediction,target)
            if any(actual[key]!=int(row[key]) for key in COUNTS):
                raise ValueError(f'Saved prediction confusion counts differ from CSV: {case_id}, {method}')
            if any(not np.isclose(actual[key],row[key],atol=1e-9,rtol=0) for key in SCORES):
                raise ValueError(f'Saved prediction metrics differ from CSV: {case_id}, {method}')
            if not np.isclose(float(target.mean()),row['target_fraction'],atol=1e-12,rtol=0):
                raise ValueError('Target area differs from CSV')
            hashes['evaluation/'+path.relative_to(evaluation_dir).as_posix()]=sha256(path)
            predictions[method]=prediction
        with Image.open(image_path) as original:
            rgb=np.asarray(original.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
        prediction=predictions[primary]
        errors=np.zeros((256,256,3),dtype=np.uint8)
        errors[prediction & target]=COLORS['TP']
        errors[prediction & ~target]=COLORS['FP']
        errors[~prediction & target]=COLORS['FN']
        assets={'source':rgb,'reference':target.astype(np.uint8)*255,
                'primary':prediction.astype(np.uint8)*255,'baseline':predictions['baseline'].astype(np.uint8)*255,'errors':errors}
        prepared.append((case_id,selection,assets))
    # All selected input hashes and CSV/mask confusion matrices agree before output creation.
    output.mkdir(parents=True)
    cases=[]
    for case_id,selection,arrays in prepared:
        paths={}
        for role,values in arrays.items():
            name=f'{case_id}__{role}.png';Image.fromarray(values).save(output/name);paths[role]=name
        left,right=selection['row'],baseline_rows[case_id]
        cases.append({'id':case_id,'selection_reasons':selection['reasons'],'assets':paths,
                      'primary_dice':left['dice'],'baseline_dice':right['dice'],
                      'delta_dice':left['dice']-right['dice'],'target_fraction':left['target_fraction'],
                      'primary_prediction_fraction':left['prediction_fraction'],
                      'primary_confusion':{key:int(left[key]) for key in COUNTS},
                      'interpretation':'Image inspection is required before assigning a visual cause; no causal claim is inferred from rank.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plates=[]
    for offset in range(0,len(cases),2):
        group=cases[offset:offset+2]
        figure,axes=plt.subplots(len(group),4,figsize=(6.1,2.2*len(group)),squeeze=False)
        for row,case in enumerate(group):
            for column,(role,title) in enumerate([('source','RGB'),('reference','Reference'),('primary','Compact U-Net'),('errors','TP / FP / FN')]):
                with Image.open(output/case['assets'][role]) as image:values=np.asarray(image).copy()
                axes[row,column].imshow(values,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[row,column].set_title(title,fontsize=8);axes[row,column].axis('off')
            axes[row,0].text(0,1.28,f'{case["id"]}: Dice {case["primary_dice"]:.4f}; baseline {case["baseline_dice"]:.4f}',
                             transform=axes[row,0].transAxes,fontsize=8,clip_on=False)
        figure.subplots_adjust(left=.01,right=.99,top=.80,bottom=.12,hspace=.65,wspace=.04)
        figure.text(.015,.025,'TP xanh, FP đỏ, FN lam. Ví dụ được chọn sau đánh giá.',fontsize=8)
        name=f'case_plate_{offset//2+1}.png';figure.savefig(output/name,dpi=220);plt.close(figure);plates.append(name)
    result={'format':'cs406-saved-error-cases-v1','primary_method':primary,'prediction_performed':False,
            'parameter_selection_performed':False,'cases':cases,'plates':plates,'colors':COLORS,
            'selection_rules':['highest Dice among nonempty targets','lowest Dice','largest paired Dice difference','smallest paired Dice difference'],
            'source_sha256':hashes,'renderer_sha256':sha256(__file__),
            'output_sha256':{path.name:sha256(path) for path in sorted(output.iterdir()) if path.is_file()}}
    (output/'case-manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--primary',default=PRIMARY)
    args=parser.parse_args(argv)
    result=render_cases(args.evaluation_dir,args.data_dir,args.output,args.primary)
    print(json.dumps({'cases':len(result['cases']),'output':str(args.output.resolve())}))


if __name__=='__main__':main()
