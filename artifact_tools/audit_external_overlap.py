"""Read-only cross-source byte/hash candidate audit; never performs inference."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'cs406-segmentation-lab'))
from scripts.audit_data import file_sha256
from scripts.audit_leakage import _perceptual_hashes


def main():
    source=ROOT/'results/completion/leakage/inventory.csv'
    external=ROOT/'data_external/isic2018_validation'
    output=ROOT/'results/completion/external-overlap'
    if output.exists() and any(output.iterdir()):raise FileExistsError('Choose a new output; preserve the existing audit')
    local=list(csv.DictReader(source.open(encoding='utf-8',newline='')))
    manifest=json.loads((external/'manifest.json').read_text(encoding='utf-8'))
    cv2.setNumThreads(1)
    inventory=[];exact=[];candidates=[]
    for sample in manifest['samples']:
        path=external/sample['image']
        digest=file_sha256(path)
        if digest!=sample['image_sha256']:raise ValueError('External image differs from prepared manifest')
        with Image.open(path) as image:phash,dhash,deviation=_perceptual_hashes(image)
        inventory.append({'id':sample['id'],'image':sample['image'],'sha256':digest,
                          'phash':f'{phash:016x}','dhash':f'{dhash:016x}','grayscale_std_32':deviation})
        for row in local:
            if row['sha256']==digest:
                exact.append({'external_id':sample['id'],'local_id':row['id'],'local_split':row['split']})
            pd=(phash^int(row['phash'],16)).bit_count()
            if pd<=6:
                dd=(dhash^int(row['dhash'],16)).bit_count()
                if dd<=8:candidates.append({'external_id':sample['id'],'local_id':row['id'],'local_split':row['split'],
                                           'phash_distance':pd,'dhash_distance':dd,
                                           'low_information':min(deviation,float(row['grayscale_std_32']))<5,
                                           'review_status':'pending_visual_review'})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'prediction_performed':False,
            'counts':{'external_images':len(inventory),'local_images':len(local),'pairs':len(inventory)*len(local),
                      'exact_matches':len(exact),'perceptual_candidates':len(candidates)},
            'protocol':{'byte_identity':'SHA-256 equality','phash_max_hamming':6,'dhash_max_hamming':8,
                        'selection':'Both thresholds required; identical implementation/thresholds as local cross-split audit'},
            'exact_matches':exact,'perceptual_candidates':candidates,
            'limitations':['No candidate at these thresholds does not establish patient independence or exclude every crop/transformation.',
                           'Supplied TransUNet training history remains unknown; no claim that it never saw ISIC.'],
            'source_sha256':{'local_inventory':file_sha256(source),'external_manifest':file_sha256(external/'manifest.json'),
                             'hash_implementation':file_sha256(ROOT.parent/'cs406-segmentation-lab/scripts/audit_leakage.py'),
                             'audit_script':file_sha256(Path(__file__))}}
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    with (output/'external_inventory.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    print(json.dumps(result['counts']))


if __name__=='__main__':main()
