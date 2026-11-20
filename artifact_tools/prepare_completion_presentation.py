"""Freeze verified final measurements and figures for the editable defense deck."""
import argparse
import json
from pathlib import Path

from prepare_completion_report import PRIMARY, REPO, ROOT, sha, verified_cases, verified_inputs


def prepare(args):
    local,outside,analysis,external_analysis = verified_inputs(
        args.evaluation_dir,args.analysis_dir,args.external_dir,args.external_analysis_dir)
    cases = verified_cases(args.case_dir,args.evaluation_dir)
    runs = {}
    for method,item in local['methods'].items():
        if method in {'baseline','pixel_logistic','transunet'}:
            continue
        directory = args.runs_dir/method
        frozen = item['provenance']['frozen']
        history_path = directory/'history.json'
        if sha(history_path) != frozen['history_sha256']:
            raise ValueError('Training curve differs from frozen evaluation: '+method)
        runs[method] = {'history':json.loads(history_path.read_text(encoding='utf-8')),
                        'frozen':frozen}
    ui_path = ROOT/'output/completion/ui-final-valid/browser-checks.json'
    ui = json.loads(ui_path.read_text(encoding='utf-8'))
    if (ui['compact_checkpoint_sha256'] != runs[PRIMARY]['frozen']['checkpoint_sha256']
            or ui['compact_training_epochs'] != 20 or ui['console_errors']):
        raise ValueError('Demo screenshot is not verified against the final primary model')
    screenshot = ui_path.parent/'compact-prediction.png'
    if not screenshot.is_file():
        raise ValueError('Missing actual-model demo screenshot')
    assets = {str((args.case_dir/name).resolve()):expected for name,expected in cases['output_sha256'].items()}
    assets[str(screenshot.resolve())] = sha(screenshot)
    worked_dir = ROOT/'output/presentation-evidence'
    worked = json.loads((worked_dir/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in worked['files'].items():
        path = (worked_dir/name).resolve()
        if not path.is_relative_to(worked_dir.resolve()) or sha(path) != expected:
            raise ValueError('Worked-example image changed: '+name)
        assets[str(path)] = expected
    result = {'format':'cs406-defense-input-v1','draft':False,'primary':PRIMARY,
              'local':local,'external':outside,'analysis':analysis,'external_analysis':external_analysis,
              'runs':runs,'cases':cases,'case_dir':str(args.case_dir.resolve()),
              'demo_screenshot':str(screenshot.resolve()),
              'team':json.loads((REPO/'config/team.json').read_text(encoding='utf-8')),
              'asset_sha256':assets,
              'evidence_sha256':{str((directory/name).resolve()):sha(directory/name)
                                  for directory,name in [(args.evaluation_dir,'summary.json'),
                                                         (args.external_dir,'summary.json'),
                                                         (args.analysis_dir,'analysis.json'),
                                                         (args.external_analysis_dir,'analysis.json'),
                                                         (args.case_dir,'case-manifest.json')]}}
    if len(result['team']['members']) != 4 or any(row['placeholder'] for row in result['team']['members']):
        raise ValueError('All four actual members are required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise FileExistsError('Presentation evidence already exists; use a new output path')
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'methods':len(local['methods'])}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-dir','analysis-dir','external-dir','external-analysis-dir','case-dir','runs-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    prepare(parser.parse_args())
