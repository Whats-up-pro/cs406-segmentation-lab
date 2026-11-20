import copy
import hashlib
import json
from pathlib import Path

import pytest

from artifact_tools import prepare_completion_report as report


def test_reporting_rejects_intact_runs_assigned_to_wrong_ablation_names(tmp_path, monkeypatch):
    protocol = json.loads((Path(__file__).resolve().parents[1]/'config/experiments/compact_ablation.json').read_text())
    methods = {name: {'threshold': .5, 'provenance': {}} for name in ('baseline','pixel_logistic','transunet')}
    for recipe in protocol['runs']:
        methods[recipe['name']] = {'threshold': .5, 'provenance': {
            'frozen': {'epochs': 20, 'counts': {'train': 2179, 'valid': 157}},
            'config': {**protocol, **recipe},
        }}
    local = {'methods': methods, 'available_images': 295}
    external = {'methods': copy.deepcopy(methods), 'available_images': 100}
    local_dir, external_dir, analysis_dir, outside_dir = [tmp_path/name for name in ('local','external','analysis','outside')]
    monkeypatch.setattr(report, 'load_verified_evaluation', lambda path: (local if path==local_dir else external, {}))
    for evaluation, analysis in [(local_dir, analysis_dir), (external_dir, outside_dir)]:
        evaluation.mkdir();analysis.mkdir()
        for name in ('summary.json','per_image.csv'):
            (evaluation/name).write_bytes(b'verified fixture')
        digest = hashlib.sha256(b'verified fixture').hexdigest()
        (analysis/'analysis.json').write_text(json.dumps({'primary_method': report.PRIMARY,
            'model_selection_performed': False, 'input_sha256': {'summary.json':digest,'per_image.csv':digest}}))
    report.verified_inputs(local_dir, analysis_dir, external_dir, outside_dir)
    for data in (local, external):
        a,b = 'bce_plain_s406','bce_dice_s406'
        data['methods'][a],data['methods'][b] = data['methods'][b],data['methods'][a]
    with pytest.raises(ValueError, match='Declared recipe differs'):
        report.verified_inputs(local_dir, analysis_dir, external_dir, outside_dir)
