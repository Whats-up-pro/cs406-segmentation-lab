import csv
import hashlib
import json

import numpy as np
import pytest


def row(name, value, group='', method='primary'):
    return {'id': name, 'encounter': group, 'method': method, 'dice': value,
            'iou': value, 'precision': value, 'recall': value,
            'pixel_accuracy': value, 'boundary_f1': value, 'seconds': .1,
            'target_fraction': .2, 'width': 100, 'height': 100, 'annotations': 1}


def test_paired_bootstrap_preserves_encounters_and_image_weighting():
    from scripts.analyze_completion import paired_cluster_bootstrap

    left = [row('a', 1, 'A'), row('b', 1, 'A'), row('c', 0, 'B')]
    right = [row(x['id'], 0, x['encounter']) for x in left]
    result = paired_cluster_bootstrap(left, right, iterations=2000, seed=406)
    assert result['images'] == 3 and result['clusters'] == 2
    assert result['delta'] == pytest.approx(2/3)
    assert result['ci95'] == pytest.approx([0, 1])
    assert result == paired_cluster_bootstrap(left, right, iterations=2000, seed=406)
    identical = paired_cluster_bootstrap(left, left, iterations=2000)
    assert identical['delta'] == 0 and identical['ci95'] == [0, 0]


def test_pairing_is_by_id_and_missing_groups_are_individual_images():
    from scripts.analyze_completion import paired_cluster_bootstrap

    left = [row('a', .8), row('b', .2)]
    right = [row('b', .1), row('a', .5)]
    assert paired_cluster_bootstrap(left, right)['delta'] == pytest.approx(.2)
    assert paired_cluster_bootstrap(left, right)['clusters'] == 2
    with pytest.raises(ValueError, match='IDs'):
        paired_cluster_bootstrap(left, right[:1])
    with pytest.raises(ValueError, match='Duplicate'):
        paired_cluster_bootstrap(left + left[:1], right)
    right[0]['encounter'] = 'other'
    with pytest.raises(ValueError, match='encounter'):
        paired_cluster_bootstrap(left, right)


def test_bootstrap_rejects_invalid_values_and_iteration_counts():
    from scripts.analyze_completion import paired_cluster_bootstrap

    with pytest.raises(ValueError):
        paired_cluster_bootstrap([row('a', np.nan)], [row('a', .1)])
    with pytest.raises(ValueError):
        paired_cluster_bootstrap([row('a', .2)], [row('a', .1)], iterations=0)


def fixture_evaluation(path):
    path.mkdir()
    records = [row('a', .8), row('b', .6), row('a', .4, method='baseline'), row('b', .4, method='baseline')]
    with (path/'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    digest = hashlib.sha256((path/'per_image.csv').read_bytes()).hexdigest()
    summary = {'format': 'cs406-completion-evaluation-v1', 'comparison_complete': True,
               'failures': [], 'available_images': 2, 'dataset_fingerprint': 'a'*64,
               'methods': {name: {'sample_count': 2, 'evaluated_ids': ['a','b'],
                                  'macro': {'mean': {'dice': value, 'iou': value}}}
                           for name, value in [('primary', .7), ('baseline', .4)]},
               'output_sha256': {'per_image.csv': digest}}
    (path/'summary.json').write_text(json.dumps(summary), encoding='utf-8')


def test_analysis_uses_verified_complete_csv_and_refuses_overwrite_or_tampering(tmp_path):
    from scripts.analyze_completion import analyze

    source, output = tmp_path/'evaluation', tmp_path/'analysis'
    fixture_evaluation(source)
    result = analyze(source, output, primary='primary', plots=True)
    assert result['comparisons']['primary-minus-baseline']['dice']['delta'] == pytest.approx(.3)
    assert result['methods']['primary']['latency']['p95_seconds'] == pytest.approx(.1)
    assert (output/'analysis.json').is_file() and (output/'method_summary.csv').is_file()
    from PIL import Image
    with Image.open(output/'method_comparison.png') as figure:
        assert figure.width >= 1000 and figure.height >= 600
    with pytest.raises(FileExistsError):
        analyze(source, output, primary='primary', plots=False)
    with (source/'per_image.csv').open('a') as stream:
        stream.write('tampered\n')
    with pytest.raises(ValueError, match='hash'):
        analyze(source, tmp_path/'other', primary='primary', plots=False)
    assert not (tmp_path/'other').exists()


def test_incomplete_evaluation_and_inconsistent_summary_are_rejected_before_output(tmp_path):
    from scripts.analyze_completion import analyze

    source = tmp_path/'evaluation'
    fixture_evaluation(source)
    path = source/'summary.json'
    summary = json.loads(path.read_text())
    summary['comparison_complete'] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(source, tmp_path/'analysis', primary='primary')
    assert not (tmp_path/'analysis').exists()
    summary['comparison_complete'] = True
    summary['methods']['primary']['macro']['mean']['dice'] = .9
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='disagree'):
        analyze(source, tmp_path/'analysis', primary='primary')
