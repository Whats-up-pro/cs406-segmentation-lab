"""Small, hand-computable checks for descriptive result aggregation."""

import pytest

from scripts import analyze_results as analysis


def test_cluster_bootstrap_keeps_pairs_and_distinguishes_image_weighting():
    # A has four images, B one: image mean=1/5, group mean=1/2.
    # Sampling images independently would also produce a narrower interval.
    rows = [
        {'id': 'a1', 'subject': 'A', 'baseline_dice': 0., 'transunet_dice': 0.},
        {'id': 'a2', 'subject': 'A', 'baseline_dice': 0., 'transunet_dice': 0.},
        {'id': 'a3', 'subject': 'A', 'baseline_dice': 0., 'transunet_dice': 0.},
        {'id': 'a4', 'subject': 'A', 'baseline_dice': 0., 'transunet_dice': 0.},
        {'id': 'b1', 'subject': 'B', 'baseline_dice': 0., 'transunet_dice': 1.},
    ]
    result = analysis.paired_cluster_bootstrap(rows, n_resamples=400, seed=9)
    assert result == analysis.paired_cluster_bootstrap(rows, n_resamples=400, seed=9)
    assert result['groups'] == 2
    assert result['image_macro']['difference']['estimate'] == pytest.approx(1 / 5)
    assert result['filename_subject_macro']['difference']['estimate'] == .5
    assert result['image_macro']['baseline']['ci95'] == [0., 0.]
    assert result['image_macro']['transunet']['ci95'] == [0., 1.]
    assert result['image_macro']['difference']['ci95'] == result['image_macro']['transunet']['ci95']


def test_duplicate_sensitivity_balances_hashes_without_discarding_annotation_versions():
    rows = [
        {'id': 'a', 'sha256': 'same', 'target_fraction': .1, 'baseline_dice': .2, 'transunet_dice': .4},
        {'id': 'b', 'sha256': 'same', 'target_fraction': .8, 'baseline_dice': .6, 'transunet_dice': .8},
        {'id': 'c', 'sha256': 'other', 'target_fraction': .5, 'baseline_dice': 1., 'transunet_dice': 1.},
    ]
    result = analysis.duplicate_sensitivity(rows)
    assert result['unique_hashes'] == 2
    assert result['redundant_files'] == 1
    assert result['hash_balanced']['baseline_dice'] == pytest.approx(.7)
    assert result['lexicographic_first_per_hash']['baseline_dice'] == pytest.approx(.6)
    assert result['duplicate_groups'][0]['target_fraction_range'] == pytest.approx(.7)
    assert result['duplicate_groups'][0]['ids'] == ['a', 'b']


@pytest.mark.parametrize(('fraction', 'expected'), [
    (0., '[0, 0.05)'), (.0499, '[0, 0.05)'), (.05, '[0.05, 0.20)'),
    (.2, '[0.20, 0.50)'), (.5, '[0.50, 1.00]'), (1., '[0.50, 1.00]'),
])
def test_foreground_strata_have_fixed_nonoverlapping_edges(fraction, expected):
    assert analysis.foreground_stratum(fraction) == expected


def test_foreground_strata_reject_out_of_range_values():
    with pytest.raises(ValueError):
        analysis.foreground_stratum(1.01)


def test_agreement_boundary_is_high_and_missing_remains_missing():
    assert analysis.agreement_stratum(.499) == 'low (<0.50)'
    assert analysis.agreement_stratum(.5) == 'high (>=0.50)'
    assert analysis.agreement_stratum(None) == 'unavailable'


def test_spearman_uses_average_ranks_for_ties():
    assert analysis.spearman([1, 2, 2, 4], [4, 2, 2, 1]) == pytest.approx(-1.)
    assert analysis.spearman([1, 1, 1], [1, 2, 3]) is None


def test_bootstrap_rejects_missing_or_nonfinite_data():
    with pytest.raises(ValueError):
        analysis.paired_cluster_bootstrap([], n_resamples=10, seed=1)
    with pytest.raises(ValueError):
        analysis.paired_cluster_bootstrap([
            {'subject': 'A', 'baseline_dice': float('nan'), 'transunet_dice': 1.}
        ], n_resamples=10, seed=1)
