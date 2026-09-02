"""Paper-, source-, and brute-force-derived tests for mini_clues."""

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp

from watchgen.mini_clues import (
    backward_mean,
    backward_std,
    build_frequency_bins,
    build_normal_cdf_lookup,
    build_transition_matrix,
    build_transition_matrix_fast,
    compute_trajectory_summary,
    forward_log_hmm,
    genotype_likelihood_emission,
    haplotype_likelihood_emission,
    importance_log_likelihood_ratio,
    internal_diploid_size,
    likelihood_ratio_test,
    log_coalescent_density,
)


def test_public_haploid_size_is_halved_for_internal_kernels():
    assert internal_diploid_size(20_000) == 10_000


def test_backward_mean_matches_clues2_source_fixture():
    assert backward_mean(0.3, 0.02, 0.5) == pytest.approx(0.29791252485089464)
    assert backward_mean(0.3, 0.0, 0.5) == 0.3


def test_backward_standard_deviation_matches_paper_parameterization():
    assert backward_std(0.3, 20_000) == pytest.approx(
        np.sqrt(0.3 * 0.7 / 20_000)
    )


def test_frequency_grid_and_safe_logs():
    frequencies, log_frequencies, log_complements = build_frequency_bins(9)
    assert frequencies == pytest.approx([
        0.0, 0.0380602337443566, 0.1464466094067262,
        0.3086582838174551, 0.5, 0.6913417161825449,
        0.8535533905932737, 0.9619397662556434, 1.0,
    ])
    assert np.all(np.isfinite(log_frequencies))
    assert np.all(np.isfinite(log_complements))


def test_source_transition_fixture_commit_b20dc5d():
    frequencies, _, _ = build_frequency_bins(9)
    log_matrix = build_transition_matrix(frequencies, 200.0, 0.02)
    expected = np.array([
        0.0, 0.0, 0.0, 0.0042011496622390, 0.9930368885174641,
        0.0027619618202968, 0.0, 0.0, 0.0,
    ])
    assert np.exp(log_matrix[4]) == pytest.approx(expected, abs=2e-15)


def test_transition_rows_normalize_and_boundaries_absorb():
    frequencies, _, _ = build_frequency_bins(31)
    probability = np.exp(build_transition_matrix(frequencies, 20_000, 0.01))
    assert probability.sum(axis=1) == pytest.approx(np.ones(31))
    assert probability[0, 0] == probability[-1, -1] == 1


def test_fast_compatibility_wrapper_returns_finite_bands():
    frequencies, _, _ = build_frequency_bins(15)
    z, cdf = build_normal_cdf_lookup()
    matrix, lower, upper = build_transition_matrix_fast(
        frequencies, 2_000, 0.01, z, cdf
    )
    for i in range(len(frequencies)):
        finite = np.flatnonzero(np.isfinite(matrix[i]))
        assert (lower[i], upper[i]) == (finite[0], finite[-1] + 1)


def test_source_style_coalescent_fixture_commit_b20dc5d():
    score = log_coalescent_density(
        np.array([0.25]), 3, 0.0, 1.0, 0.3, 200.0
    )
    assert score == pytest.approx(1.1789728043259362)


def test_coalescent_constants_are_intentionally_not_absolute_density():
    one = log_coalescent_density(np.array([0.25]), 3, 0, 1, 0.3, 200)
    two = log_coalescent_density(np.array([0.25]), 3, 0, 1, 0.6, 200)
    assert one != two
    with pytest.raises(ValueError, match="sorted"):
        log_coalescent_density(np.array([0.5, 0.2]), 3, 0, 1, 0.3, 200)


def test_genotype_emission_matches_source_fixture():
    result = genotype_likelihood_emission(
        np.log([0.01, 0.98, 0.01]), np.log(0.6), np.log(0.4)
    )
    assert result == pytest.approx(-0.7431781141655104)


def test_haplotype_emission_has_expected_mixture():
    result = haplotype_likelihood_emission(
        np.log([0.1, 0.9]), np.log(0.6), np.log(0.4)
    )
    assert np.exp(result) == pytest.approx(0.4 * 0.1 + 0.6 * 0.9)


def test_forward_recursion_matches_path_enumeration():
    initial = np.log([0.4, 0.6])
    transition = np.log([[0.8, 0.2], [0.3, 0.7]])
    emission = np.log([[0.9, 0.2], [0.5, 0.4], [0.1, 0.8]])
    observed, _ = forward_log_hmm(initial, transition, emission)
    terms = []
    for path in product(range(2), repeat=3):
        term = initial[path[0]] + emission[0, path[0]]
        term += transition[path[0], path[1]] + emission[1, path[1]]
        term += transition[path[1], path[2]] + emission[2, path[2]]
        terms.append(term)
    assert observed == pytest.approx(logsumexp(terms))


def test_importance_aggregation_is_log_mean_of_tree_ratios():
    selected = np.log([0.2, 0.6, 0.4])
    neutral = np.log([0.1, 0.3, 0.8])
    observed = importance_log_likelihood_ratio(selected, neutral)
    assert np.exp(observed) == pytest.approx(np.mean([2.0, 2.0, 0.5]))


def test_likelihood_ratio_convention_doubles_delta_log_likelihood():
    statistic, p_value, neg_log10 = likelihood_ratio_test(12.0, 10.0)
    assert statistic == 4.0
    assert 0 < p_value < 1
    assert neg_log10 == pytest.approx(-np.log10(p_value))


def test_trajectory_summary_normalizes_columns():
    frequencies = np.array([0.0, 0.5, 1.0])
    posterior = np.array([[1, 0], [2, 2], [1, 0]], dtype=float)
    mean, lower, upper = compute_trajectory_summary(posterior, frequencies)
    assert mean == pytest.approx([0.5, 0.5])
    assert np.all(lower <= mean)
    assert np.all(mean <= upper)
