"""Equation-level checks for the source-guided SINGER miniature."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from watchgen.mini_singer import (
    BranchState, F_bar_approx, branch_recombination_probability,
    branch_transition_prob, build_state_space, compute_arg_length_in_window,
    compute_scaling_factors, count_mutations_per_window,
    count_mutations_with_rate_variation, emission_probability, f_approx,
    forward_linearized, joining_prob_approx, joining_probability_exact,
    lambda_approx, lambda_inverse, partition_branch, partition_time_axis,
    poisson_edge_probability, psmc_transition_cdf, psmc_transition_density,
    recombination_time_median, representative_time, representative_times_ts,
    rescale_times, split_branch_transition, time_transition_matrix,
    type_b_transition, type_c_transition,
)


def test_exact_joining_probability_on_cherry_leaf():
    intervals = [(0, 1), (0, 1), (1, 3)]
    assert joining_probability_exact(0, 1, intervals) == pytest.approx(
        (1 - math.exp(-2)) / 2)


def test_exact_joining_probability_includes_prior_hazard():
    intervals = [(0, 1), (0, 1), (1, 3)]
    assert joining_probability_exact(1, 3, intervals) == pytest.approx(
        math.exp(-2) * (1 - math.exp(-2)))


def test_lineage_approximation_endpoints_and_density_identity():
    assert lambda_approx(0, 50) == pytest.approx(50)
    assert lambda_approx(100, 50) == pytest.approx(1)
    t = np.linspace(0.01, 4, 20)
    np.testing.assert_allclose(f_approx(t, 50),
                               lambda_approx(t, 50) * F_bar_approx(t, 50))


def test_lambda_inverse_round_trip():
    for time in (0.1, 0.7, 2.0):
        ell = float(lambda_approx(time, 20))
        assert lambda_inverse(ell, 20) == pytest.approx(time)


def test_representative_time_is_inside_branch():
    assert 0.2 < representative_time(0.2, 1.3, 100) < 1.3


def test_approx_joining_probability_is_integral_of_survival():
    direct, _ = quad(lambda t: F_bar_approx(t, 30), 0.2, 0.9)
    assert joining_prob_approx(0.2, 0.9, 30) == pytest.approx(direct)


def test_poisson_edge_probability_is_exactly_one_not_at_least_one():
    mean = 0.4
    got = poisson_edge_probability(1, length=2, theta=0.4)
    assert got == pytest.approx(mean * math.exp(-mean))
    assert got != pytest.approx(1 - math.exp(-mean))


def test_three_edge_emission_matches_manual_product():
    got = emission_probability(1, 0, 0, 0.5, 0.2, 1.0, 0.4)
    expected = (poisson_edge_probability(1, 0.5, 0.4) *
                poisson_edge_probability(0, 0.3, 0.4) *
                poisson_edge_probability(0, 0.5, 0.4))
    assert got == pytest.approx(expected)


def test_state_space_prunes_only_low_probability_partial_states():
    full = [BranchState(0, 2, 0, 1)]
    keep = BranchState(0, 3, 0, 0.5, True)
    drop = BranchState(3, 4, 0.5, 1, True)
    assert build_state_space(full, [(keep, 0.02), (drop, 0.005)]) == [full[0], keep]


def test_branch_transition_rows_sum_to_one():
    taus, p, rho = [0.3, 0.8, 1.5], [0.2, 0.3, 0.5], 0.4
    q = [branch_recombination_probability(t, rho) * mass
         for t, mass in zip(taus, p)]
    for i, tau_i in enumerate(taus):
        row = [branch_transition_prob(tau_i, tau_j, p_j, rho, False,
                                      sum(q), i == j)
               for j, (tau_j, p_j) in enumerate(zip(taus, p))]
        assert sum(row) == pytest.approx(1)


def test_partial_target_gets_no_new_recombination_mass():
    assert branch_transition_prob(0.5, 1, 0.3, 0.2, True, 0.5, False) == 0


def test_split_branch_weights_joining_mass():
    full = BranchState(0, 2, 0, 1)
    segments = [BranchState(0, 3, 0, 0.4, True),
                BranchState(3, 2, 0.4, 1, True)]
    weights = split_branch_transition(full, segments, 50)
    assert sum(weights) == pytest.approx(1)
    assert all(w > 0 for w in weights)


def test_time_partition_has_equal_exponential_mass():
    boundaries = partition_branch(0.2, 2, d=20)
    masses = np.exp(-boundaries[:-1]) - np.exp(-boundaries[1:])
    np.testing.assert_allclose(masses, masses[0])
    taus = representative_times_ts(boundaries)
    assert np.all((boundaries[:-1] < taus) & (taus < boundaries[1:]))


def test_psmc_cdf_has_correct_atom_and_limits():
    s, rho = 1.2, 0.3
    jump = psmc_transition_cdf(s, s, rho) - psmc_transition_cdf(
        np.nextafter(s, -np.inf), s, rho)
    assert jump == pytest.approx(math.exp(-rho * s))
    assert psmc_transition_cdf(100, s, rho) == pytest.approx(1)
    assert psmc_transition_density(s, s, rho) == pytest.approx(math.exp(-rho * s))


def test_time_transition_matrix_is_conditional_and_stochastic():
    boundaries = partition_branch(0.1, 2, 20)
    taus = representative_times_ts(boundaries)
    matrix = time_transition_matrix(boundaries, taus, boundaries, rho=0.2)
    np.testing.assert_allclose(matrix.sum(axis=1), 1)
    assert np.all(matrix >= 0)


def test_linearized_forward_equals_dense_forward():
    boundaries = partition_branch(0.1, 2, 20)
    taus = representative_times_ts(boundaries)
    matrix = time_transition_matrix(boundaries, taus, boundaries, rho=0.2)
    alpha = np.linspace(1, 2, 20)
    emissions = np.linspace(0.5, 1, 20)
    np.testing.assert_allclose(forward_linearized(alpha, matrix, emissions),
                               (alpha @ matrix) * emissions,
                               rtol=1e-11, atol=1e-13)


def test_type_b_transfers_only_compatible_intervals():
    got = type_b_transition(np.array([0.2, 0.3, 0.5]), None,
                            np.arange(5), [0, None, 3])
    np.testing.assert_allclose(got, [0.2, 0, 0, 0.5])


def test_type_c_is_conditioned_on_recombination():
    boundaries = partition_branch(0.1, 2, 20)
    taus = representative_times_ts(boundaries)
    got = type_c_transition(np.ones(20) / 20, taus, boundaries)
    assert got.sum() == pytest.approx(1)


def test_recombination_time_median_splits_truncated_mass():
    lower, upper = 0.2, 1.1
    median = recombination_time_median(lower, upper, 1.5)
    assert math.exp(median) - math.exp(lower) == pytest.approx(
        math.exp(upper) - math.exp(median))


def test_arg_length_and_equal_length_windows():
    branches = [(10, 0, 1), (10, 0, 1), (10, 1, 2)]
    assert compute_arg_length_in_window(branches, 0, 2) == pytest.approx(30)
    boundaries = partition_time_axis(branches, J=3)
    lengths = [compute_arg_length_in_window(branches, a, b)
               for a, b in zip(boundaries[:-1], boundaries[1:])]
    np.testing.assert_allclose(lengths, [10, 10, 10])


def test_fractional_mutation_mapping_and_scaling_equation():
    boundaries = np.array([0, 1, 2])
    counts = count_mutations_per_window([(0.5, 1.5)], boundaries)
    np.testing.assert_allclose(counts, [0.5, 0.5])
    np.testing.assert_allclose(compute_scaling_factors(counts, 20, 0.1, 2),
                               [1, 1])


def test_rescaling_is_continuous_and_piecewise_linear():
    got = rescale_times({0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2},
                        [0, 1, 2], [2, 0.5])
    assert got == pytest.approx({0: 0, 1: 1, 2: 2, 3: 2.25, 4: 2.5})


def test_rate_map_is_integrated_not_sampled_at_midpoint():
    expected, observed = count_mutations_with_rate_variation(
        [(0, 2, 0, 1)], [(1, 0, 1)], [0, 1], lambda x: x**2)
    assert expected[0] == pytest.approx(8 / 3)
    assert observed[0] == pytest.approx(1)


@pytest.mark.parametrize("call", [
    lambda: partition_branch(1, 0, 20),
    lambda: branch_recombination_probability(-1, 0.2),
    lambda: compute_scaling_factors([1], 0, 1, 1),
    lambda: recombination_time_median(0.5, 1, 0.8),
])
def test_invalid_parameters_raise(call):
    with pytest.raises(ValueError):
        call()
