import itertools

import numpy as np
import pytest

from watchgen.mini_phlash import (
    afs_log_score,
    coalescence_probabilities,
    composite_log_density,
    effective_population_size,
    fisher_transition_score,
    forward_log_likelihood,
    logarithmic_grid,
    rbf_kernel,
    structured_smc_matvec,
    svgd_direction,
)


def _dense_from_structure(lower, diagonal, upper_scale, column_scale):
    m = len(diagonal)
    matrix = np.zeros((m, m))
    for i in range(m):
        matrix[i, i] = diagonal[i]
        matrix[i + 1 :, i] = lower[i]
        matrix[i, i + 1 :] = upper_scale[i] * column_scale[i + 1 :]
    return matrix


def _brute_force_score(observations, transition, emission, initial, feature):
    likelihood = 0.0
    weighted_feature = 0.0
    m = len(initial)
    for path in itertools.product(range(m), repeat=len(observations) + 1):
        weight = initial[path[0]]
        total = 0.0
        for t, symbol in enumerate(observations):
            i, j = path[t : t + 2]
            weight *= transition[i, j] * emission[j, symbol]
            total += feature[i, j]
        likelihood += weight
        weighted_feature += weight * total
    return np.log(likelihood), weighted_feature / likelihood


def test_rate_parameterisation_matches_paper():
    np.testing.assert_allclose(effective_population_size([1 / 20, 1 / 100]), [10, 50])
    with pytest.raises(ValueError):
        effective_population_size([0])


def test_coalescence_masses_match_integrated_hazard():
    times = [0, 1, 3]
    rates = [0.2, 0.5, 3.0]
    got = coalescence_probabilities(times, rates)
    expected = [1 - np.exp(-0.2), np.exp(-0.2) - np.exp(-1.2), np.exp(-1.2)]
    np.testing.assert_allclose(got, expected)
    assert got.sum() == pytest.approx(1)


def test_grid_is_geometric_between_random_endpoints():
    grid = logarithmic_grid(np.log(1e-4), np.log(15), intervals=16)
    assert len(grid) == 16
    assert grid[0] == 0
    np.testing.assert_allclose(np.diff(np.log(grid[1:])), np.diff(np.log(grid[1:]))[0])


def test_afs_score_matches_released_source_expression():
    observed = np.array([12, 7, 3])
    expected = np.array([5, 3, 2])
    expected_value = np.sum(observed * np.log(expected / expected.sum()))
    assert afs_log_score(observed, expected) == pytest.approx(expected_value)
    transform = np.array([[1, 0, 1], [0, 1, 0]])
    transformed = transform @ (expected / expected.sum())
    expected_transformed = np.sum((transform @ observed) * np.log(transformed))
    assert afs_log_score(observed, expected, transform) == pytest.approx(expected_transformed)


def test_composite_weights_match_minibatch_scaling_contract():
    got = composite_log_density(-2, [-3, -4], -5, sequence_weight=10)
    assert got == -77


def test_structured_matvec_matches_dense_transition():
    vector = np.array([0.2, 0.3, 0.1, 0.4])
    lower = np.array([0.05, 0.07, 0.08, 0.0])
    diagonal = np.array([0.6, 0.5, 0.55, 0.7])
    upper_scale = np.array([0.15, 0.12, 0.09, 0.0])
    column_scale = np.array([0.0, 1.0, 0.8, 0.6])
    dense = _dense_from_structure(lower, diagonal, upper_scale, column_scale)
    np.testing.assert_allclose(
        structured_smc_matvec(vector, lower, diagonal, upper_scale, column_scale),
        vector @ dense,
    )


def test_forward_likelihood_matches_enumeration():
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.6, 0.4])
    observations = np.array([0, 1, 1])
    zeros = np.zeros_like(transition)
    expected, _ = _brute_force_score(observations, transition, emission, initial, zeros)
    assert forward_log_likelihood(observations, transition, emission, initial) == pytest.approx(expected)


def test_linear_memory_fisher_score_matches_path_enumeration():
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.6, 0.4])
    observations = np.array([0, 1, 1])
    feature = np.array([[0.0, 1.5], [-0.25, 0.4]])
    expected = _brute_force_score(observations, transition, emission, initial, feature)
    got = fisher_transition_score(observations, transition, emission, initial, feature)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-13)


def test_fisher_score_is_log_likelihood_derivative():
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emission = np.array([[0.9, 0.1], [0.2, 0.8]])
    initial = np.array([0.6, 0.4])
    observations = np.array([0, 1, 1])
    feature = np.array([[0.0, 1.0], [0.0, 0.0]])
    _, score = fisher_transition_score(observations, transition, emission, initial, feature)
    eps = 1e-6
    plus = transition * np.exp(eps * feature)
    minus = transition * np.exp(-eps * feature)
    finite_difference = (
        forward_log_likelihood(observations, plus, emission, initial)
        - forward_log_likelihood(observations, minus, emission, initial)
    ) / (2 * eps)
    assert score == pytest.approx(finite_difference, rel=1e-8)


def test_rbf_kernel_is_symmetric_and_unit_diagonal():
    particles = np.array([[-1.0], [0.0], [2.0]])
    kernel, bandwidth = rbf_kernel(particles)
    np.testing.assert_allclose(kernel, kernel.T)
    np.testing.assert_allclose(np.diag(kernel), 1)
    assert bandwidth > 0


def test_svgd_repels_two_particles_when_score_is_zero():
    particles = np.array([[-1.0], [1.0]])
    direction = svgd_direction(particles, np.zeros_like(particles), bandwidth=1)
    assert direction[0, 0] < 0
    assert direction[1, 0] > 0
    np.testing.assert_allclose(direction.sum(axis=0), 0)
