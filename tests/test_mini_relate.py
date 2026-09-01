"""Independent invariants for the source-guided mini-Relate kernels."""

import math

import numpy as np
import pytest

from watchgen import mini_relate as relate


def balanced_tree():
    left = relate.TreeNode(4, relate.TreeNode(0), relate.TreeNode(1))
    right = relate.TreeNode(5, relate.TreeNode(2), relate.TreeNode(3))
    return relate.TreeNode(6, left, right)


@pytest.mark.parametrize(
    ("target", "reference", "expected"),
    [(1, 0, 0.025), (0, 0, 0.975), (1, 1, 0.975), (0, 1, 0.975)],
)
def test_modified_emission_is_one_sided(target, reference, expected):
    assert relate.modified_emission(target, reference, 0.025) == expected


def test_modified_emission_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        relate.modified_emission(2, 0, 0.025)
    with pytest.raises(ValueError):
        relate.modified_emission(1, 0, 0.75)


def test_copying_posterior_normalizes_every_site():
    target = np.array([0, 1, 0, 1])
    panel = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    posterior = relate.copying_posterior(target, panel, [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    assert np.all(posterior > 0.0)


def test_no_recombination_posterior_uses_whole_haplotype_likelihood():
    p = 0.1
    target = np.array([1, 1])
    panel = np.array([[1, 0], [1, 0]])
    posterior = relate.copying_posterior(target, panel, [0.0, 0.0], p)
    expected = np.array([(1 - p) ** 2, p**2])
    expected /= expected.sum()
    np.testing.assert_allclose(posterior[0], expected)
    np.testing.assert_allclose(posterior[1], expected)


def test_complete_switch_makes_local_emission_decisive():
    p = 0.1
    target = np.array([1, 1, 1])
    panel = np.array([[1, 0], [0, 1], [1, 0]])
    posterior = relate.copying_posterior(target, panel, [0.0, 1.0, 1.0], p)
    np.testing.assert_allclose(posterior[1], [p, 1 - p])


def test_relative_distance_is_row_centered_and_reverses_probability_order():
    distance = relate.relative_distance_row([0.8, 0.15, 0.05], mismatch=0.025)
    assert distance[0] == pytest.approx(0.0)
    assert distance[0] < distance[1] < distance[2]


def test_directional_mutation_distance_counts_derived_only():
    haplotypes = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    observed = relate.directional_mutation_distance(haplotypes)
    expected = np.array([[0, 1, 2], [0, 0, 1], [1, 1, 0]], dtype=float)
    np.testing.assert_array_equal(observed, expected)
    assert not np.array_equal(observed, observed.T)


def test_painting_distance_matrix_has_zero_row_minimum_off_diagonal():
    haplotypes = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    matrix = relate.painting_distance_matrix(haplotypes, [0, 0.1, 0.1], 1)
    for row in range(3):
        off_diagonal = np.delete(matrix[row], row)
        assert off_diagonal.min() == pytest.approx(0.0)


def test_cluster_distance_is_cardinality_weighted_mean():
    distance = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0]], dtype=float)
    assert relate.cluster_distance(distance, {0, 1}, {2}) == pytest.approx(3.0)
    assert relate.cluster_distance(distance, {2}, {0, 1}) == pytest.approx(5.5)


def test_pair_selection_requires_mutual_row_minima():
    distance = np.array([[0, 0, 1], [5, 0, 0], [1, 0.5, 0]], dtype=float)
    clusters = {i: frozenset({i}) for i in range(3)}
    assert relate.find_mutual_minimum_pair(distance, clusters, 0.0) == (1, 2)


def test_pair_selection_has_symmetrized_fallback_for_directional_cycle():
    distance = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    clusters = {i: frozenset({i}) for i in range(3)}
    assert relate.find_mutual_minimum_pair(distance, clusters, 0.0) == (0, 1)


def test_tree_builder_recovers_mutation_supported_clade():
    haplotypes = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    root = relate.build_tree(relate.directional_mutation_distance(haplotypes))
    clades = {node.leaves for node in relate.iter_nodes(root)}
    assert frozenset({0, 1}) in clades
    assert frozenset({2, 3}) in clades
    assert len(relate.iter_nodes(root)) == 7


def test_exact_mutation_mapping_and_flip():
    root = balanced_tree()
    assert relate.map_mutation_exact(root, {0, 1}) == (4, False)
    assert relate.map_mutation_exact(root, {0, 1}, allow_flip=False) == (4, False)
    assert relate.map_mutation_exact(root, {0, 2}) == (None, False)
    assert relate.map_mutation_exact(root, {0, 1, 2}) == (3, True)


def test_compatible_event_order_respects_descendants():
    root = balanced_tree()
    order = relate.compatible_event_orders(root)
    assert set(order) == {4, 5, 6}
    assert order.index(4) < order.index(6)
    assert order.index(5) < order.index(6)


def test_node_times_and_branch_lengths_are_ultrametric():
    root = balanced_tree()
    order = [4, 5, 6]
    times = relate.node_times_from_intervals(root, order, [0.2, 0.3, 0.5])
    assert times == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0.2, 5: 0.5, 6: 1.0}
    lengths = relate.branch_lengths(root, times)
    assert lengths[0] == pytest.approx(0.2)
    assert lengths[4] == pytest.approx(0.8)
    assert lengths[2] == pytest.approx(0.5)
    assert lengths[5] == pytest.approx(0.5)


def test_incompatible_event_order_is_rejected():
    with pytest.raises(ValueError):
        relate.node_times_from_intervals(balanced_tree(), [6, 4, 5], [0.2, 0.3, 0.5])


def test_standard_coalescent_interval_prior_uses_choose_two_rates():
    intervals = np.array([0.1, 0.2, 0.3])
    rates = np.array([6.0, 3.0, 1.0])
    expected = np.sum(np.log(rates) - rates * intervals)
    assert relate.log_coalescent_interval_prior(intervals, 4) == pytest.approx(expected)


def test_poisson_branch_likelihood_matches_direct_calculation():
    root = balanced_tree()
    times = relate.node_times_from_intervals(root, [4, 5, 6], [0.2, 0.3, 0.5])
    exposure = {branch: 2.0 for branch in relate.branch_lengths(root, times)}
    mutations = {branch: 1.0 for branch in exposure}
    theta = 0.5
    means = [theta * 2.0 * length / 2 for length in relate.branch_lengths(root, times).values()]
    expected = sum(math.log(mean) - mean for mean in means)
    assert relate.log_branch_mutation_likelihood(
        root, times, mutations, exposure, theta
    ) == pytest.approx(expected)


def test_ranked_tree_posterior_is_finite():
    root = balanced_tree()
    exposure = {branch: 1.0 for branch in relate._parent_map(root)}
    value = relate.log_ranked_tree_posterior(
        root, [4, 5, 6], [0.2, 0.3, 0.5], {0: 1}, exposure, 1.0
    )
    assert math.isfinite(value)


def test_sampler_is_seeded_and_returns_positive_intervals():
    root = balanced_tree()
    exposure = {branch: 1.0 for branch in relate._parent_map(root)}
    kwargs = dict(
        root=root,
        event_order=[4, 5, 6],
        mutations={0: 1, 1: 1, 2: 1},
        exposure=exposure,
        theta=1.0,
        iterations=300,
        burn_in=100,
        seed=7,
    )
    first, rate1 = relate.sample_ranked_branch_lengths(**kwargs)
    second, rate2 = relate.sample_ranked_branch_lengths(**kwargs)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (200, 3)
    assert np.all(first > 0.0)
    assert rate1 == rate2 and 0.0 < rate1 < 1.0


def test_piecewise_rate_mle_is_events_over_exposure():
    rates, events, exposure = relate.piecewise_coalescence_rate_mle(
        [0.2, 0.4, 1.2, 1.8], [0.0, 0.5, 1.0, 2.0]
    )
    np.testing.assert_allclose(events, [2, 0, 2])
    np.testing.assert_allclose(exposure, [1.6, 1.0, 1.0])
    np.testing.assert_allclose(rates, [1.25, 0.0, 2.0])


def test_piecewise_rate_requires_complete_epoch_coverage():
    with pytest.raises(ValueError):
        relate.piecewise_coalescence_rate_mle([2.0], [0.0, 1.0, 2.0])


def test_demo_runs_all_kernels():
    result = relate.demo(seed=3)
    assert result["newick"].endswith(";")
    assert np.all(np.asarray(result["mean_intervals"]) > 0.0)
    assert 0.0 < result["acceptance"] < 1.0
    np.testing.assert_allclose(result["coalescence_rates"], [1.25, 0.0, 2.0])
