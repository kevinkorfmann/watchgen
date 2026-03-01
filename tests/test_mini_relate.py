"""
Tests for watchgen.mini_relate -- the Relate algorithm implementation.

Covers:
- asymmetric_painting.rst: directional emission, forward-backward, distance matrix
- tree_building.rst: find_pair_to_merge, update_distances, build_tree, to_newick
- branch_lengths.rst: mutation mapping, Poisson likelihood, coalescent prior, MCMC
- population_size.rst: epoch construction, integrated rate, M-step
"""

import numpy as np
import pytest
from scipy.special import gammaln

from watchgen.mini_relate import (
    directional_emission,
    forward_backward_relate,
    compute_distance_matrix,
    TreeNode,
    find_pair_to_merge,
    update_distances,
    build_tree,
    to_newick,
    build_local_trees,
    get_descendants,
    map_mutations,
    log_mutation_likelihood,
    log_coalescent_prior,
    log_posterior,
    mcmc_branch_lengths,
    posterior_summary,
    make_epochs,
    integrated_rate,
    log_coalescent_prior_variable,
    m_step,
    em_population_size,
)


# ===========================================================================
# Tests for directional_emission
# ===========================================================================

class TestDirectionalEmission:
    def test_match_probability(self):
        """Match should give 1 - mu."""
        assert np.isclose(directional_emission(0, 0, 0.01), 0.99)
        assert np.isclose(directional_emission(1, 1, 0.01), 0.99)

    def test_asymmetry(self):
        """Target-derived and target-ancestral mismatches should differ."""
        e_d = directional_emission(1, 0, 0.01)
        e_a = directional_emission(0, 1, 0.01)
        assert e_d != e_a

    def test_mismatch_values(self):
        """Mismatch probabilities should follow w_d and w_a."""
        mu = 0.01
        e_d = directional_emission(1, 0, mu, w_d=1.0, w_a=0.5)
        e_a = directional_emission(0, 1, mu, w_d=1.0, w_a=0.5)
        assert np.isclose(e_d, 0.01)
        assert np.isclose(e_a, 0.005)

    def test_symmetric_weights(self):
        """Equal weights should give symmetric emissions."""
        e_d = directional_emission(1, 0, 0.01, w_d=1.0, w_a=1.0)
        e_a = directional_emission(0, 1, 0.01, w_d=1.0, w_a=1.0)
        assert np.isclose(e_d, e_a)


# ===========================================================================
# Tests for forward_backward_relate
# ===========================================================================

class TestForwardBackwardRelate:
    def test_posterior_sums_to_one(self):
        """Posterior should sum to 1 at each site."""
        np.random.seed(42)
        K, L = 5, 20
        panel = np.random.binomial(1, 0.3, size=(L, K))
        target = np.random.binomial(1, 0.3, size=L)
        rho = np.full(L, 0.05)
        rho[0] = 0.0
        posterior = forward_backward_relate(target, panel, rho, mu=0.01)
        assert np.allclose(posterior.sum(axis=1), 1.0, atol=1e-6)

    def test_output_shape(self):
        """Posterior should have shape (L, K)."""
        np.random.seed(42)
        K, L = 5, 20
        panel = np.random.binomial(1, 0.3, size=(L, K))
        target = np.random.binomial(1, 0.3, size=L)
        rho = np.full(L, 0.05)
        rho[0] = 0.0
        posterior = forward_backward_relate(target, panel, rho, mu=0.01)
        assert posterior.shape == (L, K)

    def test_nonnegative(self):
        """Posterior probabilities should be non-negative."""
        np.random.seed(42)
        K, L = 5, 10
        panel = np.random.binomial(1, 0.3, size=(L, K))
        target = np.random.binomial(1, 0.3, size=L)
        rho = np.full(L, 0.05)
        rho[0] = 0.0
        posterior = forward_backward_relate(target, panel, rho, mu=0.01)
        assert np.all(posterior >= -1e-10)


# ===========================================================================
# Tests for find_pair_to_merge
# ===========================================================================

class TestFindPairToMerge:
    def test_closest_pair(self):
        """Should find the pair with smallest min distance."""
        D = np.array([
            [0.0, 1.2, 3.5, 3.8],
            [0.8, 0.0, 3.2, 3.5],
            [3.5, 3.2, 0.0, 0.5],
            [3.8, 3.5, 0.7, 0.0],
        ])
        i, j = find_pair_to_merge(D, {0, 1, 2, 3})
        assert {i, j} == {2, 3}

    def test_symmetric_case(self):
        """Symmetric distances should still find a valid pair."""
        D = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ])
        i, j = find_pair_to_merge(D, {0, 1, 2})
        assert {i, j} == {0, 1}


# ===========================================================================
# Tests for build_tree
# ===========================================================================

class TestBuildTree:
    def test_correct_topology(self):
        """Should cluster close pairs first."""
        D = np.array([
            [0.0, 1.2, 3.5, 3.8],
            [0.8, 0.0, 3.2, 3.5],
            [3.5, 3.2, 0.0, 0.5],
            [3.8, 3.5, 0.7, 0.0],
        ])
        root, merges = build_tree(D, N=4)
        # First merge should be (2, 3) since they have smallest d_min
        assert {merges[0][0], merges[0][1]} == {2, 3}

    def test_root_has_all_leaves(self):
        """Root should contain all leaf IDs."""
        D = np.array([
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 2.0, 3.0],
            [2.0, 2.0, 0.0, 1.0],
            [3.0, 3.0, 1.0, 0.0],
        ])
        root, _ = build_tree(D, N=4)
        assert root.leaf_ids == {0, 1, 2, 3}

    def test_n_minus_1_merges(self):
        """Should produce exactly N-1 merges."""
        N = 5
        D = np.random.rand(N, N) + 1.0
        np.fill_diagonal(D, 0.0)
        _, merges = build_tree(D, N)
        assert len(merges) == N - 1

    def test_root_not_leaf(self):
        """Root should not be a leaf."""
        D = np.array([[0.0, 1.0], [1.0, 0.0]])
        root, _ = build_tree(D, N=2)
        assert not root.is_leaf


# ===========================================================================
# Tests for to_newick
# ===========================================================================

class TestToNewick:
    def test_single_leaf(self):
        """Single leaf should give just its ID."""
        node = TreeNode(0)
        assert to_newick(node) == "0"

    def test_simple_tree(self):
        """Simple tree should give valid Newick."""
        left = TreeNode(0)
        right = TreeNode(1)
        root = TreeNode(2, left=left, right=right, is_leaf=False)
        newick = to_newick(root)
        assert newick == "(0,1)"

    def test_nested_tree(self):
        """Nested tree should produce nested Newick."""
        leaf0 = TreeNode(0)
        leaf1 = TreeNode(1)
        leaf2 = TreeNode(2)
        inner = TreeNode(3, left=leaf0, right=leaf1, is_leaf=False)
        root = TreeNode(4, left=inner, right=leaf2, is_leaf=False)
        newick = to_newick(root)
        assert newick == "((0,1),2)"


# ===========================================================================
# Tests for map_mutations
# ===========================================================================

class TestMapMutations:
    def test_known_tree(self):
        """Mutations should map correctly to a known tree."""
        leaf0 = TreeNode(0)
        leaf1 = TreeNode(1)
        leaf2 = TreeNode(2)
        leaf3 = TreeNode(3)
        node4 = TreeNode(4, left=leaf0, right=leaf1, is_leaf=False)
        node4.leaf_ids = {0, 1}
        node5 = TreeNode(5, left=node4, right=leaf2, is_leaf=False)
        node5.leaf_ids = {0, 1, 2}
        root = TreeNode(6, left=node5, right=leaf3, is_leaf=False)
        root.leaf_ids = {0, 1, 2, 3}

        haps = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        branch_muts, unmapped = map_mutations(root, haps, list(range(4)))
        # Site 0: carriers = {0, 1} -> branch (4, ?) where descendants = {0,1}
        assert branch_muts[(4, 0)] == 1 or branch_muts[(4, 1)] == 0
        # No unmapped mutations for compatible data
        assert unmapped == 0

    def test_monomorphic_skipped(self):
        """Monomorphic sites should be skipped."""
        leaf0 = TreeNode(0)
        leaf1 = TreeNode(1)
        root = TreeNode(2, left=leaf0, right=leaf1, is_leaf=False)
        root.leaf_ids = {0, 1}
        haps = np.array([[0, 0], [0, 0]])
        branch_muts, unmapped = map_mutations(root, haps, [0, 1])
        total = sum(branch_muts.values())
        assert total == 0
        assert unmapped == 0


# ===========================================================================
# Tests for log_mutation_likelihood
# ===========================================================================

class TestLogMutationLikelihood:
    def test_finite_for_valid_input(self):
        """Should return finite value for valid inputs."""
        branch_muts = {(2, 0): 3, (2, 1): 2}
        node_times = {0: 0, 1: 0, 2: 500}
        ll = log_mutation_likelihood(branch_muts, node_times,
                                     mu=1.25e-8, span=1e4)
        assert np.isfinite(ll)

    def test_neg_inf_for_invalid_times(self):
        """Should return -inf if parent younger than child."""
        branch_muts = {(0, 1): 1}
        node_times = {0: 0, 1: 100}  # parent younger
        ll = log_mutation_likelihood(branch_muts, node_times,
                                     mu=1.25e-8, span=1e4)
        assert ll == -np.inf

    def test_more_mutations_longer_branch(self):
        """More mutations should favor longer branches."""
        branch_muts_3 = {(1, 0): 3}
        branch_muts_10 = {(1, 0): 10}
        # Longer branch should be preferred for 10 mutations
        node_times_short = {0: 0, 1: 100}
        node_times_long = {0: 0, 1: 1000}
        ll_10_long = log_mutation_likelihood(branch_muts_10, node_times_long,
                                              mu=1.25e-8, span=1e4)
        ll_10_short = log_mutation_likelihood(branch_muts_10, node_times_short,
                                               mu=1.25e-8, span=1e4)
        # At 10 mutations, longer branch is more likely
        assert ll_10_long > ll_10_short


# ===========================================================================
# Tests for log_coalescent_prior
# ===========================================================================

class TestLogCoalescentPrior:
    def test_finite(self):
        """Prior should be finite for valid times."""
        coal_times = [100, 300, 500]
        lp = log_coalescent_prior(coal_times, N_e=10000)
        assert np.isfinite(lp)

    def test_negative(self):
        """Log-prior should be negative (probability < 1)."""
        coal_times = [100, 300, 500]
        lp = log_coalescent_prior(coal_times, N_e=10000)
        assert lp < 0

    def test_out_of_order_returns_neg_inf(self):
        """Non-sorted times should return -inf."""
        coal_times = [500, 100, 300]  # not sorted
        lp = log_coalescent_prior(coal_times, N_e=10000)
        assert lp == -np.inf

    def test_smaller_ne_faster_coalescence(self):
        """Smaller N_e should assign higher prior to early coalescences."""
        early_times = [10, 20, 30]
        lp_small = log_coalescent_prior(early_times, N_e=100)
        lp_large = log_coalescent_prior(early_times, N_e=100000)
        assert lp_small > lp_large


# ===========================================================================
# Tests for make_epochs
# ===========================================================================

class TestMakeEpochs:
    def test_starts_at_zero(self):
        """First boundary should be 0."""
        boundaries = make_epochs(100000, 20)
        assert boundaries[0] == 0.0

    def test_ends_at_max_time(self):
        """Last boundary should equal max_time."""
        boundaries = make_epochs(100000, 20)
        assert np.isclose(boundaries[-1], 100000)

    def test_length(self):
        """Should have n_epochs + 1 boundaries."""
        boundaries = make_epochs(100000, 20)
        assert len(boundaries) == 21

    def test_sorted(self):
        """Boundaries should be strictly increasing."""
        boundaries = make_epochs(100000, 20)
        assert np.all(np.diff(boundaries) > 0)


# ===========================================================================
# Tests for integrated_rate
# ===========================================================================

class TestIntegratedRate:
    def test_constant_ne(self):
        """With constant N_e, integral should be dt / N_e."""
        boundaries = make_epochs(100000, 10)
        N_e_values = np.full(10, 10000.0)
        result = integrated_rate(0, 1000, boundaries, N_e_values)
        assert np.isclose(result, 1000.0 / 10000.0, rtol=0.01)

    def test_zero_interval(self):
        """Zero-length interval should give 0."""
        boundaries = make_epochs(100000, 10)
        N_e_values = np.full(10, 10000.0)
        result = integrated_rate(100, 100, boundaries, N_e_values)
        assert result == 0.0


# ===========================================================================
# Tests for m_step
# ===========================================================================

class TestMStep:
    def test_constant_ne_recovery(self):
        """M-step should recover approximately constant N_e."""
        np.random.seed(42)
        true_N_e = 10000
        n_trees = 200
        n_leaves = 8
        coal_times_all = []
        for _ in range(n_trees):
            times = []
            prev_t = 0.0
            for k in range(n_leaves, 1, -1):
                rate = k * (k - 1) / (2.0 * true_N_e)
                dt = np.random.exponential(1.0 / rate)
                prev_t += dt
                times.append(prev_t)
            coal_times_all.append(times)
        boundaries = make_epochs(100000, 10)
        spans = np.full(n_trees, 1e4)
        N_e_est = m_step(coal_times_all, [n_leaves] * n_trees,
                          boundaries, spans)
        mean_est = np.nanmean(N_e_est)
        rel_error = abs(mean_est - true_N_e) / true_N_e
        assert rel_error < 0.3

    def test_positive_estimates(self):
        """All N_e estimates should be positive."""
        np.random.seed(42)
        coal_times = []
        for _ in range(50):
            times = sorted(np.random.exponential(5000, size=3))
            coal_times.append(list(np.cumsum(times)))
        boundaries = make_epochs(100000, 5)
        spans = np.full(50, 1e4)
        N_e_est = m_step(coal_times, [4] * 50, boundaries, spans)
        assert np.all(N_e_est > 0)
