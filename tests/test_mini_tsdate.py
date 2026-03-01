"""
Tests for watchgen.mini_tsdate -- the tsdate mini-implementation.

Imports all functions from watchgen.mini_tsdate and tests them.
Adapts test logic from tests/test_timepieces_tsdate_inside_outside.py,
importing functions rather than redefining them.

Covers:
- make_time_grid (time discretization for inside-outside algorithm)
- edge_likelihood_matrix (Poisson likelihood on a time grid)
- compute_posteriors (combining inside and outside values)
- posterior_mean (computing E[t | D] from a posterior distribution)
- inside_pass_logspace (single inside message in log space)
- conditional_coalescent_moments (coalescent prior moments)
- gamma_params_from_moments (moment matching to gamma parameters)
- build_prior_grid (prior lookup table)
- edge_likelihood (single-edge Poisson likelihood)
- gamma_poisson_update (conjugate update)
- GammaDistribution (natural parameterization class)
- numerical_hessian (finite-difference Hessian)
- compute_tilted_moments (Laplace approximation for EP)
- compute_scaling_factors (rescaling)
- apply_rescaling (piecewise time rescaling)
"""

import numpy as np
import pytest
from scipy.stats import poisson, gamma as gamma_dist
from scipy.special import logsumexp

from watchgen.mini_tsdate import (
    # Chapter 1: Coalescent Prior
    conditional_coalescent_mean,
    conditional_coalescent_moments,
    gamma_params_from_moments,
    build_prior_grid,
    _transition_prob,
    # Chapter 2: Mutation Likelihood
    edge_likelihood,
    gamma_poisson_update,
    # Chapter 3: Inside-Outside
    make_time_grid,
    edge_likelihood_matrix,
    compute_posteriors,
    posterior_mean,
    inside_pass_logspace,
    # Chapter 4: Variational Gamma
    GammaDistribution,
    numerical_hessian,
    compute_tilted_moments,
    # Chapter 5: Rescaling
    compute_scaling_factors,
    apply_rescaling,
)


# =========================================================================
# Tests for make_time_grid
# =========================================================================

class TestMakeTimeGrid:
    """Tests for the make_time_grid function."""

    def test_starts_at_zero(self):
        """Both linear and logarithmic grids should start at 0."""
        for grid_type in ["linear", "logarithmic"]:
            grid = make_time_grid(n=50, num_points=20, grid_type=grid_type)
            assert grid[0] == 0.0

    def test_correct_number_of_points(self):
        """Grid should have the requested number of points."""
        for num_points in [5, 10, 20, 50]:
            grid = make_time_grid(n=50, num_points=num_points)
            assert len(grid) == num_points

    def test_linear_grid_equally_spaced(self):
        """Linear grid points should be equally spaced."""
        grid = make_time_grid(n=50, num_points=20, grid_type="linear")
        diffs = np.diff(grid)
        assert np.allclose(diffs, diffs[0])

    def test_grid_is_sorted(self):
        """Grid should be strictly increasing."""
        for grid_type in ["linear", "logarithmic"]:
            grid = make_time_grid(n=50, num_points=20, grid_type=grid_type)
            assert np.all(np.diff(grid) > 0)

    def test_all_values_non_negative(self):
        """All grid values should be non-negative."""
        for grid_type in ["linear", "logarithmic"]:
            grid = make_time_grid(n=50, num_points=20, grid_type=grid_type)
            assert np.all(grid >= 0)

    def test_max_is_4x_expected_tmrca(self):
        """Grid maximum should be 4 * expected TMRCA = 4 * 2 * Ne * (1 - 1/n)."""
        n, Ne = 100, 1.0
        expected_tmrca = 2 * Ne * (1 - 1.0 / n)
        t_max = expected_tmrca * 4

        grid_lin = make_time_grid(n=n, Ne=Ne, num_points=20, grid_type="linear")
        assert grid_lin[-1] == pytest.approx(t_max)

        grid_log = make_time_grid(n=n, Ne=Ne, num_points=20, grid_type="logarithmic")
        assert grid_log[-1] == pytest.approx(t_max)

    def test_ne_scales_grid(self):
        """Changing Ne should scale the grid proportionally."""
        grid1 = make_time_grid(n=50, Ne=1.0, num_points=10, grid_type="linear")
        grid2 = make_time_grid(n=50, Ne=2.0, num_points=10, grid_type="linear")
        np.testing.assert_allclose(grid2, grid1 * 2.0)

    def test_log_grid_denser_near_zero(self):
        """Logarithmic grid should have more points near 0 than a linear grid."""
        grid_lin = make_time_grid(n=50, num_points=20, grid_type="linear")
        grid_log = make_time_grid(n=50, num_points=20, grid_type="logarithmic")
        assert np.diff(grid_log)[0] < np.diff(grid_lin)[0]

    def test_large_n_grid_approaches_max_8(self):
        """For large n and Ne=1, t_max should approach 4 * 2 * 1 = 8."""
        grid = make_time_grid(n=10000, Ne=1.0, num_points=20, grid_type="linear")
        assert grid[-1] == pytest.approx(8.0, rel=0.01)

    def test_n_equals_2(self):
        """For n=2, expected TMRCA = 2*Ne*(1 - 1/2) = Ne."""
        Ne = 1.0
        grid = make_time_grid(n=2, Ne=Ne, num_points=10, grid_type="linear")
        expected_max = 4 * 2 * Ne * (1 - 0.5)
        assert grid[-1] == pytest.approx(expected_max)


# =========================================================================
# Tests for edge_likelihood_matrix
# =========================================================================

class TestEdgeLikelihoodMatrix:
    """Tests for the edge_likelihood_matrix function."""

    def test_shape(self):
        """Output should be K x K."""
        grid = np.linspace(0, 5, 10)
        L = edge_likelihood_matrix(m_e=1, lambda_e=0.5, grid=grid)
        assert L.shape == (10, 10)

    def test_lower_triangular(self):
        """Matrix should be lower triangular (L[i,j]=0 for j>i)."""
        grid = np.linspace(0, 5, 10)
        L = edge_likelihood_matrix(m_e=1, lambda_e=0.5, grid=grid)
        for i in range(10):
            for j in range(i + 1, 10):
                assert L[i, j] == 0.0

    def test_all_values_non_negative(self):
        """All likelihood values should be non-negative."""
        grid = np.linspace(0, 5, 10)
        L = edge_likelihood_matrix(m_e=2, lambda_e=0.5, grid=grid)
        assert np.all(L >= 0)

    def test_zero_mutations_diagonal(self):
        """With m_e=0, L[i,i] should be 1.0 (delta_t=0, no mutations expected)."""
        grid = np.linspace(0, 5, 10)
        L = edge_likelihood_matrix(m_e=0, lambda_e=0.5, grid=grid)
        for i in range(10):
            assert L[i, i] == 1.0

    def test_nonzero_mutations_diagonal_zero(self):
        """With m_e>0, L[i,i] should be 0.0 (can't have mutations in zero time)."""
        grid = np.linspace(0, 5, 10)
        L = edge_likelihood_matrix(m_e=1, lambda_e=0.5, grid=grid)
        for i in range(10):
            assert L[i, i] == 0.0

    def test_poisson_values(self):
        """Verify entries match the Poisson PMF directly."""
        grid = np.array([0.0, 1.0, 2.0, 3.0])
        lambda_e = 0.5
        m_e = 2
        L = edge_likelihood_matrix(m_e=m_e, lambda_e=lambda_e, grid=grid)

        expected_val = poisson.pmf(2, 1.0)
        assert L[3, 1] == pytest.approx(expected_val)

        expected_val = poisson.pmf(2, 1.0)
        assert L[2, 0] == pytest.approx(expected_val)

    def test_likelihood_peaks_at_correct_delta_t(self):
        """The maximum likelihood should occur near delta_t = m_e / lambda_e."""
        m_e = 5
        lambda_e = 1.0
        grid = np.linspace(0, 20, 100)
        L = edge_likelihood_matrix(m_e=m_e, lambda_e=lambda_e, grid=grid)

        max_idx = np.unravel_index(L.argmax(), L.shape)
        delta_t_at_max = grid[max_idx[0]] - grid[max_idx[1]]
        expected_delta = m_e / lambda_e

        assert delta_t_at_max == pytest.approx(expected_delta, abs=0.5)

    def test_zero_mutations_monotone_decrease(self):
        """With m_e=0, first column values should decrease with increasing parent time."""
        grid = np.linspace(0, 5, 20)
        L = edge_likelihood_matrix(m_e=0, lambda_e=1.0, grid=grid)
        first_col = L[:, 0]
        for i in range(2, len(first_col)):
            assert first_col[i] <= first_col[i - 1]

    def test_high_mutation_rate_concentrates_likelihood(self):
        """Higher mutation rate should concentrate likelihood at smaller delta_t."""
        grid = np.linspace(0, 10, 50)
        L_low = edge_likelihood_matrix(m_e=3, lambda_e=0.1, grid=grid)
        L_high = edge_likelihood_matrix(m_e=3, lambda_e=10.0, grid=grid)

        max_low = np.unravel_index(L_low.argmax(), L_low.shape)
        max_high = np.unravel_index(L_high.argmax(), L_high.shape)
        delta_low = grid[max_low[0]] - grid[max_low[1]]
        delta_high = grid[max_high[0]] - grid[max_high[1]]

        assert delta_high < delta_low

    def test_small_grid(self):
        """Test with a minimal 2-point grid."""
        grid = np.array([0.0, 1.0])
        L = edge_likelihood_matrix(m_e=0, lambda_e=1.0, grid=grid)
        assert L.shape == (2, 2)
        assert L[0, 0] == 1.0
        assert L[1, 0] == pytest.approx(np.exp(-1.0))
        assert L[1, 1] == 1.0


# =========================================================================
# Tests for compute_posteriors
# =========================================================================

class TestComputePosteriors:
    """Tests for the compute_posteriors function."""

    def test_output_rows_sum_to_one(self):
        """Each node's posterior should be a valid probability distribution."""
        np.random.seed(42)
        K = 10
        inside = np.random.rand(5, K) + 0.01
        outside = np.random.rand(5, K) + 0.01
        post = compute_posteriors(inside, outside)

        for u in range(5):
            assert post[u, :].sum() == pytest.approx(1.0)

    def test_all_values_non_negative(self):
        """Posterior values should all be non-negative."""
        np.random.seed(42)
        K = 10
        inside = np.random.rand(5, K)
        outside = np.random.rand(5, K)
        post = compute_posteriors(inside, outside)
        assert np.all(post >= 0)

    def test_delta_inside_preserves_peak(self):
        """If inside is a delta function at grid point k, posterior peaks at k."""
        K = 10
        inside = np.zeros((1, K))
        inside[0, 5] = 1.0
        outside = np.ones((1, K))

        post = compute_posteriors(inside, outside)
        assert np.argmax(post[0]) == 5
        assert post[0, 5] == pytest.approx(1.0)

    def test_delta_outside_preserves_peak(self):
        """If outside is a delta function, posterior should peak there."""
        K = 10
        inside = np.ones((1, K))
        outside = np.zeros((1, K))
        outside[0, 3] = 1.0

        post = compute_posteriors(inside, outside)
        assert np.argmax(post[0]) == 3
        assert post[0, 3] == pytest.approx(1.0)

    def test_uniform_inside_outside(self):
        """Uniform inside and outside should give uniform posterior."""
        K = 10
        inside = np.ones((1, K))
        outside = np.ones((1, K))
        post = compute_posteriors(inside, outside)
        expected = np.ones(K) / K
        np.testing.assert_allclose(post[0], expected)

    def test_zero_row_handled_gracefully(self):
        """A node with all-zero inside*outside should not cause division errors."""
        K = 5
        inside = np.zeros((2, K))
        inside[1, :] = 1.0
        outside = np.ones((2, K))

        post = compute_posteriors(inside, outside)
        assert np.all(post[0, :] == 0.0)
        assert not np.any(np.isnan(post))

    def test_product_of_narrow_distributions(self):
        """Product of two peaked distributions should be even more peaked."""
        K = 20
        inside = np.zeros((1, K))
        inside[0, 7:10] = [0.2, 0.6, 0.2]

        outside = np.zeros((1, K))
        outside[0, 8:11] = [0.2, 0.6, 0.2]

        post = compute_posteriors(inside, outside)
        assert post[0, 8] > 0
        assert post[0, 9] > 0
        assert post[0, 8] + post[0, 9] > 0.9

    def test_multiple_nodes_independent(self):
        """Posteriors for different nodes should be computed independently."""
        K = 5
        inside = np.array([[1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1]], dtype=float)
        outside = np.ones((2, K))

        post = compute_posteriors(inside, outside)
        assert np.argmax(post[0]) == 0
        assert np.argmax(post[1]) == 4


# =========================================================================
# Tests for posterior_mean
# =========================================================================

class TestPosteriorMean:
    """Tests for the posterior_mean function."""

    def test_delta_posterior(self):
        """Delta posterior at grid[k] should give mean = grid[k]."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        post = np.zeros((1, 5))
        post[0, 3] = 1.0

        means = posterior_mean(post, grid)
        assert means[0] == pytest.approx(3.0)

    def test_uniform_posterior(self):
        """Uniform posterior should give mean = average of grid points."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        post = np.ones((1, 5)) / 5.0

        means = posterior_mean(post, grid)
        assert means[0] == pytest.approx(2.0)

    def test_two_point_mean(self):
        """A posterior split between two grid points should give weighted average."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        post = np.zeros((1, 5))
        post[0, 1] = 0.3
        post[0, 3] = 0.7

        means = posterior_mean(post, grid)
        expected = 0.3 * 1.0 + 0.7 * 3.0
        assert means[0] == pytest.approx(expected)

    def test_multiple_nodes(self):
        """Test with multiple nodes simultaneously."""
        grid = np.array([0.0, 1.0, 2.0])
        post = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1 / 3, 1 / 3, 1 / 3],
        ])

        means = posterior_mean(post, grid)
        np.testing.assert_allclose(means, [0.0, 1.0, 2.0, 1.0])

    def test_mean_within_grid_range(self):
        """Posterior mean must lie within [grid[0], grid[-1]]."""
        np.random.seed(42)
        grid = np.linspace(0, 10, 20)
        post = np.random.rand(10, 20)
        post /= post.sum(axis=1, keepdims=True)

        means = posterior_mean(post, grid)
        assert np.all(means >= grid[0])
        assert np.all(means <= grid[-1])

    def test_zero_posterior_gives_zero_mean(self):
        """An all-zero posterior row (degenerate) should yield mean = 0."""
        grid = np.array([0.0, 1.0, 2.0])
        post = np.zeros((1, 3))

        means = posterior_mean(post, grid)
        assert means[0] == pytest.approx(0.0)

    def test_log_grid_posterior_mean(self):
        """Test posterior mean with a logarithmic grid."""
        grid = make_time_grid(n=50, num_points=15, grid_type="logarithmic")
        post = np.zeros((1, 15))
        post[0, 7] = 1.0

        means = posterior_mean(post, grid)
        assert means[0] == pytest.approx(grid[7])


# =========================================================================
# Tests for inside_pass_logspace
# =========================================================================

class TestInsidePassLogspace:
    """Tests for the inside_pass_logspace function."""

    def test_consistency_with_linear_space(self):
        """Log-space computation should match linear-space computation."""
        np.random.seed(42)
        K = 8
        grid = np.linspace(0, 5, K)

        inside_linear = np.random.rand(K) + 0.01
        L_linear = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1):
                L_linear[i, j] = np.random.rand() + 0.01

        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * inside_linear[j]

        inside_log = np.log(inside_linear)
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            for j in range(i + 1):
                L_log[i, j] = np.log(L_linear[i, j])

        msg_log = inside_pass_logspace(inside_log, L_log, K)
        np.testing.assert_allclose(np.exp(msg_log), msg_linear, rtol=1e-10)

    def test_output_shape(self):
        """Output message should have shape (K,)."""
        K = 5
        inside_log = np.zeros(K)
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            L_log[i, 0] = 0.0

        msg = inside_pass_logspace(inside_log, L_log, K)
        assert msg.shape == (K,)

    def test_first_entry_uses_only_first_element(self):
        """msg[0] should be logsumexp of only L_log[0,0] + inside_log[0]."""
        K = 5
        inside_log = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        L_log = np.full((K, K), -np.inf)
        L_log[0, 0] = 0.5

        msg = inside_pass_logspace(inside_log, L_log, K)
        expected = L_log[0, 0] + inside_log[0]
        assert msg[0] == pytest.approx(expected)

    def test_all_neg_inf_gives_neg_inf(self):
        """If the likelihood matrix is all -inf, messages should be -inf."""
        K = 4
        inside_log = np.zeros(K)
        L_log = np.full((K, K), -np.inf)

        msg = inside_pass_logspace(inside_log, L_log, K)
        assert np.all(np.isneginf(msg))

    def test_handles_very_small_values(self):
        """Log-space should handle very small probabilities without underflow."""
        K = 5
        inside_log = np.array([-1000.0, -999.0, -998.0, -997.0, -996.0])
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            for j in range(i + 1):
                L_log[i, j] = -500.0

        msg = inside_pass_logspace(inside_log, L_log, K)
        assert not np.any(np.isnan(msg))
        assert not np.any(np.isposinf(msg))
        assert np.all(msg < 0)

    def test_delta_child_inside(self):
        """If child's inside is a delta at index 0, message should be L_log[i, 0]."""
        K = 5
        inside_log = np.full(K, -np.inf)
        inside_log[0] = 0.0

        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            L_log[i, 0] = -float(i)

        msg = inside_pass_logspace(inside_log, L_log, K)
        for i in range(K):
            assert msg[i] == pytest.approx(L_log[i, 0] + inside_log[0])

    def test_with_real_likelihood_matrix(self):
        """Test using a real Poisson likelihood matrix in log space."""
        K = 8
        grid = np.linspace(0, 4, K)
        m_e = 2
        lambda_e = 1.0

        L_linear = edge_likelihood_matrix(m_e, lambda_e, grid)

        L_log = np.full_like(L_linear, -np.inf)
        mask = L_linear > 0
        L_log[mask] = np.log(L_linear[mask])

        inside_log = np.zeros(K)
        msg_log = inside_pass_logspace(inside_log, L_log, K)

        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * 1.0

        for i in range(K):
            if msg_linear[i] > 0:
                assert np.exp(msg_log[i]) == pytest.approx(msg_linear[i], rel=1e-8)


# =========================================================================
# Integration tests (inside-outside pipeline)
# =========================================================================

class TestIntegration:
    """Integration tests combining multiple inside-outside functions."""

    def test_full_pipeline_mock(self):
        """Simulate a simplified inside-outside pipeline on mock data."""
        K = 15
        n_nodes = 10
        grid = make_time_grid(n=50, num_points=K, grid_type="logarithmic")

        np.random.seed(42)
        inside = np.random.rand(n_nodes, K) + 0.01
        outside = np.random.rand(n_nodes, K) + 0.01

        post = compute_posteriors(inside, outside)
        means = posterior_mean(post, grid)

        for u in range(n_nodes):
            assert post[u, :].sum() == pytest.approx(1.0)

        assert np.all(means >= grid[0])
        assert np.all(means <= grid[-1])

    def test_leaf_node_at_time_zero(self):
        """A leaf node (delta at grid[0]=0) should have posterior mean 0."""
        K = 10
        grid = make_time_grid(n=50, num_points=K)

        inside = np.zeros((1, K))
        inside[0, 0] = 1.0

        outside = np.ones((1, K))

        post = compute_posteriors(inside, outside)
        means = posterior_mean(post, grid)

        assert means[0] == pytest.approx(0.0)

    def test_likelihood_concentrates_posterior(self):
        """Edge likelihood with many mutations should concentrate the
        posterior around the expected time delta_t = m_e / lambda_e."""
        K = 30
        grid = np.linspace(0, 10, K)

        m_e = 10
        lambda_e = 2.0
        expected_delta = m_e / lambda_e

        L = edge_likelihood_matrix(m_e, lambda_e, grid)

        child_inside = np.zeros(K)
        child_inside[0] = 1.0

        msg = np.zeros(K)
        for i in range(K):
            msg[i] = L[i, 0] * child_inside[0]

        inside = msg.reshape(1, -1)
        outside = np.ones((1, K))
        post = compute_posteriors(inside, outside)
        mean_t = posterior_mean(post, grid)

        assert mean_t[0] == pytest.approx(expected_delta, abs=1.0)

    def test_logspace_matches_linear_for_real_edge(self):
        """End-to-end check that logspace message matches linear message."""
        K = 12
        grid = make_time_grid(n=50, num_points=K, grid_type="linear")

        m_e = 3
        lambda_e = 0.5
        L_linear = edge_likelihood_matrix(m_e, lambda_e, grid)

        child_inside = np.zeros(K)
        child_inside[1:4] = [0.2, 0.6, 0.2]

        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * child_inside[j]

        child_inside_log = np.full(K, -np.inf)
        nonzero = child_inside > 0
        child_inside_log[nonzero] = np.log(child_inside[nonzero])

        L_log = np.full((K, K), -np.inf)
        pos = L_linear > 0
        L_log[pos] = np.log(L_linear[pos])

        msg_log = inside_pass_logspace(child_inside_log, L_log, K)

        for i in range(K):
            if msg_linear[i] > 1e-300:
                assert np.exp(msg_log[i]) == pytest.approx(msg_linear[i], rel=1e-8)

    def test_grid_types_give_same_posterior_structure(self):
        """Linear and logarithmic grids should both produce valid posteriors."""
        for grid_type in ["linear", "logarithmic"]:
            K = 15
            grid = make_time_grid(n=50, num_points=K, grid_type=grid_type)

            np.random.seed(42)
            inside = np.random.rand(3, K) + 0.01
            outside = np.random.rand(3, K) + 0.01

            post = compute_posteriors(inside, outside)
            means = posterior_mean(post, grid)

            for u in range(3):
                assert post[u, :].sum() == pytest.approx(1.0)
            assert np.all(means >= 0)
            assert np.all(means <= grid[-1])


# =========================================================================
# Tests for Coalescent Prior (Chapter 1)
# =========================================================================

class TestConditionalCoalescentMoments:
    """Tests for the conditional_coalescent_moments function."""

    def test_root_mean(self):
        """Root (k=n) mean should be sum of 2/(j*(j-1)) for j=2..n."""
        n = 10
        moments = conditional_coalescent_moments(n)
        expected_root_mean = sum(2.0 / (j * (j - 1)) for j in range(2, n + 1))
        assert moments[n][0] == pytest.approx(expected_root_mean)

    def test_root_variance(self):
        """Root (k=n) variance should be sum of 4/(j*(j-1))^2 for j=2..n."""
        n = 10
        moments = conditional_coalescent_moments(n)
        expected_root_var = sum(4.0 / (j * (j - 1))**2 for j in range(2, n + 1))
        assert moments[n][1] == pytest.approx(expected_root_var)

    def test_all_means_positive(self):
        """All means should be positive."""
        n = 15
        moments = conditional_coalescent_moments(n)
        for k in range(2, n + 1):
            assert moments[k][0] > 0

    def test_all_variances_positive(self):
        """All variances should be positive."""
        n = 15
        moments = conditional_coalescent_moments(n)
        for k in range(2, n + 1):
            assert moments[k][1] > 0

    def test_keys_range(self):
        """Moments dict should have keys 2, 3, ..., n."""
        n = 8
        moments = conditional_coalescent_moments(n)
        assert set(moments.keys()) == set(range(2, n + 1))


class TestGammaParamsFromMoments:
    """Tests for the gamma_params_from_moments function."""

    def test_exponential_case(self):
        """When mean=1 and var=1 (exponential), alpha=1, beta=1."""
        alpha, beta = gamma_params_from_moments(1.0, 1.0)
        assert alpha == pytest.approx(1.0)
        assert beta == pytest.approx(1.0)

    def test_moments_roundtrip(self):
        """Gamma(alpha, beta) should have mean=alpha/beta, var=alpha/beta^2."""
        mean, var = 3.0, 2.0
        alpha, beta = gamma_params_from_moments(mean, var)
        assert alpha / beta == pytest.approx(mean)
        assert alpha / beta**2 == pytest.approx(var)

    def test_large_mean_small_var(self):
        """High alpha (peaked distribution): large mean, small variance."""
        mean, var = 100.0, 1.0
        alpha, beta = gamma_params_from_moments(mean, var)
        assert alpha > 1000  # very peaked
        assert beta > 0


class TestBuildPriorGrid:
    """Tests for the build_prior_grid function."""

    def test_shape(self):
        """Grid should be (n+1, 4)."""
        n = 10
        grid = build_prior_grid(n)
        assert grid.shape == (n + 1, 4)

    def test_rows_0_and_1_are_zero(self):
        """Rows 0 and 1 should be all zeros (unused)."""
        n = 10
        grid = build_prior_grid(n)
        np.testing.assert_array_equal(grid[0], [0, 0, 0, 0])
        np.testing.assert_array_equal(grid[1], [0, 0, 0, 0])

    def test_all_alphas_positive_for_valid_rows(self):
        """Alpha (column 0) should be positive for k=2..n."""
        n = 10
        grid = build_prior_grid(n)
        for k in range(2, n + 1):
            assert grid[k, 0] > 0

    def test_all_betas_positive_for_valid_rows(self):
        """Beta (column 1) should be positive for k=2..n."""
        n = 10
        grid = build_prior_grid(n)
        for k in range(2, n + 1):
            assert grid[k, 1] > 0


class TestTransitionProb:
    """Tests for the _transition_prob function."""

    def test_same_a(self):
        """Transition a' -> a where a == a'."""
        assert _transition_prob(3, 3) == pytest.approx(2.0 / 4.0)

    def test_decrease_by_one(self):
        """Transition a' -> a' - 1."""
        assert _transition_prob(3, 2) == pytest.approx(2.0 / 4.0)

    def test_invalid_a_greater_than_a_prime(self):
        """a > a' should give 0."""
        assert _transition_prob(3, 5) == 0.0

    def test_invalid_a_less_than_2(self):
        """a < 2 should give 0."""
        assert _transition_prob(3, 1) == 0.0

    def test_decrease_by_two_or_more(self):
        """Decrease by more than 1 should give 0."""
        assert _transition_prob(5, 3) == 0.0


# =========================================================================
# Tests for Mutation Likelihood (Chapter 2)
# =========================================================================

class TestEdgeLikelihood:
    """Tests for the edge_likelihood function."""

    def test_zero_branch_length(self):
        """Parent younger than child should give 0 likelihood."""
        assert edge_likelihood(1, 0.5, 1.0, 2.0) == 0.0

    def test_equal_times(self):
        """Equal parent and child times should give 0 likelihood."""
        assert edge_likelihood(1, 0.5, 1.0, 1.0) == 0.0

    def test_zero_mutations_positive_branch(self):
        """P(0 | lambda*dt) = exp(-lambda*dt)."""
        t_p, t_c = 500, 0
        lambda_e = 1e-4
        expected = np.exp(-lambda_e * (t_p - t_c))
        assert edge_likelihood(0, lambda_e, t_p, t_c) == pytest.approx(expected)

    def test_matches_scipy_poisson(self):
        """Should match scipy.stats.poisson.pmf."""
        m_e = 3
        lambda_e = 0.01
        t_p, t_c = 500, 100
        dt = t_p - t_c
        expected_val = poisson.pmf(m_e, lambda_e * dt)
        assert edge_likelihood(m_e, lambda_e, t_p, t_c) == pytest.approx(expected_val)


class TestGammaPoissonUpdate:
    """Tests for the gamma_poisson_update function."""

    def test_conjugate_update(self):
        """Posterior should be Gamma(alpha+m, beta+lambda)."""
        alpha_post, beta_post = gamma_poisson_update(2.0, 3.0, 5, 0.01)
        assert alpha_post == pytest.approx(7.0)
        assert beta_post == pytest.approx(3.01)

    def test_no_observations(self):
        """With 0 mutations and 0 rate, posterior should equal prior."""
        alpha_post, beta_post = gamma_poisson_update(2.0, 3.0, 0, 0.0)
        assert alpha_post == pytest.approx(2.0)
        assert beta_post == pytest.approx(3.0)

    def test_posterior_mean_shifts_toward_data(self):
        """Posterior mean should shift toward m/lambda when data is strong."""
        alpha_prior, beta_prior = 2.0, 3.0
        m_e, lambda_e = 100, 1.0
        alpha_post, beta_post = gamma_poisson_update(alpha_prior, beta_prior,
                                                      m_e, lambda_e)
        prior_mean = alpha_prior / beta_prior
        post_mean = alpha_post / beta_post
        data_mle = m_e / lambda_e
        # Posterior mean should be between prior mean and MLE, closer to MLE
        assert post_mean > prior_mean
        assert abs(post_mean - data_mle) < abs(prior_mean - data_mle)


# =========================================================================
# Tests for GammaDistribution (Chapter 4)
# =========================================================================

class TestGammaDistribution:
    """Tests for the GammaDistribution class."""

    def test_mean(self):
        """Mean should be alpha/beta."""
        g = GammaDistribution(3.0, 2.0)
        assert g.mean == pytest.approx(1.5)

    def test_variance(self):
        """Variance should be alpha/beta^2."""
        g = GammaDistribution(3.0, 2.0)
        assert g.variance == pytest.approx(0.75)

    def test_natural_parameters(self):
        """eta1 = alpha - 1, eta2 = -beta."""
        g = GammaDistribution(3.0, 2.0)
        assert g.eta1 == pytest.approx(2.0)
        assert g.eta2 == pytest.approx(-2.0)

    def test_multiply(self):
        """Product of two gammas: alpha_new = a1+a2-1, beta_new = b1+b2."""
        g1 = GammaDistribution(3.0, 2.0)
        g2 = GammaDistribution(2.0, 1.5)
        g_prod = g1.multiply(g2)
        assert g_prod.alpha == pytest.approx(4.0)
        assert g_prod.beta == pytest.approx(3.5)

    def test_divide(self):
        """Division is inverse of multiply."""
        g1 = GammaDistribution(3.0, 2.0)
        g2 = GammaDistribution(2.0, 1.5)
        g_prod = g1.multiply(g2)
        g_back = g_prod.divide(g2)
        assert g_back.alpha == pytest.approx(g1.alpha)
        assert g_back.beta == pytest.approx(g1.beta)

    def test_multiply_numerically(self):
        """Product of two gamma PDFs should be proportional to Gamma(a1+a2-1, b1+b2)."""
        a1, b1 = 3.0, 2.0
        a2, b2 = 2.0, 1.5

        x = np.linspace(0.01, 5.0, 1000)
        f1 = gamma_dist.pdf(x, a=a1, scale=1 / b1)
        f2 = gamma_dist.pdf(x, a=a2, scale=1 / b2)
        product = f1 * f2

        a_new, b_new = a1 + a2 - 1, b1 + b2
        f_new = gamma_dist.pdf(x, a=a_new, scale=1 / b_new)

        ratio = product / f_new
        ratio = ratio[f_new > 1e-10]
        # Ratio should be approximately constant
        assert ratio.max() / ratio.min() == pytest.approx(1.0, abs=1e-4)

    def test_from_moments(self):
        """from_moments should produce correct alpha and beta."""
        g = GammaDistribution.from_moments(2.0, 0.5)
        assert g.mean == pytest.approx(2.0)
        assert g.variance == pytest.approx(0.5)

    def test_from_moments_roundtrip(self):
        """from_moments(mean, var) -> mean/var should roundtrip."""
        g = GammaDistribution(5.0, 3.0)
        g2 = GammaDistribution.from_moments(g.mean, g.variance)
        assert g2.alpha == pytest.approx(g.alpha)
        assert g2.beta == pytest.approx(g.beta)

    def test_exponential_special_case(self):
        """Gamma(1, beta) is Exponential(beta)."""
        g = GammaDistribution(1.0, 5.0)
        assert g.mean == pytest.approx(0.2)
        assert g.variance == pytest.approx(0.04)
        assert g.eta1 == pytest.approx(0.0)


# =========================================================================
# Tests for numerical_hessian (Chapter 4)
# =========================================================================

class TestNumericalHessian:
    """Tests for the numerical_hessian function."""

    def test_quadratic(self):
        """Hessian of f(x,y) = x^2 + y^2 should be [[2,0],[0,2]]."""
        f = lambda x: x[0]**2 + x[1]**2
        H = numerical_hessian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(H, [[2.0, 0.0], [0.0, 2.0]], atol=1e-4)

    def test_cross_terms(self):
        """Hessian of f(x,y) = x*y should be [[0,1],[1,0]]."""
        f = lambda x: x[0] * x[1]
        H = numerical_hessian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(H, [[0.0, 1.0], [1.0, 0.0]], atol=1e-4)

    def test_symmetric(self):
        """Hessian should be symmetric."""
        f = lambda x: x[0]**2 * x[1] + x[0] * x[1]**3
        H = numerical_hessian(f, np.array([2.0, 3.0]))
        np.testing.assert_allclose(H, H.T, atol=1e-4)


# =========================================================================
# Tests for compute_scaling_factors (Chapter 5)
# =========================================================================

class TestComputeScalingFactors:
    """Tests for the compute_scaling_factors function."""

    def test_perfect_match(self):
        """When observed == expected, all scales should be 1.0."""
        obs = np.array([10.0, 20.0, 15.0])
        exp = np.array([10.0, 20.0, 15.0])
        scales = compute_scaling_factors(obs, exp)
        np.testing.assert_allclose(scales, [1.0, 1.0, 1.0])

    def test_double_observed(self):
        """When observed = 2 * expected, scales should be 2.0."""
        obs = np.array([20.0, 40.0])
        exp = np.array([10.0, 20.0])
        scales = compute_scaling_factors(obs, exp)
        np.testing.assert_allclose(scales, [2.0, 2.0])

    def test_min_count_filter(self):
        """Windows below min_count should have scale = 1.0."""
        obs = np.array([0.5, 10.0])
        exp = np.array([5.0, 10.0])
        scales = compute_scaling_factors(obs, exp, min_count=1.0)
        assert scales[0] == 1.0  # below min_count
        assert scales[1] == pytest.approx(1.0)

    def test_zero_expected(self):
        """Windows with zero expected should have scale = 1.0."""
        obs = np.array([5.0, 10.0])
        exp = np.array([0.0, 10.0])
        scales = compute_scaling_factors(obs, exp)
        assert scales[0] == 1.0
        assert scales[1] == pytest.approx(1.0)


# =========================================================================
# Tests for apply_rescaling (Chapter 5)
# =========================================================================

class TestApplyRescaling:
    """Tests for the apply_rescaling function."""

    def test_identity_scaling(self):
        """With all scales=1.0, times should be unchanged."""
        times = np.array([0.0, 1.0, 2.5, 4.0])
        breakpoints = np.array([0.0, 2.0, 5.0])
        scales = np.array([1.0, 1.0])
        new_times = apply_rescaling(times, breakpoints, scales, set())
        np.testing.assert_allclose(new_times, times)

    def test_fixed_nodes_unchanged(self):
        """Fixed nodes should keep their original times."""
        times = np.array([0.0, 0.0, 3.0])
        breakpoints = np.array([0.0, 2.0, 5.0])
        scales = np.array([2.0, 2.0])
        new_times = apply_rescaling(times, breakpoints, scales, {0, 1})
        assert new_times[0] == 0.0
        assert new_times[1] == 0.0

    def test_double_scaling(self):
        """With all scales=2.0, times should double."""
        times = np.array([0.0, 1.0, 2.0])
        breakpoints = np.array([0.0, 5.0])
        scales = np.array([2.0])
        new_times = apply_rescaling(times, breakpoints, scales, set())
        np.testing.assert_allclose(new_times, [0.0, 2.0, 4.0])

    def test_piecewise_rescaling(self):
        """Verify piecewise rescaling across window boundaries."""
        times = np.array([0.0, 0.5, 1.5, 3.0])
        breakpoints = np.array([0.0, 1.0, 2.0, 4.0])
        scales = np.array([1.0, 2.0, 0.5])
        new_times = apply_rescaling(times, breakpoints, scales, set())

        # t=0.0: in window 0, scale=1.0 -> 0.0
        assert new_times[0] == pytest.approx(0.0)
        # t=0.5: in window 0, scale=1.0 -> 0.5
        assert new_times[1] == pytest.approx(0.5)
        # t=1.5: in window 1 (after window 0 contributes 1.0*1.0=1.0)
        # -> 1.0 + 2.0 * 0.5 = 2.0
        assert new_times[2] == pytest.approx(2.0)
        # t=3.0: windows 0 (1.0*1.0=1.0) + 1 (2.0*1.0=2.0) + 2 (0.5*1.0=0.5)
        # -> 1.0 + 2.0 + 0.5 = 3.5
        assert new_times[3] == pytest.approx(3.5)

    def test_monotonicity_preserved(self):
        """Rescaling with positive scales should preserve time ordering."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        breakpoints = np.array([0.0, 1.5, 3.0, 5.0])
        scales = np.array([0.5, 2.0, 1.0])
        new_times = apply_rescaling(times, breakpoints, scales, set())
        assert np.all(np.diff(new_times) >= 0)
