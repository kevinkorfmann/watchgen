"""
Tests for code extracted from docs/timepieces/tsdate/inside_outside.rst

Covers the self-contained functions:
- make_time_grid (time discretization for inside-outside algorithm)
- edge_likelihood_matrix (Poisson likelihood on a time grid)
- compute_posteriors (combining inside and outside values)
- posterior_mean (computing E[t | D] from a posterior distribution)
- inside_pass_logspace (single inside message in log space)

Functions that depend on tskit (inside_pass, outside_pass,
inside_outside_date, build_discrete_prior, is_root) are not tested.
"""

import numpy as np
import pytest
from scipy.stats import poisson
from scipy.special import logsumexp


# =========================================================================
# Code extracted from inside_outside.rst
# =========================================================================

def make_time_grid(n, Ne=1.0, num_points=20, grid_type="logarithmic"):
    """Create a time grid for the inside-outside algorithm.

    Parameters
    ----------
    n : int
        Number of samples (sets the expected TMRCA).
    Ne : float
        Effective population size.
    num_points : int
        Number of grid points.
    grid_type : str
        "linear" or "logarithmic".

    Returns
    -------
    grid : np.ndarray
        Array of timepoints, starting at 0.
    """
    # Expected TMRCA under standard coalescent: 2*Ne*(1 - 1/n)
    expected_tmrca = 2 * Ne * (1 - 1.0 / n)
    t_max = expected_tmrca * 4  # go well beyond expected TMRCA

    if grid_type == "linear":
        return np.linspace(0, t_max, num_points)
    else:
        # Log-spaced: more points near 0, fewer far out
        # Start from a small positive number to avoid log(0)
        t_min = t_max / (10 * num_points)
        return np.concatenate([[0], np.geomspace(t_min, t_max, num_points - 1)])


def edge_likelihood_matrix(m_e, lambda_e, grid):
    """Compute the likelihood matrix for an edge on the time grid.

    Parameters
    ----------
    m_e : int
        Mutation count on this edge.
    lambda_e : float
        Span-weighted mutation rate (mu * span_bp).
    grid : np.ndarray
        Time grid.

    Returns
    -------
    L : np.ndarray, shape (K, K)
        L[i, j] = P(m_e | parent_time=grid[i], child_time=grid[j])
        Lower triangular (i >= j).
    """
    K = len(grid)
    L = np.zeros((K, K))

    for i in range(K):
        for j in range(i + 1):  # j <= i (child younger than parent)
            delta_t = grid[i] - grid[j]
            if delta_t > 0:
                expected = lambda_e * delta_t       # Poisson mean
                L[i, j] = poisson.pmf(m_e, expected)  # evaluate PMF
            elif m_e == 0:
                # delta_t = 0, only possible if no mutations
                L[i, j] = 1.0

    return L


def compute_posteriors(inside, outside):
    """Combine inside and outside to get marginal posteriors.

    Parameters
    ----------
    inside : np.ndarray, shape (num_nodes, K)
    outside : np.ndarray, shape (num_nodes, K)

    Returns
    -------
    posterior : np.ndarray, shape (num_nodes, K)
        posterior[u, :] is the marginal posterior distribution over
        grid points for node u.
    """
    posterior = inside * outside  # element-wise product

    # Normalize each node's posterior to sum to 1
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    posterior /= row_sums

    return posterior


def posterior_mean(posterior, grid):
    """Compute posterior mean age for each node.

    Parameters
    ----------
    posterior : np.ndarray, shape (num_nodes, K)
    grid : np.ndarray, shape (K,)

    Returns
    -------
    means : np.ndarray, shape (num_nodes,)
        E[t_u | D] for each node.
    """
    return posterior @ grid  # weighted sum: sum_i posterior[u,i] * grid[i]


def inside_pass_logspace(inside_log, L_log, K):
    """Compute a single inside message in log space.

    Parameters
    ----------
    inside_log : np.ndarray, shape (K,)
        Log inside values for child node.
    L_log : np.ndarray, shape (K, K)
        Log likelihood matrix.

    Returns
    -------
    msg_log : np.ndarray, shape (K,)
        Log message from child to parent.
    """
    msg_log = np.full(K, -np.inf)    # start at log(0) = -inf
    for i in range(K):
        terms = L_log[i, :i+1] + inside_log[:i+1]  # log(L * inside) = log(L) + log(inside)
        msg_log[i] = logsumexp(terms)               # log-sum-exp for numerical stability
    return msg_log


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
        """Logarithmic grid should have more points near 0 than a linear grid.

        Specifically, the first few intervals should be smaller for the
        logarithmic grid compared to the linear grid.
        """
        grid_lin = make_time_grid(n=50, num_points=20, grid_type="linear")
        grid_log = make_time_grid(n=50, num_points=20, grid_type="logarithmic")

        # The first interval of log grid should be smaller
        assert np.diff(grid_log)[0] < np.diff(grid_lin)[0]

    def test_large_n_grid_approaches_max_8(self):
        """For large n and Ne=1, t_max should approach 4 * 2 * 1 = 8."""
        grid = make_time_grid(n=10000, Ne=1.0, num_points=20, grid_type="linear")
        assert grid[-1] == pytest.approx(8.0, rel=0.01)

    def test_n_equals_2(self):
        """For n=2, expected TMRCA = 2*Ne*(1 - 1/2) = Ne."""
        Ne = 1.0
        grid = make_time_grid(n=2, Ne=Ne, num_points=10, grid_type="linear")
        expected_max = 4 * 2 * Ne * (1 - 0.5)  # = 4.0
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

        # Check L[3, 1]: delta_t = 3.0 - 1.0 = 2.0, expected = 0.5 * 2.0 = 1.0
        expected_val = poisson.pmf(2, 1.0)
        assert L[3, 1] == pytest.approx(expected_val)

        # Check L[2, 0]: delta_t = 2.0, expected = 0.5 * 2.0 = 1.0
        expected_val = poisson.pmf(2, 1.0)
        assert L[2, 0] == pytest.approx(expected_val)

    def test_likelihood_peaks_at_correct_delta_t(self):
        """The maximum likelihood should occur near delta_t = m_e / lambda_e.

        For Poisson(mu), the mode is at floor(mu) or ceil(mu).
        So for m_e mutations, we expect the peak at delta_t ~ m_e / lambda_e.
        """
        m_e = 5
        lambda_e = 1.0
        grid = np.linspace(0, 20, 100)
        L = edge_likelihood_matrix(m_e=m_e, lambda_e=lambda_e, grid=grid)

        # Find the maximum entry (in the lower triangle)
        max_idx = np.unravel_index(L.argmax(), L.shape)
        delta_t_at_max = grid[max_idx[0]] - grid[max_idx[1]]
        expected_delta = m_e / lambda_e  # = 5.0

        assert delta_t_at_max == pytest.approx(expected_delta, abs=0.5)

    def test_zero_mutations_monotone_decrease(self):
        """With m_e=0, P(0 | lambda * delta_t) = exp(-lambda*delta_t).

        For the first column (child at time 0), values should decrease
        with increasing parent time.
        """
        grid = np.linspace(0, 5, 20)
        L = edge_likelihood_matrix(m_e=0, lambda_e=1.0, grid=grid)
        # First column: L[i, 0] for i = 0, 1, ..., 19
        first_col = L[:, 0]
        # Should be monotonically decreasing (after position 0)
        for i in range(2, len(first_col)):
            assert first_col[i] <= first_col[i - 1]

    def test_high_mutation_rate_concentrates_likelihood(self):
        """Higher mutation rate lambda_e should concentrate the likelihood.

        The Poisson variance equals the mean, so higher lambda * delta_t
        means larger spread, but the peak is sharper relative to the range.
        """
        grid = np.linspace(0, 10, 50)
        L_low = edge_likelihood_matrix(m_e=3, lambda_e=0.1, grid=grid)
        L_high = edge_likelihood_matrix(m_e=3, lambda_e=10.0, grid=grid)

        # With high lambda, peak should be at a smaller delta_t
        max_low = np.unravel_index(L_low.argmax(), L_low.shape)
        max_high = np.unravel_index(L_high.argmax(), L_high.shape)
        delta_low = grid[max_low[0]] - grid[max_low[1]]
        delta_high = grid[max_high[0]] - grid[max_high[1]]

        # m_e/lambda_e: 3/0.1 = 30 vs 3/10 = 0.3
        assert delta_high < delta_low

    def test_small_grid(self):
        """Test with a minimal 2-point grid."""
        grid = np.array([0.0, 1.0])
        L = edge_likelihood_matrix(m_e=0, lambda_e=1.0, grid=grid)
        assert L.shape == (2, 2)
        assert L[0, 0] == 1.0  # m_e=0, delta_t=0
        assert L[1, 0] == pytest.approx(np.exp(-1.0))  # Poisson(0; 1) = e^{-1}
        assert L[1, 1] == 1.0  # m_e=0, delta_t=0


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
        posterior = compute_posteriors(inside, outside)

        for u in range(5):
            assert posterior[u, :].sum() == pytest.approx(1.0)

    def test_all_values_non_negative(self):
        """Posterior values should all be non-negative."""
        np.random.seed(42)
        K = 10
        inside = np.random.rand(5, K)
        outside = np.random.rand(5, K)
        posterior = compute_posteriors(inside, outside)
        assert np.all(posterior >= 0)

    def test_delta_inside_preserves_peak(self):
        """If inside is a delta function at grid point k, the posterior
        should peak at k (given uniform outside)."""
        K = 10
        inside = np.zeros((1, K))
        inside[0, 5] = 1.0
        outside = np.ones((1, K))

        posterior = compute_posteriors(inside, outside)
        assert np.argmax(posterior[0]) == 5
        assert posterior[0, 5] == pytest.approx(1.0)

    def test_delta_outside_preserves_peak(self):
        """If outside is a delta function, posterior should peak there."""
        K = 10
        inside = np.ones((1, K))
        outside = np.zeros((1, K))
        outside[0, 3] = 1.0

        posterior = compute_posteriors(inside, outside)
        assert np.argmax(posterior[0]) == 3
        assert posterior[0, 3] == pytest.approx(1.0)

    def test_uniform_inside_outside(self):
        """Uniform inside and outside should give uniform posterior."""
        K = 10
        inside = np.ones((1, K))
        outside = np.ones((1, K))
        posterior = compute_posteriors(inside, outside)
        expected = np.ones(K) / K
        np.testing.assert_allclose(posterior[0], expected)

    def test_zero_row_handled_gracefully(self):
        """A node with all-zero inside*outside should not cause division errors."""
        K = 5
        inside = np.zeros((2, K))
        inside[1, :] = 1.0
        outside = np.ones((2, K))

        posterior = compute_posteriors(inside, outside)
        # Row 0 is all zeros -- should remain all zeros (not NaN)
        assert np.all(posterior[0, :] == 0.0)
        assert not np.any(np.isnan(posterior))

    def test_product_of_narrow_distributions(self):
        """Product of two peaked distributions should be even more peaked."""
        K = 20
        # inside peaks at index 8
        inside = np.zeros((1, K))
        inside[0, 7:10] = [0.2, 0.6, 0.2]

        # outside peaks at index 9
        outside = np.zeros((1, K))
        outside[0, 8:11] = [0.2, 0.6, 0.2]

        posterior = compute_posteriors(inside, outside)
        # The product peaks at index 8 (overlap of both distributions)
        # inside[8] * outside[8] = 0.6 * 0.2 = 0.12
        # inside[9] * outside[9] = 0.2 * 0.6 = 0.12
        # These should be the dominant entries
        assert posterior[0, 8] > 0
        assert posterior[0, 9] > 0
        # Most mass should be at indices 8 and 9
        assert posterior[0, 8] + posterior[0, 9] > 0.9

    def test_multiple_nodes_independent(self):
        """Posteriors for different nodes should be computed independently."""
        K = 5
        inside = np.array([[1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1]], dtype=float)
        outside = np.ones((2, K))

        posterior = compute_posteriors(inside, outside)
        assert np.argmax(posterior[0]) == 0
        assert np.argmax(posterior[1]) == 4


# =========================================================================
# Tests for posterior_mean
# =========================================================================

class TestPosteriorMean:
    """Tests for the posterior_mean function."""

    def test_delta_posterior(self):
        """Delta posterior at grid[k] should give mean = grid[k]."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        posterior = np.zeros((1, 5))
        posterior[0, 3] = 1.0

        means = posterior_mean(posterior, grid)
        assert means[0] == pytest.approx(3.0)

    def test_uniform_posterior(self):
        """Uniform posterior should give mean = average of grid points."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        posterior = np.ones((1, 5)) / 5.0

        means = posterior_mean(posterior, grid)
        assert means[0] == pytest.approx(2.0)

    def test_two_point_mean(self):
        """A posterior split between two grid points should give
        the weighted average."""
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        posterior = np.zeros((1, 5))
        posterior[0, 1] = 0.3  # 30% at t=1
        posterior[0, 3] = 0.7  # 70% at t=3

        means = posterior_mean(posterior, grid)
        expected = 0.3 * 1.0 + 0.7 * 3.0
        assert means[0] == pytest.approx(expected)

    def test_multiple_nodes(self):
        """Test with multiple nodes simultaneously."""
        grid = np.array([0.0, 1.0, 2.0])
        posterior = np.array([
            [1.0, 0.0, 0.0],   # mean = 0
            [0.0, 1.0, 0.0],   # mean = 1
            [0.0, 0.0, 1.0],   # mean = 2
            [1/3, 1/3, 1/3],   # mean = 1
        ])

        means = posterior_mean(posterior, grid)
        np.testing.assert_allclose(means, [0.0, 1.0, 2.0, 1.0])

    def test_mean_within_grid_range(self):
        """Posterior mean must lie within [grid[0], grid[-1]]."""
        np.random.seed(42)
        grid = np.linspace(0, 10, 20)
        posterior = np.random.rand(10, 20)
        posterior /= posterior.sum(axis=1, keepdims=True)

        means = posterior_mean(posterior, grid)
        assert np.all(means >= grid[0])
        assert np.all(means <= grid[-1])

    def test_zero_posterior_gives_zero_mean(self):
        """An all-zero posterior row (degenerate) should yield mean = 0."""
        grid = np.array([0.0, 1.0, 2.0])
        posterior = np.zeros((1, 3))

        means = posterior_mean(posterior, grid)
        assert means[0] == pytest.approx(0.0)

    def test_log_grid_posterior_mean(self):
        """Test posterior mean with a logarithmic grid."""
        grid = make_time_grid(n=50, num_points=15, grid_type="logarithmic")
        posterior = np.zeros((1, 15))
        posterior[0, 7] = 1.0

        means = posterior_mean(posterior, grid)
        assert means[0] == pytest.approx(grid[7])


# =========================================================================
# Tests for inside_pass_logspace
# =========================================================================

class TestInsidePassLogspace:
    """Tests for the inside_pass_logspace function."""

    def test_consistency_with_linear_space(self):
        """Log-space computation should match linear-space computation.

        Compute the message in both linear and log space and verify
        they give the same result after exponentiation.
        """
        np.random.seed(42)
        K = 8
        grid = np.linspace(0, 5, K)

        # Create a random inside vector (positive) and likelihood matrix
        inside_linear = np.random.rand(K) + 0.01
        L_linear = np.zeros((K, K))
        for i in range(K):
            for j in range(i + 1):
                L_linear[i, j] = np.random.rand() + 0.01

        # Compute message in linear space
        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * inside_linear[j]

        # Compute message in log space
        inside_log = np.log(inside_linear)
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            for j in range(i + 1):
                L_log[i, j] = np.log(L_linear[i, j])

        msg_log = inside_pass_logspace(inside_log, L_log, K)

        # Compare
        np.testing.assert_allclose(np.exp(msg_log), msg_linear, rtol=1e-10)

    def test_output_shape(self):
        """Output message should have shape (K,)."""
        K = 5
        inside_log = np.zeros(K)
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            L_log[i, 0] = 0.0  # at least one nonzero per row

        msg = inside_pass_logspace(inside_log, L_log, K)
        assert msg.shape == (K,)

    def test_first_entry_uses_only_first_element(self):
        """msg[0] should be logsumexp of only L_log[0,0] + inside_log[0]."""
        K = 5
        inside_log = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        L_log = np.full((K, K), -np.inf)
        L_log[0, 0] = 0.5

        msg = inside_pass_logspace(inside_log, L_log, K)
        expected = L_log[0, 0] + inside_log[0]  # = 0.5 + 1.0 = 1.5
        assert msg[0] == pytest.approx(expected)

    def test_all_neg_inf_gives_neg_inf(self):
        """If the likelihood matrix is all -inf, messages should be -inf."""
        K = 4
        inside_log = np.zeros(K)
        L_log = np.full((K, K), -np.inf)

        msg = inside_pass_logspace(inside_log, L_log, K)
        assert np.all(np.isneginf(msg))

    def test_handles_very_small_values(self):
        """Log-space should handle very small probabilities that would
        underflow in linear space."""
        K = 5
        # Values that would underflow as doubles in linear space
        inside_log = np.array([-1000.0, -999.0, -998.0, -997.0, -996.0])
        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            for j in range(i + 1):
                L_log[i, j] = -500.0

        msg = inside_pass_logspace(inside_log, L_log, K)

        # Should not produce NaN or +inf
        assert not np.any(np.isnan(msg))
        assert not np.any(np.isposinf(msg))
        # Values should be finite and very negative
        assert np.all(msg < 0)

    def test_delta_child_inside(self):
        """If child's inside is a delta at index 0, message should
        just be L_log[i, 0] for each i."""
        K = 5
        inside_log = np.full(K, -np.inf)
        inside_log[0] = 0.0  # delta at index 0

        L_log = np.full((K, K), -np.inf)
        for i in range(K):
            L_log[i, 0] = -float(i)  # L_log[i, 0] = -i

        msg = inside_pass_logspace(inside_log, L_log, K)
        for i in range(K):
            # msg[i] = logsumexp(L_log[i, :i+1] + inside_log[:i+1])
            # Only j=0 contributes since inside_log[j] = -inf for j > 0
            assert msg[i] == pytest.approx(L_log[i, 0] + inside_log[0])

    def test_with_real_likelihood_matrix(self):
        """Test using a real Poisson likelihood matrix in log space."""
        K = 8
        grid = np.linspace(0, 4, K)
        m_e = 2
        lambda_e = 1.0

        L_linear = edge_likelihood_matrix(m_e, lambda_e, grid)

        # Convert to log space (handle zeros)
        L_log = np.full_like(L_linear, -np.inf)
        mask = L_linear > 0
        L_log[mask] = np.log(L_linear[mask])

        # Uniform inside (child) in log space
        inside_log = np.zeros(K)  # log(1) = 0

        msg_log = inside_pass_logspace(inside_log, L_log, K)

        # Compute linear version for comparison
        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * 1.0  # inside = 1

        # Compare (only where msg_linear > 0)
        for i in range(K):
            if msg_linear[i] > 0:
                assert np.exp(msg_log[i]) == pytest.approx(msg_linear[i], rel=1e-8)


# =========================================================================
# Integration tests
# =========================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline_mock(self):
        """Simulate a simplified inside-outside pipeline on mock data.

        Create mock inside/outside arrays, compute posteriors and
        posterior means, and verify basic consistency.
        """
        K = 15
        n_nodes = 10
        grid = make_time_grid(n=50, num_points=K, grid_type="logarithmic")

        np.random.seed(42)
        inside = np.random.rand(n_nodes, K) + 0.01
        outside = np.random.rand(n_nodes, K) + 0.01

        posterior = compute_posteriors(inside, outside)
        means = posterior_mean(posterior, grid)

        # All posteriors should sum to 1
        for u in range(n_nodes):
            assert posterior[u, :].sum() == pytest.approx(1.0)

        # All means should be within grid range
        assert np.all(means >= grid[0])
        assert np.all(means <= grid[-1])

    def test_leaf_node_at_time_zero(self):
        """A leaf node (delta at grid[0]=0) should have posterior mean 0.

        This mimics how tsdate handles sample nodes.
        """
        K = 10
        grid = make_time_grid(n=50, num_points=K)

        inside = np.zeros((1, K))
        inside[0, 0] = 1.0  # delta at time 0

        outside = np.ones((1, K))

        posterior = compute_posteriors(inside, outside)
        means = posterior_mean(posterior, grid)

        assert means[0] == pytest.approx(0.0)

    def test_likelihood_concentrates_posterior(self):
        """Edge likelihood with many mutations should concentrate the
        posterior around the expected time delta_t = m_e / lambda_e."""
        K = 30
        grid = np.linspace(0, 10, K)

        m_e = 10
        lambda_e = 2.0
        expected_delta = m_e / lambda_e  # = 5.0

        L = edge_likelihood_matrix(m_e, lambda_e, grid)

        # Simulate: child is at time 0 (delta at grid[0])
        # The "inside" for the child is delta at index 0
        child_inside = np.zeros(K)
        child_inside[0] = 1.0

        # Message from child to parent: msg[i] = sum_j L[i,j] * child_inside[j]
        # Only j=0 contributes
        msg = np.zeros(K)
        for i in range(K):
            msg[i] = L[i, 0] * child_inside[0]

        # This msg acts as the inside for the parent
        # With a uniform outside, the posterior is proportional to msg
        inside = msg.reshape(1, -1)
        outside = np.ones((1, K))
        posterior = compute_posteriors(inside, outside)
        mean_t = posterior_mean(posterior, grid)

        # The posterior mean should be close to expected_delta
        assert mean_t[0] == pytest.approx(expected_delta, abs=1.0)

    def test_logspace_matches_linear_for_real_edge(self):
        """End-to-end check that logspace message matches linear message
        for a realistic edge."""
        K = 12
        grid = make_time_grid(n=50, num_points=K, grid_type="linear")

        m_e = 3
        lambda_e = 0.5
        L_linear = edge_likelihood_matrix(m_e, lambda_e, grid)

        # Child inside: peaked around grid index 2
        child_inside = np.zeros(K)
        child_inside[1:4] = [0.2, 0.6, 0.2]

        # Linear message
        msg_linear = np.zeros(K)
        for i in range(K):
            for j in range(i + 1):
                msg_linear[i] += L_linear[i, j] * child_inside[j]

        # Log-space message
        child_inside_log = np.full(K, -np.inf)
        nonzero = child_inside > 0
        child_inside_log[nonzero] = np.log(child_inside[nonzero])

        L_log = np.full((K, K), -np.inf)
        pos = L_linear > 0
        L_log[pos] = np.log(L_linear[pos])

        msg_log = inside_pass_logspace(child_inside_log, L_log, K)

        # Compare where linear message is positive
        for i in range(K):
            if msg_linear[i] > 1e-300:
                assert np.exp(msg_log[i]) == pytest.approx(msg_linear[i], rel=1e-8)

    def test_grid_types_give_same_posterior_structure(self):
        """Linear and logarithmic grids should both produce valid
        posteriors (though with different resolution)."""
        for grid_type in ["linear", "logarithmic"]:
            K = 15
            grid = make_time_grid(n=50, num_points=K, grid_type=grid_type)

            np.random.seed(42)
            inside = np.random.rand(3, K) + 0.01
            outside = np.random.rand(3, K) + 0.01

            posterior = compute_posteriors(inside, outside)
            means = posterior_mean(posterior, grid)

            for u in range(3):
                assert posterior[u, :].sum() == pytest.approx(1.0)
            assert np.all(means >= 0)
            assert np.all(means <= grid[-1])
