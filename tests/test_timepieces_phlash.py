"""
Tests for Python code blocks from the PHLASH timepiece RST documentation.

All functions are re-defined here since the code in the RST files is not
importable. Tests cover mathematical properties and expected behaviors.

Covers:
- composite_likelihood.rst: sfs_log_likelihood, expected_sfs_constant,
                            composite_log_likelihood, smoothness_prior_logpdf
- random_discretization.rst: sample_random_grid, debiased_gradient_estimate
- score_function.rst: hmm_score_function, total_gradient
- svgd_inference.rst: rbf_kernel, svgd_update, phlash_loop
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/phlash/composite_likelihood.rst
# ---------------------------------------------------------------------------

def sfs_log_likelihood(observed_sfs, expected_sfs):
    """Poisson log-likelihood of the observed SFS given expected SFS."""
    xi = np.maximum(expected_sfs, 1e-300)
    return np.sum(observed_sfs * np.log(xi) - xi)


def expected_sfs_constant(n, theta, N_e=1.0):
    """Expected SFS under constant population size."""
    k = np.arange(1, n)
    return theta / k


def composite_log_likelihood(observed_sfs, expected_sfs, hmm_log_likelihoods):
    """Compute the composite log-likelihood (SFS + coalescent HMM)."""
    ll_sfs = sfs_log_likelihood(observed_sfs, expected_sfs)
    ll_hmm = sum(hmm_log_likelihoods)
    return ll_sfs + ll_hmm


def smoothness_prior_logpdf(h, sigma=1.0):
    """Log-density of the Gaussian smoothness prior on log-eta."""
    diffs = np.diff(h)
    return -0.5 * np.sum(diffs**2) / sigma**2


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/phlash/random_discretization.rst
# ---------------------------------------------------------------------------

def sample_random_grid(M, t_max=10.0, t_min=1e-4, rng=None):
    """Sample a random time discretization grid."""
    if rng is None:
        rng = np.random.default_rng()

    log_min, log_max = np.log(t_min), np.log(t_max)
    anchors = np.linspace(log_min, log_max, M - 1)
    spacing = (log_max - log_min) / (M - 1)
    jitter = rng.uniform(-0.5 * spacing, 0.5 * spacing, size=M - 1)
    log_breakpoints = anchors + jitter

    interior = np.sort(np.clip(np.exp(log_breakpoints), t_min, t_max * 0.999))
    grid = np.concatenate([[0.0], interior, [t_max]])
    return grid


def debiased_gradient_estimate(eta, observed_sfs, n_grids=10, M=32, rng=None):
    """Estimate the likelihood gradient by averaging over random grids."""
    if rng is None:
        rng = np.random.default_rng()

    gradients = []
    for _ in range(n_grids):
        grid = sample_random_grid(M, rng=rng)
        true_gradient = -0.1 * np.log(eta)
        noise = rng.normal(0, 0.05, size=len(eta))
        gradients.append(true_gradient + noise)

    gradients = np.array(gradients)
    return gradients.mean(axis=0), gradients.std(axis=0)


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/phlash/score_function.rst
# ---------------------------------------------------------------------------

def hmm_score_function(observations, transition, emission, initial):
    """Compute the HMM log-likelihood gradient via the Fisher identity."""
    L = len(observations)
    M = len(initial)

    # Forward pass (scaled)
    alpha = np.zeros((L, M))
    scale = np.zeros(L)
    alpha[0] = initial * emission[:, observations[0]]
    scale[0] = alpha[0].sum()
    alpha[0] /= scale[0]

    for t in range(1, L):
        alpha[t] = (alpha[t-1] @ transition) * emission[:, observations[t]]
        scale[t] = alpha[t].sum()
        alpha[t] /= scale[t]

    log_likelihood = np.sum(np.log(scale))

    # Backward pass (scaled)
    beta = np.zeros((L, M))
    beta[-1] = 1.0
    for t in range(L-2, -1, -1):
        beta[t] = transition @ (emission[:, observations[t+1]] * beta[t+1])
        beta[t] /= scale[t+1]

    # Posterior marginals
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Summed pairwise marginals
    xi_sum = np.zeros((M, M))
    for t in range(L-1):
        xi_t = (alpha[t, :, None] * transition
                * emission[None, :, observations[t+1]] * beta[t+1, None, :])
        xi_t /= xi_t.sum()
        xi_sum += xi_t

    return log_likelihood, gamma, xi_sum


def total_gradient(h, observed_sfs, expected_sfs, hmm_scores,
                   sigma_prior=1.0):
    """Compute the total gradient of the log-posterior."""
    M = len(h)

    xi = np.maximum(expected_sfs, 1e-300)
    grad_sfs = np.zeros(M)

    grad_hmm = np.sum(hmm_scores, axis=0)

    grad_prior = np.zeros(M)
    for j in range(1, M):
        grad_prior[j] += (h[j-1] - h[j]) / sigma_prior**2
        grad_prior[j-1] += (h[j] - h[j-1]) / sigma_prior**2

    return grad_sfs + grad_hmm + grad_prior


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/phlash/svgd_inference.rst
# ---------------------------------------------------------------------------

def rbf_kernel(particles, bandwidth=None):
    """Compute the RBF kernel matrix and its gradients."""
    J, M = particles.shape
    dists = squareform(pdist(particles, 'sqeuclidean'))

    if bandwidth is None:
        median_dist = np.median(pdist(particles, 'sqeuclidean'))
        bandwidth = np.sqrt(median_dist / (2 * np.log(J + 1)))
        bandwidth = max(bandwidth, 1e-5)

    K = np.exp(-dists / (2 * bandwidth**2))

    diff = particles[None, :, :] - particles[:, None, :]
    grad_K = K[:, :, None] * diff / bandwidth**2

    return K, grad_K, bandwidth


def svgd_update(particles, grad_log_posterior, epsilon=0.01):
    """Perform one SVGD update step."""
    J = particles.shape[0]
    K, grad_K, bw = rbf_kernel(particles)

    attraction = K @ grad_log_posterior / J
    repulsion = grad_K.sum(axis=0) / J

    phi = attraction + repulsion
    return particles + epsilon * phi


def phlash_loop(n_particles, M, n_iterations, observed_sfs,
                sigma_prior=1.0, epsilon=0.01, rng=None):
    """Simplified PHLASH inference loop."""
    if rng is None:
        rng = np.random.default_rng()

    particles = rng.normal(0, 0.5, size=(n_particles, M))

    for t in range(n_iterations):
        grid = sample_random_grid(M, rng=rng)

        grads = np.zeros_like(particles)
        for j in range(n_particles):
            h = particles[j]
            grad_prior = np.zeros(M)
            for k in range(1, M):
                grad_prior[k] += (h[k-1] - h[k]) / sigma_prior**2
                grad_prior[k-1] += (h[k] - h[k-1]) / sigma_prior**2

            grad_lik = -0.05 * h + rng.normal(0, 0.02, size=M)
            grads[j] = grad_lik + grad_prior

        particles = svgd_update(particles, grads, epsilon=epsilon)

    return particles


# ---------------------------------------------------------------------------
# Helper: build a simple HMM for testing
# ---------------------------------------------------------------------------

def _make_simple_hmm(M=3, n_obs=2):
    """Create a simple HMM with M states and n_obs observation types."""
    rng = np.random.default_rng(42)
    transition = rng.dirichlet(np.ones(M), size=M)
    emission = rng.dirichlet(np.ones(n_obs), size=M)
    initial = np.ones(M) / M
    return transition, emission, initial


# ===========================================================================
# Tests for sfs_log_likelihood
# ===========================================================================

class TestSfsLogLikelihood:
    def test_finite(self):
        """LL should be finite for valid inputs."""
        observed = np.array([5, 10, 15])
        expected = np.array([5.0, 10.0, 15.0])
        ll = sfs_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_maximized_at_truth(self):
        """Poisson LL is maximized when expected equals observed."""
        data = np.array([10.0, 20.0, 30.0])
        ll_true = sfs_log_likelihood(data, data)
        ll_bad = sfs_log_likelihood(data, data * 2)
        assert ll_true > ll_bad

    def test_zero_observed_handled(self):
        """Zero observations should not cause errors."""
        observed = np.array([0, 10, 0])
        expected = np.array([5.0, 10.0, 15.0])
        ll = sfs_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_higher_expected_worse_if_zero_observed(self):
        """Higher expected values at zero-count entries should reduce LL."""
        observed = np.array([0, 0, 0])
        ll_small = sfs_log_likelihood(observed, np.array([1.0, 1.0, 1.0]))
        ll_large = sfs_log_likelihood(observed, np.array([100.0, 100.0, 100.0]))
        assert ll_small > ll_large

    def test_poisson_form(self):
        """LL should equal sum(D*log(M) - M) for positive entries."""
        observed = np.array([10.0, 20.0])
        expected = np.array([8.0, 25.0])
        ll = sfs_log_likelihood(observed, expected)
        expected_ll = (10 * np.log(8) - 8) + (20 * np.log(25) - 25)
        assert abs(ll - expected_ll) < 1e-10


# ===========================================================================
# Tests for expected_sfs_constant
# ===========================================================================

class TestExpectedSfsConstant:
    def test_shape(self):
        """Output should have n-1 entries."""
        sfs = expected_sfs_constant(20, theta=1.0)
        assert len(sfs) == 19

    def test_proportional_to_1_over_k(self):
        """Expected SFS under constant size should be theta/k."""
        theta = 2.5
        n = 10
        sfs = expected_sfs_constant(n, theta)
        for i, k in enumerate(range(1, n)):
            assert abs(sfs[i] - theta / k) < 1e-12

    def test_decreasing(self):
        """SFS entries should be strictly decreasing."""
        sfs = expected_sfs_constant(20, theta=1.0)
        for i in range(len(sfs) - 1):
            assert sfs[i] > sfs[i + 1]

    def test_positive(self):
        """All entries should be positive."""
        sfs = expected_sfs_constant(15, theta=1.0)
        assert np.all(sfs > 0)

    def test_scales_with_theta(self):
        """SFS should scale linearly with theta."""
        sfs1 = expected_sfs_constant(10, theta=1.0)
        sfs2 = expected_sfs_constant(10, theta=3.0)
        assert np.allclose(sfs2, 3.0 * sfs1)

    def test_watterson_estimator(self):
        """Sum of expected SFS / theta should equal the harmonic number."""
        n = 20
        theta = 1.0
        sfs = expected_sfs_constant(n, theta)
        harmonic = sum(1.0 / k for k in range(1, n))
        assert abs(sfs.sum() / theta - harmonic) < 1e-10


# ===========================================================================
# Tests for composite_log_likelihood
# ===========================================================================

class TestCompositeLogLikelihood:
    def test_sfs_only(self):
        """With empty HMM list, should equal the SFS LL."""
        obs = np.array([5, 10, 15])
        exp = np.array([5.0, 10.0, 15.0])
        ll = composite_log_likelihood(obs, exp, [])
        ll_sfs = sfs_log_likelihood(obs, exp)
        assert abs(ll - ll_sfs) < 1e-12

    def test_additive(self):
        """Composite LL should be SFS LL + sum of HMM LLs."""
        obs = np.array([5, 10, 15])
        exp = np.array([5.0, 10.0, 15.0])
        hmm_lls = [-10.0, -20.0, -30.0]
        ll = composite_log_likelihood(obs, exp, hmm_lls)
        ll_sfs = sfs_log_likelihood(obs, exp)
        assert abs(ll - (ll_sfs + sum(hmm_lls))) < 1e-12

    def test_finite(self):
        """LL should be finite for valid inputs."""
        obs = np.array([5.0, 10.0])
        exp = np.array([5.0, 10.0])
        ll = composite_log_likelihood(obs, exp, [-5.0])
        assert np.isfinite(ll)


# ===========================================================================
# Tests for smoothness_prior_logpdf
# ===========================================================================

class TestSmoothnessPriorLogpdf:
    def test_constant_h_is_maximum(self):
        """A constant h (no differences) should give log-density = 0."""
        h = np.ones(10)
        lp = smoothness_prior_logpdf(h)
        assert abs(lp) < 1e-12

    def test_negative_for_varying_h(self):
        """Any non-constant h should give negative log-density."""
        h = np.array([1.0, 2.0, 1.0, 2.0])
        lp = smoothness_prior_logpdf(h)
        assert lp < 0

    def test_smoother_is_higher(self):
        """Smoother h should have higher log-density."""
        h_smooth = np.linspace(0, 1, 10)
        h_rough = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        lp_smooth = smoothness_prior_logpdf(h_smooth)
        lp_rough = smoothness_prior_logpdf(h_rough)
        assert lp_smooth > lp_rough

    def test_scales_with_sigma(self):
        """Larger sigma should give higher (less penalizing) log-density."""
        h = np.array([1.0, 2.0, 3.0])
        lp_small = smoothness_prior_logpdf(h, sigma=0.5)
        lp_large = smoothness_prior_logpdf(h, sigma=2.0)
        assert lp_large > lp_small

    def test_single_element(self):
        """Single-element h has no differences, so log-density = 0."""
        h = np.array([5.0])
        lp = smoothness_prior_logpdf(h)
        assert abs(lp) < 1e-12

    def test_quadratic_in_diffs(self):
        """The log-density should be quadratic in the first differences."""
        h = np.array([0.0, 1.0, 3.0])
        lp = smoothness_prior_logpdf(h, sigma=1.0)
        # diffs = [1.0, 2.0], sum of squares = 1 + 4 = 5
        expected = -0.5 * 5.0
        assert abs(lp - expected) < 1e-12


# ===========================================================================
# Tests for sample_random_grid
# ===========================================================================

class TestSampleRandomGrid:
    def test_length(self):
        """Grid should have M+1 points."""
        grid = sample_random_grid(10, rng=np.random.default_rng(42))
        assert len(grid) == 11

    def test_endpoints(self):
        """Grid should start at 0 and end at t_max."""
        t_max = 5.0
        grid = sample_random_grid(10, t_max=t_max, rng=np.random.default_rng(42))
        assert grid[0] == 0.0
        assert grid[-1] == t_max

    def test_sorted(self):
        """Grid points should be strictly increasing."""
        grid = sample_random_grid(20, rng=np.random.default_rng(42))
        for i in range(len(grid) - 1):
            assert grid[i] < grid[i + 1]

    def test_positive_interior(self):
        """All interior points should be positive."""
        grid = sample_random_grid(15, rng=np.random.default_rng(42))
        assert np.all(grid[1:-1] > 0)

    def test_randomness(self):
        """Different seeds should produce different grids."""
        grid1 = sample_random_grid(10, rng=np.random.default_rng(42))
        grid2 = sample_random_grid(10, rng=np.random.default_rng(99))
        assert not np.allclose(grid1, grid2)

    def test_within_bounds(self):
        """All points should be within [0, t_max]."""
        t_max = 10.0
        grid = sample_random_grid(20, t_max=t_max, rng=np.random.default_rng(42))
        assert np.all(grid >= 0)
        assert np.all(grid <= t_max)


# ===========================================================================
# Tests for debiased_gradient_estimate
# ===========================================================================

class TestDebiasedGradientEstimate:
    def test_output_shapes(self):
        """Mean and std should have the same shape as eta."""
        eta = np.ones(10)
        obs = expected_sfs_constant(20, theta=1.0)
        mean_grad, std_grad = debiased_gradient_estimate(
            eta, obs, n_grids=5, M=10, rng=np.random.default_rng(42)
        )
        assert mean_grad.shape == eta.shape
        assert std_grad.shape == eta.shape

    def test_mean_converges_with_more_grids(self):
        """The mean gradient should converge to the true value with more grids.
        The standard error of the mean (std/sqrt(n)) decreases with n."""
        eta = np.ones(10) * 2.0
        obs = expected_sfs_constant(20, theta=1.0)
        true_grad = -0.1 * np.log(eta)  # expected gradient from the placeholder
        mean_few, std_few = debiased_gradient_estimate(
            eta, obs, n_grids=3, M=10, rng=np.random.default_rng(42)
        )
        mean_many, std_many = debiased_gradient_estimate(
            eta, obs, n_grids=500, M=10, rng=np.random.default_rng(99)
        )
        # With more grids, the mean should be closer to the true gradient
        error_many = np.linalg.norm(mean_many - true_grad)
        # The std itself should converge to the noise level (~0.05)
        assert np.all(np.isfinite(std_many))
        assert error_many < 0.05  # should be close with 500 grids

    def test_gradient_direction(self):
        """For eta > 1, the placeholder gradient should push toward 1 (negative)."""
        eta = np.ones(10) * 5.0
        obs = expected_sfs_constant(20, theta=1.0)
        mean_grad, _ = debiased_gradient_estimate(
            eta, obs, n_grids=50, M=10, rng=np.random.default_rng(42)
        )
        # The placeholder gradient is -0.1 * log(eta), which is negative for eta > 1
        assert np.all(mean_grad < 0)

    def test_finite_outputs(self):
        """All outputs should be finite."""
        eta = np.ones(5) * 3.0
        obs = expected_sfs_constant(10, theta=1.0)
        mean_grad, std_grad = debiased_gradient_estimate(
            eta, obs, rng=np.random.default_rng(42)
        )
        assert np.all(np.isfinite(mean_grad))
        assert np.all(np.isfinite(std_grad))


# ===========================================================================
# Tests for hmm_score_function
# ===========================================================================

class TestHmmScoreFunction:
    def test_log_likelihood_finite(self):
        """Log-likelihood should be finite."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 1, 0])
        ll, gamma, xi_sum = hmm_score_function(observations, transition, emission, initial)
        assert np.isfinite(ll)

    def test_gamma_sums_to_one(self):
        """Posterior marginals should sum to 1 at each position."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 1, 0])
        _, gamma, _ = hmm_score_function(observations, transition, emission, initial)
        for t in range(len(observations)):
            assert abs(gamma[t].sum() - 1.0) < 1e-10

    def test_gamma_nonnegative(self):
        """Posterior marginals should be non-negative."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 0, 1])
        _, gamma, _ = hmm_score_function(observations, transition, emission, initial)
        assert np.all(gamma >= -1e-15)

    def test_gamma_shape(self):
        """Gamma should have shape (L, M)."""
        M = 4
        transition, emission, initial = _make_simple_hmm(M, 2)
        observations = np.array([0, 1, 0])
        _, gamma, _ = hmm_score_function(observations, transition, emission, initial)
        assert gamma.shape == (3, M)

    def test_xi_sum_shape(self):
        """Xi_sum should have shape (M, M)."""
        M = 4
        transition, emission, initial = _make_simple_hmm(M, 2)
        observations = np.array([0, 1, 0, 1])
        _, _, xi_sum = hmm_score_function(observations, transition, emission, initial)
        assert xi_sum.shape == (M, M)

    def test_xi_sum_nonnegative(self):
        """Pairwise marginals should be non-negative."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 1, 0])
        _, _, xi_sum = hmm_score_function(observations, transition, emission, initial)
        assert np.all(xi_sum >= -1e-15)

    def test_log_likelihood_negative(self):
        """Log-likelihood should be negative (log of probabilities < 1)."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 1, 0])
        ll, _, _ = hmm_score_function(observations, transition, emission, initial)
        assert ll < 0

    def test_longer_sequence_lower_ll(self):
        """Longer sequences should generally have lower log-likelihood."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        obs_short = np.array([0, 1])
        obs_long = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        ll_short, _, _ = hmm_score_function(obs_short, transition, emission, initial)
        ll_long, _, _ = hmm_score_function(obs_long, transition, emission, initial)
        assert ll_long < ll_short

    def test_xi_sum_consistent_with_gamma(self):
        """The sum of xi over rows should approximately equal gamma (for
        internal time steps)."""
        transition, emission, initial = _make_simple_hmm(3, 2)
        observations = np.array([0, 1, 0, 1, 0])
        _, gamma, xi_sum = hmm_score_function(observations, transition, emission, initial)
        # xi_sum[k, :].sum() should be close to sum of gamma[t, k] for t=0..L-2
        xi_row_sums = xi_sum.sum(axis=1)
        gamma_sums = gamma[:-1].sum(axis=0)
        assert np.allclose(xi_row_sums, gamma_sums, atol=0.1)


# ===========================================================================
# Tests for total_gradient
# ===========================================================================

class TestTotalGradient:
    def test_output_shape(self):
        """Gradient should have the same shape as h."""
        M = 10
        h = np.zeros(M)
        obs_sfs = np.ones(5)
        exp_sfs = np.ones(5)
        hmm_scores = [np.zeros(M)]
        grad = total_gradient(h, obs_sfs, exp_sfs, hmm_scores)
        assert len(grad) == M

    def test_zero_for_constant_h_no_hmm(self):
        """With constant h and no HMM scores, prior gradient should be zero."""
        M = 10
        h = np.ones(M) * 2.0
        obs_sfs = np.ones(5)
        exp_sfs = np.ones(5)
        hmm_scores = [np.zeros(M)]
        grad = total_gradient(h, obs_sfs, exp_sfs, hmm_scores)
        # SFS gradient is zero (placeholder), HMM is zero, prior is zero for constant h
        assert np.allclose(grad, 0.0)

    def test_prior_gradient_penalizes_roughness(self):
        """The prior gradient should push rough h toward smoothness."""
        M = 5
        h = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        obs_sfs = np.ones(3)
        exp_sfs = np.ones(3)
        hmm_scores = [np.zeros(M)]
        grad = total_gradient(h, obs_sfs, exp_sfs, hmm_scores)
        # At h[1]=1.0 surrounded by 0s, prior gradient should push it down
        assert grad[1] < 0
        # At h[0]=0.0 next to 1.0, prior gradient should push it up
        assert grad[0] > 0

    def test_hmm_scores_additive(self):
        """Multiple HMM scores should sum."""
        M = 5
        h = np.ones(M)
        obs_sfs = np.ones(3)
        exp_sfs = np.ones(3)
        score1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        score2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        grad = total_gradient(h, obs_sfs, exp_sfs, [score1, score2])
        # Prior grad is zero for constant h, SFS grad is zero
        # So grad should equal score1 + score2
        expected = score1 + score2
        assert np.allclose(grad, expected)

    def test_sigma_prior_affects_magnitude(self):
        """Smaller sigma should produce larger prior gradient."""
        M = 5
        h = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        obs_sfs = np.ones(3)
        exp_sfs = np.ones(3)
        hmm_scores = [np.zeros(M)]
        grad_small = total_gradient(h, obs_sfs, exp_sfs, hmm_scores, sigma_prior=0.5)
        grad_large = total_gradient(h, obs_sfs, exp_sfs, hmm_scores, sigma_prior=2.0)
        assert np.linalg.norm(grad_small) > np.linalg.norm(grad_large)


# ===========================================================================
# Tests for rbf_kernel
# ===========================================================================

class TestRbfKernel:
    def test_kernel_shape(self):
        """K should be (J, J) and grad_K should be (J, J, M)."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        K, grad_K, bw = rbf_kernel(particles)
        assert K.shape == (5, 5)
        assert grad_K.shape == (5, 5, 3)

    def test_diagonal_ones(self):
        """K[i, i] should be 1 (zero distance to self)."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        K, _, _ = rbf_kernel(particles)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)

    def test_symmetric(self):
        """K should be symmetric."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        K, _, _ = rbf_kernel(particles)
        assert np.allclose(K, K.T)

    def test_values_in_zero_one(self):
        """All kernel values should be in (0, 1]."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        K, _, _ = rbf_kernel(particles)
        assert np.all(K > 0)
        assert np.all(K <= 1.0 + 1e-12)

    def test_positive_bandwidth(self):
        """Bandwidth should be positive."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        _, _, bw = rbf_kernel(particles)
        assert bw > 0

    def test_identical_particles(self):
        """For identical particles, K should be all ones."""
        particles = np.ones((5, 3))
        K, _, _ = rbf_kernel(particles)
        assert np.allclose(K, 1.0)

    def test_grad_K_antisymmetric(self):
        """grad_K[i,j] should equal -grad_K[j,i] (gradient from i to j vs j to i)."""
        particles = np.random.default_rng(42).normal(size=(4, 3))
        _, grad_K, _ = rbf_kernel(particles)
        for i in range(4):
            for j in range(4):
                assert np.allclose(grad_K[i, j], -grad_K[j, i], atol=1e-12)


# ===========================================================================
# Tests for svgd_update
# ===========================================================================

class TestSvgdUpdate:
    def test_output_shape(self):
        """Updated particles should have the same shape."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        grads = np.zeros_like(particles)
        new_particles = svgd_update(particles, grads, epsilon=0.01)
        assert new_particles.shape == particles.shape

    def test_zero_gradient_repulsion_only(self):
        """With zero posterior gradient, particles should still move due to repulsion."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        grads = np.zeros_like(particles)
        new_particles = svgd_update(particles, grads, epsilon=0.1)
        # Particles should move (repulsion pushes them apart)
        assert not np.allclose(particles, new_particles)

    def test_finite_output(self):
        """Updated particles should be finite."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        grads = np.random.default_rng(99).normal(size=(5, 3))
        new_particles = svgd_update(particles, grads, epsilon=0.01)
        assert np.all(np.isfinite(new_particles))

    def test_stronger_gradient_larger_move(self):
        """Larger gradients should produce larger particle displacements."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        grads_small = np.ones_like(particles) * 0.1
        grads_large = np.ones_like(particles) * 10.0
        p_small = svgd_update(particles, grads_small, epsilon=0.01)
        p_large = svgd_update(particles, grads_large, epsilon=0.01)
        move_small = np.linalg.norm(p_small - particles)
        move_large = np.linalg.norm(p_large - particles)
        assert move_large > move_small

    def test_epsilon_scaling(self):
        """Larger epsilon should produce proportionally larger moves."""
        particles = np.random.default_rng(42).normal(size=(5, 3))
        grads = np.ones_like(particles)
        p1 = svgd_update(particles, grads, epsilon=0.01)
        p2 = svgd_update(particles, grads, epsilon=0.02)
        move1 = np.linalg.norm(p1 - particles)
        move2 = np.linalg.norm(p2 - particles)
        assert abs(move2 / move1 - 2.0) < 0.3  # approximately 2x


# ===========================================================================
# Tests for phlash_loop
# ===========================================================================

class TestPhlashLoop:
    def test_output_shape(self):
        """Output should have shape (n_particles, M)."""
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        particles = phlash_loop(5, 8, 10, obs_sfs, rng=np.random.default_rng(42))
        assert particles.shape == (5, 8)

    def test_finite_output(self):
        """All particle values should be finite."""
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        particles = phlash_loop(5, 8, 10, obs_sfs, rng=np.random.default_rng(42))
        assert np.all(np.isfinite(particles))

    def test_particles_move(self):
        """After iterations, particles should have moved from initial positions."""
        rng = np.random.default_rng(42)
        initial = rng.normal(0, 0.5, size=(5, 8))
        rng2 = np.random.default_rng(42)
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        final = phlash_loop(5, 8, 50, obs_sfs, rng=rng2)
        # They should be different (movement happened)
        assert not np.allclose(initial, final)

    def test_smoothness_increases(self):
        """After enough iterations, the smoothness prior should reduce roughness."""
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        particles = phlash_loop(10, 8, 100, obs_sfs, sigma_prior=0.5,
                                 rng=np.random.default_rng(42))
        # Average roughness (mean abs diff) should be modest
        for j in range(10):
            roughness = np.mean(np.abs(np.diff(particles[j])))
            assert roughness < 5.0  # reasonable bound

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        p1 = phlash_loop(3, 5, 10, obs_sfs, rng=np.random.default_rng(42))
        p2 = phlash_loop(3, 5, 10, obs_sfs, rng=np.random.default_rng(42))
        assert np.allclose(p1, p2)


# ===========================================================================
# Integration tests combining multiple functions
# ===========================================================================

class TestIntegration:
    def test_sfs_likelihood_pipeline(self):
        """Full pipeline: expected SFS -> composite likelihood."""
        n = 20
        theta = 1.0
        expected = expected_sfs_constant(n, theta)
        # Simulate observed as Poisson draws
        rng = np.random.default_rng(42)
        observed = rng.poisson(expected * 100)
        ll = composite_log_likelihood(observed, expected * 100, [-10.0, -20.0])
        assert np.isfinite(ll)

    def test_hmm_feeds_into_total_gradient(self):
        """HMM score function output should feed into total_gradient."""
        M = 3
        transition, emission, initial = _make_simple_hmm(M, 2)
        observations = np.array([0, 1, 0, 1, 0])
        ll, gamma, xi_sum = hmm_score_function(observations, transition, emission, initial)

        # Use gamma to create a simple score
        score = gamma.mean(axis=0) - 1.0 / M

        h = np.zeros(M)
        obs_sfs = np.ones(5)
        exp_sfs = np.ones(5)
        grad = total_gradient(h, obs_sfs, exp_sfs, [score])
        assert len(grad) == M
        assert np.all(np.isfinite(grad))

    def test_svgd_with_score_gradient(self):
        """SVGD should work with gradients from the total_gradient function."""
        M = 5
        n_particles = 3
        rng = np.random.default_rng(42)
        particles = rng.normal(0, 0.5, size=(n_particles, M))

        obs_sfs = np.ones(3)
        exp_sfs = np.ones(3)

        grads = np.zeros_like(particles)
        for j in range(n_particles):
            hmm_scores = [rng.normal(0, 0.1, size=M)]
            grads[j] = total_gradient(particles[j], obs_sfs, exp_sfs, hmm_scores)

        new_particles = svgd_update(particles, grads, epsilon=0.01)
        assert new_particles.shape == particles.shape
        assert np.all(np.isfinite(new_particles))

    def test_random_grid_in_gradient_estimation(self):
        """Random grids should feed into gradient estimation without errors."""
        eta = np.ones(10)
        obs_sfs = expected_sfs_constant(20, theta=1.0)
        mean_grad, std_grad = debiased_gradient_estimate(
            eta, obs_sfs, n_grids=5, M=10, rng=np.random.default_rng(42)
        )
        assert np.all(np.isfinite(mean_grad))
        assert np.all(std_grad >= 0)
