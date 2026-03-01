"""
Mini-PHLASH: Population History Learning by Averaging Sampled Histories.

PHLASH is a Bayesian method for inferring population size history N(t) from
whole-genome sequencing data.  It extends PSMC's coalescent HMM framework
with four key innovations:

1. **Composite likelihood** -- combines the site frequency spectrum (SFS)
   from many individuals with the pairwise coalescent HMM from diploid
   genomes into a single objective.

2. **Random time discretisation** -- randomised breakpoints whose biases
   cancel when averaged, eliminating the systematic error of fixed grids.

3. **Score function algorithm** -- an O(LM^2) gradient computation that is
   30--90x faster than reverse-mode automatic differentiation.

4. **Stein Variational Gradient Descent (SVGD)** -- a GPU-parallel posterior
   sampling algorithm that maintains a set of particles (candidate
   demographic histories) and iteratively pushes them toward the posterior
   distribution.

This module provides simplified, self-contained implementations of each
component using only NumPy and SciPy.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------------------
# Composite likelihood (composite_likelihood.rst)
# ---------------------------------------------------------------------------

def sfs_log_likelihood(observed_sfs, expected_sfs):
    """Poisson log-likelihood of the observed SFS given expected SFS.

    Parameters
    ----------
    observed_sfs : ndarray, shape (n-1,)
        Observed SFS counts D_k for k = 1, ..., n-1.
    expected_sfs : ndarray, shape (n-1,)
        Expected SFS entries xi_k under the demographic model.

    Returns
    -------
    ll : float
        Poisson log-likelihood (up to a constant).
    """
    # Avoid log(0) for zero-expected entries
    xi = np.maximum(expected_sfs, 1e-300)
    return np.sum(observed_sfs * np.log(xi) - xi)


def expected_sfs_constant(n, theta, N_e=1.0):
    """Expected SFS under constant population size.

    Under the standard coalescent with constant N_e, the expected
    number of segregating sites at frequency k/n is proportional
    to 1/k (Watterson's result).

    Parameters
    ----------
    n : int
        Number of haploid chromosomes.
    theta : float
        Population-scaled mutation rate (4 * N_e * mu * L).
    N_e : float
        Effective population size (default 1.0 in coalescent units).

    Returns
    -------
    xi : ndarray, shape (n-1,)
        Expected SFS.
    """
    k = np.arange(1, n)
    return theta / k


def composite_log_likelihood(observed_sfs, expected_sfs, hmm_log_likelihoods):
    """Compute the composite log-likelihood (SFS + coalescent HMM).

    Parameters
    ----------
    observed_sfs : ndarray
        Observed SFS counts.
    expected_sfs : ndarray
        Expected SFS under the candidate history.
    hmm_log_likelihoods : list of float
        Log-likelihood from each pairwise coalescent HMM.

    Returns
    -------
    ll : float
        Composite log-likelihood.
    """
    ll_sfs = sfs_log_likelihood(observed_sfs, expected_sfs)
    ll_hmm = sum(hmm_log_likelihoods)
    return ll_sfs + ll_hmm


def smoothness_prior_logpdf(h, sigma=1.0):
    """Log-density of the Gaussian smoothness prior on log-eta.

    Penalizes large differences between adjacent time intervals.

    Parameters
    ----------
    h : ndarray, shape (M,)
        Log population sizes (h = log eta).
    sigma : float
        Smoothness scale.

    Returns
    -------
    lp : float
        Log prior density (up to a constant).
    """
    diffs = np.diff(h)
    return -0.5 * np.sum(diffs**2) / sigma**2


# ---------------------------------------------------------------------------
# Random time discretisation (random_discretization.rst)
# ---------------------------------------------------------------------------

def sample_random_grid(M, t_max=10.0, t_min=1e-4, rng=None):
    """Sample a random time discretization grid.

    Interior breakpoints are log-uniformly spaced with random jitter,
    ensuring approximately uniform density per unit log-time.

    Parameters
    ----------
    M : int
        Number of time intervals.
    t_max : float
        Maximum time (coalescent units).
    t_min : float
        Minimum positive breakpoint.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    grid : ndarray, shape (M+1,)
        Sorted breakpoints [0, t_1, ..., t_{M-1}, t_max].
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample M-1 interior breakpoints in log-space with jitter
    log_min, log_max = np.log(t_min), np.log(t_max)
    # Evenly spaced anchors in log-space
    anchors = np.linspace(log_min, log_max, M - 1)
    # Add uniform jitter of half the spacing
    spacing = (log_max - log_min) / (M - 1)
    jitter = rng.uniform(-0.5 * spacing, 0.5 * spacing, size=M - 1)
    log_breakpoints = anchors + jitter

    # Convert back, sort, and add endpoints
    interior = np.sort(np.clip(np.exp(log_breakpoints), t_min, t_max * 0.999))
    grid = np.concatenate([[0.0], interior, [t_max]])
    return grid


def debiased_gradient_estimate(eta, observed_sfs, n_grids=10, M=32,
                                rng=None):
    """Estimate the likelihood gradient by averaging over random grids.

    This demonstrates the debiasing principle: each individual gradient
    is biased, but their average converges to the true gradient.

    Parameters
    ----------
    eta : ndarray, shape (M,)
        Population sizes at each time interval.
    observed_sfs : ndarray
        Observed SFS.
    n_grids : int
        Number of random grids to average over.
    M : int
        Number of time intervals per grid.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    mean_gradient : ndarray
        Averaged gradient estimate.
    std_gradient : ndarray
        Standard deviation across grid evaluations.
    """
    if rng is None:
        rng = np.random.default_rng()

    gradients = []
    for _ in range(n_grids):
        grid = sample_random_grid(M, rng=rng)
        # In a real implementation, this would compute the HMM gradient
        # on this grid. Here we simulate a noisy gradient.
        true_gradient = -0.1 * np.log(eta)  # placeholder: pulls toward 1
        noise = rng.normal(0, 0.05, size=len(eta))
        gradients.append(true_gradient + noise)

    gradients = np.array(gradients)
    return gradients.mean(axis=0), gradients.std(axis=0)


# ---------------------------------------------------------------------------
# Score function algorithm (score_function.rst)
# ---------------------------------------------------------------------------

def hmm_score_function(observations, transition, emission, initial):
    """Compute the HMM log-likelihood gradient via the Fisher identity.

    This implements the score function algorithm: run forward-backward
    to get posterior marginals, then use them to weight the parameter
    derivatives of the complete-data log-likelihood.

    Parameters
    ----------
    observations : ndarray, shape (L,)
        Integer observation sequence.
    transition : ndarray, shape (M, M)
        Transition probability matrix p_{kl}.
    emission : ndarray, shape (M, n_obs)
        Emission probabilities e_k(x).
    initial : ndarray, shape (M,)
        Initial state distribution.

    Returns
    -------
    log_likelihood : float
        The log-likelihood of the observations.
    gamma : ndarray, shape (L, M)
        Posterior state marginals at each position.
    xi_sum : ndarray, shape (M, M)
        Summed posterior pairwise marginals (transition counts).
    """
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

    # Posterior marginals gamma_t(k) = alpha_t(k) * beta_t(k)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)

    # Summed pairwise marginals: xi_sum(k, l) = sum_t xi_t(k, l)
    xi_sum = np.zeros((M, M))
    for t in range(L-1):
        xi_t = (alpha[t, :, None] * transition
                * emission[None, :, observations[t+1]] * beta[t+1, None, :])
        xi_t /= xi_t.sum()
        xi_sum += xi_t

    return log_likelihood, gamma, xi_sum


def total_gradient(h, observed_sfs, expected_sfs, hmm_scores,
                   sigma_prior=1.0):
    """Compute the total gradient of the log-posterior.

    Combines the SFS gradient, summed HMM score functions, and the
    smoothness prior gradient.

    Parameters
    ----------
    h : ndarray, shape (M,)
        Log population sizes.
    observed_sfs : ndarray
        Observed SFS.
    expected_sfs : ndarray
        Expected SFS under current h.
    hmm_scores : list of ndarray, each shape (M,)
        Score function (gradient) from each pairwise HMM.
    sigma_prior : float
        Smoothness prior scale.

    Returns
    -------
    grad : ndarray, shape (M,)
        Gradient of the composite log-posterior.
    """
    M = len(h)

    # SFS gradient: sum_k (D_k / xi_k - 1) * d(xi_k)/d(h)
    # Simplified: for constant-size model, d(xi_k)/d(h) ~ xi_k
    xi = np.maximum(expected_sfs, 1e-300)
    grad_sfs = np.zeros(M)  # placeholder for full implementation

    # HMM gradient: sum over pairs
    grad_hmm = np.sum(hmm_scores, axis=0)

    # Prior gradient: d/dh [-0.5 * sum((h_j - h_{j-1})^2) / sigma^2]
    grad_prior = np.zeros(M)
    for j in range(1, M):
        grad_prior[j] += (h[j-1] - h[j]) / sigma_prior**2
        grad_prior[j-1] += (h[j] - h[j-1]) / sigma_prior**2

    return grad_sfs + grad_hmm + grad_prior


# ---------------------------------------------------------------------------
# SVGD inference (svgd_inference.rst)
# ---------------------------------------------------------------------------

def rbf_kernel(particles, bandwidth=None):
    """Compute the RBF kernel matrix and its gradients.

    Parameters
    ----------
    particles : ndarray, shape (J, M)
        J particles, each of dimension M.
    bandwidth : float or None
        Kernel bandwidth. If None, uses the median heuristic.

    Returns
    -------
    K : ndarray, shape (J, J)
        Kernel matrix K_{ij} = exp(-||h_i - h_j||^2 / (2 sigma^2)).
    grad_K : ndarray, shape (J, J, M)
        grad_K[i, j] = gradient of K_{ij} with respect to h_i.
    bandwidth : float
        The bandwidth used.
    """
    J, M = particles.shape
    dists = squareform(pdist(particles, 'sqeuclidean'))

    # Median heuristic for bandwidth
    if bandwidth is None:
        median_dist = np.median(pdist(particles, 'sqeuclidean'))
        bandwidth = np.sqrt(median_dist / (2 * np.log(J + 1)))
        bandwidth = max(bandwidth, 1e-5)

    K = np.exp(-dists / (2 * bandwidth**2))

    # Gradient: dK_{ij}/dh_i = K_{ij} * (h_j - h_i) / sigma^2
    diff = particles[None, :, :] - particles[:, None, :]  # (J, J, M)
    grad_K = K[:, :, None] * diff / bandwidth**2

    return K, grad_K, bandwidth


def svgd_update(particles, grad_log_posterior, epsilon=0.01):
    """Perform one SVGD update step.

    Parameters
    ----------
    particles : ndarray, shape (J, M)
        Current particle positions.
    grad_log_posterior : ndarray, shape (J, M)
        Gradient of log-posterior at each particle.
    epsilon : float
        Step size.

    Returns
    -------
    particles_new : ndarray, shape (J, M)
        Updated particle positions.
    """
    J = particles.shape[0]
    K, grad_K, bw = rbf_kernel(particles)

    # phi*(h) = (1/J) * sum_j [ K(h_j, h) * grad_j + grad_K(h_j, h) ]
    # Attraction: K @ grad_log_posterior
    attraction = K @ grad_log_posterior / J      # (J, M)
    # Repulsion: sum_j grad_K[j, :, :]
    repulsion = grad_K.sum(axis=0) / J           # (J, M)

    phi = attraction + repulsion
    return particles + epsilon * phi


def phlash_loop(n_particles, M, n_iterations, observed_sfs,
                sigma_prior=1.0, epsilon=0.01, rng=None):
    """Simplified PHLASH inference loop.

    Demonstrates the full pipeline: random discretization, score
    function, and SVGD update. Uses placeholder likelihoods.

    Parameters
    ----------
    n_particles : int
        Number of SVGD particles (J).
    M : int
        Number of time intervals.
    n_iterations : int
        Number of SVGD iterations.
    observed_sfs : ndarray
        Observed SFS.
    sigma_prior : float
        Smoothness prior scale.
    epsilon : float
        SVGD step size.
    rng : numpy.random.Generator or None

    Returns
    -------
    particles : ndarray, shape (J, M)
        Final particle positions (log population sizes).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Initialize particles near the prior mean
    particles = rng.normal(0, 0.5, size=(n_particles, M))

    for t in range(n_iterations):
        # Step 1: sample a random grid (tourbillon)
        grid = sample_random_grid(M, rng=rng)

        # Step 2: compute gradient for each particle
        grads = np.zeros_like(particles)
        for j in range(n_particles):
            h = particles[j]
            # Prior gradient
            grad_prior = np.zeros(M)
            for k in range(1, M):
                grad_prior[k] += (h[k-1] - h[k]) / sigma_prior**2
                grad_prior[k-1] += (h[k] - h[k-1]) / sigma_prior**2

            # Placeholder likelihood gradient (pulls toward 0 = constant)
            grad_lik = -0.05 * h + rng.normal(0, 0.02, size=M)
            grads[j] = grad_lik + grad_prior

        # Steps 3-4: SVGD update (kernel + attraction + repulsion)
        particles = svgd_update(particles, grads, epsilon=epsilon)

    return particles


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate all components of the PHLASH algorithm."""

    # -- Composite likelihood demo (composite_likelihood.rst) --
    n = 20
    theta = 100.0  # realistic total theta for a genomic region
    xi_expected = expected_sfs_constant(n, theta)
    # Simulate observed SFS by Poisson sampling
    np.random.seed(42)
    D_observed = np.random.poisson(xi_expected)
    ll = sfs_log_likelihood(D_observed, xi_expected)
    print(f"Sample size n = {n}, theta = {theta}")
    print(f"Expected SFS (first 5): {xi_expected[:5].round(2)}")
    print(f"Observed SFS (first 5): {D_observed[:5]}")
    print(f"SFS log-likelihood: {ll:.2f}")

    # Composite log-posterior
    M = 32  # number of time intervals
    h_true = np.zeros(M)  # log(eta) = 0 means eta = 1 (constant size)
    h_true[10:20] = -1.0  # a bottleneck: eta = exp(-1) ~ 0.37

    lp_prior = smoothness_prior_logpdf(h_true, sigma=1.0)
    ll_sfs = sfs_log_likelihood(D_observed, xi_expected)
    # Placeholder HMM log-likelihoods for 5 pairs
    ll_hmms = [-500.0, -480.0, -510.0, -490.0, -505.0]
    ll_comp = composite_log_likelihood(D_observed, xi_expected, ll_hmms)
    log_posterior = ll_comp + lp_prior

    print(f"\nSFS log-likelihood:       {ll_sfs:.2f}")
    print(f"HMM log-likelihood (sum): {sum(ll_hmms):.2f}")
    print(f"Composite log-likelihood: {ll_comp:.2f}")
    print(f"Prior log-density:        {lp_prior:.2f}")
    print(f"Log-posterior:            {log_posterior:.2f}")

    # -- Random discretisation demo (random_discretization.rst) --
    print()
    rng = np.random.default_rng(42)
    for i in range(3):
        grid = sample_random_grid(M=32, t_max=10.0, rng=rng)
        print(f"Grid {i}: {len(grid)} breakpoints, "
              f"t_1={grid[1]:.5f}, t_mid={grid[16]:.4f}, "
              f"t_max={grid[-1]:.1f}")

    # Show that the grids differ (tourbillon rotates through configurations)
    g1 = sample_random_grid(32, rng=np.random.default_rng(0))
    g2 = sample_random_grid(32, rng=np.random.default_rng(1))
    max_diff = np.max(np.abs(g1 - g2))
    print(f"\nMax difference between two grids: {max_diff:.4f}")
    print("(Different grids = different biases = cancellation when averaged)")

    # Variance reduction via averaging
    print()
    eta = np.exp(np.zeros(32))  # constant population size
    rng = np.random.default_rng(42)
    for K in [1, 5, 20]:
        mean_grad, std_grad = debiased_gradient_estimate(
            eta, D_observed, n_grids=K, rng=rng
        )
        print(f"K={K:2d} grids: mean |grad| = {np.mean(np.abs(mean_grad)):.4f}, "
              f"std = {np.mean(std_grad):.4f}")
    print("(Variance decreases as 1/sqrt(K) -- more grids, less noise)")

    # -- Score function demo (score_function.rst) --
    print()
    M_hmm = 2
    transition = np.array([[0.99, 0.01],
                            [0.02, 0.98]])
    emission = np.array([[0.999, 0.001],   # state 0: mostly hom
                          [0.95,  0.05]])   # state 1: some hets
    initial = np.array([0.5, 0.5])

    # Synthetic observation sequence
    np.random.seed(42)
    obs = np.zeros(200, dtype=int)
    obs[50] = obs[120] = obs[180] = 1  # three het sites

    ll_hmm, gamma, xi_sum = hmm_score_function(obs, transition, emission, initial)
    print(f"Log-likelihood: {ll_hmm:.2f}")
    print(f"Posterior at het site (pos 50): "
          f"state 0 = {gamma[50, 0]:.3f}, state 1 = {gamma[50, 1]:.3f}")
    print(f"Transition counts (sum of xi):")
    print(f"  0->0: {xi_sum[0, 0]:.1f}, 0->1: {xi_sum[0, 1]:.2f}")
    print(f"  1->0: {xi_sum[1, 0]:.2f}, 1->1: {xi_sum[1, 1]:.1f}")

    # Total gradient demo
    print()
    M_grad = 32
    h = np.zeros(M_grad)
    h[10:20] = -0.5  # mild bottleneck
    # Simulate HMM score functions from 5 pairs
    rng = np.random.default_rng(42)
    hmm_scores = [rng.normal(0, 0.1, size=M_grad) for _ in range(5)]
    xi_expected_grad = expected_sfs_constant(20, 100.0)
    grad = total_gradient(h, D_observed, xi_expected_grad, hmm_scores)
    print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
    print(f"Gradient at bottleneck (interval 15): {grad[15]:.4f}")
    print(f"Gradient at constant (interval 5):    {grad[5]:.4f}")

    # -- SVGD demo (svgd_inference.rst) --
    print()
    J = 16   # particles
    M_svgd = 2    # dimensions (for visualization clarity)
    rng = np.random.default_rng(42)
    particles = rng.normal(0, 2, size=(J, M_svgd))

    # Target: standard normal (grad log p = -h)
    for step in range(50):
        grad_lp = -particles  # gradient of log N(0, I)
        particles = svgd_update(particles, grad_lp, epsilon=0.1)

    print(f"After 50 SVGD steps ({J} particles, {M_svgd}D):")
    print(f"  Particle mean: {particles.mean(axis=0).round(3)}")
    print(f"  Particle std:  {particles.std(axis=0).round(3)}")
    print(f"  (Target: mean ~ 0, std ~ 1)")

    # Full PHLASH loop
    print()
    rng = np.random.default_rng(42)
    particles = phlash_loop(
        n_particles=8, M=16, n_iterations=100,
        observed_sfs=D_observed, epsilon=0.05, rng=rng
    )
    eta_particles = np.exp(particles)  # convert to population size
    posterior_mean = eta_particles.mean(axis=0)
    posterior_std = eta_particles.std(axis=0)

    print(f"PHLASH result ({8} particles, {16} intervals, 100 iterations):")
    print(f"  Posterior mean N_e (first 5): "
          f"{posterior_mean[:5].round(3)}")
    print(f"  Posterior std  N_e (first 5): "
          f"{posterior_std[:5].round(3)}")
    print(f"  (All particles provide uncertainty quantification)")


if __name__ == "__main__":
    demo()
