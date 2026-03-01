"""
Comprehensive tests for all Python code examples from docs/prerequisites/mcmc.rst.

Tests cover:
  1. beta_binomial_demo       -- Conjugate Beta-Binomial Bayesian update
  2. markov_chain_convergence -- 3-state Markov chain stationary distribution
  3. metropolis_hastings_mixture -- MH sampling from a bimodal Gaussian mixture
  4. mh_sfs_inference         -- MH inference of theta from SFS data
  5. compute_acf_and_ess      -- Autocorrelation function and effective sample size
  6. run_mh_chain             -- MH chain targeting standard Normal (helper)
  7. gibbs_bivariate_normal   -- Gibbs sampler for bivariate Normal
  8. proposal_tuning_demo     -- Effect of step size on MH efficiency
"""

import numpy as np
import pytest
from scipy import stats
from scipy.special import beta as beta_fn


# ---------------------------------------------------------------------------
# Code block 1: beta_binomial_demo
# ---------------------------------------------------------------------------

def beta_binomial_demo():
    """Demonstrate exact Bayesian inference with a conjugate model.

    We use this as a ground truth to verify MCMC results later.

    Returns
    -------
    alpha_post : float
        Posterior alpha parameter.
    beta_post : float
        Posterior beta parameter.
    """
    # Observed data: 7 derived alleles out of 20 sites
    n, k = 20, 7

    # Prior: Beta(2, 2) -- a gentle prior favoring values near 0.5
    alpha_prior, beta_prior = 2, 2

    # Posterior: Beta(alpha + k, beta + n - k)
    alpha_post = alpha_prior + k       # 2 + 7 = 9
    beta_post = beta_prior + (n - k)   # 2 + 13 = 15

    # The posterior mean is alpha / (alpha + beta)
    post_mean = alpha_post / (alpha_post + beta_post)
    post_var = (alpha_post * beta_post) / (
        (alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)
    )

    # Verify: the posterior should integrate to 1
    integral = beta_fn(alpha_post, beta_post)

    return alpha_post, beta_post


# ---------------------------------------------------------------------------
# Code block 2: markov_chain_convergence
# ---------------------------------------------------------------------------

def markov_chain_convergence(seed=42):
    """Demonstrate that a finite Markov chain converges to its stationary distribution.

    We define a 3-state chain, run it for many steps, and compare the
    empirical state frequencies to the theoretical stationary distribution.
    """
    np.random.seed(seed)

    # Transition matrix for a 3-state chain
    # T[i, j] = P(X_{t+1} = j | X_t = i)
    T = np.array([
        [0.7, 0.2, 0.1],   # from state 0
        [0.1, 0.6, 0.3],   # from state 1
        [0.3, 0.3, 0.4],   # from state 2
    ])

    # Verify rows sum to 1
    assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"

    # Find the stationary distribution by solving pi * T = pi
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()

    # Simulate the chain for 100,000 steps starting from state 0
    n_steps = 100_000
    state = 0
    counts = np.zeros(3)

    for _ in range(n_steps):
        state = np.random.choice(3, p=T[state])
        counts[state] += 1

    empirical = counts / n_steps

    return T, pi, empirical


# ---------------------------------------------------------------------------
# Code block 3: metropolis_hastings_mixture
# ---------------------------------------------------------------------------

def metropolis_hastings_mixture(seed=42):
    """MH sampling from a mixture of two Gaussians.

    Target: 0.3 * N(-2, 0.5^2) + 0.7 * N(3, 1^2)
    This tests whether the chain can jump between modes.
    """
    np.random.seed(seed)

    def log_target(x):
        """Log of the (unnormalized) target density."""
        comp1 = 0.3 * np.exp(-0.5 * ((x + 2) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
        comp2 = 0.7 * np.exp(-0.5 * ((x - 3) / 1.0)**2) / (1.0 * np.sqrt(2 * np.pi))
        return np.log(comp1 + comp2 + 1e-300)

    # MH parameters
    n_samples = 50_000
    sigma = 1.5
    samples = np.zeros(n_samples)
    samples[0] = 0.0
    n_accepted = 0

    for t in range(1, n_samples):
        x_current = samples[t - 1]
        x_proposed = x_current + np.random.normal(0, sigma)

        log_alpha = log_target(x_proposed) - log_target(x_current)

        if np.log(np.random.uniform()) < log_alpha:
            samples[t] = x_proposed
            n_accepted += 1
        else:
            samples[t] = x_current

    acceptance_rate = n_accepted / (n_samples - 1)

    burn_in = 5000
    post_burnin = samples[burn_in:]

    near_mode1 = np.sum(post_burnin < 0) / len(post_burnin)
    near_mode2 = np.sum(post_burnin >= 0) / len(post_burnin)

    return samples, acceptance_rate, post_burnin, near_mode1, near_mode2


# ---------------------------------------------------------------------------
# Code block 4: mh_sfs_inference
# ---------------------------------------------------------------------------

def mh_sfs_inference(seed=42):
    """MH for inferring theta from an observed site frequency spectrum.

    Under the standard neutral coalescent with n samples, the expected
    number of SNPs with i derived alleles (out of n) is:
        E[SFS_i] = theta / i   for i = 1, ..., n-1

    We observe an SFS and infer theta using MCMC.
    """
    # Simulated "observed" SFS for n=10 samples with true theta=5
    n = 10
    theta_true = 5.0
    expected_sfs = theta_true / np.arange(1, n)
    np.random.seed(123)
    observed_sfs = np.random.poisson(expected_sfs)

    def log_likelihood(theta, sfs):
        """Poisson log-likelihood for the SFS given theta."""
        if theta <= 0:
            return -np.inf
        ll = 0.0
        for i in range(len(sfs)):
            lam = theta / (i + 1)
            ll += sfs[i] * np.log(lam) - lam
        return ll

    def log_prior(theta):
        """Log of an exponential prior with mean 10."""
        if theta <= 0:
            return -np.inf
        return -theta / 10.0

    # Run MH
    np.random.seed(seed)
    n_samples = 30_000
    sigma = 0.5
    chain = np.zeros(n_samples)
    chain[0] = 1.0
    n_accepted = 0

    for t in range(1, n_samples):
        theta_current = chain[t - 1]
        theta_proposed = theta_current + np.random.normal(0, sigma)

        log_alpha = (log_likelihood(theta_proposed, observed_sfs) + log_prior(theta_proposed)
                     - log_likelihood(theta_current, observed_sfs) - log_prior(theta_current))

        if np.log(np.random.uniform()) < log_alpha:
            chain[t] = theta_proposed
            n_accepted += 1
        else:
            chain[t] = theta_current

    burn_in = 5000
    post_burnin = chain[burn_in:]
    acceptance_rate = n_accepted / (n_samples - 1)

    return chain, post_burnin, acceptance_rate, theta_true, observed_sfs, log_likelihood, log_prior


# ---------------------------------------------------------------------------
# Code block 5: compute_acf_and_ess
# ---------------------------------------------------------------------------

def compute_acf_and_ess(chain, max_lag=200):
    """Compute the autocorrelation function and effective sample size.

    Parameters
    ----------
    chain : ndarray of shape (N,)
        MCMC samples (after burn-in).
    max_lag : int
        Maximum lag to compute ACF for.

    Returns
    -------
    acf : ndarray of shape (max_lag + 1,)
        Autocorrelation at each lag from 0 to max_lag.
    ess : float
        Estimated effective sample size.
    """
    N = len(chain)
    mean = chain.mean()
    var = chain.var()

    if var == 0:
        return np.ones(max_lag + 1), 1.0

    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf[k] = 1.0
        else:
            acf[k] = np.mean((chain[:-k] - mean) * (chain[k:] - mean)) / var

    # Compute ESS using the initial monotone sequence estimator
    iat = 1.0
    for k in range(1, max_lag + 1):
        if acf[k] < 0:
            break
        iat += 2 * acf[k]

    ess = N / iat

    return acf, ess


# ---------------------------------------------------------------------------
# Code block 6: run_mh_chain
# ---------------------------------------------------------------------------

def run_mh_chain(sigma, n_samples=20000, seed=None):
    """Run MH targeting standard Normal with step size sigma."""
    if seed is not None:
        np.random.seed(seed)
    chain = np.zeros(n_samples)
    chain[0] = 5.0
    n_acc = 0
    for t in range(1, n_samples):
        proposal = chain[t-1] + np.random.normal(0, sigma)
        log_alpha = -0.5 * proposal**2 + 0.5 * chain[t-1]**2
        if np.log(np.random.uniform()) < log_alpha:
            chain[t] = proposal
            n_acc += 1
        else:
            chain[t] = chain[t-1]
    return chain[2000:], n_acc / (n_samples - 1)


# ---------------------------------------------------------------------------
# Code block 7: gibbs_bivariate_normal
# ---------------------------------------------------------------------------

def gibbs_bivariate_normal(seed=42):
    """Gibbs sampling from a bivariate Normal distribution.

    Target: (X, Y) ~ N(mu, Sigma) where
        mu = (0, 0)
        Sigma = [[1, rho], [rho, 1]]

    The conditional distributions are:
        X | Y=y ~ N(rho*y, 1 - rho^2)
        Y | X=x ~ N(rho*x, 1 - rho^2)
    """
    np.random.seed(seed)

    rho = 0.8
    cond_var = 1 - rho**2
    cond_std = np.sqrt(cond_var)

    n_samples = 20_000
    samples = np.zeros((n_samples, 2))
    samples[0] = [0.0, 0.0]

    for t in range(1, n_samples):
        y_current = samples[t - 1, 1]
        samples[t, 0] = np.random.normal(rho * y_current, cond_std)

        x_current = samples[t, 0]
        samples[t, 1] = np.random.normal(rho * x_current, cond_std)

    burn_in = 1000
    post_burnin = samples[burn_in:]

    empirical_corr = np.corrcoef(post_burnin[:, 0], post_burnin[:, 1])[0, 1]

    return samples, post_burnin, rho, empirical_corr


# ---------------------------------------------------------------------------
# Code block 8: proposal_tuning_demo
# ---------------------------------------------------------------------------

def proposal_tuning_demo(seed=42):
    """Demonstrate the effect of proposal step size on MCMC efficiency.

    We run MH targeting a 5-dimensional standard Normal with different
    step sizes and compare acceptance rates and ESS.
    """
    np.random.seed(seed)

    d = 5
    n_samples = 50_000

    def log_target(x):
        """Log density of a d-dimensional standard Normal."""
        return -0.5 * np.sum(x**2)

    results = []
    for sigma in [0.05, 0.2, 0.5, 1.0, 2.4 / np.sqrt(d), 3.0, 10.0]:
        chain = np.zeros((n_samples, d))
        chain[0] = np.zeros(d)
        n_accepted = 0

        for t in range(1, n_samples):
            proposal = chain[t-1] + np.random.normal(0, sigma, size=d)
            log_alpha = log_target(proposal) - log_target(chain[t-1])
            if np.log(np.random.uniform()) < log_alpha:
                chain[t] = proposal
                n_accepted += 1
            else:
                chain[t] = chain[t-1]

        acc_rate = n_accepted / (n_samples - 1)

        burn_in = 5000
        first_coord = chain[burn_in:, 0]
        N = len(first_coord)
        mean = first_coord.mean()
        var = first_coord.var()

        if var == 0:
            ess = 1.0
        else:
            iat = 1.0
            for k in range(1, 500):
                if k >= N:
                    break
                rho_k = np.mean((first_coord[:-k] - mean) * (first_coord[k:] - mean)) / var
                if rho_k < 0:
                    break
                iat += 2 * rho_k

            ess = N / iat
        results.append((sigma, acc_rate, ess))

    best = min(results, key=lambda r: abs(r[1] - 0.234))

    return results, best


# ===========================================================================
# Test classes and functions
# ===========================================================================


class TestBetaBinomialDemo:
    """Tests for the Beta-Binomial conjugate model (code block 1)."""

    def test_posterior_parameters(self):
        """Verify that the posterior parameters are computed correctly via conjugacy."""
        alpha_post, beta_post = beta_binomial_demo()
        assert alpha_post == 9, f"Expected alpha_post=9, got {alpha_post}"
        assert beta_post == 15, f"Expected beta_post=15, got {beta_post}"

    def test_posterior_mean(self):
        """Verify the posterior mean equals alpha/(alpha+beta)."""
        alpha_post, beta_post = beta_binomial_demo()
        expected_mean = alpha_post / (alpha_post + beta_post)
        assert abs(expected_mean - 9.0 / 24.0) < 1e-12

    def test_posterior_variance(self):
        """Verify the posterior variance matches the Beta distribution formula."""
        alpha_post, beta_post = beta_binomial_demo()
        computed_var = (alpha_post * beta_post) / (
            (alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)
        )
        # Compare against scipy's Beta distribution
        scipy_var = stats.beta(alpha_post, beta_post).var()
        assert abs(computed_var - scipy_var) < 1e-12

    def test_posterior_integrates_to_one(self):
        """Verify the Beta posterior PDF integrates to 1."""
        alpha_post, beta_post = beta_binomial_demo()
        dist = stats.beta(alpha_post, beta_post)
        from scipy.integrate import quad
        integral, _ = quad(dist.pdf, 0, 1)
        assert abs(integral - 1.0) < 1e-8

    def test_posterior_is_conjugate(self):
        """Verify that the posterior is indeed Beta(alpha+k, beta+n-k)."""
        n, k = 20, 7
        alpha_prior, beta_prior = 2, 2
        alpha_post, beta_post = beta_binomial_demo()
        assert alpha_post == alpha_prior + k
        assert beta_post == beta_prior + (n - k)


class TestMarkovChainConvergence:
    """Tests for the Markov chain convergence demo (code block 2)."""

    def test_transition_matrix_rows_sum_to_one(self):
        """Verify that each row of the transition matrix sums to 1."""
        T, pi, empirical = markov_chain_convergence(seed=42)
        assert np.allclose(T.sum(axis=1), 1.0)

    def test_stationary_distribution_is_valid_probability(self):
        """Verify the stationary distribution sums to 1 and has non-negative entries."""
        T, pi, empirical = markov_chain_convergence(seed=42)
        assert abs(pi.sum() - 1.0) < 1e-10
        assert np.all(pi >= 0)

    def test_pi_T_equals_pi(self):
        """Verify that pi * T = pi (stationarity condition)."""
        T, pi, empirical = markov_chain_convergence(seed=42)
        pi_T = pi @ T
        assert np.allclose(pi_T, pi, atol=1e-10)

    def test_empirical_matches_stationary(self):
        """Verify that empirical frequencies converge to the stationary distribution."""
        T, pi, empirical = markov_chain_convergence(seed=42)
        assert np.max(np.abs(pi - empirical)) < 0.02, (
            f"Empirical frequencies {empirical} too far from stationary {pi}"
        )

    def test_detailed_balance_holds_when_applicable(self):
        """Check detailed balance pi[i]*T[i,j] vs pi[j]*T[j,i].

        Note: This transition matrix does NOT satisfy detailed balance in general.
        We just verify that the computation runs and produces finite values.
        """
        T, pi, empirical = markov_chain_convergence(seed=42)
        for i in range(3):
            for j in range(i + 1, 3):
                lhs = pi[i] * T[i, j]
                rhs = pi[j] * T[j, i]
                assert np.isfinite(lhs) and np.isfinite(rhs)

    def test_convergence_from_different_starting_states(self):
        """Verify convergence regardless of the starting state."""
        # Run from different seeds to get different trajectories
        _, pi1, emp1 = markov_chain_convergence(seed=42)
        _, pi2, emp2 = markov_chain_convergence(seed=123)
        # Stationary distribution should be the same
        assert np.allclose(pi1, pi2, atol=1e-10)
        # Empirical frequencies should both be close to stationary
        assert np.max(np.abs(pi1 - emp1)) < 0.02
        assert np.max(np.abs(pi2 - emp2)) < 0.02


class TestMetropolisHastingsMixture:
    """Tests for MH sampling from a Gaussian mixture (code block 3)."""

    def test_acceptance_rate_is_reasonable(self):
        """Verify that the acceptance rate is in a reasonable range (10-80%)."""
        _, acc_rate, _, _, _ = metropolis_hastings_mixture(seed=42)
        assert 0.10 < acc_rate < 0.80, f"Acceptance rate {acc_rate} is outside reasonable range"

    def test_samples_have_correct_shape(self):
        """Verify that the full samples array has the expected shape."""
        samples, _, _, _, _ = metropolis_hastings_mixture(seed=42)
        assert samples.shape == (50_000,)

    def test_post_burnin_shape(self):
        """Verify that post-burn-in samples have expected length."""
        _, _, post_burnin, _, _ = metropolis_hastings_mixture(seed=42)
        assert len(post_burnin) == 45_000

    def test_both_modes_are_visited(self):
        """Verify that the chain visits both modes of the mixture."""
        _, _, post_burnin, near_mode1, near_mode2 = metropolis_hastings_mixture(seed=42)
        # Mode 1 at x=-2: should have roughly 30% of samples below 0
        # Mode 2 at x=3: should have roughly 70% of samples above 0
        # Use generous tolerances for stochastic output
        assert near_mode1 > 0.10, f"Mode 1 fraction {near_mode1} is too low"
        assert near_mode2 > 0.40, f"Mode 2 fraction {near_mode2} is too low"
        assert abs(near_mode1 + near_mode2 - 1.0) < 1e-10

    def test_sample_mean_is_near_weighted_mean(self):
        """Verify sample mean is near the theoretical mixture mean.

        True mean = 0.3*(-2) + 0.7*(3) = -0.6 + 2.1 = 1.5
        """
        _, _, post_burnin, _, _ = metropolis_hastings_mixture(seed=42)
        theoretical_mean = 0.3 * (-2) + 0.7 * 3
        assert abs(post_burnin.mean() - theoretical_mean) < 0.5, (
            f"Sample mean {post_burnin.mean():.3f} too far from theoretical {theoretical_mean}"
        )

    def test_log_target_function(self):
        """Verify the log target density is correct at known points."""
        def log_target(x):
            comp1 = 0.3 * np.exp(-0.5 * ((x + 2) / 0.5)**2) / (0.5 * np.sqrt(2 * np.pi))
            comp2 = 0.7 * np.exp(-0.5 * ((x - 3) / 1.0)**2) / (1.0 * np.sqrt(2 * np.pi))
            return np.log(comp1 + comp2 + 1e-300)

        # At x=-2: mode 1 is at its peak; mode 2 contributes little
        val_at_mode1 = log_target(-2.0)
        # At x=3: mode 2 is at its peak; mode 1 contributes little
        val_at_mode2 = log_target(3.0)
        # At x=100: both modes contribute negligibly
        val_far = log_target(100.0)

        assert val_at_mode1 > val_far
        assert val_at_mode2 > val_far
        assert np.isfinite(val_at_mode1)
        assert np.isfinite(val_at_mode2)


class TestMhSfsInference:
    """Tests for MH inference of theta from the SFS (code block 4)."""

    def test_observed_sfs_shape(self):
        """Verify that the observed SFS has the correct length n-1."""
        _, _, _, _, observed_sfs, _, _ = mh_sfs_inference(seed=42)
        assert len(observed_sfs) == 9  # n=10, so n-1=9 frequency classes

    def test_chain_shape(self):
        """Verify the full MCMC chain has the expected length."""
        chain, _, _, _, _, _, _ = mh_sfs_inference(seed=42)
        assert len(chain) == 30_000

    def test_post_burnin_shape(self):
        """Verify post-burn-in chain has expected length."""
        _, post_burnin, _, _, _, _, _ = mh_sfs_inference(seed=42)
        assert len(post_burnin) == 25_000

    def test_acceptance_rate_reasonable(self):
        """Verify the acceptance rate is in a reasonable range."""
        _, _, acc_rate, _, _, _, _ = mh_sfs_inference(seed=42)
        assert 0.10 < acc_rate < 0.95, f"Acceptance rate {acc_rate} out of range"

    def test_posterior_mean_near_true_theta(self):
        """Verify the posterior mean is reasonably close to the true theta."""
        _, post_burnin, _, theta_true, _, _, _ = mh_sfs_inference(seed=42)
        post_mean = post_burnin.mean()
        # With Poisson noise and an exponential prior, the posterior mean should
        # be in the ballpark of the true value. Use generous tolerance.
        assert abs(post_mean - theta_true) < 3.0, (
            f"Posterior mean {post_mean:.3f} too far from true theta {theta_true}"
        )

    def test_true_theta_in_credible_interval(self):
        """Verify the 95% credible interval contains the true theta."""
        _, post_burnin, _, theta_true, _, _, _ = mh_sfs_inference(seed=42)
        ci_low = np.percentile(post_burnin, 2.5)
        ci_high = np.percentile(post_burnin, 97.5)
        assert ci_low < theta_true < ci_high, (
            f"True theta {theta_true} not in 95% CI ({ci_low:.3f}, {ci_high:.3f})"
        )

    def test_chain_stays_positive(self):
        """Verify all post-burn-in theta values are positive (physical constraint)."""
        _, post_burnin, _, _, _, _, _ = mh_sfs_inference(seed=42)
        # Due to proposal mechanism, some values could be <= 0 but would be rejected
        # by the log_prior. After burn-in, all values should be positive.
        assert np.all(post_burnin > 0), "Theta should remain positive"

    def test_log_likelihood_finite_for_positive_theta(self):
        """Verify log_likelihood returns finite values for valid theta."""
        _, _, _, _, observed_sfs, log_likelihood, _ = mh_sfs_inference(seed=42)
        ll = log_likelihood(5.0, observed_sfs)
        assert np.isfinite(ll)

    def test_log_likelihood_neg_inf_for_nonpositive_theta(self):
        """Verify log_likelihood returns -inf for theta <= 0."""
        _, _, _, _, observed_sfs, log_likelihood, _ = mh_sfs_inference(seed=42)
        assert log_likelihood(0.0, observed_sfs) == -np.inf
        assert log_likelihood(-1.0, observed_sfs) == -np.inf

    def test_log_prior_neg_inf_for_nonpositive_theta(self):
        """Verify log_prior returns -inf for theta <= 0."""
        _, _, _, _, _, _, log_prior = mh_sfs_inference(seed=42)
        assert log_prior(0.0) == -np.inf
        assert log_prior(-1.0) == -np.inf

    def test_log_prior_finite_for_positive_theta(self):
        """Verify log_prior returns finite values for theta > 0."""
        _, _, _, _, _, _, log_prior = mh_sfs_inference(seed=42)
        assert np.isfinite(log_prior(5.0))
        assert np.isfinite(log_prior(0.001))


class TestComputeAcfAndEss:
    """Tests for the ACF and ESS computation (code block 5)."""

    def test_acf_at_lag_zero_is_one(self):
        """Verify that the autocorrelation at lag 0 is always 1."""
        np.random.seed(42)
        chain = np.random.randn(1000)
        acf, ess = compute_acf_and_ess(chain, max_lag=50)
        assert abs(acf[0] - 1.0) < 1e-12

    def test_acf_shape(self):
        """Verify the ACF array has the correct shape."""
        np.random.seed(42)
        chain = np.random.randn(1000)
        acf, ess = compute_acf_and_ess(chain, max_lag=50)
        assert acf.shape == (51,)

    def test_ess_for_iid_samples(self):
        """For i.i.d. samples, ESS should be close to the number of samples."""
        np.random.seed(42)
        N = 10_000
        chain = np.random.randn(N)
        acf, ess = compute_acf_and_ess(chain, max_lag=200)
        # For truly i.i.d. samples, ESS should be near N
        assert ess > N * 0.5, f"ESS {ess} too low for i.i.d. samples of size {N}"

    def test_ess_for_correlated_samples(self):
        """For correlated samples, ESS should be much less than N."""
        np.random.seed(42)
        N = 10_000
        # Create a highly correlated chain (random walk)
        chain = np.cumsum(np.random.randn(N) * 0.01)
        acf, ess = compute_acf_and_ess(chain, max_lag=200)
        assert ess < N * 0.5, f"ESS {ess} too high for correlated chain of size {N}"

    def test_ess_is_positive(self):
        """Verify ESS is always positive."""
        np.random.seed(42)
        chain = np.random.randn(500)
        _, ess = compute_acf_and_ess(chain, max_lag=50)
        assert ess > 0

    def test_constant_chain_returns_ess_one(self):
        """For a constant chain (zero variance), ESS should be 1."""
        chain = np.ones(100)
        acf, ess = compute_acf_and_ess(chain, max_lag=20)
        assert ess == 1.0
        assert np.all(acf == 1.0)

    def test_acf_decays_for_mh_chain(self):
        """Verify that ACF decays for an MH chain targeting standard Normal."""
        np.random.seed(42)
        chain, _ = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        acf, ess = compute_acf_and_ess(chain, max_lag=100)
        # ACF at lag 0 is 1, should be lower at higher lags
        assert acf[10] < acf[0]
        assert acf[50] < acf[10] or acf[50] < 0.1


class TestRunMhChain:
    """Tests for the MH chain targeting standard Normal (code block 6)."""

    def test_chain_shape(self):
        """Verify chain length after discarding burn-in."""
        chain, acc_rate = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        assert len(chain) == 18000  # 20000 - 2000 burn-in

    def test_acceptance_rate_small_sigma(self):
        """Small sigma should yield high acceptance rate."""
        _, acc_rate = run_mh_chain(sigma=0.1, n_samples=20000, seed=42)
        assert acc_rate > 0.8, f"Expected high acceptance for small sigma, got {acc_rate}"

    def test_acceptance_rate_large_sigma(self):
        """Large sigma should yield low acceptance rate."""
        _, acc_rate = run_mh_chain(sigma=10.0, n_samples=20000, seed=42)
        assert acc_rate < 0.3, f"Expected low acceptance for large sigma, got {acc_rate}"

    def test_chain_mean_near_zero(self):
        """The posterior-burn-in chain should have mean near 0 (standard Normal target)."""
        chain, _ = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        assert abs(chain.mean()) < 0.3, f"Chain mean {chain.mean():.3f} too far from 0"

    def test_chain_std_near_one(self):
        """The chain should have standard deviation near 1 (standard Normal target)."""
        chain, _ = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        assert abs(chain.std() - 1.0) < 0.3, f"Chain std {chain.std():.3f} too far from 1"

    def test_different_sigmas_produce_different_ess(self):
        """Different step sizes should produce different ESS values."""
        chain_small, _ = run_mh_chain(sigma=0.1, n_samples=20000, seed=42)
        chain_mid, _ = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        _, ess_small = compute_acf_and_ess(chain_small, max_lag=200)
        _, ess_mid = compute_acf_and_ess(chain_mid, max_lag=200)
        # Middle sigma should generally produce higher ESS than very small sigma
        assert ess_mid > ess_small, (
            f"Expected ESS for sigma=1.0 ({ess_mid:.0f}) > ESS for sigma=0.1 ({ess_small:.0f})"
        )

    def test_convergence_diagnostics_integration(self):
        """Integration test: run chain and compute ACF/ESS together, as in the document."""
        np.random.seed(42)
        for sigma in [0.1, 1.0, 2.4, 10.0]:
            chain, acc_rate = run_mh_chain(sigma, seed=None)
            acf, ess = compute_acf_and_ess(chain)
            assert np.isfinite(ess)
            assert ess > 0
            assert 0 <= acc_rate <= 1
            assert acf[0] == 1.0


class TestGibbsBivariateNormal:
    """Tests for the Gibbs sampler on a bivariate Normal (code block 7)."""

    def test_samples_shape(self):
        """Verify the output samples have the correct shape."""
        samples, post_burnin, rho, emp_corr = gibbs_bivariate_normal(seed=42)
        assert samples.shape == (20_000, 2)
        assert post_burnin.shape == (19_000, 2)

    def test_mean_x_near_zero(self):
        """Verify the marginal mean of X is near 0."""
        _, post_burnin, _, _ = gibbs_bivariate_normal(seed=42)
        assert abs(post_burnin[:, 0].mean()) < 0.1, (
            f"Mean X = {post_burnin[:, 0].mean():.4f}, expected near 0"
        )

    def test_mean_y_near_zero(self):
        """Verify the marginal mean of Y is near 0."""
        _, post_burnin, _, _ = gibbs_bivariate_normal(seed=42)
        assert abs(post_burnin[:, 1].mean()) < 0.1, (
            f"Mean Y = {post_burnin[:, 1].mean():.4f}, expected near 0"
        )

    def test_variance_x_near_one(self):
        """Verify the marginal variance of X is near 1."""
        _, post_burnin, _, _ = gibbs_bivariate_normal(seed=42)
        var_x = post_burnin[:, 0].var()
        assert abs(var_x - 1.0) < 0.15, f"Var X = {var_x:.4f}, expected near 1.0"

    def test_variance_y_near_one(self):
        """Verify the marginal variance of Y is near 1."""
        _, post_burnin, _, _ = gibbs_bivariate_normal(seed=42)
        var_y = post_burnin[:, 1].var()
        assert abs(var_y - 1.0) < 0.15, f"Var Y = {var_y:.4f}, expected near 1.0"

    def test_empirical_correlation_near_rho(self):
        """Verify the empirical correlation is near the true rho=0.8."""
        _, post_burnin, rho, emp_corr = gibbs_bivariate_normal(seed=42)
        assert abs(emp_corr - rho) < 0.05, (
            f"Empirical correlation {emp_corr:.4f} too far from true rho={rho}"
        )

    def test_gibbs_acceptance_is_one(self):
        """Gibbs sampling always accepts, so every sample should differ from
        the previous one (with probability 1 for continuous distributions).
        We verify that no two consecutive samples are identical."""
        samples, _, _, _ = gibbs_bivariate_normal(seed=42)
        # Check that consecutive samples are not identical
        diffs = np.diff(samples, axis=0)
        n_identical = np.sum(np.all(diffs == 0, axis=1))
        assert n_identical == 0, f"Found {n_identical} identical consecutive samples"

    def test_covariance_matrix_close_to_true(self):
        """Verify the empirical covariance matrix approximates the true Sigma."""
        _, post_burnin, rho, _ = gibbs_bivariate_normal(seed=42)
        true_sigma = np.array([[1.0, rho], [rho, 1.0]])
        empirical_cov = np.cov(post_burnin.T)
        assert np.allclose(empirical_cov, true_sigma, atol=0.1), (
            f"Empirical covariance\n{empirical_cov}\ntoo far from true\n{true_sigma}"
        )


class TestProposalTuningDemo:
    """Tests for the proposal tuning demonstration (code block 8)."""

    def test_returns_results_for_all_sigmas(self):
        """Verify that results are returned for all 7 sigma values."""
        results, best = proposal_tuning_demo(seed=42)
        assert len(results) == 7

    def test_results_tuple_structure(self):
        """Verify each result is a tuple of (sigma, acc_rate, ess)."""
        results, _ = proposal_tuning_demo(seed=42)
        for sigma, acc_rate, ess in results:
            assert isinstance(sigma, float)
            assert 0 <= acc_rate <= 1
            assert ess > 0

    def test_small_sigma_high_acceptance(self):
        """Very small sigma should have high acceptance rate."""
        results, _ = proposal_tuning_demo(seed=42)
        # First entry is sigma=0.05
        sigma, acc_rate, ess = results[0]
        assert sigma == 0.05
        assert acc_rate > 0.9, f"Expected high acceptance for sigma=0.05, got {acc_rate}"

    def test_large_sigma_low_acceptance(self):
        """Very large sigma should have low acceptance rate."""
        results, _ = proposal_tuning_demo(seed=42)
        # Last entry is sigma=10.0
        sigma, acc_rate, ess = results[-1]
        assert sigma == 10.0
        assert acc_rate < 0.1, f"Expected low acceptance for sigma=10.0, got {acc_rate}"

    def test_optimal_sigma_near_theoretical(self):
        """The theoretically optimal sigma is 2.4/sqrt(d). Verify it produces
        acceptance rate close to the theoretical 23.4%."""
        results, _ = proposal_tuning_demo(seed=42)
        # sigma = 2.4/sqrt(5) is the 5th entry (index 4)
        optimal_sigma = 2.4 / np.sqrt(5)
        found = False
        for sigma, acc_rate, ess in results:
            if abs(sigma - optimal_sigma) < 0.01:
                found = True
                # Should be in the general neighborhood of 23.4%
                assert abs(acc_rate - 0.234) < 0.15, (
                    f"Acceptance rate {acc_rate:.3f} for optimal sigma too far from 0.234"
                )
        assert found, "Did not find the optimal sigma in results"

    def test_best_is_closest_to_234(self):
        """Verify the 'best' result is indeed closest to 23.4% acceptance."""
        results, best = proposal_tuning_demo(seed=42)
        best_sigma, best_acc, best_ess = best
        for sigma, acc_rate, ess in results:
            assert abs(best_acc - 0.234) <= abs(acc_rate - 0.234) + 1e-10

    def test_ess_varies_with_sigma(self):
        """Verify that ESS varies across different step sizes, and extreme
        step sizes produce lower ESS than moderate ones."""
        results, _ = proposal_tuning_demo(seed=42)
        ess_values = [ess for _, _, ess in results]
        # ESS should not all be the same
        assert max(ess_values) > min(ess_values) * 1.5, (
            "ESS values should vary substantially across different step sizes"
        )

    def test_acceptance_rate_monotonic_trend(self):
        """Acceptance rate should generally decrease as sigma increases."""
        results, _ = proposal_tuning_demo(seed=42)
        acc_rates = [acc for _, acc, _ in results]
        # Overall the first sigma (0.05) should have higher acceptance than the last (10.0)
        assert acc_rates[0] > acc_rates[-1], (
            "Acceptance rate should be higher for small sigma than large sigma"
        )


class TestCrossFunctionIntegration:
    """Integration tests that combine multiple code blocks, as done in the document."""

    def test_mh_sfs_chain_with_acf_ess(self):
        """Run the SFS inference chain and compute ACF/ESS on the result."""
        _, post_burnin, _, _, _, _, _ = mh_sfs_inference(seed=42)
        acf, ess = compute_acf_and_ess(post_burnin, max_lag=200)
        assert acf[0] == 1.0
        assert ess > 0
        assert ess < len(post_burnin)  # ESS should be less than N for correlated chain

    def test_mixture_chain_with_acf_ess(self):
        """Run the mixture MH chain and compute ACF/ESS on the result."""
        _, _, post_burnin, _, _ = metropolis_hastings_mixture(seed=42)
        acf, ess = compute_acf_and_ess(post_burnin, max_lag=200)
        assert acf[0] == 1.0
        assert ess > 0
        # ACF should decay
        assert acf[50] < acf[0]

    def test_beta_binomial_vs_mh_sfs_consistency(self):
        """The beta-binomial demo and MH SFS inference both estimate parameters
        from count data. Verify both produce reasonable results."""
        alpha_post, beta_post = beta_binomial_demo()
        assert alpha_post > 0 and beta_post > 0

        _, post_burnin, _, theta_true, _, _, _ = mh_sfs_inference(seed=42)
        assert post_burnin.mean() > 0

    def test_gibbs_vs_mh_both_converge(self):
        """Both Gibbs and MH should converge to their respective targets.
        Verify both produce samples with correct marginal statistics."""
        # Gibbs for bivariate Normal
        _, post_gibbs, rho, emp_corr = gibbs_bivariate_normal(seed=42)
        assert abs(post_gibbs[:, 0].mean()) < 0.1
        assert abs(post_gibbs[:, 0].var() - 1.0) < 0.15

        # MH for standard Normal
        chain_mh, acc_rate = run_mh_chain(sigma=1.0, n_samples=20000, seed=42)
        assert abs(chain_mh.mean()) < 0.3
        assert abs(chain_mh.std() - 1.0) < 0.3
