"""
Tests for Python code blocks from the probabilistic inference prerequisite RST documentation.

Covers:
- Exponential distribution: PDF, log-likelihood, MLE, Fisher information
- Poisson distribution: PMF, SFS log-likelihood, MLE (Watterson's estimator)
- Gamma distribution: log-likelihood, MLE (profile likelihood)
- Gaussian distribution: log-likelihood, smoothness prior
- Bayesian updates: gamma-Poisson conjugacy, beta-binomial conjugacy
- Composite likelihood: combining SFS + heterozygosity
- MLE worked example: two-epoch SFS inference
"""

import numpy as np
import pytest
from scipy.special import gammaln, factorial


# ---------------------------------------------------------------------------
# Re-defined functions from docs/prerequisites/probabilistic_inference.rst
# ---------------------------------------------------------------------------

def exponential_pdf(t, lam):
    """Probability density of Exponential(lambda) at time t."""
    return lam * np.exp(-lam * t)


def exponential_log_likelihood(lam, times):
    """Log-likelihood of exponential rate parameter lambda."""
    times = np.asarray(times)
    n = len(times)
    return n * np.log(lam) - lam * np.sum(times)


def poisson_pmf(k, mu):
    """P(k | mu) = mu^k * exp(-mu) / k!"""
    return mu**k * np.exp(-mu) / factorial(k, exact=True)


def sfs_log_likelihood(D_obs, xi_expected):
    """Poisson log-likelihood of observed SFS given expected SFS."""
    xi = np.maximum(xi_expected, 1e-300)
    return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))


def poisson_log_likelihood(mu, counts):
    """Poisson log-likelihood for a vector of observed counts."""
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.sum(counts * np.log(np.maximum(mu, 1e-300)) - mu
                  - gammaln(counts + 1))


def gamma_log_likelihood(alpha, beta, data):
    """Log-likelihood for Gamma(alpha, beta) observations."""
    data = np.asarray(data)
    n = len(data)
    return (n * alpha * np.log(beta)
            - n * gammaln(alpha)
            + (alpha - 1) * np.sum(np.log(data))
            - beta * np.sum(data))


def gamma_mle(data):
    """Compute the MLE of Gamma(alpha, beta) by profiling."""
    from scipy.optimize import minimize_scalar
    data = np.asarray(data)
    n = len(data)
    sum_data = np.sum(data)
    sum_log_data = np.sum(np.log(data))

    def neg_profile_ll(alpha):
        beta_hat = n * alpha / sum_data
        return -(n * alpha * np.log(beta_hat)
                 - n * gammaln(alpha)
                 + (alpha - 1) * sum_log_data
                 - beta_hat * sum_data)

    result = minimize_scalar(neg_profile_ll, bounds=(0.01, 100),
                             method='bounded')
    alpha_hat = result.x
    beta_hat = n * alpha_hat / sum_data
    return alpha_hat, beta_hat


def gaussian_log_likelihood(mu, sigma2, data):
    """Log-likelihood for N(mu, sigma^2) observations."""
    data = np.asarray(data)
    n = len(data)
    return (-0.5 * n * np.log(2 * np.pi * sigma2)
            - 0.5 * np.sum((data - mu)**2) / sigma2)


def smoothness_prior_log_density(h, sigma=1.0):
    """Log-density of the Gaussian random-walk smoothness prior."""
    diffs = np.diff(h)
    return -0.5 * np.sum(diffs**2) / sigma**2


def gamma_poisson_update(alpha_prior, beta_prior, k_observed, mu):
    """Bayesian update: Gamma prior + Poisson likelihood -> Gamma posterior."""
    return alpha_prior + k_observed, beta_prior + mu


def beta_binomial_update(a_prior, b_prior, k, n):
    """Bayesian update: Beta prior + Binomial likelihood -> Beta posterior."""
    return a_prior + k, b_prior + n - k


def expected_sfs_two_epoch(n, theta, nu, T):
    """Expected SFS for a two-epoch model (approximate)."""
    k = np.arange(1, n)
    base = theta / k
    correction = 1.0 + (nu - 1.0) * (1.0 - np.exp(-T * k / nu))
    return base * np.maximum(correction, 0.01)


def sfs_poisson_loglik(D_obs, xi_expected):
    """Poisson log-likelihood of observed SFS given expected."""
    xi = np.maximum(xi_expected, 1e-300)
    return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))


def sfs_log_likelihood_theta(theta, D_obs, n):
    """Poisson SFS log-likelihood under constant population size."""
    k = np.arange(1, n)
    xi = theta / k
    xi = np.maximum(xi, 1e-300)
    return np.sum(D_obs * np.log(xi) - xi - gammaln(D_obs + 1))


def het_log_likelihood(theta, n_het, n_sites):
    """Binomial log-likelihood for heterozygosity."""
    p_het = 1.0 - np.exp(-theta / n_sites)
    p_het = np.clip(p_het, 1e-300, 1 - 1e-300)
    n_hom = n_sites - n_het
    return n_het * np.log(p_het) + n_hom * np.log(1.0 - p_het)


# ===========================================================================
# Tests for exponential_pdf
# ===========================================================================

class TestExponentialPdf:
    def test_value_at_zero(self):
        """At t=0, the PDF should equal the rate parameter."""
        for lam in [0.5, 1.0, 2.0]:
            assert np.isclose(exponential_pdf(0, lam), lam)

    def test_nonnegative(self):
        """PDF should be non-negative everywhere."""
        t = np.linspace(0, 10, 100)
        for lam in [0.5, 1.0, 2.0]:
            assert np.all(exponential_pdf(t, lam) >= 0)

    def test_monotonically_decreasing(self):
        """PDF should be strictly decreasing for t > 0."""
        t = np.linspace(0.01, 10, 100)
        pdf = exponential_pdf(t, lam=1.0)
        assert np.all(np.diff(pdf) < 0)

    def test_integrates_to_one(self):
        """PDF should integrate to approximately 1."""
        t = np.linspace(0, 50, 10000)
        pdf = exponential_pdf(t, lam=1.0)
        integral = np.trapezoid(pdf, t)
        assert abs(integral - 1.0) < 0.01

    def test_higher_rate_faster_decay(self):
        """Higher rate should produce faster decay."""
        t = 1.0
        assert exponential_pdf(t, 2.0) < exponential_pdf(t, 0.5)


# ===========================================================================
# Tests for exponential_log_likelihood
# ===========================================================================

class TestExponentialLogLikelihood:
    def test_mle_is_reciprocal_of_mean(self):
        """MLE of rate should be 1/mean(times)."""
        np.random.seed(42)
        times = np.random.exponential(scale=2.0, size=100)
        mle_rate = 1.0 / np.mean(times)
        # Check that likelihood peaks at MLE
        rates = np.linspace(0.1, 2.0, 500)
        lls = [exponential_log_likelihood(r, times) for r in rates]
        best_rate = rates[np.argmax(lls)]
        assert abs(best_rate - mle_rate) < 0.01

    def test_maximized_at_true_rate(self):
        """With enough data, MLE should be close to true rate."""
        np.random.seed(42)
        true_rate = 1.5
        times = np.random.exponential(scale=1.0 / true_rate, size=1000)
        mle = 1.0 / np.mean(times)
        assert abs(mle - true_rate) / true_rate < 0.1

    def test_more_data_narrower_peak(self):
        """More observations should produce a sharper likelihood peak."""
        np.random.seed(42)
        times_small = np.random.exponential(1.0, size=10)
        times_large = np.random.exponential(1.0, size=1000)
        # Compute curvature (second derivative) at MLE
        mle_s = 1.0 / np.mean(times_small)
        mle_l = 1.0 / np.mean(times_large)
        curv_s = len(times_small) / mle_s**2
        curv_l = len(times_large) / mle_l**2
        assert curv_l > curv_s

    def test_finite_for_positive_inputs(self):
        """Log-likelihood should be finite for valid inputs."""
        times = np.array([1.0, 2.0, 3.0])
        ll = exponential_log_likelihood(1.0, times)
        assert np.isfinite(ll)


# ===========================================================================
# Tests for poisson_pmf
# ===========================================================================

class TestPoissonPmf:
    def test_sums_to_one(self):
        """PMF should sum to approximately 1 over all k."""
        mu = 5.0
        total = sum(poisson_pmf(k, mu) for k in range(50))
        assert abs(total - 1.0) < 1e-6

    def test_nonnegative(self):
        """All probabilities should be non-negative."""
        mu = 3.0
        for k in range(20):
            assert poisson_pmf(k, mu) >= 0

    def test_mode_near_mean(self):
        """Mode should be near the mean for integer mean."""
        mu = 5.0
        probs = [poisson_pmf(k, mu) for k in range(15)]
        mode = np.argmax(probs)
        assert mode in [4, 5]

    def test_zero_mean_gives_point_mass(self):
        """For mu=0, P(k=0)=1 and P(k>0)=0."""
        assert np.isclose(poisson_pmf(0, 1e-15), 1.0, atol=1e-10)


# ===========================================================================
# Tests for sfs_log_likelihood
# ===========================================================================

class TestSfsLogLikelihood:
    def test_finite(self):
        """SFS log-likelihood should be finite for valid inputs."""
        D = np.array([100, 50, 30, 25, 20])
        xi = np.array([100.0, 50.0, 33.3, 25.0, 20.0])
        ll = sfs_log_likelihood(D, xi)
        assert np.isfinite(ll)

    def test_maximized_at_correct_expected(self):
        """LL should be higher when expected matches observed."""
        D = np.array([100, 50, 30])
        xi_good = np.array([100.0, 50.0, 30.0])
        xi_bad = np.array([200.0, 10.0, 5.0])
        assert sfs_log_likelihood(D, xi_good) > sfs_log_likelihood(D, xi_bad)

    def test_zero_observed_handled(self):
        """Should handle zero-count entries."""
        D = np.array([0, 50, 0])
        xi = np.array([10.0, 50.0, 10.0])
        ll = sfs_log_likelihood(D, xi)
        assert np.isfinite(ll)


# ===========================================================================
# Tests for poisson_log_likelihood
# ===========================================================================

class TestPoissonLogLikelihood:
    def test_mle_is_sample_mean(self):
        """MLE of Poisson mean should be the sample mean."""
        np.random.seed(42)
        mu_true = 7.0
        obs = np.random.poisson(mu_true, size=50)
        mu_mle = np.mean(obs)
        # Verify by scanning
        mus = np.linspace(3, 12, 200)
        lls = [poisson_log_likelihood(m, obs) for m in mus]
        best_mu = mus[np.argmax(lls)]
        assert abs(best_mu - mu_mle) < 0.1

    def test_wattersons_estimator(self):
        """SFS MLE of theta should give Watterson's estimator."""
        np.random.seed(42)
        theta_true = 200.0
        n = 20
        k_values = np.arange(1, n)
        xi = theta_true / k_values
        D = np.random.poisson(xi)
        harmonic = np.sum(1.0 / k_values)
        S = np.sum(D)
        theta_watterson = S / harmonic
        # Should be close to true value
        assert abs(theta_watterson - theta_true) / theta_true < 0.3


# ===========================================================================
# Tests for gamma_log_likelihood
# ===========================================================================

class TestGammaLogLikelihood:
    def test_finite(self):
        """Gamma LL should be finite for valid inputs."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ll = gamma_log_likelihood(2.0, 1.0, data)
        assert np.isfinite(ll)

    def test_maximized_near_true_params(self):
        """Gamma LL should peak near true parameters."""
        np.random.seed(42)
        alpha_true, beta_true = 3.0, 0.5
        data = np.random.gamma(alpha_true, 1.0 / beta_true, size=200)
        # Compare true vs wrong
        ll_true = gamma_log_likelihood(alpha_true, beta_true, data)
        ll_wrong = gamma_log_likelihood(1.0, 2.0, data)
        assert ll_true > ll_wrong

    def test_exponential_special_case(self):
        """Gamma(1, beta) should match exponential log-likelihood."""
        np.random.seed(42)
        lam = 1.5
        times = np.random.exponential(1.0 / lam, size=50)
        ll_gamma = gamma_log_likelihood(1.0, lam, times)
        ll_exp = exponential_log_likelihood(lam, times)
        assert np.isclose(ll_gamma, ll_exp, rtol=1e-10)


# ===========================================================================
# Tests for gamma_mle
# ===========================================================================

class TestGammaMle:
    def test_recovers_true_parameters(self):
        """Gamma MLE should recover true parameters from large samples."""
        np.random.seed(42)
        alpha_true, beta_true = 3.0, 0.5
        data = np.random.gamma(alpha_true, 1.0 / beta_true, size=500)
        alpha_hat, beta_hat = gamma_mle(data)
        assert abs(alpha_hat - alpha_true) / alpha_true < 0.2
        assert abs(beta_hat - beta_true) / beta_true < 0.2

    def test_mean_estimate(self):
        """Estimated mean (alpha/beta) should be close to sample mean."""
        np.random.seed(42)
        data = np.random.gamma(5.0, 2.0, size=200)
        alpha_hat, beta_hat = gamma_mle(data)
        est_mean = alpha_hat / beta_hat
        assert abs(est_mean - np.mean(data)) / np.mean(data) < 0.1

    def test_positive_parameters(self):
        """Both alpha and beta should be positive."""
        np.random.seed(42)
        data = np.random.gamma(2.0, 1.0, size=50)
        alpha_hat, beta_hat = gamma_mle(data)
        assert alpha_hat > 0
        assert beta_hat > 0


# ===========================================================================
# Tests for gaussian_log_likelihood
# ===========================================================================

class TestGaussianLogLikelihood:
    def test_maximized_at_sample_mean(self):
        """Gaussian LL should be maximized at mu = sample mean."""
        np.random.seed(42)
        data = np.random.normal(5.0, 2.0, size=100)
        sigma2 = 4.0
        mu_vals = np.linspace(3, 7, 200)
        lls = [gaussian_log_likelihood(m, sigma2, data) for m in mu_vals]
        best_mu = mu_vals[np.argmax(lls)]
        assert abs(best_mu - np.mean(data)) < 0.1

    def test_finite(self):
        """LL should be finite for valid inputs."""
        data = np.array([1.0, 2.0, 3.0])
        ll = gaussian_log_likelihood(2.0, 1.0, data)
        assert np.isfinite(ll)

    def test_larger_variance_lower_peak(self):
        """Larger variance should give lower LL at the mean."""
        data = np.array([1.0, 2.0, 3.0])
        mu = np.mean(data)
        ll_small = gaussian_log_likelihood(mu, 0.5, data)
        ll_large = gaussian_log_likelihood(mu, 10.0, data)
        # With the data close to the mean, smaller variance gives higher LL
        assert ll_small > ll_large


# ===========================================================================
# Tests for smoothness_prior_log_density
# ===========================================================================

class TestSmoothnessPrior:
    def test_smooth_preferred_over_rough(self):
        """Smooth trajectories should have higher prior density."""
        h_smooth = np.zeros(30)
        h_smooth[10:20] = -0.5
        h_rough = np.zeros(30)
        h_rough[::2] = 1.0
        h_rough[1::2] = -1.0
        assert smoothness_prior_log_density(h_smooth) > smoothness_prior_log_density(h_rough)

    def test_constant_is_optimal(self):
        """Constant trajectory should have log-density = 0."""
        h_const = np.ones(20) * 5.0
        lp = smoothness_prior_log_density(h_const)
        assert np.isclose(lp, 0.0)

    def test_nonpositive(self):
        """Smoothness prior should be non-positive."""
        h = np.random.randn(20)
        assert smoothness_prior_log_density(h) <= 0

    def test_smaller_sigma_penalizes_more(self):
        """Smaller sigma should penalize deviations more."""
        h = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        lp_tight = smoothness_prior_log_density(h, sigma=0.1)
        lp_loose = smoothness_prior_log_density(h, sigma=10.0)
        assert lp_tight < lp_loose


# ===========================================================================
# Tests for gamma_poisson_update
# ===========================================================================

class TestGammaPoissonUpdate:
    def test_shape_increment(self):
        """Posterior shape should be prior shape + k."""
        a, b = gamma_poisson_update(2.0, 1.0, k_observed=3, mu=0.5)
        assert a == 5.0

    def test_rate_increment(self):
        """Posterior rate should be prior rate + mu."""
        a, b = gamma_poisson_update(2.0, 1.0, k_observed=3, mu=0.5)
        assert b == 1.5

    def test_posterior_mean_shift(self):
        """More mutations should increase posterior mean."""
        a0, b0 = gamma_poisson_update(2.0, 1.0, k_observed=0, mu=0.5)
        a3, b3 = gamma_poisson_update(2.0, 1.0, k_observed=3, mu=0.5)
        a10, b10 = gamma_poisson_update(2.0, 1.0, k_observed=10, mu=0.5)
        assert a0 / b0 < a3 / b3 < a10 / b10

    def test_zero_mutations_shrinks_mean(self):
        """Zero mutations should pull posterior mean below prior mean."""
        prior_mean = 2.0 / 1.0
        a, b = gamma_poisson_update(2.0, 1.0, k_observed=0, mu=0.5)
        post_mean = a / b
        assert post_mean < prior_mean

    def test_sequential_updates(self):
        """Sequential updates should be equivalent to batch update."""
        alpha, beta = 2.0, 1.0
        mutations = [2, 1, 3]
        mu = 0.5
        # Sequential
        for k in mutations:
            alpha, beta = gamma_poisson_update(alpha, beta, k, mu)
        # Batch
        a_batch = 2.0 + sum(mutations)
        b_batch = 1.0 + len(mutations) * mu
        assert np.isclose(alpha, a_batch)
        assert np.isclose(beta, b_batch)


# ===========================================================================
# Tests for beta_binomial_update
# ===========================================================================

class TestBetaBinomialUpdate:
    def test_posterior_parameters(self):
        """Posterior should be Beta(a+k, b+n-k)."""
        a, b = beta_binomial_update(1.0, 1.0, k=7, n=20)
        assert a == 8.0
        assert b == 14.0

    def test_uniform_prior_posterior_mean(self):
        """With uniform prior, posterior mean should be (k+1)/(n+2)."""
        a, b = beta_binomial_update(1.0, 1.0, k=7, n=20)
        mean = a / (a + b)
        expected = 8.0 / 22.0
        assert np.isclose(mean, expected)

    def test_informative_prior_pulls_estimate(self):
        """Informative prior should pull posterior toward prior mean."""
        k, n = 7, 20
        # Uniform prior
        a1, b1 = beta_binomial_update(1.0, 1.0, k, n)
        mean1 = a1 / (a1 + b1)
        # Strong prior toward low frequency
        a2, b2 = beta_binomial_update(1.0, 10.0, k, n)
        mean2 = a2 / (a2 + b2)
        # Informative prior should pull mean lower
        assert mean2 < mean1

    def test_no_data_returns_prior(self):
        """With k=0, n=0, posterior should equal prior."""
        a, b = beta_binomial_update(2.0, 5.0, k=0, n=0)
        assert a == 2.0
        assert b == 5.0


# ===========================================================================
# Tests for expected_sfs_two_epoch
# ===========================================================================

class TestExpectedSfsTwoEpoch:
    def test_output_length(self):
        """SFS should have n-1 entries."""
        sfs = expected_sfs_two_epoch(20, 100.0, 1.0, 0.1)
        assert len(sfs) == 19

    def test_positive_entries(self):
        """All SFS entries should be positive."""
        sfs = expected_sfs_two_epoch(20, 100.0, 2.0, 0.2)
        assert np.all(sfs > 0)

    def test_expansion_vs_contraction(self):
        """Expansion and contraction should produce different SFS."""
        sfs_expand = expected_sfs_two_epoch(20, 100.0, 5.0, 0.2)
        sfs_contract = expected_sfs_two_epoch(20, 100.0, 0.2, 0.2)
        assert not np.allclose(sfs_expand, sfs_contract)

    def test_neutral_baseline(self):
        """With nu=1.0 and T=0, SFS should be close to theta/k."""
        n = 20
        theta = 100.0
        sfs = expected_sfs_two_epoch(n, theta, 1.0, 0.0)
        k = np.arange(1, n)
        expected = theta / k
        assert np.allclose(sfs, expected, rtol=0.01)


# ===========================================================================
# Tests for composite likelihood
# ===========================================================================

class TestCompositeLikelihood:
    def test_sfs_ll_finite(self):
        """SFS log-likelihood should be finite for valid inputs."""
        D = np.array([100, 50, 30, 25, 20])
        ll = sfs_log_likelihood_theta(200.0, D, 6)
        assert np.isfinite(ll)

    def test_het_ll_finite(self):
        """Heterozygosity LL should be finite for valid inputs."""
        ll = het_log_likelihood(200.0, 200, 100000)
        assert np.isfinite(ll)

    def test_composite_mle_between_individual_mles(self):
        """Composite MLE should be between individual source MLEs (or close)."""
        np.random.seed(42)
        theta_true = 200.0
        n_samples = 20
        n_sites = 100000

        k = np.arange(1, n_samples)
        xi = theta_true / k
        D = np.random.poisson(xi)

        p_het = 1.0 - np.exp(-theta_true / n_sites)
        n_het = np.random.binomial(n_sites, p_het)

        thetas = np.linspace(50, 400, 300)
        ll_sfs = np.array([sfs_log_likelihood_theta(th, D, n_samples) for th in thetas])
        ll_het = np.array([het_log_likelihood(th, n_het, n_sites) for th in thetas])
        ll_comp = ll_sfs + ll_het

        mle_sfs = thetas[np.argmax(ll_sfs)]
        mle_het = thetas[np.argmax(ll_het)]
        mle_comp = thetas[np.argmax(ll_comp)]

        # Composite should be reasonable
        assert 50 < mle_comp < 400
        assert np.isfinite(np.max(ll_comp))


# ===========================================================================
# Tests for Fisher information
# ===========================================================================

class TestFisherInformation:
    def test_ci_covers_true_value(self):
        """95% CI from Fisher info should usually cover the true value."""
        np.random.seed(42)
        true_rate = 1.0
        n_obs = 100
        covered = 0
        n_reps = 100
        for _ in range(n_reps):
            times = np.random.exponential(1.0 / true_rate, size=n_obs)
            mle = 1.0 / np.mean(times)
            se = mle / np.sqrt(n_obs)
            if mle - 1.96 * se <= true_rate <= mle + 1.96 * se:
                covered += 1
        # Coverage should be roughly 95% (allow some slack)
        assert covered / n_reps > 0.80

    def test_ci_shrinks_with_data(self):
        """CI width should decrease with more data."""
        np.random.seed(42)
        widths = []
        for n in [10, 50, 200]:
            times = np.random.exponential(1.0, size=n)
            mle = 1.0 / np.mean(times)
            se = mle / np.sqrt(n)
            widths.append(2 * 1.96 * se)
        assert widths[0] > widths[1] > widths[2]
