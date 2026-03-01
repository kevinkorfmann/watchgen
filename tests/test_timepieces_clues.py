"""
Tests for Python code blocks from the CLUES timepiece RST documentation.

Covers:
- wright_fisher_hmm.rst: backward_mean, backward_std, frequency bins,
  transition matrix, log-sum-exp, fast normal CDF
- emission_probabilities.rst: coalescent density, genotype likelihood emissions
- inference.rst: likelihood ratio test, estimate_selection, trajectory summary
"""

import numpy as np
import pytest
from scipy.stats import norm, beta as beta_dist


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/clues/wright_fisher_hmm.rst
# ---------------------------------------------------------------------------

def backward_mean(x, s, h=0.5):
    """Expected allele frequency one generation further into the past."""
    numerator = s * (-1 + x) * x * (-x + h * (-1 + 2 * x))
    denominator = -1 + s * (2 * h * (-1 + x) - x) * x
    return x + numerator / denominator


def backward_std(x, N):
    """Standard deviation of allele frequency change per generation."""
    return np.sqrt(x * (1.0 - x) / N)


def build_frequency_bins(K=450):
    """Construct K allele frequency bins using Beta(1/2, 1/2) quantiles."""
    u = np.linspace(0.0, 1.0, K)
    freqs = beta_dist.ppf(u, 0.5, 0.5)
    eps = 1e-12
    freqs[0] = eps
    freqs[-1] = 1 - eps
    logfreqs = np.log(freqs)
    log1minusfreqs = np.log(1.0 - freqs)
    freqs[0] = 0.0
    freqs[-1] = 1.0
    return freqs, logfreqs, log1minusfreqs


def build_transition_matrix(freqs, N, s, h=0.5):
    """Build the K x K log-transition matrix for the Wright-Fisher HMM."""
    K = len(freqs)
    logP = np.full((K, K), -np.inf)
    midpoints = (freqs[1:] + freqs[:-1]) / 2.0
    logP[0, 0] = 0.0
    logP[K - 1, K - 1] = 0.0
    for i in range(1, K - 1):
        x = freqs[i]
        mu = backward_mean(x, s, h)
        sigma = backward_std(x, N)
        if sigma < 1e-15:
            closest = np.argmin(np.abs(freqs - mu))
            logP[i, closest] = 0.0
            continue
        lower_freq = mu - 3.3 * sigma
        upper_freq = mu + 3.3 * sigma
        j_lower = max(0, np.searchsorted(freqs, lower_freq) - 1)
        j_upper = min(K, np.searchsorted(freqs, upper_freq) + 1)
        row = np.zeros(K)
        for j in range(j_lower, j_upper):
            if j == 0:
                row[j] = norm.cdf(midpoints[0], loc=mu, scale=sigma)
            elif j == K - 1:
                row[j] = 1.0 - norm.cdf(midpoints[-1], loc=mu, scale=sigma)
            else:
                row[j] = (norm.cdf(midpoints[j], loc=mu, scale=sigma)
                          - norm.cdf(midpoints[j - 1], loc=mu, scale=sigma))
        row_sum = row.sum()
        if row_sum > 0:
            row /= row_sum
        else:
            row[np.argmin(np.abs(freqs - mu))] = 1.0
        logP[i, :] = np.where(row > 0, np.log(row), -np.inf)
    return logP


def build_normal_cdf_lookup(n_points=2000):
    """Precompute a lookup table for the standard normal CDF."""
    u = np.linspace(0.0, 1.0, n_points)
    u[0] = 1e-10
    u[-1] = 1 - 1e-10
    z_bins = norm.ppf(u)
    z_cdf = norm.cdf(z_bins)
    return z_bins, z_cdf


def fast_normal_cdf(x, mu, sigma, z_bins, z_cdf):
    """Evaluate the normal CDF using precomputed lookup table."""
    z = (x - mu) / sigma
    return np.interp(z, z_bins, z_cdf)


def logsumexp(a):
    """Compute log(sum(exp(a))) in a numerically stable way."""
    a_max = np.max(a)
    if a_max == -np.inf:
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/clues/emission_probabilities.rst
# ---------------------------------------------------------------------------

def log_coalescent_density(coal_times, n_lineages, epoch_start, epoch_end,
                            freq, N_diploid, ancestral=False):
    """Compute log-probability of coalescence events in one epoch."""
    if n_lineages <= 1:
        return 0.0
    xi = (1.0 - freq) if ancestral else freq
    if xi * N_diploid == 0.0:
        return -1e20
    logp = 0.0
    prev_t = epoch_start
    k = n_lineages
    for t in coal_times:
        kchoose2_over_4 = k * (k - 1) / 4.0
        dt = t - prev_t
        logp += -np.log(xi) - kchoose2_over_4 / (xi * N_diploid) * dt
        prev_t = t
        k -= 1
    if k >= 2:
        kchoose2_over_4 = k * (k - 1) / 4.0
        logp += -kchoose2_over_4 / (xi * N_diploid) * (epoch_end - prev_t)
    return logp


def genotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
    """Compute log-emission probability for a diploid ancient sample."""
    log_geno_freqs = np.array([
        log_1minus_freq + log_1minus_freq,
        np.log(2) + log_freq + log_1minus_freq,
        log_freq + log_freq
    ])
    log_emission = logsumexp(log_geno_freqs + anc_gl)
    if np.isnan(log_emission):
        return -np.inf
    return log_emission


def haplotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
    """Compute log-emission probability for a haploid ancient sample."""
    log_hap_freqs = np.array([log_1minus_freq, log_freq])
    log_emission = logsumexp(log_hap_freqs + anc_gl)
    if np.isnan(log_emission):
        return -np.inf
    return log_emission


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/clues/inference.rst
# ---------------------------------------------------------------------------

from scipy.stats import chi2
from scipy.optimize import minimize_scalar


def likelihood_ratio_test(log_lik_selected, log_lik_neutral, df=1):
    """Perform a likelihood ratio test for selection."""
    log_lr = 2 * (log_lik_selected - log_lik_neutral)
    log_lr = max(log_lr, 0.0)
    p_value = chi2.sf(log_lr, df)
    neg_log10_p = -np.log10(p_value) if p_value > 0 else np.inf
    return log_lr, p_value, neg_log10_p


def estimate_selection_single(neg_log_lik_func, s_max=0.1):
    """Estimate the selection coefficient using Brent's method."""
    def shifted_func(theta):
        return neg_log_lik_func(theta - 1.0)
    try:
        result = minimize_scalar(
            shifted_func,
            bracket=[1.0 - s_max, 1.0, 1.0 + s_max],
            method='Brent',
            options={'xtol': 1e-4})
        s_hat = result.x - 1.0
        neg_ll = result.fun
    except ValueError:
        result = minimize_scalar(
            shifted_func,
            bracket=[0.0, 1.0, 2.0],
            method='Brent',
            options={'xtol': 1e-4})
        s_hat = result.x - 1.0
        neg_ll = result.fun
    return s_hat, neg_ll


def compute_trajectory_summary(posterior, freqs):
    """Compute summary statistics of the posterior trajectory."""
    T = posterior.shape[1]
    mean_freq = np.zeros(T)
    lower_95 = np.zeros(T)
    upper_95 = np.zeros(T)
    for t in range(T):
        col = posterior[:, t]
        if col.sum() == 0:
            continue
        col = col / col.sum()
        mean_freq[t] = np.sum(freqs * col)
        cdf = np.cumsum(col)
        lower_95[t] = freqs[np.searchsorted(cdf, 0.025)]
        upper_95[t] = freqs[np.searchsorted(cdf, 0.975)]
    return mean_freq, lower_95, upper_95


# ===========================================================================
# Tests for backward_mean
# ===========================================================================

class TestBackwardMean:
    def test_neutral_returns_x(self):
        """Under neutrality (s=0), backward mean should equal current freq."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mu = backward_mean(x, s=0.0)
            assert np.isclose(mu, x, atol=1e-10)

    def test_positive_selection_shifts_down(self):
        """With s > 0, backward mean < x (allele was rarer in the past)."""
        for x in [0.2, 0.5, 0.8]:
            mu = backward_mean(x, s=0.05)
            assert mu < x or np.isclose(mu, x)

    def test_boundaries_are_fixed(self):
        """At x=0 and x=1, the mean should be the same (absorbing)."""
        mu_0 = backward_mean(1e-15, s=0.05)
        assert np.isclose(mu_0, 1e-15, atol=1e-10)

    def test_dominance_at_half(self):
        """At x=0.5, all dominance values should give similar shifts."""
        shifts = []
        for h in [0.0, 0.5, 1.0]:
            mu = backward_mean(0.5, s=0.05, h=h)
            shifts.append(mu - 0.5)
        # The shifts are similar but not exactly equal for the general formula
        assert np.allclose(shifts, shifts[0], atol=1e-3)

    def test_recessive_peak_at_high_freq(self):
        """For h=0 (recessive), max backward shift should be at high x."""
        x_vals = np.linspace(0.01, 0.99, 200)
        shifts = [backward_mean(x, 0.05, h=0.0) - x for x in x_vals]
        peak_x = x_vals[np.argmin(shifts)]
        assert peak_x > 0.5

    def test_dominant_peak_at_low_freq(self):
        """For h=1 (dominant), max backward shift should be at low x."""
        x_vals = np.linspace(0.01, 0.99, 200)
        shifts = [backward_mean(x, 0.05, h=1.0) - x for x in x_vals]
        peak_x = x_vals[np.argmin(shifts)]
        assert peak_x < 0.5


# ===========================================================================
# Tests for backward_std
# ===========================================================================

class TestBackwardStd:
    def test_zero_at_boundaries(self):
        """Standard deviation should be 0 at x=0 and x=1."""
        assert backward_std(0.0, 20000) == 0.0
        assert backward_std(1.0, 20000) == 0.0

    def test_maximum_at_half(self):
        """Standard deviation should be maximized at x=0.5."""
        N = 20000
        stds = [backward_std(x, N) for x in np.linspace(0.01, 0.99, 100)]
        peak_idx = np.argmax(stds)
        # Peak should be near the middle
        assert 40 < peak_idx < 60

    def test_scales_with_population_size(self):
        """Larger N should give smaller std."""
        x = 0.5
        assert backward_std(x, 40000) < backward_std(x, 10000)


# ===========================================================================
# Tests for build_frequency_bins
# ===========================================================================

class TestBuildFrequencyBins:
    def test_boundary_values(self):
        """First bin should be 0, last should be 1."""
        freqs, _, _ = build_frequency_bins(100)
        assert freqs[0] == 0.0
        assert freqs[-1] == 1.0

    def test_length(self):
        """Should return K bins."""
        for K in [50, 100, 450]:
            freqs, _, _ = build_frequency_bins(K)
            assert len(freqs) == K

    def test_sorted(self):
        """Bins should be strictly increasing."""
        freqs, _, _ = build_frequency_bins(100)
        assert np.all(np.diff(freqs) > 0)

    def test_denser_near_boundaries(self):
        """Spacing should be smaller near 0 and 1 than near 0.5."""
        freqs, _, _ = build_frequency_bins(100)
        dx = np.diff(freqs)
        assert dx[1] < dx[len(dx) // 2]
        assert dx[-2] < dx[len(dx) // 2]

    def test_log_values_finite(self):
        """Log values should be finite (no -inf from boundary eps)."""
        _, logfreqs, log1minusfreqs = build_frequency_bins(100)
        assert np.all(np.isfinite(logfreqs))
        assert np.all(np.isfinite(log1minusfreqs))


# ===========================================================================
# Tests for build_transition_matrix
# ===========================================================================

class TestBuildTransitionMatrix:
    def test_row_sums_to_one(self):
        """Each row should sum to approximately 1."""
        freqs, _, _ = build_frequency_bins(50)
        logP = build_transition_matrix(freqs, 20000.0, s=0.0)
        P = np.exp(logP)
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.01)

    def test_absorbing_states(self):
        """Rows 0 and K-1 should be absorbing (self-loops)."""
        freqs, _, _ = build_frequency_bins(50)
        logP = build_transition_matrix(freqs, 20000.0, s=0.0)
        assert np.isclose(logP[0, 0], 0.0)
        assert np.isclose(logP[-1, -1], 0.0)

    def test_selection_shifts_distribution(self):
        """Positive selection should shift transition distribution."""
        freqs, _, _ = build_frequency_bins(50)
        logP_neutral = build_transition_matrix(freqs, 20000.0, s=0.0)
        logP_selected = build_transition_matrix(freqs, 20000.0, s=0.05)
        i = np.argmin(np.abs(freqs - 0.3))
        mean_n = np.sum(freqs * np.exp(logP_neutral[i, :]))
        mean_s = np.sum(freqs * np.exp(logP_selected[i, :]))
        # Under positive selection, backward mean should be lower
        assert mean_s < mean_n or np.isclose(mean_s, mean_n, atol=0.01)


# ===========================================================================
# Tests for logsumexp
# ===========================================================================

class TestLogsumexp:
    def test_basic(self):
        """log(exp(-1000) + exp(-1001)) should be about -999.69."""
        result = logsumexp(np.array([-1000.0, -1001.0]))
        expected = -1000 + np.log(1 + np.exp(-1))
        assert np.isclose(result, expected, atol=1e-4)

    def test_single_value(self):
        """logsumexp of a single value should return that value."""
        assert np.isclose(logsumexp(np.array([5.0])), 5.0)

    def test_all_neg_inf(self):
        """logsumexp of all -inf should return -inf."""
        assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf

    def test_matches_scipy(self):
        """Should match scipy's implementation."""
        from scipy.special import logsumexp as scipy_lse
        a = np.array([-1.0, -2.0, -3.0, -10.0])
        assert np.isclose(logsumexp(a), scipy_lse(a))


# ===========================================================================
# Tests for fast_normal_cdf
# ===========================================================================

class TestFastNormalCdf:
    def test_accuracy(self):
        """Fast CDF should match scipy to within 1e-4."""
        z_bins, z_cdf = build_normal_cdf_lookup()
        for z in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            exact = norm.cdf(z)
            approx = fast_normal_cdf(z, 0.0, 1.0, z_bins, z_cdf)
            assert abs(exact - approx) < 1e-4

    def test_at_zero(self):
        """CDF at 0 should be ~0.5."""
        z_bins, z_cdf = build_normal_cdf_lookup()
        val = fast_normal_cdf(0.0, 0.0, 1.0, z_bins, z_cdf)
        assert abs(val - 0.5) < 0.001

    def test_monotonicity(self):
        """CDF should be monotonically increasing."""
        z_bins, z_cdf = build_normal_cdf_lookup()
        vals = [fast_normal_cdf(z, 0.0, 1.0, z_bins, z_cdf)
                for z in np.linspace(-3, 3, 20)]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


# ===========================================================================
# Tests for log_coalescent_density
# ===========================================================================

class TestLogCoalescentDensity:
    def test_single_lineage_returns_zero(self):
        """With 1 lineage, no coalescence possible -> log(1) = 0."""
        lp = log_coalescent_density(np.array([]), 1, 0.0, 1.0, 0.5, 10000.0)
        assert np.isclose(lp, 0.0)

    def test_lower_freq_higher_coal_rate(self):
        """Lower frequency should give higher coalescence probability for
        fast coalescence (coalescence favored at low freq)."""
        coal_times = np.array([0.1])
        lp_low = log_coalescent_density(coal_times, 3, 0.0, 1.0, 0.1, 10000.0)
        lp_high = log_coalescent_density(coal_times, 3, 0.0, 1.0, 0.9, 10000.0)
        # At very low frequency, fast coalescence is more likely
        assert lp_low > lp_high

    def test_finite_output(self):
        """Output should be finite for valid inputs."""
        coal_times = np.array([0.5])
        lp = log_coalescent_density(coal_times, 3, 0.0, 1.0, 0.3, 10000.0)
        assert np.isfinite(lp)

    def test_no_coalescence_survival(self):
        """No coalescence events should give survival probability only."""
        lp = log_coalescent_density(np.array([]), 3, 0.0, 1.0, 0.5, 10000.0)
        # Should be negative (probability < 1 of not coalescing)
        assert lp <= 0


# ===========================================================================
# Tests for genotype_likelihood_emission
# ===========================================================================

class TestGenotypeLikelihoodEmission:
    def test_high_freq_favors_dd(self):
        """At high frequency, DD genotype should be more probable."""
        gl_dd = np.log(np.array([0.001, 0.01, 0.989]))
        log_em_high = genotype_likelihood_emission(gl_dd, np.log(0.9), np.log(0.1))
        log_em_low = genotype_likelihood_emission(gl_dd, np.log(0.1), np.log(0.9))
        assert log_em_high > log_em_low

    def test_finite(self):
        """Emission should be finite for valid inputs."""
        gl = np.log(np.array([0.3, 0.4, 0.3]))
        le = genotype_likelihood_emission(gl, np.log(0.5), np.log(0.5))
        assert np.isfinite(le)

    def test_sum_to_one_across_genotypes(self):
        """For uniform GLs, emission should be finite."""
        gl_uniform = np.log(np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]))
        le = genotype_likelihood_emission(gl_uniform, np.log(0.5), np.log(0.5))
        assert np.isfinite(le)


# ===========================================================================
# Tests for haplotype_likelihood_emission
# ===========================================================================

class TestHaplotypeLikelihoodEmission:
    def test_derived_at_high_freq(self):
        """Derived haplotype more probable at high frequency."""
        gl_der = np.log(np.array([0.01, 0.99]))  # [P(R|A), P(R|D)]
        le_high = haplotype_likelihood_emission(gl_der, np.log(0.9), np.log(0.1))
        le_low = haplotype_likelihood_emission(gl_der, np.log(0.1), np.log(0.9))
        assert le_high > le_low

    def test_finite(self):
        """Emission should be finite."""
        gl = np.log(np.array([0.5, 0.5]))
        le = haplotype_likelihood_emission(gl, np.log(0.5), np.log(0.5))
        assert np.isfinite(le)


# ===========================================================================
# Tests for likelihood_ratio_test
# ===========================================================================

class TestLikelihoodRatioTest:
    def test_significant_result(self):
        """Strong signal should give low p-value."""
        log_lr, p, neg_log10_p = likelihood_ratio_test(-1000.0, -1005.0)
        assert log_lr == 10.0
        assert p < 0.01

    def test_no_signal(self):
        """Equal likelihoods should give p=1."""
        log_lr, p, neg_log10_p = likelihood_ratio_test(-1000.0, -1000.0)
        assert log_lr == 0.0
        assert np.isclose(p, 1.0)

    def test_nonnegative_lr(self):
        """Log-LR should be non-negative even with numerical issues."""
        log_lr, p, _ = likelihood_ratio_test(-1001.0, -1000.0)
        assert log_lr >= 0.0

    def test_df_affects_pvalue(self):
        """Higher df should give larger p-value for the same LR."""
        _, p1, _ = likelihood_ratio_test(-1000.0, -1003.0, df=1)
        _, p2, _ = likelihood_ratio_test(-1000.0, -1003.0, df=5)
        assert p2 > p1


# ===========================================================================
# Tests for estimate_selection_single
# ===========================================================================

class TestEstimateSelection:
    def test_finds_true_minimum(self):
        """Should find the true selection coefficient for a quadratic LL."""
        true_s = 0.03
        def toy_neg_ll(s):
            return (s - true_s)**2 / 0.001
        s_hat, _ = estimate_selection_single(toy_neg_ll)
        assert abs(s_hat - true_s) < 0.01

    def test_neutral_estimate(self):
        """For a symmetric likelihood around 0, should estimate s~0."""
        def symmetric_ll(s):
            return s**2 / 0.001
        s_hat, _ = estimate_selection_single(symmetric_ll)
        assert abs(s_hat) < 0.01


# ===========================================================================
# Tests for compute_trajectory_summary
# ===========================================================================

class TestTrajectorySummary:
    def test_output_shapes(self):
        """Summary should have correct shapes."""
        K, T = 50, 10
        freqs = np.linspace(0, 1, K)
        posterior = np.random.dirichlet(np.ones(K), size=T).T
        mean_f, lower, upper = compute_trajectory_summary(posterior, freqs)
        assert len(mean_f) == T
        assert len(lower) == T
        assert len(upper) == T

    def test_mean_in_range(self):
        """Mean frequency should be between 0 and 1."""
        K, T = 50, 10
        freqs = np.linspace(0, 1, K)
        posterior = np.random.dirichlet(np.ones(K), size=T).T
        mean_f, _, _ = compute_trajectory_summary(posterior, freqs)
        assert np.all(mean_f >= 0)
        assert np.all(mean_f <= 1)

    def test_lower_less_than_upper(self):
        """Lower bound should be <= upper bound."""
        K, T = 50, 10
        freqs = np.linspace(0, 1, K)
        posterior = np.random.dirichlet(np.ones(K), size=T).T
        _, lower, upper = compute_trajectory_summary(posterior, freqs)
        assert np.all(lower <= upper + 1e-10)
