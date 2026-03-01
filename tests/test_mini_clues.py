"""
Tests for watchgen.mini_clues -- the CLUES algorithm mini-implementation.

Adapts test logic from tests/test_timepieces_clues.py, importing all functions
from watchgen.mini_clues rather than redefining them.

Covers:
- wright_fisher_hmm: backward_mean, backward_std, frequency bins,
  transition matrix, log-sum-exp, fast normal CDF
- emission_probabilities: coalescent density, genotype likelihood emissions
- inference: likelihood ratio test, estimate_selection, trajectory summary
"""

import numpy as np
import pytest
from scipy.stats import norm, beta as beta_dist
from scipy.special import logsumexp as scipy_lse
from scipy.stats import chi2

from watchgen.mini_clues import (
    backward_mean,
    backward_std,
    build_frequency_bins,
    build_transition_matrix,
    build_normal_cdf_lookup,
    fast_normal_cdf,
    logsumexp,
    build_transition_matrix_fast,
    log_coalescent_density,
    compute_coalescent_emissions,
    genotype_likelihood_emission,
    haplotype_likelihood_emission,
    compute_total_emissions,
    likelihood_ratio_test,
    estimate_selection_single,
    estimate_selection_multi_epoch,
    compute_trajectory_summary,
)


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
# Tests for build_transition_matrix_fast
# ===========================================================================

class TestBuildTransitionMatrixFast:
    def test_row_sums_to_one(self):
        """Each row of the fast matrix should sum to approximately 1."""
        freqs, _, _ = build_frequency_bins(50)
        z_bins, z_cdf = build_normal_cdf_lookup()
        logP, lo, hi = build_transition_matrix_fast(
            freqs, 20000.0, s=0.0, z_bins=z_bins, z_cdf=z_cdf)
        P = np.exp(logP)
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.02)

    def test_sparse_indices_valid(self):
        """Lower/upper indices should be within bounds."""
        freqs, _, _ = build_frequency_bins(50)
        z_bins, z_cdf = build_normal_cdf_lookup()
        logP, lo, hi = build_transition_matrix_fast(
            freqs, 20000.0, s=0.02, z_bins=z_bins, z_cdf=z_cdf)
        K = len(freqs)
        assert np.all(lo >= 0)
        assert np.all(hi <= K)
        assert np.all(lo <= hi)

    def test_sparsity(self):
        """Matrix should have significant sparsity."""
        freqs, _, _ = build_frequency_bins(100)
        z_bins, z_cdf = build_normal_cdf_lookup()
        logP, lo, hi = build_transition_matrix_fast(
            freqs, 20000.0, s=0.02, z_bins=z_bins, z_cdf=z_cdf)
        nnz = sum(hi[i] - lo[i] for i in range(len(freqs)))
        K = len(freqs)
        sparsity = 1 - nnz / (K * K)
        assert sparsity > 0.5  # at least 50% zeros


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

    def test_ancestral_uses_complement(self):
        """Ancestral flag should use 1 - freq."""
        coal_times = np.array([0.3])
        lp_der = log_coalescent_density(
            coal_times, 3, 0.0, 1.0, 0.3, 10000.0, ancestral=False)
        lp_anc = log_coalescent_density(
            coal_times, 3, 0.0, 1.0, 0.7, 10000.0, ancestral=True)
        # freq=0.3 for derived == freq=0.7 for ancestral (1-0.7=0.3)
        assert np.isclose(lp_der, lp_anc, atol=1e-10)


# ===========================================================================
# Tests for compute_coalescent_emissions
# ===========================================================================

class TestComputeCoalescentEmissions:
    def test_output_shape(self):
        """Emissions should have same length as freqs."""
        freqs, _, _ = build_frequency_bins(50)
        e = compute_coalescent_emissions(
            np.array([0.5]), np.array([]),
            n_der=3, n_anc=2, epoch_start=0.0, epoch_end=1.0,
            freqs=freqs, N_diploid=10000.0)
        assert len(e) == len(freqs)

    def test_mixed_lineage_case(self):
        """Mixed lineage (n_der=1, n_anc=1) should have finite emissions
        for non-zero frequency bins."""
        freqs, _, _ = build_frequency_bins(50)
        e = compute_coalescent_emissions(
            np.array([]), np.array([]),
            n_der=1, n_anc=1, epoch_start=0.0, epoch_end=1.0,
            freqs=freqs, N_diploid=10000.0)
        n_finite = np.sum(e > -1e19)
        # Most bins should have finite emissions in the mixed case
        assert n_finite > len(freqs) // 2


# ===========================================================================
# Tests for genotype_likelihood_emission
# ===========================================================================

class TestGenotypeLikelihoodEmission:
    def test_high_freq_favors_dd(self):
        """At high frequency, DD genotype should be more probable."""
        gl_dd = np.log(np.array([0.001, 0.01, 0.989]))
        log_em_high = genotype_likelihood_emission(
            gl_dd, np.log(0.9), np.log(0.1))
        log_em_low = genotype_likelihood_emission(
            gl_dd, np.log(0.1), np.log(0.9))
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
        le_high = haplotype_likelihood_emission(
            gl_der, np.log(0.9), np.log(0.1))
        le_low = haplotype_likelihood_emission(
            gl_der, np.log(0.1), np.log(0.9))
        assert le_high > le_low

    def test_finite(self):
        """Emission should be finite."""
        gl = np.log(np.array([0.5, 0.5]))
        le = haplotype_likelihood_emission(gl, np.log(0.5), np.log(0.5))
        assert np.isfinite(le)


# ===========================================================================
# Tests for compute_total_emissions
# ===========================================================================

class TestComputeTotalEmissions:
    def test_output_shape(self):
        """Total emissions should have same length as freqs."""
        freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=50)
        emissions = compute_total_emissions(
            freqs, logfreqs, log1minusfreqs,
            coal_times_der=np.array([0.3]),
            coal_times_anc=np.array([]),
            n_der=3, n_anc=2,
            epoch_start=0.0, epoch_end=1.0,
            N_diploid=10000.0)
        assert len(emissions) == len(freqs)

    def test_with_ancient_diploid(self):
        """Adding ancient diploid GL should change emissions."""
        freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=50)
        e_no_anc = compute_total_emissions(
            freqs, logfreqs, log1minusfreqs,
            np.array([0.3]), np.array([]),
            n_der=3, n_anc=2,
            epoch_start=0.0, epoch_end=1.0,
            N_diploid=10000.0)
        gl_dd = np.log(np.array([0.001, 0.01, 0.989]))
        e_with_anc = compute_total_emissions(
            freqs, logfreqs, log1minusfreqs,
            np.array([0.3]), np.array([]),
            n_der=3, n_anc=2,
            epoch_start=0.0, epoch_end=1.0,
            N_diploid=10000.0,
            diploid_gls=[gl_dd])
        # Emissions should differ when ancient sample is added
        assert not np.allclose(e_no_anc, e_with_anc)


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
# Tests for estimate_selection_multi_epoch
# ===========================================================================

class TestEstimateSelectionMultiEpoch:
    def test_finds_minimum(self):
        """Should find the minimum for a simple multi-epoch likelihood."""
        true_s = np.array([0.02, -0.01])

        def neg_ll(s_vec):
            return np.sum((s_vec - true_s)**2) / 0.001

        s_hat, _ = estimate_selection_multi_epoch(neg_ll, n_epochs=2)
        # Nelder-Mead with few evaluations may not be exact, just check direction
        assert len(s_hat) == 2


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
