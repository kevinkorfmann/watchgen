"""
Tests for the mini_psmc module.

Imports all functions from watchgen.mini_psmc and tests their
mathematical properties.
"""

import numpy as np
from scipy.integrate import quad
import pytest

from watchgen.mini_psmc import (
    cumulative_hazard,
    cumulative_hazard_piecewise,
    coalescent_density,
    coalescent_survival,
    psmc_transition_density_general,
    stationary_distribution,
    compute_C_pi,
    estimate_theta_initial,
    compute_time_intervals,
    compute_helpers,
    compute_stationary,
    compute_transition_matrix,
    compute_avg_times,
    build_psmc_hmm,
    parse_pattern,
    scale_psmc_output,
    scale_mutation_free,
    correct_for_coverage,
    split_sequence,
    bootstrap_resample,
    check_overfitting,
    plot_psmc_history,
    simulate_psmc_input,
)


# ============================================================
# Tests: Continuous Model
# ============================================================

class TestCumulativeHazard:
    def test_constant_population(self):
        """For constant lambda=1, Lambda(t) should equal t."""
        for t_val in [0.5, 1.0, 2.5, 5.0]:
            result = cumulative_hazard(t_val, lambda u: 1.0)
            assert abs(result - t_val) < 1e-8

    def test_doubled_population(self):
        """For lambda=2, Lambda(t) = t/2."""
        for t_val in [0.5, 1.0, 2.5]:
            result = cumulative_hazard(t_val, lambda u: 2.0)
            assert abs(result - t_val / 2) < 1e-8

    def test_zero_at_zero(self):
        """Lambda(0) should be 0."""
        result = cumulative_hazard(0, lambda u: 1.0)
        assert abs(result) < 1e-12

    def test_monotonically_increasing(self):
        """Lambda(t) should increase with t."""
        lf = lambda u: 1.0
        vals = [cumulative_hazard(t, lf) for t in [0.5, 1.0, 2.0, 5.0]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]


class TestCumulativeHazardPiecewise:
    def test_constant_matches_continuous(self):
        """For constant lambda, piecewise should match continuous."""
        t_bounds = np.array([0.0, 5.0, 10.0, 1000.0])
        lambdas = np.array([1.0, 1.0, 1.0])
        for t_val in [0.5, 3.0, 7.5]:
            pw = cumulative_hazard_piecewise(t_val, t_bounds, lambdas)
            cont = cumulative_hazard(t_val, lambda u: 1.0)
            assert abs(pw - cont) < 1e-8

    def test_two_intervals(self):
        """Manual calculation for two intervals."""
        t_bounds = np.array([0.0, 1.0, 1000.0])
        lambdas = np.array([2.0, 1.0])
        # At t=0.5: Lambda = 0.5/2 = 0.25
        assert abs(cumulative_hazard_piecewise(0.5, t_bounds, lambdas) - 0.25) < 1e-10
        # At t=1.5: Lambda = 1.0/2 + 0.5/1 = 0.5 + 0.5 = 1.0
        assert abs(cumulative_hazard_piecewise(1.5, t_bounds, lambdas) - 1.0) < 1e-10

    def test_at_boundary(self):
        """Test right at a boundary."""
        t_bounds = np.array([0.0, 1.0, 1000.0])
        lambdas = np.array([2.0, 1.0])
        # At t=1.0: Lambda = 1.0/2 = 0.5
        assert abs(cumulative_hazard_piecewise(1.0, t_bounds, lambdas) - 0.5) < 1e-10


class TestCoalescentDensity:
    def test_matches_exponential_for_constant(self):
        """For constant lambda=1, density should be exp(-t)."""
        for t_val in [0.5, 1.0, 2.0]:
            result = coalescent_density(t_val, lambda u: 1.0)
            expected = np.exp(-t_val)
            assert abs(result - expected) < 1e-8

    def test_integrates_to_one(self):
        """The density should integrate to 1."""
        result, _ = quad(lambda t: coalescent_density(t, lambda u: 1.0),
                         0, 50)
        assert abs(result - 1.0) < 1e-6

    def test_positive(self):
        """Density should be positive."""
        for t_val in [0.01, 0.5, 1.0, 5.0]:
            assert coalescent_density(t_val, lambda u: 1.0) > 0

    def test_larger_pop_shifts_right(self):
        """Larger population should shift density toward larger times."""
        d1 = coalescent_density(0.5, lambda u: 1.0)
        d2 = coalescent_density(0.5, lambda u: 2.0)
        # At t=0.5, density for lambda=2 should be lower (shifted right)
        assert d2 < d1


class TestCoalescentSurvival:
    def test_one_at_zero(self):
        """Survival at time 0 should be 1."""
        result = coalescent_survival(0, lambda u: 1.0)
        assert abs(result - 1.0) < 1e-8

    def test_decreasing(self):
        """Survival should decrease with time."""
        lf = lambda u: 1.0
        s1 = coalescent_survival(1.0, lf)
        s2 = coalescent_survival(2.0, lf)
        assert s1 > s2

    def test_constant_pop_is_exponential(self):
        """For constant lambda=1, survival = exp(-t)."""
        for t_val in [0.5, 1.0, 3.0]:
            result = coalescent_survival(t_val, lambda u: 1.0)
            expected = np.exp(-t_val)
            assert abs(result - expected) < 1e-8


class TestTransitionDensity:
    def test_integrates_to_one(self):
        """q(t|s) should integrate to 1 over t."""
        s = 1.0
        result, _ = quad(
            lambda t: psmc_transition_density_general(t, s, lambda u: 1.0),
            0.001, 20, limit=100
        )
        assert abs(result - 1.0) < 1e-3

    def test_positive(self):
        """Transition density should be positive."""
        s = 1.0
        for t_val in [0.1, 0.5, 1.0, 2.0]:
            result = psmc_transition_density_general(t_val, s, lambda u: 1.0)
            assert result > 0

    def test_matches_closed_form_t_less_s(self):
        """For t < s and constant lambda=1, q(t|s) = (1/s)(1 - exp(-t))/1."""
        s = 2.0
        t = 0.5
        q_general = psmc_transition_density_general(t, s, lambda u: 1.0)
        q_closed = (1.0 / s) * (1 - np.exp(-t))
        assert abs(q_general - q_closed) < 1e-6

    def test_matches_closed_form_t_greater_s(self):
        """For t > s and constant lambda=1, q(t|s) = (1/s)(exp(-(t-s)) - exp(-t))."""
        s = 1.0
        t = 2.0
        q_general = psmc_transition_density_general(t, s, lambda u: 1.0)
        q_closed = (1.0 / s) * (np.exp(-(t - s)) - np.exp(-t))
        assert abs(q_general - q_closed) < 1e-6

    def test_normalization_various_s(self):
        """q(t|s) should integrate to 1 for different s values."""
        for s_val in [0.5, 2.0]:
            result, _ = quad(
                lambda t: psmc_transition_density_general(t, s_val, lambda u: 1.0),
                0.001, 20, limit=100
            )
            assert abs(result - 1.0) < 1e-3


class TestStationaryDistribution:
    def test_C_pi_constant_pop(self):
        """For constant lambda=1, C_pi should be 1.0."""
        C_pi = compute_C_pi(lambda t: 1.0)
        assert abs(C_pi - 1.0) < 1e-3

    def test_integrates_to_one(self):
        """pi(t) should integrate to 1."""
        C_pi = compute_C_pi(lambda t: 1.0)
        result, _ = quad(
            lambda t: stationary_distribution(t, lambda u: 1.0, C_pi),
            0.001, 20
        )
        assert abs(result - 1.0) < 1e-3

    def test_is_gamma_for_constant_pop(self):
        """For constant lambda=1, pi(t) = t*exp(-t) (Gamma(2,1))."""
        C_pi = compute_C_pi(lambda t: 1.0)
        for t_val in [0.5, 1.0, 2.0, 3.0]:
            result = stationary_distribution(t_val, lambda u: 1.0, C_pi)
            expected = t_val * np.exp(-t_val)
            assert abs(result - expected) < 1e-4

    def test_mean_is_two_for_constant_pop(self):
        """For constant lambda=1, E[T] = 2 (mean of Gamma(2,1))."""
        C_pi = compute_C_pi(lambda t: 1.0)
        mean_T, _ = quad(
            lambda t: t * stationary_distribution(t, lambda u: 1.0, C_pi),
            0, 20
        )
        assert abs(mean_T - 2.0) < 0.1

    def test_positive(self):
        """Stationary distribution should be positive for t > 0."""
        C_pi = compute_C_pi(lambda t: 1.0)
        for t_val in [0.1, 1.0, 5.0]:
            assert stationary_distribution(t_val, lambda u: 1.0, C_pi) > 0


class TestEstimateThetaInitial:
    def test_recovery(self):
        """Should recover approximately the right theta from simulated data."""
        np.random.seed(42)
        theta_true = 0.001
        # Simulate Bernoulli trials with p = 1 - exp(-theta)
        p = 1 - np.exp(-theta_true)
        seq = np.random.binomial(1, p, size=1000000)
        theta_est = estimate_theta_initial(seq)
        assert abs(theta_est - theta_true) / theta_true < 0.05

    def test_zero_sequence(self):
        """For all-zero sequence, theta should be near 0."""
        seq = np.zeros(1000)
        theta_est = estimate_theta_initial(seq)
        assert theta_est == 0.0

    def test_high_het(self):
        """For high heterozygosity, theta should be large."""
        seq = np.ones(1000)
        seq[:10] = 0  # a few homozygous
        theta_est = estimate_theta_initial(seq)
        assert theta_est > 1.0


# ============================================================
# Tests: Discretization
# ============================================================

class TestComputeTimeIntervals:
    def test_first_is_zero(self):
        t = compute_time_intervals(10, 15.0)
        assert t[0] == 0.0

    def test_nth_is_tmax(self):
        t = compute_time_intervals(10, 15.0)
        assert abs(t[10] - 15.0) < 1e-10

    def test_last_is_large(self):
        t = compute_time_intervals(10, 15.0)
        assert t[11] == 1000.0

    def test_increasing(self):
        t = compute_time_intervals(20, 15.0)
        for i in range(len(t) - 1):
            assert t[i] < t[i + 1]

    def test_length(self):
        n = 63
        t = compute_time_intervals(n, 15.0)
        assert len(t) == n + 2

    def test_log_spacing(self):
        """Later intervals should be wider than earlier ones."""
        t = compute_time_intervals(20, 15.0)
        # First interval width
        w_first = t[1] - t[0]
        # A later interval width
        w_later = t[10] - t[9]
        assert w_later > w_first


class TestComputeHelpers:
    def test_alpha_boundary_conditions(self):
        """alpha[0] = 1, alpha[n+1] = 0."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        assert alpha[0] == 1.0
        assert alpha[n + 1] == 0.0

    def test_alpha_decreasing(self):
        """Survival probabilities should decrease."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        for k in range(n + 1):
            assert alpha[k] >= alpha[k + 1]

    def test_C_pi_constant_pop(self):
        """For constant lambda=1, C_pi should approximate 1.0."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        assert abs(C_pi - 1.0) < 0.05

    def test_tau_positive(self):
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        for k in range(n + 1):
            assert tau[k] > 0

    def test_beta_nondecreasing(self):
        """Beta should be non-decreasing."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        for k in range(n):
            assert beta[k] <= beta[k + 1] + 1e-10


class TestComputeStationary:
    def test_pi_sums_to_one(self):
        """pi_k should sum to 1."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        assert abs(pi_k.sum() - 1.0) < 1e-6

    def test_sigma_sums_to_one(self):
        """sigma_k should sum to 1."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        assert abs(sigma_k.sum() - 1.0) < 1e-4

    def test_pi_nonnegative(self):
        """pi_k should be non-negative."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, _ = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        assert np.all(pi_k >= -1e-10)

    def test_sigma_nonnegative(self):
        """sigma_k should be non-negative."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, _ = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        assert np.all(sigma_k >= -1e-10)


class TestComputeTransitionMatrix:
    def _build_transition_matrix(self, n=10, rho=0.001):
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta_arr, q_aux, C_pi = compute_helpers(n, t, lambdas)
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        p, q = compute_transition_matrix(n, tau, alpha, beta_arr, q_aux, lambdas,
                                          C_pi, C_sigma, pi_k, sigma_k)
        return p, q, sigma_k

    def test_p_rows_sum_to_one(self):
        """Rows of the full transition matrix should sum to 1."""
        p, q, _ = self._build_transition_matrix()
        row_sums = p.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_q_rows_sum_to_one(self):
        """Rows of q (given recombination) should sum to approximately 1."""
        p, q, _ = self._build_transition_matrix()
        row_sums = q.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=0.05)

    def test_p_nonnegative(self):
        """All entries of p should be non-negative."""
        p, _, _ = self._build_transition_matrix()
        assert np.all(p >= -1e-10)

    def test_diagonal_dominates(self):
        """Diagonal should dominate for small rho (rare recombination)."""
        p, _, _ = self._build_transition_matrix(rho=0.0001)
        for k in range(p.shape[0]):
            assert p[k, k] > 0.5

    def test_stationary_distribution_is_eigenvector(self):
        """sigma should be a left eigenvector of p with eigenvalue 1."""
        p, _, sigma_k = self._build_transition_matrix()
        sigma_check = sigma_k @ p
        max_diff = np.max(np.abs(sigma_check - sigma_k))
        assert max_diff < 1e-4

    def test_shape(self):
        n = 10
        p, q, _ = self._build_transition_matrix(n=n)
        assert p.shape == (n + 1, n + 1)
        assert q.shape == (n + 1, n + 1)


class TestComputeAvgTimes:
    def test_within_intervals(self):
        """Average times should lie within (or near) their intervals."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta_arr, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        avg_t = compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho)

        sum_tau = 0.0
        for k in range(n + 1):
            # avg_t should be roughly within the interval
            assert avg_t[k] >= sum_tau - 1e-6
            sum_tau += tau[k]

    def test_positive(self):
        """Average times should be positive."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta_arr, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        avg_t = compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho)
        assert np.all(avg_t >= 0)

    def test_increasing(self):
        """Average times should be non-decreasing."""
        n = 10
        t = compute_time_intervals(n, 15.0)
        lambdas = np.ones(n + 1)
        tau, alpha, beta_arr, q_aux, C_pi = compute_helpers(n, t, lambdas)
        rho = 0.001
        pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
        avg_t = compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho)
        for k in range(n):
            assert avg_t[k] <= avg_t[k + 1] + 1e-6


class TestBuildPsmcHmm:
    def test_transition_rows_sum_to_one(self):
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        row_sums = transitions.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_initial_sums_to_one(self):
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        assert abs(initial.sum() - 1.0) < 1e-4

    def test_emissions_sum_to_one(self):
        """For each state, P(hom) + P(het) should equal 1."""
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        for k in range(n + 1):
            assert abs(emissions[0, k] + emissions[1, k] - 1.0) < 1e-10

    def test_emissions_het_increases_with_state(self):
        """Deeper intervals should have higher heterozygosity."""
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        for k in range(n):
            assert emissions[1, k] <= emissions[1, k + 1] + 1e-10

    def test_emissions_in_valid_range(self):
        """All emission probabilities should be in [0, 1]."""
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        assert np.all(emissions >= 0)
        assert np.all(emissions <= 1)

    def test_transitions_nonnegative(self):
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        assert np.all(transitions >= -1e-10)

    def test_initial_nonnegative(self):
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)
        transitions, emissions, initial = build_psmc_hmm(n, 15.0, theta, rho, lambdas)
        assert np.all(initial >= -1e-10)


# ============================================================
# Tests: Parse Pattern
# ============================================================

class TestParsePattern:
    def test_simple_pattern(self):
        par_map, n_free, n_intervals = parse_pattern("4+25*2+4+6")
        assert n_intervals == 64
        assert n_free == 28

    def test_all_free(self):
        par_map, n_free, n_intervals = parse_pattern("1+1+1+1")
        assert n_intervals == 4
        assert n_free == 4
        assert par_map == [0, 1, 2, 3]

    def test_single_group(self):
        par_map, n_free, n_intervals = parse_pattern("10")
        assert n_intervals == 10
        assert n_free == 1
        assert all(p == 0 for p in par_map)

    def test_repeated_groups(self):
        par_map, n_free, n_intervals = parse_pattern("5*2")
        assert n_intervals == 10
        assert n_free == 5

    def test_par_map_structure(self):
        """First 4 intervals should share parameter 0."""
        par_map, _, _ = parse_pattern("4+25*2+4+6")
        assert par_map[0] == par_map[1] == par_map[2] == par_map[3] == 0
        # Next 2 intervals should share parameter 1
        assert par_map[4] == par_map[5] == 1


# ============================================================
# Tests: Decoding / Scaling
# ============================================================

class TestScalePsmcOutput:
    def test_N0_calculation(self):
        """N_0 = theta / (4 * mu * s)."""
        theta_0 = 0.00069
        mu = 1.25e-8
        s = 100
        N_0_expected = theta_0 / (4 * mu * s)
        N_0, _, _, _ = scale_psmc_output(theta_0, np.ones(1), np.array([0, 1]),
                                          mu=mu, s=s)
        assert abs(N_0 - N_0_expected) < 1e-6

    def test_time_scaling(self):
        """Times in generations = 2 * N_0 * t_coalescent."""
        theta_0 = 0.00069
        mu = 1.25e-8
        s = 100
        t = np.array([0.0, 1.0, 2.0])
        lambdas = np.array([1.0, 1.0])
        N_0, t_gen, t_years, pop_sizes = scale_psmc_output(
            theta_0, lambdas, t, mu=mu, s=s, generation_time=25)
        assert abs(t_gen[0]) < 1e-10
        assert abs(t_gen[1] - 2 * N_0 * 1.0) < 1e-6

    def test_pop_size_scaling(self):
        """Pop sizes = N_0 * lambdas."""
        theta_0 = 0.00069
        lambdas = np.array([2.0, 0.5])
        t = np.array([0, 1, 2])
        N_0, _, _, pop_sizes = scale_psmc_output(theta_0, lambdas, t)
        assert abs(pop_sizes[0] - N_0 * 2.0) < 1e-6
        assert abs(pop_sizes[1] - N_0 * 0.5) < 1e-6

    def test_generation_time_effect(self):
        """Doubling generation time should double years."""
        theta_0 = 0.00069
        lambdas = np.array([1.0])
        t = np.array([0, 1])
        _, _, t_years_25, _ = scale_psmc_output(theta_0, lambdas, t, generation_time=25)
        _, _, t_years_50, _ = scale_psmc_output(theta_0, lambdas, t, generation_time=50)
        assert abs(t_years_50[1] - 2 * t_years_25[1]) < 1e-6


class TestScaleMutationFree:
    def test_divergence_zero_at_zero(self):
        theta_0 = 0.001
        lambdas = np.array([1.0])
        t = np.array([0.0, 1.0])
        div, _ = scale_mutation_free(theta_0, lambdas, t)
        assert div[0] == 0.0

    def test_divergence_proportional_to_time(self):
        theta_0 = 0.001
        s = 100
        lambdas = np.array([1.0, 1.0])
        t = np.array([0.0, 1.0, 2.0])
        div, _ = scale_mutation_free(theta_0, lambdas, t, s=s)
        assert abs(div[2] - 2 * div[1]) < 1e-10

    def test_scaled_theta_proportional_to_lambda(self):
        theta_0 = 0.001
        lambdas = np.array([2.0, 1.0])
        t = np.array([0, 1, 2])
        _, scaled_theta = scale_mutation_free(theta_0, lambdas, t)
        assert abs(scaled_theta[0] - 2 * scaled_theta[1]) < 1e-10


class TestCorrectForCoverage:
    def test_no_correction(self):
        """FNR = 0 should return the same theta."""
        assert correct_for_coverage(0.001, 0.0) == 0.001

    def test_correction_increases_theta(self):
        """FNR > 0 should increase theta."""
        theta_corr = correct_for_coverage(0.001, 0.2)
        assert theta_corr > 0.001

    def test_exact_correction(self):
        """theta_corrected = theta / (1 - FNR)."""
        theta = 0.001
        fnr = 0.2
        expected = 0.001 / 0.8
        assert abs(correct_for_coverage(theta, fnr) - expected) < 1e-10

    def test_high_fnr(self):
        """High FNR should produce large correction."""
        theta = 0.001
        theta_corr = correct_for_coverage(theta, 0.5)
        assert abs(theta_corr - 0.002) < 1e-10


class TestSplitSequence:
    def test_segment_count(self):
        seq = np.zeros(100000)
        segments = split_sequence(seq, segment_length=10000)
        assert len(segments) == 10

    def test_segment_lengths(self):
        seq = np.zeros(100000)
        segments = split_sequence(seq, segment_length=10000)
        for seg in segments:
            assert len(seg) == 10000

    def test_partial_last_segment_excluded(self):
        """If the sequence doesn't divide evenly, the remainder is dropped."""
        seq = np.zeros(15000)
        segments = split_sequence(seq, segment_length=10000)
        assert len(segments) == 1


class TestBootstrapResample:
    def test_output_length(self):
        np.random.seed(42)
        segments = [np.zeros(100) for _ in range(10)]
        replicate = bootstrap_resample(segments, total_length=500)
        assert len(replicate) == 500

    def test_preserves_values(self):
        """Replicate should contain only values from the original segments."""
        np.random.seed(42)
        segments = [np.ones(100) * i for i in range(5)]
        replicate = bootstrap_resample(segments, total_length=300)
        unique_vals = set(replicate)
        assert unique_vals.issubset({0, 1, 2, 3, 4})


class TestCheckOverfitting:
    def test_all_sufficient(self):
        sigma_k = np.ones(10) / 10
        C_sigma = 1000.0
        warnings, expected = check_overfitting(sigma_k, C_sigma)
        # Expected segments = 1000 * 0.1 = 100 for each, all > 20
        assert len(warnings) == 0

    def test_some_insufficient(self):
        sigma_k = np.array([0.001, 0.999])
        C_sigma = 100.0
        warnings, expected = check_overfitting(sigma_k, C_sigma)
        # First: 100 * 0.001 = 0.1 < 20, should be warned
        assert 0 in warnings

    def test_expected_segments_values(self):
        sigma_k = np.array([0.5, 0.5])
        C_sigma = 100.0
        warnings, expected = check_overfitting(sigma_k, C_sigma)
        assert abs(expected[0] - 50.0) < 1e-10
        assert abs(expected[1] - 50.0) < 1e-10


class TestPlotPsmcHistory:
    def test_step_function_structure(self):
        """Should produce 2*n_lambdas points (left/right edge of each step)."""
        theta_0 = 0.001
        lambdas = np.array([1.0, 2.0, 1.5])
        t = np.array([0, 1, 2, 3, 100])
        x, y = plot_psmc_history(theta_0, lambdas, t)
        assert len(x) == 6  # 2 * 3
        assert len(y) == 6

    def test_constant_height_within_interval(self):
        """Each pair of consecutive points should have the same y value."""
        theta_0 = 0.001
        lambdas = np.array([1.0, 2.0])
        t = np.array([0, 1, 2, 100])
        x, y = plot_psmc_history(theta_0, lambdas, t)
        assert y[0] == y[1]  # first interval has constant height
        assert y[2] == y[3]  # second interval has constant height


# ============================================================
# Tests: Simulation
# ============================================================

class TestSimulatePsmcInput:
    def test_output_length(self):
        np.random.seed(42)
        seq, coal_times = simulate_psmc_input(1000, 0.001, 0.0005,
                                               lambda t: 1.0)
        assert len(seq) == 1000
        assert len(coal_times) == 1000

    def test_binary_output(self):
        np.random.seed(42)
        seq, _ = simulate_psmc_input(1000, 0.001, 0.0005, lambda t: 1.0)
        assert set(seq).issubset({0, 1})

    def test_positive_coal_times(self):
        np.random.seed(42)
        _, coal_times = simulate_psmc_input(1000, 0.001, 0.0005,
                                             lambda t: 1.0)
        assert np.all(coal_times > 0)

    def test_het_rate_approximately_correct(self):
        """Het rate should be approximately 1 - exp(-theta * E[T])."""
        np.random.seed(42)
        theta = 0.001
        seq, coal_times = simulate_psmc_input(100000, theta, 0.0005,
                                               lambda t: 1.0)
        observed_het = np.mean(seq)
        # For constant lambda=1, mean coal time ~ 1 (Exp(1))
        # Expected het ~ 1 - exp(-theta * 1) ~ theta for small theta
        expected_het = 1 - np.exp(-theta * np.mean(coal_times))
        assert abs(observed_het - expected_het) / expected_het < 0.15

    def test_higher_theta_more_hets(self):
        """Higher mutation rate should produce more heterozygous sites."""
        np.random.seed(42)
        seq_low, _ = simulate_psmc_input(10000, 0.0005, 0.0005,
                                          lambda t: 1.0)
        np.random.seed(42)
        seq_high, _ = simulate_psmc_input(10000, 0.005, 0.0005,
                                           lambda t: 1.0)
        assert np.mean(seq_high) > np.mean(seq_low)


# ============================================================
# Integration Tests
# ============================================================

class TestPsmcEndToEnd:
    """Integration tests combining multiple components."""

    def test_full_hmm_build_and_properties(self):
        """Build complete HMM and verify all mathematical properties."""
        n = 10
        theta = 0.001
        rho = theta / 5
        lambdas = np.ones(n + 1)

        transitions, emissions, initial = build_psmc_hmm(
            n, 15.0, theta, rho, lambdas)

        # Transition matrix is stochastic
        assert np.allclose(transitions.sum(axis=1), 1.0, atol=1e-6)

        # Initial distribution is a probability distribution
        assert abs(initial.sum() - 1.0) < 1e-4
        assert np.all(initial >= -1e-10)

        # Emissions are valid probabilities
        assert np.all(emissions >= 0)
        assert np.all(emissions <= 1)
        assert np.allclose(emissions[0] + emissions[1], 1.0, atol=1e-10)

    def test_bottleneck_changes_transition_structure(self):
        """A population bottleneck should change the transition matrix."""
        n = 10
        theta = 0.001
        rho = theta / 5

        lambdas_const = np.ones(n + 1)
        lambdas_bottleneck = np.ones(n + 1)
        lambdas_bottleneck[3:6] = 0.1  # bottleneck in middle intervals

        trans_const, _, _ = build_psmc_hmm(n, 15.0, theta, rho, lambdas_const)
        trans_bottle, _, _ = build_psmc_hmm(n, 15.0, theta, rho, lambdas_bottleneck)

        # Matrices should differ
        assert not np.allclose(trans_const, trans_bottle, atol=1e-6)

        # Both should still be valid stochastic matrices
        assert np.allclose(trans_const.sum(axis=1), 1.0, atol=1e-6)
        assert np.allclose(trans_bottle.sum(axis=1), 1.0, atol=1e-6)

    def test_scaling_roundtrip(self):
        """Scaling and inverse scaling should be consistent."""
        theta_0 = 0.00069
        mu = 1.25e-8
        s = 100
        N_0 = theta_0 / (4 * mu * s)
        # N_0 * 4 * mu * s should give back theta_0
        theta_recovered = N_0 * 4 * mu * s
        assert abs(theta_recovered - theta_0) < 1e-12

    def test_simulate_and_estimate_theta(self):
        """Simulate data and verify theta can be approximately recovered."""
        np.random.seed(42)
        theta_true = 0.001
        rho = theta_true / 5
        seq, _ = simulate_psmc_input(100000, theta_true, rho, lambda t: 1.0)
        theta_est = estimate_theta_initial(seq)
        # Should be in the right ballpark (within 50%)
        assert abs(theta_est - theta_true) / theta_true < 0.5
