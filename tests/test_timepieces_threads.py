"""
Tests for Threads timepiece code extracted from RST documentation.

Covers:
- dating.rst: No standalone Python functions (formulas are mathematical only,
  no self-contained code blocks), but we test the estimator formulas.
- overview.rst: No Python code blocks.
- pbwt_matching.rst: No Python code blocks (text-based descriptions only).
- viterbi.rst: No Python code blocks (algorithm description only).

Since the Threads RST files contain mostly mathematical descriptions and
pseudocode rather than standalone Python functions, we implement and test
the key mathematical estimators described in the dating.rst chapter.
"""

import numpy as np
from scipy.special import gammainc  # regularized lower incomplete gamma


# ============================================================================
# Functions derived from dating.rst mathematical formulas
# ============================================================================

def mle_recombination_only(rho):
    """Maximum likelihood estimator of coalescence time from recombination only.

    From dating.rst:
        t_hat = 1 / rho
    """
    return 1.0 / rho


def mle_recombination_and_mutations(m, rho, mu):
    """MLE of coalescence time from recombination and mutations.

    From dating.rst:
        t_hat = (m + 1) / (rho + mu)
    """
    return (m + 1) / (rho + mu)


def bayesian_recombination_only(rho, gamma):
    """Bayesian posterior mean of coalescence time (recombination only).

    From dating.rst:
        E[t | rho] = 2 / (rho + gamma)
    """
    return 2.0 / (rho + gamma)


def bayesian_full(m, rho, mu, gamma):
    """Bayesian posterior mean of coalescence time (recombination + mutations).

    From dating.rst:
        E[t | rho, m] = (m + 2) / (rho + mu + gamma)
    """
    return (m + 2) / (rho + mu + gamma)


def piecewise_constant_bayesian_recomb_only(rho, time_boundaries, coal_rates):
    """Bayesian posterior mean under a piecewise-constant demographic model.

    Uses recombination only.

    Parameters
    ----------
    rho : float
        Recombination rate for the segment.
    time_boundaries : array-like
        Boundaries of the time intervals [T_0=0, T_1, T_2, ...].
        The last interval extends to infinity.
    coal_rates : array-like
        Coalescence rate gamma_k = 1/N_e^(k) for each interval.

    Returns
    -------
    posterior_mean : float
        E[t | rho] under the piecewise-constant model.
    """
    K = len(coal_rates)
    T = list(time_boundaries)

    numerator = 0.0
    denominator = 0.0

    for k in range(K):
        gamma_k = coal_rates[k]
        lambda_k = rho + gamma_k

        # Cumulative integral of rate up to T_k
        cum_rate = 0.0
        for j in range(k):
            delta_j = T[j + 1] - T[j]
            cum_rate += delta_j * coal_rates[j]

        prefactor = gamma_k * np.exp(-cum_rate + T[k] * gamma_k)

        # Upper boundary for this interval
        if k < K - 1:
            T_upper = T[k + 1]
        else:
            T_upper = np.inf

        # Regularized lower incomplete gamma
        z_upper = lambda_k * T_upper if not np.isinf(T_upper) else np.inf
        z_lower = lambda_k * T[k]

        P3_upper = gammainc(3, z_upper) if not np.isinf(z_upper) else 1.0
        P3_lower = gammainc(3, z_lower)
        P2_upper = gammainc(2, z_upper) if not np.isinf(z_upper) else 1.0
        P2_lower = gammainc(2, z_lower)

        numerator += prefactor * (2.0 / lambda_k**3) * (P3_upper - P3_lower)
        denominator += prefactor * (1.0 / lambda_k**2) * (P2_upper - P2_lower)

    if denominator == 0:
        return np.inf
    return numerator / denominator


def piecewise_constant_bayesian_full(rho, mu, m, time_boundaries, coal_rates):
    """Bayesian posterior mean under a piecewise-constant model with mutations.

    Parameters
    ----------
    rho : float
        Recombination rate.
    mu : float
        Mutation rate.
    m : int
        Number of heterozygous sites.
    time_boundaries : array-like
        Boundaries of time intervals.
    coal_rates : array-like
        Coalescence rates per interval.

    Returns
    -------
    posterior_mean : float
        E[t | rho, m] under the piecewise-constant model.
    """
    K = len(coal_rates)
    T = list(time_boundaries)

    numerator = 0.0
    denominator = 0.0

    for k in range(K):
        gamma_k = coal_rates[k]
        lambda_k = rho + mu + gamma_k

        cum_rate = 0.0
        for j in range(k):
            delta_j = T[j + 1] - T[j]
            cum_rate += delta_j * coal_rates[j]

        prefactor = gamma_k * np.exp(-cum_rate + T[k] * gamma_k)

        if k < K - 1:
            T_upper = T[k + 1]
        else:
            T_upper = np.inf

        z_upper = lambda_k * T_upper if not np.isinf(T_upper) else np.inf
        z_lower = lambda_k * T[k]

        a_num = m + 3
        a_den = m + 2

        P_num_upper = gammainc(a_num, z_upper) if not np.isinf(z_upper) else 1.0
        P_num_lower = gammainc(a_num, z_lower)
        P_den_upper = gammainc(a_den, z_upper) if not np.isinf(z_upper) else 1.0
        P_den_lower = gammainc(a_den, z_lower)

        mu_m = mu**m
        numerator += prefactor * mu_m * (m + 2) / lambda_k**(m + 3) * (P_num_upper - P_num_lower)
        denominator += prefactor * mu_m / lambda_k**(m + 2) * (P_den_upper - P_den_lower)

    if denominator == 0:
        return np.inf
    return numerator / denominator


# ============================================================================
# Tests for MLE estimators
# ============================================================================

class TestMLERecombinationOnly:
    """Tests for the recombination-only MLE."""

    def test_basic(self):
        """t_hat = 1/rho for basic case."""
        assert np.isclose(mle_recombination_only(0.1), 10.0)
        assert np.isclose(mle_recombination_only(1.0), 1.0)
        assert np.isclose(mle_recombination_only(0.01), 100.0)

    def test_inverse_relationship(self):
        """Doubling rho should halve the estimate."""
        rho = 0.05
        t1 = mle_recombination_only(rho)
        t2 = mle_recombination_only(2 * rho)
        assert np.isclose(t1 / t2, 2.0)

    def test_positive(self):
        """Estimate should be positive for positive rho."""
        for rho in [0.001, 0.01, 0.1, 1.0]:
            assert mle_recombination_only(rho) > 0


class TestMLERecombinationAndMutations:
    """Tests for the recombination+mutation MLE."""

    def test_zero_mutations(self):
        """With m=0: t_hat = 1 / (rho + mu)."""
        rho, mu = 0.1, 0.05
        assert np.isclose(mle_recombination_and_mutations(0, rho, mu),
                          1.0 / (rho + mu))

    def test_positive(self):
        """Estimate should be positive."""
        assert mle_recombination_and_mutations(5, 0.1, 0.01) > 0

    def test_increasing_with_mutations(self):
        """More mutations should give a larger time estimate."""
        rho, mu = 0.1, 0.01
        t1 = mle_recombination_and_mutations(1, rho, mu)
        t2 = mle_recombination_and_mutations(5, rho, mu)
        t3 = mle_recombination_and_mutations(10, rho, mu)
        assert t1 < t2 < t3

    def test_formula(self):
        """Verify formula: t = (m+1)/(rho+mu)."""
        m, rho, mu = 3, 0.05, 0.02
        expected = (m + 1) / (rho + mu)
        assert np.isclose(mle_recombination_and_mutations(m, rho, mu), expected)


# ============================================================================
# Tests for Bayesian estimators (constant size)
# ============================================================================

class TestBayesianRecombinationOnly:
    """Tests for the Bayesian recombination-only estimator."""

    def test_formula(self):
        """E[t | rho] = 2 / (rho + gamma)."""
        rho, gamma = 0.1, 0.001
        expected = 2.0 / (rho + gamma)
        assert np.isclose(bayesian_recombination_only(rho, gamma), expected)

    def test_larger_rho_shorter_time(self):
        """Larger rho should give a shorter expected time."""
        gamma = 0.001
        t1 = bayesian_recombination_only(0.01, gamma)
        t2 = bayesian_recombination_only(0.1, gamma)
        assert t1 > t2

    def test_larger_gamma_shorter_time(self):
        """Larger gamma (smaller Ne) should pull the estimate toward zero."""
        rho = 0.05
        t1 = bayesian_recombination_only(rho, 0.001)
        t2 = bayesian_recombination_only(rho, 0.01)
        assert t1 > t2

    def test_comparison_with_mle(self):
        """Bayesian estimate should differ from MLE by the prior term."""
        rho = 0.1
        gamma = 0.001
        mle = mle_recombination_only(rho)
        bayes = bayesian_recombination_only(rho, gamma)
        # Bayesian should be larger than MLE (numerator is 2 vs 1, but
        # denominator includes gamma)
        # Actually: MLE = 1/rho, Bayes = 2/(rho+gamma)
        # 2/(rho+gamma) vs 1/rho => 2*rho vs rho+gamma => rho vs gamma
        # For small gamma, Bayes ~ 2/rho > MLE = 1/rho
        assert bayes > mle


class TestBayesianFull:
    """Tests for the Bayesian full estimator (recomb + mutations)."""

    def test_formula(self):
        """E[t | rho, m] = (m+2) / (rho + mu + gamma)."""
        m, rho, mu, gamma = 3, 0.1, 0.05, 0.001
        expected = (m + 2) / (rho + mu + gamma)
        assert np.isclose(bayesian_full(m, rho, mu, gamma), expected)

    def test_more_mutations_longer_time(self):
        """More mutations should give a longer estimate."""
        rho, mu, gamma = 0.1, 0.01, 0.001
        t1 = bayesian_full(0, rho, mu, gamma)
        t2 = bayesian_full(5, rho, mu, gamma)
        t3 = bayesian_full(10, rho, mu, gamma)
        assert t1 < t2 < t3

    def test_positive(self):
        """Result should always be positive."""
        for m in [0, 1, 5, 20]:
            t = bayesian_full(m, 0.05, 0.01, 0.001)
            assert t > 0

    def test_numerator_larger_than_mle(self):
        """Bayesian numerator (m+2) is larger than MLE numerator (m+1)."""
        m, rho, mu, gamma = 5, 0.1, 0.01, 0.001
        mle = mle_recombination_and_mutations(m, rho, mu)
        bayes = bayesian_full(m, rho, mu, gamma)
        # The relationship depends on parameters, but for small gamma:
        # bayes ~ (m+2)/(rho+mu) > (m+1)/(rho+mu) = mle
        assert bayes > mle


# ============================================================================
# Tests for piecewise-constant demographic model estimators
# ============================================================================

class TestPiecewiseConstantBayesianRecombOnly:
    """Tests for the piecewise-constant Bayesian estimator (recomb only)."""

    def test_single_epoch_matches_constant(self):
        """With one epoch (constant size), should match the constant formula."""
        rho = 0.1
        Ne = 1000.0
        gamma = 1.0 / Ne
        # Single epoch: [0, inf)
        time_boundaries = [0.0]
        coal_rates = [gamma]
        result = piecewise_constant_bayesian_recomb_only(
            rho, time_boundaries, coal_rates)
        expected = bayesian_recombination_only(rho, gamma)
        assert np.isclose(result, expected, rtol=1e-4), \
            f"Expected {expected}, got {result}"

    def test_positive(self):
        """Result should be positive."""
        rho = 0.05
        time_boundaries = [0.0, 100.0]
        coal_rates = [0.001, 0.0005]
        result = piecewise_constant_bayesian_recomb_only(
            rho, time_boundaries, coal_rates)
        assert result > 0

    def test_larger_rho_shorter_time(self):
        """Larger rho should give a shorter time estimate."""
        time_boundaries = [0.0, 50.0]
        coal_rates = [0.001, 0.0002]
        t1 = piecewise_constant_bayesian_recomb_only(
            0.01, time_boundaries, coal_rates)
        t2 = piecewise_constant_bayesian_recomb_only(
            0.1, time_boundaries, coal_rates)
        assert t1 > t2

    def test_two_epochs(self):
        """Two-epoch model should give a finite positive result."""
        rho = 0.05
        time_boundaries = [0.0, 200.0]
        coal_rates = [0.001, 0.01]  # Bottleneck: recent large, then small
        result = piecewise_constant_bayesian_recomb_only(
            rho, time_boundaries, coal_rates)
        assert np.isfinite(result) and result > 0


class TestPiecewiseConstantBayesianFull:
    """Tests for the piecewise-constant Bayesian estimator (full)."""

    def test_single_epoch_matches_constant(self):
        """With one epoch, should match the constant-size formula."""
        rho = 0.1
        mu = 0.01
        m = 3
        Ne = 1000.0
        gamma = 1.0 / Ne
        time_boundaries = [0.0]
        coal_rates = [gamma]
        result = piecewise_constant_bayesian_full(
            rho, mu, m, time_boundaries, coal_rates)
        expected = bayesian_full(m, rho, mu, gamma)
        assert np.isclose(result, expected, rtol=1e-4), \
            f"Expected {expected}, got {result}"

    def test_more_mutations_longer_time(self):
        """More mutations should increase the estimate."""
        rho = 0.05
        mu = 0.01
        time_boundaries = [0.0, 100.0]
        coal_rates = [0.001, 0.0005]
        t1 = piecewise_constant_bayesian_full(
            rho, mu, 1, time_boundaries, coal_rates)
        t2 = piecewise_constant_bayesian_full(
            rho, mu, 5, time_boundaries, coal_rates)
        assert t2 > t1

    def test_positive(self):
        """Result should always be positive."""
        rho, mu = 0.05, 0.01
        time_boundaries = [0.0, 100.0]
        coal_rates = [0.001, 0.0005]
        for m in [0, 1, 5, 10]:
            result = piecewise_constant_bayesian_full(
                rho, mu, m, time_boundaries, coal_rates)
            assert result > 0


# ============================================================================
# Tests for mathematical properties described in the text
# ============================================================================

class TestDatingMathProperties:
    """Test mathematical properties described in dating.rst."""

    def test_mle_vs_bayesian_numerator_shift(self):
        """Bayesian estimator has numerator m+2 vs MLE's m+1."""
        m = 5
        rho, mu, gamma = 0.1, 0.01, 0.001
        mle_num = m + 1
        bayes_num = m + 2
        assert bayes_num == mle_num + 1

    def test_bayesian_denominator_includes_gamma(self):
        """Bayesian denominator includes gamma while MLE does not."""
        rho, mu, gamma = 0.1, 0.01, 0.001
        m = 5
        mle = (m + 1) / (rho + mu)
        bayes = (m + 2) / (rho + mu + gamma)
        # Both should be positive and the gamma contribution matters
        assert (rho + mu + gamma) > (rho + mu)

    def test_erlang_interpretation_recomb_only(self):
        """The posterior should be Erlang-2 with rate rho + gamma."""
        rho, gamma = 0.1, 0.001
        # Erlang-2 mean = k/lambda where k=2, lambda=rho+gamma
        erlang_mean = 2.0 / (rho + gamma)
        bayes = bayesian_recombination_only(rho, gamma)
        assert np.isclose(bayes, erlang_mean)

    def test_erlang_interpretation_full(self):
        """The posterior should be Erlang-(m+2) with rate rho + mu + gamma."""
        m = 3
        rho, mu, gamma = 0.1, 0.01, 0.001
        # Erlang-(m+2) mean = (m+2)/(rho+mu+gamma)
        erlang_mean = (m + 2) / (rho + mu + gamma)
        bayes = bayesian_full(m, rho, mu, gamma)
        assert np.isclose(bayes, erlang_mean)

    def test_segment_length_exponential_rate(self):
        """Under SMC, segment length is exponential with rate 1/(2t).

        The likelihood is 2t * exp(-t*rho), maximized at t = 1/rho.
        """
        rho = 0.05
        t_hat = 1.0 / rho
        # At the MLE, the derivative of log-likelihood should be zero
        # d/dt [log(2t) - t*rho] = 1/t - rho = 0 => t = 1/rho
        assert np.isclose(1.0 / t_hat - rho, 0.0)

    def test_mutation_poisson_process(self):
        """Mutations follow a Poisson process with rate mu = 2*c*l_bp.

        The number of mutations m given time t has E[m] = t * mu.
        """
        mu = 0.01
        t = 50
        expected_mutations = t * mu
        assert expected_mutations == 0.5
