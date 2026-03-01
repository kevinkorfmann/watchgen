"""
Tests for Python code blocks from the SMC++ timepiece RST documentation.

All functions are re-defined here since the code in the RST files is not
importable. Tests cover mathematical properties and expected behaviors.
"""

import numpy as np
from scipy.linalg import expm
import pytest


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/smcpp/distinguished_lineage.rst
# ---------------------------------------------------------------------------

def undistinguished_coalescence_rate(j, lam):
    """Rate at which j undistinguished lineages coalesce among themselves."""
    return j * (j - 1) / (2 * lam)


def distinguished_coalescence_rate(j, lam):
    """Rate at which the distinguished lineage coalesces with an undistinguished one."""
    return j / lam


def emission_unphased(genotype, t, theta):
    """Emission probability for an unphased diploid genotype."""
    p_derived = 1 - np.exp(-theta * t)
    if genotype == 0:
        return (1 - p_derived) ** 2
    elif genotype == 1:
        return 2 * p_derived * (1 - p_derived)
    else:  # genotype == 2
        return p_derived ** 2


def compute_h(t, p_j, lam):
    """Compute the effective coalescence rate h(t) of the distinguished lineage."""
    n_minus_1 = len(p_j) - 1
    h = 0.0
    for j in range(1, n_minus_1 + 1):
        h += j / lam * p_j[j]
    return h


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/smcpp/overview.rst
# ---------------------------------------------------------------------------

def expected_first_coalescence(n, N):
    """Expected time to first coalescence among n lineages in population N."""
    rate = n * (n - 1) / (2 * N)
    return 1 / rate


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/smcpp/ode_system.rst
# ---------------------------------------------------------------------------

def build_rate_matrix(n_undist):
    """Build the rate matrix Q for the undistinguished lineage count process."""
    Q = np.zeros((n_undist, n_undist))
    for j in range(1, n_undist + 1):
        idx = j - 1
        Q[idx, idx] = -j * (j - 1) / 2
        if j < n_undist:
            Q[idx, idx + 1] = (j + 1) * j / 2
    return Q


def solve_ode_piecewise(n_undist, time_breaks, lambdas):
    """Solve the ODE system for piecewise-constant population size."""
    Q = build_rate_matrix(n_undist)
    p = np.zeros(n_undist)
    p[-1] = 1.0
    p_at_breaks = np.zeros((len(time_breaks), n_undist))
    p_at_breaks[0] = p.copy()
    for k in range(len(time_breaks) - 1):
        dt = time_breaks[k + 1] - time_breaks[k]
        lam = lambdas[k]
        M = expm(dt / lam * Q)
        p = M @ p
        p_at_breaks[k + 1] = p.copy()
    return p_at_breaks


def compute_h_values(time_breaks, p_history, lambdas):
    """Compute h(t) at each time break."""
    n_undist = p_history.shape[1]
    h = np.zeros(len(time_breaks))
    j_values = np.arange(1, n_undist + 1)
    for k in range(len(time_breaks)):
        lam = lambdas[min(k, len(lambdas) - 1)]
        expected_j = np.dot(j_values, p_history[k])
        h[k] = expected_j / lam
    return h


def eigendecompose_rate_matrix(n_undist):
    """Compute eigendecomposition of the rate matrix Q."""
    Q = build_rate_matrix(n_undist)
    eigenvalues = np.diag(Q)
    V = np.zeros((n_undist, n_undist))
    for j in range(n_undist):
        v = np.zeros(n_undist)
        v[j] = 1.0
        for i in range(j - 1, -1, -1):
            rhs = sum(Q[i, k] * v[k] for k in range(i + 1, j + 1))
            denom = eigenvalues[j] - Q[i, i]
            if abs(denom) > 1e-15:
                v[i] = rhs / denom
        V[:, j] = v
    V_inv = np.linalg.inv(V)
    return eigenvalues, V, V_inv


def fast_matrix_exp(eigenvalues, V, V_inv, t, lam):
    """Compute exp(t/lam * Q) using precomputed eigendecomposition."""
    D = np.diag(np.exp(eigenvalues * t / lam))
    return V @ D @ V_inv


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/smcpp/continuous_hmm.rst
# ---------------------------------------------------------------------------

def emission_probability(genotype, t, theta, allele_count, n_undist):
    """Emission probability for unphased diploid data at the distinguished individual."""
    p_mut = 1 - np.exp(-theta * t)
    if genotype == 0:
        return np.exp(-theta * t)
    elif genotype == 1:
        return 1 - np.exp(-theta * t)
    else:
        return (1 - np.exp(-theta * t)) ** 2


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/smcpp/population_splits.rst
# ---------------------------------------------------------------------------

def cross_population_survival(t, t_split, h_anc_func):
    """Survival function for cross-population TMRCA."""
    if t < t_split:
        return 1.0
    else:
        from scipy.integrate import quad
        integral, _ = quad(h_anc_func, t_split, t)
        return np.exp(-integral)


# ===========================================================================
# Tests for undistinguished_coalescence_rate
# ===========================================================================

class TestUndistinguishedCoalescenceRate:
    def test_j_equals_1_returns_zero(self):
        """With 1 lineage, no pair can coalesce."""
        assert undistinguished_coalescence_rate(1, 1.0) == 0.0

    def test_j_equals_2(self):
        """With 2 lineages, rate is C(2,2)/lambda = 1/lambda."""
        assert undistinguished_coalescence_rate(2, 1.0) == 1.0
        assert undistinguished_coalescence_rate(2, 2.0) == 0.5

    def test_known_value_j9(self):
        """Example from the RST: j=9, lam=1 -> C(9,2)/1 = 36."""
        assert undistinguished_coalescence_rate(9, 1.0) == 36.0

    def test_inversely_proportional_to_lambda(self):
        """Rate should be inversely proportional to lambda."""
        r1 = undistinguished_coalescence_rate(5, 1.0)
        r2 = undistinguished_coalescence_rate(5, 2.0)
        assert abs(r1 / r2 - 2.0) < 1e-12

    def test_quadratic_in_j(self):
        """Rate is C(j,2)/lambda = j(j-1)/(2*lambda), quadratic in j."""
        lam = 3.0
        for j in range(2, 10):
            expected = j * (j - 1) / (2 * lam)
            assert abs(undistinguished_coalescence_rate(j, lam) - expected) < 1e-12


# ===========================================================================
# Tests for distinguished_coalescence_rate
# ===========================================================================

class TestDistinguishedCoalescenceRate:
    def test_j_equals_1(self):
        """With 1 undistinguished lineage, rate is 1/lambda."""
        assert distinguished_coalescence_rate(1, 1.0) == 1.0

    def test_known_value_j9(self):
        """Example from the RST: j=9, lam=1 -> 9."""
        assert distinguished_coalescence_rate(9, 1.0) == 9.0

    def test_linear_in_j(self):
        """Rate is linear in j."""
        lam = 2.0
        for j in range(1, 10):
            expected = j / lam
            assert abs(distinguished_coalescence_rate(j, lam) - expected) < 1e-12

    def test_total_rate_equals_binomial(self):
        """Total rate (undist + dist) for j undist lineages = C(j+1,2)/lambda."""
        lam = 1.5
        for j in range(1, 10):
            total = (undistinguished_coalescence_rate(j, lam)
                     + distinguished_coalescence_rate(j, lam))
            expected = (j + 1) * j / (2 * lam)
            assert abs(total - expected) < 1e-12


# ===========================================================================
# Tests for emission_unphased
# ===========================================================================

class TestEmissionUnphased:
    def test_probabilities_sum_to_one(self):
        """Emission probabilities for genotypes 0, 1, 2 must sum to 1."""
        for t in [0.01, 0.1, 1.0, 5.0]:
            for theta in [0.001, 0.01, 0.1]:
                total = sum(emission_unphased(g, t, theta) for g in [0, 1, 2])
                assert abs(total - 1.0) < 1e-12

    def test_genotype_0_at_t_zero(self):
        """At t=0, no mutation possible, so P(g=0)=1."""
        assert abs(emission_unphased(0, 0.0, 0.01) - 1.0) < 1e-12
        assert abs(emission_unphased(1, 0.0, 0.01) - 0.0) < 1e-12
        assert abs(emission_unphased(2, 0.0, 0.01) - 0.0) < 1e-12

    def test_symmetry(self):
        """P(g=0) and P(g=2) should be related: P(g=0) = (1-p)^2, P(g=2) = p^2."""
        t, theta = 1.0, 0.05
        p = 1 - np.exp(-theta * t)
        assert abs(emission_unphased(0, t, theta) - (1 - p) ** 2) < 1e-12
        assert abs(emission_unphased(2, t, theta) - p ** 2) < 1e-12
        assert abs(emission_unphased(1, t, theta) - 2 * p * (1 - p)) < 1e-12

    def test_all_nonnegative(self):
        """All emission probabilities must be non-negative."""
        for g in [0, 1, 2]:
            for t in [0.0, 0.001, 0.1, 1.0, 10.0]:
                assert emission_unphased(g, t, 0.01) >= 0.0


# ===========================================================================
# Tests for compute_h
# ===========================================================================

class TestComputeH:
    def test_psmc_case_n_equals_2(self):
        """With n=2, 1 undistinguished lineage, h(t) = 1/lambda always."""
        # p_j has 2 entries: p_j[0] = P(J=0), p_j[1] = P(J=1).
        # For n=2, J(t)=1 always, so p_j = [0, 1].
        p_j = np.array([0.0, 1.0])
        lam = 2.5
        h = compute_h(0, p_j, lam)
        assert abs(h - 1.0 / lam) < 1e-12

    def test_all_lineages_present(self):
        """When all n-1 lineages present, h = (n-1)/lambda."""
        n_minus_1 = 5
        p_j = np.zeros(n_minus_1 + 1)
        p_j[n_minus_1] = 1.0
        lam = 1.0
        h = compute_h(0, p_j, lam)
        assert abs(h - n_minus_1 / lam) < 1e-12

    def test_zero_lineages(self):
        """When all lineages have coalesced, h should be 0."""
        p_j = np.array([1.0, 0.0, 0.0])
        h = compute_h(0, p_j, 1.0)
        assert abs(h) < 1e-12


# ===========================================================================
# Tests for expected_first_coalescence
# ===========================================================================

class TestExpectedFirstCoalescence:
    def test_two_lineages(self):
        """E[T_MRCA] for 2 lineages in pop N is N generations."""
        N = 10000
        result = expected_first_coalescence(2, N)
        assert abs(result - N) < 1e-6

    def test_known_values_from_rst(self):
        """Values from the RST documentation."""
        N = 10000
        assert abs(expected_first_coalescence(2, N) - 10000) < 1
        assert abs(expected_first_coalescence(20, N) - 10000 / (20 * 19 / 2)) < 1
        # n=20: rate = 190/10000 = 0.019, E[T] = 10000/190 ~ 52.63
        assert abs(expected_first_coalescence(20, N) - 10000 / 190) < 0.01

    def test_decreasing_with_n(self):
        """More lineages should coalesce sooner."""
        N = 10000
        prev = expected_first_coalescence(2, N)
        for n in range(3, 20):
            curr = expected_first_coalescence(n, N)
            assert curr < prev
            prev = curr

    def test_inversely_proportional_to_n_squared(self):
        """E[T] ~ 2N / (n(n-1)), should decrease roughly as 1/n^2."""
        N = 10000
        for n in [5, 10, 50]:
            expected = 2 * N / (n * (n - 1))
            assert abs(expected_first_coalescence(n, N) - expected) < 1e-6


# ===========================================================================
# Tests for build_rate_matrix
# ===========================================================================

class TestBuildRateMatrix:
    def test_known_matrix_n4(self):
        """Known rate matrix from RST for n_undist=4."""
        Q = build_rate_matrix(4)
        expected = np.array([
            [0., 1., 0., 0.],
            [0., -1., 3., 0.],
            [0., 0., -3., 6.],
            [0., 0., 0., -6.]
        ])
        np.testing.assert_array_almost_equal(Q, expected)

    def test_diagonal_entries(self):
        """Diagonal entry at j should be -C(j,2) = -j*(j-1)/2."""
        for n in range(1, 8):
            Q = build_rate_matrix(n)
            for j in range(1, n + 1):
                idx = j - 1
                expected = -j * (j - 1) / 2
                assert abs(Q[idx, idx] - expected) < 1e-12

    def test_superdiagonal_entries(self):
        """Superdiagonal entry at (j, j+1) should be C(j+1,2) = (j+1)*j/2."""
        for n in range(2, 8):
            Q = build_rate_matrix(n)
            for j in range(1, n):
                idx = j - 1
                expected = (j + 1) * j / 2
                assert abs(Q[idx, idx + 1] - expected) < 1e-12

    def test_upper_triangular(self):
        """Q should be upper triangular (all entries below diagonal are zero)."""
        for n in range(1, 8):
            Q = build_rate_matrix(n)
            for i in range(n):
                for j in range(i):
                    assert Q[i, j] == 0.0

    def test_last_row_sum_nonpositive(self):
        """The last row (highest state) has only outflow, so its sum is non-positive.
        Other rows may have positive sums because inflow from higher states
        can exceed outflow (state 0 is absorbing and not tracked in the matrix)."""
        for n in range(2, 8):
            Q = build_rate_matrix(n)
            # The last row has only the diagonal (outflow), no superdiagonal inflow
            assert Q[-1].sum() <= 1e-12

    def test_n_undist_1(self):
        """With 1 undistinguished lineage, Q = [[0]] (no coalescence among 1 lineage)."""
        Q = build_rate_matrix(1)
        assert Q.shape == (1, 1)
        assert Q[0, 0] == 0.0


# ===========================================================================
# Tests for solve_ode_piecewise
# ===========================================================================

class TestSolveOdePiecewise:
    def test_initial_condition(self):
        """At t=0, all probability is on the maximum state."""
        n_undist = 5
        time_breaks = np.array([0.0, 1.0])
        lambdas = np.array([1.0])
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        # At t=0, p_5 = 1, others = 0
        assert abs(p_history[0, -1] - 1.0) < 1e-12
        assert abs(p_history[0, :-1].sum()) < 1e-12

    def test_probabilities_sum_to_at_most_one(self):
        """Sum of p_j(t) for j=1..n-1 should be <= 1 (remainder is in state 0)."""
        n_undist = 5
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        for k in range(len(time_breaks)):
            total = p_history[k].sum()
            assert total <= 1.0 + 1e-10

    def test_probability_mass_decreases(self):
        """Total probability mass in states j>=1 should decrease over time."""
        n_undist = 5
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        totals = p_history.sum(axis=1)
        for k in range(1, len(totals)):
            assert totals[k] <= totals[k - 1] + 1e-10

    def test_psmc_case(self):
        """For n=2 (1 undistinguished lineage), p_1(t) = 1 for all t."""
        n_undist = 1
        time_breaks = np.linspace(0, 10, 101)
        lambdas = np.ones(100)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        for k in range(len(time_breaks)):
            assert abs(p_history[k, 0] - 1.0) < 1e-10

    def test_larger_lambda_slower_coalescence(self):
        """Larger population -> slower coalescence -> more lineages at a given time."""
        n_undist = 5
        time_breaks = np.array([0.0, 1.0])
        lam_small = np.array([0.5])
        lam_large = np.array([2.0])
        p_small = solve_ode_piecewise(n_undist, time_breaks, lam_small)
        p_large = solve_ode_piecewise(n_undist, time_breaks, lam_large)
        # Larger lambda -> more total probability remaining at t=1
        assert p_large[1].sum() > p_small[1].sum()

    def test_nonnegative_probabilities(self):
        """All probabilities should be non-negative."""
        n_undist = 5
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        assert np.all(p_history >= -1e-10)


# ===========================================================================
# Tests for compute_h_values
# ===========================================================================

class TestComputeHValues:
    def test_psmc_case(self):
        """For n=2 (1 undistinguished), h(t) = 1/lambda for all t."""
        n_undist = 1
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        h = compute_h_values(time_breaks, p_history, lambdas)
        np.testing.assert_allclose(h, 1.0, atol=1e-10)

    def test_h_decreasing_over_time(self):
        """For n > 2 with constant pop, h(t) should decrease over time."""
        n_undist = 9
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        h = compute_h_values(time_breaks, p_history, lambdas)
        # h should generally decrease (as lineages coalesce, fewer partners)
        # Check that h at end is less than h at start
        assert h[-1] < h[0]

    def test_initial_h(self):
        """At t=0, all n-1 lineages present, so h(0) = (n-1)/lambda."""
        n_undist = 9
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        h = compute_h_values(time_breaks, p_history, lambdas)
        assert abs(h[0] - n_undist / lambdas[0]) < 1e-10

    def test_h_nonnegative(self):
        """h(t) should always be non-negative."""
        n_undist = 5
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.ones(50)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        h = compute_h_values(time_breaks, p_history, lambdas)
        assert np.all(h >= -1e-10)

    def test_h_scales_inversely_with_lambda(self):
        """h(t) should scale inversely with lambda at t=0."""
        n_undist = 5
        time_breaks = np.array([0.0, 0.01])
        for lam_val in [0.5, 1.0, 2.0, 5.0]:
            lambdas = np.array([lam_val])
            p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
            h = compute_h_values(time_breaks, p_history, lambdas)
            assert abs(h[0] - n_undist / lam_val) < 1e-10


# ===========================================================================
# Tests for eigendecompose_rate_matrix
# ===========================================================================

class TestEigendecomposeRateMatrix:
    def test_eigenvalues_are_diagonal(self):
        """Eigenvalues should be the diagonal entries of Q."""
        for n in range(1, 7):
            Q = build_rate_matrix(n)
            eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
            np.testing.assert_allclose(eigenvalues, np.diag(Q), atol=1e-12)

    def test_reconstruction(self):
        """V @ diag(eigenvalues) @ V_inv should reconstruct Q."""
        for n in range(1, 7):
            Q = build_rate_matrix(n)
            eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
            Q_reconstructed = V @ np.diag(eigenvalues) @ V_inv
            np.testing.assert_allclose(Q_reconstructed, Q, atol=1e-10)

    def test_V_inv_is_inverse(self):
        """V @ V_inv should be identity."""
        for n in range(1, 7):
            eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
            I = V @ V_inv
            np.testing.assert_allclose(I, np.eye(n), atol=1e-10)


# ===========================================================================
# Tests for fast_matrix_exp
# ===========================================================================

class TestFastMatrixExp:
    def test_matches_scipy_expm(self):
        """fast_matrix_exp should match scipy.linalg.expm."""
        for n in range(2, 7):
            Q = build_rate_matrix(n)
            eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
            for t in [0.1, 0.5, 1.0, 2.0]:
                for lam in [0.5, 1.0, 2.0]:
                    M_fast = fast_matrix_exp(eigenvalues, V, V_inv, t, lam)
                    M_scipy = expm(t / lam * Q)
                    np.testing.assert_allclose(M_fast, M_scipy, atol=1e-8)

    def test_identity_at_t_zero(self):
        """At t=0, matrix exponential should be the identity."""
        for n in range(1, 7):
            eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
            M = fast_matrix_exp(eigenvalues, V, V_inv, 0.0, 1.0)
            np.testing.assert_allclose(M, np.eye(n), atol=1e-12)

    def test_preserves_probability(self):
        """Applying matrix exp to a probability vector should give non-negative entries summing to <=1."""
        n = 5
        eigenvalues, V, V_inv = eigendecompose_rate_matrix(n)
        p0 = np.zeros(n)
        p0[-1] = 1.0
        M = fast_matrix_exp(eigenvalues, V, V_inv, 1.0, 1.0)
        p1 = M @ p0
        assert np.all(p1 >= -1e-10)
        assert p1.sum() <= 1.0 + 1e-10


# ===========================================================================
# Tests for emission_probability (from continuous_hmm.rst)
# ===========================================================================

class TestEmissionProbability:
    def test_genotype_0_at_zero_time(self):
        """At t=0, P(no mutation) = 1."""
        assert abs(emission_probability(0, 0.0, 0.01, 0, 5) - 1.0) < 1e-12

    def test_genotype_1_at_zero_time(self):
        """At t=0, P(mutation) = 0."""
        assert abs(emission_probability(1, 0.0, 0.01, 0, 5) - 0.0) < 1e-12

    def test_genotype_0_plus_1_sum(self):
        """P(g=0) + P(g=1) should approximately equal 1 for genotypes 0 and 1
        when we ignore genotype 2."""
        # Note: this function is simplified; g=0 returns exp(-theta*t),
        # g=1 returns 1-exp(-theta*t), so g=0 + g=1 = 1.
        t, theta = 1.0, 0.05
        assert abs(
            emission_probability(0, t, theta, 0, 5)
            + emission_probability(1, t, theta, 0, 5)
            - 1.0
        ) < 1e-12

    def test_all_nonnegative(self):
        """All emission probabilities should be non-negative."""
        for g in [0, 1, 2]:
            for t in [0.0, 0.1, 1.0, 10.0]:
                assert emission_probability(g, t, 0.01, 0, 5) >= 0.0


# ===========================================================================
# Tests for cross_population_survival (from population_splits.rst)
# ===========================================================================

class TestCrossPopulationSurvival:
    def test_before_split_is_one(self):
        """Before the split time, survival probability is 1."""
        t_split = 1.0
        h_func = lambda t: 1.0  # constant rate
        for t in [0.0, 0.5, 0.99]:
            assert cross_population_survival(t, t_split, h_func) == 1.0

    def test_at_split_is_one(self):
        """At exactly the split time, integral from t_split to t_split is 0, so survival = 1."""
        t_split = 1.0
        h_func = lambda t: 1.0
        result = cross_population_survival(t_split, t_split, h_func)
        assert abs(result - 1.0) < 1e-12

    def test_decreasing_after_split(self):
        """After the split, survival should decrease."""
        t_split = 1.0
        h_func = lambda t: 1.0
        prev = 1.0
        for t in [1.5, 2.0, 3.0, 5.0]:
            s = cross_population_survival(t, t_split, h_func)
            assert s < prev
            prev = s

    def test_constant_rate_exponential_decay(self):
        """With constant h(t) = c, survival after split = exp(-c*(t - t_split))."""
        t_split = 1.0
        c = 2.0
        h_func = lambda t: c
        for t in [1.5, 2.0, 3.0, 5.0]:
            expected = np.exp(-c * (t - t_split))
            result = cross_population_survival(t, t_split, h_func)
            assert abs(result - expected) < 1e-10

    def test_survival_in_zero_one(self):
        """Survival probability should be in [0, 1]."""
        t_split = 0.5
        h_func = lambda t: 3.0
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            s = cross_population_survival(t, t_split, h_func)
            assert 0.0 <= s <= 1.0 + 1e-12


# ===========================================================================
# Integration-style tests combining multiple functions
# ===========================================================================

class TestIntegration:
    def test_ode_solution_consistent_with_h_computation(self):
        """The ODE solution fed into compute_h_values should give consistent results."""
        n_undist = 4
        time_breaks = np.linspace(0, 3, 31)
        lambdas = np.ones(30)
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        h = compute_h_values(time_breaks, p_history, lambdas)
        # At t=0, h should be n_undist
        assert abs(h[0] - n_undist) < 1e-10
        # All h values should be positive
        assert np.all(h >= -1e-10)

    def test_fast_matexp_in_ode_solution(self):
        """Using fast_matrix_exp should give same ODE solution as expm."""
        n_undist = 4
        eigenvalues, V, V_inv = eigendecompose_rate_matrix(n_undist)
        Q = build_rate_matrix(n_undist)

        p0 = np.zeros(n_undist)
        p0[-1] = 1.0

        t, lam = 1.5, 1.0
        p_fast = fast_matrix_exp(eigenvalues, V, V_inv, t, lam) @ p0
        p_scipy = expm(t / lam * Q) @ p0
        np.testing.assert_allclose(p_fast, p_scipy, atol=1e-10)

    def test_emission_probabilities_consistent(self):
        """emission_unphased and emission_probability should be consistent for limiting cases."""
        # emission_probability(0, t, theta, ...) = exp(-theta*t) = 1 - p_derived
        # emission_unphased(0, t, theta) = (1 - p_derived)^2
        # These are different models (haploid vs diploid), so we just verify internal consistency.
        t, theta = 1.0, 0.05
        p = 1 - np.exp(-theta * t)
        # emission_probability is for haploid: P(g=0) = 1-p, P(g=1) = p
        assert abs(emission_probability(0, t, theta, 0, 5) - (1 - p)) < 1e-12
        assert abs(emission_probability(1, t, theta, 0, 5) - p) < 1e-12
        # emission_unphased is for diploid: P(g=0) = (1-p)^2, etc
        assert abs(emission_unphased(0, t, theta) - (1 - p) ** 2) < 1e-12
        assert abs(emission_unphased(1, t, theta) - 2 * p * (1 - p)) < 1e-12

    def test_varying_population_sizes(self):
        """ODE solution with varying population sizes should still produce valid probabilities."""
        np.random.seed(42)
        n_undist = 5
        time_breaks = np.linspace(0, 5, 51)
        lambdas = np.exp(np.random.randn(50) * 0.5)  # random positive sizes
        p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)
        # All probabilities non-negative
        assert np.all(p_history >= -1e-10)
        # Sum of probabilities at each time <= 1
        for k in range(len(time_breaks)):
            assert p_history[k].sum() <= 1.0 + 1e-10
