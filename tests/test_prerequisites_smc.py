"""Tests for code examples in docs/prerequisites/smc.rst.

This module extracts and tests every Python code block from the SMC
(Sequentially Markov Coalescent) prerequisites documentation.

Code blocks tested:
1. sample_smc_pruning_point -- Uniform pruning point on a marginal tree
2. smc_branch_transition -- SMC branch-level transition matrix
3. psmc_transition_density -- PSMC continuous transition density
4. psmc_transition_cdf  -- PSMC cumulative distribution function
"""

import numpy as np
import pytest
from pathlib import Path


def adaptive_simpson(f, a, b, atol=1e-11, max_depth=20):
    """Integrate a scalar function without sharing implementation formulas."""
    fa, fb = f(a), f(b)
    midpoint = (a + b) / 2
    fm = f(midpoint)
    whole = (b - a) * (fa + 4 * fm + fb) / 6

    def refine(left, right, f_left, f_mid, f_right, estimate, tol, depth):
        mid = (left + right) / 2
        left_mid = (left + mid) / 2
        right_mid = (mid + right) / 2
        f_left_mid = f(left_mid)
        f_right_mid = f(right_mid)
        left_estimate = (mid - left) * (f_left + 4 * f_left_mid + f_mid) / 6
        right_estimate = (right - mid) * (f_mid + 4 * f_right_mid + f_right) / 6
        error = left_estimate + right_estimate - estimate
        if depth == 0 or abs(error) <= 15 * tol:
            return left_estimate + right_estimate + error / 15
        return refine(
            left, mid, f_left, f_left_mid, f_mid, left_estimate, tol / 2, depth - 1
        ) + refine(
            mid, right, f_mid, f_right_mid, f_right, right_estimate, tol / 2, depth - 1
        )

    return refine(a, b, fa, fm, fb, whole, atol, max_depth)


def execute_rst_python_blocks(path):
    """Execute Python code blocks exactly as published in an RST file."""
    lines = Path(path).read_text().splitlines()
    namespace = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() != ".. code-block:: python":
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        indent = len(lines[i]) - len(lines[i].lstrip())
        block = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) < indent:
                break
            block.append(line[indent:] if line.strip() else "")
            i += 1
        exec("\n".join(block), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Code block 1: sample_smc_pruning_point
# ---------------------------------------------------------------------------

def sample_smc_pruning_point(tree_branches):
    """Sample a point uniformly on total branch length."""
    branch_lengths = np.array([u - l for _, _, l, u in tree_branches])
    if len(branch_lengths) == 0 or np.any(branch_lengths <= 0):
        raise ValueError("tree must contain positive-length branches")
    probs = branch_lengths / branch_lengths.sum()
    idx = np.random.choice(len(tree_branches), p=probs)
    _, _, lower, upper = tree_branches[idx]
    return tree_branches[idx], np.random.uniform(lower, upper)


# Example tree used in the documentation
EXAMPLE_TREE_BRANCHES = [
    (0, 4, 0.0, 0.3),
    (1, 4, 0.0, 0.3),
    (2, 5, 0.0, 0.7),
    (3, 5, 0.0, 0.7),
    (4, 6, 0.3, 1.5),
    (5, 6, 0.7, 1.5),
]


class TestSMCPruningPoint:

    def test_returns_branch_and_time(self):
        np.random.seed(42)
        branch, pruning_time = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
        assert branch in EXAMPLE_TREE_BRANCHES
        assert branch[2] <= pruning_time <= branch[3]

    def test_recomb_branch_is_valid(self):
        """Verify that the recombination branch is one of the tree branches."""
        np.random.seed(42)
        branch, _ = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
        assert branch in EXAMPLE_TREE_BRANCHES

    def test_recomb_time_within_branch(self):
        """Verify that recombination time falls within the chosen branch's interval."""
        np.random.seed(42)
        branch, pruning_time = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
        assert branch[2] <= pruning_time <= branch[3]

    def test_branch_selection_proportional_to_length(self):
        """Verify branches are selected roughly proportional to their length."""
        np.random.seed(123)
        counts = {i: 0 for i in range(len(EXAMPLE_TREE_BRANCHES))}
        n_trials = 10000
        for _ in range(n_trials):
            branch, _ = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
            counts[EXAMPLE_TREE_BRANCHES.index(branch)] += 1

        # Compute expected proportions from branch lengths
        lengths = [u - l for _, _, l, u in EXAMPLE_TREE_BRANCHES]
        total = sum(lengths)
        expected_probs = [bl / total for bl in lengths]

        total_results = sum(counts.values())
        for i, expected_p in enumerate(expected_probs):
            observed_p = counts[i] / total_results
            assert abs(observed_p - expected_p) < 0.05, (
                f"Branch {i}: expected ~{expected_p:.3f}, got {observed_p:.3f}"
            )

    def test_reproducibility_with_seed(self):
        """Verify that results are reproducible when using the same random seed."""
        np.random.seed(42)
        result1 = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
        np.random.seed(42)
        result2 = sample_smc_pruning_point(EXAMPLE_TREE_BRANCHES)
        assert result1 == result2

    @pytest.mark.parametrize("branches", [[], [(0, 1, 1.0, 1.0)]])
    def test_rejects_invalid_trees(self, branches):
        with pytest.raises(ValueError):
            sample_smc_pruning_point(branches)


# ---------------------------------------------------------------------------
# Code block 2: smc_branch_transition
# ---------------------------------------------------------------------------

def smc_branch_transition(tau, p, rho, n_branches):
    """Compute SINGER's stay-or-switch transition between branch states.

    A[i,j] = (1 - r_i) * delta(i,j) + r_i * q_j / sum(q)

    Parameters
    ----------
    tau : ndarray of shape (K,)
    p : ndarray of shape (K,)
    rho : float
    n_branches : int

    Returns
    -------
    T : ndarray of shape (K, K)
    """
    tau = np.asarray(tau, dtype=float)
    p = np.asarray(p, dtype=float)
    K = n_branches
    if tau.shape != (K,) or p.shape != (K,):
        raise ValueError("tau and p must both have length n_branches")
    if np.any(tau < 0) or rho < 0 or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("require tau >= 0, rho >= 0, and normalized p >= 0")
    if rho == 0:
        return np.eye(K)
    r = 1 - np.exp(-rho / 2 * tau)
    q = r * p
    q_sum = q.sum()
    if q_sum <= 0:
        raise ValueError("at least one positive-probability branch must have tau > 0")
    T = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                T[i, j] = (1 - r[i]) + r[i] * q[j] / q_sum
            else:
                T[i, j] = r[i] * q[j] / q_sum
    if not np.allclose(T.sum(axis=1), 1.0):
        raise RuntimeError("transition rows do not sum to one")
    return T


class TestSMCBranchTransition:
    """Tests for the smc_branch_transition function."""

    def test_basic_example_runs(self):
        """Verify that the documented example runs without error."""
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        assert T.shape == (K, K)

    def test_rows_sum_to_one(self):
        """Verify that each row of the transition matrix sums to 1 (stochastic matrix)."""
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        np.testing.assert_allclose(T.sum(axis=1), np.ones(K), atol=1e-12)

    def test_all_entries_non_negative(self):
        """Verify that all transition probabilities are non-negative."""
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        assert np.all(T >= 0)

    def test_diagonal_dominance(self):
        """Verify that diagonal entries are the largest in each row (stay > switch)."""
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        for i in range(K):
            assert T[i, i] == T[i, :].max(), (
                f"Row {i}: diagonal {T[i, i]:.4f} is not the maximum"
            )

    def test_stationary_distribution(self):
        """Verify that p is the stationary distribution of the transition matrix.

        The stationary distribution pi satisfies pi @ T = pi. For the SMC
        transition with q_j = r_j * p_j, the stationary distribution is p.
        """
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        result = p @ T
        np.testing.assert_allclose(result, p, atol=1e-12)

    def test_off_diagonal_column_structure(self):
        """Verify that off-diagonal entries in each column share a common factor q_j/sum(q).

        For column j, all off-diagonal entries T[i,j] (i != j) should equal
        r[i] * q[j] / q_sum, meaning T[i,j] / r[i] is the same for all i != j.
        """
        K = 5
        tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])
        p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        rho = 0.5
        T = smc_branch_transition(tau, p, rho, K)
        r = 1 - np.exp(-rho / 2 * tau)
        for j in range(K):
            ratios = []
            for i in range(K):
                if i != j:
                    ratios.append(T[i, j] / r[i])
            # All ratios should be equal (q_j / q_sum)
            np.testing.assert_allclose(ratios, ratios[0], atol=1e-12)

    def test_zero_recombination_rate(self):
        """At rho=0 the transition is exactly the identity."""
        K = 3
        tau = np.array([0.5, 1.0, 1.5])
        p = np.array([0.4, 0.3, 0.3])
        rho = 0.0
        T = smc_branch_transition(tau, p, rho, K)
        np.testing.assert_array_equal(T, np.eye(K))

    def test_detailed_balance(self):
        """The constructed kernel is reversible with respect to p."""
        tau = np.array([0.2, 0.7, 1.4, 2.0])
        p = np.array([0.1, 0.2, 0.3, 0.4])
        T = smc_branch_transition(tau, p, 0.8, len(tau))
        np.testing.assert_allclose(p[:, None] * T, p[None, :] * T.T, atol=1e-12)

    @pytest.mark.parametrize("tau,p,rho,k", [
        ([1.0], [1.0], -0.1, 1),
        ([-1.0], [1.0], 0.1, 1),
        ([1.0, 2.0], [1.0], 0.1, 2),
        ([1.0, 2.0], [0.2, 0.2], 0.1, 2),
        ([0.0, 0.0], [0.5, 0.5], 0.1, 2),
    ])
    def test_rejects_invalid_parameters(self, tau, p, rho, k):
        with pytest.raises(ValueError):
            smc_branch_transition(tau, p, rho, k)

    def test_large_recombination_rate(self):
        """With very high rho, T[i,j] should approach q_j / sum(q) for all i."""
        K = 4
        tau = np.array([1.0, 1.0, 1.0, 1.0])
        p = np.array([0.4, 0.3, 0.2, 0.1])
        rho = 1000.0  # very high recombination rate
        T = smc_branch_transition(tau, p, rho, K)
        r = 1 - np.exp(-rho / 2 * tau)
        q = r * p
        q_sum = q.sum()
        expected_row = q / q_sum
        for i in range(K):
            np.testing.assert_allclose(T[i, :], expected_row, atol=1e-6)

    def test_various_parameter_sizes(self):
        """Test with different numbers of branches to ensure generality."""
        for K in [2, 3, 7, 15]:
            tau = np.random.uniform(0.1, 2.0, K)
            p = np.random.dirichlet(np.ones(K))
            rho = np.random.uniform(0.01, 2.0)
            T = smc_branch_transition(tau, p, rho, K)
            assert T.shape == (K, K)
            np.testing.assert_allclose(T.sum(axis=1), np.ones(K), atol=1e-12)
            assert np.all(T >= 0)


# ---------------------------------------------------------------------------
# Code block 3: psmc_transition_density
# ---------------------------------------------------------------------------

def psmc_transition_density(t, s, rho):
    """PSMC transition density q_rho(t | s).

    Computes the probability density of the new coalescence time t,
    given that the previous coalescence time was s.
    Returns only the continuous part (not the point mass at t = s).

    Parameters
    ----------
    t : float or ndarray
    s : float
    rho : float

    Returns
    -------
    density : float or ndarray
    """
    p_no_recomb = np.exp(-rho * s)
    p_recomb = -np.expm1(-rho * s)
    t = np.asarray(t, dtype=float)
    if s <= 0 or rho < 0 or np.any(t < 0):
        raise ValueError("require t >= 0, s > 0, and rho >= 0")
    density = np.zeros_like(t)
    mask_lt = t < s
    density[mask_lt] = (p_recomb / s) * (-np.expm1(-t[mask_lt]))
    mask_ge = t >= s
    density[mask_ge] = (p_recomb / s) * (
        np.exp(-(t[mask_ge] - s)) - np.exp(-t[mask_ge])
    )
    return density


class TestPSMCTransitionDensity:
    """Tests for the psmc_transition_density function."""

    def test_basic_example_runs(self):
        """Verify that the documented example runs correctly."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.01, 4.0, 200)
        densities = psmc_transition_density(t_values, s, rho)
        assert densities.shape == t_values.shape
        assert np.all(densities >= 0)

    def test_density_non_negative(self):
        """Verify that the density is non-negative everywhere."""
        for s in [0.5, 1.0, 2.0]:
            for rho in [0.1, 0.5, 1.0, 2.0]:
                t_values = np.linspace(0.001, 10.0, 500)
                densities = psmc_transition_density(t_values, s, rho)
                assert np.all(densities >= -1e-15), (
                    f"Negative density for s={s}, rho={rho}"
                )

    def test_continuous_density_integrates_to_recomb_prob(self):
        """The continuous part should integrate to (1 - exp(-rho*s)),
        the probability that recombination occurs."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.001, 20.0, 5000)
        densities = psmc_transition_density(t_values, s, rho)
        integral = np.trapezoid(densities, t_values)
        expected = 1 - np.exp(-rho * s)
        np.testing.assert_allclose(integral, expected, atol=0.01)

    def test_total_probability_is_one(self):
        """Continuous integral + point mass at t=s should sum to 1."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.001, 30.0, 10000)
        densities = psmc_transition_density(t_values, s, rho)
        integral = np.trapezoid(densities, t_values)
        point_mass = np.exp(-rho * s)
        total = integral + point_mass
        np.testing.assert_allclose(total, 1.0, atol=0.02)

    def test_density_at_zero(self):
        """At t=0, the density should be 0 (since 1 - exp(0) = 0)."""
        s = 1.0
        rho = 0.5
        density = psmc_transition_density(np.array([0.0]), s, rho)
        np.testing.assert_allclose(density[0], 0.0, atol=1e-12)

    def test_continuity_at_s(self):
        """The density should be continuous at t = s (both cases give the same value)."""
        s = 1.0
        rho = 0.5
        eps = 1e-8
        d_below = psmc_transition_density(np.array([s - eps]), s, rho)[0]
        d_above = psmc_transition_density(np.array([s + eps]), s, rho)[0]
        np.testing.assert_allclose(d_below, d_above, atol=1e-5)

    def test_density_decays_for_large_t(self):
        """The density should decay to zero for large t."""
        s = 1.0
        rho = 0.5
        large_t = np.array([50.0, 100.0, 200.0])
        densities = psmc_transition_density(large_t, s, rho)
        np.testing.assert_allclose(densities, 0.0, atol=1e-10)

    def test_peak_location(self):
        """The documented example notes the peak location; verify it is near t=s."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.01, 4.0, 200)
        densities = psmc_transition_density(t_values, s, rho)
        peak_t = t_values[np.argmax(densities)]
        # The peak should be near s (within a reasonable range)
        assert 0.1 < peak_t < 3.0

    def test_various_parameters(self):
        """Verify density properties across a range of (s, rho) values."""
        for s in [0.3, 0.5, 1.0, 2.0, 5.0]:
            for rho in [0.1, 0.5, 1.0, 3.0]:
                t_values = np.linspace(0.001, max(30.0, 5 * s), 5000)
                densities = psmc_transition_density(t_values, s, rho)
                assert np.all(densities >= -1e-15)
                integral = np.trapezoid(densities, t_values)
                expected = 1 - np.exp(-rho * s)
                np.testing.assert_allclose(integral, expected, atol=0.05)


# ---------------------------------------------------------------------------
# Code block 4: psmc_transition_cdf
# ---------------------------------------------------------------------------

def psmc_transition_cdf(t, s, rho):
    """PSMC transition CDF Q_rho(t | s).

    Computes the cumulative distribution function of the new coalescence
    time t, given the previous coalescence time s. This CDF includes the
    point mass at t = s (no recombination).

    Parameters
    ----------
    t : float or ndarray
    s : float
    rho : float

    Returns
    -------
    cdf : float or ndarray
    """
    p_no_recomb = np.exp(-rho * s)
    p_recomb = -np.expm1(-rho * s)
    t = np.asarray(t, dtype=float)
    if s <= 0 or rho < 0 or np.any(t < 0):
        raise ValueError("require t >= 0, s > 0, and rho >= 0")
    cdf = np.zeros_like(t)
    mask_lt = t < s
    cdf[mask_lt] = (p_recomb / s) * (
        t[mask_lt] + np.expm1(-t[mask_lt])
    )
    mask_ge = t >= s
    cdf[mask_ge] = (p_recomb / s) * (
        s - np.exp(-(t[mask_ge] - s)) + np.exp(-t[mask_ge])
    ) + p_no_recomb
    return cdf


class TestPSMCTransitionCDF:
    """Tests for the psmc_transition_cdf function."""

    def test_documented_example(self):
        """Verify the documented CDF values at t=10 and t=100 approach 1."""
        s = 1.0
        rho = 0.5
        cdf_10 = psmc_transition_cdf(np.array([10.0]), s, rho)[0]
        cdf_100 = psmc_transition_cdf(np.array([100.0]), s, rho)[0]
        np.testing.assert_allclose(cdf_10, 1.0, atol=1e-4)
        np.testing.assert_allclose(cdf_100, 1.0, atol=1e-10)

    def test_cdf_at_zero(self):
        """CDF at t=0 should be 0."""
        s = 1.0
        rho = 0.5
        cdf_0 = psmc_transition_cdf(np.array([0.0]), s, rho)[0]
        np.testing.assert_allclose(cdf_0, 0.0, atol=1e-12)

    def test_cdf_approaches_one(self):
        """CDF should approach 1 as t -> infinity."""
        for s in [0.5, 1.0, 2.0]:
            for rho in [0.1, 0.5, 1.0, 2.0]:
                cdf_large = psmc_transition_cdf(np.array([100.0]), s, rho)[0]
                np.testing.assert_allclose(cdf_large, 1.0, atol=1e-6)

    def test_cdf_monotonically_increasing(self):
        """CDF must be monotonically non-decreasing."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.0, 10.0, 1000)
        cdf_values = psmc_transition_cdf(t_values, s, rho)
        diffs = np.diff(cdf_values)
        assert np.all(diffs >= -1e-12), "CDF is not monotonically increasing"

    def test_cdf_jump_at_s(self):
        """CDF should jump by exp(-rho*s) at t = s (the point mass)."""
        s = 1.0
        rho = 0.5
        eps = 1e-10
        cdf_below = psmc_transition_cdf(np.array([s - eps]), s, rho)[0]
        cdf_at = psmc_transition_cdf(np.array([s]), s, rho)[0]
        jump = cdf_at - cdf_below
        expected_jump = np.exp(-rho * s)
        np.testing.assert_allclose(jump, expected_jump, atol=0.01)

    def test_cdf_between_zero_and_one(self):
        """CDF values should always be between 0 and 1."""
        s = 1.0
        rho = 0.5
        t_values = np.linspace(0.0, 20.0, 1000)
        cdf_values = psmc_transition_cdf(t_values, s, rho)
        assert np.all(cdf_values >= -1e-12)
        assert np.all(cdf_values <= 1.0 + 1e-12)

    def test_cdf_consistent_with_density(self):
        """Verify that the CDF is the integral of the density.

        Numerically integrate the density and compare with the CDF.
        """
        s = 1.0
        rho = 0.5
        t_test = 0.5  # test point before s
        t_grid = np.linspace(0.001, t_test, 2000)
        densities = psmc_transition_density(t_grid, s, rho)
        numerical_cdf = np.trapezoid(densities, t_grid)
        analytical_cdf = psmc_transition_cdf(np.array([t_test]), s, rho)[0]
        np.testing.assert_allclose(numerical_cdf, analytical_cdf, atol=0.005)

        # Test point after s
        t_test2 = 2.0
        t_grid2 = np.linspace(0.001, t_test2, 5000)
        densities2 = psmc_transition_density(t_grid2, s, rho)
        numerical_cdf2 = np.trapezoid(densities2, t_grid2)
        # The analytical CDF at t >= s includes the point mass
        analytical_cdf2 = psmc_transition_cdf(np.array([t_test2]), s, rho)[0]
        # The numerical integral does not include the point mass, so subtract it
        point_mass = np.exp(-rho * s)
        np.testing.assert_allclose(
            numerical_cdf2 + point_mass, analytical_cdf2, atol=0.01
        )

    def test_various_parameters(self):
        """Verify CDF properties across a range of (s, rho) values."""
        np.random.seed(99)
        for s in [0.2, 0.5, 1.0, 3.0]:
            for rho in [0.1, 0.5, 1.0, 5.0]:
                t_values = np.linspace(0.0, 30.0, 500)
                cdf_values = psmc_transition_cdf(t_values, s, rho)
                # Non-decreasing
                assert np.all(np.diff(cdf_values) >= -1e-10)
                # Starts near 0
                np.testing.assert_allclose(cdf_values[0], 0.0, atol=1e-10)
                # Ends near 1
                np.testing.assert_allclose(cdf_values[-1], 1.0, atol=0.01)


class TestPSMCIndependentValidation:
    """Independent quadrature, calculus, limit, and simulation checks."""

    @pytest.mark.parametrize("s,rho", [(0.2, 0.01), (1.0, 0.5), (3.0, 4.0)])
    def test_adaptive_quadrature_plus_atom_is_one(self, s, rho):
        f = lambda x: float(psmc_transition_density(np.array([x]), s, rho)[0])
        # Split at the density's piecewise boundary. At s + 50 the omitted
        # exponential tail is below 2e-22 times its finite prefactor.
        continuous = adaptive_simpson(f, 0, s) + adaptive_simpson(f, s, s + 50)
        assert continuous + np.exp(-rho * s) == pytest.approx(1.0, abs=1e-10)

    @pytest.mark.parametrize("s,rho,t", [(1.0, 0.5, 0.3), (1.0, 0.5, 2.0)])
    def test_cdf_derivative_matches_density_away_from_atom(self, s, rho, t):
        h = 1e-5
        derivative = (
            psmc_transition_cdf(np.array([t + h]), s, rho)[0]
            - psmc_transition_cdf(np.array([t - h]), s, rho)[0]
        ) / (2 * h)
        density = psmc_transition_density(np.array([t]), s, rho)[0]
        assert derivative == pytest.approx(density, rel=1e-7, abs=1e-9)

    def test_zero_rho_is_a_point_mass_at_s(self):
        s = 1.7
        np.testing.assert_array_equal(
            psmc_transition_density(np.array([0.2, s, 4.0]), s, 0),
            np.zeros(3),
        )
        np.testing.assert_array_equal(
            psmc_transition_cdf(np.array([0.2, s - 1e-9, s, 4.0]), s, 0),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )

    def test_monte_carlo_sampler_matches_analytic_cdf_and_mean(self):
        """Generate from the defining mixture without using either formula."""
        rng = np.random.default_rng(20260901)
        s, rho, n = 1.3, 0.8, 200_000
        recomb = rng.random(n) < -np.expm1(-rho * s)
        draws = np.full(n, s)
        count = recomb.sum()
        draws[recomb] = rng.uniform(0, s, count) + rng.exponential(1, count)
        for x in [0.2, 0.9, s, 2.5, 5.0]:
            empirical = np.mean(draws <= x)
            analytic = psmc_transition_cdf(np.array([x]), s, rho)[0]
            assert empirical == pytest.approx(analytic, abs=0.004)
        p_recomb = -np.expm1(-rho * s)
        expected_mean = (1 - p_recomb) * s + p_recomb * (s / 2 + 1)
        assert draws.mean() == pytest.approx(expected_mean, abs=0.008)

    @pytest.mark.parametrize("s,rho,t", [(0, 1, 1), (1, -1, 1), (1, 1, -1)])
    def test_invalid_domains_are_rejected(self, s, rho, t):
        with pytest.raises(ValueError):
            psmc_transition_density(np.array([t]), s, rho)
        with pytest.raises(ValueError):
            psmc_transition_cdf(np.array([t]), s, rho)


class TestPublishedCodeParity:
    def test_rst_blocks_execute_and_match_validated_functions(self):
        rst = Path(__file__).parents[1] / "docs/prerequisites/smc.rst"
        published = execute_rst_python_blocks(rst)
        assert "sample_smc_pruning_point" in published
        assert "smc_branch_transition" in published
        assert "psmc_transition_density" in published
        assert "psmc_transition_cdf" in published
        tau = np.array([0.2, 0.5, 1.0])
        p = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(
            published["smc_branch_transition"](tau, p, 0.7, 3),
            smc_branch_transition(tau, p, 0.7, 3),
        )
        grid = np.array([0.0, 0.4, 1.0, 2.0])
        np.testing.assert_allclose(
            published["psmc_transition_density"](grid, 1.0, 0.7),
            psmc_transition_density(grid, 1.0, 0.7),
        )
        np.testing.assert_allclose(
            published["psmc_transition_cdf"](grid, 1.0, 0.7),
            psmc_transition_cdf(grid, 1.0, 0.7),
        )
