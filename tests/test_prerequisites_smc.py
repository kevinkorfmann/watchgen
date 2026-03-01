"""Tests for code examples in docs/prerequisites/smc.rst.

This module extracts and tests every Python code block from the SMC
(Sequentially Markov Coalescent) prerequisites documentation.

Code blocks tested:
1. smc_transition       -- SMC transition on a marginal tree
2. smc_branch_transition -- SMC branch-level transition matrix
3. psmc_transition_density -- PSMC continuous transition density
4. psmc_transition_cdf  -- PSMC cumulative distribution function
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Code block 1: smc_transition
# ---------------------------------------------------------------------------

def smc_transition(tree_branches, recomb_rate, coal_rates):
    """Compute SMC transition: given a marginal tree, produce the next one.

    Under the SMC, recombination picks a branch (proportional to length),
    snips above the recombination point, and the detached lineage re-coalesces
    with one of the remaining branches -- never with a 'ghost' lineage outside
    the current tree.

    Parameters
    ----------
    tree_branches : list of (child, parent, lower_time, upper_time)
    recomb_rate : float
    coal_rates : callable

    Returns
    -------
    dict or None
    """
    total_length = sum(u - l for _, _, l, u in tree_branches)
    branch_lengths = [(u - l) for _, _, l, u in tree_branches]
    probs = np.array(branch_lengths) / total_length
    idx = np.random.choice(len(tree_branches), p=probs)
    child, parent, lower, upper = tree_branches[idx]
    recomb_time = np.random.uniform(lower, upper)
    available = [(c, p, l, u) for c, p, l, u in tree_branches
                 if u > recomb_time and (c, p, l, u) != tree_branches[idx]]
    if not available:
        return None
    rejoin_idx = np.random.randint(len(available))
    rejoin_branch = available[rejoin_idx]
    return {
        'recomb_branch': tree_branches[idx],
        'recomb_time': recomb_time,
        'rejoin_branch': rejoin_branch,
    }


# Example tree used in the documentation
EXAMPLE_TREE_BRANCHES = [
    (0, 4, 0.0, 0.3),
    (1, 4, 0.0, 0.3),
    (2, 5, 0.0, 0.7),
    (3, 5, 0.0, 0.7),
    (4, 6, 0.3, 1.5),
    (5, 6, 0.7, 1.5),
]


class TestSMCTransition:
    """Tests for the smc_transition function."""

    def test_returns_dict_or_none(self):
        """Verify that smc_transition returns a dict with expected keys or None."""
        np.random.seed(42)
        result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
        assert result is not None
        assert 'recomb_branch' in result
        assert 'recomb_time' in result
        assert 'rejoin_branch' in result

    def test_recomb_branch_is_valid(self):
        """Verify that the recombination branch is one of the tree branches."""
        np.random.seed(42)
        result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
        assert result['recomb_branch'] in EXAMPLE_TREE_BRANCHES

    def test_recomb_time_within_branch(self):
        """Verify that recombination time falls within the chosen branch's interval."""
        np.random.seed(42)
        result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
        _, _, lower, upper = result['recomb_branch']
        assert lower <= result['recomb_time'] <= upper

    def test_rejoin_branch_is_different(self):
        """Verify that the rejoin branch is not the same as the recombination branch."""
        np.random.seed(42)
        for _ in range(50):
            result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
            if result is not None:
                assert result['rejoin_branch'] != result['recomb_branch']

    def test_rejoin_branch_extends_above_recomb_time(self):
        """Verify that the rejoin branch has upper_time > recomb_time (SMC constraint)."""
        np.random.seed(42)
        for _ in range(50):
            result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
            if result is not None:
                _, _, _, u = result['rejoin_branch']
                assert u > result['recomb_time']

    def test_branch_selection_proportional_to_length(self):
        """Verify branches are selected roughly proportional to their length."""
        np.random.seed(123)
        counts = {i: 0 for i in range(len(EXAMPLE_TREE_BRANCHES))}
        n_trials = 10000
        for _ in range(n_trials):
            result = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
            if result is not None:
                idx = EXAMPLE_TREE_BRANCHES.index(result['recomb_branch'])
                counts[idx] += 1

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
        result1 = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
        np.random.seed(42)
        result2 = smc_transition(EXAMPLE_TREE_BRANCHES, 0.01, None)
        assert result1 == result2


# ---------------------------------------------------------------------------
# Code block 2: smc_branch_transition
# ---------------------------------------------------------------------------

def smc_branch_transition(tau, p, rho, n_branches):
    """Compute SMC transition probabilities between branches.

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
    K = n_branches
    r = 1 - np.exp(-rho / 2 * tau)
    q = r * p
    q_sum = q.sum()
    T = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                T[i, j] = (1 - r[i]) + r[i] * q[j] / q_sum
            else:
                T[i, j] = r[i] * q[j] / q_sum
    assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"
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

    def test_very_small_recombination_rate(self):
        """When rho is very small, T should be close to the identity matrix.

        Note: rho=0 exactly causes a 0/0 division in q_j / q_sum, so we test
        with a very small positive rho instead.
        """
        K = 3
        tau = np.array([0.5, 1.0, 1.5])
        p = np.array([0.4, 0.3, 0.3])
        rho = 1e-12  # near-zero recombination rate
        T = smc_branch_transition(tau, p, rho, K)
        np.testing.assert_allclose(T, np.eye(K), atol=1e-6)

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
    p_recomb = 1 - p_no_recomb
    t = np.asarray(t, dtype=float)
    density = np.zeros_like(t)
    mask_lt = t < s
    density[mask_lt] = (p_recomb / s) * (1 - np.exp(-t[mask_lt]))
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
    p_recomb = 1 - p_no_recomb
    t = np.asarray(t, dtype=float)
    cdf = np.zeros_like(t)
    mask_lt = t < s
    cdf[mask_lt] = (p_recomb / s) * (
        t[mask_lt] + np.exp(-t[mask_lt]) - 1
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
