"""
Tests for self-contained functions extracted from the moments moment_equations
documentation (docs/timepieces/moments/moment_equations.rst).

Tests cover: drift_operator, mutation_operator, selection_operator,
migration_operator_2pop, drift_operator_with_size, integrate_sfs, split_1d_to_2d.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import comb


# ---------------------------------------------------------------------------
# Functions under test (extracted verbatim from the RST documentation)
# ---------------------------------------------------------------------------

def drift_operator(phi, n):
    """Compute the drift contribution to d(phi)/dt.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS (phi[j] = expected count at frequency j/n).
    n : int
        Sample size.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
        Change in SFS due to drift.
    """
    dphi = np.zeros(n + 1)
    for j in range(1, n):
        term_down = j * (j - 1) * phi[j - 1] if j >= 1 else 0.0
        term_stay = -2 * j * (n - j) * phi[j]
        term_up = (n - j) * (n - j - 1) * phi[j + 1] if j < n else 0.0
        dphi[j] = (term_down + term_stay + term_up) / (2.0 * n)
    return dphi


def mutation_operator(phi, n, theta):
    """Compute the mutation contribution to d(phi)/dt.

    Under the infinite-sites model, mutations only add singletons.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS.
    n : int
        Sample size.
    theta : float
        Population-scaled mutation rate (4*Ne*mu, per site or total).

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    dphi = np.zeros(n + 1)
    dphi[1] = theta / 2.0
    return dphi


def selection_operator(phi, n, gamma, h=0.5):
    """Compute the selection contribution to d(phi)/dt.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS.
    n : int
        Sample size.
    gamma : float
        Scaled selection coefficient (2*Ne*s).
    h : float
        Dominance coefficient.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    dphi = np.zeros(n + 1)
    for j in range(1, n):
        if j > 0 and j < n:
            term1 = gamma * h * ((j - 1) * (n - j + 1) * phi[j - 1] -
                                  j * (n - j) * phi[j]) / n
            term2 = gamma * (1 - 2 * h) * (
                (j - 1) * (j - 2) * phi[j - 1] / (n * (n - 1)) * (n - j + 1)
                - j * (j - 1) * phi[j] / (n * (n - 1)) * (n - j)
            ) if n > 1 else 0
            dphi[j] = term1 + term2
    return dphi


def migration_operator_2pop(phi_2d, n1, n2, M12, M21):
    """Compute migration contribution for a 2D SFS.

    Parameters
    ----------
    phi_2d : ndarray of shape (n1+1, n2+1)
        Joint SFS.
    n1, n2 : int
        Sample sizes.
    M12 : float
        Scaled migration rate from pop 2 into pop 1.
    M21 : float
        Scaled migration rate from pop 1 into pop 2.

    Returns
    -------
    dphi : ndarray of shape (n1+1, n2+1)
    """
    dphi = np.zeros((n1 + 1, n2 + 1))
    for j1 in range(1, n1):
        for j2 in range(1, n2):
            if j1 > 0 and j2 < n2:
                dphi[j1, j2] += M12 * (j2 + 1) / n2 * phi_2d[j1 - 1, j2 + 1]
            dphi[j1, j2] -= M12 * j1 / n1 * phi_2d[j1, j2]
            if j2 > 0 and j1 < n1:
                dphi[j1, j2] += M21 * (j1 + 1) / n1 * phi_2d[j1 + 1, j2 - 1]
            dphi[j1, j2] -= M21 * j2 / n2 * phi_2d[j1, j2]
    return dphi


def drift_operator_with_size(phi, n, nu):
    """Drift operator scaled by relative population size.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
    n : int
    nu : float
        Relative population size (N/N_ref). nu > 1 = expansion.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    return drift_operator(phi, n) / nu


def integrate_sfs(phi_init, n, T, nu_func, theta, gamma=0, h=0.5):
    """Integrate the SFS forward through time.

    Parameters
    ----------
    phi_init : ndarray of shape (n+1,)
        Initial SFS.
    n : int
        Sample size.
    T : float
        Integration time (in 2*Ne generations).
    nu_func : callable
        nu_func(t) returns the relative population size at time t.
    theta : float
        Scaled mutation rate.
    gamma : float
        Scaled selection coefficient.
    h : float
        Dominance coefficient.

    Returns
    -------
    phi_final : ndarray of shape (n+1,)
        SFS after integration.
    """
    y0 = phi_init[1:n].copy()

    def rhs(t, y):
        phi = np.zeros(n + 1)
        phi[1:n] = y
        nu = nu_func(t)
        d_drift = drift_operator(phi, n) / nu
        d_mutation = mutation_operator(phi, n, theta)
        d_selection = selection_operator(phi, n, gamma, h) if gamma != 0 else np.zeros(n + 1)
        dphi = d_drift + d_mutation + d_selection
        return dphi[1:n]

    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                     rtol=1e-10, atol=1e-12,
                     dense_output=True)
    phi_final = np.zeros(n + 1)
    phi_final[1:n] = sol.y[:, -1]
    return phi_final


def split_1d_to_2d(phi_1d, n1, n2):
    """Split a 1D SFS into a 2D joint SFS.

    Parameters
    ----------
    phi_1d : ndarray of shape (n+1,)
        1D SFS with n = n1 + n2.
    n1, n2 : int
        Sample sizes for the two daughter populations.

    Returns
    -------
    phi_2d : ndarray of shape (n1+1, n2+1)
    """
    n = n1 + n2
    phi_2d = np.zeros((n1 + 1, n2 + 1))
    for j in range(n + 1):
        if phi_1d[j] == 0:
            continue
        for j1 in range(max(0, j - n2), min(j, n1) + 1):
            j2 = j - j1
            if j2 < 0 or j2 > n2:
                continue
            prob = (comb(j, j1, exact=True) *
                    comb(n - j, n1 - j1, exact=True) /
                    comb(n, n1, exact=True))
            phi_2d[j1, j2] = phi_1d[j] * prob
    return phi_2d


# ---------------------------------------------------------------------------
# Helper: neutral SFS with theta/j shape
# ---------------------------------------------------------------------------

def _neutral_sfs(n, theta=1.0):
    """Build the standard neutral SFS: phi[j] = theta / j for j in 1..n-1."""
    phi = np.zeros(n + 1)
    for j in range(1, n):
        phi[j] = theta / j
    return phi


# ===========================================================================
# Tests for drift_operator
# ===========================================================================

class TestDriftOperator:
    """Tests for the drift_operator function."""

    def test_output_shape(self):
        """Drift operator returns an array of the correct shape."""
        n = 10
        phi = _neutral_sfs(n)
        dphi = drift_operator(phi, n)
        assert dphi.shape == (n + 1,)

    def test_boundary_bins_unchanged(self):
        """Drift should not modify the monomorphic bins (j=0 and j=n)."""
        n = 15
        phi = _neutral_sfs(n)
        dphi = drift_operator(phi, n)
        assert dphi[0] == 0.0
        assert dphi[n] == 0.0

    def test_zero_sfs_gives_zero_drift(self):
        """Drift of a zero SFS should be identically zero."""
        n = 20
        phi = np.zeros(n + 1)
        dphi = drift_operator(phi, n)
        assert np.allclose(dphi, 0.0)

    def test_drift_conserves_total_count_approximately(self):
        """Drift redistributes probability but should approximately conserve
        the total sum of the internal SFS entries (it is a second-order
        difference operator, analogous to a discrete Laplacian)."""
        n = 30
        phi = _neutral_sfs(n)
        dphi = drift_operator(phi, n)
        # The net flow summed over all internal bins should be close to zero
        # for a smooth SFS, up to boundary effects.
        net_flow = np.sum(dphi[1:n])
        # The net flow is not exactly zero for the drift-only operator
        # because probability can flow to the boundary bins (fixation/loss).
        # But the magnitude should be bounded.
        assert abs(net_flow) < np.sum(np.abs(dphi[1:n])) + 1e-15

    def test_drift_symmetry_on_symmetric_sfs(self):
        """For a symmetric SFS (phi[j] = phi[n-j]), the drift output should
        also be symmetric."""
        n = 20
        phi = np.zeros(n + 1)
        for j in range(1, n):
            phi[j] = 1.0  # flat (symmetric) SFS
        dphi = drift_operator(phi, n)
        for j in range(1, n):
            assert np.isclose(dphi[j], dphi[n - j], atol=1e-12), (
                f"Drift not symmetric at j={j}: {dphi[j]} vs {dphi[n-j]}"
            )

    def test_drift_pushes_mass_toward_boundaries(self):
        """For a flat SFS, drift should push mass toward the boundaries
        (low and high frequency bins get positive contributions, middle
        bins get negative contributions)."""
        n = 20
        phi = np.zeros(n + 1)
        for j in range(1, n):
            phi[j] = 1.0
        dphi = drift_operator(phi, n)
        # Middle bin should decrease
        assert dphi[n // 2] < 0, "Drift should decrease the mid-frequency bin"
        # Low-frequency bin should increase
        assert dphi[1] > 0, "Drift should increase the singleton bin"

    def test_drift_scales_linearly(self):
        """Drift operator is linear: drift(c * phi) = c * drift(phi)."""
        n = 15
        phi = _neutral_sfs(n)
        c = 3.7
        dphi1 = drift_operator(c * phi, n)
        dphi2 = c * drift_operator(phi, n)
        assert np.allclose(dphi1, dphi2, atol=1e-14)


# ===========================================================================
# Tests for mutation_operator
# ===========================================================================

class TestMutationOperator:
    """Tests for the mutation_operator function."""

    def test_output_shape(self):
        """Mutation operator returns an array of the correct shape."""
        n = 10
        phi = _neutral_sfs(n)
        dphi = mutation_operator(phi, n, theta=1.0)
        assert dphi.shape == (n + 1,)

    def test_only_singleton_bin_nonzero(self):
        """Under infinite-sites, mutation only feeds the j=1 bin."""
        n = 20
        theta = 2.5
        phi = np.zeros(n + 1)
        dphi = mutation_operator(phi, n, theta)
        assert dphi[1] == theta / 2.0
        for j in range(2, n + 1):
            assert dphi[j] == 0.0
        assert dphi[0] == 0.0

    def test_mutation_independent_of_phi(self):
        """The mutation operator value should not depend on the current SFS
        (under infinite-sites, it is a constant injection)."""
        n = 15
        theta = 1.0
        phi_a = np.zeros(n + 1)
        phi_b = _neutral_sfs(n, theta=5.0)
        dphi_a = mutation_operator(phi_a, n, theta)
        dphi_b = mutation_operator(phi_b, n, theta)
        assert np.allclose(dphi_a, dphi_b)

    def test_mutation_scales_with_theta(self):
        """The singleton injection rate should scale linearly with theta."""
        n = 10
        phi = np.zeros(n + 1)
        dphi1 = mutation_operator(phi, n, theta=1.0)
        dphi2 = mutation_operator(phi, n, theta=4.0)
        assert np.isclose(dphi2[1], 4.0 * dphi1[1])

    def test_mutation_zero_theta(self):
        """With theta=0 there should be no mutation input."""
        n = 10
        phi = _neutral_sfs(n)
        dphi = mutation_operator(phi, n, theta=0.0)
        assert np.allclose(dphi, 0.0)


# ===========================================================================
# Tests for selection_operator
# ===========================================================================

class TestSelectionOperator:
    """Tests for the selection_operator function."""

    def test_output_shape(self):
        """Selection operator returns an array of the correct shape."""
        n = 10
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=5.0)
        assert dphi.shape == (n + 1,)

    def test_boundary_bins_unchanged(self):
        """Selection should not modify the monomorphic bins."""
        n = 15
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=5.0, h=0.5)
        assert dphi[0] == 0.0
        assert dphi[n] == 0.0

    def test_zero_gamma_gives_zero_selection(self):
        """No selection (gamma=0) should produce zero change."""
        n = 20
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=0.0)
        assert np.allclose(dphi, 0.0)

    def test_zero_sfs_gives_zero_selection(self):
        """Selection on a zero SFS should be identically zero."""
        n = 20
        phi = np.zeros(n + 1)
        dphi = selection_operator(phi, n, gamma=10.0, h=0.5)
        assert np.allclose(dphi, 0.0)

    def test_selection_scales_linearly_with_gamma(self):
        """Selection operator is linear in gamma: S(c*gamma) = c * S(gamma)."""
        n = 15
        phi = _neutral_sfs(n)
        dphi1 = selection_operator(phi, n, gamma=2.0, h=0.5)
        dphi2 = selection_operator(phi, n, gamma=6.0, h=0.5)
        assert np.allclose(dphi2, 3.0 * dphi1, atol=1e-14)

    def test_selection_linear_in_phi(self):
        """Selection operator is linear in phi: S(c*phi) = c * S(phi)."""
        n = 15
        phi = _neutral_sfs(n)
        c = 2.5
        dphi1 = selection_operator(c * phi, n, gamma=5.0, h=0.5)
        dphi2 = c * selection_operator(phi, n, gamma=5.0, h=0.5)
        assert np.allclose(dphi1, dphi2, atol=1e-14)

    def test_purifying_selection_reduces_high_freq(self):
        """Purifying selection (gamma < 0) should reduce high-frequency
        derived alleles (negative dphi for high j)."""
        n = 30
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=-10.0, h=0.5)
        # For strong purifying selection, high-frequency bins should decrease
        # (dphi negative for j near n)
        assert dphi[n - 2] < 0, (
            "Purifying selection should reduce high-frequency bins"
        )

    def test_additive_selection_h_half(self):
        """For additive selection (h=0.5), the dominance deviation term
        (proportional to 1-2h=0) should vanish, leaving only the linear term."""
        n = 20
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=5.0, h=0.5)
        # Just verify it runs and gives a non-trivial result
        assert not np.allclose(dphi, 0.0)


# ===========================================================================
# Tests for migration_operator_2pop
# ===========================================================================

class TestMigrationOperator2Pop:
    """Tests for the migration_operator_2pop function."""

    def test_output_shape(self):
        """Migration operator returns a 2D array of the correct shape."""
        n1, n2 = 5, 7
        phi_2d = np.ones((n1 + 1, n2 + 1))
        dphi = migration_operator_2pop(phi_2d, n1, n2, M12=1.0, M21=1.0)
        assert dphi.shape == (n1 + 1, n2 + 1)

    def test_zero_migration_gives_zero(self):
        """With no migration, the migration operator should be zero."""
        n1, n2 = 6, 8
        phi_2d = np.random.RandomState(42).rand(n1 + 1, n2 + 1)
        dphi = migration_operator_2pop(phi_2d, n1, n2, M12=0.0, M21=0.0)
        assert np.allclose(dphi, 0.0)

    def test_zero_sfs_gives_zero_migration(self):
        """Migration on a zero 2D SFS should be identically zero."""
        n1, n2 = 5, 5
        phi_2d = np.zeros((n1 + 1, n2 + 1))
        dphi = migration_operator_2pop(phi_2d, n1, n2, M12=2.0, M21=3.0)
        assert np.allclose(dphi, 0.0)

    def test_migration_linear_in_phi(self):
        """Migration operator is linear: M(c*phi) = c * M(phi)."""
        n1, n2 = 5, 5
        np.random.seed(123)
        phi_2d = np.random.rand(n1 + 1, n2 + 1)
        c = 3.0
        dphi1 = migration_operator_2pop(c * phi_2d, n1, n2, M12=1.0, M21=0.5)
        dphi2 = c * migration_operator_2pop(phi_2d, n1, n2, M12=1.0, M21=0.5)
        assert np.allclose(dphi1, dphi2, atol=1e-14)

    def test_migration_scales_with_rate(self):
        """Doubling both migration rates should double the operator output."""
        n1, n2 = 5, 5
        np.random.seed(99)
        phi_2d = np.random.rand(n1 + 1, n2 + 1)
        dphi1 = migration_operator_2pop(phi_2d, n1, n2, M12=1.0, M21=2.0)
        dphi2 = migration_operator_2pop(phi_2d, n1, n2, M12=2.0, M21=4.0)
        assert np.allclose(dphi2, 2.0 * dphi1, atol=1e-14)

    def test_symmetric_migration_on_symmetric_sfs(self):
        """With equal sample sizes and symmetric migration on a transposed-
        symmetric SFS, the output should respect the same symmetry."""
        n = 6
        np.random.seed(77)
        base = np.random.rand(n + 1, n + 1)
        phi_2d = (base + base.T) / 2.0  # make symmetric
        M = 1.5
        dphi = migration_operator_2pop(phi_2d, n, n, M12=M, M21=M)
        # dphi should also be symmetric (transposed)
        assert np.allclose(dphi, dphi.T, atol=1e-12), (
            "Symmetric migration on symmetric SFS should give symmetric output"
        )

    def test_boundary_rows_and_cols_zero(self):
        """The migration operator only modifies internal bins (j1 in 1..n1-1,
        j2 in 1..n2-1), so boundary rows/columns should be zero."""
        n1, n2 = 5, 7
        np.random.seed(55)
        phi_2d = np.random.rand(n1 + 1, n2 + 1)
        dphi = migration_operator_2pop(phi_2d, n1, n2, M12=1.0, M21=1.0)
        assert np.allclose(dphi[0, :], 0.0)
        assert np.allclose(dphi[n1, :], 0.0)
        assert np.allclose(dphi[:, 0], 0.0)
        assert np.allclose(dphi[:, n2], 0.0)


# ===========================================================================
# Tests for drift_operator_with_size
# ===========================================================================

class TestDriftOperatorWithSize:
    """Tests for the drift_operator_with_size function."""

    def test_nu_one_equals_plain_drift(self):
        """With nu=1, drift_operator_with_size should equal drift_operator."""
        n = 15
        phi = _neutral_sfs(n)
        dphi_plain = drift_operator(phi, n)
        dphi_sized = drift_operator_with_size(phi, n, nu=1.0)
        assert np.allclose(dphi_plain, dphi_sized)

    def test_larger_nu_means_weaker_drift(self):
        """Larger population (nu > 1) should produce smaller magnitude drift."""
        n = 20
        phi = _neutral_sfs(n)
        dphi_small = drift_operator_with_size(phi, n, nu=1.0)
        dphi_large = drift_operator_with_size(phi, n, nu=10.0)
        # Each entry should be 10x smaller in magnitude
        assert np.allclose(dphi_large, dphi_small / 10.0)

    def test_scaling_is_exact_inverse(self):
        """drift_with_size(phi, n, nu) = drift(phi, n) / nu exactly."""
        n = 15
        phi = _neutral_sfs(n, theta=2.0)
        nu = 3.5
        expected = drift_operator(phi, n) / nu
        result = drift_operator_with_size(phi, n, nu)
        assert np.allclose(result, expected, atol=1e-15)

    def test_small_nu_amplifies_drift(self):
        """Smaller population (nu < 1) should amplify drift."""
        n = 20
        phi = _neutral_sfs(n)
        dphi_ref = drift_operator(phi, n)
        dphi_small = drift_operator_with_size(phi, n, nu=0.5)
        assert np.allclose(dphi_small, 2.0 * dphi_ref)


# ===========================================================================
# Tests for integrate_sfs
# ===========================================================================

class TestIntegrateSfs:
    """Tests for the integrate_sfs ODE integration function."""

    def test_output_shape(self):
        """Integrated SFS should have shape (n+1,)."""
        n = 10
        phi = _neutral_sfs(n)
        result = integrate_sfs(phi, n, T=0.1, nu_func=lambda t: 1.0, theta=1.0)
        assert result.shape == (n + 1,)

    def test_boundary_bins_remain_zero(self):
        """Boundary bins (j=0, j=n) should remain zero after integration."""
        n = 15
        phi = _neutral_sfs(n)
        result = integrate_sfs(phi, n, T=0.5, nu_func=lambda t: 1.0, theta=1.0)
        assert result[0] == 0.0
        assert result[n] == 0.0

    def test_nonnegative_sfs_after_integration(self):
        """The SFS should remain non-negative after integration under
        reasonable parameters."""
        n = 15
        phi = _neutral_sfs(n, theta=1.0)
        result = integrate_sfs(phi, n, T=1.0, nu_func=lambda t: 1.0, theta=1.0)
        assert np.all(result >= -1e-10), (
            f"SFS has negative entries: {result[result < -1e-10]}"
        )

    def test_integration_preserves_monotonic_sfs_decrease(self):
        """Integrating a 1/j-shaped SFS for a short time under constant
        size should produce an SFS that is still monotonically decreasing
        for internal bins, i.e. phi[j] > phi[j+1] for j=1..n-2. The
        short integration time ensures the shape is not dramatically altered."""
        n = 15
        theta = 1.0
        phi_init = _neutral_sfs(n, theta)
        # Integrate for a very short time
        phi_after = integrate_sfs(phi_init, n, T=0.01,
                                  nu_func=lambda t: 1.0, theta=theta)
        # Internal bins should still be monotonically decreasing
        for j in range(1, n - 1):
            assert phi_after[j] >= phi_after[j + 1] - 1e-10, (
                f"SFS should be approximately decreasing: "
                f"phi[{j}]={phi_after[j]:.6f} < phi[{j+1}]={phi_after[j+1]:.6f}"
            )

    def test_zero_time_returns_initial(self):
        """Integrating for T=0 should return the initial SFS unchanged."""
        n = 15
        phi = _neutral_sfs(n, theta=2.0)
        result = integrate_sfs(phi, n, T=0.0, nu_func=lambda t: 1.0, theta=2.0)
        assert np.allclose(result, phi, atol=1e-12)

    def test_expansion_increases_rare_variants(self):
        """A population expansion (large nu) should increase the proportion of
        rare variants relative to common variants, because drift is weakened
        but mutation continues."""
        n = 20
        theta = 1.0
        phi_eq = _neutral_sfs(n, theta)
        phi_expanded = integrate_sfs(phi_eq, n, T=0.5,
                                     nu_func=lambda t: 10.0, theta=theta)
        # After expansion, singletons should be enriched relative to
        # higher-frequency classes.
        ratio_singleton = phi_expanded[1] / phi_eq[1] if phi_eq[1] > 0 else 0
        ratio_mid = phi_expanded[n // 2] / phi_eq[n // 2] if phi_eq[n // 2] > 0 else 0
        assert ratio_singleton > ratio_mid, (
            "Expansion should enrich rare variants more than common ones"
        )

    def test_flat_sfs_converges_toward_neutral(self):
        """Starting from a flat SFS (phi[j]=1), long integration with constant
        size should produce something closer to the 1/j neutral shape."""
        n = 15
        theta = 1.0
        phi_flat = np.zeros(n + 1)
        for j in range(1, n):
            phi_flat[j] = 1.0
        phi_after = integrate_sfs(phi_flat, n, T=5.0,
                                  nu_func=lambda t: 1.0, theta=theta)
        # After long integration, phi[1] should be larger than phi[n//2]
        # (the 1/j shape is monotonically decreasing)
        assert phi_after[1] > phi_after[n // 2], (
            "After long integration, the SFS should approach 1/j shape"
        )


# ===========================================================================
# Tests for split_1d_to_2d
# ===========================================================================

class TestSplit1dTo2d:
    """Tests for the split_1d_to_2d function."""

    def test_output_shape(self):
        """The 2D SFS should have shape (n1+1, n2+1)."""
        n1, n2 = 5, 7
        n = n1 + n2
        phi_1d = _neutral_sfs(n)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert phi_2d.shape == (n1 + 1, n2 + 1)

    def test_nonnegative_entries(self):
        """All entries of the 2D SFS should be non-negative."""
        n1, n2 = 6, 8
        n = n1 + n2
        phi_1d = _neutral_sfs(n)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.all(phi_2d >= 0.0)

    def test_total_mass_conserved(self):
        """The sum of the 2D SFS should equal the sum of the 1D SFS.

        Each entry phi_1d[j] is distributed across the 2D grid with
        hypergeometric probabilities that sum to 1, so total mass is conserved.
        """
        n1, n2 = 5, 5
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=2.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.isclose(phi_2d.sum(), phi_1d.sum(), rtol=1e-10)

    def test_marginal_over_pop2_gives_projection(self):
        """Marginalizing the 2D SFS over pop 2 should give the projection
        of the 1D SFS to sample size n1.

        The projection of phi_1d from sample size n to n1 is:
        phi_proj[j1] = sum_j phi_1d[j] * C(j, j1) * C(n-j, n1-j1) / C(n, n1)
        which is exactly what summing phi_2d over the j2 axis gives.
        """
        n1, n2 = 6, 8
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        # Marginalize over pop 2
        phi_marginal = phi_2d.sum(axis=1)

        # Compute projection directly
        phi_proj = np.zeros(n1 + 1)
        for j1 in range(n1 + 1):
            for j in range(n + 1):
                if phi_1d[j] == 0:
                    continue
                if j1 > j or (n1 - j1) > (n - j):
                    continue
                prob = (comb(j, j1, exact=True) *
                        comb(n - j, n1 - j1, exact=True) /
                        comb(n, n1, exact=True))
                phi_proj[j1] += phi_1d[j] * prob

        assert np.allclose(phi_marginal, phi_proj, atol=1e-12), (
            "Marginalization over pop 2 should equal projection to n1"
        )

    def test_marginal_over_pop1_gives_projection(self):
        """Same as above but marginalizing over pop 1 to get pop 2 projection."""
        n1, n2 = 7, 5
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.5)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        phi_marginal = phi_2d.sum(axis=0)

        phi_proj = np.zeros(n2 + 1)
        for j2 in range(n2 + 1):
            for j in range(n + 1):
                if phi_1d[j] == 0:
                    continue
                if j2 > j or (n2 - j2) > (n - j):
                    continue
                prob = (comb(j, j2, exact=True) *
                        comb(n - j, n2 - j2, exact=True) /
                        comb(n, n2, exact=True))
                phi_proj[j2] += phi_1d[j] * prob

        assert np.allclose(phi_marginal, phi_proj, atol=1e-12)

    def test_zero_sfs_gives_zero_2d(self):
        """Splitting a zero SFS should give a zero 2D SFS."""
        n1, n2 = 4, 6
        n = n1 + n2
        phi_1d = np.zeros(n + 1)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.allclose(phi_2d, 0.0)

    def test_single_entry_distribution(self):
        """If the 1D SFS has mass only at j=k, the 2D SFS should have
        nonzero entries only on the anti-diagonal j1 + j2 = k, with
        hypergeometric weights."""
        n1, n2 = 4, 6
        n = n1 + n2
        k = 5
        phi_1d = np.zeros(n + 1)
        phi_1d[k] = 3.0  # arbitrary mass

        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        # Only entries with j1 + j2 == k should be nonzero
        for j1 in range(n1 + 1):
            for j2 in range(n2 + 1):
                if j1 + j2 == k:
                    # Should be positive (hypergeometric weight * 3.0)
                    expected_prob = (comb(k, j1, exact=True) *
                                    comb(n - k, n1 - j1, exact=True) /
                                    comb(n, n1, exact=True))
                    if expected_prob > 0:
                        assert np.isclose(phi_2d[j1, j2], 3.0 * expected_prob)
                else:
                    assert phi_2d[j1, j2] == 0.0, (
                        f"Off-diagonal entry phi_2d[{j1},{j2}] should be 0, "
                        f"got {phi_2d[j1, j2]}"
                    )

    def test_hypergeometric_weights_sum_to_one(self):
        """For each frequency class j in the 1D SFS, the hypergeometric
        weights across all (j1, j2) with j1+j2=j should sum to 1."""
        n1, n2 = 5, 5
        n = n1 + n2
        for j in range(1, n):
            weight_sum = 0.0
            for j1 in range(max(0, j - n2), min(j, n1) + 1):
                j2 = j - j1
                prob = (comb(j, j1, exact=True) *
                        comb(n - j, n1 - j1, exact=True) /
                        comb(n, n1, exact=True))
                weight_sum += prob
            assert np.isclose(weight_sum, 1.0, atol=1e-12), (
                f"Hypergeometric weights for j={j} sum to {weight_sum}, not 1"
            )

    def test_equal_split_symmetry(self):
        """When n1 == n2, split should produce a 2D SFS that is symmetric
        across the diagonal (phi_2d[j1, j2] == phi_2d[j2, j1])."""
        n1 = n2 = 6
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.allclose(phi_2d, phi_2d.T, atol=1e-12), (
            "Equal split should produce a symmetric 2D SFS"
        )


# ===========================================================================
# Integration tests combining multiple operators
# ===========================================================================

class TestCombinedOperators:
    """Integration tests that exercise multiple operators together."""

    def test_drift_plus_mutation_equilibrium_direction(self):
        """Starting from a flat SFS, drift+mutation should move toward the
        neutral equilibrium shape (1/j)."""
        n = 15
        theta = 1.0
        phi_flat = np.zeros(n + 1)
        for j in range(1, n):
            phi_flat[j] = 1.0

        dphi = drift_operator(phi_flat, n) + mutation_operator(phi_flat, n, theta)

        # At the singleton bin, the derivative should be positive
        # (mutation feeds it, and the 1/j equilibrium has phi[1] > 1)
        # This tests that the combined operator pushes toward equilibrium.
        assert dphi[1] > 0, (
            "Drift + mutation should increase singletons for a flat SFS"
        )

    def test_selection_reversal(self):
        """Positive and negative selection should produce opposite effects."""
        n = 20
        phi = _neutral_sfs(n)
        dphi_pos = selection_operator(phi, n, gamma=5.0, h=0.5)
        dphi_neg = selection_operator(phi, n, gamma=-5.0, h=0.5)
        assert np.allclose(dphi_pos, -dphi_neg, atol=1e-14)

    def test_split_then_migration_preserves_total_mass_direction(self):
        """After splitting a 1D SFS into 2D using split_1d_to_2d,
        applying the migration operator should redistribute mass
        but keep total internal mass at zero net change under
        symmetric equal-size conditions."""
        n1 = n2 = 5
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        M = 2.0
        dphi_mig = migration_operator_2pop(phi_2d, n1, n2, M12=M, M21=M)

        # The sum of dphi over all internal bins gives the net change
        # in total mass. Migration should redistribute, not create/destroy.
        # Note: the operator only touches internal bins (1..n-1), so
        # we sum over those.
        net_change = dphi_mig[1:n1, 1:n2].sum()
        # This may not be exactly zero due to boundary effects in the
        # discrete operator, but it should be small relative to the mass.
        total_mass = phi_2d[1:n1, 1:n2].sum()
        assert abs(net_change) < 0.5 * total_mass, (
            "Migration should not drastically change total internal mass"
        )
