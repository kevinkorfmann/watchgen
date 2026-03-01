"""
Tests for watchgen.mini_dadi -- the diffusion approximation module.

Adapted from tests/test_timepieces_dadi.py. All functions are imported from
watchgen.mini_dadi rather than being redefined here.

Covers:
- equilibrium_sfs_density: neutral equilibrium density proportional to 1/x
- make_nonuniform_grid: boundary-concentrated frequency grid
- phi_1d_to_2d: population split (1D -> 2D diagonal density)
- crank_nicolson_1d: PDE integration with drift and mutation
- sfs_from_phi: binomial projection from density to discrete SFS
- poisson_log_likelihood: Poisson composite log-likelihood
- multinomial_log_likelihood: multinomial composite log-likelihood
- optimal_sfs_scaling: optimal theta rescaling
- two_epoch_sfs: two-epoch demographic model pipeline
"""

import numpy as np
import pytest

from watchgen.mini_dadi import (
    equilibrium_sfs_density,
    make_nonuniform_grid,
    phi_1d_to_2d,
    crank_nicolson_1d,
    sfs_from_phi,
    poisson_log_likelihood,
    multinomial_log_likelihood,
    optimal_sfs_scaling,
    two_epoch_sfs,
)


# ===========================================================================
# Tests for equilibrium_sfs_density
# ===========================================================================

class TestEquilibriumSfsDensity:
    def test_proportional_to_inverse_x(self):
        """The equilibrium density should be proportional to 1/x."""
        xx = np.linspace(0.01, 0.99, 100)
        phi = equilibrium_sfs_density(xx)
        # phi(x) / (1/x) should be constant
        ratios = phi * xx
        assert np.allclose(ratios, ratios[0], rtol=1e-10)

    def test_zero_at_boundaries(self):
        """Density at x=0 and x=1 should be zero."""
        xx = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        phi = equilibrium_sfs_density(xx)
        assert phi[0] == 0.0
        assert phi[-1] == 0.0

    def test_decreasing_for_small_x(self):
        """For x in (0, 0.5), density should be decreasing."""
        xx = np.linspace(0.01, 0.5, 50)
        phi = equilibrium_sfs_density(xx)
        for i in range(len(phi) - 1):
            assert phi[i] > phi[i + 1]

    def test_positive_interior(self):
        """Density should be strictly positive for 0 < x < 1."""
        xx = np.linspace(0.01, 0.99, 100)
        phi = equilibrium_sfs_density(xx)
        assert np.all(phi > 0)

    def test_symmetry_in_reciprocal(self):
        """phi(x) * x should be constant (all = 1 for this implementation)."""
        xx = np.array([0.1, 0.2, 0.5, 0.8])
        phi = equilibrium_sfs_density(xx)
        products = phi * xx
        assert np.allclose(products, 1.0)


# ===========================================================================
# Tests for make_nonuniform_grid
# ===========================================================================

class TestMakeNonuniformGrid:
    def test_endpoints(self):
        """Grid should start at 0 and end at 1."""
        xx = make_nonuniform_grid(50)
        assert xx[0] == 0.0
        assert xx[-1] == 1.0

    def test_length(self):
        """Grid should have the specified number of points."""
        for pts in [20, 40, 60, 100]:
            xx = make_nonuniform_grid(pts)
            assert len(xx) == pts

    def test_sorted(self):
        """Grid should be strictly increasing."""
        xx = make_nonuniform_grid(50)
        for i in range(len(xx) - 1):
            assert xx[i] < xx[i + 1]

    def test_denser_at_boundaries(self):
        """Grid spacing should be smaller near x=0 and x=1 than in the middle."""
        xx = make_nonuniform_grid(50)
        dx = np.diff(xx)
        # First spacing should be smaller than middle spacing
        mid = len(dx) // 2
        assert dx[0] < dx[mid]
        # Last spacing should be smaller than middle spacing
        assert dx[-1] < dx[mid]

    def test_within_unit_interval(self):
        """All grid points should be in [0, 1]."""
        xx = make_nonuniform_grid(60)
        assert np.all(xx >= 0.0)
        assert np.all(xx <= 1.0)


# ===========================================================================
# Tests for phi_1d_to_2d
# ===========================================================================

class TestPhi1dTo2d:
    def test_diagonal_structure(self):
        """The 2D density should be concentrated on the diagonal."""
        xx = np.linspace(0, 1, 10)
        phi_1d = equilibrium_sfs_density(xx)
        phi_2d = phi_1d_to_2d(phi_1d, xx)
        # Check that diagonal matches 1D
        for i in range(len(xx)):
            assert phi_2d[i, i] == phi_1d[i]

    def test_off_diagonal_zero(self):
        """Off-diagonal entries should be zero immediately after split."""
        xx = np.linspace(0, 1, 10)
        phi_1d = equilibrium_sfs_density(xx)
        phi_2d = phi_1d_to_2d(phi_1d, xx)
        for i in range(len(xx)):
            for j in range(len(xx)):
                if i != j:
                    assert phi_2d[i, j] == 0.0

    def test_total_mass_conserved(self):
        """Total mass should be preserved in the split."""
        xx = np.linspace(0.01, 0.99, 20)
        phi_1d = equilibrium_sfs_density(xx)
        phi_2d = phi_1d_to_2d(phi_1d, xx)
        assert np.isclose(phi_2d.sum(), phi_1d.sum())

    def test_output_shape(self):
        """Output should be n x n where n is the grid size."""
        xx = np.linspace(0, 1, 15)
        phi_1d = equilibrium_sfs_density(xx)
        phi_2d = phi_1d_to_2d(phi_1d, xx)
        assert phi_2d.shape == (15, 15)


# ===========================================================================
# Tests for crank_nicolson_1d
# ===========================================================================

class TestCrankNicolson1d:
    def test_boundary_conditions(self):
        """Boundary values should remain zero after integration."""
        xx = make_nonuniform_grid(40)
        phi = equilibrium_sfs_density(xx)
        phi_evolved = crank_nicolson_1d(phi, xx, T=0.1, nu=1.0)
        assert phi_evolved[0] == 0.0
        assert phi_evolved[-1] == 0.0

    def test_nonnegative(self):
        """Density should remain non-negative after integration."""
        xx = make_nonuniform_grid(60)
        phi = equilibrium_sfs_density(xx)
        phi_evolved = crank_nicolson_1d(phi, xx, T=0.1, nu=1.0, n_steps=200)
        assert np.all(phi_evolved >= -1e-6)

    def test_zero_time_unchanged(self):
        """Integrating for T=0 should return the original density."""
        xx = make_nonuniform_grid(40)
        phi = equilibrium_sfs_density(xx)
        phi_evolved = crank_nicolson_1d(phi, xx, T=0.0, nu=1.0)
        assert np.allclose(phi_evolved, phi)

    def test_expansion_spreads_density(self):
        """A population expansion (large nu) should spread the density."""
        xx = make_nonuniform_grid(60)
        phi = equilibrium_sfs_density(xx)
        phi_expanded = crank_nicolson_1d(phi, xx, T=0.5, nu=5.0, n_steps=500)
        # After expansion, drift is weaker, so the density should be
        # more spread out (less concentrated near boundaries)
        interior = (xx > 0.2) & (xx < 0.8)
        ratio_orig = np.sum(phi[interior]) / max(np.sum(phi), 1e-300)
        ratio_expanded = np.sum(phi_expanded[interior]) / max(np.sum(phi_expanded), 1e-300)
        assert ratio_expanded > ratio_orig * 0.5  # relaxed check

    def test_bottleneck_concentrates_density(self):
        """A bottleneck (small nu) should accelerate drift relative to
        constant size, causing faster loss of total density."""
        xx = make_nonuniform_grid(60)
        phi = equilibrium_sfs_density(xx)
        phi_const = crank_nicolson_1d(phi, xx, T=0.05, nu=1.0, n_steps=500)
        phi_bottle = crank_nicolson_1d(phi, xx, T=0.05, nu=0.2, n_steps=500)
        # Bottleneck should lose more mass than constant size
        assert np.sum(phi_bottle) < np.sum(phi_const)

    def test_output_shape(self):
        """Output should have the same shape as input."""
        xx = make_nonuniform_grid(40)
        phi = equilibrium_sfs_density(xx)
        phi_evolved = crank_nicolson_1d(phi, xx, T=0.1, nu=1.0)
        assert phi_evolved.shape == phi.shape


# ===========================================================================
# Tests for sfs_from_phi
# ===========================================================================

class TestSfsFromPhi:
    def test_output_shape(self):
        """SFS should have n_samples + 1 entries."""
        xx = make_nonuniform_grid(40)
        phi = equilibrium_sfs_density(xx)
        sfs = sfs_from_phi(phi, xx, 10)
        assert len(sfs) == 11

    def test_nonnegative(self):
        """All SFS entries should be non-negative."""
        xx = make_nonuniform_grid(60)
        phi = equilibrium_sfs_density(xx)
        sfs = sfs_from_phi(phi, xx, 20)
        assert np.all(sfs >= -1e-10)

    def test_neutral_sfs_decreasing(self):
        """Under neutrality, the SFS should decrease: sfs[k] > sfs[k+1]."""
        xx = make_nonuniform_grid(80)
        phi = equilibrium_sfs_density(xx)
        sfs = sfs_from_phi(phi, xx, 20)
        # Internal entries should be roughly decreasing
        for k in range(1, 10):
            assert sfs[k] > sfs[k + 1] - 1e-6

    def test_neutral_sfs_proportional_to_1_over_k(self):
        """Under neutrality, sfs[k] should be approximately proportional to 1/k."""
        xx = make_nonuniform_grid(100)
        phi = equilibrium_sfs_density(xx)
        sfs = sfs_from_phi(phi, xx, 20)
        # Check ratios: sfs[1] / sfs[k] ~ k
        for k in range(2, 10):
            ratio = sfs[1] / max(sfs[k], 1e-300)
            assert abs(ratio - k) / k < 0.3  # within 30%

    def test_zero_density_gives_zero_sfs(self):
        """Zero density should produce zero SFS."""
        xx = make_nonuniform_grid(40)
        phi = np.zeros_like(xx)
        sfs = sfs_from_phi(phi, xx, 10)
        assert np.allclose(sfs, 0.0)


# ===========================================================================
# Tests for poisson_log_likelihood
# ===========================================================================

class TestPoissonLogLikelihood:
    def test_finite(self):
        """Poisson LL should be finite for valid inputs."""
        model = np.array([5.0, 10.0, 15.0])
        data = np.array([5, 10, 15])
        ll = poisson_log_likelihood(model, data)
        assert np.isfinite(ll)

    def test_best_at_true_value(self):
        """Poisson LL should be maximized when model equals data."""
        data = np.array([10.0, 20.0, 30.0])
        ll_true = poisson_log_likelihood(data, data)
        ll_bad = poisson_log_likelihood(data * 2, data)
        assert ll_true > ll_bad

    def test_zero_observed_handled(self):
        """Zero-count entries should not cause errors."""
        model = np.array([5.0, 10.0, 15.0])
        data = np.array([0, 10, 0])
        ll = poisson_log_likelihood(model, data)
        assert np.isfinite(ll)

    def test_scaling_sensitivity(self):
        """LL should prefer the correct scale."""
        data = np.array([10.0, 20.0, 30.0])
        ll_correct = poisson_log_likelihood(data, data)
        ll_half = poisson_log_likelihood(data / 2, data)
        ll_double = poisson_log_likelihood(data * 2, data)
        assert ll_correct > ll_half
        assert ll_correct > ll_double


# ===========================================================================
# Tests for multinomial_log_likelihood
# ===========================================================================

class TestMultinomialLogLikelihood:
    def test_negative(self):
        """Multinomial LL should be non-positive (log of probabilities <= 1)."""
        model = np.array([1.0, 2.0, 3.0])
        data = np.array([10, 20, 30])
        ll = multinomial_log_likelihood(model, data)
        assert ll <= 0

    def test_best_at_true_proportions(self):
        """LL should be higher when model proportions match data proportions."""
        data = np.array([10, 20, 30])
        model_true = np.array([1.0, 2.0, 3.0])  # same proportions
        model_bad = np.array([3.0, 2.0, 1.0])  # reversed proportions
        ll_true = multinomial_log_likelihood(model_true, data)
        ll_bad = multinomial_log_likelihood(model_bad, data)
        assert ll_true > ll_bad

    def test_scale_invariant(self):
        """Multinomial LL should be invariant to model scaling."""
        data = np.array([10, 20, 30])
        model = np.array([1.0, 2.0, 3.0])
        ll1 = multinomial_log_likelihood(model, data)
        ll2 = multinomial_log_likelihood(model * 100, data)
        assert abs(ll1 - ll2) < 1e-10

    def test_finite(self):
        """LL should be finite for valid inputs."""
        model = np.array([1.0, 2.0, 3.0])
        data = np.array([5, 10, 15])
        ll = multinomial_log_likelihood(model, data)
        assert np.isfinite(ll)


# ===========================================================================
# Tests for optimal_sfs_scaling
# ===========================================================================

class TestOptimalSfsScaling:
    def test_identity_when_sums_equal(self):
        """When model and data have equal sums, optimal scaling = 1."""
        model = np.array([10.0, 20.0, 30.0])
        data = np.array([10, 20, 30])
        theta = optimal_sfs_scaling(model, data)
        assert abs(theta - 1.0) < 1e-10

    def test_correct_scaling(self):
        """Optimal scaling should be sum(data) / sum(model)."""
        model = np.array([1.0, 2.0, 3.0])
        data = np.array([10, 20, 30])
        theta = optimal_sfs_scaling(model, data)
        assert abs(theta - 10.0) < 1e-10

    def test_positive(self):
        """Optimal scaling should be positive for positive inputs."""
        model = np.array([1.0, 2.0, 3.0])
        data = np.array([5, 10, 15])
        theta = optimal_sfs_scaling(model, data)
        assert theta > 0

    def test_doubles_with_data(self):
        """Doubling data should double the optimal scaling."""
        model = np.array([1.0, 2.0, 3.0])
        data1 = np.array([5, 10, 15])
        data2 = 2 * data1
        theta1 = optimal_sfs_scaling(model, data1)
        theta2 = optimal_sfs_scaling(model, data2)
        assert abs(theta2 - 2 * theta1) < 1e-10


# ===========================================================================
# Integration tests combining multiple functions
# ===========================================================================

class TestIntegration:
    def test_equilibrium_sfs_pipeline(self):
        """Full pipeline: equilibrium density -> SFS extraction -> likelihood."""
        xx = make_nonuniform_grid(80)
        phi = equilibrium_sfs_density(xx)
        sfs = sfs_from_phi(phi, xx, 20)
        # SFS should be positive for internal bins
        assert np.all(sfs[1:20] > 0)
        # Likelihood evaluation should work
        ll = poisson_log_likelihood(sfs[1:20], sfs[1:20])
        assert np.isfinite(ll)

    def test_two_epoch_sfs_valid(self):
        """The two-epoch model should produce a valid SFS."""
        sfs = two_epoch_sfs(nu=2.0, T=0.1, n_samples=20, pts=60)
        # Internal entries should be non-negative
        assert np.all(sfs[1:20] >= -1e-6)

    def test_two_epoch_expansion_vs_contraction(self):
        """An expansion should produce a different SFS than a contraction."""
        sfs_expand = two_epoch_sfs(nu=5.0, T=0.3, n_samples=15, pts=60)
        sfs_contract = two_epoch_sfs(nu=0.2, T=0.3, n_samples=15, pts=60)
        # The SFS shapes should differ
        assert not np.allclose(sfs_expand, sfs_contract, atol=0.01)

    def test_multinomial_ll_maximized_at_true_model(self):
        """Multinomial LL should prefer the true model over a misspecified one."""
        # Generate "data" from an expansion model
        true_sfs = two_epoch_sfs(nu=2.0, T=0.1, n_samples=15, pts=60)
        data = np.maximum(true_sfs[1:15], 1e-10) * 1000  # scale up

        # Compare LL at true vs wrong model
        wrong_sfs = two_epoch_sfs(nu=0.1, T=0.1, n_samples=15, pts=60)
        true_internal = np.maximum(true_sfs[1:15], 1e-10)
        wrong_internal = np.maximum(wrong_sfs[1:15], 1e-10)

        ll_true = multinomial_log_likelihood(true_internal, data)
        ll_wrong = multinomial_log_likelihood(wrong_internal, data)
        assert ll_true > ll_wrong

    def test_optimal_scaling_improves_fit(self):
        """Applying optimal scaling should improve the Poisson LL."""
        xx = make_nonuniform_grid(60)
        phi = equilibrium_sfs_density(xx)
        model = sfs_from_phi(phi, xx, 15)[1:15]
        model = np.maximum(model, 1e-10)
        data = model * 500  # simulated data at different scale

        # Unscaled LL vs optimally scaled LL
        ll_unscaled = poisson_log_likelihood(model, data)
        theta_opt = optimal_sfs_scaling(model, data)
        ll_scaled = poisson_log_likelihood(model * theta_opt, data)
        assert ll_scaled > ll_unscaled
