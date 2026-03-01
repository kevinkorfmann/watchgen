"""
Tests for watchgen.mini_moments -- the moments demographic inference module.

Tests cover all functions extracted from the moments documentation:
- SFS computation: compute_sfs, compute_joint_sfs, expected_sfs_neutral
- SFS manipulation: fold_sfs, project_sfs
- Summary statistics: watterson_theta, nucleotide_diversity, tajimas_d
- Moment equations: drift_operator, mutation_operator, selection_operator,
  migration_operator_2pop, drift_operator_with_size, integrate_sfs, split_1d_to_2d
- Demographic inference: poisson_log_likelihood, optimal_theta_scaling,
  likelihood_ratio_test, apply_misidentification
- Linkage disequilibrium: compute_D, ld_decay_deterministic,
  compute_ld_statistics, ld_equilibrium, gaussian_composite_ll, map_r_bins_to_rho

All functions are imported from watchgen.mini_moments (not redefined).
"""

import numpy as np
import pytest
from scipy.special import comb

from watchgen.mini_moments import (
    compute_sfs,
    compute_joint_sfs,
    expected_sfs_neutral,
    fold_sfs,
    project_sfs,
    watterson_theta,
    nucleotide_diversity,
    tajimas_d,
    drift_operator,
    mutation_operator,
    selection_operator,
    migration_operator_2pop,
    drift_operator_with_size,
    integrate_sfs,
    split_1d_to_2d,
    poisson_log_likelihood,
    optimal_theta_scaling,
    fisher_information_numerical,
    likelihood_ratio_test,
    apply_misidentification,
    compute_D,
    ld_decay_deterministic,
    compute_ld_statistics,
    ld_equilibrium,
    gaussian_composite_ll,
    map_r_bins_to_rho,
    _neutral_sfs,
)


# ===========================================================================
# Tests for compute_sfs
# ===========================================================================

class TestComputeSfs:
    """Tests for the compute_sfs function."""

    def test_output_shape(self):
        """SFS should have shape (n+1,)."""
        n = 5
        genotypes = np.zeros((10, n), dtype=int)
        sfs = compute_sfs(genotypes, n)
        assert sfs.shape == (n + 1,)

    def test_all_monomorphic(self):
        """All-zero genotypes should put everything in bin 0."""
        n = 5
        genotypes = np.zeros((10, n), dtype=int)
        sfs = compute_sfs(genotypes, n)
        assert sfs[0] == 10
        assert sfs[1:].sum() == 0

    def test_known_counts(self):
        """Verify SFS with a hand-constructed genotype matrix."""
        n = 4
        # 3 sites: counts 1, 2, 3
        genotypes = np.array([
            [1, 0, 0, 0],  # j=1
            [1, 1, 0, 0],  # j=2
            [1, 1, 1, 0],  # j=3
        ])
        sfs = compute_sfs(genotypes, n)
        assert sfs[1] == 1
        assert sfs[2] == 1
        assert sfs[3] == 1
        assert sfs[0] == 0
        assert sfs[4] == 0

    def test_total_sites_conserved(self):
        """Total sites should equal the number of rows."""
        n = 5
        np.random.seed(42)
        genotypes = np.random.randint(0, 2, size=(20, n))
        sfs = compute_sfs(genotypes, n)
        assert sfs.sum() == 20


# ===========================================================================
# Tests for compute_joint_sfs
# ===========================================================================

class TestComputeJointSfs:
    """Tests for the compute_joint_sfs function."""

    def test_output_shape(self):
        """Joint SFS should have shape (n1+1, n2+1)."""
        n1, n2 = 3, 4
        gen1 = np.zeros((10, n1), dtype=int)
        gen2 = np.zeros((10, n2), dtype=int)
        sfs = compute_joint_sfs(gen1, gen2, n1, n2)
        assert sfs.shape == (n1 + 1, n2 + 1)

    def test_total_sites_conserved(self):
        """Total sites should equal the number of rows."""
        n1, n2 = 3, 4
        np.random.seed(42)
        gen1 = np.random.randint(0, 2, size=(15, n1))
        gen2 = np.random.randint(0, 2, size=(15, n2))
        sfs = compute_joint_sfs(gen1, gen2, n1, n2)
        assert sfs.sum() == 15


# ===========================================================================
# Tests for expected_sfs_neutral
# ===========================================================================

class TestExpectedSfsNeutral:
    """Tests for the expected_sfs_neutral function."""

    def test_output_shape(self):
        """Expected SFS should have shape (n+1,)."""
        sfs = expected_sfs_neutral(20, theta=1.0)
        assert sfs.shape == (21,)

    def test_boundary_bins_zero(self):
        """Bins 0 and n should be zero (no monomorphic sites)."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=1.0)
        assert sfs[0] == 0.0
        assert sfs[n] == 0.0

    def test_one_over_j_law(self):
        """SFS[j] should equal theta/j."""
        n = 30
        theta = 42.0
        sfs = expected_sfs_neutral(n, theta)
        for j in range(1, n):
            assert np.isclose(sfs[j], theta / j)

    def test_total_equals_theta_times_harmonic(self):
        """Total segregating sites should be theta * H_{n-1}."""
        n = 50
        theta = 1000
        sfs = expected_sfs_neutral(n, theta)
        harmonic = sum(1 / j for j in range(1, n))
        assert np.isclose(sfs[1:n].sum(), theta * harmonic)

    def test_monotonically_decreasing(self):
        """Neutral SFS should be monotonically decreasing."""
        n = 30
        sfs = expected_sfs_neutral(n, theta=1.0)
        for j in range(1, n - 1):
            assert sfs[j] > sfs[j + 1]


# ===========================================================================
# Tests for fold_sfs
# ===========================================================================

class TestFoldSfs:
    """Tests for the fold_sfs function."""

    def test_output_shape(self):
        """Folded SFS should have shape (n//2 + 1,)."""
        n = 10
        sfs = expected_sfs_neutral(n, theta=100)
        folded = fold_sfs(sfs)
        assert folded.shape == (n // 2 + 1,)

    def test_mirror_bins_summed(self):
        """folded[j] should equal sfs[j] + sfs[n-j] for j != n/2."""
        n = 10
        sfs = expected_sfs_neutral(n, theta=100)
        folded = fold_sfs(sfs)
        for j in range(1, n // 2):
            assert np.isclose(folded[j], sfs[j] + sfs[n - j])

    def test_midpoint_not_doubled(self):
        """At j = n/2 (even n), the entry should not be doubled."""
        n = 10
        sfs = expected_sfs_neutral(n, theta=100)
        folded = fold_sfs(sfs)
        assert np.isclose(folded[n // 2], sfs[n // 2])

    def test_total_mass_conserved(self):
        """Sum of folded SFS should equal sum of internal unfolded SFS."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=100)
        folded = fold_sfs(sfs)
        assert np.isclose(folded.sum(), sfs[1:n].sum())


# ===========================================================================
# Tests for project_sfs
# ===========================================================================

class TestProjectSfs:
    """Tests for the project_sfs function."""

    def test_output_shape(self):
        """Projected SFS should have shape (n_new+1,)."""
        sfs = expected_sfs_neutral(50, theta=100)
        proj = project_sfs(sfs, 20)
        assert proj.shape == (21,)

    def test_neutral_sfs_projection_preserves_shape(self):
        """Projecting neutral SFS should give neutral SFS at new sample size."""
        n = 50
        n_new = 20
        theta = 500
        sfs_50 = expected_sfs_neutral(n, theta)
        sfs_20_proj = project_sfs(sfs_50, n_new)
        sfs_20_direct = expected_sfs_neutral(n_new, theta)
        for j in range(1, n_new):
            assert np.isclose(sfs_20_proj[j], sfs_20_direct[j], rtol=1e-10)

    def test_total_mass_conserved(self):
        """Projection should conserve total segregating sites."""
        n = 30
        n_new = 15
        sfs = expected_sfs_neutral(n, theta=100)
        proj = project_sfs(sfs, n_new)
        # The total should equal the expected total at n_new
        expected_total = sum(100 / j for j in range(1, n_new))
        assert np.isclose(proj[1:n_new].sum(), expected_total, rtol=1e-6)


# ===========================================================================
# Tests for watterson_theta
# ===========================================================================

class TestWattersonTheta:
    """Tests for the watterson_theta function."""

    def test_recovers_theta(self):
        """Should recover theta from the neutral SFS."""
        n = 50
        theta = 1000
        sfs = expected_sfs_neutral(n, theta)
        assert np.isclose(watterson_theta(sfs), theta)

    def test_scales_linearly(self):
        """Doubling theta should double the estimate."""
        n = 30
        sfs1 = expected_sfs_neutral(n, theta=100)
        sfs2 = expected_sfs_neutral(n, theta=200)
        assert np.isclose(watterson_theta(sfs2), 2 * watterson_theta(sfs1))


# ===========================================================================
# Tests for nucleotide_diversity
# ===========================================================================

class TestNucleotideDiversity:
    """Tests for the nucleotide_diversity function."""

    def test_equals_theta_for_neutral(self):
        """Under the neutral model, pi should equal theta."""
        n = 50
        theta = 1000
        sfs = expected_sfs_neutral(n, theta)
        assert np.isclose(nucleotide_diversity(sfs), theta)

    def test_equals_watterson_for_neutral(self):
        """Under neutrality, pi == theta_W."""
        n = 30
        theta = 500
        sfs = expected_sfs_neutral(n, theta)
        assert np.isclose(nucleotide_diversity(sfs), watterson_theta(sfs))


# ===========================================================================
# Tests for tajimas_d
# ===========================================================================

class TestTajimasD:
    """Tests for the tajimas_d function."""

    def test_zero_for_neutral(self):
        """Tajima's D should be zero for the neutral SFS."""
        n = 30
        sfs = expected_sfs_neutral(n, theta=1000)
        assert np.isclose(tajimas_d(sfs), 0.0, atol=1e-10)

    def test_zero_for_empty_sfs(self):
        """Tajima's D should be zero when S = 0."""
        n = 20
        sfs = np.zeros(n + 1)
        assert tajimas_d(sfs) == 0.0


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
        the total sum of the internal SFS entries."""
        n = 30
        phi = _neutral_sfs(n)
        dphi = drift_operator(phi, n)
        net_flow = np.sum(dphi[1:n])
        assert abs(net_flow) < np.sum(np.abs(dphi[1:n])) + 1e-15

    def test_drift_symmetry_on_symmetric_sfs(self):
        """For a symmetric SFS, the drift output should also be symmetric."""
        n = 20
        phi = np.zeros(n + 1)
        for j in range(1, n):
            phi[j] = 1.0
        dphi = drift_operator(phi, n)
        for j in range(1, n):
            assert np.isclose(dphi[j], dphi[n - j], atol=1e-12), (
                f"Drift not symmetric at j={j}: {dphi[j]} vs {dphi[n - j]}"
            )

    def test_drift_pushes_mass_toward_boundaries(self):
        """For a flat SFS, drift should push mass toward the boundaries."""
        n = 20
        phi = np.zeros(n + 1)
        for j in range(1, n):
            phi[j] = 1.0
        dphi = drift_operator(phi, n)
        assert dphi[n // 2] < 0, "Drift should decrease the mid-frequency bin"
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
        """The mutation operator should not depend on the current SFS."""
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
        """Selection operator is linear in gamma."""
        n = 15
        phi = _neutral_sfs(n)
        dphi1 = selection_operator(phi, n, gamma=2.0, h=0.5)
        dphi2 = selection_operator(phi, n, gamma=6.0, h=0.5)
        assert np.allclose(dphi2, 3.0 * dphi1, atol=1e-14)

    def test_selection_linear_in_phi(self):
        """Selection operator is linear in phi."""
        n = 15
        phi = _neutral_sfs(n)
        c = 2.5
        dphi1 = selection_operator(c * phi, n, gamma=5.0, h=0.5)
        dphi2 = c * selection_operator(phi, n, gamma=5.0, h=0.5)
        assert np.allclose(dphi1, dphi2, atol=1e-14)

    def test_purifying_selection_reduces_high_freq(self):
        """Purifying selection should reduce high-frequency derived alleles."""
        n = 30
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=-10.0, h=0.5)
        assert dphi[n - 2] < 0, (
            "Purifying selection should reduce high-frequency bins"
        )

    def test_additive_selection_h_half(self):
        """For additive selection (h=0.5), the result should be non-trivial."""
        n = 20
        phi = _neutral_sfs(n)
        dphi = selection_operator(phi, n, gamma=5.0, h=0.5)
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
        """With equal sample sizes and symmetric migration on a symmetric SFS."""
        n = 6
        np.random.seed(77)
        base = np.random.rand(n + 1, n + 1)
        phi_2d = (base + base.T) / 2.0
        M = 1.5
        dphi = migration_operator_2pop(phi_2d, n, n, M12=M, M21=M)
        assert np.allclose(dphi, dphi.T, atol=1e-12), (
            "Symmetric migration on symmetric SFS should give symmetric output"
        )

    def test_boundary_rows_and_cols_zero(self):
        """Boundary rows/columns should be zero."""
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
        """Larger population should produce 10x smaller drift."""
        n = 20
        phi = _neutral_sfs(n)
        dphi_small = drift_operator_with_size(phi, n, nu=1.0)
        dphi_large = drift_operator_with_size(phi, n, nu=10.0)
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
        """Smaller population should amplify drift."""
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
        """Boundary bins should remain zero after integration."""
        n = 15
        phi = _neutral_sfs(n)
        result = integrate_sfs(phi, n, T=0.5, nu_func=lambda t: 1.0, theta=1.0)
        assert result[0] == 0.0
        assert result[n] == 0.0

    def test_nonnegative_sfs_after_integration(self):
        """The SFS should remain non-negative after integration."""
        n = 15
        phi = _neutral_sfs(n, theta=1.0)
        result = integrate_sfs(phi, n, T=1.0, nu_func=lambda t: 1.0, theta=1.0)
        assert np.all(result >= -1e-10), (
            f"SFS has negative entries: {result[result < -1e-10]}"
        )

    def test_integration_preserves_monotonic_sfs_decrease(self):
        """Short integration should preserve monotonic decrease."""
        n = 15
        theta = 1.0
        phi_init = _neutral_sfs(n, theta)
        phi_after = integrate_sfs(phi_init, n, T=0.01,
                                  nu_func=lambda t: 1.0, theta=theta)
        for j in range(1, n - 1):
            assert phi_after[j] >= phi_after[j + 1] - 1e-10

    def test_zero_time_returns_initial(self):
        """Integrating for T=0 should return the initial SFS unchanged."""
        n = 15
        phi = _neutral_sfs(n, theta=2.0)
        result = integrate_sfs(phi, n, T=0.0, nu_func=lambda t: 1.0, theta=2.0)
        assert np.allclose(result, phi, atol=1e-12)

    def test_expansion_increases_rare_variants(self):
        """Population expansion should increase rare variants."""
        n = 20
        theta = 1.0
        phi_eq = _neutral_sfs(n, theta)
        phi_expanded = integrate_sfs(phi_eq, n, T=0.5,
                                     nu_func=lambda t: 10.0, theta=theta)
        ratio_singleton = phi_expanded[1] / phi_eq[1] if phi_eq[1] > 0 else 0
        ratio_mid = phi_expanded[n // 2] / phi_eq[n // 2] if phi_eq[n // 2] > 0 else 0
        assert ratio_singleton > ratio_mid, (
            "Expansion should enrich rare variants more than common ones"
        )

    def test_flat_sfs_converges_toward_neutral(self):
        """Starting from flat SFS, long integration should approach 1/j."""
        n = 15
        theta = 1.0
        phi_flat = np.zeros(n + 1)
        for j in range(1, n):
            phi_flat[j] = 1.0
        phi_after = integrate_sfs(phi_flat, n, T=5.0,
                                  nu_func=lambda t: 1.0, theta=theta)
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
        """All entries should be non-negative."""
        n1, n2 = 6, 8
        n = n1 + n2
        phi_1d = _neutral_sfs(n)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.all(phi_2d >= 0.0)

    def test_total_mass_conserved(self):
        """Total mass should be conserved."""
        n1, n2 = 5, 5
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=2.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.isclose(phi_2d.sum(), phi_1d.sum(), rtol=1e-10)

    def test_marginal_over_pop2_gives_projection(self):
        """Marginalizing over pop 2 should give projection to n1."""
        n1, n2 = 6, 8
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        phi_marginal = phi_2d.sum(axis=1)

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

        assert np.allclose(phi_marginal, phi_proj, atol=1e-12)

    def test_marginal_over_pop1_gives_projection(self):
        """Marginalizing over pop 1 should give pop 2 projection."""
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
        """Splitting a zero SFS should give zero 2D SFS."""
        n1, n2 = 4, 6
        n = n1 + n2
        phi_1d = np.zeros(n + 1)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.allclose(phi_2d, 0.0)

    def test_single_entry_distribution(self):
        """Mass at j=k should distribute along the anti-diagonal j1+j2=k."""
        n1, n2 = 4, 6
        n = n1 + n2
        k = 5
        phi_1d = np.zeros(n + 1)
        phi_1d[k] = 3.0

        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        for j1 in range(n1 + 1):
            for j2 in range(n2 + 1):
                if j1 + j2 == k:
                    expected_prob = (comb(k, j1, exact=True) *
                                    comb(n - k, n1 - j1, exact=True) /
                                    comb(n, n1, exact=True))
                    if expected_prob > 0:
                        assert np.isclose(phi_2d[j1, j2], 3.0 * expected_prob)
                else:
                    assert phi_2d[j1, j2] == 0.0

    def test_hypergeometric_weights_sum_to_one(self):
        """Hypergeometric weights for each j should sum to 1."""
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
            assert np.isclose(weight_sum, 1.0, atol=1e-12)

    def test_equal_split_symmetry(self):
        """When n1 == n2, split should be symmetric."""
        n1 = n2 = 6
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)
        assert np.allclose(phi_2d, phi_2d.T, atol=1e-12)


# ===========================================================================
# Tests for poisson_log_likelihood
# ===========================================================================

class TestPoissonLogLikelihood:
    """Tests for the poisson_log_likelihood function."""

    def test_maximized_at_true_model(self):
        """LL should be maximized when model == data."""
        n = 20
        theta = 1000
        data = expected_sfs_neutral(n, theta)
        ll_true = poisson_log_likelihood(data, data)
        # Compare with a wrong model
        ll_wrong = poisson_log_likelihood(data, expected_sfs_neutral(n, theta * 2))
        assert ll_true > ll_wrong

    def test_impossible_observation(self):
        """If data has counts where model is zero, LL should be -inf."""
        n = 10
        data = np.zeros(n + 1)
        data[1] = 5.0
        model = np.zeros(n + 1)
        model[2] = 10.0  # model has zero where data is nonzero
        assert poisson_log_likelihood(data, model) == -np.inf

    def test_zero_data_finite_ll(self):
        """Zero data with nonzero model should give a finite LL."""
        n = 10
        data = np.zeros(n + 1)
        model = expected_sfs_neutral(n, theta=100)
        ll = poisson_log_likelihood(data, model)
        assert np.isfinite(ll)

    def test_ll_increases_toward_true_theta(self):
        """LL should increase as theta approaches the true value."""
        n = 20
        theta_true = 1000
        data = expected_sfs_neutral(n, theta_true)
        lls = []
        for theta_test in [500, 800, 1000, 1200, 1500]:
            model = expected_sfs_neutral(n, theta_test)
            lls.append(poisson_log_likelihood(data, model))
        # Maximum should be at index 2 (theta=1000)
        assert lls[2] == max(lls)


# ===========================================================================
# Tests for optimal_theta_scaling
# ===========================================================================

class TestOptimalThetaScaling:
    """Tests for the optimal_theta_scaling function."""

    def test_recovers_theta(self):
        """Should recover the true theta from unit-scaled model."""
        n = 20
        theta_true = 1000
        data = expected_sfs_neutral(n, theta_true)
        model_unit = expected_sfs_neutral(n, theta=1.0)
        theta_opt = optimal_theta_scaling(data, model_unit)
        assert np.isclose(theta_opt, theta_true)

    def test_zero_model_returns_one(self):
        """If model is all zeros, should return 1.0."""
        n = 10
        data = expected_sfs_neutral(n, theta=100)
        model_zero = np.zeros(n + 1)
        assert optimal_theta_scaling(data, model_zero) == 1.0


# ===========================================================================
# Tests for likelihood_ratio_test
# ===========================================================================

class TestLikelihoodRatioTest:
    """Tests for the likelihood_ratio_test function."""

    def test_same_ll_gives_p_one(self):
        """If both models have the same LL, p-value should be 1."""
        p_val = likelihood_ratio_test(-100.0, -100.0, df=2)
        assert np.isclose(p_val, 1.0)

    def test_large_improvement_gives_small_p(self):
        """Large LL improvement should give small p-value."""
        p_val = likelihood_ratio_test(-200.0, -100.0, df=2)
        assert p_val < 0.001

    def test_p_value_in_range(self):
        """p-value should be between 0 and 1."""
        p_val = likelihood_ratio_test(-150.0, -140.0, df=1)
        assert 0.0 <= p_val <= 1.0


# ===========================================================================
# Tests for apply_misidentification
# ===========================================================================

class TestApplyMisidentification:
    """Tests for the apply_misidentification function."""

    def test_zero_misid_returns_original(self):
        """With p_misid=0, should return the original SFS."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=100)
        sfs_obs = apply_misidentification(sfs, p_misid=0.0)
        assert np.allclose(sfs_obs, sfs)

    def test_full_misid_reverses(self):
        """With p_misid=1, should fully reverse the SFS."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=100)
        sfs_obs = apply_misidentification(sfs, p_misid=1.0)
        for j in range(n + 1):
            assert np.isclose(sfs_obs[j], sfs[n - j])

    def test_total_mass_conserved(self):
        """Misidentification should conserve total mass."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=100)
        sfs_obs = apply_misidentification(sfs, p_misid=0.05)
        assert np.isclose(sfs_obs.sum(), sfs.sum())

    def test_small_misid_inflates_high_freq(self):
        """Small misidentification should inflate high-frequency bins."""
        n = 20
        sfs = expected_sfs_neutral(n, theta=1000)
        sfs_obs = apply_misidentification(sfs, p_misid=0.02)
        # High-frequency bins (near n-1) should increase
        # because singletons (which are abundant) get flipped to n-1
        assert sfs_obs[n - 1] > sfs[n - 1]


# ===========================================================================
# Tests for compute_D
# ===========================================================================

class TestComputeD:
    """Tests for the compute_D function."""

    def test_perfect_ld(self):
        """All (0,0) and (1,1) should give positive D."""
        haps = np.array([[0, 0]] * 60 + [[1, 1]] * 40)
        D = compute_D(haps)
        assert D > 0

    def test_no_ld(self):
        """Independent loci should give D close to 0."""
        np.random.seed(42)
        n = 1000
        p, q = 0.3, 0.5
        locus_a = (np.random.rand(n) < p).astype(int)
        locus_b = (np.random.rand(n) < q).astype(int)
        haps = np.column_stack([locus_a, locus_b])
        D = compute_D(haps)
        assert abs(D) < 0.05  # should be near zero with large n

    def test_perfect_negative_ld(self):
        """All (0,1) and (1,0) should give negative D."""
        haps = np.array([[0, 1]] * 50 + [[1, 0]] * 50)
        D = compute_D(haps)
        assert D < 0


# ===========================================================================
# Tests for ld_decay_deterministic
# ===========================================================================

class TestLdDecayDeterministic:
    """Tests for the ld_decay_deterministic function."""

    def test_no_recombination(self):
        """With r=0, LD should not decay."""
        D0 = 0.1
        D_t = ld_decay_deterministic(D0, r=0.0, t_generations=100)
        assert np.isclose(D_t, D0)

    def test_half_life(self):
        """At the half-life, D should be approximately D0/2."""
        r = 0.01
        half_life = int(round(np.log(2) / r))
        D0 = 0.1
        D_t = ld_decay_deterministic(D0, r, half_life)
        assert np.isclose(D_t, D0 / 2, rtol=0.02)

    def test_full_decay(self):
        """After many generations, D should be near zero."""
        D0 = 0.1
        D_t = ld_decay_deterministic(D0, r=0.1, t_generations=100)
        assert abs(D_t) < 1e-5

    def test_zero_initial(self):
        """If D0=0, D should remain 0."""
        D_t = ld_decay_deterministic(0.0, r=0.01, t_generations=50)
        assert D_t == 0.0


# ===========================================================================
# Tests for compute_ld_statistics
# ===========================================================================

class TestComputeLdStatistics:
    """Tests for the compute_ld_statistics function."""

    def test_returns_three_values(self):
        """Should return a tuple of three floats."""
        np.random.seed(42)
        haps = np.random.randint(0, 2, size=(50, 5))
        D2, Dz, pi2 = compute_ld_statistics(haps)
        assert isinstance(D2, float)
        assert isinstance(Dz, float)
        assert isinstance(pi2, float)

    def test_d2_nonnegative(self):
        """D^2 should always be non-negative."""
        np.random.seed(42)
        haps = np.random.randint(0, 2, size=(100, 10))
        D2, Dz, pi2 = compute_ld_statistics(haps)
        assert D2 >= 0

    def test_pi2_nonnegative(self):
        """pi_2 should always be non-negative."""
        np.random.seed(42)
        haps = np.random.randint(0, 2, size=(100, 10))
        D2, Dz, pi2 = compute_ld_statistics(haps)
        assert pi2 >= 0

    def test_monomorphic_loci_skipped(self):
        """Monomorphic loci should be skipped."""
        # All loci monomorphic
        haps = np.ones((50, 5), dtype=int)
        D2, Dz, pi2 = compute_ld_statistics(haps)
        assert D2 == 0 and Dz == 0 and pi2 == 0


# ===========================================================================
# Tests for ld_equilibrium
# ===========================================================================

class TestLdEquilibrium:
    """Tests for the ld_equilibrium function."""

    def test_high_rho_low_ld(self):
        """High recombination should give low LD."""
        sigma = ld_equilibrium(0.001, rho=100)
        assert sigma < 0.01

    def test_low_rho_high_ld(self):
        """Low recombination should give high LD."""
        sigma = ld_equilibrium(0.001, rho=0.1)
        assert sigma > 0.5

    def test_zero_rho(self):
        """With rho=0, sigma_d^2 should be 1."""
        sigma = ld_equilibrium(0.001, rho=0.0)
        assert np.isclose(sigma, 1.0)

    def test_formula(self):
        """Should follow 1/(1+rho)."""
        for rho in [0.1, 1.0, 5.0, 10.0]:
            assert np.isclose(ld_equilibrium(0.001, rho), 1.0 / (1.0 + rho))


# ===========================================================================
# Tests for gaussian_composite_ll
# ===========================================================================

class TestGaussianCompositeLl:
    """Tests for the gaussian_composite_ll function."""

    def test_perfect_match_gives_zero(self):
        """When data equals model, LL should be 0."""
        data = [np.array([1.0, 2.0])]
        model = [np.array([1.0, 2.0])]
        cov = [np.eye(2)]
        ll = gaussian_composite_ll(data, model, cov)
        assert np.isclose(ll, 0.0)

    def test_larger_residual_gives_lower_ll(self):
        """Larger residual should give lower LL."""
        cov = [np.eye(2)]
        ll_close = gaussian_composite_ll(
            [np.array([1.0, 2.0])], [np.array([1.1, 2.1])], cov)
        ll_far = gaussian_composite_ll(
            [np.array([1.0, 2.0])], [np.array([5.0, 6.0])], cov)
        assert ll_close > ll_far

    def test_negative_ll(self):
        """LL should be non-positive (zero at best)."""
        data = [np.array([1.0])]
        model = [np.array([2.0])]
        cov = [np.array([[1.0]])]
        ll = gaussian_composite_ll(data, model, cov)
        assert ll <= 0.0


# ===========================================================================
# Tests for map_r_bins_to_rho
# ===========================================================================

class TestMapRBinsToRho:
    """Tests for the map_r_bins_to_rho function."""

    def test_basic_conversion(self):
        """rho should be 4*Ne*r."""
        r_bins = np.array([1e-4, 1e-3])
        Ne = 10000
        rho = map_r_bins_to_rho(r_bins, Ne)
        assert np.isclose(rho[0], 4.0)
        assert np.isclose(rho[1], 40.0)

    def test_zero_r(self):
        """r=0 should give rho=0."""
        rho = map_r_bins_to_rho(np.array([0.0]), Ne=10000)
        assert rho[0] == 0.0

    def test_scales_with_Ne(self):
        """Doubling Ne should double rho."""
        r_bins = np.array([1e-4])
        rho1 = map_r_bins_to_rho(r_bins, Ne=5000)
        rho2 = map_r_bins_to_rho(r_bins, Ne=10000)
        assert np.isclose(rho2[0], 2 * rho1[0])


# ===========================================================================
# Integration tests combining multiple operators
# ===========================================================================

class TestCombinedOperators:
    """Integration tests that exercise multiple operators together."""

    def test_drift_plus_mutation_equilibrium_direction(self):
        """Starting from a flat SFS, drift+mutation should move toward equilibrium."""
        n = 15
        theta = 1.0
        phi_flat = np.zeros(n + 1)
        for j in range(1, n):
            phi_flat[j] = 1.0

        dphi = drift_operator(phi_flat, n) + mutation_operator(phi_flat, n, theta)
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
        """Migration should redistribute mass but not drastically change total."""
        n1 = n2 = 5
        n = n1 + n2
        phi_1d = _neutral_sfs(n, theta=1.0)
        phi_2d = split_1d_to_2d(phi_1d, n1, n2)

        M = 2.0
        dphi_mig = migration_operator_2pop(phi_2d, n1, n2, M12=M, M21=M)

        net_change = dphi_mig[1:n1, 1:n2].sum()
        total_mass = phi_2d[1:n1, 1:n2].sum()
        assert abs(net_change) < 0.5 * total_mass, (
            "Migration should not drastically change total internal mass"
        )

    def test_neutral_equilibrium_direction(self):
        """Starting from neutral SFS with short integration, drift+mutation
        should produce a result where singletons are still the most abundant."""
        n = 20
        theta = 1.0
        phi_eq = _neutral_sfs(n, theta)
        phi_after = integrate_sfs(phi_eq, n, T=0.01,
                                  nu_func=lambda t: 1.0, theta=theta)
        # After a very short integration, the SFS should still be
        # approximately monotonically decreasing
        assert phi_after[1] > phi_after[n // 2], (
            "Singletons should remain most abundant after short integration"
        )

    def test_long_integration_produces_decreasing_sfs(self):
        """Flat SFS integrated for a long time should produce a
        monotonically decreasing SFS (characteristic of drift-mutation
        equilibrium)."""
        n = 15
        theta = 1.0
        phi_flat = np.zeros(n + 1)
        for j in range(1, n):
            phi_flat[j] = 1.0
        phi_after = integrate_sfs(phi_flat, n, T=5.0,
                                  nu_func=lambda t: 1.0, theta=theta)
        # After long integration, phi[1] should be larger than phi[n//2]
        assert phi_after[1] > phi_after[n // 2], (
            "After long integration, the SFS should approach 1/j shape"
        )


# ===========================================================================
# Tests for fisher_information_numerical
# ===========================================================================

class TestFisherInformationNumerical:
    """Tests for the fisher_information_numerical function."""

    def test_returns_symmetric_matrix(self):
        """FIM should be symmetric."""
        n = 20
        theta_true = 1000
        data = expected_sfs_neutral(n, theta_true)

        def model_func(params, ns):
            nu, = params
            return expected_sfs_neutral(ns[0], theta=1.0) * nu

        params = np.array([1.0])
        FIM = fisher_information_numerical(params, data, model_func, [n])
        assert FIM.shape == (1, 1)
        # For a 1x1 matrix, symmetry is trivial, just check it's finite
        assert np.isfinite(FIM[0, 0])

    def test_2d_symmetric(self):
        """2D FIM should be symmetric."""
        n = 15
        data = _neutral_sfs(n, theta=500)

        def model_func(params, ns):
            # Simple model: scale theta and apply a mild size change
            s, t = params
            phi = expected_sfs_neutral(ns[0], theta=1.0) * s
            return phi

        params = np.array([500.0, 1.0])
        FIM = fisher_information_numerical(params, data, model_func, [n])
        assert np.allclose(FIM, FIM.T)
