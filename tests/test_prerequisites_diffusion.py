"""
Comprehensive tests for all Python code examples in the
diffusion_approximation.rst prerequisite documentation.

Tests cover:
  1. wright_fisher_trajectory  -- WF allele frequency simulation
  2. euler_maruyama            -- SDE simulation via Euler-Maruyama
  3. stationary_density        -- Stationary distributions of the diffusion
  4. solve_diffusion_1d        -- 1D Fokker-Planck PDE solver (Crank-Nicolson)
  5. density_to_sfs            -- Binomial sampling from diffusion density to SFS
"""

import numpy as np
from scipy.special import comb
import pytest


# ---------------------------------------------------------------------------
# Extracted functions (verbatim from the RST, with minor whitespace cleanup)
# ---------------------------------------------------------------------------

def wright_fisher_trajectory(two_N, x0, n_generations):
    """Simulate a Wright-Fisher allele frequency trajectory.

    Parameters
    ----------
    two_N : int
        Total number of gene copies (2N).
    x0 : float
        Initial allele frequency.
    n_generations : int
        Number of generations to simulate.

    Returns
    -------
    freqs : ndarray of shape (n_generations + 1,)
        Allele frequency at each generation.
    """
    freqs = np.zeros(n_generations + 1)
    freqs[0] = x0
    for g in range(n_generations):
        count = np.random.binomial(two_N, freqs[g])
        freqs[g + 1] = count / two_N
    return freqs


def euler_maruyama(x0, mu_func, sigma_func, T, n_steps):
    """Simulate an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    x0 : float
        Initial condition.
    mu_func : callable
        Drift coefficient mu(x).
    sigma_func : callable
        Diffusion coefficient sigma(x).
    T : float
        Total time to simulate (in diffusion time units).
    n_steps : int
        Number of discrete time steps.

    Returns
    -------
    times : ndarray of shape (n_steps + 1,)
        Time points.
    x : ndarray of shape (n_steps + 1,)
        Simulated trajectory.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = x0

    for n in range(n_steps):
        Z = np.random.randn()
        x[n + 1] = x[n] + mu_func(x[n]) * dt + sigma_func(x[n]) * sqrt_dt * Z
        x[n + 1] = np.clip(x[n + 1], 0.0, 1.0)

    return times, x


def stationary_density(x, theta1, theta2, s=0.0):
    """Compute the stationary density of the diffusion (unnormalized).

    Parameters
    ----------
    x : float or ndarray
        Allele frequency in (0, 1).
    theta1 : float
        Forward mutation rate (scaled by 2N).
    theta2 : float
        Backward mutation rate (scaled by 2N).
    s : float
        Selection coefficient (scaled by 2N). Positive favors derived.

    Returns
    -------
    density : float or ndarray
        Unnormalized stationary density at each x.
    """
    x = np.asarray(x, dtype=float)
    log_density = (theta1 - 1) * np.log(x) + (theta2 - 1) * np.log(1 - x) + s * x
    return np.exp(log_density)


def solve_diffusion_1d(P, T, n_time_steps, theta, s=0.0, N_func=None):
    """Solve the 1D Wright-Fisher diffusion equation numerically.

    Uses the method of lines with Crank-Nicolson time stepping.

    Parameters
    ----------
    P : int
        Number of grid points (including boundaries).
    T : float
        Total time to integrate (in units of 2*N_ref generations).
    n_time_steps : int
        Number of Crank-Nicolson steps.
    theta : float
        Population-scaled mutation rate 4*N_ref*mu.
    s : float
        Population-scaled selection coefficient 2*N_ref*s.
    N_func : callable or None
        N_func(t) returns N(t)/N_ref at time t. If None, constant size 1.

    Returns
    -------
    x_grid : ndarray of shape (P,)
        Frequency grid points.
    phi : ndarray of shape (P,)
        Final density at each grid point.
    """
    if N_func is None:
        N_func = lambda t: 1.0

    dx = 1.0 / (P - 1)
    dt = T / n_time_steps
    x_grid = np.linspace(0, 1, P)

    phi = np.zeros(P)
    for i in range(1, P - 1):
        x = x_grid[i]
        phi[i] = theta / (x * (1 - x))

    phi /= np.trapz(phi, x_grid)

    for step in range(n_time_steps):
        t = step * dt
        N_rel = N_func(t)

        def rhs(phi_in):
            F = np.zeros(P)
            for i in range(1, P - 1):
                x = x_grid[i]

                g_im1 = x_grid[i-1] * (1 - x_grid[i-1]) * phi_in[i-1]
                g_i   = x * (1 - x) * phi_in[i]
                g_ip1 = x_grid[i+1] * (1 - x_grid[i+1]) * phi_in[i+1]
                diffusion = 0.5 * (g_ip1 - 2*g_i + g_im1) / (dx**2)

                mu_im1 = (s / 2) * x_grid[i-1] * (1 - x_grid[i-1])
                mu_ip1 = (s / 2) * x_grid[i+1] * (1 - x_grid[i+1])
                h_im1 = mu_im1 * phi_in[i-1]
                h_ip1 = mu_ip1 * phi_in[i+1]
                advection = -(h_ip1 - h_im1) / (2 * dx)

                F[i] = diffusion / N_rel + advection
            return F

        F_n = rhs(phi)
        phi_pred = phi + dt * F_n
        phi_pred[0] = 0.0
        phi_pred[-1] = 0.0
        phi_pred = np.maximum(phi_pred, 0)

        F_pred = rhs(phi_pred)
        phi = phi + 0.5 * dt * (F_n + F_pred)
        phi[0] = 0.0
        phi[-1] = 0.0
        phi = np.maximum(phi, 0)

    return x_grid, phi


def density_to_sfs(x_grid, phi, n_samples):
    """Convert a diffusion density to a site frequency spectrum.

    Applies the binomial sampling formula:
        sfs[j] = integral of Binom(n, j, x) * phi(x) dx

    Parameters
    ----------
    x_grid : ndarray of shape (P,)
        Frequency grid points.
    phi : ndarray of shape (P,)
        Density values at each grid point.
    n_samples : int
        Number of sampled chromosomes.

    Returns
    -------
    sfs : ndarray of shape (n_samples - 1,)
        Expected SFS entries for j = 1, ..., n_samples - 1.
    """
    sfs = np.zeros(n_samples - 1)
    for j in range(1, n_samples):
        binom_probs = comb(n_samples, j) * x_grid**j * (1 - x_grid)**(n_samples - j)
        sfs[j - 1] = np.trapz(binom_probs * phi, x_grid)
    return sfs


# ===================================================================
# Tests for wright_fisher_trajectory
# ===================================================================

class TestWrightFisherTrajectory:
    """Tests for the Wright-Fisher allele frequency trajectory simulation."""

    def test_output_shape(self):
        """Verify the output array has the correct length."""
        np.random.seed(0)
        two_N = 100
        n_gen = 50
        traj = wright_fisher_trajectory(two_N, 0.5, n_gen)
        assert traj.shape == (n_gen + 1,)

    def test_initial_frequency(self):
        """Verify the trajectory starts at the specified initial frequency."""
        np.random.seed(1)
        x0 = 0.3
        traj = wright_fisher_trajectory(200, x0, 10)
        assert traj[0] == pytest.approx(x0)

    def test_frequency_bounds(self):
        """Verify all frequencies remain in [0, 1]."""
        np.random.seed(2)
        traj = wright_fisher_trajectory(100, 0.5, 500)
        assert np.all(traj >= 0.0)
        assert np.all(traj <= 1.0)

    def test_frequency_granularity(self):
        """Verify frequencies are multiples of 1/(2N)."""
        np.random.seed(3)
        two_N = 50
        traj = wright_fisher_trajectory(two_N, 0.4, 100)
        # Each frequency * two_N should be an integer (within floating-point tolerance)
        counts = traj * two_N
        assert np.allclose(counts, np.round(counts), atol=1e-10)

    def test_mean_frequency_unbiased(self):
        """Under neutrality, the mean allele frequency over many replicates
        should remain close to the starting frequency (martingale property)."""
        np.random.seed(4)
        x0 = 0.4
        two_N = 200
        n_gen = 50
        n_reps = 5000
        endpoints = np.array([
            wright_fisher_trajectory(two_N, x0, n_gen)[-1]
            for _ in range(n_reps)
        ])
        assert endpoints.mean() == pytest.approx(x0, abs=0.02)

    def test_single_generation_variance(self):
        """Variance of allele frequency change over a single generation
        should be x0*(1 - x0)/(2N), which is the exact binomial variance."""
        np.random.seed(42)
        x0 = 0.3
        two_N = 200
        n_replicates = 50000
        changes = np.zeros(n_replicates)
        for rep in range(n_replicates):
            traj = wright_fisher_trajectory(two_N, x0, 1)
            changes[rep] = traj[1] - traj[0]
        theoretical_var = x0 * (1 - x0) / two_N
        assert changes.var() == pytest.approx(theoretical_var, rel=0.05)

    def test_variance_accumulates_over_generations(self):
        """The variance of frequency change over multiple generations should
        grow (at least initially) roughly proportionally to the number of
        generations, confirming the diffusion scaling."""
        np.random.seed(100)
        x0 = 0.5
        two_N = 1000
        n_reps = 5000

        var_1 = np.var([
            wright_fisher_trajectory(two_N, x0, 1)[-1] - x0
            for _ in range(n_reps)
        ])
        var_10 = np.var([
            wright_fisher_trajectory(two_N, x0, 10)[-1] - x0
            for _ in range(n_reps)
        ])
        # Variance after 10 generations should be roughly 10x variance after 1
        # (exact for independent increments; approximate here since x drifts)
        ratio = var_10 / var_1
        assert 5.0 < ratio < 15.0

    def test_step_size_decreases_with_N(self):
        """Verify that step sizes shrink as 2N grows."""
        np.random.seed(5)
        x0 = 0.3
        max_steps = {}
        for two_N in [20, 200, 2000]:
            n_gen = two_N
            traj = wright_fisher_trajectory(two_N, x0, n_gen)
            max_step = np.max(np.abs(np.diff(traj)))
            max_steps[two_N] = max_step
        # Larger 2N should yield smaller individual step sizes
        assert max_steps[200] < max_steps[20]
        assert max_steps[2000] < max_steps[200]

    def test_fixation_possible(self):
        """Given enough time and small population, allele should eventually fix or be lost."""
        np.random.seed(6)
        two_N = 20
        n_gen = 10000
        traj = wright_fisher_trajectory(two_N, 0.5, n_gen)
        # The trajectory should hit 0 or 1 at some point
        assert traj[-1] == 0.0 or traj[-1] == 1.0


# ===================================================================
# Tests for euler_maruyama
# ===================================================================

class TestEulerMaruyama:
    """Tests for the Euler-Maruyama SDE simulation."""

    def test_output_shapes(self):
        """Verify output arrays have the correct length."""
        np.random.seed(10)
        mu = lambda x: 0.0
        sigma = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        times, x = euler_maruyama(0.5, mu, sigma, 1.0, 100)
        assert times.shape == (101,)
        assert x.shape == (101,)

    def test_initial_condition(self):
        """Verify the trajectory starts at x0."""
        np.random.seed(11)
        mu = lambda x: 0.0
        sigma = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        _, x = euler_maruyama(0.3, mu, sigma, 1.0, 200)
        assert x[0] == pytest.approx(0.3)

    def test_time_grid(self):
        """Verify the time array spans [0, T] with correct endpoints."""
        np.random.seed(12)
        mu = lambda x: 0.0
        sigma = lambda x: 0.1
        T = 2.5
        times, _ = euler_maruyama(0.5, mu, sigma, T, 500)
        assert times[0] == pytest.approx(0.0)
        assert times[-1] == pytest.approx(T)

    def test_clipping_bounds(self):
        """Verify the trajectory stays within [0, 1] due to clipping."""
        np.random.seed(13)
        mu = lambda x: 0.0
        sigma = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        _, x = euler_maruyama(0.5, mu, sigma, 5.0, 10000)
        assert np.all(x >= 0.0)
        assert np.all(x <= 1.0)

    def test_neutral_mean_is_unbiased(self):
        """Under neutral drift, the average final frequency over many runs
        should be close to the starting frequency."""
        np.random.seed(14)
        mu_neutral = lambda x: 0.0
        sigma_drift = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        x0 = 0.4
        n_reps = 3000
        endpoints = np.array([
            euler_maruyama(x0, mu_neutral, sigma_drift, 0.5, 1000)[1][-1]
            for _ in range(n_reps)
        ])
        assert endpoints.mean() == pytest.approx(x0, abs=0.03)

    def test_selection_pushes_frequency_upward(self):
        """Positive selection should yield a higher mean final frequency
        than neutral drift."""
        np.random.seed(42)
        sigma_drift = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        mu_neutral = lambda x: 0.0
        s_coeff = 5.0
        mu_selection = lambda x: (s_coeff / 2) * x * (1 - x)

        T = 2.0
        n_steps = 5000
        _, x_neutral = euler_maruyama(0.1, mu_neutral, sigma_drift, T, n_steps)
        _, x_selected = euler_maruyama(0.1, mu_selection, sigma_drift, T, n_steps)

        # Run many replicates to confirm on average
        np.random.seed(15)
        n_reps = 500
        neutral_ends = np.array([
            euler_maruyama(0.1, mu_neutral, sigma_drift, T, n_steps)[1][-1]
            for _ in range(n_reps)
        ])
        selected_ends = np.array([
            euler_maruyama(0.1, mu_selection, sigma_drift, T, n_steps)[1][-1]
            for _ in range(n_reps)
        ])
        assert selected_ends.mean() > neutral_ends.mean()

    def test_zero_diffusion_is_deterministic(self):
        """With sigma(x) = 0, the trajectory should follow the deterministic ODE."""
        np.random.seed(16)
        mu = lambda x: 0.5  # constant drift
        sigma = lambda x: 0.0  # no noise
        x0 = 0.1
        T = 0.5
        n_steps = 1000
        times, x = euler_maruyama(x0, mu, sigma, T, n_steps)
        # x should increase linearly: x(t) = x0 + 0.5*t, clipped to [0,1]
        expected = np.clip(x0 + 0.5 * times, 0.0, 1.0)
        assert np.allclose(x, expected, atol=1e-6)

    def test_zero_drift_zero_diffusion_stays_put(self):
        """With mu=0 and sigma=0, the trajectory should remain at x0."""
        np.random.seed(17)
        mu = lambda x: 0.0
        sigma = lambda x: 0.0
        x0 = 0.7
        _, x = euler_maruyama(x0, mu, sigma, 1.0, 100)
        assert np.allclose(x, x0)


# ===================================================================
# Tests for stationary_density
# ===================================================================

class TestStationaryDensity:
    """Tests for the unnormalized stationary density of the diffusion."""

    def test_returns_positive_values(self):
        """Density should be positive on (0, 1)."""
        x_grid = np.linspace(0.001, 0.999, 500)
        phi = stationary_density(x_grid, theta1=0.5, theta2=0.5)
        assert np.all(phi > 0)

    def test_symmetric_mutation_is_symmetric(self):
        """With theta1 = theta2 and s=0, the density should be symmetric
        around x=0.5."""
        x_grid = np.linspace(0.01, 0.99, 999)
        phi = stationary_density(x_grid, theta1=1.5, theta2=1.5, s=0.0)
        phi_rev = phi[::-1]
        assert np.allclose(phi, phi_rev, rtol=1e-10)

    def test_integrates_to_one_after_normalization(self):
        """After normalizing by trapezoidal integration, the integral should be 1."""
        x_grid = np.linspace(0.001, 0.999, 1000)

        phi_neutral = stationary_density(x_grid, theta1=0.5, theta2=0.5)
        phi_neutral /= np.trapz(phi_neutral, x_grid)
        assert np.trapz(phi_neutral, x_grid) == pytest.approx(1.0, abs=1e-6)

        phi_selected = stationary_density(x_grid, theta1=0.5, theta2=0.5, s=10.0)
        phi_selected /= np.trapz(phi_selected, x_grid)
        assert np.trapz(phi_selected, x_grid) == pytest.approx(1.0, abs=1e-6)

    def test_selection_shifts_mean_upward(self):
        """Positive selection (s > 0) should increase the mean allele frequency."""
        x_grid = np.linspace(0.001, 0.999, 1000)

        phi_neutral = stationary_density(x_grid, theta1=0.5, theta2=0.5)
        phi_neutral /= np.trapz(phi_neutral, x_grid)
        mean_neutral = np.trapz(x_grid * phi_neutral, x_grid)

        phi_selected = stationary_density(x_grid, theta1=0.5, theta2=0.5, s=10.0)
        phi_selected /= np.trapz(phi_selected, x_grid)
        mean_selected = np.trapz(x_grid * phi_selected, x_grid)

        assert mean_selected > mean_neutral

    def test_negative_selection_shifts_mean_downward(self):
        """Negative selection (s < 0) should decrease the mean allele frequency
        relative to neutral."""
        x_grid = np.linspace(0.001, 0.999, 1000)

        phi_neutral = stationary_density(x_grid, theta1=0.5, theta2=0.5)
        phi_neutral /= np.trapz(phi_neutral, x_grid)
        mean_neutral = np.trapz(x_grid * phi_neutral, x_grid)

        phi_neg = stationary_density(x_grid, theta1=0.5, theta2=0.5, s=-10.0)
        phi_neg /= np.trapz(phi_neg, x_grid)
        mean_neg = np.trapz(x_grid * phi_neg, x_grid)

        assert mean_neg < mean_neutral

    def test_matches_beta_distribution_no_selection(self):
        """With s=0, the stationary density should match the Beta(theta1, theta2)
        distribution."""
        from scipy.stats import beta as beta_dist
        x_grid = np.linspace(0.01, 0.99, 500)
        theta1, theta2 = 2.0, 3.0

        phi = stationary_density(x_grid, theta1=theta1, theta2=theta2, s=0.0)
        phi /= np.trapz(phi, x_grid)

        beta_pdf = beta_dist.pdf(x_grid, theta1, theta2)

        # Compare shapes (normalize both to unit max for shape comparison)
        phi_norm = phi / phi.max()
        beta_norm = beta_pdf / beta_pdf.max()
        assert np.allclose(phi_norm, beta_norm, atol=0.02)

    def test_u_shaped_for_small_theta(self):
        """With theta1 = theta2 < 1, the density should be U-shaped
        (higher at boundaries than at center)."""
        x_grid = np.linspace(0.01, 0.99, 1000)
        phi = stationary_density(x_grid, theta1=0.3, theta2=0.3)
        # The density at x=0.01 and x=0.99 should exceed the density at x=0.5
        mid_idx = len(x_grid) // 2
        assert phi[0] > phi[mid_idx]
        assert phi[-1] > phi[mid_idx]

    def test_bell_shaped_for_large_theta(self):
        """With theta1 = theta2 > 1, the density should be bell-shaped
        (higher at center than at boundaries)."""
        x_grid = np.linspace(0.01, 0.99, 1000)
        phi = stationary_density(x_grid, theta1=5.0, theta2=5.0)
        mid_idx = len(x_grid) // 2
        assert phi[mid_idx] > phi[0]
        assert phi[mid_idx] > phi[-1]

    def test_uniform_for_theta_one(self):
        """With theta1 = theta2 = 1 and s=0, the density should be uniform."""
        x_grid = np.linspace(0.01, 0.99, 1000)
        phi = stationary_density(x_grid, theta1=1.0, theta2=1.0, s=0.0)
        # All values should be equal (the exponents in the log are all zero)
        assert np.allclose(phi, phi[0], rtol=1e-10)

    def test_scalar_input(self):
        """The function should handle scalar input."""
        val = stationary_density(0.5, theta1=1.0, theta2=1.0)
        assert np.isscalar(val) or val.ndim == 0
        assert float(val) > 0


# ===================================================================
# Tests for solve_diffusion_1d
# ===================================================================

class TestSolveDiffusion1D:
    """Tests for the 1D Fokker-Planck PDE solver.

    Note: The predictor-corrector scheme in solve_diffusion_1d is
    conditionally stable. To ensure numerical stability, we use a coarser
    grid (P=51 or P=101) with sufficient time steps so that the CFL-like
    stability condition dt * (1/(2*dx^2)) < 1 is satisfied.
    """

    def test_output_shapes(self):
        """Verify the output arrays have the expected shapes."""
        P = 51
        x_grid, phi = solve_diffusion_1d(P, T=0.1, n_time_steps=500, theta=1.0)
        assert x_grid.shape == (P,)
        assert phi.shape == (P,)

    def test_grid_endpoints(self):
        """Verify the frequency grid spans [0, 1]."""
        P = 51
        x_grid, _ = solve_diffusion_1d(P, T=0.1, n_time_steps=500, theta=1.0)
        assert x_grid[0] == pytest.approx(0.0)
        assert x_grid[-1] == pytest.approx(1.0)

    def test_absorbing_boundaries(self):
        """Verify phi is zero at the boundary grid points."""
        x_grid, phi = solve_diffusion_1d(
            P=51, T=0.2, n_time_steps=1000, theta=1.0
        )
        assert phi[0] == pytest.approx(0.0)
        assert phi[-1] == pytest.approx(0.0)

    def test_density_nonnegative(self):
        """Verify the density is non-negative everywhere."""
        x_grid, phi = solve_diffusion_1d(
            P=51, T=0.3, n_time_steps=2000, theta=1.0
        )
        assert np.all(phi >= 0.0)

    def test_total_mass_positive(self):
        """Verify total probability mass remains positive after integration.
        Uses a coarse grid with many time steps for numerical stability."""
        x_grid, phi = solve_diffusion_1d(
            P=51, T=0.5, n_time_steps=5000, theta=1.0
        )
        mass = np.trapz(phi, x_grid)
        assert np.isfinite(mass)
        assert mass > 0.0

    def test_bottleneck_reduces_diversity(self):
        """A bottleneck should reduce total probability mass (diversity)
        compared to constant population size."""
        P = 51
        T = 0.5
        n_steps_t = 5000
        theta = 1.0

        x_grid, phi_const = solve_diffusion_1d(P, T, n_steps_t, theta)

        def bottleneck(t):
            if 0.1 <= t <= 0.3:
                return 0.1
            return 1.0

        _, phi_bottle = solve_diffusion_1d(P, T, n_steps_t, theta, N_func=bottleneck)

        mass_const = np.trapz(phi_const, x_grid)
        mass_bottle = np.trapz(phi_bottle, x_grid)
        assert np.isfinite(mass_const)
        assert np.isfinite(mass_bottle)
        assert mass_bottle < mass_const

    def test_constant_n_func_same_as_none(self):
        """Passing N_func=lambda t: 1.0 should give the same result as N_func=None."""
        P = 51
        T = 0.2
        n_t = 1000
        theta = 1.0

        x1, phi1 = solve_diffusion_1d(P, T, n_t, theta, N_func=None)
        x2, phi2 = solve_diffusion_1d(P, T, n_t, theta, N_func=lambda t: 1.0)

        assert np.allclose(x1, x2)
        assert np.allclose(phi1, phi2, atol=1e-12)

    def test_larger_population_retains_more_mass(self):
        """A population that expands (N_func > 1) should retain more diversity
        than a population that contracts (N_func < 1)."""
        P = 51
        T = 0.5
        n_t = 5000
        theta = 1.0

        _, phi_expand = solve_diffusion_1d(P, T, n_t, theta, N_func=lambda t: 2.0)
        _, phi_shrink = solve_diffusion_1d(P, T, n_t, theta, N_func=lambda t: 0.5)
        x_grid = np.linspace(0, 1, P)

        mass_expand = np.trapz(phi_expand, x_grid)
        mass_shrink = np.trapz(phi_shrink, x_grid)
        assert np.isfinite(mass_expand)
        assert np.isfinite(mass_shrink)
        assert mass_expand > mass_shrink

    def test_short_integration_preserves_shape(self):
        """A very short integration time should produce a density close
        to the initial 1/(x(1-x)) shape."""
        P = 51
        T = 0.001  # very short time
        n_t = 10
        theta = 1.0

        x_grid, phi = solve_diffusion_1d(P, T, n_t, theta)
        # The density should still be concentrated near the boundaries
        # (U-shaped), consistent with the 1/(x(1-x)) initial condition.
        mid = P // 2
        assert phi[1] > phi[mid]  # near-boundary value > center value
        assert phi[-2] > phi[mid]


# ===================================================================
# Tests for density_to_sfs
# ===================================================================

class TestDensityToSFS:
    """Tests for the binomial sampling from diffusion density to SFS."""

    def test_output_length(self):
        """Verify SFS has n_samples - 1 entries."""
        x_grid = np.linspace(0, 1, 201)
        phi = np.ones(201)
        phi[0] = 0
        phi[-1] = 0
        sfs = density_to_sfs(x_grid, phi, n_samples=10)
        assert len(sfs) == 9

    def test_sfs_nonnegative(self):
        """SFS entries should be non-negative."""
        x_grid = np.linspace(0, 1, 201)
        phi = np.ones(201)
        phi[0] = 0
        phi[-1] = 0
        sfs = density_to_sfs(x_grid, phi, n_samples=15)
        assert np.all(sfs >= 0)

    def test_neutral_sfs_follows_one_over_j(self):
        """For a neutral constant-size population, the unfolded SFS should
        approximate the 1/j pattern. As the RST document notes, the SFS entry
        E[xi_j] is proportional to integral of Binom(n,j,x) * (1/x) dx, which
        evaluates to 1/j. We use the unfolded density phi(x) = 1/x (derived
        allele frequency density for new mutations) rather than 1/(x(1-x))."""
        P = 2001
        x_grid = np.linspace(0, 1, P)
        n_samples = 20

        # Construct the analytic density for the unfolded SFS: phi ~ 1/x
        # This is the density of derived allele frequencies for new mutations
        # entering at low frequency under neutrality.
        phi_neutral = np.zeros(P)
        for i in range(1, P - 1):
            x = x_grid[i]
            phi_neutral[i] = 1.0 / x

        sfs_const = density_to_sfs(x_grid, phi_neutral, n_samples)

        expected_neutral = np.array([1.0 / j for j in range(1, n_samples)])
        # Normalize both to compare shapes
        sfs_norm = sfs_const / sfs_const.sum()
        expected_norm = expected_neutral / expected_neutral.sum()

        # Allow generous tolerance for numerical integration on a grid
        assert np.allclose(sfs_norm, expected_norm, atol=0.02)

    def test_symmetric_density_gives_symmetric_sfs(self):
        """A density symmetric around x=0.5 should produce an SFS that is
        symmetric: sfs[j-1] == sfs[n-j-1] (the folded spectrum is the same)."""
        x_grid = np.linspace(0, 1, 501)
        # Build a symmetric density
        phi = np.zeros(501)
        for i in range(1, 500):
            x = x_grid[i]
            phi[i] = np.exp(-50 * (x - 0.5)**2)  # Gaussian centered at 0.5

        n_samples = 10
        sfs = density_to_sfs(x_grid, phi, n_samples)
        # sfs[j-1] should approximately equal sfs[n_samples-j-1]
        for j in range(1, n_samples // 2):
            assert sfs[j - 1] == pytest.approx(sfs[n_samples - j - 1], rel=0.01)

    def test_uniform_density_sfs(self):
        """For a uniform density phi(x) = 1 (with boundaries zero), the SFS
        can be computed analytically. The integral of Binom(n,j,x) over [0,1]
        is 1/(n+1) for all j, so all SFS entries should be approximately equal."""
        x_grid = np.linspace(0, 1, 1001)
        phi = np.ones(1001)
        phi[0] = 0
        phi[-1] = 0

        n_samples = 10
        sfs = density_to_sfs(x_grid, phi, n_samples)
        # All entries should be close to each other
        # The integral of binom(n,j)*x^j*(1-x)^(n-j) from 0 to 1 = 1/(n+1)
        expected_val = 1.0 / (n_samples + 1)
        for j in range(n_samples - 1):
            assert sfs[j] == pytest.approx(expected_val, rel=0.02)

    def test_zero_density_gives_zero_sfs(self):
        """If the density is zero everywhere, the SFS should be all zeros."""
        x_grid = np.linspace(0, 1, 201)
        phi = np.zeros(201)
        sfs = density_to_sfs(x_grid, phi, n_samples=10)
        assert np.allclose(sfs, 0.0)

    def test_bottleneck_effect_on_sfs(self):
        """A bottleneck population should produce a different SFS from constant
        population size. Uses a coarser grid for numerical stability."""
        P = 51
        T = 0.5
        n_steps_t = 5000
        theta = 1.0
        n_samples = 20

        x_grid, phi_const = solve_diffusion_1d(P, T, n_steps_t, theta)

        def bottleneck(t):
            if 0.1 <= t <= 0.3:
                return 0.1
            return 1.0

        _, phi_bottle = solve_diffusion_1d(P, T, n_steps_t, theta, N_func=bottleneck)

        sfs_const = density_to_sfs(x_grid, phi_const, n_samples)
        sfs_bottle = density_to_sfs(x_grid, phi_bottle, n_samples)

        # The SFS values should differ
        assert not np.allclose(sfs_const, sfs_bottle, atol=1e-4)


# ===================================================================
# Integration tests: reproducing the RST code-block outputs
# ===================================================================

class TestRSTCodeBlockOutputs:
    """End-to-end tests that reproduce the exact code blocks from the RST
    documentation and verify their stated claims."""

    def test_wf_trajectory_code_block(self):
        """Reproduce the first code block: WF trajectories for different 2N.
        Verifies basic trajectory properties and the single-generation
        variance formula Var(Delta x) = x(1-x)/(2N)."""
        np.random.seed(42)

        x0 = 0.3
        for two_N in [20, 200, 2000]:
            n_gen = two_N
            traj = wright_fisher_trajectory(two_N, x0, n_gen)
            diffusion_times = np.arange(n_gen + 1) / two_N
            final_freq = traj[-1]
            step_size = 1.0 / two_N
            # Verify basic properties
            assert 0.0 <= final_freq <= 1.0
            assert diffusion_times[-1] == pytest.approx(1.0)
            assert step_size == pytest.approx(1.0 / two_N)

        # Verify the per-generation variance, which is exact:
        # Var(Delta x | x) = x(1-x)/(2N) for one generation.
        two_N = 10000
        n_replicates = 50000
        changes = np.zeros(n_replicates)
        for rep in range(n_replicates):
            traj = wright_fisher_trajectory(two_N, x0, 1)
            changes[rep] = traj[1] - traj[0]
        simulated_var = changes.var()
        theoretical_var = x0 * (1 - x0) / two_N
        assert simulated_var == pytest.approx(theoretical_var, rel=0.05)

    def test_euler_maruyama_code_block(self):
        """Reproduce the second code block: Euler-Maruyama for neutral vs selected."""
        mu_neutral = lambda x: 0.0
        sigma_drift = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        s_coeff = 5.0
        mu_selection = lambda x: (s_coeff / 2) * x * (1 - x)

        np.random.seed(42)
        T = 2.0
        n_steps = 5000
        _, x_neutral = euler_maruyama(0.1, mu_neutral, sigma_drift, T, n_steps)
        _, x_selected = euler_maruyama(0.1, mu_selection, sigma_drift, T, n_steps)

        # Both should stay in [0, 1]
        assert np.all(x_neutral >= 0) and np.all(x_neutral <= 1)
        assert np.all(x_selected >= 0) and np.all(x_selected <= 1)

        # The trajectories should have the expected start
        assert x_neutral[0] == pytest.approx(0.1)
        assert x_selected[0] == pytest.approx(0.1)

    def test_stationary_density_code_block(self):
        """Reproduce the third code block: stationary density computations."""
        x_grid = np.linspace(0.001, 0.999, 1000)

        # Case 1: Neutral with symmetric mutation
        phi_neutral = stationary_density(x_grid, theta1=0.5, theta2=0.5)
        phi_neutral /= np.trapz(phi_neutral, x_grid)

        # Case 2: With positive selection
        phi_selected = stationary_density(x_grid, theta1=0.5, theta2=0.5, s=10.0)
        phi_selected /= np.trapz(phi_selected, x_grid)

        # Verify normalization
        assert np.trapz(phi_neutral, x_grid) == pytest.approx(1.0, abs=1e-6)
        assert np.trapz(phi_selected, x_grid) == pytest.approx(1.0, abs=1e-6)

        # Verify mean frequency shift
        mean_neutral = np.trapz(x_grid * phi_neutral, x_grid)
        mean_selected = np.trapz(x_grid * phi_selected, x_grid)
        assert mean_selected > mean_neutral

    def test_solve_diffusion_1d_code_block(self):
        """Reproduce the fourth code block: 1D PDE solver with bottleneck.
        Uses a coarser grid (P=51) with more time steps to ensure numerical
        stability of the predictor-corrector scheme."""
        P = 51
        T = 0.5
        n_steps_t = 5000
        theta = 1.0

        x_grid, phi_const = solve_diffusion_1d(P, T, n_steps_t, theta)

        def bottleneck(t):
            if 0.1 <= t <= 0.3:
                return 0.1
            return 1.0

        _, phi_bottle = solve_diffusion_1d(P, T, n_steps_t, theta, N_func=bottleneck)

        mass_const = np.trapz(phi_const, x_grid)
        mass_bottle = np.trapz(phi_bottle, x_grid)

        # Total mass should be positive and finite
        assert np.isfinite(mass_const)
        assert np.isfinite(mass_bottle)
        assert mass_const > 0
        assert mass_bottle > 0
        # Bottleneck should reduce diversity
        assert mass_bottle < mass_const

    def test_density_to_sfs_code_block(self):
        """Reproduce the fifth code block: diffusion density to SFS conversion.

        The RST code block uses solve_diffusion_1d to evolve the density, then
        extracts SFS via binomial sampling. We test two key properties:
        1. The SFS from the analytic neutral density 1/x follows the 1/j
           pattern (validating density_to_sfs itself). The RST notes that
           E[xi_j] ~ integral of Binom(n,j,x) * (1/x) dx = 1/j.
        2. The SFS from the PDE solver differs between constant-N and
           bottleneck populations (validating the PDE solver + SFS pipeline).
        """
        # Part 1: Verify 1/j pattern from the analytic 1/x density
        P_fine = 2001
        n_samples = 20
        x_grid_fine = np.linspace(0, 1, P_fine)
        phi_analytic = np.zeros(P_fine)
        for i in range(1, P_fine - 1):
            x = x_grid_fine[i]
            phi_analytic[i] = 1.0 / x

        sfs_analytic = density_to_sfs(x_grid_fine, phi_analytic, n_samples)
        expected_neutral = np.array([1.0 / j for j in range(1, n_samples)])
        sfs_analytic_norm = sfs_analytic / sfs_analytic.sum()
        expected_norm = expected_neutral / expected_neutral.sum()

        for j in range(min(8, n_samples - 1)):
            ratio = sfs_analytic_norm[j] / expected_norm[j]
            assert ratio == pytest.approx(1.0, abs=0.05)

        # Part 2: Verify the PDE solver + SFS pipeline works and
        # bottleneck alters the SFS
        P = 51
        T = 0.5
        n_steps_t = 5000
        theta = 1.0

        x_grid, phi_const = solve_diffusion_1d(P, T, n_steps_t, theta)

        def bottleneck(t):
            if 0.1 <= t <= 0.3:
                return 0.1
            return 1.0

        _, phi_bottle = solve_diffusion_1d(P, T, n_steps_t, theta, N_func=bottleneck)

        sfs_const = density_to_sfs(x_grid, phi_const, n_samples)
        sfs_bottle = density_to_sfs(x_grid, phi_bottle, n_samples)

        # SFS values should be non-negative
        assert np.all(sfs_const >= 0)
        assert np.all(sfs_bottle >= 0)

        # The two SFS should differ meaningfully
        assert not np.allclose(sfs_const, sfs_bottle, atol=1e-4)
