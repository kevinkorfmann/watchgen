"""Independent validation for docs/prerequisites/diffusion_approximation.rst."""

from math import comb
from pathlib import Path

import numpy as np
import pytest


def execute_rst_python_blocks(path):
    """Execute every Python code block exactly as published."""
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


def wright_fisher_trajectory(two_N, x0, n_generations):
    if (
        not isinstance(two_N, (int, np.integer))
        or two_N <= 0
        or not 0 <= x0 <= 1
        or not isinstance(n_generations, (int, np.integer))
        or n_generations < 0
    ):
        raise ValueError
    freqs = np.zeros(n_generations + 1)
    freqs[0] = x0
    for generation in range(n_generations):
        freqs[generation + 1] = np.random.binomial(two_N, freqs[generation]) / two_N
    return freqs


def euler_maruyama(x0, mu_func, sigma_func, T, n_steps):
    if not 0 <= x0 <= 1 or T < 0 or n_steps <= 0:
        raise ValueError
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    path = np.zeros(n_steps + 1)
    path[0] = x0
    for step in range(n_steps):
        path[step + 1] = (
            path[step]
            + mu_func(path[step]) * dt
            + sigma_func(path[step]) * np.sqrt(dt) * np.random.randn()
        )
        path[step + 1] = np.clip(path[step + 1], 0.0, 1.0)
    return times, path


def stationary_density(x, theta1, theta2, gamma=0.0):
    x = np.asarray(x, dtype=float)
    if theta1 <= 0 or theta2 <= 0 or np.any((x <= 0) | (x >= 1)):
        raise ValueError
    return np.exp(
        (theta1 - 1) * np.log(x)
        + (theta2 - 1) * np.log1p(-x)
        + 2 * gamma * x
    )


def solve_neutral_diffusion(P, T, x0, cfl=0.9):
    if P < 3 or T < 0 or not 0 <= x0 <= 1 or not 0 < cfl <= 1:
        raise ValueError
    dx = 1.0 / (P - 1)
    grid = np.linspace(0, 1, P)
    rates = 0.5 * grid * (1 - grid) / dx**2
    max_total_rate = 2 * rates.max()
    n_steps = max(1, int(np.ceil(T * max_total_rate / cfl)))
    dt = T / n_steps
    jump = dt * rates
    probability = np.zeros(P)
    coordinate = x0 / dx
    lower = min(int(np.floor(coordinate)), P - 1)
    upper = min(lower + 1, P - 1)
    fraction = coordinate - lower
    probability[lower] += 1 - fraction
    probability[upper] += fraction
    for _ in range(n_steps):
        updated = probability * (1 - 2 * jump)
        updated[:-1] += probability[1:] * jump[1:]
        updated[1:] += probability[:-1] * jump[:-1]
        probability = updated
    return grid, probability


def density_to_sfs(x_grid, phi, n_samples):
    x_grid = np.asarray(x_grid, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if (
        x_grid.ndim != 1
        or phi.shape != x_grid.shape
        or n_samples < 2
        or np.any(np.diff(x_grid) <= 0)
        or x_grid[0] < 0
        or x_grid[-1] > 1
        or np.any(phi < 0)
    ):
        raise ValueError
    sfs = np.zeros(n_samples - 1)
    for j in range(1, n_samples):
        kernel = comb(n_samples, j) * x_grid**j * (1 - x_grid) ** (n_samples - j)
        sfs[j - 1] = np.trapezoid(kernel * phi, x_grid)
    return sfs


class TestWrightFisherMoments:
    def test_one_generation_matches_exact_binomial_moments(self):
        two_N, x = 20, 0.35
        counts = np.arange(two_N + 1)
        probabilities = np.array(
            [comb(two_N, k) * x**k * (1 - x) ** (two_N - k) for k in counts]
        )
        frequencies = counts / two_N
        assert np.dot(probabilities, frequencies) == pytest.approx(x, abs=1e-14)
        variance = np.dot(probabilities, (frequencies - x) ** 2)
        assert variance == pytest.approx(x * (1 - x) / two_N, abs=1e-14)

    @pytest.mark.parametrize("two_N", [50, 500, 5000])
    def test_finite_variance_converges_to_diffusion_limit(self, two_N):
        x0, tau = 0.3, 0.4
        generations = round(tau * two_N)
        finite = x0 * (1 - x0) * (1 - (1 - 1 / two_N) ** generations)
        limit = x0 * (1 - x0) * (1 - np.exp(-tau))
        assert abs(finite - limit) < 0.002

    def test_monte_carlo_endpoint_matches_exact_finite_variance(self):
        rng = np.random.default_rng(20260901)
        two_N, x0, generations, replicates = 400, 0.3, 80, 50_000
        counts = np.full(replicates, round(two_N * x0))
        for _ in range(generations):
            counts = rng.binomial(two_N, counts / two_N)
        endpoints = counts / two_N
        expected = x0 * (1 - x0) * (1 - (1 - 1 / two_N) ** generations)
        assert endpoints.mean() == pytest.approx(x0, abs=0.003)
        assert endpoints.var() == pytest.approx(expected, rel=0.025)

    def test_trajectory_is_discrete_bounded_and_absorbing(self):
        np.random.seed(7)
        trajectory = wright_fisher_trajectory(40, 0.5, 100)
        assert np.all((trajectory >= 0) & (trajectory <= 1))
        np.testing.assert_allclose(trajectory * 40, np.round(trajectory * 40))
        assert wright_fisher_trajectory(40, 0.0, 10).tolist() == [0.0] * 11
        assert wright_fisher_trajectory(40, 1.0, 10).tolist() == [1.0] * 11

    @pytest.mark.parametrize("args", [(0, 0.5, 1), (10, -0.1, 1), (10, 0.5, -1)])
    def test_invalid_trajectory_inputs(self, args):
        with pytest.raises(ValueError):
            wright_fisher_trajectory(*args)


class TestEulerMaruyamaGuardrails:
    def test_zero_noise_matches_euler_deterministic_update(self):
        times, path = euler_maruyama(0.2, lambda x: 0.1, lambda x: 0.0, 0.5, 100)
        np.testing.assert_allclose(path, 0.2 + 0.1 * times, atol=1e-14)

    def test_boundaries_are_absorbing_for_neutral_coefficients(self):
        sigma = lambda x: np.sqrt(max(x * (1 - x), 0.0))
        for boundary in [0.0, 1.0]:
            _, path = euler_maruyama(boundary, lambda x: 0.0, sigma, 1.0, 100)
            np.testing.assert_array_equal(path, np.full(101, boundary))


class TestStationaryDensity:
    def test_neutral_kernel_is_beta_kernel(self):
        grid = np.linspace(0.01, 0.99, 100)
        result = stationary_density(grid, 2.5, 3.5)
        expected = grid**1.5 * (1 - grid) ** 2.5
        np.testing.assert_allclose(result, expected, rtol=1e-14)

    def test_selection_is_exact_exponential_tilt(self):
        grid = np.linspace(0.05, 0.95, 50)
        neutral = stationary_density(grid, 1.2, 0.8)
        selected = stationary_density(grid, 1.2, 0.8, gamma=1.7)
        np.testing.assert_allclose(selected / neutral, np.exp(3.4 * grid), rtol=1e-14)

    def test_log_derivative_satisfies_zero_flux_stationary_equation(self):
        theta1, theta2, gamma = 1.3, 2.1, -0.4
        grid = np.linspace(0.1, 0.9, 10_000)
        density = stationary_density(grid, theta1, theta2, gamma)
        numerical = np.gradient(np.log(density), grid)
        expected = (theta1 - 1) / grid - (theta2 - 1) / (1 - grid) + 2 * gamma
        np.testing.assert_allclose(
            numerical[10:-10], expected[10:-10], rtol=2e-6, atol=2e-6
        )


class TestNeutralDiffusionSolver:
    @pytest.mark.parametrize("P,T,x0", [(51, 0, 0.31), (51, 0.2, 0.3), (101, 1, 0.7)])
    def test_probability_and_martingale_invariants(self, P, T, x0):
        grid, probability = solve_neutral_diffusion(P, T, x0)
        assert np.all(probability >= 0)
        assert probability.sum() == pytest.approx(1.0, abs=2e-13)
        assert np.dot(grid, probability) == pytest.approx(x0, abs=2e-13)

    @pytest.mark.parametrize("P", [51, 101, 201])
    def test_variance_matches_wright_fisher_diffusion_moment(self, P):
        T, x0 = 0.4, 0.3
        grid, probability = solve_neutral_diffusion(P, T, x0)
        variance = np.dot((grid - x0) ** 2, probability)
        expected = x0 * (1 - x0) * (1 - np.exp(-T))
        assert variance == pytest.approx(expected, rel=0.006)

    def test_boundary_atoms_accumulate(self):
        masses = []
        for T in [0.1, 0.5, 1.0]:
            _, probability = solve_neutral_diffusion(101, T, 0.3)
            masses.append(probability[0] + probability[-1])
        assert masses[0] < masses[1] < masses[2]

    def test_discrete_generator_matches_first_two_polynomials(self):
        P = 101
        grid = np.linspace(0, 1, P)
        dx = 1 / (P - 1)
        rate = 0.5 * grid * (1 - grid) / dx**2
        linear_generator = rate[1:-1] * (
            grid[2:] - 2 * grid[1:-1] + grid[:-2]
        )
        quadratic_generator = rate[1:-1] * (
            grid[2:] ** 2 - 2 * grid[1:-1] ** 2 + grid[:-2] ** 2
        )
        np.testing.assert_allclose(linear_generator, 0, atol=2e-13)
        np.testing.assert_allclose(
            quadratic_generator, grid[1:-1] * (1 - grid[1:-1]), atol=2e-13
        )


class TestSFSProjection:
    @pytest.mark.parametrize("n", [4, 10, 20])
    def test_neutral_theta_over_x_projects_to_theta_over_j(self, n):
        theta = 1.7
        grid = np.linspace(1e-7, 1, 200_001)
        sfs = density_to_sfs(grid, theta / grid, n)
        expected = theta / np.arange(1, n)
        np.testing.assert_allclose(sfs, expected, atol=4e-6)

    def test_linearity(self):
        grid = np.linspace(0.001, 1, 10_000)
        density = 1 / grid
        np.testing.assert_allclose(
            density_to_sfs(grid, 3 * density, 8),
            3 * density_to_sfs(grid, density, 8),
        )

    @pytest.mark.parametrize(
        "grid,density,n",
        [([0.1, 0.2], [1], 4), ([0.2, 0.1], [1, 1], 4), ([0.1, 0.2], [1, -1], 4)],
    )
    def test_invalid_inputs(self, grid, density, n):
        with pytest.raises(ValueError):
            density_to_sfs(grid, density, n)


class TestPublishedCodeParity:
    def test_all_published_blocks_execute_and_match_validated_functions(self):
        rst = Path(__file__).parents[1] / "docs/prerequisites/diffusion_approximation.rst"
        published = execute_rst_python_blocks(rst)
        for name in [
            "wright_fisher_trajectory",
            "euler_maruyama",
            "stationary_density",
            "solve_neutral_diffusion",
            "density_to_sfs",
        ]:
            assert name in published
        grid = np.linspace(0.1, 0.9, 9)
        np.testing.assert_allclose(
            published["stationary_density"](grid, 1.2, 2.1, 0.3),
            stationary_density(grid, 1.2, 2.1, 0.3),
        )
        published_grid, published_probability = published["solve_neutral_diffusion"](
            51, 0.2, 0.3
        )
        test_grid, test_probability = solve_neutral_diffusion(51, 0.2, 0.3)
        np.testing.assert_array_equal(published_grid, test_grid)
        np.testing.assert_allclose(published_probability, test_probability)

    def test_methodological_guardrails_remain_in_text(self):
        rst = Path(__file__).parents[1] / "docs/prerequisites/diffusion_approximation.rst"
        text = rst.read_text()
        assert "Euler--Maruyama is not an exact" in text
        assert "boundary atoms" in text
        assert "does **not** use the Crank--Nicolson" in text
        assert "\\phi_{\\mathrm{SFS}}(x)=\\frac{\\theta}{x}" in text
