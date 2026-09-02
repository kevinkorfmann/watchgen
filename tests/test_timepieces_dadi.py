"""Executable checks for the dadi Timepiece documentation.

The chapter's snippets call the maintained mini implementation instead of
duplicating a second, historically divergent solver inside the test suite.
The numerical references below follow dadi's public source conventions.
"""

import numpy as np
import pytest
from scipy.special import gammaln

from watchgen.mini_dadi import (
    equilibrium_sfs_density,
    implicit_1d,
    make_nonuniform_grid,
    multinomial_log_likelihood,
    optimal_sfs_scaling,
    phi_1d_to_2d,
    poisson_log_likelihood,
    sfs_from_phi,
    two_epoch_sfs,
)


def test_default_grid_matches_dadi_exponential_grid():
    uniform = np.linspace(-1, 1, 51)
    expected = 1 / (1 + np.exp(-8 * uniform))
    expected = (expected - expected[0]) / (expected[-1] - expected[0])
    assert np.array_equal(make_nonuniform_grid(51), expected)


def test_neutral_phi_matches_dadi_snm_convention():
    xx = make_nonuniform_grid(31)
    phi = equilibrium_sfs_density(xx, nu=2.0, theta=3.0)
    assert phi[0] == phi[1]
    assert np.allclose(phi[1:] * xx[1:], 6.0)


def test_split_is_a_trapezoid_normalized_delta():
    xx = make_nonuniform_grid(31)
    phi = equilibrium_sfs_density(xx)
    split = phi_1d_to_2d(phi, xx)
    assert np.count_nonzero(split - np.diag(np.diag(split))) == 0
    expected_diag = phi[1:-1] * 2 / (xx[2:] - xx[:-2])
    assert np.allclose(np.diag(split)[1:-1], expected_diag)
    interior_mass = np.sum(phi[1:-1] * (xx[2:] - xx[:-2]) / 2)
    split_mass = np.trapezoid(np.trapezoid(split, xx, axis=1), xx)
    assert split_mass == pytest.approx(interior_mass)


def test_neutral_equilibrium_is_stationary_under_one_pop_integration():
    xx = make_nonuniform_grid(41)
    phi = equilibrium_sfs_density(xx)
    evolved = implicit_1d(phi, xx, T=0.2, nu=1, theta=1, n_steps=100)
    assert np.allclose(evolved[1:], phi[1:], rtol=1e-10, atol=1e-10)


def test_one_step_matches_official_dadi_neutral_coefficients():
    """Independent dense solve of dadi's _one_pop_const_params equations."""
    xx = make_nonuniform_grid(27)
    phi = np.linspace(0.2, 2.0, len(xx))
    nu, theta, dt = 0.7, 1.3, 0.017
    dx = np.diff(xx)
    dfactor = np.empty(len(xx))
    dfactor[1:-1] = 2 / (dx[:-1] + dx[1:])
    dfactor[0], dfactor[-1] = 2 / dx[0], 2 / dx[-1]
    variance = xx * (1 - xx) / nu

    lower = np.zeros_like(phi)
    upper = np.zeros_like(phi)
    diagonal = np.zeros_like(phi)
    lower[1:] = -dfactor[1:] * variance[:-1] / (2 * dx)
    upper[:-1] = -dfactor[:-1] * variance[1:] / (2 * dx)
    diagonal[:-1] = dfactor[:-1] * variance[:-1] / (2 * dx)
    diagonal[1:] += dfactor[1:] * variance[1:] / (2 * dx)
    diagonal[0] += (0.5 / nu) * 2 / dx[0]
    diagonal[-1] += (0.5 / nu) * 2 / dx[-1]

    system = np.diag(diagonal + 1 / dt)
    system += np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    rhs = phi.copy()
    rhs[1] += dt / xx[1] * theta / 2 * 2 / (xx[2] - xx[0])
    expected = np.linalg.solve(system, rhs / dt)
    actual = implicit_1d(phi, xx, T=dt, nu=nu, theta=theta, n_steps=1)
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_projection_recovers_neutral_internal_spectrum():
    xx = make_nonuniform_grid(201)
    fs = sfs_from_phi(equilibrium_sfs_density(xx), xx, 20)
    expected = 1 / np.arange(1, 20)
    assert np.allclose(fs[1:20], expected, rtol=4e-3)


def test_poisson_likelihood_includes_factorial_constant():
    model = np.array([0.0, 2.5, 7.0])
    data = np.array([0, 3, 5])
    expected = np.sum(
        -model
        + np.where(data > 0, data * np.log(np.maximum(model, 1)), 0)
        - gammaln(data + 1)
    )
    assert poisson_log_likelihood(model, data) == pytest.approx(expected)


def test_multinomial_is_optimally_scaled_poisson_likelihood():
    model = np.array([1.0, 2.0, 4.0])
    data = np.array([5.0, 8.0, 20.0])
    scale = optimal_sfs_scaling(model, data)
    assert scale == pytest.approx(data.sum() / model.sum())
    assert multinomial_log_likelihood(model, data) == pytest.approx(
        poisson_log_likelihood(scale * model, data)
    )
    assert multinomial_log_likelihood(100 * model, data) == pytest.approx(
        multinomial_log_likelihood(model, data)
    )


def test_two_epoch_size_changes_produce_distinct_spectra():
    expansion = two_epoch_sfs(5.0, 0.2, 15, pts=61)
    contraction = two_epoch_sfs(0.2, 0.2, 15, pts=61)
    assert np.all(expansion[1:-1] >= 0)
    assert np.all(contraction[1:-1] >= 0)
    assert not np.allclose(expansion[1:-1], contraction[1:-1])
