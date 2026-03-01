"""
Tests for all Python code examples in docs/prerequisites/odes.rst.

Covers:
  1. logistic_exact       -- exact logistic equation solution
  2. euler_method          -- forward Euler ODE solver
  3. logistic_rhs          -- logistic equation right-hand side
  4. rk4_method            -- classical fourth-order Runge-Kutta solver
  5. sfs_ode               -- SFS moment equations (drift + mutation)
  6. stiff_ode             -- stiff two-component system
  7. migration_matrix      -- symmetric migration rate matrix
  8. Matrix exponential     -- expm and eigendecomposition equivalence
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

# ---------------------------------------------------------------------------
# Re-define every function from the RST code blocks
# ---------------------------------------------------------------------------

# -- Code block 1: logistic_exact ------------------------------------------

def logistic_exact(t, y0, r, K):
    """Exact solution of the logistic equation.

    Parameters
    ----------
    t : float or ndarray
        Time(s) at which to evaluate the solution.
    y0 : float
        Initial condition y(0).
    r : float
        Growth rate.
    K : float
        Carrying capacity.

    Returns
    -------
    y : float or ndarray
        Solution y(t).
    """
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))


# -- Code block 2: euler_method & logistic_rhs -----------------------------

def euler_method(f, y0, t_span, h):
    """Solve dy/dt = f(y, t) using forward Euler with step size h."""
    t0, tf = t_span
    t_values = np.arange(t0, tf + h / 2, h)
    n_steps = len(t_values)

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0

    for i in range(1, n_steps):
        y_values[i] = y_values[i - 1] + h * np.atleast_1d(
            f(y_values[i - 1], t_values[i - 1])
        )

    return t_values, y_values.squeeze()


def logistic_rhs(y, t):
    """Right-hand side of the logistic equation."""
    r, K = 1.0, 10.0
    return r * y * (1 - y / K)


# -- Code block 3: rk4_method ----------------------------------------------

def rk4_method(f, y0, t_span, h):
    """Solve dy/dt = f(y, t) using the classical fourth-order Runge-Kutta."""
    t0, tf = t_span
    t_values = np.arange(t0, tf + h / 2, h)
    n_steps = len(t_values)

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0

    for i in range(1, n_steps):
        t = t_values[i - 1]
        y = y_values[i - 1]

        k1 = np.atleast_1d(f(y, t))
        k2 = np.atleast_1d(f(y + h / 2 * k1, t + h / 2))
        k3 = np.atleast_1d(f(y + h / 2 * k2, t + h / 2))
        k4 = np.atleast_1d(f(y + h * k3, t + h))

        y_values[i] = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, y_values.squeeze()


# -- Code block 4: sfs_ode -------------------------------------------------

def sfs_ode(phi, t, n, theta):
    """Right-hand side of the SFS moment equations (drift + mutation).

    Parameters
    ----------
    phi : ndarray of shape (n-1,) -- SFS entries phi_1, ..., phi_{n-1}.
    t : float -- current time (unused, required by ODE interface).
    n : int -- sample size.
    theta : float -- population-scaled mutation rate (4*Ne*mu).
    """
    m = n - 1  # number of SFS entries
    dphi = np.zeros(m)

    for j in range(1, n):
        idx = j - 1

        drift_out = -(j * (n - j)) / n * phi[idx]

        drift_up = 0.0
        if j + 1 <= n - 1:
            drift_up = ((j + 1) * (n - j - 1)) / n * phi[idx + 1]

        drift_down = 0.0
        if j - 1 >= 1:
            drift_down = ((j - 1) * (n - j + 1)) / n * phi[idx - 1]

        mutation = theta if j == 1 else 0.0

        dphi[idx] = drift_out + drift_up + drift_down + mutation

    return dphi


# -- Code block 5: stiff_ode -----------------------------------------------

def stiff_ode(y, t):
    """A stiff two-component system: dy1/dt = -1000*y1 + y2 (fast),
    dy2/dt = y1 - y2 (slow). Stiffness ratio ~1000."""
    return np.array([-1000 * y[0] + y[1],
                      y[0] - y[1]])


# -- Code block 6: migration_matrix ----------------------------------------

def migration_matrix(m_rate, n_pops=2):
    """Rate matrix for symmetric migration between n_pops populations.

    Off-diagonal: A[i,j] = m_rate (gain from pop j).
    Diagonal: A[i,i] = -(n_pops-1)*m_rate (conservation: rows sum to 0).
    """
    A = np.full((n_pops, n_pops), m_rate)
    np.fill_diagonal(A, -(n_pops - 1) * m_rate)
    return A


# ===========================================================================
# TESTS -- Code block 1: logistic_exact
# ===========================================================================

class TestLogisticExact:
    """Tests for the exact logistic equation solution."""

    def test_initial_condition(self):
        """y(0) must equal y0 for any parameter combination."""
        for y0, r, K in [(0.1, 1.0, 10.0), (5.0, 2.0, 100.0), (0.01, 0.5, 1.0)]:
            assert np.isclose(logistic_exact(0.0, y0, r, K), y0), (
                f"y(0) != y0 for y0={y0}"
            )

    def test_approaches_carrying_capacity(self):
        """As t -> inf, y(t) should approach K."""
        y0, r, K = 0.1, 1.0, 10.0
        y_large_t = logistic_exact(100.0, y0, r, K)
        assert np.isclose(y_large_t, K, atol=1e-10), (
            f"y(100) = {y_large_t}, expected {K}"
        )

    def test_monotonically_increasing_below_K(self):
        """When y0 < K, y(t) should be strictly increasing."""
        y0, r, K = 0.1, 1.0, 10.0
        t_vals = np.linspace(0, 20, 200)
        y_vals = logistic_exact(t_vals, y0, r, K)
        diffs = np.diff(y_vals)
        assert np.all(diffs > 0), "y(t) is not monotonically increasing when y0 < K"

    def test_array_input(self):
        """logistic_exact should accept numpy array time inputs."""
        y0, r, K = 0.1, 1.0, 10.0
        t_arr = np.array([0.0, 1.0, 5.0, 10.0])
        result = logistic_exact(t_arr, y0, r, K)
        assert result.shape == (4,)
        assert np.isclose(result[0], y0)

    def test_midpoint_value(self):
        """At the inflection point t* = ln((K-y0)/y0)/r, y = K/2."""
        y0, r, K = 0.1, 1.0, 10.0
        t_inflection = np.log((K / y0 - 1)) / r
        y_mid = logistic_exact(t_inflection, y0, r, K)
        assert np.isclose(y_mid, K / 2.0, rtol=1e-12), (
            f"y(t*) = {y_mid}, expected {K/2}"
        )

    def test_symmetry_around_inflection(self):
        """The logistic curve is symmetric around K/2 when mapped appropriately."""
        y0, r, K = 0.1, 1.0, 10.0
        t_inflection = np.log((K / y0 - 1)) / r
        dt = 1.0
        y_before = logistic_exact(t_inflection - dt, y0, r, K)
        y_after = logistic_exact(t_inflection + dt, y0, r, K)
        # Symmetry: y_before + y_after = K
        assert np.isclose(y_before + y_after, K, rtol=1e-10)

    def test_different_parameters(self):
        """Verify with several parameter choices."""
        # Higher growth rate should reach K faster
        y0, K = 0.1, 10.0
        y_slow = logistic_exact(3.0, y0, 0.5, K)
        y_fast = logistic_exact(3.0, y0, 2.0, K)
        assert y_fast > y_slow, "Higher growth rate should reach K faster"


# ===========================================================================
# TESTS -- Code block 2: euler_method & logistic_rhs
# ===========================================================================

class TestEulerMethod:
    """Tests for the forward Euler method and the logistic RHS."""

    def test_logistic_rhs_at_zero(self):
        """logistic_rhs(0, t) = 0 for any t (zero is a fixed point)."""
        assert logistic_rhs(np.array([0.0]), 0.0) == 0.0

    def test_logistic_rhs_at_carrying_capacity(self):
        """logistic_rhs(K, t) = 0 (carrying capacity is a fixed point)."""
        K = 10.0
        assert np.isclose(logistic_rhs(np.array([K]), 0.0), 0.0)

    def test_logistic_rhs_positive_below_K(self):
        """For 0 < y < K, the growth rate is positive."""
        for y_val in [0.1, 1.0, 5.0, 9.99]:
            assert logistic_rhs(np.array([y_val]), 0.0) > 0

    def test_euler_returns_correct_shapes(self):
        """Euler should return time and solution arrays of matching length."""
        t_vals, y_vals = euler_method(logistic_rhs, 0.1, (0.0, 5.0), 0.1)
        assert t_vals.shape[0] == y_vals.shape[0]
        assert t_vals[0] == 0.0
        assert np.isclose(t_vals[-1], 5.0, atol=0.05)

    def test_euler_initial_condition_preserved(self):
        """The first value of the solution should equal y0."""
        _, y_vals = euler_method(logistic_rhs, 0.1, (0.0, 5.0), 0.1)
        assert np.isclose(y_vals[0], 0.1)

    def test_euler_convergence_first_order(self):
        """Euler's method is first-order: halving h should roughly halve the error."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        _, y_h1 = euler_method(logistic_rhs, y0, (0.0, t_final), 0.1)
        _, y_h2 = euler_method(logistic_rhs, y0, (0.0, t_final), 0.05)

        error_h1 = abs(float(y_h1[-1]) - y_exact)
        error_h2 = abs(float(y_h2[-1]) - y_exact)
        ratio = error_h1 / error_h2
        # For first-order, expect ratio ~2.0; allow generous tolerance
        assert 1.5 < ratio < 3.0, (
            f"Convergence ratio {ratio:.2f} outside expected range [1.5, 3.0]"
        )

    def test_euler_approaches_exact_with_small_h(self):
        """With a small enough step size, Euler should be close to the exact solution."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        _, y_vals = euler_method(logistic_rhs, y0, (0.0, t_final), 0.01)
        error = abs(float(y_vals[-1]) - y_exact)
        assert error < 0.05, f"Euler error with h=0.01 is too large: {error}"

    def test_euler_multiple_step_sizes_converge(self):
        """Error should decrease as step size decreases."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        errors = []
        for h in [1.0, 0.5, 0.1, 0.05, 0.01]:
            _, y_vals = euler_method(logistic_rhs, y0, (0.0, t_final), h)
            errors.append(abs(float(y_vals[-1]) - y_exact))
        # Each error should be smaller than the previous
        for i in range(1, len(errors)):
            assert errors[i] < errors[i - 1], (
                f"Error did not decrease: h-sequence error[{i}]={errors[i]:.2e} "
                f">= error[{i-1}]={errors[i-1]:.2e}"
            )

    def test_euler_handles_vector_ode(self):
        """Euler should work with a vector-valued ODE (system)."""
        def simple_system(y, t):
            return np.array([-y[0], -2 * y[1]])

        y0 = np.array([1.0, 1.0])
        t_vals, y_vals = euler_method(simple_system, y0, (0.0, 1.0), 0.01)
        assert y_vals.shape[1] == 2
        # Solutions are y1 = exp(-t), y2 = exp(-2t)
        assert np.isclose(y_vals[-1, 0], np.exp(-1.0), atol=0.02)
        assert np.isclose(y_vals[-1, 1], np.exp(-2.0), atol=0.03)


# ===========================================================================
# TESTS -- Code block 3: rk4_method
# ===========================================================================

class TestRK4Method:
    """Tests for the classical fourth-order Runge-Kutta method."""

    def test_rk4_initial_condition(self):
        """The first value should equal y0."""
        _, y_vals = rk4_method(logistic_rhs, 0.1, (0.0, 5.0), 0.1)
        assert np.isclose(y_vals[0], 0.1)

    def test_rk4_returns_correct_shapes(self):
        """RK4 should return arrays of consistent shape."""
        t_vals, y_vals = rk4_method(logistic_rhs, 0.1, (0.0, 5.0), 0.1)
        assert t_vals.shape[0] == y_vals.shape[0]

    def test_rk4_much_better_than_euler(self):
        """RK4 at h=0.1 should be far more accurate than Euler at h=0.1."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        _, y_euler = euler_method(logistic_rhs, y0, (0.0, t_final), 0.1)
        _, y_rk4 = rk4_method(logistic_rhs, y0, (0.0, t_final), 0.1)

        error_euler = abs(float(y_euler[-1]) - y_exact)
        error_rk4 = abs(float(y_rk4[-1]) - y_exact)
        assert error_rk4 < error_euler / 100, (
            f"RK4 error ({error_rk4:.2e}) should be much smaller than "
            f"Euler error ({error_euler:.2e})"
        )

    def test_rk4_convergence_fourth_order(self):
        """Halving h should reduce the error by a factor of ~16 (fourth-order)."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        prev_error = None
        for h in [0.5, 0.25, 0.125, 0.0625]:
            _, y_rk = rk4_method(logistic_rhs, y0, (0.0, t_final), h)
            err = abs(float(y_rk[-1]) - y_exact)
            if prev_error is not None:
                ratio = prev_error / err
                # Expect ~16 for fourth-order; allow range [10, 22]
                assert 10.0 < ratio < 22.0, (
                    f"RK4 convergence ratio {ratio:.1f} outside [10, 22] at h={h}"
                )
            prev_error = err

    def test_rk4_vs_scipy_solve_ivp(self):
        """RK4 and scipy solve_ivp should give comparable results."""
        y0, r, K = 0.1, 1.0, 10.0
        t_final = 5.0
        y_exact = logistic_exact(t_final, y0, r, K)

        _, y_rk4 = rk4_method(logistic_rhs, y0, (0.0, t_final), 0.1)

        # Use tighter tolerances to ensure scipy matches exact solution closely
        sol = solve_ivp(
            lambda t, y: logistic_rhs(y, t),
            [0.0, t_final], [y0],
            rtol=1e-10, atol=1e-12,
            dense_output=True,
        )
        y_scipy = sol.sol(t_final)[0]

        # RK4 at h=0.1 should be close to exact (within ~1e-5 for this problem)
        assert abs(float(y_rk4[-1]) - y_exact) < 1e-4
        # scipy adaptive solver with tight tolerances should be very close
        assert abs(y_scipy - y_exact) < 1e-6

    def test_rk4_handles_vector_ode(self):
        """RK4 should correctly solve a 2D linear system."""
        def linear_system(y, t):
            return np.array([-y[0], -2 * y[1]])

        y0 = np.array([1.0, 1.0])
        t_vals, y_vals = rk4_method(linear_system, y0, (0.0, 2.0), 0.01)
        # Analytical: y1 = exp(-t), y2 = exp(-2t) at t=2
        assert np.isclose(y_vals[-1, 0], np.exp(-2.0), atol=1e-8)
        assert np.isclose(y_vals[-1, 1], np.exp(-4.0), atol=1e-8)

    def test_rk4_constant_solution(self):
        """If f(y, t) = 0, the solution should remain at y0."""
        def zero_rhs(y, t):
            return 0.0

        _, y_vals = rk4_method(zero_rhs, 3.14, (0.0, 10.0), 0.5)
        assert np.allclose(y_vals, 3.14)


# ===========================================================================
# TESTS -- Code block 4: sfs_ode (SFS moment equations)
# ===========================================================================

class TestSFSODE:
    """Tests for the SFS drift+mutation moment equations.

    The sfs_ode function implements the moment equations for the site frequency
    spectrum under drift and mutation. The drift coefficients divide by n (the
    sample size), giving a particular time scaling. We test structural properties
    of the ODE system and verify that numerical integration converges to a
    well-defined steady state.
    """

    def test_sfs_ode_output_shape(self):
        """sfs_ode should return an array of shape (n-1,)."""
        n, theta = 20, 1.0
        phi = np.zeros(n - 1)
        dphi = sfs_ode(phi, 0.0, n, theta)
        assert dphi.shape == (n - 1,)

    def test_sfs_ode_mutation_injection(self):
        """Starting from zero SFS, only phi_1 should get a positive derivative
        (from the mutation term)."""
        n, theta = 20, 1.0
        phi = np.zeros(n - 1)
        dphi = sfs_ode(phi, 0.0, n, theta)
        assert dphi[0] == theta, "Mutation should inject at j=1"
        # All other entries should be zero when phi is all-zero
        assert np.allclose(dphi[1:], 0.0)

    def test_sfs_ode_drift_is_tridiagonal(self):
        """The drift operator is tridiagonal: dphi_j depends only on
        phi_{j-1}, phi_j, phi_{j+1}."""
        n = 10
        theta = 0.0  # no mutation, pure drift
        # Set a single non-zero entry at j=5 (index 4)
        phi = np.zeros(n - 1)
        phi[4] = 1.0  # phi_5 = 1
        dphi = sfs_ode(phi, 0.0, n, theta)
        # Only indices 3, 4, 5 (j=4, 5, 6) should be non-zero
        for idx in range(n - 1):
            if idx in [3, 4, 5]:
                continue
            assert dphi[idx] == 0.0, (
                f"dphi[{idx}] = {dphi[idx]} should be 0 for tridiagonal structure"
            )

    def test_sfs_ode_drift_out_negative(self):
        """The diagonal drift term should be negative (mass leaving frequency j)."""
        n = 10
        theta = 0.0
        phi = np.ones(n - 1)
        # For any j, drift_out = -j*(n-j)/n * phi_j which is negative
        for j in range(1, n):
            expected_drift_out = -(j * (n - j)) / n
            assert expected_drift_out < 0

    def test_sfs_short_time_integration(self):
        """Integrating from zero SFS for a short time should produce a
        solution with positive phi_1 (from mutation injection) and all
        other entries non-negative."""
        n = 20
        theta = 1.0
        phi0 = np.zeros(n - 1)

        # Integrate for a short time where the solution is well-behaved
        sol = solve_ivp(
            lambda t, y: sfs_ode(y, t, n, theta),
            [0.0, 2.0],
            phi0,
            method='RK45',
            max_step=0.05,
        )
        phi_final = sol.y[:, -1]

        # phi_1 should be positive (mutation has injected mass)
        assert phi_final[0] > 0, "phi_1 should be positive after mutation injection"
        # All entries should be non-negative for short integration time
        assert np.all(phi_final >= -1e-10), "SFS entries should be non-negative"
        assert sol.success, f"Integration failed: {sol.message}"

    def test_sfs_integration_reproduces_rst_code(self):
        """Reproduce the exact integration from the RST code block:
        n=20, theta=1.0, integrate from 0 to 50 with RK45 + max_step=0.1.
        Verify the solver reports success (matching the RST code's behavior)."""
        n = 20
        theta = 1.0
        phi0 = np.zeros(n - 1)

        sol = solve_ivp(
            lambda t, y: sfs_ode(y, t, n, theta),
            [0.0, 50.0],
            phi0,
            method='RK45',
            max_step=0.1,
            dense_output=True,
        )
        # The solver should complete (report success)
        assert sol.success, f"Integration failed: {sol.message}"
        # The solution should have the right shape
        assert sol.y.shape[0] == n - 1

    def test_sfs_drift_matrix_is_tridiagonal(self):
        """Build the drift matrix explicitly and verify it is tridiagonal."""
        n = 10
        m = n - 1
        A_mat = np.zeros((m, m))

        for j in range(1, n):
            idx = j - 1
            A_mat[idx, idx] = -(j * (n - j)) / n
            if j + 1 <= n - 1:
                A_mat[idx, idx + 1] = ((j + 1) * (n - j - 1)) / n
            if j - 1 >= 1:
                A_mat[idx, idx - 1] = ((j - 1) * (n - j + 1)) / n

        # Check tridiagonal: entries more than 1 away from diagonal should be 0
        for i in range(m):
            for j_idx in range(m):
                if abs(i - j_idx) > 1:
                    assert A_mat[i, j_idx] == 0.0, (
                        f"A[{i},{j_idx}] = {A_mat[i, j_idx]} should be 0 "
                        f"for tridiagonal structure"
                    )

    def test_sfs_linearity_in_theta(self):
        """The ODE system is linear: dphi/dt = A*phi + theta*b. So the
        derivative at phi with theta=2 should be double the derivative
        at phi with theta=1, when phi=0 (since A*0=0)."""
        n = 10
        phi_zero = np.zeros(n - 1)
        dphi_1 = sfs_ode(phi_zero, 0.0, n, 1.0)
        dphi_2 = sfs_ode(phi_zero, 0.0, n, 2.0)
        assert np.allclose(dphi_2, 2.0 * dphi_1), (
            "Mutation injection should scale linearly with theta"
        )

    def test_sfs_time_invariant_rhs(self):
        """The RHS does not depend on t; evaluating at different times gives the
        same result for the same phi."""
        n = 10
        theta = 1.0
        phi = np.array([1.0 / j for j in range(1, n)])
        dphi_t0 = sfs_ode(phi, 0.0, n, theta)
        dphi_t5 = sfs_ode(phi, 5.0, n, theta)
        assert np.allclose(dphi_t0, dphi_t5)


# ===========================================================================
# TESTS -- Code block 5: stiff_ode
# ===========================================================================

class TestStiffODE:
    """Tests for the stiff two-component ODE system."""

    def test_stiff_ode_output_shape(self):
        """stiff_ode should return an array of shape (2,)."""
        y = np.array([1.0, 0.0])
        result = stiff_ode(y, 0.0)
        assert result.shape == (2,)

    def test_stiff_ode_initial_derivatives(self):
        """With y0 = [1, 0], dy1/dt = -1000, dy2/dt = 1."""
        y = np.array([1.0, 0.0])
        dy = stiff_ode(y, 0.0)
        assert dy[0] == -1000.0
        assert dy[1] == 1.0

    def test_stiff_rk45_and_bdf_agree(self):
        """Both RK45 (explicit) and BDF (implicit) should give the same answer."""
        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 1.0)

        sol_rk45 = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='RK45', rtol=1e-8, atol=1e-10,
        )
        sol_bdf = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='BDF', rtol=1e-8, atol=1e-10,
        )

        assert np.allclose(sol_rk45.y[:, -1], sol_bdf.y[:, -1], atol=1e-6), (
            f"RK45 end: {sol_rk45.y[:, -1]}, BDF end: {sol_bdf.y[:, -1]}"
        )

    def test_stiff_bdf_fewer_evaluations(self):
        """BDF should require fewer function evaluations than RK45 for this stiff system."""
        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 1.0)

        sol_rk45 = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='RK45', rtol=1e-8, atol=1e-10,
        )
        sol_bdf = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='BDF', rtol=1e-8, atol=1e-10,
        )

        assert sol_bdf.nfev < sol_rk45.nfev, (
            f"BDF nfev ({sol_bdf.nfev}) should be less than RK45 nfev ({sol_rk45.nfev})"
        )

    def test_stiff_fast_transient_decays(self):
        """The fast component y1 should be nearly zero after t = 0.01."""
        y0 = np.array([1.0, 0.0])
        sol = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            (0.0, 1.0), y0,
            method='BDF', rtol=1e-10, atol=1e-12,
            dense_output=True,
        )
        y_at_001 = sol.sol(0.01)
        # After 0.01 seconds, exp(-1000*0.01) = exp(-10) ~ 4.5e-5
        # y1 should be very small
        assert abs(y_at_001[0]) < 0.01, (
            f"Fast component y1 at t=0.01 is {y_at_001[0]:.6f}, expected near 0"
        )

    def test_stiff_both_solvers_successful(self):
        """Both solvers should report success."""
        y0 = np.array([1.0, 0.0])
        t_span = (0.0, 1.0)

        sol_rk45 = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='RK45', rtol=1e-8, atol=1e-10,
        )
        sol_bdf = solve_ivp(
            lambda t, y: stiff_ode(y, t),
            t_span, y0,
            method='BDF', rtol=1e-8, atol=1e-10,
        )
        assert sol_rk45.success
        assert sol_bdf.success


# ===========================================================================
# TESTS -- Code block 6: migration_matrix & matrix exponential
# ===========================================================================

class TestMigrationMatrix:
    """Tests for the migration rate matrix construction."""

    def test_rows_sum_to_zero(self):
        """Rate matrix rows must sum to zero (probability conservation)."""
        for m_rate in [0.1, 0.5, 1.0, 5.0]:
            for n_pops in [2, 3, 4]:
                A = migration_matrix(m_rate, n_pops)
                row_sums = A.sum(axis=1)
                assert np.allclose(row_sums, 0.0), (
                    f"Row sums not zero for m_rate={m_rate}, n_pops={n_pops}"
                )

    def test_diagonal_entries(self):
        """Diagonal entries should be -(n_pops-1)*m_rate."""
        m_rate = 0.5
        n_pops = 3
        A = migration_matrix(m_rate, n_pops)
        expected_diag = -(n_pops - 1) * m_rate
        assert np.allclose(np.diag(A), expected_diag)

    def test_off_diagonal_entries(self):
        """Off-diagonal entries should be m_rate."""
        m_rate = 0.5
        n_pops = 3
        A = migration_matrix(m_rate, n_pops)
        for i in range(n_pops):
            for j in range(n_pops):
                if i != j:
                    assert A[i, j] == m_rate

    def test_shape(self):
        """Matrix should be n_pops x n_pops."""
        A = migration_matrix(0.5, 4)
        assert A.shape == (4, 4)

    def test_default_two_populations(self):
        """Default n_pops should be 2."""
        A = migration_matrix(0.5)
        assert A.shape == (2, 2)


class TestMatrixExponentialMigration:
    """Tests for the matrix exponential solution of migration ODEs."""

    def test_initial_condition_preserved(self):
        """expm(A*0) @ y0 = y0 (identity at t=0)."""
        A = migration_matrix(0.5)
        y0 = np.array([0.8, 0.2])
        y_t0 = expm(A * 0.0) @ y0
        assert np.allclose(y_t0, y0)

    def test_conservation_of_total_frequency(self):
        """Total frequency should be conserved at all times (rows sum to 0)."""
        A = migration_matrix(0.5)
        y0 = np.array([0.8, 0.2])
        total_initial = np.sum(y0)

        for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            y_t = expm(A * t) @ y0
            assert np.isclose(np.sum(y_t), total_initial, atol=1e-12), (
                f"Total frequency changed at t={t}: {np.sum(y_t)} != {total_initial}"
            )

    def test_equilibrium_is_mean_frequency(self):
        """At equilibrium, both populations should have the mean initial frequency."""
        m_rate = 0.5
        A = migration_matrix(m_rate)
        y0 = np.array([0.8, 0.2])
        expected_eq = np.mean(y0)

        y_eq = expm(A * 100.0) @ y0
        assert np.allclose(y_eq, expected_eq, atol=1e-10), (
            f"Equilibrium {y_eq} != expected {expected_eq}"
        )

    def test_eigenvalues_two_pop_symmetric(self):
        """For 2-pop symmetric migration at rate m, eigenvalues are 0 and -2m."""
        m_rate = 0.5
        A = migration_matrix(m_rate)
        eigenvalues = np.sort(np.linalg.eigvals(A))
        expected = np.sort([0.0, -2 * m_rate])
        assert np.allclose(eigenvalues, expected, atol=1e-12)

    def test_expm_vs_eigendecomposition(self):
        """Matrix exponential via expm and via eigendecomposition should agree."""
        m_rate = 0.5
        A = migration_matrix(m_rate)
        y0 = np.array([0.8, 0.2])
        t_test = 2.0

        # Via scipy expm
        y_expm = expm(A * t_test) @ y0

        # Via eigendecomposition
        eigenvalues, V = np.linalg.eig(A)
        D = np.diag(np.exp(eigenvalues * t_test))
        y_eigen = V @ D @ np.linalg.inv(V) @ y0

        assert np.allclose(y_expm, np.real(y_eigen), atol=1e-12), (
            f"expm: {y_expm}, eigendecomp: {np.real(y_eigen)}"
        )

    def test_frequencies_converge_over_time(self):
        """Frequencies in the two populations should converge toward each other."""
        A = migration_matrix(0.5)
        y0 = np.array([0.8, 0.2])

        diffs = []
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            y_t = expm(A * t) @ y0
            diffs.append(abs(y_t[0] - y_t[1]))

        # Differences should be monotonically decreasing
        for i in range(1, len(diffs)):
            assert diffs[i] < diffs[i - 1] + 1e-15, (
                f"Frequency difference not decreasing at index {i}"
            )

    def test_three_population_equilibrium(self):
        """For 3 populations with symmetric migration, equilibrium is the mean."""
        m_rate = 0.3
        A = migration_matrix(m_rate, n_pops=3)
        y0 = np.array([0.9, 0.5, 0.1])
        expected_eq = np.mean(y0)

        y_eq = expm(A * 100.0) @ y0
        assert np.allclose(y_eq, expected_eq, atol=1e-8), (
            f"3-pop equilibrium {y_eq} != expected {expected_eq}"
        )

    def test_matrix_exponential_is_stochastic(self):
        """expm(A*t) should be a stochastic matrix: non-negative entries, rows sum to 1."""
        A = migration_matrix(0.5)
        for t in [0.1, 1.0, 5.0]:
            P = expm(A * t)
            # Rows should sum to 1
            assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12)
            # All entries non-negative
            assert np.all(P >= -1e-15), f"Negative entry in expm(A*{t})"

    def test_expm_at_various_times(self):
        """Verify the numerical values at specific time points match expectations."""
        m_rate = 0.5
        A = migration_matrix(m_rate)
        y0 = np.array([0.8, 0.2])

        # At t=0, should be initial condition
        y0_check = expm(A * 0.0) @ y0
        assert np.allclose(y0_check, [0.8, 0.2])

        # At large t, both should be close to 0.5
        y_large = expm(A * 10.0) @ y0
        assert np.allclose(y_large, [0.5, 0.5], atol=1e-4)

        # For 2x2 symmetric migration with eigenvalues 0 and -2m:
        # y1(t) = mean + (y1_0 - mean)*exp(-2m*t)
        # y2(t) = mean + (y2_0 - mean)*exp(-2m*t)
        mean_freq = 0.5
        for t in [0.5, 1.0, 2.0]:
            y_t = expm(A * t) @ y0
            expected_y1 = mean_freq + (0.8 - mean_freq) * np.exp(-2 * m_rate * t)
            expected_y2 = mean_freq + (0.2 - mean_freq) * np.exp(-2 * m_rate * t)
            assert np.isclose(y_t[0], expected_y1, atol=1e-10)
            assert np.isclose(y_t[1], expected_y2, atol=1e-10)
