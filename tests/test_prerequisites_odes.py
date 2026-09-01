"""Independent validation for docs/prerequisites/odes.rst."""

from pathlib import Path

import numpy as np
import pytest


def execute_rst_python_blocks(path):
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


def coalescence_event_rate(k):
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError
    return k * (k - 1) / 2


def logistic_exact(t, y0, r, K):
    if y0 <= 0 or K <= 0 or r < 0:
        raise ValueError
    t = np.asarray(t, dtype=float)
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))


def euler_method(f, y0, t_span, h):
    t0, tf = t_span
    if h <= 0 or tf < t0:
        raise ValueError
    times = [float(t0)]
    values = [np.atleast_1d(np.asarray(y0, dtype=float))]
    while times[-1] < tf:
        step = min(h, tf - times[-1])
        slope = np.atleast_1d(f(values[-1], times[-1]))
        if slope.shape != values[-1].shape:
            raise ValueError
        values.append(values[-1] + step * slope)
        times.append(times[-1] + step)
    return np.array(times), np.array(values).squeeze()


def rk4_method(f, y0, t_span, h):
    t0, tf = t_span
    if h <= 0 or tf < t0:
        raise ValueError
    times = [float(t0)]
    values = [np.atleast_1d(np.asarray(y0, dtype=float))]
    while times[-1] < tf:
        step = min(h, tf - times[-1])
        t = times[-1]
        y = values[-1]
        k1 = np.atleast_1d(f(y, t))
        k2 = np.atleast_1d(f(y + step / 2 * k1, t + step / 2))
        k3 = np.atleast_1d(f(y + step / 2 * k2, t + step / 2))
        k4 = np.atleast_1d(f(y + step * k3, t + step))
        if not all(k.shape == y.shape for k in (k1, k2, k3, k4)):
            raise ValueError
        values.append(y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        times.append(t + step)
    return np.array(times), np.array(values).squeeze()


def coalescent_count_ode(probabilities, t):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) < 1:
        raise ValueError
    n = len(probabilities)
    derivative = np.zeros(n)
    for k in range(1, n + 1):
        loss = k * (k - 1) / 2 * probabilities[k - 1]
        gain = (k + 1) * k / 2 * probabilities[k] if k < n else 0.0
        derivative[k - 1] = gain - loss
    return derivative


def backward_euler_decay(rate, y0, T, h):
    if rate < 0 or h <= 0 or T < 0:
        raise ValueError
    t = 0.0
    y = float(y0)
    while t < T:
        step = min(h, T - t)
        y /= 1 + rate * step
        t += step
    return y


def migration_matrix(m_rate, n_pops=2):
    if m_rate < 0 or not isinstance(n_pops, (int, np.integer)) or n_pops < 2:
        raise ValueError
    matrix = np.full((n_pops, n_pops), m_rate, dtype=float)
    np.fill_diagonal(matrix, -(n_pops - 1) * m_rate)
    return matrix


def symmetric_matrix_exponential(A, t):
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.allclose(A, A.T):
        raise ValueError
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    return (eigenvectors * np.exp(eigenvalues * t)) @ eigenvectors.T


class TestClosedFormAndBasicSolvers:
    def test_logistic_initial_condition_and_ode_residual(self):
        y0, r, K = 0.2, 1.3, 7.0
        assert logistic_exact(0, y0, r, K) == pytest.approx(y0)
        t = np.linspace(0.1, 3, 10_000)
        y = logistic_exact(t, y0, r, K)
        numerical_derivative = np.gradient(y, t)
        rhs = r * y * (1 - y / K)
        np.testing.assert_allclose(
            numerical_derivative[2:-2], rhs[2:-2], rtol=2e-7, atol=2e-7
        )

    @pytest.mark.parametrize("args", [(0, 1, 2), (1, -1, 2), (1, 1, 0)])
    def test_logistic_rejects_invalid_parameters(self, args):
        with pytest.raises(ValueError):
            logistic_exact(1, *args)

    def test_methods_land_exactly_on_nondivisible_endpoint(self):
        for method in [euler_method, rk4_method]:
            times, _ = method(lambda y, t: -y, 1.0, (0.0, 1.0), 0.3)
            assert times[-1] == pytest.approx(1.0)
            assert np.all(np.diff(times) > 0)
            assert np.diff(times).max() <= 0.3 + 1e-15

    def test_euler_has_first_order_global_convergence(self):
        errors = []
        for h in [0.1, 0.05, 0.025]:
            _, values = euler_method(lambda y, t: -y, 1.0, (0, 1), h)
            errors.append(abs(values[-1] - np.exp(-1)))
        ratios = np.array(errors[:-1]) / errors[1:]
        np.testing.assert_allclose(ratios, 2, rtol=0.06)

    def test_rk4_has_fourth_order_global_convergence(self):
        errors = []
        for h in [0.1, 0.05, 0.025]:
            _, values = rk4_method(lambda y, t: -y, 1.0, (0, 1), h)
            errors.append(abs(values[-1] - np.exp(-1)))
        ratios = np.array(errors[:-1]) / errors[1:]
        np.testing.assert_allclose(ratios, 16, rtol=0.06)

    def test_vector_linear_system_against_closed_form(self):
        rates = np.array([1.0, 3.0])
        _, values = rk4_method(lambda y, t: -rates * y, [2.0, 4.0], (0, 2), 0.02)
        np.testing.assert_allclose(values[-1], np.array([2, 4]) * np.exp(-2 * rates), rtol=1e-6)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            euler_method(lambda y, t: np.ones(2), 1.0, (0, 1), 0.1)


class TestCoalescentProbabilityODE:
    @pytest.mark.parametrize("k,expected", [(1, 0), (2, 1), (3, 3), (10, 45)])
    def test_event_rate(self, k, expected):
        assert coalescence_event_rate(k) == expected

    def test_forward_equation_conserves_total_probability(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(8))
        assert coalescent_count_ode(p, 7).sum() == pytest.approx(0, abs=2e-15)

    def test_n3_solution_matches_independent_closed_form(self):
        p0 = np.array([0.0, 0.0, 1.0])
        times, values = rk4_method(coalescent_count_ode, p0, (0, 2), 0.002)
        t = times[-1]
        p3 = np.exp(-3 * t)
        p2 = 1.5 * (np.exp(-t) - np.exp(-3 * t))
        exact = np.array([1 - p2 - p3, p2, p3])
        np.testing.assert_allclose(values[-1], exact, atol=2e-12)
        assert values[-1].sum() == pytest.approx(1, abs=2e-14)
        assert np.all(values >= -1e-14)

    def test_mean_field_is_not_the_exact_mean_derivative(self):
        p = np.array([0.5, 0.0, 0.5])
        lineage_counts = np.arange(1, 4)
        exact_mean_derivative = np.dot(lineage_counts, coalescent_count_ode(p, 0))
        mean = np.dot(lineage_counts, p)
        mean_field = -mean * (mean - 1) / 2
        assert exact_mean_derivative == pytest.approx(-1.5)
        assert mean_field == pytest.approx(-1.0)

    def test_generator_is_lower_bidiagonal_with_zero_column_sums(self):
        n = 6
        generator = np.column_stack(
            [coalescent_count_ode(np.eye(n)[:, j], 0) for j in range(n)]
        )
        np.testing.assert_allclose(generator.sum(axis=0), 0, atol=1e-15)
        assert np.all(generator[np.tril_indices(n, -1)] == 0)


class TestStability:
    def test_forward_euler_instability_and_backward_euler_stability(self):
        rate, T, h = 1000.0, 0.1, 0.01
        _, explicit = euler_method(lambda y, t: -rate * y, 1.0, (0, T), h)
        implicit = backward_euler_decay(rate, 1.0, T, h)
        assert abs(explicit[-1]) > 1e8
        assert 0 < implicit < 1

    def test_backward_euler_formula_and_first_order_convergence(self):
        rate, T = 4.0, 1.0
        for h in [0.25, 0.125]:
            steps = round(T / h)
            assert backward_euler_decay(rate, 2.0, T, h) == pytest.approx(
                2 * (1 + rate * h) ** (-steps), abs=1e-15
            )
        errors = [
            abs(backward_euler_decay(rate, 2.0, T, h) - 2 * np.exp(-rate * T))
            for h in [0.1, 0.05, 0.025]
        ]
        assert 1.7 < errors[0] / errors[1] < 2.1
        assert 1.8 < errors[1] / errors[2] < 2.1


class TestMatrixExponential:
    def test_migration_generator_conserves_under_column_convention(self):
        for populations in [2, 3, 7]:
            matrix = migration_matrix(0.4, populations)
            np.testing.assert_allclose(matrix.sum(axis=0), 0, atol=1e-15)
            np.testing.assert_allclose(matrix, matrix.T)

    def test_two_population_solution_matches_scalar_closed_form(self):
        rate = 0.5
        matrix = migration_matrix(rate)
        initial = np.array([0.8, 0.2])
        mean = initial.mean()
        for t in [0, 0.2, 1, 4]:
            observed = symmetric_matrix_exponential(matrix, t) @ initial
            expected = mean + (initial - mean) * np.exp(-2 * rate * t)
            np.testing.assert_allclose(observed, expected, atol=4e-16)

    def test_semigroup_identity_and_conservation(self):
        matrix = migration_matrix(0.3, 4)
        left = symmetric_matrix_exponential(matrix, 0.7) @ symmetric_matrix_exponential(matrix, 0.4)
        right = symmetric_matrix_exponential(matrix, 1.1)
        np.testing.assert_allclose(left, right, atol=1e-15)
        np.testing.assert_allclose(right.sum(axis=0), 1, atol=1e-15)
        assert np.all(right >= -1e-15)

    def test_exponential_derivative_at_zero_recovers_generator(self):
        matrix = migration_matrix(0.2, 3)
        h = 1e-6
        derivative = (
            symmetric_matrix_exponential(matrix, h)
            - symmetric_matrix_exponential(matrix, -h)
        ) / (2 * h)
        np.testing.assert_allclose(derivative, matrix, atol=2e-10)

    def test_rejects_non_symmetric_matrix(self):
        with pytest.raises(ValueError):
            symmetric_matrix_exponential([[0, 1], [0, 0]], 1)


class TestPublishedCodeParity:
    def test_every_published_block_executes_and_matches_validated_functions(self):
        rst = Path(__file__).parents[1] / "docs/prerequisites/odes.rst"
        published = execute_rst_python_blocks(rst)
        for name in [
            "coalescence_event_rate",
            "logistic_exact",
            "euler_method",
            "rk4_method",
            "coalescent_count_ode",
            "backward_euler_decay",
            "migration_matrix",
            "symmetric_matrix_exponential",
        ]:
            assert name in published
        p = np.array([0.1, 0.2, 0.7])
        np.testing.assert_array_equal(
            published["coalescent_count_ode"](p, 0), coalescent_count_ode(p, 0)
        )
        matrix = migration_matrix(0.5, 3)
        np.testing.assert_allclose(
            published["symmetric_matrix_exponential"](matrix, 0.8),
            symmetric_matrix_exponential(matrix, 0.8),
        )

    def test_guardrails_remain_in_text(self):
        text = (Path(__file__).parents[1] / "docs/prerequisites/odes.rst").read_text()
        assert "not** the exact ODE" in text
        assert "A-stable" in text
        assert "defective matrices are not diagonalizable" in text
        assert "columns** of :math:`A` to sum to zero" in text
