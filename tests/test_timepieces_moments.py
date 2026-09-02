"""Documentation-level invariants for the moments teaching implementation.

The documentation displays functions from :mod:`watchgen.mini_moments`; tests
must exercise that maintained implementation rather than a second pasted copy.
The neutral operator fixtures below follow moments-popgen 1.6.1
``LinearSystem.calcD``/``Integration.integrate_nD`` scaling.
"""

import demes
import numpy as np

from watchgen.mini_moments import (
    _neutral_sfs,
    drift_operator,
    integrate_sfs,
    mutation_operator,
    split_1d_to_2d,
)


def test_published_demes_builder_example_resolves_to_graph():
    """The chapter must pass a resolved graph, not a nonexistent builder export."""
    builder = demes.Builder(description="Two-epoch expansion model")
    builder.add_deme(
        "ancestral", epochs=[{"start_size": 10000, "end_time": 5000}]
    )
    builder.add_deme(
        "modern",
        ancestors=["ancestral"],
        epochs=[{"start_size": 50000, "end_time": 0}],
    )

    graph = builder.resolve()
    assert isinstance(graph, demes.Graph)
    assert [deme.name for deme in graph.demes] == ["ancestral", "modern"]


def test_neutral_operator_matches_moments_1_6_1_matrix_fixture():
    """The teaching operator is D/2 and mutation enters at n*theta/2."""
    n = 6
    theta = 1.7
    phi = np.array([0.0, 2.0, 3.0, 5.0, 7.0, 11.0, 0.0])

    expected_drift = np.zeros(n + 1)
    for j in range(1, n):
        expected_drift[j] = 0.5 * (
            (j - 1) * (n - j + 1) * phi[j - 1]
            - 2 * j * (n - j) * phi[j]
            + (j + 1) * (n - j - 1) * phi[j + 1]
        )

    expected_mutation = np.zeros(n + 1)
    expected_mutation[1] = n * theta / 2
    assert np.array_equal(drift_operator(phi, n), expected_drift)
    assert np.array_equal(mutation_operator(phi, n, theta), expected_mutation)


def test_theta_over_j_is_stationary_at_reference_size():
    n = 20
    theta = 2.3
    phi = _neutral_sfs(n, theta)
    derivative = drift_operator(phi, n) + mutation_operator(phi, n, theta)
    assert np.allclose(derivative[1:n], 0.0, atol=1e-13)


def test_constant_size_integration_preserves_neutral_spectrum():
    n = 24
    theta = 0.8
    phi = _neutral_sfs(n, theta)
    observed = integrate_sfs(phi, n, 0.2, lambda _t: 1.0, theta)
    assert np.allclose(observed, phi, rtol=2e-9, atol=2e-11)


def test_split_matches_binomial_partition_identity():
    n1, n2 = 3, 4
    ancestral = np.zeros(n1 + n2 + 1)
    ancestral[3] = 12.0
    split = split_1d_to_2d(ancestral, n1, n2)
    assert np.isclose(split.sum(), ancestral.sum())
    assert np.allclose(
        [split[j1, 3 - j1] for j1 in range(4)],
        [48 / 35, 216 / 35, 144 / 35, 12 / 35],
    )
