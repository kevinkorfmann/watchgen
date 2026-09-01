"""Independent checks for the source-guided discoal kernels."""

import math

import numpy as np
import pytest

from watchgen.mini_discoal import (
    SweepTrajectory,
    coalescence_rate,
    deterministic_trajectory,
    discoal_deterministic_frequency,
    discoal_to_msprime,
    escape_probability,
    fixation_probability,
    migration_rates,
    msprime_to_discoal,
    neutral_coalescent,
    pairwise_diversity_profile,
    simulate_linked_locus_genealogy,
    stochastic_trajectory,
    structured_coalescent_sweep,
    structured_event_probabilities,
)


class FixedRNG:
    """Small deterministic RNG used to check the two-point jump law."""

    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def small_deterministic(alpha=50.0, N=200):
    return deterministic_trajectory(alpha, sweep_N=N, dt_scalar=40)


def test_fixation_probability_neutral_limit_and_boundaries():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(fixation_probability(x, 0.0), x)
    assert fixation_probability(0.0, 20.0) == pytest.approx(0.0)
    assert fixation_probability(1.0, 20.0) == pytest.approx(1.0)


def test_fixation_probability_stable_for_tiny_alpha():
    assert fixation_probability(0.37, 1e-12) == pytest.approx(0.37)


def test_deterministic_frequency_is_exact_c_translation():
    alpha = 200.0
    epsilon = 0.05 / alpha
    t_s = -2 * math.log(epsilon) / alpha
    assert discoal_deterministic_frequency(t_s, alpha) == pytest.approx(epsilon)
    expected_at_zero = 1 / (1 + (1 - epsilon) * epsilon)
    assert discoal_deterministic_frequency(0, alpha) == pytest.approx(
        expected_at_zero
    )


def test_deterministic_trajectory_uses_production_step_and_boundaries():
    N = 200
    trajectory = small_deterministic(N=N)
    assert trajectory.dt_2N == pytest.approx(1 / (40 * N))
    assert trajectory.dt_generations == pytest.approx(1 / 20)
    assert trajectory.frequencies[0] == pytest.approx(1 - 1 / (2 * N))
    assert trajectory.frequencies[-1] == pytest.approx(1 / (2 * N))
    assert np.all(np.diff(trajectory.frequencies) <= 0)


def test_trajectory_duration_units_are_consistent():
    trajectory = small_deterministic()
    assert trajectory.duration_generations == pytest.approx(
        trajectory.duration_2N * 2 * trajectory.sweep_N
    )


def test_stochastic_trajectory_has_correct_endpoints_and_reproducibility():
    kwargs = dict(alpha=50, sweep_N=200, dt_scalar=40)
    a = stochastic_trajectory(**kwargs, rng=np.random.default_rng(123))
    b = stochastic_trajectory(**kwargs, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(a.frequencies, b.frequencies)
    assert a.frequencies[0] == pytest.approx(1 - 1 / 400)
    assert a.frequencies[-1] == pytest.approx(1 / 400)
    assert np.all((a.frequencies > 0) & (a.frequencies < 1))


def test_standing_variation_adds_neutral_trajectory_phase():
    trajectory = stochastic_trajectory(
        50,
        sweep_N=200,
        selection_start_frequency=0.1,
        rng=np.random.default_rng(5),
    )
    assert trajectory.selected_steps < len(trajectory.frequencies) - 1
    assert np.any(trajectory.frequencies <= 0.1)


def test_coalescence_and_migration_rates_from_structured_coalescent():
    N = 10_000
    assert coalescence_rate(2, 0.25, N) == pytest.approx(1 / 5_000)
    assert coalescence_rate(5, 0.25, N) == pytest.approx(10 / 5_000)
    assert migration_rates(4, 3, 1e-3, 0.25) == pytest.approx(
        (0.003, 0.00075)
    )


def test_event_probabilities_are_rates_times_dt():
    probabilities = structured_event_probabilities(4, 3, 0.25, 1e-3, 10_000, 0.05)
    expected_rates = np.array(
        [6 / 5_000, 3 / 15_000, 0.003, 0.00075]
    )
    np.testing.assert_allclose(probabilities, expected_rates * 0.05)


def test_coarse_event_grid_is_rejected_instead_of_silently_biased():
    with pytest.raises(ValueError, match="too coarse"):
        structured_event_probabilities(100, 0, 1e-4, 0.0, 100, 1.0)


@pytest.mark.parametrize("r_site", [0.0, 1e-4, 1e-3])
def test_structured_kernel_conserves_lineages(r_site):
    n = 12
    trajectory = small_deterministic()
    times, n_B, n_b = structured_coalescent_sweep(
        trajectory, n, r_site, 200, np.random.default_rng(8)
    )
    assert len(times) + n_B + n_b == n
    assert all(0 <= t <= trajectory.duration_generations for t in times)


def test_no_recombination_leaves_no_wild_type_lineages():
    trajectory = small_deterministic()
    times, n_B, n_b = structured_coalescent_sweep(
        trajectory, 10, 0.0, 200, np.random.default_rng(2)
    )
    assert n_B == 1
    assert n_b == 0
    assert len(times) == 9


def test_escape_probability_is_exact_discrete_survival_product():
    trajectory = SweepTrajectory(
        np.array([0.9, 0.8, 0.6]), dt_2N=0.01, sweep_N=50, selected_steps=2
    )
    # dt is one generation; hazards are 0.02 and 0.04.
    assert escape_probability(0.1, trajectory) == pytest.approx(
        1 - (1 - 0.02) * (1 - 0.04)
    )
    assert escape_probability(0.0, trajectory) == 0.0


def test_neutral_pairwise_tmrca_has_mean_two_N():
    N = 200
    rng = np.random.default_rng(71)
    tmrcas = [neutral_coalescent(2, N, rng)[0][-1] for _ in range(20_000)]
    assert np.mean(tmrcas) == pytest.approx(2 * N, rel=0.025)


def test_linked_locus_genealogy_has_n_minus_one_coalescences():
    trajectory = small_deterministic()
    times = simulate_linked_locus_genealogy(
        10, 200, trajectory, 1e-3, rng=np.random.default_rng(91)
    )
    assert len(times) == 9
    assert np.all(np.diff(times) >= 0)


def test_pairwise_profile_is_labelled_against_pairwise_neutral_expectation():
    trajectory = small_deterministic(alpha=80)
    positions = np.array([50_000, 500_000])
    profile = pairwise_diversity_profile(
        200,
        trajectory,
        recombination_rate=1e-6,
        positions=positions,
        selected_position=50_000,
        replicates=2_000,
        seed=17,
    )
    assert profile[0] < profile[1]
    assert profile[1] == pytest.approx(1.0, abs=0.15)


def test_parameter_translation_roundtrip_and_haploid_samples():
    raw = dict(n=20, L=100_000, mu=1.25e-8, r=1e-8, s=0.01, N=10_000)
    scaled = msprime_to_discoal(**raw)
    converted = discoal_to_msprime(N=raw["N"], **scaled)
    assert converted["samples"] == raw["n"]
    assert converted["ploidy"] == 1
    assert converted["mutation_rate"] == pytest.approx(raw["mu"])
    assert converted["recombination_rate"] == pytest.approx(raw["r"])
    assert converted["selection_coefficient"] == pytest.approx(raw["s"])
    assert converted["start_frequency"] == pytest.approx(1 / (2 * raw["N"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 0, "sweep_N": 100},
        {"alpha": 10, "sweep_N": 0},
        {"alpha": 10, "sweep_N": 100, "start_frequency": 1.0},
    ],
)
def test_invalid_trajectory_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        deterministic_trajectory(**kwargs)
