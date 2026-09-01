"""Source-derived tests for the bounded SLiM teaching mechanisms."""

from math import isclose

import numpy as np
import pytest

from watchgen.mini_slim import (
    Individual,
    Mutation,
    add_mutations,
    breakpoint_intensity,
    calculate_fitness,
    draw_breakpoints,
    gamete_from_breakpoints,
    mutation_frequency,
    parent_probabilities,
    select_parents,
    simulate,
    wf_tick_stages,
    wright_fisher_generation,
)


def test_same_lineage_on_both_haplosomes_is_homozygous():
    mutation = Mutation(position=7, s=0.2, h=0.25)
    assert calculate_fitness(Individual([mutation], [mutation])) == pytest.approx(1.2)


def test_recurrent_mutations_at_same_position_are_not_homozygous():
    first = Mutation(position=7, s=0.2, h=0.25)
    recurrent = Mutation(position=7, s=0.2, h=0.25)
    assert first.mutation_id != recurrent.mutation_id
    assert calculate_fitness(Individual([first], [recurrent])) == pytest.approx(1.05**2)


def test_default_mutation_effects_multiply_and_clip_final_fitness():
    one = Mutation(position=1, s=0.2, h=0.5)
    two = Mutation(position=2, s=-0.4, h=0.25)
    assert calculate_fitness(Individual([one, two], [])) == pytest.approx(1.1 * 0.9)
    lethal = Mutation(position=3, s=-2.0, h=1.0)
    assert calculate_fitness(Individual([lethal], [])) == 0.0


def test_parent_probabilities_use_relative_cached_fitness():
    population = [Individual(fitness=value) for value in (1.0, 2.0, 7.0)]
    assert parent_probabilities(population) == pytest.approx([0.1, 0.2, 0.7])


def test_parent_draws_are_independent_and_can_self():
    assert select_parents([Individual()], np.random.default_rng(4)) == (0, 0)


def test_zero_total_fitness_is_extinction():
    with pytest.raises(RuntimeError, match="extinction"):
        parent_probabilities([Individual(fitness=0.0)])


@pytest.mark.parametrize("probability", [0.0, 1e-8, 0.01, 0.5])
def test_breakpoint_parameterization(probability):
    intensity = breakpoint_intensity(probability)
    assert 1.0 - np.exp(-intensity) == pytest.approx(probability)


def test_breakpoint_rate_domain_matches_slim():
    with pytest.raises(ValueError):
        breakpoint_intensity(-0.1)
    with pytest.raises(ValueError):
        breakpoint_intensity(0.50001)


def test_one_base_chromosome_has_no_breakpoint_intervals():
    assert draw_breakpoints(0.5, 1, np.random.default_rng(9)) == []


def test_breakpoints_use_coordinates_one_through_length_minus_one():
    points = draw_breakpoints(0.5, 8, np.random.default_rng(12))
    assert points == sorted(set(points))
    assert all(1 <= point < 8 for point in points)


def test_observed_probability_of_a_single_breakpoint_matches_input():
    rng = np.random.default_rng(91)
    observed = np.mean([bool(draw_breakpoints(0.2, 2, rng)) for _ in range(20_000)])
    assert observed == pytest.approx(0.2, abs=0.01)


def test_breakpoint_coordinate_is_immediately_left_of_base():
    left_a, right_a = Mutation(position=0), Mutation(position=1)
    left_b, right_b = Mutation(position=0), Mutation(position=1)
    parent = Individual([left_a, right_a], [left_b, right_b])
    gamete = gamete_from_breakpoints(parent, [1], start_haplosome=0, length=2)
    assert gamete == [left_a, right_b]


def test_new_mutations_are_distinct_even_when_the_position_recurs():
    mutations = add_mutations([], mu=10, L=1, tick=8, rng=np.random.default_rng(31))
    assert {mutation.position for mutation in mutations} == {0}
    assert len({mutation.mutation_id for mutation in mutations}) == len(mutations)


def test_wf_tick_order_places_fitness_near_end_of_tick():
    stages = wf_tick_stages()
    assert stages.index("offspring generation") < stages.index("late events")
    assert stages.index("late events") < stages.index("fitness recalculation")
    assert stages[-1] == "tick increment"


def test_one_generation_preserves_requested_population_size():
    result = wright_fisher_generation(
        [Individual() for _ in range(8)], 8, 20, 0, 0, 1,
        rng=np.random.default_rng(1),
    )
    assert len(result) == 8


def test_simulation_is_reproducible_with_seed():
    first, first_stats = simulate(10, 30, 0.01, 0.02, 3, seed=17)
    second, second_stats = simulate(10, 30, 0.01, 0.02, 3, seed=17)
    assert first_stats == second_stats
    assert [[m.position for m in i.haplosome_1] for i in first] == [
        [m.position for m in i.haplosome_1] for i in second
    ]


def test_mutation_frequency_tracks_lineage_not_position():
    target, recurrent = Mutation(position=5), Mutation(position=5)
    population = [Individual([target], [recurrent]), Individual([target], [])]
    assert isclose(mutation_frequency(population, target), 0.5)
