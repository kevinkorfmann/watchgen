"""
Tests for watchgen.mini_slim -- the mini SLiM forward-time Wright-Fisher simulator.

Adapted from tests/test_timepieces_slim.py. All functions are imported from
watchgen.mini_slim rather than redefined here.

Covers:
- wright_fisher.rst: Mutation/Individual dataclasses, fitness calculation,
  parent selection, recombination, mutation addition, generation cycle,
  full simulation, neutrality verification
- recipes.rst: fixation probability formula, tree-sequence recording
"""

import numpy as np
import pytest

from watchgen.mini_slim import (
    Mutation,
    Individual,
    calculate_fitness,
    select_parents,
    recombine,
    recombine_v2,
    add_mutations,
    wright_fisher_generation,
    simulate,
    verify_neutrality,
    simulate_sweep,
    estimate_fixation_probability,
    simulate_bgs,
    simulate_with_tree_recording,
    add_mutations_to_tree_sequence,
)


# ===========================================================================
# Tests for Mutation dataclass
# ===========================================================================

class TestMutation:
    def test_creation(self):
        """Should create a mutation with all fields."""
        m = Mutation(position=500, s=0.1, h=0.5, origin_tick=10)
        assert m.position == 500
        assert m.s == 0.1
        assert m.h == 0.5
        assert m.origin_tick == 10

    def test_default_origin(self):
        """Default origin_tick should be 0."""
        m = Mutation(position=100, s=0.0, h=0.5)
        assert m.origin_tick == 0


# ===========================================================================
# Tests for Individual dataclass
# ===========================================================================

class TestIndividual:
    def test_default_empty(self):
        """Default individual should have empty haplosomes and fitness=1."""
        ind = Individual()
        assert ind.haplosome_1 == []
        assert ind.haplosome_2 == []
        assert ind.fitness == 1.0


# ===========================================================================
# Tests for calculate_fitness
# ===========================================================================

class TestCalculateFitness:
    def test_no_mutations(self):
        """Individual with no mutations should have fitness 1.0."""
        ind = Individual()
        assert calculate_fitness(ind) == 1.0

    def test_heterozygous_beneficial(self):
        """One het beneficial mutation (s=0.1, h=0.5) -> fitness 1.05."""
        ind = Individual()
        ind.haplosome_1 = [Mutation(position=500, s=0.1, h=0.5)]
        w = calculate_fitness(ind)
        assert np.isclose(w, 1.05)

    def test_homozygous_deleterious(self):
        """Homozygous deleterious (s=-0.05) -> fitness 0.95."""
        ind = Individual()
        m1 = Mutation(position=100, s=-0.05, h=0.5)
        m2 = Mutation(position=100, s=-0.05, h=0.5)
        ind.haplosome_1 = [m1]
        ind.haplosome_2 = [m2]
        w = calculate_fitness(ind)
        assert np.isclose(w, 0.95)

    def test_fitness_clamped_to_zero(self):
        """Fitness should never be negative."""
        ind = Individual()
        ind.haplosome_1 = [Mutation(position=100, s=-2.0, h=1.0)]
        w = calculate_fitness(ind)
        assert w >= 0.0

    def test_multiplicative_effects(self):
        """Two het mutations should multiply: (1+h*s1)*(1+h*s2)."""
        ind = Individual()
        ind.haplosome_1 = [
            Mutation(position=100, s=0.1, h=0.5),
            Mutation(position=200, s=0.2, h=0.5),
        ]
        w = calculate_fitness(ind)
        expected = (1 + 0.5 * 0.1) * (1 + 0.5 * 0.2)
        assert np.isclose(w, expected)

    def test_recessive_heterozygote(self):
        """Fully recessive (h=0) het should have no fitness effect."""
        ind = Individual()
        ind.haplosome_1 = [Mutation(position=100, s=0.1, h=0.0)]
        w = calculate_fitness(ind)
        assert np.isclose(w, 1.0)


# ===========================================================================
# Tests for select_parents
# ===========================================================================

class TestSelectParents:
    def test_returns_valid_indices(self):
        """Parent indices should be within population range."""
        np.random.seed(42)
        pop = [Individual() for _ in range(10)]
        for ind in pop:
            ind.fitness = 1.0
        p1, p2 = select_parents(pop)
        assert 0 <= p1 < 10
        assert 0 <= p2 < 10

    def test_fitter_individual_selected_more(self):
        """A much fitter individual should be selected more often."""
        np.random.seed(42)
        pop = [Individual() for _ in range(10)]
        for ind in pop:
            ind.fitness = 1.0
        pop[0].fitness = 100.0  # much fitter
        counts = np.zeros(10)
        for _ in range(1000):
            p1, p2 = select_parents(pop)
            counts[p1] += 1
            counts[p2] += 1
        # Individual 0 should be selected much more often
        assert counts[0] > counts[1] * 5

    def test_all_zero_fitness_raises(self):
        """All-zero fitnesses should raise RuntimeError."""
        pop = [Individual() for _ in range(5)]
        for ind in pop:
            ind.fitness = 0.0
        with pytest.raises(RuntimeError):
            select_parents(pop)


# ===========================================================================
# Tests for recombine_v2
# ===========================================================================

class TestRecombineV2:
    def test_no_recombination_copies_one_haplosome(self):
        """With r=0, should return a copy of one haplosome."""
        np.random.seed(42)
        parent = Individual()
        m1 = Mutation(position=100, s=0.0, h=0.5)
        m2 = Mutation(position=200, s=0.0, h=0.5)
        parent.haplosome_1 = [m1]
        parent.haplosome_2 = [m2]
        child = recombine_v2(parent, r=0, L=1000)
        # Should get exactly one of the two mutations
        assert len(child) == 1

    def test_empty_parent(self):
        """Recombining an empty parent should produce empty haplosome."""
        np.random.seed(42)
        parent = Individual()
        child = recombine_v2(parent, r=1e-4, L=10000)
        assert child == []


# ===========================================================================
# Tests for recombine (v1)
# ===========================================================================

class TestRecombine:
    def test_no_recombination_copies_one_haplosome(self):
        """With r=0, should return a copy of one haplosome."""
        np.random.seed(42)
        parent = Individual()
        m1 = Mutation(position=100, s=0.0, h=0.5)
        m2 = Mutation(position=200, s=0.0, h=0.5)
        parent.haplosome_1 = [m1]
        parent.haplosome_2 = [m2]
        child = recombine(parent, r=0, L=1000)
        # Should get exactly one of the two mutations
        assert len(child) == 1

    def test_empty_parent(self):
        """Recombining an empty parent should produce empty haplosome."""
        np.random.seed(42)
        parent = Individual()
        child = recombine(parent, r=1e-4, L=10000)
        assert child == []


# ===========================================================================
# Tests for add_mutations
# ===========================================================================

class TestAddMutations:
    def test_neutral_mutations_have_s_zero(self):
        """Neutral DFE should produce s=0 mutations."""
        np.random.seed(42)
        hap = []
        hap = add_mutations(hap, mu=0.01, L=1000, tick=1, dfe='neutral')
        for m in hap:
            assert m.s == 0.0

    def test_fixed_dfe(self):
        """Fixed DFE should produce mutations with specified s."""
        np.random.seed(42)
        hap = []
        hap = add_mutations(hap, mu=0.01, L=1000, tick=1,
                            dfe='fixed', dfe_params={'s': -0.01})
        for m in hap:
            assert m.s == -0.01

    def test_sorted_by_position(self):
        """Mutations should be sorted by position."""
        np.random.seed(42)
        hap = []
        hap = add_mutations(hap, mu=0.1, L=1000, tick=1)
        positions = [m.position for m in hap]
        assert positions == sorted(positions)

    def test_expected_number(self):
        """Mean number of mutations should be close to mu*L."""
        np.random.seed(42)
        counts = []
        for _ in range(1000):
            hap = []
            hap = add_mutations(hap, mu=0.001, L=10000, tick=1)
            counts.append(len(hap))
        mean_count = np.mean(counts)
        expected = 0.001 * 10000
        assert abs(mean_count - expected) / expected < 0.1


# ===========================================================================
# Tests for wright_fisher_generation
# ===========================================================================

class TestWrightFisherGeneration:
    def test_population_size_preserved(self):
        """Population size should remain constant."""
        np.random.seed(42)
        N = 20
        pop = [Individual() for _ in range(N)]
        new_pop = wright_fisher_generation(pop, N, L=1000, mu=0, r=0, tick=0)
        assert len(new_pop) == N

    def test_mutations_accumulate(self):
        """After several generations, mutations should accumulate."""
        np.random.seed(42)
        N = 50
        pop = [Individual() for _ in range(N)]
        for tick in range(10):
            pop = wright_fisher_generation(pop, N, L=5000, mu=1e-3,
                                           r=1e-4, tick=tick)
        total_muts = sum(len(ind.haplosome_1) + len(ind.haplosome_2)
                         for ind in pop)
        assert total_muts > 0


# ===========================================================================
# Tests for simulate (full simulation)
# ===========================================================================

class TestSimulate:
    def test_returns_population_and_stats(self):
        """simulate() should return population and stats dict."""
        np.random.seed(42)
        pop, stats = simulate(N=20, L=1000, mu=1e-3, r=1e-4, T=10,
                              track_every=5)
        assert len(pop) == 20
        assert 'generation' in stats
        assert 'mean_fitness' in stats
        assert len(stats['generation']) > 0

    def test_neutral_fitness_stays_one(self):
        """Under neutrality, mean fitness should remain 1.0."""
        np.random.seed(42)
        pop, stats = simulate(N=20, L=1000, mu=1e-3, r=1e-4, T=10,
                              dfe='neutral', track_every=5)
        for w in stats['mean_fitness']:
            assert np.isclose(w, 1.0)


# ===========================================================================
# Tests for tree-sequence recording
# ===========================================================================

class TestTreeSequenceRecording:
    def test_node_count(self):
        """Should have 2*N*(T+1) total nodes."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 0, T)
        expected_nodes = 2 * N * (T + 1)
        assert len(nodes) == expected_nodes

    def test_sample_count(self):
        """Should have 2*N sample nodes."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 0, T)
        assert len(samples) == 2 * N

    def test_edges_created(self):
        """Should create edges during simulation."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 1e-4, T)
        assert len(edges) > 0

    def test_edge_spans_valid(self):
        """All edges should have left < right."""
        np.random.seed(42)
        N, L, T = 10, 1000, 3
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 1e-4, T)
        for left, right, parent, child in edges:
            assert left < right

    def test_sample_nodes_are_present_time(self):
        """Sample nodes should have time = 0 (present)."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 0, T)
        for s in samples:
            assert nodes[s][0] == 0


# ===========================================================================
# Tests for add_mutations_to_tree_sequence
# ===========================================================================

class TestAddMutationsToTreeSequence:
    def test_mutations_placed(self):
        """Should place mutations on the tree sequence."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 1e-4, T)
        muts = add_mutations_to_tree_sequence(nodes, edges, samples,
                                               mu=1e-3, L=L)
        assert len(muts) > 0

    def test_mutations_sorted_by_position(self):
        """Mutations should be sorted by position."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 1e-4, T)
        muts = add_mutations_to_tree_sequence(nodes, edges, samples,
                                               mu=1e-3, L=L)
        positions = [m[0] for m in muts]
        assert positions == sorted(positions)

    def test_mutation_positions_within_range(self):
        """All mutation positions should be within [0, L]."""
        np.random.seed(42)
        N, L, T = 10, 1000, 5
        nodes, edges, samples = simulate_with_tree_recording(N, L, 0, 1e-4, T)
        muts = add_mutations_to_tree_sequence(nodes, edges, samples,
                                               mu=1e-3, L=L)
        for pos, node, time in muts:
            assert 0 <= pos <= L


# ===========================================================================
# Tests for fixation probability formula
# ===========================================================================

class TestFixationProbability:
    def test_neutral_fixation_probability(self):
        """For s=0, fixation probability should be 1/(2N)."""
        N = 1000
        s = 0.0
        p_fix = 1.0 / (2 * N)
        assert np.isclose(p_fix, 0.0005)

    def test_beneficial_fixation_probability(self):
        """For s>0 and large 4Ns, P_fix ~ 2s."""
        N = 10000
        s = 0.01
        x = 4 * N * s
        p_fix = (2 * s) / (1 - np.exp(-x))
        # For large 4Ns, should be approximately 2s
        assert abs(p_fix - 2 * s) < 0.001

    def test_fixation_time_formula(self):
        """Fixation time should be approximately (2/s)*ln(4Ns)."""
        N = 10000
        s = 0.01
        T_fix = (2 / s) * np.log(4 * N * s)
        # Should be much less than neutral fixation time (4N)
        assert T_fix < 4 * N
        assert T_fix > 0

    def test_stronger_selection_faster_fixation(self):
        """Stronger selection should give faster fixation."""
        N = 10000
        T1 = (2 / 0.01) * np.log(4 * N * 0.01)
        T2 = (2 / 0.05) * np.log(4 * N * 0.05)
        assert T2 < T1


# ===========================================================================
# Tests for verify_neutrality
# ===========================================================================

class TestVerifyNeutrality:
    def test_runs_without_error(self):
        """verify_neutrality should run without raising."""
        np.random.seed(42)
        N = 20
        pop = [Individual() for _ in range(N)]
        for tick in range(10):
            pop = wright_fisher_generation(pop, N, L=1000, mu=1e-3,
                                           r=1e-4, tick=tick)
        # Should not raise
        verify_neutrality(pop, N, L=1000, mu=1e-3)


# ===========================================================================
# Tests for background selection
# ===========================================================================

class TestBackgroundSelection:
    def test_returns_stats(self):
        """simulate_bgs should return a stats dict with expected keys."""
        np.random.seed(42)
        stats = simulate_bgs(
            N=20, L=1000,
            mu_neutral=1e-3,
            mu_deleterious=5e-3,
            s_deleterious=-0.05,
            r=1e-4,
            T=10,
            track_interval=5
        )
        assert 'generation' in stats
        assert 'mean_fitness' in stats
        assert 'neutral_diversity' in stats
        assert len(stats['generation']) > 0
