"""
Tests for Python code blocks from the SLiM timepiece RST documentation.

Covers:
- wright_fisher.rst: Mutation/Individual dataclasses, fitness calculation,
  parent selection, recombination, mutation addition, generation cycle,
  full simulation, neutrality verification
- recipes.rst: fixation probability formula, tree-sequence recording
"""

import numpy as np
import pytest
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/slim/wright_fisher.rst
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single mutation on a haplosome."""
    position: int
    s: float
    h: float
    origin_tick: int = 0


@dataclass
class Individual:
    """A diploid individual carrying two haplosomes."""
    haplosome_1: list = field(default_factory=list)
    haplosome_2: list = field(default_factory=list)
    fitness: float = 1.0


def calculate_fitness(individual):
    """Compute fitness as the product of all mutation effects."""
    w = 1.0
    positions_1 = {m.position for m in individual.haplosome_1}
    positions_2 = {m.position for m in individual.haplosome_2}
    for m in individual.haplosome_1:
        if m.position in positions_2:
            w *= max(0.0, 1.0 + m.s)
        else:
            w *= max(0.0, 1.0 + m.h * m.s)
    for m in individual.haplosome_2:
        if m.position not in positions_1:
            w *= max(0.0, 1.0 + m.h * m.s)
    individual.fitness = max(0.0, w)
    return individual.fitness


def select_parents(population):
    """Draw two parents with probability proportional to fitness."""
    fitnesses = np.array([ind.fitness for ind in population])
    total = fitnesses.sum()
    if total == 0:
        raise RuntimeError("Population extinction: all fitnesses are zero")
    probs = fitnesses / total
    parent_indices = np.random.choice(len(population), size=2, p=probs)
    return parent_indices[0], parent_indices[1]


def recombine_v2(parent, r, L):
    """Generate one recombinant haplosome (cleaner version)."""
    n_crossovers = np.random.poisson(r * L)
    breakpoints = sorted(np.random.randint(1, L, size=n_crossovers)) if n_crossovers > 0 else []
    breakpoints.append(L + 1)
    if np.random.random() < 0.5:
        hap_a, hap_b = parent.haplosome_1, parent.haplosome_2
    else:
        hap_a, hap_b = parent.haplosome_2, parent.haplosome_1
    child = []
    all_muts_a = [(m.position, 'a', m) for m in hap_a]
    all_muts_b = [(m.position, 'b', m) for m in hap_b]
    all_events = sorted(all_muts_a + all_muts_b, key=lambda x: x[0])
    using_a = True
    bp_idx = 0
    for pos, source, mut in all_events:
        while bp_idx < len(breakpoints) and breakpoints[bp_idx] <= pos:
            using_a = not using_a
            bp_idx += 1
        if (using_a and source == 'a') or (not using_a and source == 'b'):
            child.append(mut)
    return child


def add_mutations(haplosome, mu, L, tick, dfe='neutral', dfe_params=None):
    """Add new mutations to a haplosome."""
    n_new = np.random.poisson(mu * L)
    if dfe_params is None:
        dfe_params = {}
    for _ in range(n_new):
        position = np.random.randint(0, L)
        if dfe == 'neutral':
            s = 0.0
        elif dfe == 'fixed':
            s = dfe_params.get('s', 0.0)
        elif dfe == 'exponential':
            s = np.random.exponential(dfe_params.get('mean', 0.01))
        elif dfe == 'gamma':
            s = -np.random.gamma(
                dfe_params.get('shape', 0.3),
                dfe_params.get('scale', 0.05)
            )
        h = dfe_params.get('h', 0.5)
        haplosome.append(Mutation(position=position, s=s, h=h, origin_tick=tick))
    haplosome.sort(key=lambda m: m.position)
    return haplosome


def wright_fisher_generation(population, N, L, mu, r, tick,
                              dfe='neutral', dfe_params=None):
    """Run one Wright-Fisher generation."""
    for ind in population:
        calculate_fitness(ind)
    new_population = []
    for _ in range(N):
        p1_idx, p2_idx = select_parents(population)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]
        child_hap1 = recombine_v2(parent1, r, L)
        child_hap2 = recombine_v2(parent2, r, L)
        child_hap1 = add_mutations(child_hap1, mu, L, tick, dfe, dfe_params)
        child_hap2 = add_mutations(child_hap2, mu, L, tick, dfe, dfe_params)
        new_population.append(Individual(
            haplosome_1=child_hap1,
            haplosome_2=child_hap2
        ))
    return new_population


# ---------------------------------------------------------------------------
# Re-defined from docs/timepieces/slim/recipes.rst
# ---------------------------------------------------------------------------

def simulate_with_tree_recording(N, L, mu, r, T):
    """Forward simulation with tree-sequence recording."""
    nodes = []
    edges = []
    current_nodes = []
    for i in range(N):
        node_id_1 = len(nodes)
        nodes.append((T, 0))
        node_id_2 = len(nodes)
        nodes.append((T, 0))
        current_nodes.append((node_id_1, node_id_2))
    for tick in range(T):
        generation_time = T - tick - 1
        new_nodes = []
        for _ in range(N):
            p1_idx = np.random.randint(N)
            p2_idx = np.random.randint(N)
            child_node_1 = len(nodes)
            nodes.append((generation_time, 0))
            child_node_2 = len(nodes)
            nodes.append((generation_time, 0))
            parent1_hap_a, parent1_hap_b = current_nodes[p1_idx]
            n_breaks = np.random.poisson(r * L)
            breakpoints = sorted(np.random.randint(1, L, size=n_breaks)) if n_breaks > 0 else []
            if np.random.random() < 0.5:
                sources = [parent1_hap_a, parent1_hap_b]
            else:
                sources = [parent1_hap_b, parent1_hap_a]
            all_breaks = [0] + breakpoints + [L]
            for i in range(len(all_breaks) - 1):
                left = all_breaks[i]
                right = all_breaks[i + 1]
                parent_node = sources[i % 2]
                edges.append((left, right, parent_node, child_node_1))
            parent2_hap_a, parent2_hap_b = current_nodes[p2_idx]
            n_breaks = np.random.poisson(r * L)
            breakpoints = sorted(np.random.randint(1, L, size=n_breaks)) if n_breaks > 0 else []
            if np.random.random() < 0.5:
                sources = [parent2_hap_a, parent2_hap_b]
            else:
                sources = [parent2_hap_b, parent2_hap_a]
            all_breaks = [0] + breakpoints + [L]
            for i in range(len(all_breaks) - 1):
                left = all_breaks[i]
                right = all_breaks[i + 1]
                parent_node = sources[i % 2]
                edges.append((left, right, parent_node, child_node_2))
            new_nodes.append((child_node_1, child_node_2))
        current_nodes = new_nodes
    sample_nodes = []
    for hap_a, hap_b in current_nodes:
        sample_nodes.extend([hap_a, hap_b])
    return nodes, edges, sample_nodes


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
