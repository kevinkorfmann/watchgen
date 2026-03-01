"""
Mini SLiM -- A minimal forward-time Wright-Fisher population genetics simulator.

SLiM is a forward-time population genetics simulator: given a population size,
genome length, mutation rate, recombination rate, and a model of natural
selection, it evolves a population generation by generation from past to present.

This module implements the core mechanism in pure Python/NumPy:

1. The Wright-Fisher Cycle (the escapement) -- The discrete-generation engine:
   in each tick, parents are selected with probability proportional to fitness,
   offspring are generated through recombination, and new mutations are added.

2. Fitness and Selection (the mainspring) -- Each mutation carries a selection
   coefficient s and a dominance coefficient h. An individual's fitness is the
   product of the effects of all its mutations. Parents are drawn in proportion
   to their fitness.

3. Recipes (the complications) -- Practical applications: a selective sweep,
   background selection, and tree-sequence recording.

Reference: Haller & Messer (2019), SLiM 3.
"""

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model (from wright_fisher.rst, Step 1)
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single mutation on a haplosome.

    Each mutation has a genomic position, a selection coefficient s,
    and a dominance coefficient h. The mutation also records when
    (which generation) and where (which individual) it arose.
    """
    position: int         # base-pair position on the chromosome
    s: float              # selection coefficient (>0 beneficial, <0 deleterious)
    h: float              # dominance coefficient (0.5 = codominant)
    origin_tick: int = 0  # generation when the mutation arose


@dataclass
class Individual:
    """A diploid individual carrying two haplosomes.

    Each haplosome is a sorted list of Mutation objects. Fitness
    is computed from the combined effects of all mutations.
    """
    haplosome_1: list = field(default_factory=list)
    haplosome_2: list = field(default_factory=list)
    fitness: float = 1.0


# ---------------------------------------------------------------------------
# Fitness calculation (from wright_fisher.rst, Step 2)
# ---------------------------------------------------------------------------

def calculate_fitness(individual):
    """Compute fitness as the product of all mutation effects.

    For each mutation, we check whether the individual is homozygous
    (mutation on both haplosomes) or heterozygous (mutation on one).

    This mirrors SLiM's core fitness calculation in population.cpp.
    """
    w = 1.0

    # Collect mutations by (position, selection coefficient) identity
    muts_1 = {(m.position, m.s, m.h): m for m in individual.haplosome_1}
    muts_2 = {(m.position, m.s, m.h): m for m in individual.haplosome_2}

    # Mutations on haplosome 1
    for key, m in muts_1.items():
        if key in muts_2:
            # Homozygous: same mutation on both haplosomes
            w *= max(0.0, 1.0 + m.s)
        else:
            # Heterozygous: mutation only on haplosome 1
            w *= max(0.0, 1.0 + m.h * m.s)

    # Mutations on haplosome 2 that are NOT shared with haplosome 1
    for key, m in muts_2.items():
        if key not in muts_1:
            # Heterozygous: mutation only on haplosome 2
            w *= max(0.0, 1.0 + m.h * m.s)

    # Clamped to zero (SLiM also does this: fitness cannot be negative)
    individual.fitness = max(0.0, w)
    return individual.fitness


# ---------------------------------------------------------------------------
# Parent selection (from wright_fisher.rst, Step 3)
# ---------------------------------------------------------------------------

def select_parents(population):
    """Draw two parents with probability proportional to fitness.

    Returns indices into the population list. The two parents
    may be the same individual (selfing is allowed in the basic
    Wright-Fisher model).
    """
    fitnesses = np.array([ind.fitness for ind in population])
    total = fitnesses.sum()

    if total == 0:
        # All individuals have zero fitness -- population is dead.
        # This can happen with very strong purifying selection.
        raise RuntimeError("Population extinction: all fitnesses are zero")

    probs = fitnesses / total

    # Draw two parents independently (with replacement)
    parent_indices = np.random.choice(len(population), size=2, p=probs)
    return parent_indices[0], parent_indices[1]


# ---------------------------------------------------------------------------
# Recombination (from wright_fisher.rst, Step 4)
# ---------------------------------------------------------------------------

def recombine(parent, r, L):
    """Generate one recombinant haplosome from a diploid parent.

    This is the core of meiosis: start with one randomly chosen
    parental haplosome, then switch at each crossover point.

    Parameters
    ----------
    parent : Individual
        The diploid parent with two haplosomes.
    r : float
        Per-bp, per-generation recombination rate.
    L : int
        Chromosome length in base pairs.

    Returns
    -------
    child_haplosome : list of Mutation
        The recombinant haplosome for the child.
    """
    # Draw number of crossovers from Poisson distribution
    n_crossovers = np.random.poisson(r * L)

    if n_crossovers == 0:
        # No recombination: child gets a complete copy of one haplosome
        if np.random.random() < 0.5:
            return list(parent.haplosome_1)
        else:
            return list(parent.haplosome_2)

    # Draw crossover positions (sorted)
    breakpoints = sorted(np.random.randint(1, L, size=n_crossovers))

    # Start with a randomly chosen haplosome
    if np.random.random() < 0.5:
        sources = [parent.haplosome_1, parent.haplosome_2]
    else:
        sources = [parent.haplosome_2, parent.haplosome_1]

    # Build the recombinant haplosome by alternating between sources
    child_haplosome = []

    for m in sources[0] + sources[1]:
        # Determine which source this mutation falls in
        # by counting how many breakpoints are to the left of it
        n_breaks_before = sum(1 for bp in breakpoints if bp <= m.position)
        source_idx = n_breaks_before % 2
        if source_idx == 0 and m in sources[0]:
            child_haplosome.append(m)
        elif source_idx == 1 and m in sources[1]:
            child_haplosome.append(m)

    # Sort by position for consistency
    child_haplosome.sort(key=lambda m: m.position)
    return child_haplosome


def recombine_v2(parent, r, L):
    """Generate one recombinant haplosome (cleaner version).

    Walk along the chromosome, switching between parental haplosomes
    at each breakpoint. Collect mutations from whichever haplosome
    is currently active.
    """
    n_crossovers = np.random.poisson(r * L)
    breakpoints = sorted(np.random.randint(1, L, size=n_crossovers)) if n_crossovers > 0 else []
    breakpoints.append(L + 1)  # sentinel: end of chromosome

    # Choose starting haplosome randomly
    if np.random.random() < 0.5:
        hap_a, hap_b = parent.haplosome_1, parent.haplosome_2
    else:
        hap_a, hap_b = parent.haplosome_2, parent.haplosome_1

    child = []

    # Merge mutations from both haplosomes in position order
    all_muts_a = [(m.position, 'a', m) for m in hap_a]
    all_muts_b = [(m.position, 'b', m) for m in hap_b]
    all_events = sorted(all_muts_a + all_muts_b, key=lambda x: x[0])

    using_a = True  # start with hap_a
    bp_idx = 0

    for pos, source, mut in all_events:
        # Advance through breakpoints
        while bp_idx < len(breakpoints) and breakpoints[bp_idx] <= pos:
            using_a = not using_a
            bp_idx += 1

        # Keep mutation if it comes from the currently active haplosome
        if (using_a and source == 'a') or (not using_a and source == 'b'):
            child.append(mut)

    return child


# ---------------------------------------------------------------------------
# Mutation (from wright_fisher.rst, Step 5)
# ---------------------------------------------------------------------------

def add_mutations(haplosome, mu, L, tick, dfe='neutral', dfe_params=None):
    """Add new mutations to a haplosome.

    Parameters
    ----------
    haplosome : list of Mutation
        The haplosome to mutate.
    mu : float
        Per-bp, per-generation mutation rate.
    L : int
        Chromosome length.
    tick : int
        Current generation (for recording origin time).
    dfe : str
        Distribution of fitness effects: 'neutral', 'fixed',
        'exponential', or 'gamma'.
    dfe_params : dict
        Parameters for the DFE (e.g., {'s': -0.01} for fixed,
        {'mean': 0.01} for exponential,
        {'shape': 0.3, 'scale': 0.05} for gamma).

    Returns
    -------
    haplosome : list of Mutation
        The haplosome with new mutations added.
    """
    # Number of new mutations: Poisson(mu * L)
    n_new = np.random.poisson(mu * L)

    if dfe_params is None:
        dfe_params = {}

    for _ in range(n_new):
        # Random position along the chromosome
        position = np.random.randint(0, L)

        # Draw selection coefficient from the DFE
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
            )  # negative because deleterious

        h = dfe_params.get('h', 0.5)  # default: codominant
        haplosome.append(Mutation(position=position, s=s, h=h,
                                  origin_tick=tick))

    # Keep sorted by position
    haplosome.sort(key=lambda m: m.position)
    return haplosome


# ---------------------------------------------------------------------------
# Complete generation cycle (from wright_fisher.rst, Step 6)
# ---------------------------------------------------------------------------

def wright_fisher_generation(population, N, L, mu, r, tick,
                              dfe='neutral', dfe_params=None):
    """Run one Wright-Fisher generation.

    This is the complete cycle:
    1. Calculate fitness for all individuals
    2. Generate N offspring by:
       a. Selecting parents proportional to fitness
       b. Recombining parental haplosomes
       c. Adding new mutations
    3. Replace the old population with the new one

    Parameters
    ----------
    population : list of Individual
        Current generation (N diploid individuals).
    N : int
        Population size (constant).
    L : int
        Chromosome length.
    mu : float
        Per-bp, per-generation mutation rate.
    r : float
        Per-bp, per-generation recombination rate.
    tick : int
        Current generation number.
    dfe : str
        Distribution of fitness effects.
    dfe_params : dict
        DFE parameters.

    Returns
    -------
    new_population : list of Individual
        The next generation.
    """
    # Step 1: Recalculate fitness for all individuals
    for ind in population:
        calculate_fitness(ind)

    # Step 2: Generate N offspring
    new_population = []
    for _ in range(N):
        # Select two parents (fitness-proportional)
        p1_idx, p2_idx = select_parents(population)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        # Recombine: each parent contributes one haplosome
        child_hap1 = recombine_v2(parent1, r, L)
        child_hap2 = recombine_v2(parent2, r, L)

        # Add new mutations
        child_hap1 = add_mutations(child_hap1, mu, L, tick, dfe, dfe_params)
        child_hap2 = add_mutations(child_hap2, mu, L, tick, dfe, dfe_params)

        new_population.append(Individual(
            haplosome_1=child_hap1,
            haplosome_2=child_hap2
        ))

    return new_population


# ---------------------------------------------------------------------------
# Full simulation (from wright_fisher.rst, Step 7)
# ---------------------------------------------------------------------------

def simulate(N, L, mu, r, T, dfe='neutral', dfe_params=None,
             track_every=100):
    """Run a complete Wright-Fisher forward simulation.

    Parameters
    ----------
    N : int
        Diploid population size.
    L : int
        Chromosome length (bp).
    mu : float
        Per-bp, per-generation mutation rate.
    r : float
        Per-bp, per-generation recombination rate.
    T : int
        Number of generations to simulate.
    dfe : str
        Distribution of fitness effects.
    dfe_params : dict
        DFE parameters.
    track_every : int
        Record statistics every this many generations.

    Returns
    -------
    population : list of Individual
        Final population.
    stats : dict
        Time series of population statistics.
    """
    # Initialize: N individuals with empty haplosomes
    population = [Individual() for _ in range(N)]

    stats = {
        'generation': [],
        'mean_fitness': [],
        'num_segregating': [],
        'mean_mutations_per_individual': [],
    }

    for tick in range(T):
        population = wright_fisher_generation(
            population, N, L, mu, r, tick, dfe, dfe_params
        )

        # Track statistics periodically
        if tick % track_every == 0 or tick == T - 1:
            fitnesses = [ind.fitness for ind in population]
            n_muts = [len(ind.haplosome_1) + len(ind.haplosome_2)
                      for ind in population]

            # Count segregating mutations (present in at least one
            # but not all individuals)
            all_positions = set()
            for ind in population:
                for m in ind.haplosome_1 + ind.haplosome_2:
                    all_positions.add(m.position)

            stats['generation'].append(tick)
            stats['mean_fitness'].append(np.mean(fitnesses))
            stats['num_segregating'].append(len(all_positions))
            stats['mean_mutations_per_individual'].append(np.mean(n_muts))

            print(f"Gen {tick:>6d}: mean_w={np.mean(fitnesses):.4f}, "
                  f"seg_sites={len(all_positions):>5d}, "
                  f"muts/ind={np.mean(n_muts):.1f}")

    return population, stats


# ---------------------------------------------------------------------------
# Verify against coalescent theory (from wright_fisher.rst, Step 8)
# ---------------------------------------------------------------------------

def verify_neutrality(pop, N, L, mu):
    """Check that our neutral simulation matches coalescent predictions.

    Compares the number of segregating sites and heterozygosity to
    the theoretical expectations from coalescent theory.
    """
    theta = 4 * N * mu * L  # population-scaled mutation rate

    # Count segregating sites
    all_positions = set()
    for ind in pop:
        for m in ind.haplosome_1 + ind.haplosome_2:
            all_positions.add(m.position)
    S_observed = len(all_positions)

    # Harmonic number for 2N - 1
    harmonic = sum(1.0 / k for k in range(1, 2 * N))
    S_expected = theta * harmonic

    print(f"Segregating sites:")
    print(f"  Observed:  {S_observed}")
    print(f"  Expected:  {S_expected:.1f}")
    print(f"  Theta:     {theta:.1f}")


# ---------------------------------------------------------------------------
# Recipe 1: Selective sweep (from recipes.rst)
# ---------------------------------------------------------------------------

def simulate_sweep(N, L, mu, r, s_beneficial, position_selected,
                    T_burnin, T_after=2000, track_interval=50):
    """Simulate a selective sweep.

    1. Burn in under neutrality to reach mutation-drift equilibrium.
    2. Introduce a single beneficial mutation at a specified position.
    3. Continue simulation and track the beneficial allele frequency.

    Parameters
    ----------
    N : int
        Diploid population size.
    L : int
        Chromosome length.
    mu : float
        Neutral mutation rate (per bp per generation).
    r : float
        Recombination rate (per bp per generation).
    s_beneficial : float
        Selection coefficient of the sweeping allele.
    position_selected : int
        Genomic position of the beneficial mutation.
    T_burnin : int
        Generations of neutral burn-in.
    T_after : int
        Maximum generations after introducing the mutation.
    track_interval : int
        How often to record the allele frequency.

    Returns
    -------
    trajectory : list of (generation, frequency)
        Frequency trajectory of the beneficial allele.
    fixed : bool
        Whether the allele reached fixation.
    """
    # Initialize population
    population = [Individual() for _ in range(N)]

    # Phase 1: Neutral burn-in
    print(f"Burning in for {T_burnin} generations...")
    for tick in range(T_burnin):
        population = wright_fisher_generation(
            population, N, L, mu, r, tick, dfe='neutral'
        )

    # Phase 2: Introduce beneficial mutation on one haplosome
    # Pick a random individual and add the mutation to haplosome 1
    lucky = np.random.randint(N)
    sweep_mutation = Mutation(
        position=position_selected,
        s=s_beneficial,
        h=0.5,  # codominant
        origin_tick=T_burnin
    )
    population[lucky].haplosome_1.append(sweep_mutation)
    population[lucky].haplosome_1.sort(key=lambda m: m.position)

    print(f"Introduced beneficial mutation (s={s_beneficial}) "
          f"at position {position_selected}")

    # Phase 3: Simulate and track the sweep
    trajectory = [(0, 1.0 / (2 * N))]  # initial frequency = 1/(2N)

    for tick in range(T_after):
        gen = T_burnin + tick + 1
        population = wright_fisher_generation(
            population, N, L, mu, r, gen, dfe='neutral'
        )

        # Count copies of the beneficial mutation
        count = 0
        total = 2 * N
        for ind in population:
            for m in ind.haplosome_1:
                if (m.position == position_selected and
                    m.s == s_beneficial):
                    count += 1
                    break
            for m in ind.haplosome_2:
                if (m.position == position_selected and
                    m.s == s_beneficial):
                    count += 1
                    break

        freq = count / total

        if tick % track_interval == 0:
            trajectory.append((tick + 1, freq))
            print(f"  Gen {tick+1:>5d}: freq = {freq:.4f}")

        # Check for fixation or loss
        if freq >= 1.0:
            trajectory.append((tick + 1, 1.0))
            print(f"  FIXED at generation {tick + 1}")
            return trajectory, True
        elif freq <= 0.0:
            trajectory.append((tick + 1, 0.0))
            print(f"  LOST at generation {tick + 1}")
            return trajectory, False

    return trajectory, False


def estimate_fixation_probability(N, s, n_trials=500, **sim_kwargs):
    """Estimate fixation probability by repeated simulation.

    Run many independent introductions of a beneficial mutation
    and count how often it fixes.
    """
    n_fixed = 0
    for trial in range(n_trials):
        _, fixed = simulate_sweep(N=N, s_beneficial=s, **sim_kwargs)
        if fixed:
            n_fixed += 1

    p_fix_observed = n_fixed / n_trials
    # Theoretical prediction (Kimura)
    if s == 0:
        p_fix_theory = 1 / (2 * N)
    else:
        x = 4 * N * s
        p_fix_theory = (2 * s) / (1 - np.exp(-x))

    print(f"\nFixation probability (N={N}, s={s}):")
    print(f"  Observed:    {p_fix_observed:.4f} ({n_fixed}/{n_trials})")
    print(f"  Theory:      {p_fix_theory:.4f}")


# ---------------------------------------------------------------------------
# Recipe 2: Background selection (from recipes.rst)
# ---------------------------------------------------------------------------

def simulate_bgs(N, L, mu_neutral, mu_deleterious, s_deleterious,
                  r, T, track_interval=100):
    """Simulate background selection.

    Two classes of mutations:
    1. Neutral (s=0): these are what we measure diversity in.
    2. Deleterious (s<0): these create the background selection effect.

    Parameters
    ----------
    N : int
        Diploid population size.
    L : int
        Chromosome length.
    mu_neutral : float
        Per-bp neutral mutation rate.
    mu_deleterious : float
        Per-bp deleterious mutation rate.
    s_deleterious : float
        Selection coefficient (negative) for deleterious mutations.
    r : float
        Recombination rate.
    T : int
        Number of generations.

    Returns
    -------
    stats : dict
        Time series of population statistics.
    """
    population = [Individual() for _ in range(N)]
    stats = {'generation': [], 'mean_fitness': [], 'neutral_diversity': []}

    for tick in range(T):
        # Recalculate fitness
        for ind in population:
            calculate_fitness(ind)

        # Generate offspring
        new_pop = []
        for _ in range(N):
            p1, p2 = select_parents(population)
            child_h1 = recombine_v2(population[p1], r, L)
            child_h2 = recombine_v2(population[p2], r, L)

            # Add neutral mutations
            child_h1 = add_mutations(child_h1, mu_neutral, L, tick,
                                      dfe='neutral')
            child_h2 = add_mutations(child_h2, mu_neutral, L, tick,
                                      dfe='neutral')

            # Add deleterious mutations
            child_h1 = add_mutations(child_h1, mu_deleterious, L, tick,
                                      dfe='fixed',
                                      dfe_params={'s': s_deleterious,
                                                  'h': 0.5})
            child_h2 = add_mutations(child_h2, mu_deleterious, L, tick,
                                      dfe='fixed',
                                      dfe_params={'s': s_deleterious,
                                                  'h': 0.5})

            new_pop.append(Individual(haplosome_1=child_h1,
                                       haplosome_2=child_h2))
        population = new_pop

        if tick % track_interval == 0:
            fitnesses = [ind.fitness for ind in population]

            # Count neutral segregating sites (s == 0 mutations)
            neutral_positions = set()
            for ind in population:
                for m in ind.haplosome_1 + ind.haplosome_2:
                    if m.s == 0:
                        neutral_positions.add(m.position)

            stats['generation'].append(tick)
            stats['mean_fitness'].append(np.mean(fitnesses))
            stats['neutral_diversity'].append(len(neutral_positions))

            print(f"Gen {tick:>6d}: mean_w={np.mean(fitnesses):.4f}, "
                  f"neutral_seg={len(neutral_positions):>5d}")

    return stats


# ---------------------------------------------------------------------------
# Recipe 3: Tree-sequence recording (from recipes.rst)
# ---------------------------------------------------------------------------

def simulate_with_tree_recording(N, L, mu, r, T):
    """Forward simulation with tree-sequence recording.

    Instead of tracking neutral mutations, we record the parent-child
    edges during each reproduction event. At the end, we have a
    complete tree sequence that can be analyzed with tskit.

    Parameters
    ----------
    N : int
        Population size.
    L : int
        Chromosome length.
    mu : float
        Mutation rate (used only for adding mutations AFTER simulation).
    r : float
        Recombination rate.
    T : int
        Number of generations.

    Returns
    -------
    nodes : list of (time, population)
        Tree-sequence nodes (one per haplosome per generation).
    edges : list of (left, right, parent_node, child_node)
        Tree-sequence edges (genealogical relationships).
    sample_nodes : list of int
        Node IDs for the final generation's haplosomes.
    """
    # Node table: each haplosome in each generation gets a node
    # node_id -> (time, population)
    nodes = []
    edges = []

    # Current generation's node IDs
    # Each individual has two haplosomes -> two node IDs
    current_nodes = []
    for i in range(N):
        # Two nodes per individual (one per haplosome)
        node_id_1 = len(nodes)
        nodes.append((T, 0))  # time T (oldest generation)
        node_id_2 = len(nodes)
        nodes.append((T, 0))
        current_nodes.append((node_id_1, node_id_2))

    # No fitness tracking here -- purely neutral for simplicity
    for tick in range(T):
        generation_time = T - tick - 1  # counting down

        new_nodes = []
        for _ in range(N):
            # Draw parents uniformly (neutral)
            p1_idx = np.random.randint(N)
            p2_idx = np.random.randint(N)

            # Create child nodes
            child_node_1 = len(nodes)
            nodes.append((generation_time, 0))
            child_node_2 = len(nodes)
            nodes.append((generation_time, 0))

            # Parent 1 contributes child haplosome 1 (with recombination)
            parent1_hap_a, parent1_hap_b = current_nodes[p1_idx]
            n_breaks = np.random.poisson(r * L)
            breakpoints = sorted(np.random.randint(1, L, size=n_breaks)) if n_breaks > 0 else []

            # Record edges for child_node_1 <- parent1
            # Start with a random haplosome
            if np.random.random() < 0.5:
                sources = [parent1_hap_a, parent1_hap_b]
            else:
                sources = [parent1_hap_b, parent1_hap_a]

            # Create edges at breakpoint boundaries
            all_breaks = [0] + breakpoints + [L]
            for i in range(len(all_breaks) - 1):
                left = all_breaks[i]
                right = all_breaks[i + 1]
                parent_node = sources[i % 2]
                edges.append((left, right, parent_node, child_node_1))

            # Parent 2 contributes child haplosome 2 (same process)
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

    # The sample nodes are the final generation's haplosomes
    sample_nodes = []
    for hap_a, hap_b in current_nodes:
        sample_nodes.extend([hap_a, hap_b])

    return nodes, edges, sample_nodes


# ---------------------------------------------------------------------------
# Add mutations to tree sequence (from recipes.rst)
# ---------------------------------------------------------------------------

def add_mutations_to_tree_sequence(nodes, edges, samples, mu, L):
    """Sprinkle mutations onto a recorded tree sequence.

    This is the same process as msprime's Phase 2: for each edge,
    draw Poisson-distributed mutations and place them at random
    positions and times on the branch.

    Parameters
    ----------
    nodes : list of (time, population)
    edges : list of (left, right, parent, child)
    samples : list of int
        Sample node IDs.
    mu : float
        Per-bp, per-generation mutation rate.
    L : int
        Chromosome length.

    Returns
    -------
    mutations : list of (position, node, time)
    """
    mutations = []

    for left, right, parent, child in edges:
        span = right - left
        parent_time = nodes[parent][0]
        child_time = nodes[child][0]
        branch_length = parent_time - child_time

        if branch_length <= 0:
            continue

        # Number of mutations on this edge
        expected = mu * span * branch_length
        n_muts = np.random.poisson(expected)

        for _ in range(n_muts):
            pos = np.random.uniform(left, right)
            time = np.random.uniform(child_time, parent_time)
            mutations.append((pos, child, time))

    mutations.sort(key=lambda m: m[0])
    print(f"Placed {len(mutations)} mutations on the tree sequence")
    return mutations


# ---------------------------------------------------------------------------
# Demo function
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the mini SLiM forward-time simulator."""
    print("=" * 60)
    print("Mini SLiM -- Forward-time Wright-Fisher Simulator")
    print("=" * 60)

    # --- Step 2 demo: Fitness calculation ---
    print("\n--- Fitness Calculation ---")
    ind = Individual()
    ind.haplosome_1 = [Mutation(position=500, s=0.1, h=0.5)]
    print(f"Fitness with one het beneficial (s=0.1, h=0.5): "
          f"{calculate_fitness(ind):.4f}")
    # Expected: 1 + 0.5 * 0.1 = 1.05

    ind2 = Individual()
    m1 = Mutation(position=100, s=-0.05, h=0.5)
    m2 = Mutation(position=100, s=-0.05, h=0.5)  # same position = homozygous
    ind2.haplosome_1 = [m1]
    ind2.haplosome_2 = [m2]
    print(f"Fitness with one hom deleterious (s=-0.05): "
          f"{calculate_fitness(ind2):.4f}")
    # Expected: 1 + (-0.05) = 0.95

    # --- Step 7 demo: Small neutral simulation ---
    print("\n--- Small Neutral Simulation ---")
    np.random.seed(42)
    pop, stats = simulate(
        N=100,        # small population for speed
        L=10000,      # 10 kb chromosome
        mu=1e-4,      # elevated rate (for tractability)
        r=1e-4,       # same as mutation rate
        T=500,        # 500 generations
        dfe='neutral',
        track_every=100
    )

    # --- Tree-sequence recording demo ---
    print("\n--- Tree-Sequence Recording ---")
    np.random.seed(42)
    nodes, edges, samples = simulate_with_tree_recording(
        N=50, L=10000, mu=1e-4, r=1e-4, T=200
    )
    print(f"Done: {len(nodes)} total nodes, {len(edges)} total edges")
    print(f"Sample nodes: {len(samples)} (= 2N = {2*50})")

    muts = add_mutations_to_tree_sequence(nodes, edges, samples,
                                           mu=1e-4, L=10000)

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
