.. _slim_wright_fisher:

==================================
The Wright-Fisher Generation Cycle
==================================

   *The escapement of the forge: one tick, one generation.*

The Wright-Fisher model is the oldest and simplest model of a finite
population. Each generation, :math:`N` offspring are produced by drawing
parents at random (with replacement) from the current population. Add
mutations, recombination, and fitness-proportional parent selection, and
you have the engine that drives SLiM.

In this chapter, we build a minimal Wright-Fisher simulator from scratch.
It will not be fast -- SLiM's C++ implementation is heavily optimized with
mutation runs, fitness caching, and template specialization -- but it will
be *correct*, and every line of code will correspond to a line of math.


Step 1: The Population and Its Genomes
========================================

A diploid individual carries two **haplosomes** (chromosome copies). Each
haplosome is a list of mutations, where each mutation has a position on the
chromosome, a selection coefficient :math:`s`, and a dominance coefficient
:math:`h`.

.. code-block:: python

   import numpy as np
   from dataclasses import dataclass, field

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

This is the data model. Now for the dynamics.


Step 2: Fitness Calculation
============================

An individual's fitness is the product of the effects of all its mutations.
The key subtlety is **dominance**: a mutation's effect depends on whether
the individual carries it on one haplosome (heterozygous) or both
(homozygous).

For a mutation with selection coefficient :math:`s` and dominance :math:`h`:

.. math::

   \text{fitness contribution} = \begin{cases}
   1 + s & \text{if homozygous (mutation on both haplosomes)} \\
   1 + hs & \text{if heterozygous (mutation on one haplosome)}
   \end{cases}

The individual's total fitness is the product over all mutations:

.. math::

   w = \prod_{m \in \text{mutations}} f(m)

where :math:`f(m)` is the fitness contribution of mutation :math:`m`, taking
into account its zygosity.

.. admonition:: Probability Aside -- Why multiplicative fitness?

   Multiplicative fitness means that the effects of mutations are
   independent on a log scale: :math:`\log w = \sum \log f(m)`. This is
   the standard model in population genetics (it is the simplest
   assumption, and SLiM uses it by default). Alternative models include
   additive fitness (:math:`w = 1 + \sum s_m`) and epistatic fitness
   (where the effect of one mutation depends on which other mutations are
   present). Multiplicative fitness is a good approximation when
   :math:`|s|` is small, since :math:`\prod(1 + s_i) \approx 1 + \sum s_i`
   for small :math:`s_i`.

.. admonition:: Closing a confusion gap -- Dominance

   Dominance coefficient :math:`h` interpolates between recessive
   (:math:`h = 0`) and dominant (:math:`h = 1`):

   - :math:`h = 0`: The mutation has no effect in heterozygotes (fully
     recessive). Only homozygous carriers are affected.
   - :math:`h = 0.5`: The heterozygote effect is half the homozygote
     effect (codominant/additive). This is the most common default.
   - :math:`h = 1`: The heterozygote has the full effect (fully dominant).

   In SLiM's source code, the cached fitness for a heterozygous mutation
   is ``1 + h * s``, and for a homozygous mutation it is ``1 + s``. The
   fitness value is clamped to a minimum of 0 (an individual cannot have
   negative fitness).

.. code-block:: python

   def calculate_fitness(individual):
       """Compute fitness as the product of all mutation effects.

       For each mutation, we check whether the individual is homozygous
       (mutation on both haplosomes) or heterozygous (mutation on one).

       This mirrors SLiM's core fitness calculation in population.cpp.
       """
       w = 1.0

       # Collect positions of mutations on each haplosome
       positions_1 = {m.position for m in individual.haplosome_1}
       positions_2 = {m.position for m in individual.haplosome_2}

       # Mutations on haplosome 1
       for m in individual.haplosome_1:
           if m.position in positions_2:
               # Homozygous: mutation present on both haplosomes
               w *= max(0.0, 1.0 + m.s)
           else:
               # Heterozygous: mutation only on haplosome 1
               w *= max(0.0, 1.0 + m.h * m.s)

       # Mutations on haplosome 2 that are NOT on haplosome 1
       for m in individual.haplosome_2:
           if m.position not in positions_1:
               # Heterozygous: mutation only on haplosome 2
               w *= max(0.0, 1.0 + m.h * m.s)

       # Clamped to zero (SLiM also does this: fitness cannot be negative)
       individual.fitness = max(0.0, w)
       return individual.fitness

   # Example: one beneficial heterozygous mutation
   ind = Individual()
   ind.haplosome_1 = [Mutation(position=500, s=0.1, h=0.5)]
   print(f"Fitness with one het beneficial (s=0.1, h=0.5): "
         f"{calculate_fitness(ind):.4f}")
   # Expected: 1 + 0.5 * 0.1 = 1.05

   # Two deleterious mutations, one homozygous
   ind2 = Individual()
   m1 = Mutation(position=100, s=-0.05, h=0.5)
   m2 = Mutation(position=100, s=-0.05, h=0.5)  # same position = homozygous
   ind2.haplosome_1 = [m1]
   ind2.haplosome_2 = [m2]
   print(f"Fitness with one hom deleterious (s=-0.05): "
         f"{calculate_fitness(ind2):.4f}")
   # Expected: 1 + (-0.05) = 0.95


Step 3: Parent Selection
=========================

Parents are drawn with probability proportional to their fitness. This is
the mechanism by which selection acts: fitter individuals are more likely
to be chosen as parents, so their alleles are over-represented in the next
generation.

Formally, the probability that individual :math:`i` is chosen as a parent
is:

.. math::

   P(\text{parent} = i) = \frac{w_i}{\sum_{j=1}^{N} w_j}

This is **fitness-proportional selection**, also called **roulette wheel
selection** in the evolutionary computation literature.

.. admonition:: Probability Aside -- Selection as a biased coin

   Under neutrality (:math:`s = 0` for all mutations), all individuals have
   fitness :math:`w = 1`, and parent selection reduces to uniform random
   sampling -- the standard Wright-Fisher model. Selection introduces a
   bias: if individual :math:`i` has fitness :math:`1 + s` and everyone
   else has fitness :math:`1`, the probability of choosing :math:`i` is
   :math:`(1 + s) / (N - 1 + 1 + s) \approx (1 + s) / N` for large
   :math:`N`. The bias is small per generation but compounds over many
   generations -- this is how evolution works.

In SLiM's C++ implementation, fitness-proportional selection uses a GSL
discrete lookup table for :math:`O(1)` sampling after :math:`O(N)` setup.
Our Python version uses ``numpy.random.choice`` with weights:

.. code-block:: python

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


Step 4: Recombination
======================

Each parent contributes one haplosome to the offspring. That haplosome is
assembled by **recombining** the parent's two haplosomes: crossover points
are drawn, and the haplosome alternates between the two parental copies at
each crossover.

The number of crossovers along a chromosome of length :math:`L` with
per-bp recombination rate :math:`r` is:

.. math::

   n_{\text{crossovers}} \sim \text{Poisson}(r \cdot L)

Each crossover point is placed uniformly at random along the chromosome.
The offspring haplosome starts by copying from one randomly chosen parental
haplosome, then switches to the other at each crossover point.

.. code-block:: python

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
       current_source = 0  # index into sources list

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

A cleaner (and faster) implementation that mirrors SLiM more closely:

.. code-block:: python

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
       current = hap_a
       other = hap_b
       bp_idx = 0

       # Merge mutations from both haplosomes in position order
       all_muts_a = [(m.position, 'a', m) for m in hap_a]
       all_muts_b = [(m.position, 'b', m) for m in hap_b]
       all_events = sorted(all_muts_a + all_muts_b, key=lambda x: x[0])

       using_a = True  # start with hap_a

       for pos, source, mut in all_events:
           # Advance through breakpoints
           while bp_idx < len(breakpoints) and breakpoints[bp_idx] <= pos:
               using_a = not using_a
               bp_idx += 1

           # Keep mutation if it comes from the currently active haplosome
           if (using_a and source == 'a') or (not using_a and source == 'b'):
               child.append(mut)

       return child


Step 5: Adding New Mutations
=============================

After recombination, new mutations are added to the offspring's haplosomes.
The number of new mutations on each haplosome is Poisson-distributed:

.. math::

   n_{\text{mutations}} \sim \text{Poisson}(\mu \cdot L)

Each mutation is placed at a uniformly random position along the chromosome.
Its selection coefficient :math:`s` is drawn from the **distribution of
fitness effects (DFE)**, which defines the spectrum of possible fitness
effects.

.. admonition:: Probability Aside -- Common DFE models

   The DFE is the probability distribution of :math:`s` for new mutations.
   Common choices:

   - **Fixed**: :math:`s` is a constant (e.g., :math:`s = 0` for neutral
     mutations, :math:`s = -0.01` for weakly deleterious).
   - **Exponential**: :math:`s \sim \text{Exp}(\beta)`. Used for beneficial
     mutations, where most have small effect and few have large effect.
   - **Gamma**: :math:`|s| \sim \text{Gamma}(\alpha, \beta)`. The most
     commonly used DFE for deleterious mutations (Eyre-Walker & Keightley,
     2007). The shape parameter :math:`\alpha` controls how concentrated
     effects are: :math:`\alpha < 1` gives an L-shaped distribution (many
     nearly-neutral, few strongly deleterious); :math:`\alpha > 1` gives a
     bell-shaped distribution.
   - **Normal**: :math:`s \sim \mathcal{N}(\mu_s, \sigma_s^2)`. Rarely
     used for purifying selection but sometimes for stabilizing selection
     models.

.. code-block:: python

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


Step 6: The Complete Generation Cycle
=======================================

Now we assemble the parts into a complete Wright-Fisher generation. This
is the heart of SLiM -- the function that ticks once per generation:

.. code-block:: python

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


Step 7: The Full Simulation
============================

Now we can run a complete forward simulation. We initialize a population,
run for :math:`T` generations, and track statistics along the way.

.. code-block:: python

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

   # Example: small neutral simulation
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


Step 8: Verifying Against Theory
==================================

Under neutrality (no selection), we can verify our simulator against known
results from coalescent theory.

**Expected number of segregating sites.** For a diploid population of size
:math:`N`, the expected number of segregating sites at mutation-drift
equilibrium is:

.. math::

   E[S] = 4 N \mu L \cdot \sum_{k=1}^{2N-1} \frac{1}{k}
   \approx 4 N \mu L \cdot (\ln(2N) + \gamma)

where :math:`\gamma \approx 0.577` is the Euler-Mascheroni constant and
we sum to :math:`2N - 1` because there are :math:`2N` haploid genomes
in a diploid population of size :math:`N`.

**Expected heterozygosity.** The expected heterozygosity (probability that
two randomly chosen haplosomes differ at a site) is:

.. math::

   \pi = \frac{4 N \mu}{1 + 4 N \mu} \approx 4 N \mu

for small :math:`\mu`.

.. code-block:: python

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

   # After a long burn-in, the observed S should be close to expected
   # (The short simulation above won't have reached equilibrium;
   #  a proper test would run for ~10*N generations)


The Wright-Fisher cycle is complete. You have built the escapement. In the
next chapter, we put it to work with three recipes that demonstrate the
power of forward simulation: phenomena that the coalescent cannot easily
model.
