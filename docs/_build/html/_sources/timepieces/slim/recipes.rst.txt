.. _slim_recipes:

=========
Recipes
=========

   *Now fire up the forge and watch the metal bend.*

With the Wright-Fisher cycle in hand, we can simulate phenomena that the
coalescent cannot easily model. Each recipe below introduces a new
selective regime, derives what we should expect to see, and then runs the
simulation to verify.


Recipe 1: The Selective Sweep
==============================

A **selective sweep** occurs when a new beneficial mutation rises to fixation,
dragging nearby neutral variation along with it. This is one of the most
important signatures of natural selection, and it is trivial to simulate
in a forward framework.

The Math
---------

Consider a single beneficial mutation with selection coefficient :math:`s > 0`
arising in a population of size :math:`N`. Under the Wright-Fisher model with
fitness-proportional selection:

1. **Fixation probability.** The probability that a new mutation with
   selective advantage :math:`s` fixes (reaches frequency 1) is approximately:

   .. math::

      P_{\text{fix}} \approx \frac{2s}{1 - e^{-4Ns}}

   For :math:`4Ns \gg 1`, this simplifies to :math:`P_{\text{fix}} \approx 2s`.
   For a neutral mutation (:math:`s = 0`), the fixation probability is
   :math:`1/(2N)`.

   .. admonition:: Calculus Aside -- Diffusion approximation for fixation probability

      This formula comes from the diffusion approximation to the
      Wright-Fisher model (Kimura, 1962). The allele frequency
      :math:`p` follows a stochastic differential equation:

      .. math::

         dp = sp(1-p)\,dt + \sqrt{\frac{p(1-p)}{2N}}\,dW

      where the first term is deterministic selection and the second is
      random genetic drift. Solving for the probability of reaching
      :math:`p = 1` starting from :math:`p = 1/(2N)` gives the formula
      above. The key insight: for :math:`s > 0`, the fixation probability
      scales with :math:`s`, not :math:`1/(2N)` -- selection overpowers
      drift.

2. **Time to fixation.** Conditional on fixation, the expected time for
   the sweep is approximately:

   .. math::

      T_{\text{fix}} \approx \frac{2}{s} \ln(4Ns)

   This is much faster than the neutral fixation time (:math:`\approx 4N`
   generations). A mutation with :math:`s = 0.01` in a population of
   :math:`N = 10{,}000` takes about :math:`\sim 2{,}000` generations to
   fix, versus :math:`\sim 40{,}000` for a neutral mutation.

3. **The hitchhiking effect.** As the beneficial mutation sweeps to fixation,
   it drags along all neutral variants that happen to be on the same
   haplosome (linked to it). This reduces neutral diversity near the
   selected site -- a **selective sweep signature**.

   The expected reduction in heterozygosity at a neutral site at
   recombinational distance :math:`\rho = 4Nr` from the selected site is:

   .. math::

      \frac{\pi_{\text{after}}}{\pi_{\text{before}}}
      \approx 1 - \frac{2s}{2s + r}

   Close to the selected site (:math:`r \ll s`), diversity is nearly
   eliminated. Far away (:math:`r \gg s`), the effect vanishes.

The Code
---------

.. code-block:: python

   import numpy as np

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

   # Example: selective sweep with s=0.05 in a small population
   # (Small N and elevated mu/r for tractability)
   np.random.seed(123)
   traj, fixed = simulate_sweep(
       N=200,
       L=5000,
       mu=1e-4,
       r=1e-4,
       s_beneficial=0.05,
       position_selected=2500,  # middle of chromosome
       T_burnin=500,
       T_after=1000
   )

Let us verify the fixation probability:

.. code-block:: python

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

.. admonition:: What a sweep looks like in the genealogy

   After a sweep, the genealogy near the selected site looks **star-like**:
   all lineages coalesce rapidly to the single individual that carried the
   sweeping allele. Moving away from the selected site (in recombination
   distance), the genealogy gradually returns to the normal coalescent
   shape. This star-like topology creates several observable signatures:

   - **Reduced diversity**: :math:`\pi` drops near the selected site.
   - **Excess rare variants**: The SFS shifts toward singletons (because
     all new mutations arose after the recent coalescence).
   - **Extended haplotype homozygosity**: Long tracts of identical
     sequence around the selected site (because recombination has not
     yet had time to break up the sweeping haplotype).


Recipe 2: Background Selection
================================

**Background selection** (BGS) is the flip side of the sweep: purifying
selection against deleterious mutations reduces the effective population
size at linked neutral sites. Unlike a sweep (one dramatic event), BGS is
a continuous process arising from the steady rain of deleterious mutations.

The Math
---------

Consider a region of the genome where deleterious mutations arise at rate
:math:`U` per generation (the total deleterious mutation rate across the
region), each with selection coefficient :math:`-s` (and :math:`s > 0`).

The key result (Charlesworth, Morgan, and Charlesworth, 1993) is that the
effective population size at a linked neutral site is reduced to:

.. math::

   N_e \approx N \cdot \exp\left(-\frac{U}{s + r_{\text{total}}}\right)

where :math:`r_{\text{total}}` is the total recombination rate between the
neutral site and the selected region. When :math:`U / s` is large and
recombination is limited, BGS can dramatically reduce :math:`N_e` and
therefore neutral diversity.

.. admonition:: Probability Aside -- Why BGS reduces :math:`N_e`

   Deleterious mutations constantly arise and are removed by selection. An
   individual carrying a deleterious mutation has lower fitness and is less
   likely to contribute to the next generation. Neutral alleles linked to
   the deleterious mutation are "dragged down" along with it. This is
   analogous to a reverse sweep: instead of one allele rising in frequency,
   many alleles are constantly being removed. The net effect is that the
   neutral genealogy looks like it came from a smaller population.

   More precisely: in a diploid population, an individual free of
   deleterious mutations has relative fitness 1, while individuals carrying
   :math:`k` deleterious mutations have fitness :math:`(1 - s)^k`. If
   deleterious mutations are at mutation-selection balance, the fraction of
   the population that is mutation-free is :math:`e^{-U/s}` (Poisson
   approximation), and the effective population is roughly
   :math:`N \cdot e^{-U/s}`.

The Code
---------

.. code-block:: python

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

   # Compare neutral diversity WITH and WITHOUT background selection
   np.random.seed(42)

   # Without BGS (neutral only)
   print("=== No selection (neutral) ===")
   pop_neutral = [Individual() for _ in range(100)]
   for tick in range(500):
       pop_neutral = wright_fisher_generation(
           pop_neutral, 100, 5000, 1e-4, 1e-4, tick, dfe='neutral')

   # With BGS
   print("\n=== With background selection (s=-0.05) ===")
   stats_bgs = simulate_bgs(
       N=100, L=5000,
       mu_neutral=1e-4,
       mu_deleterious=5e-4,  # 5x higher than neutral
       s_deleterious=-0.05,
       r=1e-4,
       T=500,
       track_interval=100
   )

.. admonition:: What BGS looks like in the data

   Background selection produces a characteristic pattern:

   - **Reduced diversity genome-wide**, but especially in regions of low
     recombination (where linked deleterious mutations cannot be separated
     from neutral variants).
   - **Normal SFS shape**: Unlike a selective sweep, BGS does not strongly
     distort the shape of the SFS. It reduces diversity (as if :math:`N_e`
     were smaller) but the :math:`1/i` pattern is preserved.
   - **Correlation between diversity and recombination rate**: Regions with
     higher recombination have higher diversity, because they are less
     affected by linked selection. This is one of the strongest empirical
     signals of BGS in real genomes.


Recipe 3: Tree-Sequence Recording
===================================

SLiM's most powerful feature for large-scale simulations is **tree-sequence
recording**: instead of storing every mutation in every individual, SLiM
records the genealogical relationships (edges in the tree sequence) as
they are created during reproduction. This produces a compact, lossless
record of the complete ancestry.

The Math
---------

The tree-sequence recording idea (Kelleher, Thornton, Ashander, and Ralph,
2018) is simple: during each reproduction event, record an **edge** from
child to parent for each genomic interval inherited. The collection of all
edges over all generations defines the complete genealogy of the final
population.

Each edge is a tuple :math:`(l, r, p, c)`:

- :math:`l, r`: the genomic interval :math:`[l, r)` inherited
- :math:`p`: the parent node (one haplosome of the parent)
- :math:`c`: the child node (one haplosome of the child)

Recombination within a parent creates **multiple edges** from the same
child to different parent haplosomes, covering different genomic intervals.

.. admonition:: Why record trees?

   Without tree-sequence recording, a forward simulation must store every
   mutation in every individual -- a huge memory burden for large
   populations and long genomes. Many of these mutations are neutral and
   are only needed for computing summary statistics *after* the simulation.

   With tree-sequence recording, neutral mutations are not tracked during
   the simulation at all. Instead, the genealogy is recorded, and mutations
   are sprinkled onto the genealogy *after* the simulation (exactly as
   msprime does -- see :ref:`msprime_mutations`). This can reduce memory
   use by orders of magnitude.

   The genealogy also enables efficient computation of many statistics
   (diversity, divergence, :math:`F_{ST}`) without even needing mutations,
   using branch-length statistics.

The Code
---------

.. code-block:: python

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

           if tick % 100 == 0:
               print(f"Gen {tick:>5d}: {len(nodes)} nodes, {len(edges)} edges")

       # The sample nodes are the final generation's haplosomes
       sample_nodes = []
       for hap_a, hap_b in current_nodes:
           sample_nodes.extend([hap_a, hap_b])

       print(f"\nDone: {len(nodes)} total nodes, {len(edges)} total edges")
       print(f"Sample nodes: {len(sample_nodes)} (= 2N = {2*N})")

       return nodes, edges, sample_nodes

   # Run a small example
   np.random.seed(42)
   nodes, edges, samples = simulate_with_tree_recording(
       N=50, L=10000, mu=1e-4, r=1e-4, T=200
   )

After the simulation, mutations can be placed on the recorded tree sequence
exactly as in msprime:

.. code-block:: python

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

   muts = add_mutations_to_tree_sequence(nodes, edges, samples, mu=1e-4, L=10000)

.. admonition:: The tree-sequence recording tradeoff

   **Pros:**

   - Dramatically reduces memory for neutral mutations.
   - Enables post-hoc mutation placement (try different :math:`\mu` without
     re-running the simulation).
   - Enables efficient branch-length statistics.
   - The tree sequence can be **simplified** to remove extinct lineages,
     further reducing size.

   **Cons:**

   - Recording edges has a cost: each reproduction event creates 1-3 edges
     per haplosome (depending on recombination).
   - Long simulations accumulate many edges; periodic **simplification** is
     needed to keep size manageable.
   - Selected mutations must still be tracked during the simulation (they
     affect fitness and cannot be deferred to post-hoc).


Exercises
=========

.. admonition:: Exercise 1: Fixation probability

   Run 500 independent selective sweeps with :math:`N = 200` and
   :math:`s = 0.01, 0.02, 0.05, 0.1`. For each :math:`s`, estimate
   the fixation probability and compare to the theoretical prediction
   :math:`P_{\text{fix}} \approx 2s / (1 - e^{-4Ns})`.

.. admonition:: Exercise 2: Diversity reduction around a sweep

   After a completed sweep, measure neutral diversity (:math:`\pi`) in
   windows around the selected site. Verify that diversity is reduced
   near the selected site and recovers with distance, following the
   prediction :math:`\pi/\pi_0 \approx 1 - 2s/(2s + r \cdot d)` where
   :math:`d` is the distance from the selected site.

.. admonition:: Exercise 3: Background selection and :math:`N_e`

   Simulate 1000 generations with :math:`N = 500`, neutral mutation rate
   :math:`\mu = 10^{-4}`, and varying deleterious mutation rates
   :math:`U = 0.01, 0.05, 0.1, 0.5`. Measure neutral diversity at
   equilibrium and estimate the effective population size as
   :math:`\hat{N}_e = \pi / (4\mu)`. Compare to the BGS prediction
   :math:`N_e \approx N \cdot e^{-U/s}`.

.. admonition:: Exercise 4: Tree-sequence simplification

   After running a tree-sequence recording simulation for :math:`T = 500`
   generations with :math:`N = 100`, count the number of edges. Then
   implement a simplification step that removes all nodes and edges not
   ancestral to the final sample. How much smaller is the simplified tree
   sequence?

----

You have now built a forward-time population genetics simulator from
scratch. The three recipes demonstrate phenomena that are central to
population genetics but impossible (or at best very awkward) to model
with backward-time coalescent simulation:

- **Selective sweeps** -- A beneficial allele rising to fixation,
  dragging linked variation along.
- **Background selection** -- The steady erosion of neutral diversity
  by linked deleterious mutations.
- **Tree-sequence recording** -- A compact genealogical record that
  bridges forward and backward time.

The real SLiM is vastly more powerful than our toy implementation: it
supports non-Wright-Fisher life cycles, spatial structure, continuous
space, gene drive, quantitative genetics, and much more. But the
*mechanism* is the same. Every generation, the same cycle runs: fitness,
selection, recombination, mutation. The forge is hot, and now you know how
to work it.

*The metal bends. And you know exactly why.*
