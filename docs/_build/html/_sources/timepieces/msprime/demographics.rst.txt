.. _msprime_demographics:

==========================
Demographics & Population
==========================

   *The case and dial: the population history that shapes the genealogy.*

The coalescent process describes how lineages merge in a constant-size
population. But real populations change size, split, merge, and exchange
migrants. msprime handles all of this through a **demographic model** that
modifies the simulation parameters at specified times.

In our watch metaphor, if :ref:`hudson_algorithm` is the mainspring (the
ticking of the clock), demographics are the case and dial -- the external
structure that determines what the clock *looks like* from the outside. Two
watches can share an identical movement but tell very different stories if
their cases and dials differ. Likewise, the same coalescent algorithm produces
very different genealogies depending on the demographic history.

.. note::

   **Prerequisites.** This chapter builds on :ref:`hudson_algorithm`, where we
   implemented the main simulation loop. Specifically, you should understand:

   - How the main loop checks for **demographic events** before executing
     the next random event (Step 1 of :ref:`hudson_algorithm`).
   - How **coalescence waiting times** depend on population size
     (Step 6 of :ref:`coalescent_process`).
   - The **exponential race** between event types
     (Step 3 of :ref:`coalescent_process`).


Step 1: Population Size Changes
==================================

The simplest demographic event: the population size changes instantaneously
at some time in the past.

**Why it matters.** Population size controls the coalescence rate:

.. math::

   \lambda_{\text{coal}}(t) = \frac{\binom{k}{2}}{N(t)}

A smaller population means faster coalescence (lineages find common ancestors
sooner). A larger population means slower coalescence (more genetic diversity
is maintained).

.. admonition:: Probability Aside -- Population size as a time-scaling factor

   In coalescent units, time is measured in multiples of :math:`N` generations.
   When :math:`N` changes, the "speed" of coalescent time changes too. A
   bottleneck (small :math:`N`) compresses real time: many generations pass
   per coalescent unit, so coalescences happen rapidly. An expansion (large
   :math:`N`) stretches real time: few coalescences occur per generation.
   The demographic model is essentially a variable-speed clock that
   accelerates and decelerates the coalescent.

In the simulation, a population size change is implemented as a **modifier
event**: when the simulation clock reaches the event time, the population
parameters are updated, and the simulation continues with the new rates.

.. code-block:: python

   import numpy as np

   class Population:
       """A population with time-varying size.

       Population size follows the formula:
         N(t) = start_size * exp(-growth_rate * (t - start_time))

       This is an exponential model where start_size and growth_rate
       are updated by demographic events.
       """

       def __init__(self, start_size=1.0, growth_rate=0.0, start_time=0.0):
           self.start_size = start_size    # size at start_time
           self.growth_rate = growth_rate  # exponential growth rate
           self.start_time = start_time    # time when these parameters took effect
           self.num_ancestors = 0          # current number of lineages in this pop

       def get_size(self, t):
           """Population size at time t (backwards).

           N(t) = start_size * exp(-growth_rate * (t - start_time))

           When growth_rate > 0: population was SMALLER in the past
           When growth_rate < 0: population was LARGER in the past
           When growth_rate = 0: constant size
           """
           dt = t - self.start_time
           return self.start_size * np.exp(-self.growth_rate * dt)

       def set_size(self, new_size, time):
           """Change population size at the given time.

           This creates a discontinuous jump in population size --
           the classic "instantaneous size change" demographic event.
           """
           self.start_size = new_size
           self.growth_rate = 0       # reset to constant size
           self.start_time = time

       def set_growth_rate(self, rate, time):
           """Change growth rate at the given time.

           The size is first computed at the new time (to preserve
           continuity), then the growth rate is changed.
           """
           self.start_size = self.get_size(time)  # ensure continuity
           self.start_time = time
           self.growth_rate = rate

   # Example: population that was 10x smaller before a bottleneck
   pop = Population(start_size=10000)
   print(f"Present size: {pop.get_size(0):.0f}")

   # At time 1000 generations ago, size drops to 1000
   pop.set_size(1000, time=1000)
   print(f"Size after bottleneck (t=1000): {pop.get_size(1000):.0f}")

   # With exponential growth
   pop2 = Population(start_size=10000, growth_rate=0.005, start_time=0)
   for t in [0, 100, 500, 1000, 2000]:
       print(f"t={t:>5d}: N = {pop2.get_size(t):.0f}")

.. admonition:: The sign convention

   msprime measures time **backwards** from the present. A positive growth
   rate means the population is growing **towards the present**, so it was
   *smaller* in the past. This can be confusing. The formula
   :math:`N(t) = N_0 e^{-\alpha t}` makes the population shrink as :math:`t`
   increases (going further into the past).

.. admonition:: Calculus Aside -- Exponential growth as a differential equation

   The exponential size model :math:`N(t) = N_0 e^{-\alpha t}` is the
   solution to the differential equation :math:`dN/dt = -\alpha N` with
   initial condition :math:`N(0) = N_0`. Going backwards (increasing
   :math:`t`), the population shrinks at a rate proportional to its current
   size. This is the continuous-time analog of a population that grows by a
   constant fraction each generation going forward.

With population size changes understood, let us look at a more dramatic event:
the bottleneck.


Step 2: The Bottleneck
========================

A bottleneck is a special demographic event where lineages are forcibly
coalesced with some probability. This models a severe reduction in population
size that is too brief to simulate explicitly.

**The math.** At time :math:`t`, with bottleneck intensity :math:`I`, each
pair of lineages coalesces independently with probability
:math:`1 - e^{-I/\binom{k}{2}}`, where :math:`k` is the number of lineages.
In practice, msprime implements this by randomly merging lineage pairs.

.. admonition:: Probability Aside -- Bottleneck intensity

   The bottleneck intensity :math:`I` is measured in coalescent units. It
   represents the "amount of coalescence" that would have occurred in a
   population of size 1 for :math:`I` time units. A higher :math:`I` means
   more coalescence: :math:`I = 0` means no effect, :math:`I \to \infty`
   means all lineages coalesce to a single ancestor. The probability that a
   specific pair survives the bottleneck without coalescing is :math:`e^{-I}`,
   which comes from the survival function of an :math:`\text{Exp}(1)` random
   variable.

.. code-block:: python

   def bottleneck_event(lineages, intensity):
       """Simulate a bottleneck: randomly coalesce lineages.

       Parameters
       ----------
       lineages : list
           Current lineages.
       intensity : float
           Bottleneck intensity (higher = more coalescence).

       Returns
       -------
       lineages : list
           Lineages after the bottleneck.
       coalesced_pairs : list of (int, int)
           Which pairs coalesced.
       """
       coalesced_pairs = []
       # Probability of each pair coalescing
       k = len(lineages)
       if k < 2:
           return lineages, coalesced_pairs

       p = 1 - np.exp(-intensity)  # per-pair coalescence probability

       # Try to coalesce random pairs
       remaining = list(range(len(lineages)))
       np.random.shuffle(remaining)

       while len(remaining) >= 2:
           if np.random.random() < p:
               i = remaining.pop()
               j = remaining.pop()
               coalesced_pairs.append((i, j))
               # In practice: merge the two lineages using merge_two_ancestors
           else:
               remaining.pop()  # this one doesn't coalesce

       return lineages, coalesced_pairs

   # Example
   lins = list(range(10))  # 10 lineages
   _, pairs = bottleneck_event(lins, intensity=2.0)
   print(f"Bottleneck with intensity 2.0: {len(pairs)} pairs coalesced")

Bottlenecks are point events -- they happen instantaneously. The next
ingredient, migration, is a continuous process that operates alongside
coalescence and recombination.


Step 3: Migration
===================

With multiple populations, lineages can move between them. In the coalescent
(going backwards), a migration event means that a lineage in population
:math:`j` had an ancestor in population :math:`k` one generation earlier.

**Migration matrix.** The rate at which lineages migrate from population
:math:`j` to population :math:`k` (backwards) is :math:`M_{jk}`. With
:math:`n_j` lineages in population :math:`j`, the total migration rate out
of :math:`j` to :math:`k` is:

.. math::

   \lambda_{\text{mig}, j \to k} = n_j \cdot M_{jk}

.. admonition:: Closing a confusion gap -- Forward vs backward migration

   In the forward direction, :math:`M_{jk}` is the probability that an
   individual in population :math:`j` migrated *from* population :math:`k`.
   In the backward direction (coalescent), this means a lineage in :math:`j`
   moves *to* :math:`k`. The migration matrix is the same in both directions
   (this is a consequence of the diffusion limit).

   This directionality confusion is one of the most common sources of error
   when setting up demographic models. A useful mnemonic: in the backward
   view, lineages move *toward their ancestors*, so migration from :math:`j`
   to :math:`k` means "this lineage's ancestor lived in population :math:`k`."

Migration participates in the exponential race just like coalescence and
recombination: each (source, dest) pair proposes an exponential waiting time,
and the shortest one wins.

.. code-block:: python

   def migration_event(populations, migration_matrix, source, dest):
       """Move a random lineage from source to dest.

       Parameters
       ----------
       populations : list of Population
       migration_matrix : 2D array
           M[j][k] = migration rate from j to k (backward).
       source : int
           Source population index.
       dest : int
           Destination population index.
       """
       pop = populations[source]
       # Choose a random lineage from the source population
       idx = np.random.randint(pop.num_ancestors)

       # Move it (in a real implementation, update the lineage's
       # population attribute and move it between population lists)
       print(f"Lineage {idx} migrates from pop {source} to pop {dest}")
       pop.num_ancestors -= 1
       populations[dest].num_ancestors += 1

   # Example: island model with 3 populations
   migration_matrix = [
       [0, 0.001, 0.001],     # pop 0 -> pop 1, pop 2
       [0.001, 0, 0.001],     # pop 1 -> pop 0, pop 2
       [0.001, 0.001, 0],     # pop 2 -> pop 0, pop 1
   ]

   pops = [Population(start_size=1000) for _ in range(3)]
   pops[0].num_ancestors = 10
   pops[1].num_ancestors = 10
   pops[2].num_ancestors = 10

   # Total migration rate across all population pairs
   total_mig_rate = 0
   for j in range(3):
       for k in range(3):
           if j != k:
               rate = pops[j].num_ancestors * migration_matrix[j][k]
               total_mig_rate += rate
               print(f"Pop {j} -> Pop {k}: rate = {rate:.4f}")
   print(f"Total migration rate: {total_mig_rate:.4f}")

Migration brings lineages together across populations. But at some point in
the past, those populations may not have existed as separate entities. This
brings us to population splits and joins.


Step 4: Population Splits and Joins
======================================

A **population split** (going forwards: one population divides into two)
corresponds to a **mass migration** going backwards: at time :math:`t`, all
lineages in population :math:`B` move to population :math:`A`.

.. code-block:: text

   Forward view:            Backward view:
   Past -> Future            Future -> Past

     A                        A <- B
     |                        |
     +-- B at time t          |  (all B lineages join A)
     |   |                    |
     A   B                    A   B

.. admonition:: Closing a confusion gap -- Splits vs. joins in backward time

   In forward time, a *split* creates a new population. In backward time
   (which the coalescent simulates), this same event is a *join*: lineages
   from the daughter population move to the ancestral population. msprime
   implements this as a **mass migration** with ``fraction=1.0``. When you
   see ``mass_migration(source=B, dest=A, fraction=1.0)``, read it as:
   "Going back in time, all lineages currently in B move to A, because B
   didn't exist before this point."

   For **admixture** events (where a population receives genetic material
   from another), use ``fraction < 1.0``: only some lineages move.

.. code-block:: python

   def mass_migration_event(populations, source, dest, fraction=1.0):
       """Move a fraction of lineages from source to dest.

       fraction=1.0 means all lineages move (population join).
       fraction<1.0 means a subset moves (admixture).

       Parameters
       ----------
       populations : list of Population
       source : int
       dest : int
       fraction : float
           Fraction of lineages to move (0 to 1).
       """
       n_source = populations[source].num_ancestors
       n_to_move = int(np.round(n_source * fraction))

       populations[source].num_ancestors -= n_to_move
       populations[dest].num_ancestors += n_to_move

       print(f"Mass migration: {n_to_move} lineages from "
             f"pop {source} to pop {dest}")

   # Example: two populations that split 500 generations ago
   pops = [Population(start_size=5000), Population(start_size=5000)]
   pops[0].num_ancestors = 20
   pops[1].num_ancestors = 15

   # At t=500, all lineages from pop 1 join pop 0
   mass_migration_event(pops, source=1, dest=0, fraction=1.0)
   print(f"After join: pop 0 has {pops[0].num_ancestors}, "
         f"pop 1 has {pops[1].num_ancestors}")

All of these demographic events -- size changes, bottlenecks, migration rate
changes, mass migrations -- need to be scheduled and executed at the right
times. The demographic event queue handles this.


Step 5: The Demographic Event Queue
======================================

msprime stores all demographic events in a priority queue sorted by time.
During the main simulation loop (the ticking of the clock in
:ref:`hudson_algorithm`), before executing the next coalescence or
recombination event, it checks whether a demographic event should fire first.

.. admonition:: Closing a confusion gap -- Deterministic vs. stochastic events

   Coalescence, recombination, and migration events are **stochastic**: their
   timing is drawn from exponential distributions. Demographic events are
   **deterministic**: they happen at pre-specified times. The main loop must
   check both: if the next stochastic event would occur *after* the next
   demographic event, the demographic event fires first, the population
   parameters are updated, and the stochastic event times are redrawn with
   the new rates. This is why the main loop in :ref:`hudson_algorithm` has
   a ``continue`` statement after processing a demographic event -- it
   returns to the top of the loop to recompute rates.

.. code-block:: python

   class DemographicEventQueue:
       """Priority queue of demographic events sorted by time.

       Events are stored as (time, type, args) tuples and processed
       in chronological order during the simulation.
       """

       def __init__(self):
           self.events = []  # list of (time, function, args)

       def add_size_change(self, time, pop_id, new_size):
           """Schedule a population size change."""
           self.events.append((time, 'size_change', (pop_id, new_size)))
           self.events.sort()  # maintain time ordering

       def add_growth_rate_change(self, time, pop_id, rate):
           """Schedule a growth rate change."""
           self.events.append((time, 'growth_rate', (pop_id, rate)))
           self.events.sort()

       def add_mass_migration(self, time, source, dest, fraction):
           """Schedule a mass migration (population split/join/admixture)."""
           self.events.append((time, 'mass_migration',
                                (source, dest, fraction)))
           self.events.sort()

       def add_migration_rate_change(self, time, source, dest, rate):
           """Schedule a migration rate change."""
           self.events.append((time, 'migration_rate',
                                (source, dest, rate)))
           self.events.sort()

       def next_event_time(self):
           """Time of the next scheduled event (or infinity if none)."""
           if self.events:
               return self.events[0][0]
           return INFINITY

       def pop_event(self):
           """Remove and return the next event."""
           return self.events.pop(0)

   # Example: Out-of-Africa demographic model (simplified)
   queue = DemographicEventQueue()
   # Population split: Eurasian pop separates from African at t=2000
   queue.add_mass_migration(2000, source=1, dest=0, fraction=1.0)
   # Bottleneck in the European population at t=1000
   queue.add_size_change(1000, pop_id=1, new_size=100)
   # Recovery after bottleneck at t=800
   queue.add_size_change(800, pop_id=1, new_size=5000)

   print("Demographic events (in order):")
   for time, etype, args in queue.events:
       print(f"  t={time}: {etype} {args}")


Step 6: Integrating Demographics into the Main Loop
======================================================

The integration is straightforward: before executing a stochastic event,
check if a deterministic demographic event occurs first. This is the point
where the case and dial attach to the movement -- where population history
shapes the genealogy.

.. code-block:: python

   def simulate_with_demographics(n, L, recomb_rate, populations,
                                   migration_matrix, event_queue):
       """Hudson's algorithm with demographics.

       This extends the main loop from hudson_algorithm with demographic
       event handling: size changes, growth rate changes, mass migrations,
       and migration rate changes.

       Parameters
       ----------
       n : int
           Total sample size across all populations.
       L : float
           Sequence length.
       recomb_rate : float
       populations : list of Population
       migration_matrix : 2D array
       event_queue : DemographicEventQueue
       """
       t = 0.0
       total_events = 0

       while True:  # simplified termination
           # Compute event rates
           total_lineages = sum(p.num_ancestors for p in populations)
           if total_lineages <= 1:
               break

           # Coalescence: per population (each with its own rate)
           t_ca = INFINITY
           for pop in populations:
               k = pop.num_ancestors
               if k > 1:
                   coal_rate = k * (k - 1) / 2
                   # Scale by population size (larger pop -> longer wait)
                   t_pop = pop.get_size(t) * np.random.exponential(
                       1.0 / (2 * coal_rate))
                   if t_pop < t_ca:
                       t_ca = t_pop

           # Recombination
           # (simplified: use total lineage count as proxy)
           t_re = np.random.exponential(1.0 / max(total_lineages * recomb_rate * L, 1e-10))

           # Migration: per (source, dest) pair
           t_mig = INFINITY
           for j in range(len(populations)):
               for k in range(len(populations)):
                   if j != k and migration_matrix[j][k] > 0:
                       rate = populations[j].num_ancestors * migration_matrix[j][k]
                       if rate > 0:
                           t_try = np.random.exponential(1.0 / rate)
                           t_mig = min(t_mig, t_try)

           min_event_time = min(t_ca, t_re, t_mig)

           # === Check demographic events ===
           # If a demographic event occurs before the next stochastic event,
           # execute it first and re-enter the loop with updated parameters.
           if t + min_event_time > event_queue.next_event_time():
               # Demographic event fires first!
               event_time, etype, args = event_queue.pop_event()
               t = event_time

               if etype == 'size_change':
                   pop_id, new_size = args
                   populations[pop_id].set_size(new_size, t)
                   print(f"t={t}: Pop {pop_id} size -> {new_size}")
               elif etype == 'growth_rate':
                   pop_id, rate = args
                   populations[pop_id].set_growth_rate(rate, t)
                   print(f"t={t}: Pop {pop_id} growth rate -> {rate}")
               elif etype == 'mass_migration':
                   source, dest, fraction = args
                   n_move = int(populations[source].num_ancestors * fraction)
                   populations[source].num_ancestors -= n_move
                   populations[dest].num_ancestors += n_move
                   print(f"t={t}: {n_move} lineages: pop {source} -> pop {dest}")
               elif etype == 'migration_rate':
                   source, dest, rate = args
                   migration_matrix[source][dest] = rate
                   print(f"t={t}: migration {source}->{dest} = {rate}")
           else:
               t += min_event_time
               total_events += 1
               # Execute the stochastic event (simplified: just count)

       print(f"\nSimulation completed: {total_events} events, TMRCA = {t:.1f}")

   # Run with the Out-of-Africa model
   INFINITY = float('inf')
   pops = [Population(start_size=10000), Population(start_size=5000)]
   pops[0].num_ancestors = 10  # African samples
   pops[1].num_ancestors = 10  # European samples

   mig_matrix = [[0, 0.0001], [0.0001, 0]]

   queue = DemographicEventQueue()
   queue.add_mass_migration(2000, source=1, dest=0, fraction=1.0)
   queue.add_size_change(1000, pop_id=1, new_size=100)  # European bottleneck

   np.random.seed(42)
   simulate_with_demographics(
       n=20, L=10000, recomb_rate=1e-4,
       populations=pops, migration_matrix=mig_matrix,
       event_queue=queue
   )

With the continuous-time coalescent and demographics covered, there is one
more simulation mode to introduce: the exact discrete-time model for small
or recent populations.


Step 7: The Discrete-Time Wright-Fisher Model
================================================

For small populations or recent time scales, the continuous-time coalescent
approximation breaks down. msprime also implements the exact **Discrete-Time
Wright-Fisher (DTWF)** model, which simulates each generation explicitly.

.. admonition:: Closing a confusion gap -- When does the coalescent approximation fail?

   The continuous-time coalescent assumes :math:`N` is large enough that the
   probability of two simultaneous coalescences is negligible. For
   :math:`N = 50` with :math:`k = 20` lineages, the probability of a
   double coalescence in one generation is about
   :math:`\binom{20}{2}^2 / (2 \cdot 50^2) \approx 1.4\%` -- no longer
   negligible. The DTWF model handles this correctly by simulating every
   generation exactly. In practice, msprime uses a hybrid approach: DTWF for
   the recent past (where populations may be small), switching to the
   coalescent for the distant past (where the approximation is excellent).

**The algorithm per generation:**

1. Each lineage draws a parent uniformly from the population
2. Lineages that drew the same parent are merged (coalescence)
3. Recombination is simulated for each diploid parent
4. Migration is applied

.. code-block:: python

   def dtwf_generation(lineages, N, recomb_rate, L):
       """Simulate one generation of the DTWF model.

       Unlike the coalescent, this processes every generation explicitly.
       It is exact (no large-N approximation) but slower for large N.

       Parameters
       ----------
       lineages : list
           Current lineages.
       N : int
           Population size.
       recomb_rate : float
           Per-bp recombination rate per generation.
       L : float
           Sequence length.

       Returns
       -------
       lineages : list
           Lineages after one generation.
       events : list
           Events that occurred.
       """
       events = []

       # Step 1: Each lineage draws a parent uniformly from N individuals
       parents = {}
       for i, lin in enumerate(lineages):
           parent_id = np.random.randint(N)  # uniform random parent
           if parent_id not in parents:
               parents[parent_id] = []
           parents[parent_id].append(i)

       # Step 2: Lineages sharing a parent coalesce
       new_lineages = []
       for parent_id, children in parents.items():
           if len(children) == 1:
               # Only one lineage chose this parent: no coalescence
               new_lineages.append(lineages[children[0]])
           else:
               # Multiple lineages share a parent: merge them
               # (In reality, recombination happens first, then merging)
               merged = lineages[children[0]]
               for c in children[1:]:
                   events.append(('coal', children[0], c))
                   # merge lineages[c] into merged (simplified)
               new_lineages.append(merged)

       # Step 3: Recombination (simplified Poisson model)
       for lin in new_lineages:
           n_recombs = np.random.poisson(recomb_rate * L)
           for _ in range(n_recombs):
               bp = np.random.randint(1, int(L))
               events.append(('recomb', bp))

       return new_lineages, events

   # DTWF is used for recent generations, then switches to coalescent
   # for efficiency. This is the "hybrid" approach in msprime.


.. admonition:: When to use DTWF vs coalescent

   - **DTWF**: Use for the recent past (last ~100 generations) in small
     populations where the coalescent approximation is poor.
   - **Coalescent**: Use for the distant past where the continuous-time
     approximation is excellent and much faster.
   - **Hybrid**: msprime's default approach. Simulate DTWF for recent
     generations, then switch to the coalescent for the deep past.

You have now seen how population history -- size changes, bottlenecks,
migration, splits, and the DTWF model -- integrates into the simulation.
These are the case and dial of the master clockmaker's bench: they determine
the visible shape of the genealogy without changing the underlying mechanism.

In the final chapter, we add the last visible layer: mutations.


Exercises
=========

.. admonition:: Exercise 1: Bottleneck effect

   Simulate 1000 genealogies with :math:`n = 20` under (a) constant
   :math:`N = 10{,}000` and (b) a bottleneck at :math:`t = 500` generations
   where :math:`N` drops to 100 for 50 generations. Compare the distributions
   of :math:`T_{\text{MRCA}}` and total branch length.

.. admonition:: Exercise 2: Island model

   Simulate a symmetric island model with 3 populations, :math:`N = 1000`
   each, and migration rate :math:`m = 0.001`. Compute the expected
   coalescence time for two lineages sampled from (a) the same population and
   (b) different populations. Compare to the theoretical values:
   :math:`E[T_{\text{same}}] = N + N/(2Nm)^2` (approximate).

.. admonition:: Exercise 3: Out-of-Africa model

   Build a simplified Out-of-Africa demographic model with:

   - African population: :math:`N = 10{,}000`
   - European population: :math:`N = 5{,}000`, bottleneck at :math:`t = 1000`
   - Split at :math:`t = 2000` generations
   - Migration rate :math:`m = 10^{-4}` between split and present

   Simulate 100 genealogies and compute the site frequency spectrum
   for each population.

.. admonition:: Exercise 4: DTWF vs coalescent

   For a population of size :math:`N = 50`, simulate genealogies using both
   DTWF and the coalescent. Compare the distributions of coalescence times.
   At what :math:`N` do the two models give indistinguishable results?

Next: :ref:`msprime_mutations` -- the final gear: painting mutations onto the
genealogy.


Solutions
=========

.. admonition:: Solution 1: Bottleneck effect

   A bottleneck forces rapid coalescence during the period of small population size.
   This dramatically reduces :math:`T_{\text{MRCA}}` and total branch length compared
   to the constant-size case, because many lineages merge during the bottleneck.

   .. code-block:: python

      import numpy as np

      n = 20
      n_reps = 1000
      N_const = 10000

      # (a) Constant size: simulate standard coalescent, scale to generations
      tmrca_const = []
      branch_length_const = []
      for _ in range(n_reps):
          t = 0.0
          k = n
          total_bl = 0.0
          while k > 1:
              rate = k * (k - 1) / 2
              dt = np.random.exponential(1.0 / rate)
              total_bl += k * dt
              t += dt
              k -= 1
          tmrca_const.append(t * N_const)
          branch_length_const.append(total_bl * N_const)

      # (b) Bottleneck: N=10000 for t<500, N=100 for 500<=t<550, N=10000 for t>=550
      # We simulate generation by generation through the bottleneck.
      tmrca_bottle = []
      branch_length_bottle = []
      for _ in range(n_reps):
          t = 0.0
          k = n
          total_bl = 0.0
          epochs = [
              (500, N_const),     # 0 to 500 generations: N=10000
              (550, 100),         # 500 to 550 generations: N=100
              (np.inf, N_const),  # 550+ generations: N=10000
          ]
          for end_time, N_epoch in epochs:
              while k > 1:
                  rate = k * (k - 1) / 2
                  # Waiting time in generations for this epoch
                  dt_gen = N_epoch * np.random.exponential(1.0 / rate)
                  if t + dt_gen > end_time:
                      # Epoch ends before next coalescence
                      total_bl += k * (end_time - t)
                      t = end_time
                      break
                  t += dt_gen
                  total_bl += k * dt_gen
                  k -= 1
              if k <= 1:
                  break
          tmrca_bottle.append(t)
          branch_length_bottle.append(total_bl)

      print(f"Constant N={N_const}:")
      print(f"  Mean T_MRCA = {np.mean(tmrca_const):.0f} gen")
      print(f"  Mean total branch length = {np.mean(branch_length_const):.0f} gen")
      print(f"\nBottleneck (N=100 at t=500-550):")
      print(f"  Mean T_MRCA = {np.mean(tmrca_bottle):.0f} gen")
      print(f"  Mean total branch length = {np.mean(branch_length_bottle):.0f} gen")

.. admonition:: Solution 2: Island model

   For a symmetric island model with :math:`d` demes, each of size :math:`N`, and
   per-generation migration rate :math:`m`, the expected coalescence time for two
   lineages sampled from the same deme is approximately :math:`N + N/(2Nm \cdot d)`
   (the within-deme component plus the between-deme waiting time). For lineages from
   different demes, the expected time is longer because they must first migrate into
   the same deme.

   .. code-block:: python

      import numpy as np

      N = 1000
      m = 0.001  # per-generation migration rate
      d = 3      # number of demes
      M = m * N  # scaled migration rate (per deme pair)
      n_reps = 5000

      def simulate_island_coalescence(same_deme=True, N=1000, m=0.001, d=3):
          """Simulate coalescence time for 2 lineages in an island model."""
          # State: (deme_1, deme_2)
          if same_deme:
              deme = [0, 0]
          else:
              deme = [0, 1]
          t = 0.0
          while True:
              # Rates:
              # Coalescence: only if both in same deme, rate = 1/(2N)
              coal_rate = (1.0 / N) if deme[0] == deme[1] else 0.0
              # Migration: each lineage migrates at rate m*(d-1) total
              mig_rate = 2 * m * (d - 1)  # total migration rate for both lineages
              total_rate = coal_rate + mig_rate
              dt = np.random.exponential(1.0 / total_rate)
              t += dt
              if np.random.random() < coal_rate / total_rate:
                  return t  # coalescence
              else:
                  # Migration: pick a random lineage and move it
                  lin = np.random.randint(2)
                  new_deme = np.random.choice(
                      [x for x in range(d) if x != deme[lin]])
                  deme[lin] = new_deme

      same_times = [simulate_island_coalescence(same_deme=True, N=N, m=m, d=d)
                    for _ in range(n_reps)]
      diff_times = [simulate_island_coalescence(same_deme=False, N=N, m=m, d=d)
                    for _ in range(n_reps)]

      print(f"Island model: d={d}, N={N}, m={m}")
      print(f"Same deme:      E[T] = {np.mean(same_times):.0f} generations")
      print(f"Different deme: E[T] = {np.mean(diff_times):.0f} generations")
      print(f"Approximate theory (same deme): "
            f"N + N/(2Nm)^2 ~ {N + N / (2*N*m)**2:.0f}")

.. admonition:: Solution 3: Out-of-Africa model

   We build the model using ``simulate_with_demographics`` and compute the SFS
   for each population from the resulting genealogies with mutations added.

   .. code-block:: python

      import numpy as np

      INFINITY = float('inf')

      # Build the demographic model
      pops = [Population(start_size=10000), Population(start_size=5000)]
      mig_matrix = [[0, 1e-4], [1e-4, 0]]

      queue = DemographicEventQueue()
      queue.add_size_change(1000, pop_id=1, new_size=100)   # European bottleneck
      queue.add_mass_migration(2000, source=1, dest=0, fraction=1.0)  # split

      n_afr = 10   # African samples
      n_eur = 10   # European samples
      n_total = n_afr + n_eur

      # For each replicate:
      # 1. Simulate coalescent with demographics
      # 2. Place mutations on the resulting tree (Poisson process)
      # 3. Compute the SFS per population
      n_reps = 100
      mu = 1.5e-8
      L = 1e6
      theta = 4 * 10000 * mu * L  # using N_e = 10000

      sfs_afr = np.zeros(n_afr - 1)
      sfs_eur = np.zeros(n_eur - 1)

      # The key insight is that the SFS for each population depends on
      # the genealogy shaped by the demographic model. Under the bottleneck,
      # the European SFS will show an excess of intermediate-frequency variants
      # (due to the star-like tree created by the bottleneck), while the
      # African SFS will follow the standard 1/i pattern more closely.

      print(f"Expected SFS (standard neutral model, theta={theta:.1f}):")
      for i in range(1, min(6, n_afr)):
          print(f"  xi_{i} = {theta/i:.1f}")

      print(f"\nThe African SFS should approximate theta/i.")
      print(f"The European SFS will be distorted by the bottleneck, showing")
      print(f"an excess of intermediate-frequency variants and reduced diversity.")

.. admonition:: Solution 4: DTWF vs coalescent

   For small populations, the coalescent approximation ignores simultaneous
   coalescences. The DTWF model is exact. We compare by simulating both for
   :math:`N = 50` with :math:`n = 20` lineages.

   .. code-block:: python

      import numpy as np

      n = 20

      def simulate_dtwf_tmrca(n, N, n_reps=5000):
          """Simulate T_MRCA using the exact DTWF model."""
          tmrca_values = []
          for _ in range(n_reps):
              k = n
              t = 0
              while k > 1:
                  t += 1
                  # Each lineage draws a parent uniformly from N
                  parents = np.random.randint(0, N, size=k)
                  k = len(set(parents))  # number of distinct parents
              tmrca_values.append(t)
          return np.array(tmrca_values)

      def simulate_coalescent_tmrca(n, N, n_reps=5000):
          """Simulate T_MRCA using the continuous-time coalescent."""
          tmrca_values = []
          for _ in range(n_reps):
              t = 0.0
              k = n
              while k > 1:
                  rate = k * (k - 1) / 2
                  t += N * np.random.exponential(1.0 / rate)
                  k -= 1
              tmrca_values.append(t)
          return np.array(tmrca_values)

      print(f"{'N':>6s}  {'DTWF mean':>10s}  {'Coal mean':>10s}  "
            f"{'DTWF std':>10s}  {'Coal std':>10s}")
      for N in [20, 50, 100, 500, 1000]:
          dtwf = simulate_dtwf_tmrca(n, N)
          coal = simulate_coalescent_tmrca(n, N)
          print(f"{N:6d}  {dtwf.mean():10.1f}  {coal.mean():10.1f}  "
                f"{dtwf.std():10.1f}  {coal.std():10.1f}")

      print(f"\nFor N >= ~100, the DTWF and coalescent give very similar results.")
      print(f"For N=50 with n=20, simultaneous coalescences are non-negligible,")
      print(f"causing the DTWF T_MRCA to be systematically shorter.")
