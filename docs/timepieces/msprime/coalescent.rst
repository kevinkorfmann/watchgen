.. _coalescent_process:

========================
The Coalescent Process
========================

   *The escapement of the mechanism: how lineages find common ancestors.*

The coalescent is the mathematical foundation of msprime. Before we can
simulate anything, we need to understand the stochastic process that governs
how lineages merge backwards in time. We build this from the very bottom:
starting with two lineages, then :math:`n`, then adding recombination.

If you have worked through the :ref:`coalescent_theory` chapter, much of
the mathematics here will be familiar -- but the emphasis is different. There,
we developed the theory for its own sake. Here, we are building toward a
*simulation algorithm*: every formula we derive will become a line of code
in :ref:`hudson_algorithm`, the main simulation loop -- the ticking of the
clock.

Think of this chapter as calibrating the escapement: we need to know exactly
how fast the gears should turn before we can assemble the movement.


Step 1: Two Lineages, No Recombination
========================================

Start with the simplest possible case: two haploid genomes sampled from a
population of constant size :math:`N`. Going one generation back, what is the
probability that these two genomes share a parent?

In a haploid Wright-Fisher population of size :math:`N`, each individual in
generation :math:`t` chose its parent uniformly at random from the :math:`N`
individuals in generation :math:`t-1`. So the probability that our two samples
chose the **same** parent is:

.. math::

   P(\text{same parent}) = \frac{1}{N}

and the probability they chose **different** parents is:

.. math::

   P(\text{different parents}) = 1 - \frac{1}{N}

If they didn't coalesce in the first generation, the problem resets: we now
have two lineages one generation further back, in the same population. The
process is memoryless.

.. admonition:: Probability Aside -- The memoryless property

   A random variable :math:`T` is **memoryless** if
   :math:`P(T > s + t \mid T > s) = P(T > t)` for all :math:`s, t \geq 0`.
   Knowing that no coalescence has happened so far gives you no information
   about how much longer you will wait. Among continuous distributions, only
   the exponential has this property. Among discrete distributions, only the
   geometric has it. This is why the coalescent waiting time (geometric in
   discrete time, exponential in continuous time) resets perfectly after each
   failed generation.

**Waiting time distribution.** Let :math:`T_2` be the number of generations
until the two lineages coalesce. The probability that they first coalesce in
generation :math:`g` is:

.. math::

   P(T_2 = g) = \left(1 - \frac{1}{N}\right)^{g-1} \cdot \frac{1}{N}

This is a **geometric distribution** with success probability :math:`1/N`.

**The continuous-time approximation.** For large :math:`N`, we can approximate
this discrete geometric distribution with a continuous exponential. Measure
time in units of :math:`N` generations: let :math:`t = g / N`. Then:

.. math::

   P(T_2 > t) = \left(1 - \frac{1}{N}\right)^{Nt} \approx e^{-t}

as :math:`N \to \infty`. So in units of :math:`N` generations, the coalescence
time of two lineages is approximately :math:`\text{Exponential}(1)`.

.. admonition:: Calculus Aside -- The exponential limit

   The key identity is :math:`\lim_{N \to \infty} (1 - 1/N)^N = e^{-1}`.
   More generally, :math:`\lim_{N \to \infty} (1 - c/N)^{Nt} = e^{-ct}`.
   This follows from taking the logarithm:
   :math:`Nt \cdot \ln(1 - c/N) \approx Nt \cdot (-c/N) = -ct` as
   :math:`N \to \infty`, using the first-order Taylor expansion
   :math:`\ln(1 - x) \approx -x` for small :math:`x`. For
   :math:`N = 10{,}000`, the error is about :math:`0.005\%` -- far smaller
   than any biological uncertainty.

.. code-block:: python

   import numpy as np

   def simulate_coalescence_time_discrete(N, n_replicates=10000):
       """Simulate coalescence time for 2 lineages in discrete generations.

       Each generation, the two lineages independently choose a parent
       uniformly from N individuals. If they pick the same one, they coalesce.
       """
       times = np.zeros(n_replicates)
       for rep in range(n_replicates):
           g = 0
           while True:
               g += 1
               # With probability 1/N, the two lineages pick the same parent
               if np.random.random() < 1.0 / N:
                   break
           times[rep] = g
       return times

   def simulate_coalescence_time_continuous(n_replicates=10000):
       """Simulate coalescence time for 2 lineages (exponential).

       In the continuous-time limit, the waiting time is Exp(1)
       in coalescent units (units of N generations).
       """
       return np.random.exponential(1.0, size=n_replicates)

   # Compare: discrete (in units of N generations) vs continuous
   N = 10000
   discrete_times = simulate_coalescence_time_discrete(N) / N  # rescale to coalescent units
   continuous_times = simulate_coalescence_time_continuous()

   print(f"Discrete:   mean = {discrete_times.mean():.4f}, "
         f"var = {discrete_times.var():.4f}")
   print(f"Continuous: mean = {continuous_times.mean():.4f}, "
         f"var = {continuous_times.var():.4f}")
   print(f"Theory:     mean = 1.0000, var = 1.0000")

With the two-lineage case in hand, we can now generalize to :math:`n` lineages.
The key insight is that each *pair* of lineages races independently to coalesce,
and with :math:`k` lineages there are :math:`\binom{k}{2}` such pairs.


Step 2: Multiple Lineages, No Recombination
=============================================

Now take :math:`k` lineages in a population of size :math:`N`. Going one
generation back, any pair could coalesce. How many pairs are there?

.. math::

   \binom{k}{2} = \frac{k(k-1)}{2}

Each pair independently has probability :math:`1/N` of coalescing. For large
:math:`N`, the probability that *any* pair coalesces is approximately:

.. math::

   P(\text{any coalescence}) \approx \frac{\binom{k}{2}}{N}

(We ignore the possibility of two simultaneous coalescences, which has
probability :math:`O(1/N^2)`.)

.. admonition:: Probability Aside -- Why we can ignore simultaneous events

   The probability that exactly one pair coalesces is
   :math:`\binom{k}{2} \cdot \frac{1}{N} \cdot (1 - \frac{1}{N})^{\binom{k}{2}-1} \approx \binom{k}{2}/N`.
   The probability that two or more pairs coalesce simultaneously is
   :math:`O(\binom{k}{2}^2 / N^2)`. For :math:`k = 100` and :math:`N = 10{,}000`,
   the single-event probability is about :math:`0.5`, while the
   double-event probability is about :math:`0.0025` -- negligible. In the
   continuous-time limit (:math:`N \to \infty`), simultaneous events have
   probability exactly zero.

**The continuous-time limit.** In units of :math:`N` generations, the waiting
time until the next coalescence among :math:`k` lineages is:

.. math::

   T_k \sim \text{Exponential}\left(\binom{k}{2}\right)

with mean :math:`\frac{1}{\binom{k}{2}} = \frac{2}{k(k-1)}`.

**The full coalescent process** for :math:`n` samples:

1. Start with :math:`k = n` lineages
2. Wait :math:`T_k \sim \text{Exp}(\binom{k}{2})` time units
3. Choose a random pair to coalesce: :math:`k \to k - 1`
4. Repeat until :math:`k = 1`

.. code-block:: python

   def simulate_coalescent(n, n_replicates=1):
       """Simulate the standard coalescent for n samples.

       Returns
       -------
       times : list of float
           Coalescence times (in coalescent units of N generations).
       pairs : list of (int, int)
           Which pair coalesced at each event.
       """
       all_results = []

       for _ in range(n_replicates):
           lineages = list(range(n))  # each lineage gets a unique ID
           times = []
           pairs = []
           t = 0.0

           while len(lineages) > 1:
               k = len(lineages)
               rate = k * (k - 1) / 2  # binom(k, 2) -- the coalescence rate
               # Wait exponential time with this rate
               t += np.random.exponential(1.0 / rate)
               # Choose random pair to coalesce
               i, j = sorted(np.random.choice(len(lineages), 2, replace=False))
               pairs.append((lineages[i], lineages[j]))
               times.append(t)
               # Merge: replace i with new node, remove j
               new_node = max(lineages) + 1
               lineages[i] = new_node
               lineages.pop(j)

           all_results.append((times, pairs))

       return all_results

   # Simulate and show the coalescent for n=5
   results = simulate_coalescent(5, n_replicates=1)
   times, pairs = results[0]
   print("Coalescent for n=5:")
   for i, (t, (a, b)) in enumerate(zip(times, pairs)):
       print(f"  k={5-i}: t={t:.4f}, lineages {a} and {b} coalesce")


Expected time to MRCA
----------------------

The total time to the Most Recent Common Ancestor (MRCA) is:

.. math::

   T_{\text{MRCA}} = \sum_{k=2}^{n} T_k

Since each :math:`T_k` is independent with mean :math:`\frac{2}{k(k-1)}`:

.. math::

   E[T_{\text{MRCA}}] = \sum_{k=2}^{n} \frac{2}{k(k-1)}
   = 2\sum_{k=2}^{n} \left(\frac{1}{k-1} - \frac{1}{k}\right)
   = 2\left(1 - \frac{1}{n}\right)

**This is remarkable.** The expected MRCA time approaches 2 (in units of
:math:`N` generations) regardless of how large :math:`n` is. Adding more
samples barely changes the total tree height.

.. admonition:: Calculus Aside -- Telescoping sums

   The partial fraction decomposition
   :math:`\frac{2}{k(k-1)} = 2\left(\frac{1}{k-1} - \frac{1}{k}\right)`
   gives a **telescoping sum**: most terms cancel in pairs.

   .. math::

      \sum_{k=2}^{n} 2\left(\frac{1}{k-1} - \frac{1}{k}\right)
      = 2\left[\left(\frac{1}{1} - \frac{1}{2}\right) + \left(\frac{1}{2} - \frac{1}{3}\right) + \cdots + \left(\frac{1}{n-1} - \frac{1}{n}\right)\right]
      = 2\left(1 - \frac{1}{n}\right)

   Everything except the first and last terms cancels. This is a standard
   technique in combinatorics and analysis; watch for it whenever you see
   partial fractions of the form :math:`1/[k(k-1)]`.

**Why?** With many lineages, the coalescence
rate :math:`\binom{k}{2} \sim k^2/2` is so high that events happen almost
instantly. Most of the time is spent waiting for the last few lineages to merge.

.. code-block:: python

   def expected_tmrca(n):
       """Expected MRCA time for n samples (in coalescent units)."""
       return 2 * (1 - 1.0 / n)

   def expected_total_branch_length(n):
       """Expected total branch length of the coalescent tree.

       This equals 2 * H_{n-1}, where H_k is the k-th harmonic number.
       """
       return 2 * sum(1.0 / k for k in range(1, n))

   for n in [2, 5, 10, 50, 100, 1000]:
       print(f"n={n:>5d}: E[T_MRCA] = {expected_tmrca(n):.4f}, "
             f"E[total length] = {expected_total_branch_length(n):.4f}")

We now know how to simulate *when* events happen (exponential waiting times)
and *what* happens (a random pair coalesces). But in the real simulation,
coalescence is not the only possible event. Recombination and migration also
compete for the clock's next tick. To handle this competition, we need the
**exponential race**.


Step 3: The Exponential Race
==============================

The coalescent with recombination involves multiple types of events
(coalescence, recombination, migration) happening at different rates.
The simulation needs to determine which event happens next. This is where
the **exponential race** comes in.

**Fact.** If :math:`X_1 \sim \text{Exp}(\lambda_1)` and
:math:`X_2 \sim \text{Exp}(\lambda_2)` are independent, then:

1. :math:`\min(X_1, X_2) \sim \text{Exp}(\lambda_1 + \lambda_2)`
2. :math:`P(X_1 < X_2) = \frac{\lambda_1}{\lambda_1 + \lambda_2}`

**Proof of (1).** :math:`P(\min(X_1, X_2) > t) = P(X_1 > t) P(X_2 > t) = e^{-\lambda_1 t} e^{-\lambda_2 t} = e^{-(\lambda_1 + \lambda_2)t}`.

**Proof of (2).**

.. math::

   P(X_1 < X_2) = \int_0^\infty P(X_2 > t) \cdot f_{X_1}(t) \, dt
   = \int_0^\infty e^{-\lambda_2 t} \cdot \lambda_1 e^{-\lambda_1 t} \, dt
   = \lambda_1 \int_0^\infty e^{-(\lambda_1 + \lambda_2) t} \, dt
   = \frac{\lambda_1}{\lambda_1 + \lambda_2}

.. admonition:: Calculus Aside -- Evaluating the integral

   The integral :math:`\int_0^\infty e^{-\alpha t}\,dt = 1/\alpha` for
   :math:`\alpha > 0` is one of the most important integrals in probability.
   It follows immediately from the antiderivative
   :math:`\int e^{-\alpha t}\,dt = -\frac{1}{\alpha}e^{-\alpha t} + C` and
   the fact that :math:`e^{-\alpha t} \to 0` as :math:`t \to \infty`. In
   the proof above, :math:`\alpha = \lambda_1 + \lambda_2`.

This extends to any number of competing exponentials: the minimum of
:math:`\text{Exp}(\lambda_1), \ldots, \text{Exp}(\lambda_m)` is
:math:`\text{Exp}(\sum_i \lambda_i)`, and event :math:`i` wins with
probability :math:`\lambda_i / \sum_j \lambda_j`.

**This is the simulation engine.** At each step, msprime computes:

- :math:`\lambda_{\text{coal}}` = total coalescence rate
- :math:`\lambda_{\text{recomb}}` = total recombination rate
- :math:`\lambda_{\text{mig}}` = total migration rate

Then draws the minimum, advances time, and executes the winning event. This
is the heartbeat of :ref:`hudson_algorithm` -- the main simulation loop,
the ticking of the clock.

.. code-block:: python

   def exponential_race(*rates):
       """Simulate an exponential race.

       Each "competitor" proposes a random waiting time drawn from
       Exp(rate_i). The competitor with the shortest time wins.

       Parameters
       ----------
       rates : floats
           Rate of each competing process.

       Returns
       -------
       winner : int
           Index of the process that fired first.
       time : float
           Waiting time until the first event.
       """
       times = []
       for rate in rates:
           if rate > 0:
               # Draw a random waiting time: higher rate -> shorter wait on average
               times.append(np.random.exponential(1.0 / rate))
           else:
               times.append(np.inf)  # rate=0 means this event never fires

       winner = np.argmin(times)
       return winner, times[winner]

   # Example: coalescence (rate=10) vs recombination (rate=5)
   wins = np.zeros(2)
   for _ in range(10000):
       w, t = exponential_race(10.0, 5.0)
       wins[w] += 1
   print(f"Coalescence wins: {wins[0]/100:.1f}% "
         f"(expected: {10/15*100:.1f}%)")
   print(f"Recombination wins: {wins[1]/100:.1f}% "
         f"(expected: {5/15*100:.1f}%)")

.. admonition:: A practical subtlety

   msprime does **not** actually compute the minimum of several exponential
   random variables. Instead, it draws each exponential independently and
   takes the minimum. This is mathematically equivalent but allows each
   rate to be computed separately (e.g., coalescence rates per population,
   migration rates per population pair), which is important for the
   implementation.

With the exponential race in our toolkit, we are ready to add the crucial
complication that makes the coalescent interesting (and computationally
challenging): recombination.


Step 4: Adding Recombination
==============================

Now the crucial complication. In the standard coalescent without recombination,
each lineage is a single indivisible entity. With recombination, a lineage can
**split**: part of the genome traces back through one parent, part through
the other.

Going backwards in time, a recombination event on a lineage at position
:math:`x` produces:

- A **left lineage** carrying ancestry for positions :math:`[0, x)`
- A **right lineage** carrying ancestry for positions :math:`[x, L)`

This increases the number of lineages by one.

**Rate of recombination.** A lineage carrying a segment of length :math:`\ell`
has a recombination rate of :math:`\rho \cdot \ell / L` per coalescent time
unit, where :math:`\rho = 4N_e r L` is the population-scaled recombination
rate for the whole genome. Equivalently, the per-base-pair rate in coalescent
units is :math:`\rho / L`.

The **total** recombination rate across all lineages is:

.. math::

   \lambda_{\text{recomb}} = \frac{\rho}{L} \sum_{i=1}^{k} \ell_i

where :math:`\ell_i` is the total length of ancestry carried by lineage
:math:`i`.

.. admonition:: Why recombination rate depends on segment length

   A recombination can only matter where the lineage carries ancestral
   material. If a lineage has already lost some genomic segments through
   prior coalescence events, recombination in those lost regions has no
   effect. Only the remaining "active" segments contribute to the rate.

.. admonition:: Closing a confusion gap -- Why does recombination *increase* the lineage count?

   In forward time, recombination takes *two* parental chromosomes and
   shuffles them into *one* offspring chromosome. But in the coalescent, we
   trace ancestry *backwards*. A present-day chromosome that was formed by
   recombination has *two* parents -- one for the left half and one for the
   right half. Tracing backwards, one lineage becomes two. This is the
   fundamental asymmetry of the backward view: coalescence *reduces* the
   lineage count (two children share one parent), while recombination
   *increases* it (one child has two parents for different genomic regions).

.. code-block:: python

   def coalescent_with_recombination_simple(n, L, rho, max_events=10000):
       """A simple (but slow) coalescent with recombination.

       Parameters
       ----------
       n : int
           Sample size.
       L : float
           Genome length.
       rho : float
           Population-scaled recombination rate (4*Ne*r*L).

       Returns
       -------
       events : list of (time, event_type, details)
       """
       # Each lineage is a list of (left, right) segments
       # Initially, every lineage carries the full genome [0, L).
       lineages = [[(0, L)] for _ in range(n)]
       events = []
       t = 0.0

       for _ in range(max_events):
           k = len(lineages)
           if k <= 1:
               break

           # Coalescence rate: any pair of k lineages can coalesce
           coal_rate = k * (k - 1) / 2

           # Recombination rate: proportional to total ancestry length
           total_length = sum(
               sum(r - l for l, r in segs) for segs in lineages
           )
           recomb_rate = rho * total_length / L

           # Exponential race between coalescence and recombination
           winner, dt = exponential_race(coal_rate, recomb_rate)
           t += dt

           if winner == 0:
               # Coalescence: pick two random lineages and merge
               i, j = sorted(np.random.choice(k, 2, replace=False))
               merged = merge_segments(lineages[i], lineages[j])
               lineages.pop(j)
               lineages[i] = merged
               events.append((t, 'coal', len(lineages)))

           else:
               # Recombination: pick a lineage weighted by length,
               # then pick a breakpoint
               lengths = [sum(r - l for l, r in segs) for segs in lineages]
               probs = np.array(lengths) / sum(lengths)
               idx = np.random.choice(k, p=probs)

               # Pick a random position within the segments
               bp = pick_random_breakpoint(lineages[idx], L)
               left_segs, right_segs = split_at_breakpoint(lineages[idx], bp)

               if left_segs and right_segs:
                   lineages[idx] = left_segs
                   lineages.append(right_segs)
                   events.append((t, 'recomb', len(lineages)))

       return events

   def merge_segments(segs_a, segs_b):
       """Merge two segment lists (simplified: just concatenate and sort)."""
       all_segs = sorted(segs_a + segs_b)
       # Merge overlapping intervals (coalescence reduces segment count)
       merged = [all_segs[0]]
       for l, r in all_segs[1:]:
           if l <= merged[-1][1]:
               merged[-1] = (merged[-1][0], max(merged[-1][1], r))
           else:
               merged.append((l, r))
       return merged

   def pick_random_breakpoint(segs, L):
       """Pick a random breakpoint within the segment list."""
       total = sum(r - l for l, r in segs)
       target = np.random.uniform(0, total)
       cumulative = 0
       for l, r in segs:
           cumulative += r - l
           if cumulative >= target:
               bp = r - (cumulative - target)
               return bp
       return segs[-1][1]

   def split_at_breakpoint(segs, bp):
       """Split segments at breakpoint bp into left and right."""
       left, right = [], []
       for l, r in segs:
           if r <= bp:
               left.append((l, r))       # entirely left of breakpoint
           elif l >= bp:
               right.append((l, r))      # entirely right of breakpoint
           else:
               left.append((l, bp))      # straddles: split into two pieces
               right.append((bp, r))
       return left, right

   # Simulate
   events = coalescent_with_recombination_simple(n=5, L=1000, rho=10.0)
   print(f"Total events: {len(events)}")
   for t, etype, k in events[:10]:
       print(f"  t={t:.4f}: {etype:>6s}, lineages={k}")

This simple implementation works, but it is slow: computing the total segment
length requires iterating over all segments every step. In
:ref:`segments_fenwick`, we will replace this with a Fenwick tree -- a clever
indexing mechanism for fast event scheduling -- that maintains the total in
:math:`O(\log n)` time.


Step 5: The Coalescent with Recombination in Detail
=====================================================

Let's be more precise about the rates. In a population of effective size
:math:`N_e`, with :math:`k` lineages, each carrying some genomic segments:

**Coalescence rate:**

In the standard (Hudson) model, any pair of :math:`k` lineages can coalesce:

.. math::

   \lambda_{\text{coal}} = \frac{\binom{k}{2}}{N_e} = \frac{k(k-1)}{2N_e}

In coalescent time units (measured in :math:`N_e` generations), the :math:`N_e`
cancels and the rate is simply :math:`\binom{k}{2}`.

.. admonition:: SMC and SMC' approximations

   In the Sequentially Markov Coalescent (SMC), not all :math:`\binom{k}{2}`
   pairs can coalesce -- only those whose ancestral segments **overlap**
   genomically. This reduces the rate but also the state space, making
   inference algorithms (like PSMC and SINGER) tractable.

   - **SMC** (McVean & Cardin, 2005): After a recombination, the new lineage
     can only re-coalesce with lineages that share a marginal tree at the
     breakpoint.

   - **SMC'** (Marjoram & Wall, 2006): Relaxes SMC slightly to allow
     re-coalescence with lineages that have contiguous ancestral segments.

   - **SMC(k)**: A parameterized family that interpolates between SMC and
     the full coalescent. msprime implements this using "hulls" -- bounding
     boxes of each lineage's ancestry.

**Recombination rate:**

A lineage carrying total ancestral material of length :math:`\ell` (summed
over all its segments) experiences recombination at rate:

.. math::

   \lambda_{\text{recomb, lineage}} = \frac{\ell}{L} \cdot \frac{\rho}{2}

where :math:`\rho = 4N_e r L` is the population-scaled total recombination rate.
In practice, with a position-dependent recombination rate map :math:`r(x)`,
the recombination "mass" of a segment :math:`[a, b)` is:

.. math::

   m(a, b) = \int_a^b r(x) \, dx

and the total recombination rate is proportional to the total mass across all
segments.

.. admonition:: Calculus Aside -- From rate function to mass

   The integral :math:`m(a,b) = \int_a^b r(x)\,dx` is simply the area under
   the recombination rate curve between positions :math:`a` and :math:`b`.
   When the rate is constant (:math:`r(x) = r`), this reduces to
   :math:`m(a,b) = r \cdot (b - a)`. When the rate varies (e.g.,
   recombination hotspots), the integral accounts for the fact that some
   base pairs contribute more to the recombination probability than others.
   In :ref:`segments_fenwick`, we will store these masses in a Fenwick tree
   so that the total can be maintained efficiently as segments are created
   and destroyed.

**The breakpoint** is chosen uniformly over the total recombination mass
(not uniformly over genomic position). This naturally concentrates breakpoints
in recombination hotspots.

Now that we have the recombination machinery, there is one more ingredient
needed for realistic simulations: the effect of changing population size on
coalescence waiting times.


Step 6: The Coalescent Waiting Time with Growth
=================================================

When the population size changes through time, the coalescence rate changes
too. A key case is **exponential growth**:

.. math::

   N(t) = N_0 \cdot e^{-\alpha t}

where :math:`\alpha > 0` is the growth rate and :math:`t` is measured backwards
(so the population was smaller in the past for :math:`\alpha > 0`).

The coalescence rate with :math:`k` lineages at time :math:`t` is:

.. math::

   \lambda(t) = \frac{\binom{k}{2}}{N(t)} = \frac{\binom{k}{2}}{N_0 e^{-\alpha t}}

The waiting time is no longer a simple exponential. We need the cumulative
hazard: the integral of the rate from the current time :math:`t_0` to some
future time :math:`t_0 + w`:

.. math::

   \Lambda(w) = \int_{t_0}^{t_0 + w} \frac{\binom{k}{2}}{N_0 e^{-\alpha s}} \, ds

The survival function is :math:`P(W > w) = \exp(-\Lambda(w))`. To sample from
this distribution, we use the inversion method.

.. admonition:: Probability Aside -- The inversion method

   To sample from a distribution with survival function :math:`S(w) = e^{-\Lambda(w)}`,
   we draw :math:`U \sim \text{Uniform}(0,1)` and solve :math:`S(W) = U`, i.e.,
   :math:`\Lambda(W) = -\ln(U)`. Since :math:`-\ln(U) \sim \text{Exp}(1)`,
   this is equivalent to drawing :math:`E \sim \text{Exp}(1)` and solving
   :math:`\Lambda(W) = E`. The advantage is that we can solve for :math:`W`
   analytically when :math:`\Lambda` has a closed-form inverse.

**Deriving the waiting time formula.** Let :math:`c = \binom{k}{2}`. We draw
:math:`U \sim \text{Exp}(2c)` (the waiting time under constant size
:math:`N_0 = 1`). Then we need to solve:

.. math::

   U = \int_{t_0}^{t_0 + W} \frac{c}{N(s)} \, ds

For constant size: :math:`W = N_0 \cdot U / c \cdot 2 = N_0 \cdot U`, which
gives :math:`\text{Exp}(c / N_0)`.

For exponential growth, using :math:`N(s) = N_0 e^{-\alpha(s - t_0)}`:

.. math::

   U = \frac{c}{N_0} \int_0^W e^{\alpha s} \, ds = \frac{c}{N_0 \alpha}(e^{\alpha W} - 1)

.. admonition:: Calculus Aside -- Evaluating the growth integral

   The integral :math:`\int_0^W e^{\alpha s}\,ds` has antiderivative
   :math:`\frac{1}{\alpha}e^{\alpha s}`. Evaluating:

   .. math::

      \int_0^W e^{\alpha s}\,ds = \frac{1}{\alpha}\left[e^{\alpha W} - e^{0}\right] = \frac{e^{\alpha W} - 1}{\alpha}

   This integral appears frequently in demographic models. When
   :math:`\alpha \to 0`, l'Hopital's rule gives
   :math:`\frac{e^{\alpha W} - 1}{\alpha} \to W`, recovering the
   constant-size case.

Solving for :math:`W`:

.. math::

   e^{\alpha W} = 1 + \frac{\alpha N_0 U}{c}

Taking the logarithm:

.. math::

   W = \frac{1}{\alpha} \ln\left(1 + \alpha N_0 U \cdot \frac{1}{c}\right)

This is valid only if :math:`1 + \alpha N_0 U / c > 0`. When :math:`\alpha < 0`
(population was *larger* in the past), the argument can become zero or negative,
meaning the population was so large that coalescence never occurs within the
growth epoch -- the simulation must wait for a demographic event that changes
the growth rate.

.. code-block:: python

   import math

   def coalescent_waiting_time_constant(k, N):
       """Waiting time for k lineages, constant population size N."""
       rate = k * (k - 1) / 2  # binom(k,2)
       u = np.random.exponential(1.0 / (2 * rate))
       return N * u  # scale from coalescent units to generations

   def coalescent_waiting_time_growth(k, N0, alpha, t0):
       """Waiting time for k lineages, exponential growth.

       Parameters
       ----------
       k : int
           Number of lineages.
       N0 : float
           Current population size.
       alpha : float
           Growth rate (positive = population was smaller in the past).
       t0 : float
           Current time.

       Returns
       -------
       w : float
           Waiting time (can be inf if coalescence doesn't occur).
       """
       rate = k * (k - 1) / 2  # binom(k,2)
       u = np.random.exponential(1.0 / (2 * rate))  # draw from Exp(2c)

       if alpha == 0:
           return N0 * u  # constant-size case

       dt = 0  # already at t0
       # Apply the inversion formula derived above
       z = 1 + alpha * N0 * math.exp(-alpha * dt) * u
       if z <= 0:
           return np.inf  # coalescence doesn't happen in this epoch

       return math.log(z) / alpha

   # Compare waiting times: constant vs growing population
   N0, alpha = 10000, 0.01
   k = 10
   n_reps = 10000

   const_times = [coalescent_waiting_time_constant(k, N0) for _ in range(n_reps)]
   growth_times = [coalescent_waiting_time_growth(k, N0, alpha, 0)
                   for _ in range(n_reps)]

   print(f"Constant N={N0}: mean waiting time = {np.mean(const_times):.1f} gen")
   print(f"Growth alpha={alpha}: mean waiting time = {np.mean(growth_times):.1f} gen")
   print(f"(Growth concentrates coalescences in the recent past)")

.. admonition:: The formula in msprime's code

   In the reference implementation (``algorithms.py``), the waiting time
   with growth is computed in ``Population._get_common_ancestor_waiting_time``:

   .. code-block:: python

      u = random.expovariate(2 * np)
      if self.growth_rate == 0:
          ret = self.start_size * u
      else:
          z = 1 + self.growth_rate * self.start_size
              * math.exp(-self.growth_rate * dt) * u
          if z > 0:
              ret = math.log(z) / self.growth_rate

   This matches our derivation exactly.

With coalescence, recombination, and variable population size all accounted
for, there is one more biological mechanism to add: gene conversion.


Step 7: Gene Conversion
=========================

Gene conversion is a recombination-like event with a twist: instead of
exchanging everything to one side of a breakpoint, it copies a short
**tract** of DNA (typically a few hundred base pairs) from one homolog
to the other.

Going backwards in time, a gene conversion event produces two lineages:

- One carrying a short segment (the converted tract)
- One carrying everything except that tract

.. code-block:: text

   Before gene conversion (going backwards):

   Lineage:  ==========================================
                           | gene conversion at position x
                           | with tract length l

   After:
   Lineage 1: ============     ========================
                          |   |
   Lineage 2:             =====
                          x   x+l

The **rate** of gene conversion initiation is proportional to the total
genomic mass of the lineage, just like recombination. The **tract length**
is drawn from a geometric distribution (discrete genome) or exponential
distribution (continuous genome):

.. math::

   \ell \sim \text{Geometric}(1/\bar{\ell}) \quad \text{or} \quad
   \ell \sim \text{Exponential}(\bar{\ell})

where :math:`\bar{\ell}` is the mean tract length (typically ~300-500 bp in
humans).

Additionally, gene conversion can occur **to the left** of the first ancestral
segment, producing a lineage that extends the ancestry leftward:

.. code-block:: text

   Before:        ===== ======= ====
                  |
   Gene conversion starting to the left:

   After:
   Lineage 1:     ===== ======= ====
   Lineage 2: ====
              |        |
              (new)    (original start)

This "left extension" has a separate rate proportional to the number of
ancestors times the mean gene conversion rate times the tract length.

.. code-block:: python

   def gene_conversion_event(segments, gc_position, tract_length, L):
       """Simulate a gene conversion event.

       Gene conversion removes a short tract from the main lineage and
       places it on a new lineage. This is like recombination, but
       instead of splitting left/right, it punches a "hole" in the middle.

       Parameters
       ----------
       segments : list of (left, right)
           Segments of the lineage.
       gc_position : float
           Starting position of the gene conversion.
       tract_length : float
           Length of the converted tract.

       Returns
       -------
       main_segs : list of (left, right)
           Remaining segments of the main lineage.
       tract_segs : list of (left, right)
           Segments of the gene conversion tract lineage.
       """
       gc_left = gc_position
       gc_right = min(gc_position + tract_length, L)

       main_segs = []
       tract_segs = []

       for l, r in segments:
           if r <= gc_left or l >= gc_right:
               # Entirely outside tract: stays with main
               main_segs.append((l, r))
           elif l >= gc_left and r <= gc_right:
               # Entirely inside tract: goes to tract lineage
               tract_segs.append((l, r))
           elif l < gc_left and r > gc_right:
               # Straddles both sides: split into three pieces
               main_segs.append((l, gc_left))
               tract_segs.append((gc_left, gc_right))
               main_segs.append((gc_right, r))
           elif l < gc_left:
               # Overlaps left boundary only
               main_segs.append((l, gc_left))
               tract_segs.append((gc_left, r))
           else:
               # Overlaps right boundary only
               tract_segs.append((l, gc_right))
               main_segs.append((gc_right, r))

       return main_segs, tract_segs

   # Example
   segs = [(100, 500), (700, 1000)]
   main, tract = gene_conversion_event(segs, gc_position=400, tract_length=350, L=1000)
   print(f"Original: {segs}")
   print(f"Main:     {main}")
   print(f"Tract:    {tract}")

With all event types defined, let us now summarize the complete rate table.


Step 8: Putting It Together -- Event Rates Summary
=====================================================

At any point in the simulation, with :math:`k` lineages in a population of
size :math:`N(t)`, the competing event rates are:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Event
     - Rate
     - Effect
   * - Coalescence
     - :math:`\binom{k}{2} / N(t)`
     - Two lineages merge (:math:`k \to k-1`)
   * - Recombination
     - :math:`\sum_i m_i^{\text{recomb}}`
     - One lineage splits (:math:`k \to k+1`)
   * - Gene conversion (within)
     - :math:`\sum_i m_i^{\text{gc}}`
     - One lineage splits (:math:`k \to k+1`)
   * - Gene conversion (left)
     - :math:`k \cdot \bar{g} \cdot \bar{\ell}`
     - One lineage splits (:math:`k \to k+1`)
   * - Migration
     - :math:`k_j \cdot M_{j \to j'}`
     - One lineage moves between populations

where :math:`m_i^{\text{recomb}}` is the recombination mass of lineage
:math:`i` and :math:`m_i^{\text{gc}}` is the gene conversion mass.

The simulation draws independent exponential waiting times for each event
type, takes the minimum, advances time, and executes the winning event.
This is the exponential race in full generality, and it drives the main loop
of :ref:`hudson_algorithm`.

.. admonition:: Termination condition

   The simulation terminates when every genomic position has found its MRCA.
   This is tracked by an overlap counter :math:`S[x]` that records how many
   lineages carry ancestral material at position :math:`x`. When
   :math:`S[x] = 1` for all :math:`x`, we're done. (In practice, msprime
   uses ``S[x] = 0`` after a final decrement, stored in an AVL tree keyed
   by genomic position.) We will build this overlap counter in
   :ref:`segments_fenwick`.

We now have the complete mathematical specification of the coalescent with
recombination. Every rate, every distribution, every event type is defined.
But simulating this efficiently requires clever data structures -- which
brings us to the next chapter.


Exercises
=========

.. admonition:: Exercise 1: Verify the coalescent

   Simulate 10,000 coalescent trees with :math:`n = 10`. Compute the mean
   and variance of :math:`T_{\text{MRCA}}`. Compare to the theoretical
   values: :math:`E[T_{\text{MRCA}}] = 2(1 - 1/n)`,
   :math:`\text{Var}(T_{\text{MRCA}}) = \sum_{k=2}^n 4/[k(k-1)]^2`.

.. admonition:: Exercise 2: The exponential race

   Implement the exponential race for 3 competing events with rates 1, 2, 3.
   Verify empirically that event :math:`i` wins with probability
   :math:`\lambda_i / \sum \lambda_j`, and the minimum has rate
   :math:`\sum \lambda_j`.

.. admonition:: Exercise 3: Coalescent with recombination

   Use ``coalescent_with_recombination_simple`` to simulate 1000 genealogies
   with :math:`n = 5`, :math:`L = 10^4`, and varying :math:`\rho`. Plot
   the number of recombination events as a function of :math:`\rho`. Compare
   to the theoretical expectation: :math:`E[\text{recomb events}] \approx \rho \cdot \sum_{k=2}^n 1/(k-1)`.

.. admonition:: Exercise 4: Growth vs. constant size

   Simulate coalescent trees under (a) constant :math:`N = 10{,}000` and
   (b) exponential growth with :math:`N_0 = 10{,}000`, :math:`\alpha = 0.01`.
   Plot the distributions of :math:`T_{\text{MRCA}}`. How does growth affect
   the shape of the genealogy?

Next: :ref:`segments_fenwick` -- the linked-list track that follows each
lineage's ancestral material, and the Fenwick tree -- a clever indexing
mechanism for fast event scheduling.


Solutions
=========

.. admonition:: Solution 1: Verify the coalescent

   We simulate 10,000 coalescent trees using ``simulate_coalescent`` and compare the
   empirical mean and variance of :math:`T_{\text{MRCA}}` to their theoretical values.

   The theoretical variance uses the fact that the :math:`T_k` are independent, so:

   .. math::

      \text{Var}(T_{\text{MRCA}}) = \sum_{k=2}^{n} \text{Var}(T_k)
      = \sum_{k=2}^{n} \frac{1}{\binom{k}{2}^2}
      = \sum_{k=2}^{n} \frac{4}{[k(k-1)]^2}

   since :math:`T_k \sim \text{Exp}(\binom{k}{2})` and the variance of an
   :math:`\text{Exp}(\lambda)` random variable is :math:`1/\lambda^2`.

   .. code-block:: python

      import numpy as np

      n = 10
      n_replicates = 10000

      # Theoretical values
      E_tmrca = 2 * (1 - 1.0 / n)
      Var_tmrca = sum(4.0 / (k * (k - 1))**2 for k in range(2, n + 1))

      # Simulate
      tmrca_values = []
      for _ in range(n_replicates):
          results = simulate_coalescent(n, n_replicates=1)
          times, pairs = results[0]
          tmrca_values.append(times[-1])  # last coalescence time is the MRCA

      tmrca_values = np.array(tmrca_values)

      print(f"E[T_MRCA]:   simulated = {tmrca_values.mean():.4f}, "
            f"theory = {E_tmrca:.4f}")
      print(f"Var[T_MRCA]: simulated = {tmrca_values.var():.4f}, "
            f"theory = {Var_tmrca:.4f}")

.. admonition:: Solution 2: The exponential race

   We run the race 100,000 times with rates :math:`\lambda_1 = 1, \lambda_2 = 2,
   \lambda_3 = 3`. The total rate is :math:`\Lambda = 6`, so the minimum has
   mean :math:`1/6` and event :math:`i` wins with probability
   :math:`\lambda_i / \Lambda`.

   .. code-block:: python

      import numpy as np

      rates = [1.0, 2.0, 3.0]
      total_rate = sum(rates)
      n_trials = 100000

      wins = np.zeros(3)
      min_times = []

      for _ in range(n_trials):
          winner, t = exponential_race(*rates)
          wins[winner] += 1
          min_times.append(t)

      min_times = np.array(min_times)

      print("Win probabilities:")
      for i in range(3):
          print(f"  Event {i}: observed = {wins[i]/n_trials:.4f}, "
                f"expected = {rates[i]/total_rate:.4f}")

      print(f"\nMinimum time distribution:")
      print(f"  Mean: observed = {min_times.mean():.4f}, "
            f"expected = {1.0/total_rate:.4f}")
      print(f"  Var:  observed = {min_times.var():.4f}, "
            f"expected = {1.0/total_rate**2:.4f}")

.. admonition:: Solution 3: Coalescent with recombination

   The expected number of recombination events in a coalescent tree is approximately
   :math:`\frac{\rho}{2} \cdot E[L_{\text{total}}]` where
   :math:`E[L_{\text{total}}] = 2 \sum_{k=1}^{n-1} 1/k`. We simulate for several
   values of :math:`\rho` and compare.

   .. code-block:: python

      import numpy as np

      n = 5
      L = 10000
      n_replicates = 1000
      rho_values = [0.1, 1.0, 5.0, 10.0, 20.0, 50.0]

      # Theoretical expected total branch length (coalescent units)
      E_total_length = 2 * sum(1.0 / k for k in range(1, n))

      print(f"E[total branch length] = {E_total_length:.4f}")
      print(f"{'rho':>6s}  {'E[recomb] (sim)':>16s}  {'E[recomb] (theory)':>18s}")

      for rho in rho_values:
          recomb_counts = []
          for _ in range(n_replicates):
              events = coalescent_with_recombination_simple(n=n, L=L, rho=rho)
              n_recomb = sum(1 for _, etype, _ in events if etype == 'recomb')
              recomb_counts.append(n_recomb)

          mean_recomb = np.mean(recomb_counts)
          # Theory: E[recomb] = (rho/2) * E[L_total]
          expected_recomb = (rho / 2) * E_total_length
          print(f"{rho:6.1f}  {mean_recomb:16.2f}  {expected_recomb:18.2f}")

.. admonition:: Solution 4: Growth vs. constant size

   Exponential growth (:math:`\alpha > 0`) makes the population smaller in the past,
   which increases the coalescence rate at earlier times. This compresses the genealogy:
   :math:`T_{\text{MRCA}}` is shorter and the tree is more star-shaped (most coalescences
   happen at roughly the same time in the recent past).

   .. code-block:: python

      import numpy as np

      N0 = 10000
      alpha = 0.01
      n = 10
      n_reps = 10000

      # Constant size: standard coalescent, then scale to generations
      tmrca_const = []
      for _ in range(n_reps):
          results = simulate_coalescent(n, n_replicates=1)
          times, _ = results[0]
          tmrca_const.append(times[-1] * N0)  # scale to generations

      # Growth: draw waiting times with the growth formula
      tmrca_growth = []
      for _ in range(n_reps):
          t = 0.0
          k = n
          while k > 1:
              w = coalescent_waiting_time_growth(k, N0, alpha, t)
              if w == np.inf:
                  break
              t += w
              k -= 1
          tmrca_growth.append(t)

      tmrca_const = np.array(tmrca_const)
      tmrca_growth = np.array(tmrca_growth)

      print(f"Constant N={N0}:")
      print(f"  Mean T_MRCA = {tmrca_const.mean():.0f} generations")
      print(f"  Std  T_MRCA = {tmrca_const.std():.0f} generations")
      print(f"\nGrowth alpha={alpha}:")
      print(f"  Mean T_MRCA = {tmrca_growth.mean():.0f} generations")
      print(f"  Std  T_MRCA = {tmrca_growth.std():.0f} generations")
      print(f"\nGrowth reduces T_MRCA because the ancestral population was smaller, "
            f"forcing lineages to coalesce faster.")
