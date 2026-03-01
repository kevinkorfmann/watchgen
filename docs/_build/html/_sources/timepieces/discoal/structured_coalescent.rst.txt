.. _discoal_structured_coalescent:

=============================================
The Structured Coalescent Under Selection
=============================================

   *The escapement of the mechanism: how lineages coalesce and migrate between two backgrounds.*

With the allele frequency trajectory :math:`x(t)` in hand (from the previous
chapter), we can now build the core of discoal's algorithm: the **structured
coalescent** that runs backward in time through the sweep phase.

Under neutrality, the coalescent is simple: :math:`k` lineages coalesce pairwise
at rate :math:`\binom{k}{2}` in a population of size :math:`2N` (in units of
:math:`2N` generations). But during a selective sweep, the population is split into
two subpopulations that change in size over time. The coalescent becomes
**structured** and **inhomogeneous** (time-varying).

This chapter derives the event rates from first principles, implements the
structured coalescent in Python, and shows how the sweep distorts the genealogy
compared to the neutral case.


.. note::

   **Prerequisites.** This chapter builds directly on:

   - :ref:`discoal_allele_trajectory` -- the trajectory :math:`x(t)` is the input
   - :ref:`coalescent_process` -- coalescence rates for :math:`k` lineages
   - Familiarity with continuous-time Markov chains and exponential waiting times


Step 1: Partitioning the Population
=====================================

During the sweep, every chromosome in the population carries either the
**beneficial allele** or the **wild-type allele** at the selected site. This
divides the population of :math:`2N` chromosomes into two classes:

.. math::

   \text{Background } B: \quad \text{size} = 2N \cdot x(t)

.. math::

   \text{Background } b: \quad \text{size} = 2N \cdot (1 - x(t))

where :math:`x(t)` is the frequency of the beneficial allele at time :math:`t`
(forward time).

Going **backward** in time through the sweep (which is how the coalescent
operates), :math:`x` decreases from 1.0 at the moment of fixation to
:math:`1/(2N)` at the origin of the mutation. So background :math:`B` **shrinks**
and background :math:`b` **grows** as we look further into the past.

At the moment of fixation (:math:`x = 1`), all chromosomes are in :math:`B` --
every sampled lineage starts on the beneficial background. As we trace backward,
recombination can move lineages from :math:`B` to :math:`b` (and vice versa), and
coalescence can reduce the number of lineages within each background.

.. code-block:: text

   Time (backward)    x(t)      B size     b size
   ───────────────    ────      ──────     ──────
   Just fixed         1.00      2N         0
   ...                0.80      1.6N       0.4N
   ...                0.50      N          N
   ...                0.20      0.4N       1.6N
   ...                0.01      0.02N      1.98N
   Origin             1/2N      1          2N - 1

   The B background shrinks to a single chromosome: the original mutant.
   Coalescence within B accelerates dramatically as its size drops.


Step 2: Coalescence Rates Within Each Background
===================================================

Within each background, coalescence happens just as in the standard coalescent,
but with the effective population size equal to the background's size. If there
are :math:`n_B` lineages in background :math:`B` and :math:`n_b` lineages in
background :math:`b`, the pairwise coalescence rates are:

.. math::

   \lambda_B = \frac{\binom{n_B}{2}}{2N \cdot x(t)}
             = \frac{n_B(n_B - 1)}{2 \cdot 2N \cdot x(t)}

.. math::

   \lambda_b = \frac{\binom{n_b}{2}}{2N \cdot (1 - x(t))}
             = \frac{n_b(n_b - 1)}{2 \cdot 2N \cdot (1 - x(t))}

These are the standard coalescent rates :math:`\binom{k}{2}/(2N_{\text{eff}})`,
with :math:`N_{\text{eff}}` replaced by the background size.

The crucial consequence: **as** :math:`x(t) \to 0` **(going backward), coalescence
in** :math:`B` **becomes infinitely fast**. This is the bottleneck. When the
beneficial allele is at frequency 0.01, there are only :math:`0.01 \times 2N = 200`
beneficial chromosomes (for :math:`N = 10{,}000`). Two lineages in this tiny pool
coalesce 100 times faster than they would in the full population. This rapid
coalescence is what destroys diversity near the selected site.

.. admonition:: Closing a confusion gap -- Coalescence between backgrounds?

   Lineages in *different* backgrounds (:math:`B` vs :math:`b`) **cannot**
   coalesce directly. A :math:`B` lineage and a :math:`b` lineage have,
   by definition, different alleles at the selected site, so they cannot share
   a parent in the previous generation. Coalescence only occurs within a
   background.

   However, a lineage can *move* between backgrounds via recombination. Once
   two lineages are in the same background, they can then coalesce.

.. code-block:: python

   def coalescence_rate(n_lineages, background_size):
       """Coalescence rate for n_lineages in a background of given size.

       Parameters
       ----------
       n_lineages : int
           Number of lineages in this background.
       background_size : float
           Effective number of haploid chromosomes in this background.

       Returns
       -------
       rate : float
           Total pairwise coalescence rate (per generation).
       """
       if n_lineages < 2 or background_size <= 0:
           return 0.0
       return n_lineages * (n_lineages - 1) / (2.0 * background_size)

   # Example: 5 lineages in a B background of size 200 (x = 0.01, N = 10000)
   rate_B = coalescence_rate(5, 200)
   rate_neutral = coalescence_rate(5, 20000)  # same lineages, full population
   print(f"Coalescence rate in B (x=0.01): {rate_B:.4f} per generation")
   print(f"Neutral coalescence rate:       {rate_neutral:.6f} per generation")
   print(f"Ratio (speed-up):               {rate_B / rate_neutral:.0f}x")
   # Output:
   # Coalescence rate in B (x=0.01): 0.0500 per generation
   # Neutral coalescence rate:       0.000500 per generation
   # Ratio (speed-up):               100x


Step 3: Recombination as Migration
====================================

Recombination between the neutral locus and the selected site can transfer a
lineage from one background to the other. Think of it this way: going backward
in time, a lineage at the neutral locus is sitting on some chromosome. If a
recombination event occurs between the neutral and selected sites, the neutral
allele's ancestry switches to a *different* chromosome -- one that was randomly
chosen from the population.

If the lineage was on a :math:`B` chromosome and the recombinant chromosome is
from background :math:`b` (probability :math:`1 - x(t)`), the lineage "migrates"
from :math:`B` to :math:`b`. Conversely, a :math:`b` lineage can migrate to
:math:`B` with probability :math:`x(t)`.

The migration rates per lineage per generation are:

.. math::

   m_{B \to b} = r \cdot (1 - x(t))

.. math::

   m_{b \to B} = r \cdot x(t)

where :math:`r` is the recombination rate between the neutral locus and the
selected site. The **total** migration rates for all lineages are:

.. math::

   M_{B \to b} = n_B \cdot r \cdot (1 - x(t))

.. math::

   M_{b \to B} = n_b \cdot r \cdot x(t)

.. admonition:: Why does recombination act as migration?

   In the standard coalescent *without* selection, recombination splits a lineage
   into two pieces that independently trace their ancestry. But in the structured
   coalescent *with* selection, we are tracking which **background** each lineage
   is on. A recombination event does not split the lineage -- it potentially
   *moves* it to the other background.

   This is because we are simulating a single neutral locus linked to a selected
   site. When recombination occurs between them, the neutral locus's ancestry
   continues on whatever chromosome the recombinant picked up -- which could be
   from either background.

   Compare this to msprime's standard ARG, where recombination genuinely splits
   ancestry. In discoal's structured coalescent, recombination at the selected
   site acts as a population transfer.

The key parameter that governs escape from the sweep is the **ratio** :math:`r/s`:

.. code-block:: python

   def migration_rates(n_B, n_b, r, x):
       """Compute migration rates between backgrounds due to recombination.

       Parameters
       ----------
       n_B : int
           Number of lineages in the beneficial background.
       n_b : int
           Number of lineages in the wild-type background.
       r : float
           Recombination rate between neutral locus and selected site.
       x : float
           Current frequency of the beneficial allele.

       Returns
       -------
       m_B_to_b : float
           Total rate of migration from B to b (per generation).
       m_b_to_B : float
           Total rate of migration from b to B (per generation).
       """
       m_B_to_b = n_B * r * (1.0 - x)
       m_b_to_B = n_b * r * x
       return m_B_to_b, m_b_to_B


Step 4: The Complete Event Rates
==================================

At any moment during the sweep, four types of events can occur:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Event
     - Rate (per generation)
     - Effect
   * - Coalescence in :math:`B`
     - :math:`\frac{n_B(n_B-1)}{2 \cdot 2Nx}`
     - Merge two :math:`B` lineages
   * - Coalescence in :math:`b`
     - :math:`\frac{n_b(n_b-1)}{2 \cdot 2N(1-x)}`
     - Merge two :math:`b` lineages
   * - Migration :math:`B \to b`
     - :math:`n_B \cdot r \cdot (1-x)`
     - Move one lineage from :math:`B` to :math:`b`
   * - Migration :math:`b \to B`
     - :math:`n_b \cdot r \cdot x`
     - Move one lineage from :math:`b` to :math:`B`

The **total event rate** is the sum of all four:

.. math::

   \Lambda = \lambda_B + \lambda_b + M_{B \to b} + M_{b \to B}

Given that an event occurs, the probability of each type is proportional to its
rate. The time to the next event is exponentially distributed with rate
:math:`\Lambda`.

.. admonition:: Probability Aside -- Competing exponentials

   This is the same "racing exponentials" framework used in Hudson's algorithm
   (see :ref:`hudson_algorithm`). When multiple types of events each occur at
   their own rate, the time to the *first* event is exponential with rate equal
   to the *sum* of all rates. Which event actually occurs is drawn categorically,
   with probability proportional to each rate.

   Formally, if :math:`T_i \sim \text{Exp}(\lambda_i)` independently, then
   :math:`\min(T_1, \ldots, T_k) \sim \text{Exp}(\lambda_1 + \cdots + \lambda_k)`
   and :math:`P(\text{type } i \text{ occurs}) = \lambda_i / \sum_j \lambda_j`.


Step 5: The Inhomogeneous Process
====================================

There is one complication: the rates **change over time** because :math:`x(t)`
changes. This makes the process **inhomogeneous**: we cannot simply draw one
exponential waiting time and fast-forward.

discoal handles this by discretizing the trajectory into small time steps (one
generation each, or coarser intervals). Within each step, the rates are treated
as constant (using the :math:`x` value at that step). Events are tested within
each step, and if the exponential waiting time exceeds the step length, we advance
to the next step with the residual waiting time adjusted.

.. code-block:: python

   def structured_coalescent_sweep(trajectory, n_sample, r_site, N, rng=None):
       """Simulate the structured coalescent through a selective sweep.

       Runs backward through the trajectory, simulating coalescence and
       migration events. Returns the coalescence times.

       Parameters
       ----------
       trajectory : ndarray
           Allele frequency x(t) at each generation, from origin to fixation.
           We process this backward (from fixation to origin).
       n_sample : int
           Number of sampled lineages.
       r_site : float
           Recombination rate between the neutral locus and the selected site
           (per generation).
       N : int
           Diploid effective population size.
       rng : np.random.Generator, optional
           Random number generator.

       Returns
       -------
       coal_times : list of float
           Times (in generations backward from fixation) of coalescence events.
       n_remaining_B : int
           Number of lineages remaining in B at the end of the sweep.
       n_remaining_b : int
           Number of lineages remaining in b at the end of the sweep.
       """
       if rng is None:
           rng = np.random.default_rng(42)

       two_N = 2 * N

       # All lineages start in B (the sweep just fixed)
       n_B = n_sample
       n_b = 0

       coal_times = []
       residual = rng.exponential(1.0)  # time until next event (in rate-units)

       # Process trajectory backward: from fixation (end) to origin (start)
       T_sweep = len(trajectory)
       for step in range(T_sweep - 1, -1, -1):
           x = trajectory[step]

           if n_B + n_b <= 1:
               break  # MRCA found

           # Compute rates for this time step
           bg_B = two_N * x
           bg_b = two_N * (1.0 - x)

           rate_coal_B = (n_B * (n_B - 1) / (2.0 * bg_B)) if (n_B >= 2 and bg_B > 0) else 0.0
           rate_coal_b = (n_b * (n_b - 1) / (2.0 * bg_b)) if (n_b >= 2 and bg_b > 0) else 0.0
           rate_mig_Bb = n_B * r_site * (1.0 - x) if n_B > 0 else 0.0
           rate_mig_bB = n_b * r_site * x if n_b > 0 else 0.0

           total_rate = rate_coal_B + rate_coal_b + rate_mig_Bb + rate_mig_bB

           if total_rate == 0:
               continue

           # Can an event happen within this generation?
           # The residual is in units of "rate * time". One generation = total_rate.
           while residual < total_rate:
               # An event happens within this step
               backward_gen = T_sweep - step
               coal_times_entry = None

               # Determine which event
               u = rng.random() * total_rate
               if u < rate_coal_B:
                   # Coalescence in B
                   n_B -= 1
                   coal_times_entry = backward_gen
               elif u < rate_coal_B + rate_coal_b:
                   # Coalescence in b
                   n_b -= 1
                   coal_times_entry = backward_gen
               elif u < rate_coal_B + rate_coal_b + rate_mig_Bb:
                   # Migration B -> b
                   n_B -= 1
                   n_b += 1
               else:
                   # Migration b -> B
                   n_b -= 1
                   n_B += 1

               if coal_times_entry is not None:
                   coal_times.append(coal_times_entry)

               if n_B + n_b <= 1:
                   break

               # Draw next event residual
               residual = rng.exponential(1.0)

           # Subtract this step's contribution from the residual
           residual -= total_rate

       # At the origin of the sweep, all B lineages coalesce
       # (they all trace to the single original mutant)
       if n_B > 1:
           for _ in range(n_B - 1):
               coal_times.append(T_sweep)
           n_B = 1

       return coal_times, n_B, n_b


Step 6: The Bottleneck Effect
================================

The most important consequence of the structured coalescent is the **bottleneck
in background** :math:`B`. Let us compute how severe it is.

At the moment the beneficial allele was at frequency :math:`x`, the effective
size of background :math:`B` was :math:`2N \cdot x`. The allele spends most of
its time near the boundary frequencies (where :math:`x` is close to 0 or 1),
because the logistic growth is slowest there.

The expected time that the allele spends between frequency :math:`x` and
:math:`x + dx` (during the sweep) is approximately:

.. math::

   dt = \frac{dx}{s \, x(1-x)}

The "effective population size" experienced by :math:`B` lineages during this
interval is :math:`2Nx`. So the total coalescent opportunity (integrated inverse
effective size) for the :math:`B` background over the sweep is:

.. math::

   \int_0^1 \frac{1}{2Nx} \cdot \frac{dx}{s \, x(1-x)}
   = \frac{1}{2Ns} \int_0^1 \frac{dx}{x^2(1-x)}

This integral diverges at :math:`x = 0`, reflecting the fact that when the allele
is at very low frequency, the :math:`B` background is tiny and coalescence is
nearly instantaneous. In practice, the integral is bounded by the initial
frequency :math:`1/(2N)`:

.. math::

   \int_{1/(2N)}^{1} \frac{dx}{x^2(1-x)} \approx 2N + \ln(2N)

So the total coalescent opportunity is approximately:

.. math::

   \frac{2N + \ln(2N)}{2Ns} \approx \frac{1}{s} + \frac{\ln(2N)}{2Ns}

For strong selection (:math:`2Ns \gg 1`), this simplifies to :math:`\approx 1/s`.
Compare this to the neutral coalescent, where two lineages coalesce in expected
time :math:`2N` generations (:math:`= 1` coalescent unit). The ratio is:

.. math::

   \frac{1/s}{2N} = \frac{1}{2Ns} = \frac{1}{\alpha}

For :math:`\alpha = 1000`, the bottleneck compresses the expected coalescence time
by a factor of 1000. This is why strong sweeps obliterate diversity.

.. code-block:: python

   def demonstrate_bottleneck(N, alpha, n_sample=20, n_reps=500, seed=42):
       """Compare coalescence times under a sweep vs. neutrality.

       Parameters
       ----------
       N : int
           Diploid population size.
       alpha : float
           Scaled selection coefficient 2Ns.
       n_sample : int
           Number of sampled lineages.
       n_reps : int
           Number of simulation replicates.
       seed : int
           Random seed.
       """
       rng = np.random.default_rng(seed)
       s = alpha / (2 * N)

       # Simulate under sweep: coalescence times within the sweep
       sweep_mean_times = []
       for _ in range(n_reps):
           traj = deterministic_trajectory(s, N)
           # Use r_site = 0 (perfectly linked) to see pure bottleneck effect
           coal_times, _, _ = structured_coalescent_sweep(
               traj, n_sample, r_site=0.0, N=N, rng=rng
           )
           if coal_times:
               sweep_mean_times.append(np.mean(coal_times))

       # Neutral comparison: expected total coalescence time for k lineages
       # is sum of 2N/choose(k,2) for k from n to 2
       neutral_expected = sum(2*N / (k*(k-1)/2) for k in range(n_sample, 1, -1))

       print(f"alpha = {alpha}, s = {s:.5f}, N = {N}")
       print(f"Sweep: mean coalescence time = "
             f"{np.mean(sweep_mean_times):.0f} generations")
       print(f"Neutral: expected total tree height = "
             f"{neutral_expected:.0f} generations")
       print(f"Compression factor: "
             f"{neutral_expected / np.mean(sweep_mean_times):.1f}x")

   # demonstrate_bottleneck(10_000, 1000)


Step 7: The Escape Probability
=================================

Not all lineages are trapped in the bottleneck. If the recombination rate between
the neutral locus and the selected site is high enough, a lineage can **escape**
from background :math:`B` to background :math:`b` before the bottleneck crushes
it.

The probability that a lineage escapes the sweep (recombines off :math:`B` before
being forced to coalesce) depends on the ratio :math:`r/s`. Intuitively:

- During one generation of the sweep, a lineage recombines with probability
  :math:`\sim r`.
- The sweep takes :math:`\sim 1/s` coalescent time units.
- So the expected number of recombination events per lineage during the sweep is
  :math:`\sim r/s` (in appropriate units, this is :math:`\rho / \alpha` per
  site).

The probability that a lineage does *not* escape (stays in :math:`B` through the
entire sweep) is approximately :math:`e^{-r/s}`, and the probability that it
*does* escape is :math:`1 - e^{-r/s}`.

More precisely, for a hard sweep, the probability that a lineage at recombination
distance :math:`r` from the selected site is **not** affected by the sweep is:

.. math::

   P(\text{escape}) \approx 1 - \exp\!\left(-\frac{r}{s} \cdot 2\ln(2N)\right)

This means:

- At :math:`r = 0` (the selected site itself): no escape. All lineages coalesce in
  the bottleneck.
- At :math:`r \gg s/\ln(2N)`: nearly all lineages escape. The sweep has no effect.
- At :math:`r \sim s/\ln(2N)`: the transition zone.

.. code-block:: python

   def escape_probability(r_site, s, N):
       """Approximate probability that a lineage escapes a hard sweep.

       Parameters
       ----------
       r_site : float
           Recombination rate between neutral locus and selected site.
       s : float
           Selection coefficient.
       N : int
           Diploid population size.

       Returns
       -------
       p_escape : float
           Probability of escaping the sweep via recombination.
       """
       return 1.0 - np.exp(-r_site / s * 2 * np.log(2 * N))

   # How far does the sweep's effect reach?
   N = 10_000
   s = 0.01  # 2Ns = 200
   r_values = np.logspace(-7, -2, 100)
   p_esc = [escape_probability(r, s, N) for r in r_values]

   print("Recombination distance at 50% escape probability:")
   r_half = s / (2 * np.log(2 * N))
   print(f"  r_50 = s / (2*ln(2N)) = {r_half:.2e}")
   print(f"  Physical distance (at r=1e-8/bp): {r_half / 1e-8:.0f} bp")


Step 8: Putting It Together -- The Full Sweep Simulation
==========================================================

Now we can assemble the complete structured coalescent for a single neutral locus
linked to a selected site:

.. code-block:: python

   def simulate_sweep_genealogy(n_sample, N, s, r_site, tau_gen=0,
                                 trajectory_mode='deterministic', rng=None):
       """Simulate a genealogy at a neutral locus linked to a selective sweep.

       The full algorithm:
       1. Generate the allele frequency trajectory.
       2. Run neutral coalescent from present to tau (time since fixation).
       3. Run structured coalescent through the sweep.
       4. Run neutral coalescent for remaining lineages before the sweep.

       Parameters
       ----------
       n_sample : int
           Number of sampled lineages.
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient.
       r_site : float
           Recombination rate between neutral locus and selected site.
       tau_gen : int
           Generations since the sweep completed (0 = just fixed).
       trajectory_mode : str
           'deterministic' or 'stochastic'.
       rng : np.random.Generator, optional
           Random number generator.

       Returns
       -------
       total_coal_times : list of float
           All coalescence times (in generations backward from present).
       """
       if rng is None:
           rng = np.random.default_rng()

       two_N = 2 * N
       n = n_sample
       total_coal_times = []

       # Phase 1: Neutral coalescent from present to tau
       t_current = 0
       while n > 1 and t_current < tau_gen:
           # Rate of coalescence: choose(n,2) / (2N)
           rate = n * (n - 1) / (2.0 * two_N)
           wait = rng.exponential(1.0 / rate)
           if t_current + wait > tau_gen:
               break  # No coalescence before the sweep phase starts
           t_current += wait
           n -= 1
           total_coal_times.append(t_current)

       if n <= 1:
           return total_coal_times

       # Phase 2: Generate trajectory and run structured coalescent
       if trajectory_mode == 'deterministic':
           traj = deterministic_trajectory(s, N)
       else:
           traj = stochastic_trajectory(s, N, rng=rng)

       coal_times_sweep, n_B, n_b = structured_coalescent_sweep(
           traj, n, r_site, N, rng=rng
       )

       # Offset sweep coalescence times by tau
       for ct in coal_times_sweep:
           total_coal_times.append(tau_gen + ct)

       # Phase 3: Neutral coalescent for remaining lineages
       n_remaining = n_B + n_b
       t_current = tau_gen + len(traj)
       while n_remaining > 1:
           rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
           wait = rng.exponential(1.0 / rate)
           t_current += wait
           n_remaining -= 1
           total_coal_times.append(t_current)

       return total_coal_times

   # Example: compare genealogies with and without a sweep
   N = 10_000
   s = 0.01
   n = 10

   rng = np.random.default_rng(42)

   # Neutral genealogy (no sweep)
   neutral_times = []
   n_temp = n
   t = 0
   while n_temp > 1:
       rate = n_temp * (n_temp - 1) / (2.0 * 2 * N)
       t += rng.exponential(1.0 / rate)
       n_temp -= 1
       neutral_times.append(t)

   # Sweep genealogy (r = 0, perfectly linked: maximum effect)
   sweep_times = simulate_sweep_genealogy(n, N, s, r_site=0.0, rng=rng)

   print(f"Neutral TMRCA:  {max(neutral_times):,.0f} generations")
   print(f"Sweep TMRCA:    {max(sweep_times):,.0f} generations")
   print(f"Neutral total:  {sum(neutral_times):,.0f} gen")
   print(f"Sweep total:    {sum(sweep_times):,.0f} gen")

The structured coalescent is the mathematical engine that translates the allele
frequency trajectory into a distorted genealogy. In the next chapter, we explore
the different *types* of sweeps -- hard, soft, and partial -- and see how each
produces distinctive genealogical signatures.
