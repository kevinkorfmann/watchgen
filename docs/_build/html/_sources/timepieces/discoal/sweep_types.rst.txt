.. _discoal_sweep_types:

======================================
Hard, Soft, and Partial Sweeps
======================================

   *The complications of the mechanism: three ways a beneficial allele reshapes the genealogy.*

Not all selective sweeps are alike. The classic textbook picture -- a single new
mutation rising from one copy to fixation -- is only one variant. In this chapter
we explore the three sweep types that discoal can simulate and show how each
produces a distinct genealogical signature.

Think of these as complications on the watch face: the basic mechanism (structured
coalescent driven by a trajectory) is the same, but the *boundary conditions*
differ. A hard sweep starts from a single copy; a soft sweep starts from multiple
copies; a partial sweep stops before fixation. Each starting or stopping condition
reshapes the genealogy in characteristic ways.


.. note::

   **Prerequisites.** This chapter builds directly on:

   - :ref:`discoal_allele_trajectory` -- trajectory generation
   - :ref:`discoal_structured_coalescent` -- the event rates and bottleneck


Hard Sweeps
============

A **hard sweep** is the classical selective sweep:

- The beneficial mutation arises as a **single new copy**: :math:`x_0 = 1/(2N)`.
- It rises to **fixation**: :math:`x = 1`.
- All copies of the beneficial allele in the present trace back to this single
  ancestral mutation.

Going backward through the structured coalescent, all :math:`n_B` lineages in the
beneficial background are forced to coalesce at the origin of the sweep (when
:math:`x = 1/(2N)`), because they all descend from one chromosome. Any lineages
that escaped to background :math:`b` via recombination are spared, but lineages
that stayed in :math:`B` share a very recent common ancestor.

**Genealogical signature of a hard sweep:**

- **Star-like genealogy.** Near the selected site (:math:`r/s \ll 1`), most
  lineages coalesce simultaneously at the sweep origin, producing a star-shaped
  tree with one internal node.
- **Reduced diversity.** The compressed genealogy means fewer mutations, hence
  less genetic variation.
- **Excess of rare alleles.** After the sweep, new mutations arise on the long
  terminal branches of the star tree, producing many low-frequency variants.
  This skews the site frequency spectrum (SFS) toward rare alleles (negative
  Tajima's D).
- **Extended haplotype homozygosity (EHH).** Near the selected site, all
  haplotypes are identical (or nearly so), because they all trace to the same
  recent ancestor. EHH decays with recombination distance.

.. code-block:: python

   import numpy as np

   def hard_sweep_genealogy(n_sample, N, s, r_site, rng=None):
       """Simulate a hard sweep genealogy at a linked neutral locus.

       A hard sweep starts from frequency 1/(2N) and reaches fixation.

       Parameters
       ----------
       n_sample : int
           Number of sampled lineages.
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient.
       r_site : float
           Recombination rate to the selected site.
       rng : np.random.Generator, optional

       Returns
       -------
       coal_times : list of float
           Coalescence times (generations backward from present).
       n_escaped : int
           Number of lineages that escaped to background b.
       """
       if rng is None:
           rng = np.random.default_rng(42)

       traj = deterministic_trajectory(s, N, x0=1.0 / (2 * N))
       coal_times, n_B, n_b = structured_coalescent_sweep(
           traj, n_sample, r_site, N, rng=rng
       )
       return coal_times, n_b

   # Hard sweep: all lineages coalesce in the bottleneck
   N = 10_000
   s = 0.01
   coal_times, n_escaped = hard_sweep_genealogy(20, N, s, r_site=0.0)
   print(f"Hard sweep (r=0): {len(coal_times)} coalescences, "
         f"{n_escaped} lineages escaped")

   # Hard sweep with some recombination: some lineages escape
   coal_times2, n_escaped2 = hard_sweep_genealogy(20, N, s, r_site=1e-5)
   print(f"Hard sweep (r=1e-5): {len(coal_times2)} coalescences, "
         f"{n_escaped2} lineages escaped")


Soft Sweeps from Standing Variation
=====================================

A **soft sweep from standing variation** occurs when a previously neutral (or
mildly deleterious) allele that already exists at frequency :math:`x_0 > 1/(2N)`
becomes beneficial due to an environmental change. Because multiple copies of the
allele already exist on *different* haplotypic backgrounds, the sweep starts from
a diverse base.

The key difference from a hard sweep:

- At the onset of selection (going backward: at the end of the sweep phase), the
  allele is at frequency :math:`x_0`, not :math:`1/(2N)`.
- The :math:`B` lineages do **not** all coalesce at the sweep origin. Instead,
  they need to find common ancestors through the **neutral drift phase** before
  selection began. Multiple ancestral lineages persist.

In discoal, the trajectory has two phases (reading forward):

1. **Neutral drift phase** (before selection): the allele drifts from :math:`1/(2N)`
   to :math:`x_0` under neutrality.
2. **Selection phase**: the allele rises from :math:`x_0` to fixation under
   selection.

The structured coalescent operates only during the selection phase. At the end
of the selection phase (going backward), the remaining :math:`B` lineages
enter the neutral drift phase with :math:`x = x_0`, and their genealogy follows
a standard neutral coalescent in a population of size :math:`2N \cdot x_0`.

**Genealogical signature of a soft sweep:**

- **Multiple ancestral haplotypes.** Unlike the star topology of a hard sweep,
  a soft sweep preserves diversity from the standing variation phase. The genealogy
  near the selected site may have several deep lineages.
- **Moderate diversity reduction.** Diversity drops, but not as dramatically as in
  a hard sweep.
- **Less negative Tajima's D.** The SFS shift is weaker because the genealogy
  retains some ancient structure.
- **Multiple haplotype clusters.** Extended haplotype patterns show multiple
  distinct "swept" haplotypes, not just one.

.. code-block:: python

   def soft_sweep_standing_variation(n_sample, N, s, x0, r_site, rng=None):
       """Simulate a soft sweep from standing variation.

       The trajectory runs from x0 to fixation under selection, then
       the remaining lineages coalesce under neutral drift.

       Parameters
       ----------
       n_sample : int
           Number of sampled lineages.
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient.
       x0 : float
           Frequency of the allele at the onset of selection.
       r_site : float
           Recombination rate to the selected site.
       rng : np.random.Generator, optional

       Returns
       -------
       coal_times : list of float
           All coalescence times.
       n_surviving_B : int
           Number of distinct B lineages that survived the sweep
           (these still need to coalesce under neutral drift from x0).
       """
       if rng is None:
           rng = np.random.default_rng(42)

       # Step 1: trajectory from x0 to fixation (under selection)
       traj = deterministic_trajectory(s, N, x0=x0)

       # Step 2: structured coalescent through the sweep
       coal_times_sweep, n_B, n_b = structured_coalescent_sweep(
           traj, n_sample, r_site, N, rng=rng
       )

       # Step 3: the n_B surviving lineages must coalesce under neutral drift
       # in the ancestral population. Before selection started, the allele
       # was at frequency x0, so the B lineages are in a subpopulation of
       # size 2N * x0 that drifts neutrally.
       #
       # For simplicity, we model this as a neutral coalescent in a
       # population of size 2N (the full population, since before selection
       # the allele was neutral and its carriers were part of the panmictic
       # population).
       coal_times_all = list(coal_times_sweep)
       sweep_duration = len(traj)

       # Remaining lineages (from both B and b) coalesce neutrally
       n_remaining = n_B + n_b
       two_N = 2 * N
       t_offset = sweep_duration
       while n_remaining > 1:
           rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
           wait = rng.exponential(1.0 / rate)
           t_offset += wait
           n_remaining -= 1
           coal_times_all.append(t_offset)

       return coal_times_all, n_B

   # Compare hard vs soft sweep at the same selection strength
   N = 10_000
   s = 0.01
   rng = np.random.default_rng(42)

   hard_times, _ = hard_sweep_genealogy(20, N, s, 0.0, rng=rng)
   soft_times, n_surv = soft_sweep_standing_variation(20, N, s, x0=0.1, r_site=0.0, rng=rng)

   print(f"Hard sweep (x0=1/2N): TMRCA = {max(hard_times):,.0f} gen")
   print(f"Soft sweep (x0=0.1):  TMRCA = {max(soft_times):,.0f} gen, "
         f"{n_surv} ancestral B lineages survived sweep")


Soft Sweeps from Recurrent Mutation
=====================================

A second type of soft sweep occurs when the same beneficial mutation arises
**independently multiple times** by recurrent mutation at rate :math:`\mu_a`
(per generation). Each independent origin is on a different haplotypic background,
so the swept alleles carry diverse linked neutral variation.

This model is appropriate when:

- The mutation rate is not negligible (:math:`\mu_a > 0`)
- The allele is strongly beneficial (:math:`s` is large)
- The time from origin to fixation is long enough for multiple mutations to
  arise and establish

The genealogical signature is similar to the standing variation case: multiple
haplotype clusters near the selected site, with each cluster tracing to an
independent origin of the beneficial allele.

In discoal's implementation (the ``-uA`` flag), the number of independent origins
is drawn from the appropriate distribution given the trajectory and mutation rate.

.. code-block:: python

   def expected_independent_origins(s, N, mu_a):
       """Expected number of independent origins of a beneficial allele
       that contribute to a soft sweep from recurrent mutation.

       From Pennings & Hermisson (2006): the expected number of
       independent origins that survive drift and contribute to
       fixation is approximately:

           E[K] = 2 * N * mu_a * log(2Ns) / s

       Parameters
       ----------
       s : float
           Selection coefficient.
       N : int
           Diploid effective population size.
       mu_a : float
           Mutation rate to the beneficial allele (per generation).

       Returns
       -------
       E_K : float
           Expected number of independent surviving origins.
       """
       two_N_s = 2 * N * s
       return 2 * N * mu_a * np.log(two_N_s) / s

   # When do we expect hard vs soft sweeps?
   N = 10_000
   s = 0.01
   for mu_a in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
       E_K = expected_independent_origins(s, N, mu_a)
       sweep_type = "hard" if E_K < 1 else "soft"
       print(f"mu_a = {mu_a:.0e}: E[K] = {E_K:.2f} origins --> {sweep_type} sweep")


Partial Sweeps
===============

A **partial sweep** occurs when the beneficial allele has not yet reached fixation.
At the present time, the allele is at frequency :math:`c < 1`. The population is
*currently* divided into carriers (:math:`B`) and non-carriers (:math:`b`), and
the structured coalescent is active from the present backward through the sweep's
history.

This differs from a completed sweep in two ways:

1. **Initial assignment of lineages.** At the present time, each sampled lineage
   is on background :math:`B` with probability :math:`c` or background :math:`b`
   with probability :math:`1 - c`.
2. **The trajectory starts at** :math:`c`, **not** :math:`1`. The structured
   coalescent runs from :math:`x = c` backward to the origin (:math:`x = 1/(2N)`).

**Genealogical signature of a partial sweep:**

- **Current population structure.** The sample is divided into two groups
  (carriers and non-carriers) with distinct recent genealogical histories.
- **Intermediate diversity reduction.** Carriers show reduced diversity (from the
  ongoing bottleneck in :math:`B`), but not as much as after a completed sweep.
  Non-carriers show normal diversity.
- **High** :math:`F_{ST}` **between carriers and non-carriers.** The two groups
  have been partially isolated by the sweep.

.. code-block:: python

   def partial_sweep_genealogy(n_sample, N, s, c, r_site, rng=None):
       """Simulate a genealogy under a partial sweep.

       The beneficial allele is currently at frequency c < 1.

       Parameters
       ----------
       n_sample : int
           Total number of sampled lineages.
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient.
       c : float
           Current frequency of the beneficial allele (0 < c < 1).
       r_site : float
           Recombination rate to the selected site.
       rng : np.random.Generator, optional

       Returns
       -------
       coal_times : list of float
           All coalescence times.
       initial_B : int
           Number of lineages initially in background B.
       initial_b : int
           Number of lineages initially in background b.
       """
       if rng is None:
           rng = np.random.default_rng(42)

       # Assign each lineage to B or b based on current frequency c
       assignments = rng.random(n_sample) < c
       n_B = int(np.sum(assignments))
       n_b = n_sample - n_B

       # Generate trajectory from 1/(2N) to c
       traj = deterministic_trajectory(s, N, x0=1.0 / (2 * N))
       # Truncate trajectory at frequency c
       idx = np.searchsorted(traj, c)
       traj = traj[:idx + 1]
       if len(traj) > 0 and traj[-1] < c:
           traj = np.append(traj, c)

       # Run structured coalescent backward through this partial trajectory
       # (We start with n_B and n_b as given, not all in B)
       coal_times_sweep, n_B_final, n_b_final = structured_coalescent_sweep(
           traj, n_B + n_b, r_site, N, rng=rng
       )

       # Remaining lineages coalesce neutrally
       coal_times_all = list(coal_times_sweep)
       n_remaining = n_B_final + n_b_final
       two_N = 2 * N
       t_offset = len(traj)
       while n_remaining > 1:
           rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
           wait = rng.exponential(1.0 / rate)
           t_offset += wait
           n_remaining -= 1
           coal_times_all.append(t_offset)

       return coal_times_all, n_B, n_b

   # Partial sweep: allele at 60% frequency
   N = 10_000
   s = 0.01
   coal_times, n_B, n_b = partial_sweep_genealogy(20, N, s, c=0.6, r_site=0.0)
   print(f"Partial sweep (c=0.6): {n_B} carriers, {n_b} non-carriers")
   print(f"  TMRCA = {max(coal_times):,.0f} generations")


Comparing the Signatures
==========================

The three sweep types produce distinguishable patterns in the data:

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - Hard sweep
     - Soft sweep
     - Partial sweep
   * - Diversity near site
     - Strongly reduced
     - Moderately reduced
     - Reduced in carriers only
   * - Tajima's D
     - Strongly negative
     - Mildly negative
     - Negative in carriers
   * - Haplotype structure
     - Single swept haplotype
     - Multiple swept haplotypes
     - Two clusters (B and b)
   * - EHH pattern
     - Long, symmetric
     - Short, fragmented
     - Asymmetric
   * - SFS near site
     - Excess rare alleles
     - Less extreme shift
     - Intermediate
   * - Genealogy near site
     - Star-like (single deep node)
     - Bush-like (multiple deep nodes)
     - Split (B and b clades)

.. code-block:: python

   def compare_sweep_types(N, s, n_sample=50, n_reps=200, seed=42):
       """Compare diversity under hard, soft, and partial sweeps.

       Measures the total tree length (proportional to expected number
       of segregating sites) for each sweep type.

       Parameters
       ----------
       N, s, n_sample, n_reps : as above
       seed : random seed
       """
       rng = np.random.default_rng(seed)

       hard_lengths = []
       soft_lengths = []
       partial_lengths = []
       neutral_lengths = []

       for _ in range(n_reps):
           # Hard sweep (perfectly linked)
           hard_ct, _ = hard_sweep_genealogy(n_sample, N, s, 0.0, rng=rng)
           if hard_ct:
               hard_lengths.append(sum(hard_ct))

           # Soft sweep (x0 = 0.05)
           soft_ct, _ = soft_sweep_standing_variation(
               n_sample, N, s, x0=0.05, r_site=0.0, rng=rng)
           if soft_ct:
               soft_lengths.append(sum(soft_ct))

           # Neutral coalescent
           n_temp = n_sample
           t = 0
           total = 0
           while n_temp > 1:
               rate = n_temp * (n_temp - 1) / (2.0 * 2 * N)
               wait = rng.exponential(1.0 / rate)
               t += wait
               total += t
               n_temp -= 1
           neutral_lengths.append(total)

       print(f"Mean total tree length (proportional to expected S):")
       print(f"  Neutral:      {np.mean(neutral_lengths):>12,.0f} gen")
       if hard_lengths:
           print(f"  Hard sweep:   {np.mean(hard_lengths):>12,.0f} gen "
                 f"({np.mean(hard_lengths)/np.mean(neutral_lengths)*100:.1f}% of neutral)")
       if soft_lengths:
           print(f"  Soft sweep:   {np.mean(soft_lengths):>12,.0f} gen "
                 f"({np.mean(soft_lengths)/np.mean(neutral_lengths)*100:.1f}% of neutral)")

   # compare_sweep_types(10_000, 0.01)


The Genealogical Intuition
============================

To build intuition, consider what the genealogy looks like at the selected site
itself (:math:`r = 0`):

**Hard sweep:** All :math:`n` lineages trace to a single ancestor at the time
the beneficial mutation originated. The tree is a star: one root node with
:math:`n` terminal branches of approximately equal length (the sweep duration).
There is essentially zero diversity below the root.

.. code-block:: text

   Hard sweep genealogy (r = 0):

         *         <-- origin of beneficial mutation
        /|\
       / | \
      /  |  \
     /   |   \
    1  2  3  4    <-- sampled lineages (all on one branch)

    Star topology: all coalesce at one point.

**Soft sweep from standing variation (** :math:`x_0 = 0.1` **):** The :math:`n`
lineages trace to several ancestors at the onset of selection. The tree has
multiple deep lineages that connect back through the neutral standing variation
phase. Some structure survives.

.. code-block:: text

   Soft sweep genealogy (r = 0, x0 = 0.1):

       *           <-- neutral coalescent (pre-selection)
      / \
     *   *         <-- multiple lineages survive the sweep
    /|   |\
   / |   | \
  1  2   3  4     <-- sampled lineages

   Multiple ancestral lineages: diversity is partially preserved.

**Partial sweep (** :math:`c = 0.5` **):** The genealogy splits into two clades:
one for lineages on background :math:`B` (carriers) and one for background
:math:`b` (non-carriers). The :math:`B` clade shows reduced diversity (from the
ongoing sweep), while the :math:`b` clade retains normal diversity.

.. code-block:: text

   Partial sweep genealogy (c = 0.5):

           *                 <-- MRCA (deep in neutral past)
          / \
         /   \
        *     *              <-- split between B and b
       /|     |\
      / |     | \
     1  2     3  4
   (B clade) (b clade)

   Two distinct clades: carriers and non-carriers diverge.


What Comes Next
=================

We have now built every gear of the discoal mechanism: the trajectory (mainspring),
the structured coalescent (escapement), and the sweep variants (complications).

In the final chapter, we compare this machinery directly to msprime's selection
implementation, translate the core ideas into Python code that uses the same
framework as msprime, and identify what distinguishes the two approaches.
