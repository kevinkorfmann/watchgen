.. _discoal_msprime_comparison:

==========================================
discoal and msprime: Two Takes on Sweeps
==========================================

   *Two watchmakers build the same mechanism -- one from scratch, one by extending an existing movement.*

discoal (Kern & Schrider, 2016) and msprime 1.0 (Baumdicker et al., 2022) both
simulate selective sweeps using the same mathematical framework: generate an allele
frequency trajectory, then run the structured coalescent conditioned on that
trajectory. They are implementations of the same theory -- but the engineering
differs profoundly.

This chapter puts the two side by side: the shared mathematical core, the API
differences, the translation of discoal's C code into Python using msprime's
internals, and the features that distinguish each tool. If the previous chapters
built discoal's mechanism from scratch, this chapter shows how the same mechanism
was grafted onto msprime's existing neutral coalescent.


.. note::

   **Prerequisites.** This chapter assumes you have worked through:

   - :ref:`discoal_structured_coalescent` -- the two-class structured coalescent
   - :ref:`hudson_algorithm` -- msprime's simulation loop
   - Familiarity with both tools' APIs is helpful but not required


The Shared Mathematical Core
==============================

Both tools implement the same algorithm at the mathematical level:

**Step 1:** Generate an allele frequency trajectory :math:`x(t)` for the
beneficial allele.

**Step 2:** Run the structured coalescent backward in time. During the sweep
phase, lineages are on background :math:`B` (beneficial) or :math:`b`
(wild-type). Coalescence within each background scales inversely with background
size. Recombination between the neutral locus and the selected site acts as
migration between backgrounds.

The event rates are identical in both implementations:

.. math::

   \lambda_B = \frac{n_B(n_B-1)}{2 \cdot 2Nx}, \qquad
   \lambda_b = \frac{n_b(n_b-1)}{2 \cdot 2N(1-x)}

.. math::

   M_{B \to b} = n_B \cdot r \cdot (1-x), \qquad
   M_{b \to B} = n_b \cdot r \cdot x

The physics is the same. The engineering is where they diverge.


Engineering Differences
========================

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - discoal
     - msprime 1.0
   * - Language
     - C (standalone binary)
     - C with Python API
   * - Output format
     - ms-compatible text (haplotype matrix)
     - Tree sequence (tskit)
   * - Parameter convention
     - Scaled: :math:`\theta = 4N\mu`, :math:`\rho = 4Nr`, :math:`\alpha = 2Ns`
     - Raw: :math:`\mu`, :math:`r`, :math:`s` (per bp per gen), explicit :math:`N`
   * - Trajectory generation
     - Deterministic (logistic) or stochastic (conditioned jump process)
     - Stochastic only (conditioned jump process)
   * - Internal representation
     - Array of haplotypes, :math:`O(nS)` memory
     - Tree sequence, :math:`O(n + S)` memory
   * - Sweep position
     - Anywhere along the sequence (``-x`` flag)
     - Midpoint of sequence (can be customized)
   * - Multiple sweeps
     - Not supported
     - Possible via chaining sweep models
   * - Soft sweeps
     - Standing variation (``-f``), recurrent mutation (``-uA``), neutral
       fixation (``-wn``)
     - Hard sweep only (as of msprime 1.0)
   * - Partial sweeps
     - Yes (``-c`` flag)
     - Not directly supported
   * - Demographics
     - Population size changes (``-en``, ``-eN``)
     - Full demographic models (splits, admixture, migration)
   * - Gene conversion
     - Yes (``-gc`` flag)
     - Not during sweeps
   * - Performance
     - Good for moderate :math:`n`
     - Orders of magnitude faster for large :math:`n` due to tree sequences


Parameter Translation
======================

Converting between discoal and msprime parameters:

.. code-block:: python

   def discoal_to_msprime(theta, rho, alpha, n, L, N):
       """Convert discoal's scaled parameters to msprime's raw parameters.

       Parameters
       ----------
       theta : float
           Population-scaled mutation rate (4*N*mu*L).
       rho : float
           Population-scaled recombination rate (4*N*r*L).
       alpha : float
           Population-scaled selection coefficient (2*N*s).
       n : int
           Sample size.
       L : int
           Sequence length in bp.
       N : int
           Diploid effective population size.

       Returns
       -------
       params : dict
           Dictionary of msprime parameters.
       """
       mu = theta / (4 * N * L)
       r = rho / (4 * N * L)
       s = alpha / (2 * N)

       return {
           'samples': n,
           'sequence_length': L,
           'mutation_rate': mu,
           'recombination_rate': r,
           'population_size': N,
           'selection_coefficient': s,
       }

   def msprime_to_discoal(n, L, mu, r, s, N):
       """Convert msprime's raw parameters to discoal's scaled parameters.

       Parameters
       ----------
       n : int
           Sample size.
       L : int
           Sequence length in bp.
       mu : float
           Mutation rate per bp per generation.
       r : float
           Recombination rate per bp per generation.
       s : float
           Selection coefficient.
       N : int
           Diploid effective population size.

       Returns
       -------
       params : dict
           Dictionary of discoal command-line parameters.
       """
       theta = 4 * N * mu * L
       rho = 4 * N * r * L
       alpha = 2 * N * s

       return {
           'n': n,
           'L': L,
           'theta': theta,
           'rho': rho,
           'alpha': alpha,
       }

   # Example: a typical human-like simulation
   N = 10_000
   L = 100_000
   mu = 1.25e-8
   r = 1e-8
   s = 0.01

   discoal_params = msprime_to_discoal(100, L, mu, r, s, N)
   print("discoal command:")
   print(f"  discoal {discoal_params['n']} 1 {discoal_params['L']} "
         f"-t {discoal_params['theta']:.1f} "
         f"-r {discoal_params['rho']:.1f} "
         f"-ws 0 -a {discoal_params['alpha']:.0f} -x 0.5")

   msprime_params = discoal_to_msprime(
       discoal_params['theta'], discoal_params['rho'],
       discoal_params['alpha'], 100, L, N
   )
   print(f"\nmsprime equivalent:")
   print(f"  samples={msprime_params['samples']}")
   print(f"  sequence_length={msprime_params['sequence_length']}")
   print(f"  mutation_rate={msprime_params['mutation_rate']:.2e}")
   print(f"  recombination_rate={msprime_params['recombination_rate']:.2e}")
   print(f"  population_size={msprime_params['population_size']}")
   print(f"  s={msprime_params['selection_coefficient']}")


A Python Translation of the discoal Algorithm
===============================================

Here is a complete, self-contained Python implementation of discoal's core
algorithm. This is a faithful translation of the C code's logic into readable
Python, covering the main simulation loop for a hard sweep at a single locus.

.. code-block:: python

   import numpy as np
   from dataclasses import dataclass, field

   @dataclass
   class SweepState:
       """State of the structured coalescent during a sweep."""
       n_B: int = 0          # lineages in beneficial background
       n_b: int = 0          # lineages in wild-type background
       coal_times_B: list = field(default_factory=list)
       coal_times_b: list = field(default_factory=list)
       migrations_Bb: int = 0
       migrations_bB: int = 0

   def discoal_core(n, N, s, r_per_site, L, tau_4N=0.0,
                     trajectory_mode='deterministic', seed=42):
       """A Python translation of discoal's core simulation algorithm.

       Simulates a selective sweep at position L/2, with neutral sites
       along a chromosome of length L.

       Parameters
       ----------
       n : int
           Sample size (haploid).
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient (genic).
       r_per_site : float
           Recombination rate per site per generation.
       L : int
           Sequence length (number of sites).
       tau_4N : float
           Time since fixation in units of 4N generations.
       trajectory_mode : str
           'deterministic' or 'stochastic'.
       seed : int
           Random seed.

       Returns
       -------
       result : dict
           Contains 'diversity' (pi), 'n_segregating', 'tajimas_D_sign',
           and diagnostic information about the sweep.
       """
       rng = np.random.default_rng(seed)
       two_N = 2 * N
       tau_gen = int(tau_4N * 4 * N)

       # --- Step 1: Generate trajectory ---
       x0 = 1.0 / two_N
       if trajectory_mode == 'deterministic':
           traj = []
           x = x0
           while x < 1.0 - 1.0 / two_N:
               traj.append(x)
               # One generation of logistic growth
               x = x * (1 + s) / (1 + s * x)
               # Clip to [0, 1]
               x = min(x, 1.0)
           traj.append(1.0)
           traj = np.array(traj)
       else:
           traj = stochastic_trajectory(s, N, rng=rng)

       T_sweep = len(traj)

       # --- Step 2: Simulate at multiple neutral sites ---
       # For each site, compute its recombination distance to the selected
       # site at position L/2, then run the structured coalescent.
       selected_pos = L // 2
       site_tmrcas = []

       # Sample a few representative sites
       test_positions = [selected_pos,  # at the selected site
                         selected_pos + L // 10,  # nearby
                         selected_pos + L // 4,   # intermediate
                         0]                        # far away

       for pos in test_positions:
           r_site = r_per_site * abs(pos - selected_pos)

           state = SweepState(n_B=n, n_b=0)
           coal_times = []

           # Phase A: Neutral coalescent (present to tau)
           n_current = n
           t = 0
           while n_current > 1 and t < tau_gen:
               rate = n_current * (n_current - 1) / (2.0 * two_N)
               wait = rng.exponential(1.0 / rate)
               if t + wait > tau_gen:
                   break
               t += wait
               n_current -= 1
               coal_times.append(t)

           # Phase B: Structured coalescent through sweep
           state.n_B = n_current
           state.n_b = 0

           for step in range(T_sweep - 1, -1, -1):
               x = traj[step]
               if state.n_B + state.n_b <= 1:
                   break

               bg_B = max(two_N * x, 1.0)
               bg_b = max(two_N * (1.0 - x), 1.0)

               # Compute rates
               rate_cB = state.n_B * (state.n_B - 1) / (2.0 * bg_B) if state.n_B >= 2 else 0
               rate_cb = state.n_b * (state.n_b - 1) / (2.0 * bg_b) if state.n_b >= 2 else 0
               rate_mBb = state.n_B * r_site * (1.0 - x) if state.n_B > 0 else 0
               rate_mbB = state.n_b * r_site * x if state.n_b > 0 else 0
               total = rate_cB + rate_cb + rate_mBb + rate_mbB

               if total == 0:
                   continue

               # Check if event happens in this generation
               if rng.exponential(1.0 / total) < 1.0:
                   u = rng.random() * total
                   t_event = tau_gen + (T_sweep - step)
                   if u < rate_cB:
                       state.n_B -= 1
                       coal_times.append(t_event)
                   elif u < rate_cB + rate_cb:
                       state.n_b -= 1
                       coal_times.append(t_event)
                   elif u < rate_cB + rate_cb + rate_mBb:
                       state.n_B -= 1
                       state.n_b += 1
                       state.migrations_Bb += 1
                   else:
                       state.n_b -= 1
                       state.n_B += 1
                       state.migrations_bB += 1

           # Forced coalescence at sweep origin for remaining B lineages
           if state.n_B > 1:
               for _ in range(state.n_B - 1):
                   coal_times.append(tau_gen + T_sweep)
               state.n_B = 1

           # Phase C: Neutral coalescent before sweep
           n_remaining = state.n_B + state.n_b
           t = tau_gen + T_sweep
           while n_remaining > 1:
               rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
               wait = rng.exponential(1.0 / rate)
               t += wait
               n_remaining -= 1
               coal_times.append(t)

           tmrca = max(coal_times) if coal_times else 0
           site_tmrcas.append((pos, abs(pos - selected_pos), tmrca, state))

       # Report
       result = {
           'selected_position': selected_pos,
           'sweep_duration_gen': T_sweep,
           'sites': []
       }
       for pos, dist, tmrca, state in site_tmrcas:
           result['sites'].append({
               'position': pos,
               'distance_to_selected': dist,
               'r_to_selected': r_per_site * dist,
               'tmrca_gen': tmrca,
               'tmrca_coalescent_units': tmrca / two_N,
               'migrations_Bb': state.migrations_Bb,
               'migrations_bB': state.migrations_bB,
           })
       return result

   # Run the translation
   result = discoal_core(
       n=20, N=10_000, s=0.01, r_per_site=1e-8, L=100_000, seed=42
   )
   print(f"Sweep duration: {result['sweep_duration_gen']} generations\n")
   print(f"{'Position':>10} {'Distance':>10} {'r_site':>12} "
         f"{'TMRCA (gen)':>12} {'TMRCA (coal)':>14} {'Mig B->b':>10}")
   print("-" * 75)
   for site in result['sites']:
       print(f"{site['position']:>10} {site['distance_to_selected']:>10} "
             f"{site['r_to_selected']:>12.2e} {site['tmrca_gen']:>12,.0f} "
             f"{site['tmrca_coalescent_units']:>14.4f} "
             f"{site['migrations_Bb']:>10}")


The msprime Way: Using SweepGenicSelection
=============================================

msprime 1.0 provides the same functionality through a different API. Instead of
a standalone binary with command-line flags, msprime integrates sweeps into its
composable model framework.

.. code-block:: python

   # How you would run the same simulation in msprime:
   #
   # import msprime
   #
   # sweep = msprime.SweepGenicSelection(
   #     position=50_000,           # selected site position
   #     start_frequency=1/(2*N),   # start from single copy
   #     end_frequency=1 - 1/(2*N), # sweep to near-fixation
   #     s=0.01,                    # selection coefficient
   #     dt=1e-6,                   # trajectory time step
   # )
   #
   # ts = msprime.sim_ancestry(
   #     samples=20,
   #     sequence_length=100_000,
   #     recombination_rate=1e-8,
   #     population_size=10_000,
   #     model=[
   #         sweep,                    # the sweep phase
   #         msprime.StandardCoalescent(),  # neutral before and after
   #     ],
   # )
   #
   # # Add mutations
   # ts = msprime.sim_mutations(ts, rate=1.25e-8)
   #
   # # Analyze diversity around the sweep
   # diversity = ts.diversity(windows=np.linspace(0, 100_000, 51))

The key differences in the msprime API:

1. **Model composition.** The sweep is one "phase" in a list of models. msprime
   automatically transitions between phases.
2. **Tree sequence output.** The result is a tree sequence, not a haplotype matrix.
   This enables efficient downstream analysis.
3. **Raw parameters.** You specify :math:`s`, :math:`N`, :math:`r` directly, not
   :math:`\alpha`, :math:`\rho`, :math:`\theta`.
4. **No soft sweeps.** msprime's ``SweepGenicSelection`` currently supports only
   hard sweeps (from :math:`1/(2N)` to near-fixation).


What discoal Can Do That msprime Cannot
==========================================

As of msprime 1.0, discoal offers several features that msprime's sweep model
does not:

1. **Soft sweeps from standing variation** (``-f`` flag). Start the sweep from
   frequency :math:`x_0 > 1/(2N)`, modeling the case where a previously neutral
   allele becomes beneficial.

2. **Soft sweeps from recurrent mutation** (``-uA`` flag). Multiple independent
   origins of the beneficial allele, each on a different haplotypic background.

3. **Partial sweeps** (``-c`` flag). The beneficial allele is at intermediate
   frequency in the present.

4. **Neutral fixation trajectories** (``-wn`` flag). Simulate the trajectory of
   an allele that fixes by drift alone (no selection). Useful as a null model.

5. **Gene conversion during sweeps** (``-gc`` flag). Gene conversion
   (non-crossover recombination) can also move lineages between backgrounds.

6. **Deterministic trajectories** (``-wd`` flag). The logistic trajectory, which
   is faster and may be preferable for very strong selection.


What msprime Does Better
==========================

1. **Performance.** For large sample sizes (:math:`n > 1000`), msprime is orders
   of magnitude faster because it uses tree sequences internally. discoal stores
   an :math:`n \times S` haplotype matrix, which becomes prohibitive for large
   :math:`n`.

2. **Tree sequence output.** The compact, queryable tree sequence format enables
   efficient computation of any statistic (diversity, SFS, LD, :math:`F_{ST}`,
   etc.) without re-simulating.

3. **Composable models.** Sweeps can be combined with arbitrary demographic
   models (population splits, admixture, migration) and chained with neutral
   phases in a single simulation.

4. **Ecosystem integration.** Output plugs directly into tskit, tsinfer, tsdate,
   and the broader tree sequence toolkit.

5. **Active development.** msprime is actively maintained and extended. Future
   versions may add soft sweep support and other features.


The Architecture Comparison
=============================

At the deepest level, the difference is **architectural**:

.. code-block:: text

   discoal architecture:
   =====================
   1. Generate trajectory  ───> array of floats
   2. Structured coalescent ───> modify haplotype matrix in place
   3. Neutral coalescent    ───> modify haplotype matrix in place
   4. Add mutations         ───> fill haplotype matrix
   5. Output: n x S matrix  ───> print to stdout

   msprime architecture:
   =====================
   1. Initialize tree sequence tables (nodes, edges, sites, mutations)
   2. Event loop:
      - Neutral phase: Hudson's algorithm with segments and Fenwick tree
      - Sweep phase: structured coalescent with two labels
        (same math, different data structure)
      - Neutral phase: back to Hudson's algorithm
   3. Output: tree sequence ───> tskit object

   Key difference: msprime never materializes the haplotype matrix.
   It builds the *genealogy* directly, and mutations are added afterward.

discoal's approach is simpler to understand (and to implement from scratch), but
msprime's approach is more powerful: the tree sequence representation allows
:math:`O(n)` storage instead of :math:`O(nS)`, and genealogical statistics can
be computed in :math:`O(n)` time instead of :math:`O(nS)`.


Building It Yourself: The Minimal Sweep Simulator
====================================================

To consolidate everything from this Timepiece, here is a minimal but complete
sweep simulator in Python. It generates a trajectory, runs the structured
coalescent, and computes the expected diversity reduction around the selected site.

.. code-block:: python

   def minimal_discoal(n, N, s, r_per_site, L, n_sites=50, seed=42):
       """A minimal discoal-like simulator that computes the diversity
       profile around a selective sweep.

       Parameters
       ----------
       n : int
           Sample size.
       N : int
           Diploid effective population size.
       s : float
           Selection coefficient.
       r_per_site : float
           Recombination rate per site per generation.
       L : int
           Sequence length.
       n_sites : int
           Number of sites to evaluate along the sequence.
       seed : int
           Random seed.

       Returns
       -------
       positions : ndarray
           Positions along the sequence.
       relative_diversity : ndarray
           Diversity at each position relative to neutral expectation.
       """
       rng = np.random.default_rng(seed)
       two_N = 2 * N
       selected_pos = L / 2

       # Generate trajectory (deterministic for speed)
       x0 = 1.0 / two_N
       traj = []
       x = x0
       while x < 1.0 - 1.0 / two_N:
           traj.append(x)
           x = x * (1 + s) / (1 + s * x)
           x = min(x, 1.0)
       traj.append(1.0)
       traj = np.array(traj)
       T_sweep = len(traj)

       # Neutral expected TMRCA for comparison
       neutral_expected_tmrca = sum(two_N / (k * (k - 1) / 2)
                                     for k in range(n, 1, -1))

       positions = np.linspace(0, L, n_sites)
       tmrcas = np.zeros(n_sites)

       for i, pos in enumerate(positions):
           r_site = r_per_site * abs(pos - selected_pos)

           # Run structured coalescent (simplified: track only TMRCA)
           n_B = n
           n_b = 0
           n_total = n

           for step in range(T_sweep - 1, -1, -1):
               if n_total <= 1:
                   break
               x = traj[step]
               bg_B = max(two_N * x, 1.0)
               bg_b = max(two_N * (1.0 - x), 1.0)

               # Check for events this generation
               rate_cB = n_B * (n_B - 1) / (2.0 * bg_B) if n_B >= 2 else 0
               rate_cb = n_b * (n_b - 1) / (2.0 * bg_b) if n_b >= 2 else 0
               rate_mBb = n_B * r_site * (1.0 - x) if n_B > 0 else 0
               rate_mbB = n_b * r_site * x if n_b > 0 else 0
               total = rate_cB + rate_cb + rate_mBb + rate_mbB

               if total > 0 and rng.exponential(1.0 / total) < 1.0:
                   u = rng.random() * total
                   if u < rate_cB:
                       n_B -= 1; n_total -= 1
                   elif u < rate_cB + rate_cb:
                       n_b -= 1; n_total -= 1
                   elif u < rate_cB + rate_cb + rate_mBb:
                       n_B -= 1; n_b += 1
                   else:
                       n_b -= 1; n_B += 1

           # Forced coalescence of remaining B lineages
           if n_B > 1:
               n_total -= (n_B - 1)
               n_B = 1

           # Neutral coalescent for remaining lineages
           t = T_sweep
           while n_total > 1:
               rate = n_total * (n_total - 1) / (2.0 * two_N)
               t += rng.exponential(1.0 / rate)
               n_total -= 1

           tmrcas[i] = t

       relative_diversity = tmrcas / neutral_expected_tmrca
       return positions, relative_diversity

   # Generate the classic diversity valley around a sweep
   positions, rel_div = minimal_discoal(
       n=20, N=10_000, s=0.01, r_per_site=1e-8, L=200_000, n_sites=100
   )

   print("Diversity profile around a hard sweep (s=0.01, N=10000):")
   print(f"{'Position':>10} {'Relative diversity':>20}")
   for j in range(0, len(positions), 10):
       print(f"{positions[j]:>10.0f} {rel_div[j]:>20.4f}")

   # The characteristic "valley" pattern:
   # - At the selected site (center): diversity ~ 0 (compressed by the sweep)
   # - Far from the site: diversity ~ 1 (neutral)
   # - The width of the valley depends on r/s


Summary: The Same Gears, Different Cases
==========================================

discoal and msprime implement the same mathematical model -- the structured
coalescent under selection -- but wrap it in different engineering:

1. **discoal** is a standalone tool that speaks the language of population genetics
   (:math:`\theta`, :math:`\rho`, :math:`\alpha`), supports the full range of
   sweep types (hard, soft, partial), and outputs classic ms-format haplotype
   matrices. It was purpose-built for sweep simulation.

2. **msprime** is a general-purpose coalescent simulator that added sweep support
   by grafting the structured coalescent onto its existing tree sequence machinery.
   It currently supports only hard sweeps, but the tree sequence output and
   composable model framework make it the better choice for large-scale
   simulations and integration with other tools.

For training data for machine learning classifiers (as in Kern & Schrider's
diploS/HIC), discoal remains the tool of choice: its soft sweep and partial
sweep support is essential for generating the training data that distinguishes
these sweep types.

For understanding the algorithm itself, both implementations teach the same lesson:
**natural selection at one site reshapes the genealogy at linked sites, and the
structured coalescent is the mathematical lens that makes this visible**.

.. admonition:: The watchmaker's postscript

   Building this Timepiece, you have seen how a single parameter -- the selection
   coefficient :math:`s` -- transforms the neutral clockwork into something richer.
   The mainspring (trajectory) drives the escapement (structured coalescent),
   which turns the gear train (recombination as migration), which moves the hands
   on the dial (the diversity profile).

   The mechanism is the same whether you build it in C (discoal) or graft it onto
   an existing movement (msprime). What matters is understanding the gears.
