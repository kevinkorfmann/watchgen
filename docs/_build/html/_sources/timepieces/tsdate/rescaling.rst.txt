.. _tsdate_rescaling:

==========
Rescaling
==========

   *Even a well-tuned mechanism drifts; the final step is calibrating against
   the master clock.*

After belief propagation (whether inside-outside or variational gamma), tsdate
has posterior estimates for every node's age. But these estimates assume a
**constant effective population size**, which is almost never true. If the
population was larger in the past, branches are too long; if it was smaller,
they're too short.

Rescaling corrects for this by comparing the inferred times against the
**empirical mutation clock**: the observed number of mutations in different
time windows. In the watch metaphor, this is the final calibration -- checking
the movement against a master reference clock and adjusting the hands until
the ticks match.

This is the same idea used in SINGER's ARG rescaling (see
:ref:`arg_rescaling`), adapted for tsdate's continuous posteriors.

.. admonition:: Prerequisites

   This chapter assumes you have followed the full tsdate pipeline so far:
   the coalescent prior (:ref:`tsdate_coalescent_prior`), the mutation
   likelihood (:ref:`tsdate_mutation_likelihood`), and one of the two message
   passing algorithms (:ref:`tsdate_inside_outside` or
   :ref:`tsdate_variational_gamma`). Rescaling is a post-processing step
   applied to the node times that those algorithms produce.


The Problem: Mismatch Between Model and Reality
===================================================

The coalescent prior assumes constant :math:`N_e`. Under this model, the
expected time between coalescence events is fixed. But real populations have
complex histories: bottlenecks, expansions, migrations.

When :math:`N_e` was *larger* in the past:

- Real coalescence events were *slower* (more time between them)
- The constant-:math:`N_e` model underestimates deep times
- Branches in the deep past are too short

When :math:`N_e` was *smaller* in the past:

- Real coalescence events were *faster*
- The model overestimates deep times
- Branches in the deep past are too long

Either way, the mutation rate implied by the inferred times won't match the
true mutation rate. Rescaling fixes this.

Think of it as a watch whose mainspring weakens with age: the gears near the
present run at the right speed, but the deeper you go, the more the rate
drifts. Rescaling adjusts the time scale in each epoch so that the ticks
(mutations) per unit time remain constant.


The Key Insight: Mutations Don't Lie
=======================================

Whatever the population history, the molecular clock still ticks at rate
:math:`\mu` per base pair per generation. The total number of mutations
in a time window is:

.. math::

   \text{mutations in } [t_a, t_b) = \mu \cdot \text{(total branch length in } [t_a, t_b) \text{)}

If our estimated times are correct, the ratio
:math:`\text{observed mutations} / \text{expected mutations}` should be 1.0
in every time window. If it's consistently :math:`> 1` in some window, our
branch lengths there are too short, and we need to stretch time. If it's
:math:`< 1`, we need to compress.

This ratio is a direct diagnostic: any deviation from 1.0 reveals how much
the constant-:math:`N_e` assumption has distorted the time scale in that epoch.


Step 1: Partition Time into Windows
======================================

Divide the time axis :math:`[0, t_{\max})` into :math:`J` windows such that each
window contains approximately equal total branch length:

.. math::

   [t_0 = 0, t_1), \quad [t_1, t_2), \quad \ldots, \quad [t_{J-1}, t_J)

**Why equal branch length?** So that each window has comparable statistical
power for estimating the local mutation rate. A window with very little branch
length would have very few mutations and a noisy estimate.

.. code-block:: python

   import numpy as np

   def partition_time_axis(ts, node_times, J=1000):
       """Partition the time axis into J windows of roughly equal branch length.

       Parameters
       ----------
       ts : tskit.TreeSequence
       node_times : np.ndarray
           Current estimated node times (from EP or inside-outside).
       J : int
           Number of windows.

       Returns
       -------
       breakpoints : np.ndarray, shape (J+1,)
           Window boundaries: [breakpoints[j], breakpoints[j+1]) for window j.
       """
       # Collect all branch lengths weighted by span
       branch_data = []  # (midpoint_time, weighted_length)

       for edge in ts.edges():
           t_parent = node_times[edge.parent]
           t_child = node_times[edge.child]
           span = edge.right - edge.left       # genomic span in bp

           if t_parent > t_child:
               midpoint = (t_parent + t_child) / 2     # time midpoint of the edge
               weighted_length = span * (t_parent - t_child)  # bp * generations
               branch_data.append((midpoint, weighted_length))

       if not branch_data:
           return np.linspace(0, 1, J + 1)

       # Sort by time and find breakpoints with equal cumulative branch length
       branch_data.sort(key=lambda x: x[0])
       times = np.array([b[0] for b in branch_data])
       lengths = np.array([b[1] for b in branch_data])

       cum_length = np.cumsum(lengths)       # running total of branch length
       total_length = cum_length[-1]

       breakpoints = [0.0]
       target_per_window = total_length / J  # each window gets equal share

       for j in range(1, J):
           target = j * target_per_window
           idx = np.searchsorted(cum_length, target)  # find where cumulative exceeds target
           if idx < len(times):
               breakpoints.append(times[idx])
           else:
               breakpoints.append(times[-1])

       breakpoints.append(node_times.max() * 1.01)  # upper bound beyond all nodes
       return np.array(breakpoints)


Step 2: Count Mutations per Window
=====================================

For each time window, count how many mutations fall in it. A mutation on edge
:math:`e` is assigned to the time window containing the midpoint of the edge
(or, more precisely, proportionally distributed across windows that the edge
spans).

.. code-block:: python

   def count_mutations_per_window(ts, node_times, breakpoints, mutation_rate):
       """Count observed and expected mutations in each time window.

       Parameters
       ----------
       ts : tskit.TreeSequence
       node_times : np.ndarray
       breakpoints : np.ndarray, shape (J+1,)
       mutation_rate : float

       Returns
       -------
       observed : np.ndarray, shape (J,)
           Mutation count in each window.
       expected : np.ndarray, shape (J,)
           Expected mutations (mu * total branch length) in each window.
       """
       J = len(breakpoints) - 1
       observed = np.zeros(J)
       expected = np.zeros(J)

       # Count mutations per edge (once)
       mut_per_edge = np.zeros(ts.num_edges, dtype=int)
       for mut in ts.mutations():
           if mut.edge >= 0:
               mut_per_edge[mut.edge] += 1

       for edge in ts.edges():
           t_parent = node_times[edge.parent]
           t_child = node_times[edge.child]
           span = edge.right - edge.left    # genomic span
           m_e = mut_per_edge[edge.id]      # observed mutations on this edge

           if t_parent <= t_child:
               continue  # skip edges with zero or negative branch length

           # Distribute this edge's contribution across windows
           for j in range(J):
               w_lo = breakpoints[j]
               w_hi = breakpoints[j + 1]

               # Overlap of edge [t_child, t_parent] with window [w_lo, w_hi]
               overlap_lo = max(t_child, w_lo)
               overlap_hi = min(t_parent, w_hi)

               if overlap_hi > overlap_lo:
                   # Fraction of edge in this window
                   frac = (overlap_hi - overlap_lo) / (t_parent - t_child)

                   # Expected mutations: mu * span * overlap_length
                   expected[j] += mutation_rate * span * (overlap_hi - overlap_lo)

                   # Observed mutations: distribute proportionally by time overlap
                   observed[j] += m_e * frac

       return observed, expected


Step 3: Compute Scaling Factors
==================================

For each window, the scaling factor is the ratio of observed to expected
mutations:

.. math::

   s_j = \frac{\text{observed}_j}{\text{expected}_j}

If :math:`s_j > 1`, the branch lengths in window :math:`j` are too short (need
stretching). If :math:`s_j < 1`, they're too long (need compression).

.. code-block:: python

   def compute_scaling_factors(observed, expected, min_count=1.0):
       """Compute per-window scaling factors.

       Parameters
       ----------
       observed, expected : np.ndarray, shape (J,)
       min_count : float
           Minimum mutation count to trust a window.

       Returns
       -------
       scales : np.ndarray, shape (J,)
       """
       scales = np.ones(len(observed))  # default: no rescaling

       for j in range(len(observed)):
           if expected[j] > 0 and observed[j] >= min_count:
               # Ratio > 1 means branches too short; < 1 means too long
               scales[j] = observed[j] / expected[j]

       return scales

**Intuition**: This is a piecewise-constant estimate of :math:`N_e(t)`.
If the model assumed :math:`N_e = 10{,}000` but the true :math:`N_e` was
:math:`20{,}000` during window :math:`j`, then branch lengths are half what
they should be, and :math:`s_j \approx 2`.

.. admonition:: Probability Aside -- Rescaling as implicit Ne estimation

   The scaling factor :math:`s_j` is closely related to the ratio of true
   :math:`N_e` to assumed :math:`N_e` in window :math:`j`. Under the
   coalescent with variable population size, the coalescent rate is
   :math:`1/N_e(t)`. If we used the wrong :math:`N_e`, the branch lengths in
   that epoch are stretched or compressed by the ratio
   :math:`N_e^{\text{true}} / N_e^{\text{assumed}}`. By setting
   :math:`s_j = \text{observed}_j / \text{expected}_j`, we effectively
   estimate this ratio and correct for it -- without explicitly fitting a
   population-size model. This is similar in spirit to how PSMC
   (:ref:`psmc_timepiece`) estimates :math:`N_e(t)`, except here it is a
   post-processing step rather than the main inference.


Step 4: Apply the Rescaling
==============================

Each node's time is adjusted by the cumulative scaling factor up to its
current time:

.. math::

   t_u^{\text{new}} = \int_0^{t_u^{\text{old}}} s(x) \, dx

where :math:`s(x)` is the piecewise-constant scaling function. This integral
is just a sum:

.. math::

   t_u^{\text{new}} = \sum_{j : t_j < t_u} s_j \cdot \min(t_u, t_{j+1}) - t_j)
   + s_{j^*} \cdot (t_u - t_{j^*})

where :math:`j^*` is the window containing :math:`t_u`.

.. admonition:: Calculus Aside -- Piecewise integration

   The rescaling integral :math:`\int_0^{t} s(x) \, dx` with
   piecewise-constant :math:`s(x)` decomposes into a sum of rectangles:
   in each window :math:`[t_j, t_{j+1})` where :math:`s(x) = s_j`, the
   contribution is :math:`s_j \cdot (t_{j+1} - t_j)`. For the final
   (partial) window containing :math:`t`, the contribution is
   :math:`s_{j^*} \cdot (t - t_{j^*})`. The result is a piecewise-linear,
   monotonically increasing function of the original time -- a "warped" time
   axis that stretches or compresses different epochs.

.. code-block:: python

   def apply_rescaling(node_times, breakpoints, scales, fixed_nodes):
       """Apply piecewise rescaling to node times.

       Parameters
       ----------
       node_times : np.ndarray
           Current node times (will not be modified).
       breakpoints : np.ndarray, shape (J+1,)
       scales : np.ndarray, shape (J,)
       fixed_nodes : set
           Nodes whose times should not change (e.g., samples).

       Returns
       -------
       new_times : np.ndarray
           Rescaled node times.
       """
       new_times = np.zeros_like(node_times)
       J = len(scales)

       # Build cumulative scaling function
       # cum_rescaled[j] = rescaled time at window boundary j
       cum_rescaled = np.zeros(J + 1)
       for j in range(J):
           window_width = breakpoints[j + 1] - breakpoints[j]
           cum_rescaled[j + 1] = cum_rescaled[j] + scales[j] * window_width

       for u in range(len(node_times)):
           if u in fixed_nodes:
               new_times[u] = node_times[u]  # samples stay fixed
               continue

           t = node_times[u]

           # Find which window t falls in
           j = np.searchsorted(breakpoints, t, side='right') - 1
           j = min(j, J - 1)
           j = max(j, 0)

           # Rescaled time = cumulative up to window j + fraction within window
           fraction_in_window = t - breakpoints[j]
           new_times[u] = cum_rescaled[j] + scales[j] * fraction_in_window

       return new_times


Iterating the Rescaling
==========================

A single round of rescaling may not be sufficient because the window boundaries
depend on the node times, which change after rescaling. tsdate iterates:

1. Compute window boundaries from current times
2. Count mutations per window
3. Compute scaling factors
4. Apply rescaling to get new times
5. Repeat (default: 5 iterations)

Each iteration refines the time scale, like adjusting a regulator screw on a
mechanical watch -- small turns that progressively bring the rate into
alignment with the master clock.

.. code-block:: python

   def iterative_rescaling(ts, node_times, mutation_rate, fixed_nodes,
                           J=1000, num_iter=5):
       """Iteratively rescale node times to match the mutation clock.

       Parameters
       ----------
       ts : tskit.TreeSequence
       node_times : np.ndarray
       mutation_rate : float
       fixed_nodes : set
       J : int
           Number of time windows.
       num_iter : int
           Number of rescaling iterations.

       Returns
       -------
       node_times : np.ndarray
           Rescaled node times.
       """
       times = node_times.copy()

       for iteration in range(num_iter):
           # 1. Partition time into equal-branch-length windows
           breakpoints = partition_time_axis(ts, times, J)

           # 2. Count observed vs. expected mutations per window
           observed, expected = count_mutations_per_window(
               ts, times, breakpoints, mutation_rate)

           # 3. Compute scaling factors (observed / expected)
           scales = compute_scaling_factors(observed, expected)

           # 4. Apply piecewise rescaling
           times = apply_rescaling(times, breakpoints, scales, fixed_nodes)

       return times


Connection to Population Size History
========================================

The scaling factors :math:`s_j` are intimately related to the effective
population size history. Under the coalescent with variable :math:`N_e(t)`:

- The rate of coalescence at time :math:`t` is :math:`1 / N_e(t)`
- The mutation rate is constant at :math:`\mu`

If we model the coalescent under constant :math:`N_e^{(0)}` but the true
population size in window :math:`j` is :math:`N_e^{(j)}`, then:

.. math::

   s_j \approx \frac{N_e^{(j)}}{N_e^{(0)}}

So the rescaling implicitly estimates the population size history. This is
similar to what PSMC does (see the :ref:`psmc_timepiece`), but here it's a
post-processing step rather than the main inference.


Edge Cases and Robustness
============================

Several practical issues arise:

**Windows with few mutations**: If a window has very few mutations (or none),
the scaling factor is unreliable. tsdate handles this by:

- Setting a minimum count threshold
- Smoothing adjacent scaling factors
- Falling back to a scale of 1.0 for empty windows

**Negative branch lengths**: After rescaling, some edges might end up with
the parent younger than the child. tsdate enforces constraints by adjusting
times to maintain the topological ordering.

**Convergence**: Rescaling typically converges within 3-5 iterations. The
scaling factors stabilize as the times settle into their correct positions.


The Full Pipeline
==================

Putting rescaling together with EP, here is the complete tsdate pipeline from
raw tree sequence to dated genealogy.

.. code-block:: python

   def tsdate_full_pipeline(ts, mutation_rate, Ne=1.0, max_ep_iter=25,
                            rescaling_intervals=1000, rescaling_iterations=5):
       """The complete tsdate pipeline.

       Parameters
       ----------
       ts : tskit.TreeSequence
           Input (topology from tsinfer).
       mutation_rate : float
       Ne : float
       max_ep_iter : int
       rescaling_intervals : int
       rescaling_iterations : int

       Returns
       -------
       dated_ts : np.ndarray
           Posterior mean node times.
       """
       # Step 1: Build priors (Gear 1 -- the expected beat rate)
       prior_grid = build_coalescent_priors(ts, Ne)

       # Step 2: Run EP (Gear 4 -- messages flow through the gear train)
       posteriors = run_ep(ts, mutation_rate, prior_grid, max_ep_iter)

       # Step 3: Extract posterior means
       node_times = np.zeros(ts.num_nodes)
       for u in range(ts.num_nodes):
           if u in posteriors:
               node_times[u] = posteriors[u].mean

       fixed_nodes = set(ts.samples())
       for s in fixed_nodes:
           node_times[s] = 0.0

       # Step 4: Rescale (Gear 5 -- calibrate against the mutation clock)
       if rescaling_iterations > 0:
           node_times = iterative_rescaling(
               ts, node_times, mutation_rate, fixed_nodes,
               J=rescaling_intervals,
               num_iter=rescaling_iterations
           )

       return node_times


Summary
========

Rescaling is tsdate's final calibration step -- the last gear in the mechanism:

1. **Partition** the time axis into :math:`J` windows of equal branch length
2. **Count** observed vs. expected mutations per window
3. **Scale** each window by the ratio :math:`s_j = \text{observed}/\text{expected}`
4. **Apply** the piecewise scaling to all node times
5. **Iterate** until convergence (default: 5 rounds)

The key equation:

.. math::

   t_u^{\text{new}} = \int_0^{t_u^{\text{old}}} \frac{\text{observed mutations}(x)}
   {\text{expected mutations}(x)} \, dx

This corrects for variable population size without explicitly modeling it,
by letting the molecular clock be the final arbiter of time. In the watch
metaphor, the mutation clock is the master reference -- it ticks at a known
rate (:math:`\mu`) regardless of population history, and rescaling adjusts
every hand on the dial until the ticks match.

Congratulations -- you've now built every gear of the tsdate mechanism:

1. **Coalescent prior** -- the expected beat rate from coalescent theory: informed
   starting beliefs about node ages
2. **Mutation likelihood** -- evidence from the mutation clock: the Poisson model
   connecting observed data to branch lengths
3. **Inside-outside** -- messages flowing through the gear train on a discrete grid
4. **Variational gamma** -- the same messages, now carried by continuous gamma
   distributions via expectation propagation
5. **Rescaling** -- calibrating the clock against the master reference: adjusting
   for variable population size

Together, these gears transform a topology-only tree sequence (from tsinfer)
into a fully dated genealogy. You understand the math, the code, and the
intuition behind every step. The watch is assembled, calibrated, and keeping
time.
