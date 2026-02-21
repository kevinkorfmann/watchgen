.. _arg_rescaling:

==============
ARG Rescaling
==============

   *A watch with the right gears but the wrong spring tension still tells the wrong time.*

In the :ref:`previous chapters <branch_sampling>`, we built the threading
algorithm -- branch sampling finds *which branch* to join, and
:ref:`time sampling <time_sampling>` finds *when* to join. Together, they
produce an ARG with coalescence times on every node. But these times were
estimated under a constant-population-size assumption, using approximate
transition and emission models. In reality, populations change size, and the
mutation clock doesn't tick at constant speed across the genome.

ARG rescaling corrects for this by adjusting coalescence times to match the
observed mutations. In the watch metaphor, the threading algorithm built a watch
with the right gears (correct tree topologies) but potentially the wrong spring
tension (incorrect time scale). Rescaling calibrates the beat rate -- it adjusts
the tension so that the watch's tick rate matches the molecular clock, as
measured by the mutations we actually observe in the DNA.

The Idea
=========

The key insight: **mutations are a molecular clock**. If we have the right
tree, the number of mutations on each branch should be proportional to the
branch length. If the times are off, we can detect this by comparing expected
vs. observed mutation counts in different time windows.

.. admonition:: Probability Aside -- Mutations as a Poisson Process

   Recall from :ref:`coalescent theory <coalescent_theory>` that mutations
   occur as a **Poisson process** along branches of the genealogy. On a branch
   of length :math:`\ell` (in coalescent time units) spanning :math:`s` base
   pairs, the expected number of mutations is :math:`(\theta/2) \cdot \ell \cdot s`,
   where :math:`\theta = 4N_e\mu` is the population-scaled mutation rate.

   The Poisson model has a crucial property: the *expected* number of mutations
   is proportional to the branch length. This means that if we observe *more*
   mutations than expected in some time window, the branches in that window
   are probably longer than we estimated (the time scale is compressed). If we
   observe *fewer* mutations, the branches are probably shorter (the time scale
   is stretched).

   This is exactly the principle behind molecular clock calibration in
   phylogenetics: use the number of substitutions to estimate elapsed time.
   ARG rescaling applies this principle locally, in each time window.

SINGER partitions the time axis into windows and rescales each window
independently to match the empirical mutation count.

Step 1: Partition the Time Axis
================================

Given an inferred ARG :math:`\mathcal{G}`, partition the time axis
:math:`[0, t_{\max})` into :math:`J` intervals:

.. math::

   [t_0 = 0, t_1), \quad [t_1, t_2), \quad \ldots, \quad [t_{J-1}, t_J = t_{\max})

such that each interval contains :math:`\frac{1}{J}` of the total ARG branch
length.

.. admonition:: Probability Aside -- Why Equal Branch Length, Not Equal Time?

   We partition the time axis so that each window contains the *same total
   branch length*, not the same span of time. This is an instance of the
   **equal information content** principle we encountered in
   :ref:`PSMC discretization <psmc_discretization>`.

   The reason: the statistical power to estimate the scaling factor in a window
   depends on how many mutations fall in that window. The expected number of
   mutations is proportional to the branch length in the window (by the Poisson
   model). So equal branch length per window means equal expected mutations per
   window, which means equal statistical precision for each scaling factor.

   If we used equal time spans instead, recent windows (which contain many
   short branches from many samples) would have enormous branch length and
   many mutations, while ancient windows (which contain few long branches)
   would have little branch length and few mutations. The scaling factors
   for ancient windows would be very noisy.

The **ARG length in an interval** :math:`[t_i, t_{i+1})` is the sum of branch
length overlapping the interval, across all marginal trees, weighted by tree
span.

.. code-block:: python

   import numpy as np

   def compute_arg_length_in_window(branches, window_lower, window_upper):
       """Compute total ARG length overlapping a time window.

       For each branch in the ARG, compute the time overlap between
       the branch's time interval and the window, then weight by the
       branch's genomic span (number of base pairs it covers).

       Parameters
       ----------
       branches : list of (span, lower_time, upper_time)
           Each branch has a genomic span and a time interval.
       window_lower, window_upper : float
           Time window boundaries.

       Returns
       -------
       length : float
           Total branch length in this window, weighted by span.
       """
       total = 0.0
       for span, lo, hi in branches:
           # Overlap between [lo, hi) and [window_lower, window_upper)
           overlap_lo = max(lo, window_lower)
           overlap_hi = min(hi, window_upper)
           if overlap_hi > overlap_lo:
               # span * time_overlap = total branch-length contribution
               total += span * (overlap_hi - overlap_lo)
       return total

   def partition_time_axis(branches, J=100):
       """Partition time axis into J equal-ARG-length windows.

       Finds time boundaries such that each window contains
       1/J of the total ARG branch length.  Uses a sweep through
       all distinct time points in the ARG.

       Parameters
       ----------
       branches : list of (span, lower_time, upper_time)
       J : int
           Number of windows.

       Returns
       -------
       boundaries : ndarray of shape (J + 1,)
       """
       # Total ARG length (sum of span * branch_length for all branches)
       t_max = max(hi for _, _, hi in branches)
       total_length = compute_arg_length_in_window(branches, 0, t_max)
       target_per_window = total_length / J

       # Find boundaries by scanning through time.
       # Collect all distinct time points (branch endpoints) to avoid
       # missing discontinuities in the branch-length function.
       time_points = sorted(set(
           [0.0, t_max] +
           [lo for _, lo, _ in branches] +
           [hi for _, _, hi in branches]
       ))

       boundaries = [0.0]
       cumulative = 0.0

       for k in range(len(time_points) - 1):
           segment_length = compute_arg_length_in_window(
               branches, time_points[k], time_points[k + 1])
           cumulative += segment_length

           # When we've accumulated enough length, place a boundary
           while cumulative >= target_per_window and len(boundaries) < J:
               # Interpolate to find exact boundary within this segment
               overshoot = cumulative - target_per_window
               segment_total = segment_length
               if segment_total > 0:
                   fraction = 1 - overshoot / segment_total
                   boundary = (time_points[k] +
                               fraction * (time_points[k + 1] - time_points[k]))
               else:
                   boundary = time_points[k + 1]
               boundaries.append(boundary)
               cumulative -= target_per_window

       boundaries.append(t_max)
       return np.array(boundaries[:J + 1])

   # Example: simple tree with known structure
   branches = [
       (1000, 0.0, 0.3),   # leaf branch, spans 1000 bp
       (1000, 0.0, 0.3),   # leaf branch
       (1000, 0.0, 0.7),   # leaf branch
       (1000, 0.0, 0.7),   # leaf branch
       (1000, 0.3, 0.7),   # internal branch
       (1000, 0.3, 0.7),   # internal branch
       (1000, 0.7, 1.5),   # root branch
   ]

   boundaries = partition_time_axis(branches, J=5)
   print("Time window boundaries:")
   for i in range(len(boundaries) - 1):
       length = compute_arg_length_in_window(branches, boundaries[i],
                                               boundaries[i + 1])
       print(f"  [{boundaries[i]:.4f}, {boundaries[i+1]:.4f}): "
             f"ARG length = {length:.1f}")

With the time axis partitioned, we next count how many mutations fall in each
window.


Step 2: Count Mutations per Window
====================================

For each time window, count the number of mutations that fall in it. If a
mutation sits on a branch that spans multiple windows, its count is split
proportionally.

.. admonition:: Probability Aside -- Fractional Mutation Counts

   A mutation is placed on a branch, but we don't know *exactly* where on the
   branch it occurred -- only that it happened somewhere between the branch's
   lower and upper time. When a branch spans multiple time windows, we
   distribute the mutation's count proportionally to the overlap with each
   window.

   This is a form of **soft assignment**: instead of assigning each mutation
   to a single window (which would lose information), we distribute it
   fractionally. If a branch spans 30% in window :math:`i` and 70% in window
   :math:`j`, the mutation contributes 0.3 to :math:`m_i` and 0.7 to
   :math:`m_j`. This is the same logic used in EM algorithms (see
   :ref:`the PSMC EM chapter <psmc_hmm>`) where observations are
   probabilistically assigned to hidden states.

.. math::

   m_i = \sum_{\text{mutation } \mu} \text{fraction of } \mu\text{'s branch in window } i

.. code-block:: python

   def count_mutations_per_window(mutations, boundaries):
       """Count mutations in each time window, fractionally.

       Each mutation is distributed across windows proportionally
       to the overlap between its branch and each window.

       Parameters
       ----------
       mutations : list of (branch_lower, branch_upper)
           Time interval of the branch carrying each mutation.
       boundaries : ndarray of shape (J + 1,)

       Returns
       -------
       counts : ndarray of shape (J,)
       """
       J = len(boundaries) - 1
       counts = np.zeros(J)

       for branch_lo, branch_hi in mutations:
           branch_length = branch_hi - branch_lo
           if branch_length == 0:
               continue  # degenerate branch: skip

           for i in range(J):
               # How much of this branch falls in window i?
               overlap_lo = max(branch_lo, boundaries[i])
               overlap_hi = min(branch_hi, boundaries[i + 1])
               if overlap_hi > overlap_lo:
                   fraction = (overlap_hi - overlap_lo) / branch_length
                   counts[i] += fraction  # fractional mutation count

       return counts

   # Example: 20 mutations at various branch heights
   np.random.seed(42)
   mutations = [(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.5))
                for _ in range(20)]

   counts = count_mutations_per_window(mutations, boundaries)
   print("\nMutation counts per window:")
   for i in range(len(counts)):
       print(f"  Window {i}: {counts[i]:.2f} mutations")

Now that we have both the ARG branch length per window and the mutation count
per window, we can compute the scaling factors that will recalibrate the
coalescence times.


Step 3: Compute Scaling Factors
=================================

The total expected number of mutations across the entire ARG is
:math:`\frac{\theta}{2} L(\mathcal{G})` (from the Poisson mutation model: rate
:math:`\theta/2` per unit branch length per base pair, summed over all branch
length).

Since we partitioned the time axis so that each window contains
:math:`\frac{1}{J}` of the total ARG length, the ARG length in each window is
:math:`\frac{L(\mathcal{G})}{J}`. Therefore, the expected number of
mutations in each time window is:

.. math::

   \mathbb{E}[\text{mutations in window } i] = \frac{\theta}{2} \cdot \frac{L(\mathcal{G})}{J}
   = \frac{\theta L(\mathcal{G})}{2J}

If the observed count is :math:`m_i`, the ratio of observed to expected is:

.. math::

   c_i = \frac{m_i}{\mathbb{E}[\text{mutations in window } i]} = \frac{m_i}{\theta L(\mathcal{G}) / (2J)}
   = \frac{2J \cdot m_i}{\theta \cdot L(\mathcal{G})}

.. admonition:: Probability Aside -- The Scaling Factor as a Likelihood Ratio

   The scaling factor :math:`c_i` has a natural interpretation as a
   **maximum likelihood estimate** of the local time dilation in window :math:`i`.

   Under the Poisson mutation model, the number of mutations in window :math:`i`
   follows a Poisson distribution with mean :math:`(\theta/2) \cdot L_i`, where
   :math:`L_i` is the true branch length in window :math:`i`. If we observe
   :math:`m_i` mutations, the MLE for :math:`L_i` is :math:`2m_i / \theta`.
   The ratio of the estimated branch length to the assumed branch length
   :math:`L(\mathcal{G})/J` gives the scaling factor :math:`c_i`.

   In other words, :math:`c_i` answers: "by what factor should we stretch or
   compress the time axis in window :math:`i` to make the observed mutations
   consistent with the assumed mutation rate?"

**Interpretation**: :math:`c_i > 1` means more mutations than expected (the true
time window should be *wider* -- time was compressed). :math:`c_i < 1` means
fewer mutations (the true window should be *narrower* -- time was stretched).
:math:`c_i = 1` means the times are already calibrated for this window.

**Edge case**: If :math:`m_i = 0` (no mutations in a window), then :math:`c_i = 0`
and the window collapses. In practice, this is handled by using a minimum scaling
factor or by choosing :math:`J` small enough that each window contains mutations.

.. code-block:: python

   def compute_scaling_factors(counts, total_arg_length, theta, J):
       """Compute rescaling factors for each time window.

       Each factor c_i = observed / expected mutations in window i.
       A factor > 1 means time was compressed (too few mutations for
       the branch length); < 1 means time was stretched.

       Parameters
       ----------
       counts : ndarray of shape (J,)
           Mutation counts per window.
       total_arg_length : float
       theta : float
           Population-scaled mutation rate.
       J : int
           Number of windows.

       Returns
       -------
       c : ndarray of shape (J,)
           Scaling factors.
       """
       # Expected mutations per window: theta/2 * (total_length / J)
       expected_per_window = theta * total_arg_length / (2 * J)
       if expected_per_window == 0:
           return np.ones(J)  # nothing to rescale

       c = counts / expected_per_window
       return c

   total_length = sum(span * (hi - lo) for span, lo, hi in branches)
   theta = 0.001
   c = compute_scaling_factors(counts, total_length, theta, len(counts))
   print("\nScaling factors:")
   for i, ci in enumerate(c):
       print(f"  Window {i}: c = {ci:.4f}")


Step 4: Rescale Coalescence Times
===================================

Apply the scaling factors to transform the time axis. The new time boundaries are:

.. math::

   \tilde{t}_0 = 0, \qquad \tilde{t}_i = c_i(t_i - t_{i-1}) + \tilde{t}_{i-1}

A coalescence time :math:`t \in [t_{i-1}, t_i)` is rescaled to:

.. math::

   \tilde{t} = c_i(t - t_{i-1}) + \tilde{t}_{i-1}

.. admonition:: Calculus Aside -- Piecewise Linear Rescaling

   The rescaling is a **piecewise linear transformation**: within each window,
   the mapping from old time :math:`t` to new time :math:`\tilde{t}` is a
   linear function with slope :math:`c_i`. If :math:`c_i > 1`, the window
   is stretched (the slope is steeper -- more time passes per unit of old time).
   If :math:`c_i < 1`, the window is compressed.

   The transformation is continuous (no jumps at window boundaries) because
   each piece starts where the previous one ended:
   :math:`\tilde{t}_{i} = c_i(t_i - t_{i-1}) + \tilde{t}_{i-1}`. This ensures
   that the rescaled time axis is a monotonically increasing function of the
   original time axis -- the order of events is preserved.

   This is the simplest reasonable rescaling: a step function of slopes. More
   sophisticated approaches (e.g., smooth spline rescaling) would give smoother
   results but at the cost of additional complexity and potential overfitting.

.. code-block:: python

   def rescale_times(node_times, boundaries, scaling_factors):
       """Rescale all coalescence times using window-specific scaling.

       Each node's time is transformed according to the scaling factor
       of the window it falls in.  The transformation is piecewise
       linear and monotonically increasing.

       Parameters
       ----------
       node_times : dict of {node_id: time}
       boundaries : ndarray of shape (J + 1,)
       scaling_factors : ndarray of shape (J,)

       Returns
       -------
       new_times : dict of {node_id: rescaled_time}
       """
       J = len(scaling_factors)

       # Compute new window boundaries by applying the rescaling
       new_boundaries = np.zeros(J + 1)
       for i in range(J):
           # Each new boundary = previous boundary + scaled window width
           new_boundaries[i + 1] = (scaling_factors[i] *
                                     (boundaries[i + 1] - boundaries[i]) +
                                     new_boundaries[i])

       # Rescale each node time
       new_times = {}
       for node_id, t in node_times.items():
           if t <= 0:
               new_times[node_id] = 0.0  # leaf nodes stay at time 0
               continue

           # Find which window this time falls in
           for i in range(J):
               if boundaries[i] <= t < boundaries[i + 1]:
                   # Apply piecewise linear rescaling:
                   # offset within window * scaling factor + new window start
                   new_t = (scaling_factors[i] * (t - boundaries[i]) +
                            new_boundaries[i])
                   new_times[node_id] = new_t
                   break
           else:
               # Time is at or beyond t_max: map to the end
               new_times[node_id] = new_boundaries[-1]

       return new_times

   # Example
   node_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
                 4: 0.3, 5: 0.7, 6: 1.5}

   new_times = rescale_times(node_times, boundaries, c)
   print("\nRescaled coalescence times:")
   for node_id in sorted(new_times.keys()):
       print(f"  Node {node_id}: {node_times[node_id]:.4f} -> "
             f"{new_times[node_id]:.4f}")


Step 5: Handling Mutation Rate Variation
==========================================

When local mutation rates vary across the genome (mutation rate map :math:`\mu(x)`),
the expected number of mutations changes. The expected branch length in window
:math:`[t_i, t_{i+1})` becomes:

.. math::

   \sum_{k} \bar{\mu}_k \cdot \mathbb{1}(x_k < t_{i+1}, y_k > t_i) \cdot
   [\min(y_k, t_{i+1}) - \max(x_k, t_i)]

where :math:`\bar{\mu}_k = \int_{x_k}^{y_k} \mu(s) \, ds / (y_k - x_k)` is the
average mutation rate across the genomic span of branch :math:`k`.

.. admonition:: Probability Aside -- Why Mutation Rate Variation Matters

   In many organisms, the mutation rate varies significantly across the genome.
   For example, in humans, CpG sites mutate at roughly 10x the rate of other
   sites, and there are large-scale regional variations in mutation rate
   across chromosomes.

   If we ignore this variation and assume a constant mutation rate, regions
   with high mutation rates will appear to have deeper coalescence times
   (more mutations imply longer branches), and regions with low mutation rates
   will appear to have shallower times. This bias propagates through the
   entire ARG inference.

   By incorporating a **mutation rate map** :math:`\mu(x)`, the rescaling
   procedure can distinguish between "more mutations because the time is
   deeper" and "more mutations because this region has a higher mutation rate."
   The scaling factors then correct only for genuine time miscalibration, not
   for rate variation.

.. code-block:: python

   def count_mutations_with_rate_variation(branches, mutations, boundaries,
                                            mutation_rate_map):
       """Count expected mutations per window accounting for rate variation.

       Instead of assuming a constant mutation rate, this function uses
       a position-dependent rate map to compute expected mutations.

       Parameters
       ----------
       branches : list of (start_pos, end_pos, lower_time, upper_time)
       mutations : list of (position, branch_lower, branch_upper)
       boundaries : ndarray of shape (J + 1,)
       mutation_rate_map : callable
           mutation_rate_map(x) returns the local mutation rate at position x.

       Returns
       -------
       expected : ndarray of shape (J,)
           Expected mutations per window.
       observed : ndarray of shape (J,)
           Observed mutations per window.
       """
       J = len(boundaries) - 1
       expected = np.zeros(J)
       observed = np.zeros(J)

       # Expected: integrate over branches, weighting by local mutation rate
       for start, end, lo, hi in branches:
           # Average mutation rate over this branch's genomic span
           # (simplified: evaluate at midpoint instead of integrating)
           mu_avg = mutation_rate_map((start + end) / 2)
           span = end - start

           for i in range(J):
               overlap_lo = max(lo, boundaries[i])
               overlap_hi = min(hi, boundaries[i + 1])
               if overlap_hi > overlap_lo:
                   # Expected mutations = rate * span * time_overlap
                   expected[i] += mu_avg * span * (overlap_hi - overlap_lo)

       # Observed: count actual mutations (same as before)
       for pos, branch_lo, branch_hi in mutations:
           branch_length = branch_hi - branch_lo
           if branch_length == 0:
               continue
           for i in range(J):
               overlap_lo = max(branch_lo, boundaries[i])
               overlap_hi = min(branch_hi, boundaries[i + 1])
               if overlap_hi > overlap_lo:
                   observed[i] += (overlap_hi - overlap_lo) / branch_length

       return expected, observed


Putting It All Together
========================

The complete ARG rescaling procedure chains all the steps together. In the watch
metaphor, this is the calibration of the beat rate: we measure the molecular
clock (mutations), compare it to the assumed rate, and adjust the spring tension
(time scale) in each window until the watch ticks in sync with the clock.

.. code-block:: python

   def rescale_arg(arg, theta, J=100, mutation_rate_map=None):
       """Full ARG rescaling procedure.

       This is the complete calibration step. It:
       1. Partitions time into equal-branch-length windows
       2. Counts mutations in each window
       3. Computes observed/expected ratios (scaling factors)
       4. Applies piecewise linear rescaling to all node times

       Parameters
       ----------
       arg : object
           The inferred ARG (tree sequence).
       theta : float
           Population-scaled mutation rate.
       J : int
           Number of time windows.
       mutation_rate_map : callable, optional
           Local mutation rate function.

       Returns
       -------
       rescaled_arg : object
           ARG with rescaled coalescence times.
       """
       # Step 1: Extract branches with their spans
       branches = extract_branches(arg)

       # Step 2: Partition time axis into equal-length windows
       boundaries = partition_time_axis(branches, J)

       # Step 3: Count mutations per window
       mutations = extract_mutations(arg)

       if mutation_rate_map is None:
           # Constant rate: simple observed/expected ratio
           counts = count_mutations_per_window(mutations, boundaries)
           total_length = sum(s * (h - l) for s, l, h in branches)
           scaling = compute_scaling_factors(counts, total_length, theta, J)
       else:
           # Variable rate: use the rate map for expected counts
           expected, observed = count_mutations_with_rate_variation(
               branches, mutations, boundaries, mutation_rate_map)
           # Scaling = observed / expected (with fallback for zero expected)
           scaling = np.where(expected > 0, observed / expected, 1.0)

       # Step 4: Rescale all node times
       node_times = extract_node_times(arg)
       new_times = rescale_times(node_times, boundaries, scaling)

       # Step 5: Update the ARG with new times
       rescaled_arg = update_arg_times(arg, new_times)

       return rescaled_arg

.. admonition:: When is rescaling applied?

   ARG rescaling is performed:

   1. After the initial threading (building the ARG from scratch)
   2. After each MCMC thinning step (after a set of :ref:`SGPR <sgpr>` moves)

   This ensures that coalescence times stay calibrated to the mutation data
   throughout the MCMC run. Without periodic rescaling, small errors in the
   time scale would accumulate over many MCMC iterations, causing the inferred
   ARG to drift away from the truth.

   In the watch metaphor, rescaling is like periodically checking the watch
   against a reference clock and adjusting the spring tension. Even a well-made
   watch drifts over time; regular calibration keeps it accurate.

.. admonition:: Comparison with other approaches

   SINGER's rescaling is **self-contained**: it uses only mutation counts from
   the inferred ARG, without requiring a known demographic model. This contrasts
   with methods like the ARG normalization in Zhang et al. (2023), which assumes
   a known demography and performs quantile matching.

   The :ref:`tsdate <tsdate_timepiece>` approach to dating nodes in a tree
   sequence uses a similar principle (matching observed mutations to branch
   lengths), but applies it via a Bayesian message-passing algorithm rather
   than the window-based approach used here. Both methods share the fundamental
   insight that mutations are the clock, and branch lengths must be consistent
   with the clock.


Exercises
=========

.. admonition:: Exercise 1: Rescaling on a known tree

   Simulate a single coalescent tree with ``msprime`` using a bottleneck
   demographic model. Run the rescaling procedure on the resulting tree and
   compare the rescaled times to the true times.

.. admonition:: Exercise 2: Window sensitivity

   Try different values of :math:`J` (10, 50, 100, 500). How does the number of
   windows affect the accuracy and variance of the rescaled times?

.. admonition:: Exercise 3: Mutation rate heterogeneity

   Create a mutation rate map with a 10x hotspot in one region. Verify that the
   rescaling procedure with the rate map correctly handles this, while the
   constant-rate version does not.

Next: :ref:`sgpr` -- the MCMC engine that lets us explore the space of ARGs.
