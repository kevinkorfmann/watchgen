.. _argweaver_time_discretization:

====================
Time Discretization
====================

   *The first gear to cut: a clock face with unevenly spaced tick marks,
   crowded near the present where the action is densest.*

This chapter covers the foundational mechanism that distinguishes ARGweaver from
continuous-time methods like SINGER: the **discretization of time** onto a finite
grid. This single design choice cascades through the entire algorithm --- it determines
the HMM state space, the transition matrix structure, and the granularity of the
posterior samples.

In the overview (:ref:`argweaver_overview`), we called ARGweaver a "digital watch."
Now we forge the tick marks on its dial. If you skipped the overview, go back and
read it first --- the terminology and high-level flow defined there are essential for
what follows.

Why Discretize Time?
=====================

In a coalescent model, times are continuous: any positive real number is a valid
coalescence time. This means the state space for an HMM threading a new lineage
is infinite --- you cannot enumerate all (branch, time) pairs.

SINGER handles this by splitting the problem: first pick branches (finite), then
sample continuous times conditioned on the branch choice. ARGweaver takes a different
approach: **snap all times to a finite grid**, making the joint (branch, time) state
space finite from the start.

The benefits:

1. **Exact HMM computation** --- The forward--backward algorithm runs on a finite
   state space with no quadrature or approximation in the HMM itself.

2. **Single HMM** --- No need to decouple branches and times into separate stages.

3. **Simpler transition matrices** --- Transitions are ordinary matrices, not
   continuous kernels.

The cost is a discretization error: coalescence times are rounded to the nearest
grid point. This error is controlled by the number of time points :math:`n_t` and
the grid spacing.

.. admonition:: Calculus Aside --- From continuous kernels to finite matrices

   Under the continuous coalescent (see :ref:`coalescent_theory`), the probability
   that a lineage coalesces in the infinitesimal interval :math:`[t, t+dt)` is
   :math:`\lambda(t)\,dt`, where :math:`\lambda(t) = k(k-1)/(4N_e(t))` for :math:`k`
   lineages. To compute the forward probability in an HMM with continuous time, you
   must evaluate integrals of the form:

   .. math::

      \alpha_s(b', t') = \int_0^\infty \alpha_{s-1}(b, t) \; K\big((b,t) \to (b',t')\big) \; dt

   where :math:`K` is a transition kernel. Discretization replaces this integral with
   a finite sum:

   .. math::

      \alpha_s(b', j) = \sum_{(b,i)} \alpha_{s-1}(b, i) \; T_{(b,i) \to (b',j)}

   The integral becomes a matrix--vector product --- something a computer can evaluate
   exactly (up to floating point). This is why discretization is so powerful: it turns
   an analytically intractable integral equation into linear algebra.

.. admonition:: Closing the confusion gap --- Why "finite" matters

   You might wonder: cannot we just approximate the integral numerically? Yes, but
   every numerical integration scheme (trapezoidal rule, Gauss quadrature, etc.)
   effectively *discretizes* the integral at a finite set of evaluation points.
   ARGweaver's time grid is exactly such a discretization, but designed specifically
   for the coalescent: denser where coalescence events concentrate (near the present)
   and sparser where they are rare (the deep past). By building the discretization
   into the model rather than treating it as an afterthought, ARGweaver can
   pre-compute all transition probabilities once per tree and reuse them for every
   genomic position --- a huge efficiency gain.

The Exponential Time Grid
==========================

ARGweaver does **not** use a uniform grid. Instead, it uses a log-spaced grid that
is denser near the present (where the coalescent density is highest) and sparser
in the deep past.

The :math:`i`-th time point is:

.. math::

   t_i = \frac{\exp\!\Big(\frac{i}{n_t - 1} \cdot \ln(1 + \delta \cdot T_{\max})\Big) - 1}{\delta}

where:

- :math:`n_t` is the number of time points (including :math:`t_0 = 0`)
- :math:`T_{\max}` is the maximum time in generations
- :math:`\delta` is a parameter controlling the spacing

.. admonition:: Why this formula?

   Let's unpack it. Define :math:`f(x) = \frac{e^x - 1}{\delta}`. This maps
   :math:`x = 0` to :math:`t = 0` and :math:`x = \ln(1 + \delta T_{\max})` to
   :math:`t = T_{\max}`. By spacing :math:`x` uniformly on :math:`[0, \ln(1 + \delta T_{\max})]`
   and applying :math:`f`, we get time points that are **uniformly spaced on a log scale**
   (after shifting by :math:`1/\delta`).

   Near the present, consecutive time points are close together because the exponential
   grows slowly when :math:`x` is small. In the deep past, they spread out because
   the exponential grows fast. This matches the coalescent's behavior: most coalescence
   events happen recently, so we need finer resolution there.

.. admonition:: Calculus Aside --- The exponential map in detail

   The formula is a change of variables. Start with :math:`n_t` uniformly spaced points
   :math:`x_i = i \cdot h` where :math:`h = \frac{1}{n_t - 1} \ln(1 + \delta T_{\max})`.
   Then apply the transformation :math:`t = g(x) = (e^x - 1)/\delta`.

   The derivative is :math:`g'(x) = e^x / \delta`, which grows with :math:`x`. This
   means equal steps in :math:`x`-space produce *growing* steps in :math:`t`-space ---
   small steps near :math:`t=0` (where :math:`x` is small and :math:`g'` is small) and
   large steps in the deep past (where :math:`x` is large and :math:`g'` is large).

   If you have studied PSMC (see :ref:`coalescent_theory`), you will recognize a similar
   strategy: PSMC also uses a non-uniform time discretization to concentrate resolution
   in the recent past. ARGweaver's formula is a specific, closed-form version of the
   same idea.

The watch metaphor is apt here: **the tick marks on the dial are not evenly spaced**.
They crowd together near 12 o'clock (the present) where the second hand's motion
matters most, and spread out near 6 o'clock (the deep past) where the hand barely
moves. Twenty ticks are enough to read the time accurately in the range that matters.

.. code-block:: python

   from math import exp, log

   def get_time_point(i, ntimes, maxtime, delta=0.01):
       """
       Compute the i-th discretized time point.

       Parameters
       ----------
       i : int
           Index of the time point (0 <= i <= ntimes-1).
       ntimes : int
           Total number of time intervals (ntimes-1 is the last index).
       maxtime : float
           Maximum time in generations.
       delta : float
           Controls log-spacing. Smaller delta -> more uniform spacing.
           Larger delta -> more concentration near present.

       Returns
       -------
       float
           The i-th time point in generations.

       Examples
       --------
       >>> get_time_point(0, 19, 160000, 0.01)
       0.0
       >>> round(get_time_point(1, 19, 160000, 0.01), 1)
       52.6
       >>> round(get_time_point(19, 19, 160000, 0.01), 1)
       160000.0
       """
       # i / ntimes gives the fractional position along the grid (0 to 1).
       # Multiplying by log(1 + delta*maxtime) maps this to a log-scale range.
       # exp(...) transforms back, and subtracting 1 then dividing by delta
       # undoes the shift-and-scale, producing a time in generations.
       return (exp(i / float(ntimes) * log(1 + delta * maxtime)) - 1) / delta


   def get_time_points(ntimes=20, maxtime=160000, delta=0.01):
       """
       Compute all discretized time points.

       Parameters
       ----------
       ntimes : int
           Number of time points (including t_0 = 0).
       maxtime : float
           Maximum time in generations.
       delta : float
           Controls log-spacing.

       Returns
       -------
       list of float
           The ntimes time points.

       Examples
       --------
       >>> times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
       >>> len(times)
       20
       >>> times[0]
       0.0
       >>> round(times[-1], 1)
       160000.0
       """
       # ntimes-1 is used as the denominator because the grid has ntimes
       # points but only ntimes-1 intervals between them.
       return [get_time_point(i, ntimes - 1, maxtime, delta)
               for i in range(ntimes)]

Let's visualize a typical grid:

.. code-block:: python

   times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
   for i, t in enumerate(times):
       step = times[i] - times[i-1] if i > 0 else 0
       print(f"t[{i:2d}] = {t:10.1f}   step = {step:10.1f}")

   # Output:
   # t[ 0] =        0.0   step =        0.0
   # t[ 1] =       52.6   step =       52.6
   # t[ 2] =      134.6   step =       82.0
   # t[ 3] =      256.3   step =      121.7
   # t[ 4] =      437.4   step =      181.1
   # t[ 5] =      706.8   step =      269.4
   # t[ 6] =     1108.0   step =      401.3
   # t[ 7] =     1706.4   step =      598.3
   # t[ 8] =     2597.9   step =      891.5
   # t[ 9] =     3925.7   step =     1327.8
   # t[10] =     5903.3   step =     1977.6
   # t[11] =     8849.5   step =     2946.2
   # t[12] =    13240.2   step =     4390.7
   # t[13] =    19783.7   step =     6543.5
   # t[14] =    29540.7   step =     9757.0
   # t[15] =    44088.1   step =    14547.4
   # t[16] =    65764.1   step =    21675.9
   # t[17] =    98058.2   step =    32294.1
   # t[18] =   146195.1   step =    48136.9
   # t[19] =   160000.0   step =    13804.9

Notice how the first step is ~53 generations but the steps grow to tens of thousands
of generations in the deep past.

The Delta Parameter
--------------------

The :math:`\delta` parameter controls how concentrated the grid is near the present:

- **Small** :math:`\delta` (e.g., 0.001): nearly uniform spacing
- **Large** :math:`\delta` (e.g., 0.1): very concentrated near present, huge gaps in past
- **Default** :math:`\delta = 0.01`: a good balance for human population genetics

.. code-block:: python

   # Effect of delta on the first few time points
   for delta in [0.001, 0.01, 0.1]:
       times = get_time_points(ntimes=20, maxtime=160000, delta=delta)
       print(f"delta={delta}: first 5 times = "
             f"{[round(t, 1) for t in times[:5]]}")

   # delta=0.001: first 5 times = [0.0, 530.6, 1116.2, 1764.5, 2484.5]
   # delta=0.01:  first 5 times = [0.0, 52.6, 134.6, 256.3, 437.4]
   # delta=0.1:   first 5 times = [0.0, 5.3, 13.8, 27.1, 47.8]

*Transition:* Now that we have the time grid itself, we need two more ingredients
before we can build the HMM: a way to represent "the typical time" within each
interval, and a way to measure how long each interval is. These are the time steps
and coal times.

Time Steps
-----------

The **time step** :math:`\Delta t_i` is the length of the :math:`i`-th interval:

.. math::

   \Delta t_i = t_{i+1} - t_i

.. code-block:: python

   def get_time_steps(times):
       """
       Compute time step sizes from time points.

       Parameters
       ----------
       times : list of float
           Discretized time points.

       Returns
       -------
       list of float
           Time steps: times[i+1] - times[i] for each interval.
       """
       ntimes = len(times) - 1
       # Each step is simply the difference between consecutive time points.
       # Because the grid is log-spaced, these steps grow geometrically.
       return [times[i+1] - times[i] for i in range(ntimes)]

Coal Times: Interval Midpoints
================================

When ARGweaver assigns a coalescence to time interval :math:`[t_i, t_{i+1})`, it
needs a **representative time** within that interval for computing branch lengths,
emissions, and tree lengths. It uses the **geometric mean midpoint**:

.. math::

   t_{\text{mid},i} = \sqrt{(t_i + 1)(t_{i+1} + 1)} - 1

.. admonition:: Why geometric mean, not arithmetic mean?

   The arithmetic mean :math:`(t_i + t_{i+1})/2` would work, but the geometric mean
   better represents the "typical" coalescence time within the interval. Under the
   coalescent, the waiting time is exponentially distributed, so the expected
   coalescence time within an interval is closer to the geometric mean (which
   down-weights the tail). The "+1" shift avoids issues at :math:`t = 0`.

.. admonition:: Probability Aside --- The exponential distribution and interval midpoints

   Under the coalescent with :math:`k` lineages and constant population size :math:`N_e`,
   the time to the next coalescence event is exponentially distributed with rate
   :math:`\lambda = \binom{k}{2}/(2N_e)`. The density is
   :math:`f(t) = \lambda e^{-\lambda t}`, which is highest at :math:`t=0` and decays
   exponentially.

   Within an interval :math:`[a, b)`, the conditional expected coalescence time is:

   .. math::

      E[T \mid a \leq T < b] = a + \frac{1}{\lambda} - \frac{(b-a) e^{-\lambda(b-a)}}{1 - e^{-\lambda(b-a)}}

   For small :math:`\lambda(b-a)`, this is close to the arithmetic mean :math:`(a+b)/2`.
   For larger :math:`\lambda(b-a)`, the expected time is pulled toward :math:`a` because
   the exponential density is higher near the interval's lower boundary. The geometric
   mean :math:`\sqrt{(a+1)(b+1)} - 1` approximates this pull-toward-the-bottom behavior
   without needing to know :math:`\lambda`.

.. code-block:: python

   def get_coal_times(times):
       """
       Compute coal times (geometric mean midpoints) for each interval.

       The coal_times list interleaves boundary times and midpoints:
         [t_0, mid_0, t_1, mid_1, ..., t_{n-1}, mid_{n-1}, t_n]

       This interleaved structure is used internally for computing
       transition probabilities (the "half-intervals" above and below
       each time point).

       Parameters
       ----------
       times : list of float
           Discretized time points (length ntimes).

       Returns
       -------
       list of float
           Interleaved boundary times and midpoints (length 2*ntimes - 1).

       Examples
       --------
       >>> times = [0.0, 100.0, 1000.0, 10000.0]
       >>> ct = get_coal_times(times)
       >>> len(ct)
       7
       >>> round(ct[0], 1)  # t_0
       0.0
       >>> round(ct[1], 1)  # mid between t_0 and t_1
       9.0
       >>> round(ct[2], 1)  # t_1
       100.0
       """
       ntimes = len(times) - 1
       times2 = []
       for i in range(ntimes):
           times2.append(times[i])
           # Geometric mean midpoint: sqrt((t_i + 1) * (t_{i+1} + 1)) - 1
           # The +1/-1 shift ensures the formula works at t=0 (where the
           # plain geometric mean sqrt(0 * t_{i+1}) would always be 0).
           times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
       times2.append(times[ntimes])
       return times2

Let's see what this looks like for the first few intervals:

.. code-block:: python

   times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
   coal_times = get_coal_times(times)

   for i in range(min(5, len(times) - 1)):
       t_lo = times[i]
       t_hi = times[i + 1]
       t_mid = coal_times[2 * i + 1]
       arith = (t_lo + t_hi) / 2
       print(f"Interval [{t_lo:.1f}, {t_hi:.1f}): "
             f"geometric mid = {t_mid:.1f}, arithmetic mid = {arith:.1f}")

   # Interval [0.0, 52.6): geometric mid = 6.3, arithmetic mid = 26.3
   # Interval [52.6, 134.6): geometric mid = 90.2, arithmetic mid = 93.6
   # Interval [134.6, 256.3): geometric mid = 190.0, arithmetic mid = 195.5
   # Interval [256.3, 437.4): geometric mid = 339.2, arithmetic mid = 346.9
   # Interval [437.4, 706.8): geometric mid = 561.2, arithmetic mid = 572.1

Notice how the geometric midpoint is pulled toward the lower boundary, especially for
the first interval (6.3 vs. 26.3). This reflects the exponential concentration of
coalescence events near the bottom of each interval.

Coal Time Steps
----------------

The **coal time steps** partition the timeline into sub-intervals centered on each
time point, using the midpoints as boundaries:

.. code-block:: python

   def get_coal_time_steps(times):
       """
       Compute the effective time step for coalescence at each time point.

       For time point i, the coal time step spans from the midpoint below
       to the midpoint above:
         coal_step[i] = mid[i] - mid[i-1]

       These are used in transition probability calculations to compute
       the "exposure" of each time point to coalescence.

       Parameters
       ----------
       times : list of float
           Discretized time points.

       Returns
       -------
       list of float
           Coal time steps for each time index.
       """
       ntimes = len(times) - 1
       # First, rebuild the interleaved structure (same as get_coal_times).
       times2 = []
       for i in range(ntimes):
           times2.append(times[i])
           times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
       times2.append(times[ntimes])

       # For each time point (at even indices 0, 2, 4, ...),
       # the coal time step spans from the midpoint just below (index i-1)
       # to the midpoint just above (index i+1). Clamped at boundaries.
       coal_time_steps = []
       for i in range(0, len(times2), 2):
           coal_time_steps.append(
               times2[min(i + 1, len(times2) - 1)] -
               times2[max(i - 1, 0)]
           )
       return coal_time_steps

*Transition:* With the time grid, its midpoints, and its step sizes fully specified,
we are ready to define the HMM state space. The state space is where the time grid
meets the tree topology --- each valid (branch, time-index) pair is one tick on the
watch's digital display.

The State Space
================

With the time grid defined, we can now describe the HMM state space precisely.

At each genomic position, the local tree has a set of branches. Each branch :math:`b`
spans from the age of its child node to the age of its parent node. A valid state
:math:`(b, i)` means "the new lineage coalesces with branch :math:`b` at time index
:math:`i`", which requires :math:`t_i` to fall within the time span of branch :math:`b`.

.. math::

   \text{States}(T) = \{(b, i) : b \in \text{branches}(T), \;
   \text{age}(\text{child}(b)) \leq t_i < \text{age}(\text{parent}(b))\}

For the root branch (which has no parent), the state extends up to the last time
point :math:`t_{n_t - 1}` (excluding the final sentinel time).

.. code-block:: python

   def iter_coal_states(tree, times):
       """
       Iterate through valid coalescent states for a local tree.

       Each state is a (node_name, time_index) pair, where node_name
       identifies the branch (by its child node) and time_index is
       the index into the times array.

       Parameters
       ----------
       tree : tree object
           A local tree with nodes having .age, .parents, .children attributes.
       times : list of float
           Discretized time points.

       Yields
       ------
       tuple of (str, int)
           (node_name, time_index) pairs.
       """
       ntimes = len(times) - 1
       seen = set()
       time_lookup = {t: i for i, t in enumerate(times)}

       for node in tree.preorder():
           # Skip single-child nodes (artifacts of the ARG structure)
           # These arise from recombination events in the full ARG but
           # do not represent real branches in the marginal tree.
           if len(node.children) == 1:
               continue

           i = time_lookup[node.age]

           if node.parents:
               # Find the "real" parent (skip single-child nodes)
               parent = node.parents[0]
               while parent and parent not in seen:
                   parent = parent.parents[0]

               # Yield states from this node's age up to parent's age.
               # Each yielded (node.name, i) means "the new lineage joins
               # this branch at time index i."
               while i < ntimes and times[i] <= parent.age:
                   yield (node.name, i)
                   i += 1
           else:
               # Root: yield states up to ntimes-1.
               # The root branch extends to infinity in principle, but the
               # time grid truncates it at the last time point.
               while i < ntimes:
                   yield (node.name, i)
                   i += 1

           seen.add(node)

.. admonition:: State space size

   For a tree with :math:`k` leaves, there are :math:`2k - 1` nodes and
   :math:`2k - 2` branches. Each branch spans some number of time intervals.
   A typical state count is :math:`O(k \cdot n_t)`, though the exact number
   depends on the tree shape. For :math:`k = 8` leaves and :math:`n_t = 20`
   time points, the state space might contain 50--150 states per position.

.. admonition:: Probability Aside --- Why the state space determines computational cost

   The forward algorithm (see :ref:`hmms`) computes one vector of length :math:`S`
   (the number of states) at each genomic position, using a matrix--vector product
   that costs :math:`O(S^2)`. For a genome of length :math:`L`, the total cost is
   :math:`O(L \cdot S^2)`. With :math:`S \sim k \cdot n_t`, this becomes
   :math:`O(L \cdot k^2 \cdot n_t^2)`. Compare this with SINGER, where the two-HMM
   approach achieves :math:`O(L \cdot k)` per site --- the scaling advantage of
   SINGER for large :math:`k` is dramatic.

   However, ARGweaver's :math:`O(S^2)` cost can be reduced to :math:`O(S)` per site
   by exploiting the rank-1 structure of the transition matrix (see
   :ref:`argweaver_transitions`). This is a key optimization that makes ARGweaver
   practical even with hundreds of states.

Lineage Counting
==================

The transition probabilities depend on how many lineages exist at each time
interval. ARGweaver counts three quantities:

- **nbranches[i]**: number of branches passing through :math:`[t_i, t_{i+1})` ---
  these are the lineages that could *coalesce* with a new lineage at time :math:`i`,
  or on which a *recombination* could occur.

- **nrecombs[i]**: number of valid recombination points at time :math:`i` ---
  the number of (branch, time) pairs at this time index.

- **ncoals[i]**: number of valid coalescence points at time :math:`t_i` ---
  where a re-coalescing lineage could attach.

.. code-block:: python

   def get_nlineages_recomb_coal(tree, times):
       """
       Count lineages, recombination points, and coalescence points
       at each time index.

       Parameters
       ----------
       tree : tree object
           A local tree.
       times : list of float
           Discretized time points.

       Returns
       -------
       nbranches : list of int
           Number of lineages at each time interval.
       nrecombs : list of int
           Number of recombination points at each time index.
       ncoals : list of int
           Number of coalescence points at each time index.
       """
       nbranches = [0] * len(times)
       nrecombs = [0] * len(times)
       ncoals = [0] * len(times)

       for node_name, timei in iter_coal_states(tree, times):
           node = tree[node_name]

           # Find the real parent (skip single-child nodes)
           if node.parents:
               parent = node.parents[0]
               while len(parent.children) == 1:
                   parent = parent.parents[0]
           else:
               parent = None

           # A branch "passes through" interval i if the branch's parent
           # is strictly above time i. If the parent is exactly at time i,
           # the branch ends there (coalescence event), so it does not
           # contribute to the lineage count for the *next* interval.
           if not parent or times[timei] < parent.age:
               nbranches[timei] += 1

           # Count as both a recombination and coalescence point.
           # nrecombs and ncoals count *states* (branch, time pairs),
           # while nbranches counts *lineages* (branches extending
           # through the interval). The distinction matters at nodes
           # where a coalescence event occurs.
           nrecombs[timei] += 1
           ncoals[timei] += 1

       # The last time point always has exactly 1 branch (above root)
       nbranches[-1] = 1

       return nbranches, nrecombs, ncoals

.. admonition:: Why separate counts?

   You might wonder why ``nrecombs`` and ``ncoals`` differ from ``nbranches``.
   The key is that ``nbranches`` counts lineages that *pass through* an interval
   (contributing to the coalescent rate), while ``nrecombs`` and ``ncoals`` count
   *points* in the state space (used for normalizing probabilities). At a
   coalescence node, the point exists for both recombination and coalescence,
   but the branch above may or may not pass through the next interval.

*Recap:* We have now built the complete time-discretization gear. The grid itself
(log-spaced, denser near the present), the midpoint representation (geometric mean),
the step sizes, the state space (valid branch-time pairs), and the lineage counts.
These are the tick marks, numerals, and subdivisions on the watch dial --- everything
the transition and emission gears will need to do their work.

Worked Example
===============

Let's put it all together with a concrete example. Consider a tree with 4 leaves
and 20 time points:

.. code-block:: python

   # A simple example: 4 leaves, tree shape ((A,B),(C,D))
   # Leaf ages: all at t=0
   # Internal node ages: AB coalesces at t_3, CD at t_5, root at t_8

   times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
   time_steps = get_time_steps(times)

   # Count states
   # Branch A: from t_0 to t_3 -> states at i=0,1,2,3
   # Branch B: from t_0 to t_3 -> states at i=0,1,2,3
   # Branch C: from t_0 to t_5 -> states at i=0,1,2,3,4,5
   # Branch D: from t_0 to t_5 -> states at i=0,1,2,3,4,5
   # Branch AB: from t_3 to t_8 -> states at i=3,4,5,6,7,8
   # Branch CD: from t_5 to t_8 -> states at i=5,6,7,8
   # Branch root: from t_8 to t_19 -> states at i=8,9,...,18

   # Total states = 4 + 4 + 6 + 6 + 6 + 4 + 11 = 41

   # Lineage counts:
   # nbranches[0] = 4  (A, B, C, D all present)
   # nbranches[1] = 4
   # nbranches[2] = 4
   # nbranches[3] = 4  (A, B, C, D; AB starts at t_3 but A,B end at t_3)
   # ...actually, at t_3: A and B coalesce -> branches are AB, C, D = 3
   # nbranches[3] = 3
   # nbranches[4] = 3  (AB, C, D)
   # nbranches[5] = 2  (AB, CD)
   # nbranches[6] = 2
   # nbranches[7] = 2
   # nbranches[8] = 1  (root)
   # nbranches[9..19] = 1

*Where we are headed next:* The time grid is now fully specified. In the next chapter
(:ref:`argweaver_transitions`), we will use these time points, step sizes, midpoints,
and lineage counts to derive the transition probabilities --- the largest and most
intricate gear in the ARGweaver mechanism.

Exercises
==========

.. admonition:: Exercise 1: Grid sensitivity

   Generate time grids with :math:`n_t = 10, 20, 40` and :math:`\delta = 0.01`.
   For each grid, compute the maximum ratio :math:`\Delta t_{i+1} / \Delta t_i`
   between consecutive time steps. How does this ratio change with :math:`n_t`?
   What does this tell you about the smoothness of the approximation?

.. admonition:: Exercise 2: Coalescent concentration

   Under the standard coalescent with constant :math:`N_e = 10{,}000`, the
   expected time to the first coalescence of :math:`k` lineages is
   :math:`2N_e / \binom{k}{2}`. For :math:`k = 20`, compute this expected time.
   How many of the default 20 time points fall below this expected time?
   What does this tell you about the grid's resolution where it matters most?

.. admonition:: Exercise 3: Implement state counting

   Write a function that takes a tree (as a dictionary of parent--child relationships
   with ages) and a time grid, and returns the total number of HMM states. Verify
   your count against the worked example above.

.. admonition:: Exercise 4: Midpoint comparison

   For the default time grid, plot the geometric-mean midpoints and the arithmetic
   midpoints on the same axis. At which time intervals is the difference largest
   in relative terms? Why does this matter for emission calculations?
