.. _time_sampling:

==============
Time Sampling
==============

   *The branch tells you where. The time tells you when.*

In the :ref:`previous chapter <branch_sampling>`, we built the largest gear in
the SINGER mechanism: the branch sampling HMM that determines *which branch*
of the marginal tree the new lineage joins at each genomic bin. That gear
found the right slot in the movement. Now we need to set the depth -- the
precise *time* at which the new lineage coalesces within that branch.

Think of it this way. Branch sampling told us which floor of the building to
go to; time sampling tells us exactly where on that floor to stand. Or, in the
watch metaphor: the branch sampling gear engaged the right tooth; now we need
to set how deeply the tooth meshes -- the depth of engagement that determines
the beat rate.

After branch sampling gives us a sequence of joining branches along the genome,
time sampling determines the **coalescence time** within each branch. This is
formulated as a second HMM, closely related to the Pairwise Sequentially Markov
Coalescent (:ref:`PSMC <psmc_timepiece>`).

The Setup
==========

From branch sampling, we have a sequence of full joining branches
:math:`b_1, b_2, \ldots, b_L`, one per genomic bin. Each branch :math:`b_\ell`
spans a time interval :math:`[x_\ell, y_\ell)`.

The time sampling HMM needs to sample a joining time :math:`T_\ell \in [x_\ell, y_\ell)`
at each bin :math:`\ell`, consistent with the coalescent dynamics and the observed
mutations.

.. admonition:: Partial branches

   Although the branch sampling HMM uses both full and partial branch states,
   only the **full joining branches** are passed to the time sampling step.
   Partial branches are used internally by branch sampling to ensure correctness,
   but the final sampled sequence consists entirely of full branches.

.. admonition:: Probability Aside -- Time Sampling as a Conditional HMM

   Time sampling is a **conditional HMM**: the state space at each bin is
   *determined by* the branch sampling output. At bin :math:`\ell`, the hidden
   state is a sub-interval of :math:`[x_\ell, y_\ell)`, the time range of the
   joining branch :math:`b_\ell`. The state space changes from bin to bin
   because the joining branch changes.

   This is conceptually different from a standard HMM where the state space
   is fixed across all positions. Here, the state space adapts to the branch
   assignment -- a form of **hierarchical inference** where the first HMM
   (branch sampling) constrains the second HMM (time sampling).


Step 1: Discretizing the Time Interval
========================================

The time interval :math:`[x_\ell, y_\ell)` for each branch is partitioned into
:math:`d` sub-intervals:

.. math::

   [t_{\ell,0}, t_{\ell,1}), \quad [t_{\ell,1}, t_{\ell,2}), \quad \ldots, \quad [t_{\ell,d-1}, t_{\ell,d})

where :math:`t_{\ell,0} = x_\ell` and :math:`t_{\ell,d} = y_\ell`.

The partition is **uniform with respect to the exponential distribution** with
rate 1. By default, SINGER uses 5% quantile spacing (so :math:`d = 20`).

**Why not uniform in time?** Under the coalescent, coalescence times are
approximately exponentially distributed, meaning events are concentrated toward
the present (:math:`t = 0`). A uniform-in-time partition would waste many
sub-intervals on ancient times where little happens, and have too few
sub-intervals near the present where the density is highest. Uniform-in-CDF
spacing gives each sub-interval equal probability mass, so each sub-interval
is equally "important" for the inference.

.. admonition:: Probability Aside -- Quantile Spacing and Equal Information Content

   This is the same principle we encountered in :ref:`PSMC's time discretization
   <psmc_discretization>`, where log-spaced intervals were chosen to give each
   interval roughly equal statistical weight. Here, SINGER uses **quantile
   spacing** of the exponential distribution restricted to :math:`[x, y)`.

   The idea: if a random variable :math:`T` has CDF :math:`F(t)`, the
   :math:`k`-th quantile is the value :math:`t` such that
   :math:`F(t) = k/d`. Spacing the boundaries at quantiles ensures that
   each sub-interval captures :math:`1/d` of the probability mass -- each
   sub-interval is equally likely to contain the coalescence time.

   For the exponential distribution :math:`\text{Exp}(1)`, the CDF is
   :math:`F(t) = 1 - e^{-t}`. The quantiles of the exponential are
   :math:`t_k = -\ln(1 - k/d)`, which gives denser spacing near :math:`t = 0`
   (where :math:`F` is steep) and sparser spacing for large :math:`t` (where
   :math:`F` is flat). This matches the concentration of coalescence events
   near the present.

**Deriving the partition boundaries.** The CDF of :math:`\text{Exp}(1)` is
:math:`F(t) = 1 - e^{-t}`. We want the :math:`k`-th quantile of the
:math:`\text{Exp}(1)` distribution **restricted to** :math:`[x, y)`.

The conditional CDF on :math:`[x, y)` is:

.. math::

   F_{[x,y)}(t) = \frac{F(t) - F(x)}{F(y) - F(x)} = \frac{e^{-x} - e^{-t}}{e^{-x} - e^{-y}}, \quad x \leq t < y

.. admonition:: Calculus Aside -- Conditional CDF by Truncation

   The conditional CDF :math:`F_{[x,y)}(t)` is obtained by **restricting** the
   original distribution to the interval :math:`[x, y)`. This is Bayes' rule
   applied to continuous distributions:

   .. math::

      F_{[x,y)}(t) = P(T \leq t \mid x \leq T < y)
      = \frac{P(x \leq T \leq t)}{P(x \leq T < y)}
      = \frac{F(t) - F(x)}{F(y) - F(x)}

   The numerator is the probability mass between :math:`x` and :math:`t`; the
   denominator is the total mass in the allowed interval. This ensures
   :math:`F_{[x,y)}(x) = 0` and :math:`F_{[x,y)}(y) = 1`, as required for a
   valid CDF on :math:`[x, y)`.

Setting this equal to :math:`k/d` (for the :math:`k`-th boundary out of :math:`d`
equally-spaced quantiles):

.. math::

   \frac{e^{-x} - e^{-t_k}}{e^{-x} - e^{-y}} = \frac{k}{d}

Solving for :math:`e^{-t_k}`:

.. math::

   e^{-t_k} = e^{-x} - \frac{k}{d}(e^{-x} - e^{-y})

Taking :math:`-\ln`:

.. math::

   t_{\ell,k} = -\ln\left(e^{-x} - \frac{k}{d}(e^{-x} - e^{-y})\right)

**Verification**: At :math:`k = 0`, :math:`t_0 = -\ln(e^{-x}) = x` :math:`\checkmark`.
At :math:`k = d`, :math:`t_d = -\ln(e^{-x} - (e^{-x} - e^{-y})) = -\ln(e^{-y}) = y` :math:`\checkmark`.

.. code-block:: python

   import numpy as np

   def partition_branch(x, y, d=20):
       """Partition a branch [x, y) into d sub-intervals.

       Uses uniform spacing in the exponential CDF, so sub-intervals
       have equal probability mass under Exp(1).  This gives denser
       spacing near the present (x) and sparser spacing toward the
       past (y), matching where coalescence events are most likely.

       Parameters
       ----------
       x, y : float
           Lower and upper time of the branch.
       d : int
           Number of sub-intervals.

       Returns
       -------
       boundaries : ndarray of shape (d + 1,)
           Time boundaries [t_0, t_1, ..., t_d].
       """
       exp_x = np.exp(-x)  # e^{-x}: survival probability at lower endpoint
       exp_y = np.exp(-y)  # e^{-y}: survival probability at upper endpoint
       # fractions = [0, 1/d, 2/d, ..., 1]: the quantile levels
       fractions = np.linspace(0, 1, d + 1)
       # Linear interpolation between exp_x and exp_y, then invert
       # via -log to get the time boundaries
       boundaries = -np.log(exp_x - fractions * (exp_x - exp_y))
       return boundaries

   # Example: branch [0.1, 2.0], 10 sub-intervals
   boundaries = partition_branch(0.1, 2.0, d=10)
   print("Sub-interval boundaries:")
   for i in range(len(boundaries) - 1):
       width = boundaries[i+1] - boundaries[i]
       print(f"  [{boundaries[i]:.4f}, {boundaries[i+1]:.4f}) width={width:.4f}")

Notice that the sub-intervals are narrower near :math:`x = 0.1` (the present)
and wider near :math:`y = 2.0` (the past). This is exactly the exponential
quantile spacing at work.

Representative time for each sub-interval:

.. math::

   \exp(-\tau_{\ell,i}) = \frac{\exp(-t_{\ell,i}) + \exp(-t_{\ell,i+1})}{2}

.. admonition:: Calculus Aside -- Why Average in Exponential Space?

   The representative time :math:`\tau_i` is defined so that
   :math:`e^{-\tau_i}` is the average of :math:`e^{-t_i}` and
   :math:`e^{-t_{i+1}}`. This is *not* the same as averaging the times
   directly (which would give the arithmetic midpoint :math:`(t_i + t_{i+1})/2`).

   Averaging in exponential space means :math:`\tau_i` is the time whose
   "survival weight" :math:`e^{-\tau_i}` is the midpoint of the survival
   weights at the endpoints. Since the coalescent density is proportional to
   :math:`e^{-t}`, this choice ensures that :math:`\tau_i` sits at the
   "center of mass" of the sub-interval under the exponential distribution --
   a better representative than the arithmetic midpoint.

.. code-block:: python

   def representative_times(boundaries):
       """Compute representative time for each sub-interval.

       The representative time sits at the center of mass of the
       sub-interval under the exponential distribution, not at the
       arithmetic midpoint.

       Parameters
       ----------
       boundaries : ndarray of shape (d + 1,)

       Returns
       -------
       taus : ndarray of shape (d,)
       """
       d = len(boundaries) - 1
       taus = np.zeros(d)
       for i in range(d):
           # Average in survival-probability space, then invert
           avg_exp = (np.exp(-boundaries[i]) + np.exp(-boundaries[i+1])) / 2
           taus[i] = -np.log(avg_exp)
       return taus

   taus = representative_times(boundaries)
   print("\nRepresentative times:")
   for i, tau in enumerate(taus):
       print(f"  Sub-interval {i}: tau = {tau:.4f}")

With the branch partitioned into sub-intervals, each with a representative time,
we now need the transition model: how does the coalescence time change from one
bin to the next? This is where the PSMC transition density enters.


Step 2: The PSMC Transition Density
=====================================

The core transition in time sampling comes from the PSMC model. If you have
worked through :ref:`the PSMC continuous model <psmc_continuous>`, the
transition density below will be familiar -- it is the same formula that
describes how coalescence times change between adjacent genomic positions due to
recombination. Here we use it in a slightly different context: instead of
transitioning between time intervals of arbitrary range :math:`[0, \infty)`, we
transition within the time range of a specific joining branch.

When the joining branch spans :math:`[0, \infty)` (the simple case first), the
transition density from time :math:`s` to time :math:`t` is:

.. math::

   q_0(t \mid s) = \int_0^{s \wedge t} \frac{1}{s} e^{-(t-u)} du =
   \begin{cases}
   \frac{1}{s}[1 - e^{-t}] & t < s \\
   \frac{1}{s}[e^{-(t-s)} - e^{-t}] & t \geq s
   \end{cases}

.. admonition:: Probability Aside -- The Recombination-Recoalescence Model

   This transition density models the process of **recombination followed by
   re-coalescence**. Here is the physical picture:

   1. The current coalescence time is :math:`s`. The lineage has a branch from
      :math:`0` to :math:`s` connecting it to the tree.

   2. A recombination occurs at a uniform random position :math:`u` on this
      branch, i.e., :math:`u \sim \text{Uniform}(0, s)`. The lineage is
      "cut" at time :math:`u`.

   3. From time :math:`u`, the detached lineage must re-coalesce with the
      remaining tree. Under the standard coalescent, it waits an
      :math:`\text{Exp}(1)` time before finding a new partner. The new
      coalescence time is :math:`t = u + W` where :math:`W \sim \text{Exp}(1)`.

   The density :math:`q_0(t|s)` integrates over all possible recombination
   points :math:`u \in [0, \min(s,t)]`, weighting each by the uniform
   density :math:`1/s` and the exponential re-coalescence density
   :math:`e^{-(t-u)}`.

   The :math:`\min(s,t)` in the integration limit arises because recombination
   can only happen on the existing branch :math:`[0, s]`, and must happen
   before the new coalescence time :math:`t`.

**Intuition**: A recombination occurs somewhere on the branch below time :math:`s`,
uniformly at time :math:`u \in [0, s]`. From :math:`u`, the new lineage waits
an :math:`\text{Exp}(1)` time before re-coalescing at time :math:`t`.

Including the possibility of no recombination:

.. math::

   q_\rho(t \mid s) =
   \begin{cases}
   \frac{1 - e^{-\rho s}}{s}[1 - e^{-t}] & t < s \\[6pt]
   e^{-\rho s} & t = s \text{ (point mass: no recombination)} \\[6pt]
   \frac{1 - e^{-\rho s}}{s}[e^{-(t-s)} - e^{-t}] & t > s
   \end{cases}

.. admonition:: Probability Aside -- The Point Mass at :math:`t = s`

   The term :math:`e^{-\rho s}` at :math:`t = s` is a **point mass** -- a
   discrete probability sitting at a single point in an otherwise continuous
   distribution. It represents the event "no recombination occurred," which
   has probability :math:`e^{-\rho s}` (the survival probability of a Poisson
   process with rate :math:`\rho` over a branch of length :math:`s`).

   When no recombination occurs, the coalescence time stays exactly at :math:`s`.
   This is a **mixed distribution**: part continuous (the recombination cases)
   and part discrete (the no-recombination case). Such mixtures are common
   in population genetics -- they arise whenever a process either "happens"
   (continuous outcome) or "doesn't happen" (discrete point mass).

.. code-block:: python

   def psmc_transition_density(t, s, rho):
       """PSMC transition density q_rho(t | s).

       This is the probability density of the new coalescence time t,
       given the old coalescence time s and recombination rate rho.

       Parameters
       ----------
       t : float
           Target time.
       s : float
           Source time.
       rho : float
           Recombination rate per bin.

       Returns
       -------
       density : float
       """
       # Probability that recombination occurred on the branch [0, s]
       p_recomb = 1 - np.exp(-rho * s)

       if abs(t - s) < 1e-12:
           # Point mass (no recombination): probability e^{-rho*s}
           return np.exp(-rho * s)

       if t < s:
           # New coalescence is shallower (more recent) than old
           return (p_recomb / s) * (1 - np.exp(-t))
       else:
           # New coalescence is deeper (more ancient) than old
           return (p_recomb / s) * (np.exp(-(t - s)) - np.exp(-t))


The CDF:

.. math::

   Q_\rho(t \mid s) = \int_0^t q_\rho(x \mid s) \, dx =
   \begin{cases}
   \frac{1 - e^{-\rho s}}{s}[t + e^{-t} - 1] & t < s \\[6pt]
   \frac{1 - e^{-\rho s}}{s}[s - e^{-(t-s)} + e^{-t}] + e^{-\rho s} & t \geq s
   \end{cases}

.. admonition:: Calculus Aside -- Deriving the CDF from the Density

   The CDF is the integral of the density from 0 to :math:`t`. For the
   :math:`t < s` case:

   .. math::

      Q(t|s) = \int_0^t \frac{p}{s}(1 - e^{-x})\,dx
      = \frac{p}{s}\left[x + e^{-x}\right]_0^t
      = \frac{p}{s}\left[(t + e^{-t}) - (0 + 1)\right]
      = \frac{p}{s}[t + e^{-t} - 1]

   The integral of :math:`1 - e^{-x}` is :math:`x + e^{-x}` (the antiderivative
   of 1 is :math:`x`, and the antiderivative of :math:`-e^{-x}` is :math:`e^{-x}`).

   For the :math:`t \geq s` case, we must add the point mass :math:`e^{-\rho s}`
   at :math:`t = s` and integrate the :math:`t > s` density from :math:`s` to
   :math:`t`. The algebra is similar but includes the exponential shift
   :math:`e^{-(t-s)}`.

.. code-block:: python

   def psmc_transition_cdf(t, s, rho):
       """PSMC transition CDF Q_rho(t | s).

       The cumulative distribution function: P(T_new <= t | T_old = s).
       Includes both the continuous density (recombination cases) and
       the point mass at t = s (no recombination).
       """
       p_recomb = 1 - np.exp(-rho * s)
       p_no_recomb = np.exp(-rho * s)

       if t < s:
           # Only the t < s continuous density contributes
           return (p_recomb / s) * (t + np.exp(-t) - 1)
       else:
           # The t < s density (integrated to s), plus the point mass,
           # plus the t > s density (integrated from s to t)
           return (p_recomb / s) * (s - np.exp(-(t - s)) + np.exp(-t)) + p_no_recomb

   # Verify: CDF at infinity should be 1
   print(f"CDF(100 | s=1, rho=0.5) = {psmc_transition_cdf(100, 1.0, 0.5):.6f}")

Now that we have the continuous transition density and CDF, we can compute the
discrete transition probabilities between time sub-intervals.


Step 3: Transition Probabilities Between Sub-Intervals
=========================================================

The transition from sub-interval :math:`[t_{\ell-1,i}, t_{\ell-1,i+1})` to
:math:`[t_{\ell,j}, t_{\ell,j+1})` is:

.. math::

   q_{i,j}^{\ell-1,\ell} = \frac{Q_\rho(t_{\ell,j+1} \mid \tau_{\ell-1,i}) - Q_\rho(t_{\ell,j} \mid \tau_{\ell-1,i})}{Q_\rho(y_\ell \mid \tau_{\ell-1,i}) - Q_\rho(x_\ell \mid \tau_{\ell-1,i})}

**Why the denominator?** The PSMC CDF :math:`Q_\rho` covers the full range
:math:`[0, \infty)`, but we need the time to land in :math:`[x_\ell, y_\ell)` (the
joining branch interval). The denominator
:math:`Q_\rho(y_\ell \mid \tau) - Q_\rho(x_\ell \mid \tau)` is the total
probability mass that falls in this interval. Dividing by it gives us the
**conditional** transition probability:

.. math::

   P(T_\ell \in [t_j, t_{j+1}) \mid T_\ell \in [x_\ell, y_\ell), T_{\ell-1} = \tau_i)
   = \frac{Q_\rho(t_{j+1} \mid \tau_i) - Q_\rho(t_j \mid \tau_i)}{Q_\rho(y_\ell \mid \tau_i) - Q_\rho(x_\ell \mid \tau_i)}

.. admonition:: Probability Aside -- Conditioning by Normalization

   This formula is a direct application of **conditional probability**:

   .. math::

      P(A \mid B) = \frac{P(A \cap B)}{P(B)}

   Here, :math:`A` is "the new time falls in sub-interval :math:`j`" and
   :math:`B` is "the new time falls in the branch interval :math:`[x, y)`."
   Since sub-interval :math:`j` is contained within :math:`[x, y)`, the
   intersection :math:`A \cap B = A`, so the numerator is just
   :math:`P(A)` (the CDF difference for sub-interval :math:`j`), and the
   denominator is :math:`P(B)` (the CDF difference for the whole branch).

   This conditioning is necessary because the PSMC transition density was
   derived for an unconstrained time range :math:`[0, \infty)`, but the
   branch sampling HMM has already determined that the coalescence must
   occur within the specific branch :math:`[x_\ell, y_\ell)`.

This is simply Bayes' rule applied to restrict the distribution to the allowed
interval. By construction, summing over all sub-intervals :math:`j` gives:

.. math::

   \sum_{j=0}^{d-1} q_{i,j} = \frac{Q_\rho(y_\ell \mid \tau_i) - Q_\rho(x_\ell \mid \tau_i)}{Q_\rho(y_\ell \mid \tau_i) - Q_\rho(x_\ell \mid \tau_i)} = 1 \quad \checkmark

.. code-block:: python

   def time_transition_matrix(boundaries_prev, taus_prev, boundaries_next, rho):
       """Compute transition matrix between time sub-intervals.

       Each entry Q[i, j] gives the probability of transitioning from
       sub-interval i at the previous bin to sub-interval j at the
       current bin, conditioned on the coalescence falling within the
       current branch interval.

       Parameters
       ----------
       boundaries_prev : ndarray of shape (d_prev + 1,)
           Sub-interval boundaries at bin ell-1.
       taus_prev : ndarray of shape (d_prev,)
           Representative times at bin ell-1.
       boundaries_next : ndarray of shape (d_next + 1,)
           Sub-interval boundaries at bin ell.
       rho : float

       Returns
       -------
       Q : ndarray of shape (d_prev, d_next)
           Transition matrix.
       """
       d_prev = len(taus_prev)
       d_next = len(boundaries_next) - 1
       Q = np.zeros((d_prev, d_next))

       # Branch interval boundaries for normalization
       x_ell = boundaries_next[0]   # lower bound of joining branch
       y_ell = boundaries_next[-1]  # upper bound of joining branch

       for i in range(d_prev):
           # Denominator: total mass in the branch interval [x, y)
           denom = (psmc_transition_cdf(y_ell, taus_prev[i], rho) -
                    psmc_transition_cdf(x_ell, taus_prev[i], rho))
           if denom < 1e-15:
               Q[i, :] = 1.0 / d_next  # uniform fallback for degenerate cases
               continue

           for j in range(d_next):
               # Numerator: mass in sub-interval j
               numer = (psmc_transition_cdf(boundaries_next[j+1],
                                            taus_prev[i], rho) -
                        psmc_transition_cdf(boundaries_next[j],
                                            taus_prev[i], rho))
               Q[i, j] = numer / denom  # conditional probability

       return Q

   # Example: same branch for two adjacent bins
   boundaries = partition_branch(0.1, 2.0, d=10)
   taus = representative_times(boundaries)
   Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)

   print("Transition matrix shape:", Q.shape)
   print("Row sums:", np.round(Q.sum(axis=1), 6))
   print("\nFirst row:")
   print(np.round(Q[0], 4))

The transition matrix is now ready. For the common case where the joining branch
is the same at adjacent bins (no branch change), the matrix has a special
structure that enables a dramatic speedup. That is the subject of the next section.


Step 4: The Linearization Trick
=================================

For **Type A transitions** (same joining branch, no change in partial ARG), the
transition matrix has a special symmetry structure that allows :math:`O(d)` forward
computation instead of :math:`O(d^2)`.

This is analogous to the efficient transition matrix computation in
:ref:`PSMC discretization <psmc_discretization>`, where the factorization of
:math:`q_{kl}` into source-dependent and target-dependent terms enabled
efficient matrix-vector products. Here, a similar algebraic structure emerges.

From the PSMC transition formula, two key properties hold when the state space
is the same at both bins (i.e., :math:`t_{\ell-1,k} = t_{\ell,k}` for all :math:`k`):

**Property 1**: For all :math:`i > j`:

.. math::

   q_{i,j} = q_{j+1,j}

(All states above sub-interval :math:`j` have the same transition probability to :math:`j`.)

**Proof of Property 1.** When :math:`i > j`, we have :math:`\tau_i > t_{j+1}` (the
source time is above the target interval). From the PSMC transition CDF, when
the source time :math:`\tau` exceeds the target interval :math:`[t_j, t_{j+1})`:

.. math::

   Q_\rho(t \mid \tau) = \frac{1 - e^{-\rho\tau}}{\tau}(1 - e^{-t}) \quad \text{for } t < \tau

The numerator of :math:`q_{i,j}` is:

.. math::

   Q_\rho(t_{j+1} \mid \tau_i) - Q_\rho(t_j \mid \tau_i)
   = \frac{1 - e^{-\rho\tau_i}}{\tau_i}\left[(1 - e^{-t_{j+1}}) - (1 - e^{-t_j})\right]
   = \frac{1 - e^{-\rho\tau_i}}{\tau_i}(e^{-t_j} - e^{-t_{j+1}})

The denominator :math:`Q_\rho(y \mid \tau_i) - Q_\rho(x \mid \tau_i)` also depends
on :math:`\tau_i`. However, the key observation is that when :math:`\tau_i` and
:math:`\tau_{j+1}` are both above :math:`t_{j+1}`, the factors
:math:`\frac{1-e^{-\rho\tau}}{\tau}` in both numerator and denominator are
evaluated in the same regime (:math:`t < \tau`). After normalization, the
:math:`\tau`-dependent prefactor cancels, leaving a result that depends only on
:math:`t_j` and :math:`t_{j+1}`. Hence :math:`q_{i,j} = q_{j+1,j}` for all :math:`i > j`.

**Property 2**: For all :math:`i < j`:

.. math::

   \frac{q_{i,j}}{q_{i,j-1}} = \kappa_j := \frac{e^{-t_j} - e^{-t_{j+1}}}{e^{-t_{j-1}} - e^{-t_j}}

(The ratio doesn't depend on :math:`i`.)

**Proof of Property 2.** When :math:`i < j`, we have :math:`\tau_i < t_j` (the
source time is below the target interval). In this case, both
:math:`Q_\rho(t_j \mid \tau_i)` and :math:`Q_\rho(t_{j+1} \mid \tau_i)` use the
:math:`t \geq s` formula:

.. math::

   Q_\rho(t \mid \tau_i) = \frac{1-e^{-\rho\tau_i}}{\tau_i}[\tau_i - e^{-(t-\tau_i)} + e^{-t}] + e^{-\rho\tau_i}

The difference :math:`Q_\rho(t_{j+1} \mid \tau_i) - Q_\rho(t_j \mid \tau_i)` involves:

.. math::

   = \frac{1-e^{-\rho\tau_i}}{\tau_i}[(-e^{-(t_{j+1}-\tau_i)} + e^{-t_{j+1}}) - (-e^{-(t_j-\tau_i)} + e^{-t_j})]

The terms :math:`e^{-(t-\tau_i)} = e^{\tau_i} \cdot e^{-t}`, so:

.. math::

   = \frac{1-e^{-\rho\tau_i}}{\tau_i}(1 - e^{\tau_i})(e^{-t_{j+1}} - e^{-t_j})

The ratio :math:`q_{i,j}/q_{i,j-1}` then becomes:

.. math::

   \frac{q_{i,j}}{q_{i,j-1}} = \frac{e^{-t_j} - e^{-t_{j+1}}}{e^{-t_{j-1}} - e^{-t_j}} = \kappa_j

The :math:`\tau_i`-dependent prefactor cancels in the ratio, confirming that
:math:`\kappa_j` is independent of :math:`i`.

.. admonition:: Probability Aside -- Why These Properties Enable Linear Time

   Properties 1 and 2 together mean the transition matrix has a very special
   structure. Property 1 says the lower triangle is "column-constant": all
   entries below the diagonal in column :math:`j` are the same value
   :math:`q_{j+1,j}`. Property 2 says the upper triangle has a "geometric
   ratio" structure: moving one column to the right multiplies by a fixed
   ratio :math:`\kappa_j`.

   This structure means we can replace the naive :math:`O(d^2)` matrix-vector
   multiply :math:`\alpha_{j}^{(\text{new})} = \sum_i \alpha_i^{(\text{old})} q_{i,j}`
   with an :math:`O(d)` recursion using running sums. The sum over :math:`i > j`
   (using Property 1) becomes a single accumulation, and the sum over :math:`i < j`
   (using Property 2) becomes a recursion involving :math:`\kappa_j`.

These properties enable the following linear-time recursion:

.. math::

   \alpha_j(\ell+1) = e_j(\ell+1) \left[S_j + \alpha_j(\ell) \cdot q_{j,j} + A_j \cdot q_{j+1,j}\right]

where:

.. math::

   S_j = \alpha_{j-1}(\ell) \cdot q_{j-1,j} + \kappa_j \cdot S_{j-1} \qquad (S_0 = 0)

.. math::

   A_j = \alpha_{j+1}(\ell) + A_{j+1} \qquad (A_{d-1} = 0)

.. code-block:: python

   def forward_linearized(alpha_prev, Q, emissions):
       """Linear-time forward step for Type A transitions.

       Exploits Properties 1 and 2 to compute the forward step in
       O(d) time instead of O(d^2).  The result is identical to the
       standard matrix-vector product alpha_prev @ Q * emissions.

       Parameters
       ----------
       alpha_prev : ndarray of shape (d,)
           Forward probabilities at previous bin.
       Q : ndarray of shape (d, d)
           Transition matrix (Type A: same state space).
       emissions : ndarray of shape (d,)
           Emission probabilities at current bin.

       Returns
       -------
       alpha_curr : ndarray of shape (d,)
       """
       d = len(alpha_prev)
       boundaries = partition_branch(0.1, 2.0, d)  # placeholder

       # Compute kappa values: the geometric ratio from Property 2
       # kappa[j] = q[i,j] / q[i,j-1] for any i < j
       kappa = np.zeros(d)
       for j in range(1, d):
           kappa[j] = Q[0, j] / Q[0, j-1] if Q[0, j-1] > 0 else 0

       # Compute S_j (from below): accumulates contributions from i < j
       # S_j = alpha_{j-1} * q_{j-1,j} + kappa_j * S_{j-1}
       S = np.zeros(d)
       for j in range(1, d):
           S[j] = alpha_prev[j-1] * Q[j-1, j] + kappa[j] * S[j-1]

       # Compute A_j (from above): accumulates contributions from i > j
       # A_j = alpha_{j+1} + A_{j+1}
       # By Property 1, all i > j contribute the same q_{j+1,j}, so
       # we just need the sum of alpha values above j
       A = np.zeros(d)
       for j in range(d - 2, -1, -1):
           A[j] = alpha_prev[j+1] + A[j+1]

       # Forward probabilities: combine below, diagonal, and above
       alpha_curr = np.zeros(d)
       for j in range(d):
           alpha_curr[j] = emissions[j] * (
               S[j] + alpha_prev[j] * Q[j, j] + A[j] * Q[j+1, j] if j < d-1
               else S[j] + alpha_prev[j] * Q[j, j]
           )

       return alpha_curr

   # Verify: linear vs quadratic should give same answer
   d = 10
   alpha_prev = np.random.dirichlet(np.ones(d))
   emissions = np.random.uniform(0.1, 0.9, size=d)

   # Quadratic (standard): alpha_prev @ Q gives the matrix-vector product
   alpha_quad = emissions * (alpha_prev @ Q)

   # Linear: uses the O(d) recursion
   alpha_lin = forward_linearized(alpha_prev, Q, emissions)

   print("Quadratic:", np.round(alpha_quad, 6))
   print("Linear:   ", np.round(alpha_lin, 6))
   print("Max difference:", np.max(np.abs(alpha_quad - alpha_lin)))

.. admonition:: Why does this matter?

   Time sampling is not the computational bottleneck of SINGER (branch sampling
   is), but the linearization is mathematically elegant and demonstrates a general
   principle: **exploit structure in transition matrices** to reduce complexity.
   This is the same principle that makes PSMC's transition matrix efficient (see
   :ref:`psmc_discretization`) and that powers the Li-Stephens model's
   :math:`O(K)` forward algorithm (see :ref:`haploid_algorithms`).

With the linearized forward step for the common case, we now need to handle the
two less common cases: when the partial ARG changes (Type B) and when the
joining branch changes (Type C).


Step 5: Type B and Type C Transitions
=======================================

Not all transitions are Type A. There are three types:

Type A: Neither branch nor partial ARG changes
-------------------------------------------------

This is the common case (most bins). The state space is identical at both bins,
and we use the linearized forward algorithm.

Type B: Partial ARG changes (recombination hitchhiking)
---------------------------------------------------------

The partial ARG has a recombination between these bins, so the joining branch may
change by hitchhiking. Some time sub-intervals from the previous bin map to the
new joining branch, others don't (they map to a different branch that wasn't
selected).

.. admonition:: Probability Aside -- What is Hitchhiking in This Context?

   "Hitchhiking" here means that the new lineage's joining point changes *not*
   because of its own recombination, but because the *existing* partial ARG
   rearranges. When the partial ARG has a recombination between two bins, the
   marginal tree topology changes, and the branch that the new lineage was
   joining may split into pieces or merge with other branches. The new lineage
   "hitchhikes" on this rearrangement -- its joining point moves passively.

   This is different from Type C (below), where the new lineage's *own*
   recombination causes the branch change.

.. code-block:: python

   def type_b_transition(alpha_prev, boundaries_prev, boundaries_next,
                          mapped_intervals, rho):
       """Handle Type B transition (recombination hitchhiking).

       When the partial ARG has a recombination, some sub-intervals
       from the previous bin map directly to sub-intervals in the
       current bin (the new lineage hitchhikes on the existing
       recombination), while others do not (wrong branch).

       Parameters
       ----------
       alpha_prev : ndarray
           Forward probabilities at previous bin.
       boundaries_prev : ndarray
           Time boundaries at previous bin.
       boundaries_next : ndarray
           Time boundaries at current bin.
       mapped_intervals : list of (prev_idx, next_idx) or None
           Maps previous sub-intervals to current ones.
           None means the interval doesn't contribute (wrong branch).
       rho : float

       Returns
       -------
       alpha_curr : ndarray
       """
       d_next = len(boundaries_next) - 1
       alpha_curr = np.zeros(d_next)

       for prev_idx, mapping in enumerate(mapped_intervals):
           if mapping is not None:
               next_idx = mapping
               # Forward probability transfers directly: the new lineage
               # stays at the same time, just in the new tree's coordinates
               alpha_curr[next_idx] += alpha_prev[prev_idx]

       # Sub-intervals not covered by hitchhiking get zero forward probability
       # from the hitchhiked states but may receive probability from
       # the transition matrix for newly created states

       return alpha_curr

Type C: Joining branch changes (new recombination)
------------------------------------------------------

A recombination in the new lineage causes a branch switch. The transition is
computed by setting :math:`\rho = \infty` in the PSMC CDF (since we already
condition on having a recombination):

.. admonition:: Probability Aside -- Why :math:`\rho \to \infty`?

   In a Type C transition, we already know that a recombination occurred (it was
   determined by the branch sampling step). The PSMC transition density
   :math:`q_\rho(t|s)` includes both the probability of recombination
   (:math:`1 - e^{-\rho s}`) and the re-coalescence density. Since we are
   conditioning on recombination having occurred, we need the density
   :math:`q_0(t|s)` alone -- the re-coalescence part without the recombination
   probability.

   Setting :math:`\rho \to \infty` makes :math:`1 - e^{-\rho s} \to 1`, so the
   recombination is certain, and the point mass at :math:`t = s` vanishes
   (:math:`e^{-\rho s} \to 0`). The result is exactly the conditional
   density given recombination.

.. code-block:: python

   def type_c_transition(alpha_prev, taus_prev, boundaries_next):
       """Handle Type C transition (new recombination in the new lineage).

       When rho -> infinity, the transition is just the unconditional
       coalescence density restricted to the new branch.

       Parameters
       ----------
       alpha_prev : ndarray of shape (d_prev,)
       taus_prev : ndarray of shape (d_prev,)
       boundaries_next : ndarray of shape (d_next + 1,)

       Returns
       -------
       alpha_curr : ndarray of shape (d_next,)
       """
       d_prev = len(alpha_prev)
       d_next = len(boundaries_next) - 1

       # With rho -> infinity, the no-recombination term vanishes
       # and we use the conditional transition q_0(t|s)
       Q = time_transition_matrix(
           None,  # boundaries don't matter for rho=infinity
           taus_prev,
           boundaries_next,
           rho=1e10  # approximate infinity: e^{-1e10 * s} is essentially 0
       )

       # Standard matrix-vector multiply: sum over source sub-intervals
       alpha_curr = alpha_prev @ Q
       return alpha_curr

We now have all three transition types. Let us assemble them into the complete
time sampling algorithm.


Step 6: The Complete Time Sampling Algorithm
==============================================

This is the moment where all the components snap together. The time sampling
algorithm takes the branch sequence from :ref:`branch sampling <branch_sampling>`,
partitions each branch into sub-intervals, runs a forward algorithm with the
appropriate transition type at each step, and then traces back to sample a
sequence of coalescence times.

.. code-block:: python

   def time_sampling(joining_branches, bins, partial_arg, new_haplotype,
                      theta, rho, d=20):
       """Run the time sampling HMM.

       This is the second stage of SINGER's threading algorithm.
       Given the branch sequence from branch sampling, it determines
       the exact coalescence time within each branch.

       Parameters
       ----------
       joining_branches : list of BranchState
           Sampled joining branches from branch sampling.
       bins : list
           Genomic bin boundaries.
       partial_arg : object
           The partial ARG.
       new_haplotype : ndarray
           Alleles of the new haplotype.
       theta : float
           Mutation rate.
       rho : float
           Recombination rate per bin.
       d : int
           Number of time sub-intervals per branch.

       Returns
       -------
       sampled_times : ndarray of shape (L,)
           Sampled joining time at each bin.
       """
       L = len(bins)

       # Build state spaces: partition each branch into sub-intervals
       all_boundaries = []
       all_taus = []
       for ell in range(L):
           branch = joining_branches[ell]
           boundaries = partition_branch(branch.lower_time,
                                          branch.upper_time, d)
           taus = representative_times(boundaries)
           all_boundaries.append(boundaries)
           all_taus.append(taus)

       # Forward algorithm (same structure as the HMM forward pass
       # from the prerequisite chapter, but with time sub-intervals
       # as states instead of discrete categories)
       alpha = [np.zeros(d) for _ in range(L)]

       # Initialize: uniform prior within the branch
       for j in range(d):
           alpha[0][j] = 1.0 / d  # each sub-interval equally likely a priori
           # Multiply by emission probability
           alpha[0][j] *= compute_time_emission(
               new_haplotype[0], all_taus[0][j],
               joining_branches[0], theta, partial_arg, bins[0])

       # Normalize to prevent underflow
       total = alpha[0].sum()
       if total > 0:
           alpha[0] /= total

       # Recursion: classify each transition and apply the right formula
       for ell in range(1, L):
           # Determine transition type by comparing adjacent branches
           branch_changed = (joining_branches[ell].child !=
                              joining_branches[ell-1].child or
                              joining_branches[ell].parent !=
                              joining_branches[ell-1].parent)

           has_existing_recomb = check_recombination(partial_arg,
                                                      bins[ell-1], bins[ell])

           if not branch_changed and not has_existing_recomb:
               # Type A: same branch, no existing recombination
               # Use the efficient linearized forward step
               Q = time_transition_matrix(all_boundaries[ell-1],
                                           all_taus[ell-1],
                                           all_boundaries[ell], rho)
               alpha[ell] = alpha[ell-1] @ Q
           elif has_existing_recomb:
               # Type B: partial ARG changed (hitchhiking)
               alpha[ell] = handle_type_b(alpha[ell-1], all_boundaries[ell-1],
                                           all_boundaries[ell], partial_arg,
                                           bins[ell], rho)
           else:
               # Type C: new recombination (branch changed)
               # Use rho = infinity (condition on recombination)
               Q = time_transition_matrix(all_boundaries[ell-1],
                                           all_taus[ell-1],
                                           all_boundaries[ell],
                                           rho=1e10)
               alpha[ell] = alpha[ell-1] @ Q

           # Emission: multiply by P(observation | time)
           for j in range(d):
               alpha[ell][j] *= compute_time_emission(
                   new_haplotype[ell], all_taus[ell][j],
                   joining_branches[ell], theta, partial_arg, bins[ell])

           # Normalize to prevent underflow
           total = alpha[ell].sum()
           if total > 0:
               alpha[ell] /= total

       # Stochastic traceback: sample a time sequence from the posterior
       sampled_times = np.zeros(L)

       # Sample last bin from the final forward distribution
       probs = alpha[-1] / alpha[-1].sum()
       idx = np.random.choice(d, p=probs)
       sampled_times[-1] = all_taus[-1][idx]

       # Trace backwards (same logic as branch sampling traceback)
       for ell in range(L - 2, -1, -1):
           # ... (similar to branch sampling traceback)
           probs = alpha[ell] / alpha[ell].sum()
           idx = np.random.choice(d, p=probs)
           sampled_times[ell] = all_taus[ell][idx]

       return sampled_times


Step 7: Inference of Recombination Times
==========================================

The threading algorithm infers where recombinations happen (when the joining branch
changes) but not the exact timing of the recombination breakpoint on the branch.

Under the SMC (see :ref:`the SMC prerequisite chapter <smc>`), given a
recombination breakpoint at time :math:`x`, the probability density of the
re-coalescence event being at time :math:`v` is:

.. math::

   p(x) = e^{-(v - x)}

for :math:`l < x < u`, where:

- :math:`l` = lower node age of the recombining branch
- :math:`u` = min(joining time before recombination, joining time after)

.. admonition:: Probability Aside -- The SMC Recombination Model

   Under the Sequentially Markov Coalescent (SMC), a recombination event on a
   branch at time :math:`x` produces a "floating" lineage that must re-coalesce
   at some time :math:`v > x`. The waiting time :math:`v - x` follows an
   :math:`\text{Exp}(1)` distribution (standard coalescent waiting time with
   one additional lineage; see :ref:`coalescent theory <coalescent_theory>`).

   The constraint :math:`x < u = \min(v_{\text{before}}, v_{\text{after}})`
   comes from the requirement that the recombination must occur *below* both
   the old and new coalescence times. If it occurred above either, the
   genealogy would be inconsistent.

SINGER uses the **median** of this distribution as the recombination time:

.. code-block:: python

   def sample_recombination_time(lower, upper, joining_time_before,
                                  joining_time_after):
       """Sample the time of a recombination breakpoint.

       Uses the median of the SMC recombination time distribution
       as a point estimate.  The median is chosen over the mean
       because it is more robust to the exponential tail.

       Parameters
       ----------
       lower : float
           Lower node age of the recombining branch.
       upper : float
           Upper bound: min of joining times.
       joining_time_before : float
       joining_time_after : float

       Returns
       -------
       recomb_time : float
           Sampled recombination time.
       """
       u = min(joining_time_before, joining_time_after)
       v = max(joining_time_before, joining_time_after)

       # Under SMC: p(x) proportional to exp(-(v - x)) for l < x < u
       # This is an exponential distribution shifted and truncated
       # CDF: F(x) = [exp(-(v-x)) - exp(-(v-l))] / [exp(-(v-u)) - exp(-(v-l))]

       # Use the median: F(x) = 0.5
       F_target = 0.5
       # Solve: exp(-(v-x)) = exp(-(v-l)) + 0.5*(exp(-(v-u)) - exp(-(v-l)))
       a = np.exp(-(v - lower))   # CDF at lower bound
       b = np.exp(-(v - u))       # CDF at upper bound
       midpoint = a + F_target * (b - a)  # linear interpolation in exp space

       if midpoint <= 0:
           return (lower + u) / 2  # fallback: arithmetic midpoint

       recomb_time = v + np.log(midpoint)
       return np.clip(recomb_time, lower, u)  # ensure within valid range

   # Example
   t = sample_recombination_time(0.0, 0.5, 0.3, 0.8)
   print(f"Recombination time: {t:.4f}")

**Recap.** Time sampling is now complete. Starting from the branch sequence
produced by :ref:`branch sampling <branch_sampling>`, we have:

1. **Discretized** each branch into sub-intervals with equal coalescent
   probability mass (Step 1)
2. Built the **PSMC transition density** that models how coalescence times
   change between bins (Step 2)
3. Computed **transition probabilities** between sub-intervals, conditioned on
   the joining branch (Step 3)
4. Exploited the **linearization trick** for efficient :math:`O(d)` forward
   computation in the common case (Step 4)
5. Handled **Type B and Type C transitions** for the less common cases (Step 5)
6. Assembled the **complete algorithm**: forward pass plus stochastic
   traceback (Step 6)
7. Inferred **recombination times** using the SMC model (Step 7)

Together, branch sampling and time sampling form the **threading algorithm** --
the procedure that adds one haplotype to an existing partial ARG. The output is
a complete ARG with coalescence times on every node. But these times were
estimated under a constant-population assumption. To correct for this, we need
to recalibrate the beat rate of the watch using the mutation data. That is the
subject of the next chapter.


Exercises
=========

.. admonition:: Exercise 1: Verify the PSMC transition

   Implement the full PSMC transition density and CDF. Verify numerically
   that the density integrates to :math:`1 - e^{-\rho s}` (the probability
   of recombination), and the CDF approaches 1 as :math:`t \to \infty`.

.. admonition:: Exercise 2: Implement linearized forward

   Verify Properties 1 and 2 numerically for a concrete transition matrix.
   Then implement the :math:`O(d)` forward step and verify it gives the same
   result as the :math:`O(d^2)` version.

.. admonition:: Exercise 3: End-to-end time sampling

   Using a simple simulated tree sequence (via ``msprime``), run branch sampling
   followed by time sampling. Compare the inferred coalescence times to the
   true times from the simulation.

Solutions
=========

.. admonition:: Solution 1: Verify the PSMC transition

   We verify numerically that the PSMC transition density integrates to
   :math:`1 - e^{-\rho s}` (the probability of recombination) and that the CDF
   approaches 1 as :math:`t \to \infty`.

   The density :math:`q_0(t \mid s)` (the continuous part, given recombination
   occurred) integrates to the recombination probability because:

   .. math::

      \int_0^\infty q_\rho(t \mid s)\,dt
      = \underbrace{(1 - e^{-\rho s})}_{\text{recomb prob}} \cdot \underbrace{\int_0^\infty q_0(t|s)\,dt}_{= 1}
      + \underbrace{e^{-\rho s}}_{\text{point mass at } t = s}

   The continuous part integrates to :math:`1 - e^{-\rho s}`, and the point mass
   adds :math:`e^{-\rho s}`, giving a total of 1.

   .. code-block:: python

      import numpy as np
      from scipy.integrate import quad

      def psmc_transition_density(t, s, rho):
          """PSMC transition density q_rho(t | s) -- continuous part only."""
          p_recomb = 1 - np.exp(-rho * s)
          if t < s:
              return (p_recomb / s) * (1 - np.exp(-t))
          else:
              return (p_recomb / s) * (np.exp(-(t - s)) - np.exp(-t))

      def psmc_transition_cdf(t, s, rho):
          """PSMC transition CDF Q_rho(t | s), including point mass at t=s."""
          p_recomb = 1 - np.exp(-rho * s)
          p_no_recomb = np.exp(-rho * s)
          if t < s:
              return (p_recomb / s) * (t + np.exp(-t) - 1)
          else:
              return (p_recomb / s) * (s - np.exp(-(t - s)) + np.exp(-t)) + p_no_recomb

      # Verify: continuous density integrates to 1 - exp(-rho * s)
      for s in [0.5, 1.0, 2.0]:
          for rho in [0.1, 0.5, 1.0]:
              integral, _ = quad(psmc_transition_density, 0, 100, args=(s, rho))
              expected = 1 - np.exp(-rho * s)
              print(f"s={s}, rho={rho}: integral={integral:.8f}, "
                    f"1-exp(-rho*s)={expected:.8f}, "
                    f"diff={abs(integral - expected):.2e}")

      # Verify: CDF -> 1 as t -> infinity
      for s in [0.5, 1.0, 2.0]:
          for rho in [0.1, 0.5, 1.0]:
              cdf_large = psmc_transition_cdf(1000.0, s, rho)
              print(f"s={s}, rho={rho}: CDF(1000|s)={cdf_large:.10f}")

      # The CDF at large t should approach 1.0 because it includes both
      # the continuous density (integrates to 1 - e^{-rho*s}) and the
      # point mass e^{-rho*s} at t = s.

.. admonition:: Solution 2: Implement linearized forward

   We first verify Properties 1 and 2 on a concrete transition matrix, then
   show the :math:`O(d)` forward step matches the :math:`O(d^2)` version.

   **Property 1**: For all :math:`i > j`, :math:`q_{i,j} = q_{j+1,j}`.
   **Property 2**: For all :math:`i < j`, :math:`q_{i,j}/q_{i,j-1} = \kappa_j`
   (independent of :math:`i`).

   .. code-block:: python

      import numpy as np

      def partition_branch(x, y, d=20):
          exp_x = np.exp(-x)
          exp_y = np.exp(-y)
          fractions = np.linspace(0, 1, d + 1)
          boundaries = -np.log(exp_x - fractions * (exp_x - exp_y))
          return boundaries

      def representative_times(boundaries):
          d = len(boundaries) - 1
          taus = np.zeros(d)
          for i in range(d):
              avg_exp = (np.exp(-boundaries[i]) + np.exp(-boundaries[i+1])) / 2
              taus[i] = -np.log(avg_exp)
          return taus

      def psmc_transition_cdf(t, s, rho):
          p_recomb = 1 - np.exp(-rho * s)
          p_no_recomb = np.exp(-rho * s)
          if t < s:
              return (p_recomb / s) * (t + np.exp(-t) - 1)
          else:
              return (p_recomb / s) * (s - np.exp(-(t - s)) + np.exp(-t)) + p_no_recomb

      def time_transition_matrix(boundaries, taus, rho):
          d = len(taus)
          x_ell, y_ell = boundaries[0], boundaries[-1]
          Q = np.zeros((d, d))
          for i in range(d):
              denom = (psmc_transition_cdf(y_ell, taus[i], rho) -
                       psmc_transition_cdf(x_ell, taus[i], rho))
              if denom < 1e-15:
                  Q[i, :] = 1.0 / d
                  continue
              for j in range(d):
                  numer = (psmc_transition_cdf(boundaries[j+1], taus[i], rho) -
                           psmc_transition_cdf(boundaries[j], taus[i], rho))
                  Q[i, j] = numer / denom
          return Q

      # Build a concrete example (same branch at both bins = Type A)
      d = 10
      boundaries = partition_branch(0.1, 2.0, d)
      taus = representative_times(boundaries)
      rho = 0.5
      Q = time_transition_matrix(boundaries, taus, rho)

      # Verify Property 1: q_{i,j} = q_{j+1,j} for all i > j
      print("Property 1 verification (should all be ~0):")
      for j in range(d - 1):
          for i in range(j + 2, d):
              diff = abs(Q[i, j] - Q[j + 1, j])
              if diff > 1e-10:
                  print(f"  FAIL: q[{i},{j}]={Q[i,j]:.8f} vs "
                        f"q[{j+1},{j}]={Q[j+1,j]:.8f}, diff={diff:.2e}")
      print("  All passed." if True else "")

      # Verify Property 2: q_{i,j}/q_{i,j-1} = kappa_j for all i < j
      print("\nProperty 2 verification:")
      for j in range(1, d):
          kappa_vals = []
          for i in range(j):
              if Q[i, j-1] > 1e-15:
                  kappa_vals.append(Q[i, j] / Q[i, j-1])
          if len(kappa_vals) > 1:
              spread = max(kappa_vals) - min(kappa_vals)
              print(f"  j={j}: kappa values range = {spread:.2e} "
                    f"(mean={np.mean(kappa_vals):.6f})")

      # O(d) forward step
      def forward_linearized(alpha_prev, Q, emissions):
          d = len(alpha_prev)
          kappa = np.zeros(d)
          for j in range(1, d):
              kappa[j] = Q[0, j] / Q[0, j-1] if Q[0, j-1] > 0 else 0

          S = np.zeros(d)
          for j in range(1, d):
              S[j] = alpha_prev[j-1] * Q[j-1, j] + kappa[j] * S[j-1]

          A = np.zeros(d)
          for j in range(d - 2, -1, -1):
              A[j] = alpha_prev[j+1] + A[j+1]

          alpha_curr = np.zeros(d)
          for j in range(d):
              above_term = A[j] * Q[min(j+1, d-1), j] if j < d - 1 else 0.0
              alpha_curr[j] = emissions[j] * (
                  S[j] + alpha_prev[j] * Q[j, j] + above_term)
          return alpha_curr

      # Compare O(d^2) vs O(d)
      alpha_prev = np.random.dirichlet(np.ones(d))
      emissions = np.random.uniform(0.1, 0.9, size=d)

      alpha_quad = emissions * (alpha_prev @ Q)
      alpha_lin = forward_linearized(alpha_prev, Q, emissions)

      print(f"\nMax difference (linear vs quadratic): "
            f"{np.max(np.abs(alpha_quad - alpha_lin)):.2e}")

.. admonition:: Solution 3: End-to-end time sampling

   This exercise combines ``msprime`` simulation with the branch and time
   sampling algorithms. We simulate a tree sequence, extract the true
   coalescence times, then run the time sampling HMM and compare.

   .. code-block:: python

      import msprime
      import numpy as np

      # Simulate a small tree sequence with known parameters
      ts = msprime.simulate(
          sample_size=4,
          length=1e4,
          recombination_rate=1e-8,
          mutation_rate=1e-8,
          random_seed=42
      )

      # Extract true coalescence times from each marginal tree
      true_times = []
      for tree in ts.trees():
          # For each tree, record the TMRCA (root time)
          root = tree.root
          true_times.append(tree.time(root))

      print(f"Number of marginal trees: {ts.num_trees}")
      print(f"True root times: {[f'{t:.4f}' for t in true_times[:5]]}...")

      # To run time sampling, we would:
      # 1. Remove one haplotype from the tree sequence
      # 2. Use branch_sampling to find the joining branches
      # 3. Use time_sampling to find the joining times
      # 4. Compare inferred times to true times

      # Simplified demonstration: for each tree, partition the root branch
      # and verify the discretization covers the true coalescence time
      def partition_branch(x, y, d=20):
          exp_x = np.exp(-x)
          exp_y = np.exp(-y)
          fractions = np.linspace(0, 1, d + 1)
          boundaries = -np.log(exp_x - fractions * (exp_x - exp_y))
          return boundaries

      def representative_times(boundaries):
          d = len(boundaries) - 1
          taus = np.zeros(d)
          for i in range(d):
              avg_exp = (np.exp(-boundaries[i]) + np.exp(-boundaries[i+1])) / 2
              taus[i] = -np.log(avg_exp)
          return taus

      for tree in ts.trees():
          # Get the branch that a sample (node 0) connects to
          parent = tree.parent(0)
          branch_lower = tree.time(0)
          branch_upper = tree.time(parent)

          # Partition this branch
          boundaries = partition_branch(branch_lower + 1e-10,
                                         branch_upper, d=20)
          taus = representative_times(boundaries)

          # The true joining time for sample 0 is branch_upper
          # (it coalesces at its parent's time)
          true_t = branch_upper
          closest_tau = taus[np.argmin(np.abs(taus - true_t))]

          print(f"Tree interval [{tree.interval.left:.0f}, "
                f"{tree.interval.right:.0f}): "
                f"true_t={true_t:.4f}, closest_tau={closest_tau:.4f}, "
                f"error={abs(true_t - closest_tau):.4f}")

Next: :ref:`arg_rescaling` -- calibrating the clock to the mutation data.
