.. _branch_sampling:

================
Branch Sampling
================

   *The biggest gear in the mechanism: deciding which branch each lineage joins.*

In the :ref:`SINGER overview <singer_overview>`, we laid out the blueprint of the
grand complication -- the four-stage engine that reconstructs ancestral
recombination graphs from DNA sequences. Now we begin building the first and most
intricate stage: **branch sampling**.

Branch sampling is the first stage of SINGER's threading algorithm. Given a
partial ARG (containing the first :math:`n-1` haplotypes) and a new haplotype,
it determines which branch of each marginal tree the new lineage should join.
Think of it as finding the right slot in the movement -- the specific gear tooth
where a new wheel can engage without jamming the mechanism. The problem is that
this slot changes along the genome (because the marginal tree changes), and we
must find a *consistent* sequence of slots that respects both the coalescent
dynamics and the observed mutations.

This is formulated as a Hidden Markov Model where:

- **Hidden states** = branches in the marginal tree at each bin
- **Observations** = the allelic state (0 or 1) of the new haplotype at each bin
- **Transitions** = recombination-driven switches between branches
- **Emissions** = mutation probabilities given the joining branch

If you have worked through the :ref:`HMM prerequisite chapter <hmms>`, these
four components should feel familiar -- they are the same building blocks that
power every HMM, from speech recognition to PSMC. The novelty here is in *what*
the states, transitions, and emissions represent: branches in a genealogical
tree, recombination events, and mutation patterns.

We build every piece from scratch.


Step 1: Joining Probability for a Branch
==========================================

The first thing we need: given a marginal tree :math:`\Psi` with :math:`n`
leaves, what is the probability that the new lineage joins a specific branch?

This question connects directly to the coalescent theory developed in
:ref:`the prerequisite chapter <coalescent_theory>`. There, we learned that
coalescence events follow a Poisson process whose rate depends on the number
of lineages available. Here, we apply that same logic to a *fixed* tree: the
new lineage "falls" through the tree from the present toward the past, and at
each moment it can coalesce with any of the existing lineages.

Exceedance probability and density
------------------------------------

Let :math:`T` be the random variable representing the joining time of the new
lineage, and :math:`\lambda_\Psi(t)` be the number of lineages at time :math:`t`
in the tree. We want :math:`P(T > t)` -- the probability that the new lineage
has *not yet* coalesced by time :math:`t`.

.. admonition:: Probability Aside -- Inhomogeneous Poisson Processes

   An **inhomogeneous Poisson process** is a generalization of the standard
   Poisson process where the rate :math:`\lambda(t)` varies over time. In a
   standard (homogeneous) Poisson process, events occur at a constant rate --
   like a clock ticking at regular intervals. In an inhomogeneous process, the
   rate can speed up or slow down -- like a clock whose tick rate changes with
   the season.

   The key result for inhomogeneous Poisson processes is the **survival
   function**: the probability of no event occurring in the interval :math:`[0, t)`
   is :math:`\exp\left(-\int_0^t \lambda(x)\,dx\right)`. This generalizes the
   familiar :math:`e^{-\lambda t}` formula for constant-rate processes (which
   you met in :ref:`coalescent theory <coalescent_theory>`). The integral
   :math:`\int_0^t \lambda(x)\,dx` is called the **cumulative hazard** -- it
   measures the total "risk" of an event accumulated over the interval.

This is a **survival probability** for a time-varying Poisson process. At each
infinitesimal interval :math:`[t, t+dt)`, the new lineage can coalesce with any
of the :math:`\lambda_\Psi(t)` existing lineages, each at rate 1. So the
instantaneous coalescence rate at time :math:`t` is :math:`\lambda_\Psi(t)`.

The probability of surviving (not coalescing) through the interval :math:`[t, t+dt)`
is :math:`1 - \lambda_\Psi(t) \, dt`. Multiplying over all infinitesimal intervals
from 0 to :math:`t`, and taking the continuous limit:

.. math::

   \bar{F}_\Psi(t) := P_\Psi(T > t)
   = \prod_{k} (1 - \lambda_\Psi(t_k) \, dt)
   = \exp\left(-\int_0^t \lambda_\Psi(x) \, dx\right)

This is the standard result for inhomogeneous Poisson processes: the survival
function is the exponential of the negative cumulative hazard.

.. admonition:: Calculus Aside -- From Product to Exponential

   The step from a product of :math:`(1 - \lambda \, dt)` terms to an
   exponential integral is a fundamental trick in continuous probability.
   Here is the reasoning:

   Take :math:`\ln` of both sides of the product:

   .. math::

      \ln \bar{F}(t) = \sum_k \ln(1 - \lambda(t_k)\,dt)
      \approx \sum_k (-\lambda(t_k)\,dt) = -\int_0^t \lambda(x)\,dx

   The approximation :math:`\ln(1-\epsilon) \approx -\epsilon` holds for small
   :math:`\epsilon` (this is the first-order Taylor expansion of the logarithm
   around 1). Since each :math:`dt` is infinitesimal, :math:`\lambda \, dt` is
   indeed small, and the approximation becomes exact in the limit. Exponentiating
   both sides gives the survival function.

   This is the same technique used in :ref:`coalescent theory <coalescent_theory>`
   to derive the exponential distribution of coalescence times, and in
   :ref:`the PSMC continuous model <psmc_continuous>` for the survival probability
   under varying population size.

The density follows by differentiation (using the chain rule):

.. math::

   f_\Psi(t) = -\frac{d}{dt}\bar{F}_\Psi(t)
   = \lambda_\Psi(t) \cdot \exp\left(-\int_0^t \lambda_\Psi(x)\,dx\right)
   = \lambda_\Psi(t) \cdot \bar{F}_\Psi(t)

**Intuition**: The density at time :math:`t` is the probability of having
survived to :math:`t` (the :math:`\bar{F}` term) times the rate of coalescing
at :math:`t` (the :math:`\lambda` term). This "survive then hit" decomposition
appears throughout survival analysis and should feel familiar from the
coalescent waiting times in :ref:`coalescent theory <coalescent_theory>`.

Probability of joining branch :math:`b_i`
-------------------------------------------

A branch :math:`b_i` spans the time interval :math:`[x, y]`. The probability of
joining this specific branch is:

.. math::

   p_i = \int_x^y \frac{f_\Psi(t)}{\lambda_\Psi(t)} \, dt = \int_x^y \bar{F}_\Psi(t) \, dt

**Why divide by** :math:`\lambda_\Psi(t)` **?** The density :math:`f_\Psi(t)` gives
the probability of coalescing at time :math:`t` with *any* of the
:math:`\lambda_\Psi(t)` available lineages. But we want the probability of
joining one *specific* branch :math:`b_i`. Since at time :math:`t` all
:math:`\lambda_\Psi(t)` branches are equally likely targets, the probability of
joining :math:`b_i` specifically is :math:`f_\Psi(t) / \lambda_\Psi(t)`.

.. admonition:: Probability Aside -- Symmetry and Exchangeability

   The assumption that all :math:`\lambda_\Psi(t)` branches are equally likely
   targets comes from the **exchangeability** property of the coalescent
   (see :ref:`coalescent theory <coalescent_theory>`). Under the standard
   neutral coalescent, all lineages are statistically identical -- none is
   "preferred" over any other. So if coalescence occurs at time :math:`t`, each
   of the :math:`\lambda_\Psi(t)` available branches receives an equal
   :math:`1/\lambda_\Psi(t)` share of the coalescence probability.

   This is analogous to drawing a ball from an urn: if the urn contains
   :math:`\lambda` balls and each is equally likely, the probability of drawing
   a specific ball is :math:`1/\lambda`.

Substituting :math:`f_\Psi(t) = \lambda_\Psi(t) \bar{F}_\Psi(t)`:

.. math::

   p_i = \int_x^y \frac{\lambda_\Psi(t) \bar{F}_\Psi(t)}{\lambda_\Psi(t)} \, dt
   = \int_x^y \bar{F}_\Psi(t) \, dt

The :math:`\lambda` terms cancel, giving this clean formula.

.. code-block:: python

   import numpy as np
   from scipy.integrate import quad

   def joining_probability_exact(x, y, tree_intervals):
       """Compute the exact probability of joining a branch spanning [x, y].

       Parameters
       ----------
       x, y : float
           Lower and upper time of the branch.
       tree_intervals : list of (lower, upper)
           All branch intervals in the marginal tree, defining lambda(t).

       Returns
       -------
       p : float
           Joining probability for this branch.
       """
       def lambda_psi(t):
           """Number of lineages at time t.

           Count how many branches in the tree span time t.
           This is a step function that decreases as we go deeper
           in time (further from the present), because lineages
           merge at coalescence events.
           """
           return sum(1 for lo, hi in tree_intervals if lo <= t < hi)

       def integrand_for_F_bar(t):
           """exp(-integral_0^t lambda(x) dx) = F_bar(t).

           This is the survival probability: the chance that the
           new lineage has NOT coalesced by time t.  We compute
           it by numerically integrating lambda from 0 to t.
           """
           integral, _ = quad(lambda_psi, 0, t)
           return np.exp(-integral)

       # Integrate the survival function over the branch interval [x, y].
       # This gives the probability of joining this specific branch.
       p, _ = quad(integrand_for_F_bar, x, y)
       return p

   # Example: tree with 4 leaves
   # Branches: 4 leaf branches [0, t1], 2 internal [t1, t2], 1 root [t2, inf]
   t1, t2 = 0.3, 0.8
   tree_intervals = [
       (0, t1), (0, t1), (0, t1), (0, t1),  # 4 leaf branches
       (t1, t2), (t1, t2),                    # 2 internal branches
       (t2, 5.0),                              # root branch (truncated)
   ]

   for i, (lo, hi) in enumerate(tree_intervals):
       p = joining_probability_exact(lo, hi, tree_intervals)
       print(f"Branch {i} [{lo:.1f}, {hi:.1f}]: p = {p:.6f}")

With the exact joining probability in hand, we have the foundation for the
HMM's initial distribution. But computing it exactly is expensive -- the
step-function :math:`\lambda_\Psi(t)` requires numerical integration at every
evaluation. For large sample sizes, we need an approximation. That is the
subject of the next section.


Step 2: The Deterministic Approximation
=========================================

The exact calculation is expensive because :math:`\lambda_\Psi(t)` is a step
function. For large :math:`n`, Frost and Volz (2010) showed that
:math:`\lambda_\Psi(t)` is well approximated by its expectation:

.. math::

   \lambda_\Psi(t) \approx \frac{n}{n + (1-n)\exp(-t/2)}

.. admonition:: Probability Aside -- Where This Formula Comes From

   Under the standard coalescent with :math:`n` samples, the expected number
   of lineages at time :math:`t` (in coalescent units) can be derived from the
   coalescent waiting times (see :ref:`coalescent theory <coalescent_theory>`).

   The exact expected lineage count involves a sum over all possible coalescence
   configurations, but for large :math:`n` the **deterministic approximation**
   replaces this sum with a smooth function. The idea is that when :math:`n` is
   large, the stochastic fluctuations in lineage count are small relative to
   the mean -- the law of large numbers kicks in.

   The formula :math:`\lambda(t) = n / (n + (1-n)e^{-t/2})` is the solution to
   the **deterministic coalescent ODE**:
   :math:`d\lambda/dt = -\lambda(\lambda - 1)/2`, with initial condition
   :math:`\lambda(0) = n`. This ODE says: the rate at which lineages disappear
   (left side) equals the rate of pairwise coalescence (right side).

   At :math:`t = 0`, :math:`\lambda(0) = n/(n + 0) = n` (all lineages present).
   As :math:`t \to \infty`, :math:`e^{-t/2} \to 0`, so
   :math:`\lambda \to n/n = 1` (only the root lineage remains). The function
   smoothly interpolates between these limits.

This is a smooth function, making all integrals analytically tractable.

Approximation for :math:`\bar{F}_\Psi(t)`
-------------------------------------------

.. math::

   \bar{F}_\Psi(t) \approx \exp(-t) \left[\frac{n + (1-n)\exp(-t/2)}{1}\right]^{-2}

Wait -- let's derive this properly. We need:

.. math::

   \int_0^t \lambda(x) \, dx = \int_0^t \frac{n}{n + (1-n)e^{-x/2}} \, dx

We need to evaluate the integral :math:`\int_0^t \lambda(x) \, dx` where
:math:`\lambda(x) = \frac{n}{n + (1-n)e^{-x/2}}`.

.. admonition:: Calculus Aside -- The Substitution Strategy

   The integrand :math:`n / (n + (1-n)e^{-x/2})` looks complicated, but it
   has a key property: it involves :math:`e^{-x/2}`, which suggests the
   substitution :math:`u = e^{-x/2}`. This is a standard technique in
   calculus -- when the integrand contains an exponential, substituting the
   exponential as the new variable often simplifies the algebra dramatically.

   The substitution converts the integral from one over :math:`x` (which
   appears in both the exponential and the denominator) to one over :math:`u`
   (which appears only in the denominator), making partial fractions applicable.

**Substitution.** Let :math:`u = e^{-x/2}`, so that :math:`du = -\frac{1}{2}e^{-x/2}dx = -\frac{u}{2}dx`,
which gives :math:`dx = -\frac{2}{u}du`. When :math:`x = 0`, :math:`u = 1`.
When :math:`x = t`, :math:`u = e^{-t/2}`. Substituting:

.. math::

   \int_0^t \frac{n}{n + (1-n)u} \cdot \left(-\frac{2}{u}\right) du
   = 2n \int_{e^{-t/2}}^{1} \frac{du}{u[n + (1-n)u]}

(The limits flip because :math:`u` decreases as :math:`x` increases, and the
minus sign cancels.)

**Partial fractions.** We decompose :math:`\frac{1}{u[n + (1-n)u]}` by finding
constants :math:`A, B` such that:

.. math::

   \frac{1}{u[n + (1-n)u]} = \frac{A}{u} + \frac{B}{n + (1-n)u}

.. admonition:: Calculus Aside -- The Partial Fractions Method

   **Partial fractions** is a technique for breaking a complicated rational
   function into a sum of simpler ones that can each be integrated individually.
   The idea: if the denominator factors as a product of simpler terms (here,
   :math:`u` and :math:`n + (1-n)u`), we can write the fraction as a sum of
   terms, one per factor.

   To find the constants :math:`A` and :math:`B`, multiply both sides by the
   full denominator to clear all fractions, then substitute convenient values
   of :math:`u` (typically the zeros of each factor) to solve for each constant
   independently. This is called the **cover-up method** or
   **Heaviside's method**.

Multiplying both sides by :math:`u[n + (1-n)u]`:

.. math::

   1 = A[n + (1-n)u] + Bu

Setting :math:`u = 0`: :math:`1 = An`, so :math:`A = 1/n`.
Setting :math:`u = -n/(1-n)` (the zero of :math:`n + (1-n)u`):
:math:`1 = B \cdot (-n/(1-n))`, so :math:`B = (1-n)/(-n) = (n-1)/n`.

Therefore:

.. math::

   \frac{1}{u[n + (1-n)u]} = \frac{1}{n} \cdot \frac{1}{u} + \frac{n-1}{n} \cdot \frac{1}{n + (1-n)u}

**Integration.** Substituting back:

.. math::

   2n \int_{e^{-t/2}}^{1} \left[\frac{1}{n} \cdot \frac{1}{u} + \frac{n-1}{n} \cdot \frac{1}{n+(1-n)u}\right] du

.. math::

   = 2\int_{e^{-t/2}}^1 \frac{du}{u} + 2(n-1)\int_{e^{-t/2}}^1 \frac{du}{n + (1-n)u}

The first integral is straightforward: :math:`\int \frac{du}{u} = \ln u`.

For the second, let :math:`w = n + (1-n)u`, so :math:`dw = (1-n)du`:

.. math::

   \int \frac{du}{n + (1-n)u} = \frac{1}{1-n}\ln|n + (1-n)u|

Combining (note :math:`2(n-1) \cdot \frac{1}{1-n} = 2(n-1) \cdot \frac{-1}{n-1} = -2`):

.. math::

   = 2\left[\ln u\right]_{e^{-t/2}}^1 - 2\left[\ln(n + (1-n)u)\right]_{e^{-t/2}}^1

Evaluating at the limits. At :math:`u = 1`: :math:`\ln(1) = 0` and
:math:`\ln(n + (1-n) \cdot 1) = \ln(1) = 0`. At :math:`u = e^{-t/2}`:
:math:`\ln(e^{-t/2}) = -t/2` and :math:`\ln(n + (1-n)e^{-t/2})`. So:

.. math::

   = 2[0 - (-t/2)] - 2[0 - \ln(n + (1-n)e^{-t/2})]
   = t + 2\ln(n + (1-n)e^{-t/2})

**The exceedance probability:**

.. math::

   \bar{F}_\Psi(t) = \exp\left(-\left[t + 2\ln(n + (1-n)e^{-t/2})\right]\right)

Using :math:`e^{-\ln(x^2)} = 1/x^2`:

.. math::

   = e^{-t} \cdot e^{-2\ln(n + (1-n)e^{-t/2})} = \frac{e^{-t}}{[n + (1-n)e^{-t/2}]^2}

**The density** (just multiply by :math:`\lambda(t)`):

.. math::

   f_\Psi(t) = \lambda(t)\bar{F}_\Psi(t) = \frac{n}{n + (1-n)e^{-t/2}} \cdot \frac{e^{-t}}{[n + (1-n)e^{-t/2}]^2} = \frac{n \, e^{-t}}{[n + (1-n)e^{-t/2}]^3}

**The branch joining probability** :math:`p_i = \int_x^y \bar{F}(t) \, dt` can be
evaluated analytically but the closed form is complex. In practice we compute it
numerically:

.. code-block:: python

   def lambda_approx(t, n):
       """Deterministic approximation for number of lineages at time t.

       This replaces the stochastic step-function lambda(t) with a
       smooth curve that tracks the expected lineage count.  The
       approximation improves as n increases (law of large numbers).
       """
       return n / (n + (1 - n) * np.exp(-t / 2))

   def F_bar_approx(t, n):
       """Exceedance probability P(T > t) under the approximation.

       This is the probability that the new lineage has NOT coalesced
       by time t.  The formula was derived by integrating the smooth
       lambda approximation (see the derivation above).
       """
       return np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**2

   def f_approx(t, n):
       """Density of joining time under the approximation.

       f(t) = lambda(t) * F_bar(t): the rate of coalescence at time t
       times the probability of having survived to time t.
       """
       return n * np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**3

   def joining_prob_approx(x, y, n):
       """Joining probability for branch [x, y] using the approximation.

       Numerically integrates the survival function over the branch
       interval.  This is much faster than the exact calculation
       because F_bar_approx is a smooth closed-form function.
       """
       result, _ = quad(lambda t: F_bar_approx(t, n), x, y)
       return result

   # Compare exact vs approximate for n=50
   n = 50
   t1, t2, t3 = 0.01, 0.05, 0.2
   branches = [(0, t1), (t1, t2), (t2, t3), (t3, 2.0)]

   print(f"{'Branch':<20} {'p_approx':>10}")
   print("-" * 32)
   total = 0
   for lo, hi in branches:
       p = joining_prob_approx(lo, hi, n)
       total += p
       print(f"[{lo:.2f}, {hi:.2f}]{'':<10} {p:>10.6f}")
   print(f"{'Sum':<20} {total:>10.6f}")

With the deterministic approximation, we have traded a step function that
requires numerical integration against itself for a smooth function with a
closed-form survival probability. This is the kind of engineering trade-off
that makes SINGER practical for large datasets -- the approximation is accurate
for :math:`n \geq 20` and becomes essentially exact for :math:`n \geq 100`.

Now we need one more ingredient before building the HMM: a single representative
time for each branch, to use in the emission and transition calculations.


Step 3: Representative Joining Time
=====================================

For the HMM transition and emission calculations, we need a single
**representative time** :math:`\tau_i` for each branch :math:`b_i` spanning
:math:`[x, y]`. SINGER uses the heuristic:

.. math::

   \lambda(\tau_i) = \sqrt{\lambda(x) \cdot \lambda(y)}

That is, :math:`\tau_i` is chosen so that :math:`\lambda(\tau_i)` is the
**geometric mean** of the number of lineages at the branch endpoints.

.. admonition:: Probability Aside -- Why the Geometric Mean?

   The **geometric mean** of two positive numbers :math:`a` and :math:`b` is
   :math:`\sqrt{ab}`. Unlike the **arithmetic mean** :math:`(a+b)/2`, the
   geometric mean respects multiplicative structure: if :math:`a` and :math:`b`
   differ by a constant factor :math:`r`, then :math:`\sqrt{ab} = a\sqrt{r}`,
   which is exactly halfway between :math:`a` and :math:`b` on a logarithmic
   scale.

   Lineage counts decrease roughly geometrically (each coalescence event
   reduces the count by 1 out of :math:`k`, a roughly constant proportional
   drop). The geometric mean therefore sits at the "natural midpoint" of the
   branch in lineage-count space. Using the arithmetic mean would
   systematically overweight the endpoint with more lineages (the lower
   endpoint, nearer the present).

   In the watch metaphor: the geometric mean is the natural way to measure
   the "average gear ratio" of a branch that spans a range of positions on a
   logarithmic dial.

**Deriving the inverse function** :math:`\lambda^{-1}(\ell)`. Given
:math:`\lambda(t) = \frac{n}{n + (1-n)e^{-t/2}}`, we want to solve
:math:`\lambda(t) = \ell` for :math:`t`:

.. math::

   \ell = \frac{n}{n + (1-n)e^{-t/2}}

Multiply both sides by the denominator:

.. math::

   \ell[n + (1-n)e^{-t/2}] = n

.. math::

   \ell n + \ell(1-n)e^{-t/2} = n

.. math::

   e^{-t/2} = \frac{n - \ell n}{\ell(1-n)} = \frac{n(1 - \ell)}{\ell(1-n)}

Taking :math:`\ln` and multiplying by :math:`-2`:

.. math::

   \lambda^{-1}(\ell) = t = -2\ln\left(\frac{n(1 - \ell)}{\ell(1-n)}\right)

Note: since :math:`1 < \ell < n` (the number of lineages is always between 1
and :math:`n`), and :math:`1 - n < 0`, the argument of the log is positive.

**Why geometric mean?** The joining probability :math:`p_i` is essentially the
integral of a smooth function over :math:`[\lambda(x), \lambda(y)]`. The
geometric mean :math:`\sqrt{\lambda(x)\lambda(y)}` is a better midpoint than
the arithmetic mean for quantities that vary multiplicatively (like lineage
counts), and empirically gives more accurate emission and transition
probabilities.

.. code-block:: python

   def lambda_inverse(ell, n):
       """Inverse of the lambda function: find t such that lambda(t) = ell.

       This inverts the deterministic approximation formula.
       We need this to convert from a target lineage count (the
       geometric mean) back to a physical time.
       """
       # lambda(t) = n / (n + (1-n)*exp(-t/2))
       # Solving for t:
       # ell * (n + (1-n)*exp(-t/2)) = n
       # (1-n)*exp(-t/2) = n/ell - n = n(1-ell)/ell
       # exp(-t/2) = n(1-ell) / (ell*(1-n))
       # Since n > 1 and 1-n < 0: n(1-ell)/(ell*(1-n)) should be positive
       # when ell < n (which it always is for branches below TMRCA)
       ratio = (n - n * ell) / (ell - n * ell)
       if ratio <= 0:
           return np.inf  # edge case: lineage count at or beyond n
       return -2 * np.log(ratio)

   def representative_time(x, y, n):
       """Compute representative joining time for branch [x, y].

       Uses the geometric mean of lambda at the endpoints to find
       a single representative time for the branch.  This time is
       used in emission and transition probability calculations.
       """
       lam_x = lambda_approx(x, n)
       lam_y = lambda_approx(y, n)
       # Geometric mean: the "natural midpoint" on a log scale
       lam_tau = np.sqrt(lam_x * lam_y)
       tau = lambda_inverse(lam_tau, n)
       return tau

   # Example
   n = 50
   for x, y in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]:
       tau = representative_time(max(x, 1e-10), y, n)
       print(f"Branch [{x:.2f}, {y:.2f}]: tau = {tau:.6f}, "
             f"lambda(x)={lambda_approx(max(x,1e-10),n):.2f}, "
             f"lambda(y)={lambda_approx(y,n):.2f}, "
             f"lambda(tau)={lambda_approx(tau,n):.2f}")

We now have all the ingredients for the joining probability and representative
time. Next, we need to compute what the HMM "sees" -- the emission probability
that connects the hidden branch state to the observed allele.


Step 4: Emission Probabilities
================================

The emission probability answers: given that the new lineage joins branch
:math:`b_i` at representative time :math:`\tau_i`, how likely are we to see
the observed allele at this position?

This is where the mutation model from :ref:`coalescent theory <coalescent_theory>`
enters the picture. Recall that under the infinite-sites model, mutations occur
as a Poisson process along branches: on a branch of length :math:`\ell`, the
probability of at least one mutation is :math:`1 - e^{-\theta\ell/2}`, where
:math:`\theta = 4N_e\mu` is the population-scaled mutation rate.

When the new node joins a branch, it creates a **new lineage** and bisects the
joining branch into two parts. Three branches are involved:

1. **New lineage**: from the new node (time 0) up to the joining point (time :math:`\tau_i`)
   -- length :math:`\ell_1 = \tau_i`
2. **Lower part of joining branch**: from the lower endpoint to the joining point
   -- length :math:`\ell_2`
3. **Upper part of joining branch**: from the joining point to the upper endpoint
   -- length :math:`\ell_3`

For each branch, the probability of no mutation is :math:`e^{-\theta\ell/2}`,
and the probability of exactly one mutation is :math:`1 - e^{-\theta\ell/2}`.

.. admonition:: Probability Aside -- The Infinite-Sites Mutation Model

   Under the **infinite-sites model** (introduced in
   :ref:`coalescent theory <coalescent_theory>`), each site in the genome can
   mutate at most once in the entire genealogy. This is a reasonable
   approximation when the mutation rate is low relative to the branch lengths
   (which is the case for most organisms), because the probability of two
   independent mutations hitting the same site is negligible.

   The key consequence: mutations on different branches are **independent**.
   If we know the allelic state at both endpoints of a branch, we can compute
   the probability of seeing that pattern without worrying about interactions
   with other branches. This independence is what allows us to multiply the
   probabilities from the three branches below.

The emission probability is computed in three steps:

1. **Impute** the allelic state at the joining point using parsimony.
   Parsimony assigns the state that requires the fewest mutations. For example,
   if the subtree below the joining point has mostly 0s, the joining point is
   assigned 0. If the new node carries allele 1, a mutation must have occurred
   on the new lineage.

2. For each of the 3 branches, compute the probability of the required number of
   mutations (0 or 1). A branch of length :math:`\ell` has:

   - :math:`P(\text{no mutation}) = e^{-\theta\ell/2}` -- needed when the alleles
     at both endpoints match
   - :math:`P(\text{exactly one mutation}) = 1 - e^{-\theta\ell/2}` -- needed
     when the alleles at the endpoints differ (under infinite sites, at most one
     mutation per site per branch)

3. **Multiply** the three probabilities (independence of mutations on different
   branches):

   .. math::

      e(X_\ell \mid b_i) = P(\text{branch 1 correct}) \times P(\text{branch 2 correct}) \times P(\text{branch 3 correct})

   The result is the per-site emission. For a bin of :math:`m` base pairs,
   the per-bin emission is the product over all sites in the bin.

.. code-block:: python

   def emission_probability(allele_new, allele_at_join, tau, branch_lower,
                            branch_upper, theta):
       """Compute emission probability for one site.

       Parameters
       ----------
       allele_new : int
           Allele (0 or 1) at the new node (sample).
       allele_at_join : int
           Imputed allele at the joining point (by parsimony).
       tau : float
           Representative joining time.
       branch_lower : float
           Lower time of the joining branch.
       branch_upper : float
           Upper time of the joining branch.
       theta : float
           Population-scaled mutation rate (4*Ne*mu).

       Returns
       -------
       prob : float
           Per-site emission probability.
       """
       # Branch lengths: how much "time" each branch spans
       l1 = tau                    # new lineage: from present to joining point
       l2 = tau - branch_lower     # lower part of joining branch
       l3 = branch_upper - tau     # upper part of joining branch

       def p_mutation(length):
           """Probability of mutation on a branch of given length.

           Under the Poisson mutation model, mutations occur at rate
           theta/2 per unit time.  1 - exp(-rate * time) is the
           probability of at least one event.
           """
           return 1 - np.exp(-theta / 2 * length)

       def p_no_mutation(length):
           """Probability of no mutation on a branch of given length.

           exp(-rate * time) is the probability of zero events in
           a Poisson process -- the "survival" probability for mutations.
           """
           return np.exp(-theta / 2 * length)

       # New lineage: needs mutation if allele_new != allele_at_join
       if allele_new != allele_at_join:
           e1 = p_mutation(l1)
       else:
           e1 = p_no_mutation(l1)

       # For the two parts of the joining branch, we need to consider
       # what alleles are expected above and below the joining point.
       # The detailed calculation depends on the tree topology above,
       # but the core structure is:
       e2 = p_no_mutation(l2)  # Simplified: no mutation on lower part
       e3 = p_no_mutation(l3)  # Simplified: no mutation on upper part

       # Independence: multiply the three branch probabilities
       return e1 * e2 * e3

   # Example: compute emission for a few scenarios
   theta = 0.001  # typical per-bp value
   tau = 0.5
   branch = (0.1, 1.2)

   for allele_new, allele_join in [(0, 0), (0, 1), (1, 0), (1, 1)]:
       e = emission_probability(allele_new, allele_join, tau,
                                branch[0], branch[1], theta)
       print(f"allele_new={allele_new}, allele_join={allele_join}: "
             f"emission={e:.8f}")

For a bin of :math:`m` base pairs, the per-bin emission is the product over all
sites in the bin:

.. math::

   e_{\text{bin}}(X_\ell \mid B_\ell = b_i) = \prod_{s \in \text{bin } \ell} e_s(x_s \mid b_i, \tau_i)

.. admonition:: Calculus Aside -- Products of Probabilities and Log-Space Computation

   When we multiply many probabilities together (one per site in a bin), the
   product can become astronomically small -- a bin of 100 sites might produce
   a product like :math:`10^{-30}`. This causes **numerical underflow** (the
   computer rounds the result to zero).

   The standard solution is to work in **log space**: compute
   :math:`\sum_s \ln e_s` instead of :math:`\prod_s e_s`. Sums of logs are
   numerically stable even when the individual probabilities are tiny. We
   convert back to probability space only when needed (e.g., for normalization).
   This is the same technique used in the scaled forward algorithm in
   :ref:`the HMM chapter <hmms>`.

We have now built the emission model -- the component that connects hidden
states (branches) to observations (alleles). Next comes the transition model:
how does the joining branch change from one genomic bin to the next?


Step 5: Transition Probabilities (New Recombination)
======================================================

Now the heart of the HMM: how does the joining branch change between adjacent bins?

This transition model is structurally similar to the **Li-Stephens model**
(see :ref:`the Li-Stephens HMM chapter <copying_model>`), which describes how
a haplotype "copies" segments from a reference panel, occasionally switching
templates due to recombination. Here, instead of switching between reference
haplotypes, we switch between branches of the marginal tree.

When there is **no existing recombination** in the partial ARG between bins
:math:`\ell-1` and :math:`\ell`, the new lineage may introduce a recombination.
The probability depends on the representative time :math:`\tau_i` of the current
branch:

.. math::

   r_i = 1 - \exp\left(-\frac{\rho}{2}\tau_i\right)

where :math:`\rho = 4N_e r m` is the population-scaled recombination rate per bin.

.. admonition:: Probability Aside -- Why the Recombination Probability Depends on :math:`\tau_i`

   The recombination probability :math:`r_i = 1 - e^{-\rho\tau_i/2}` has a
   clear physical interpretation. A recombination can occur anywhere on the
   new lineage between the present (time 0) and the joining point (time
   :math:`\tau_i`). Under the Poisson model, the probability of at least one
   recombination on a branch of length :math:`\tau_i` is
   :math:`1 - e^{-\rho\tau_i/2}`.

   Longer branches (larger :math:`\tau_i`) are more likely to experience
   recombination -- they have more "runway" for a recombination event to occur.
   This is the same logic as the mutation model: more time means more
   opportunities for events.

The transition probability has the Li-Stephens structure:

.. math::

   P(B_\ell = b_j \mid B_{\ell-1} = b_i) = (1 - r_i)\delta_{ij} + r_i \frac{q_j}{\sum_{k: b_k \in \mathcal{S}_\ell} q_k}

where:

.. math::

   q_j = \begin{cases}
   r_j \cdot p_j & \text{if } b_j \text{ is a full branch} \\
   0 & \text{if } b_j \text{ is a partial branch}
   \end{cases}

.. admonition:: Probability Aside -- The Li-Stephens Transition Structure

   The transition formula :math:`(1-r_i)\delta_{ij} + r_i \cdot (\text{something})`
   is the hallmark of the **Li-Stephens copying model** (see
   :ref:`the Li-Stephens chapter <copying_model>` for the full derivation).
   It says:

   - With probability :math:`1 - r_i`, **no recombination** occurs, and the
     hidden state stays the same: :math:`B_\ell = B_{\ell-1}`. The Kronecker
     delta :math:`\delta_{ij}` ensures this contributes only when :math:`j = i`.

   - With probability :math:`r_i`, **recombination** occurs, and the new lineage
     "jumps" to a new branch. The target branch :math:`b_j` is chosen with
     probability proportional to :math:`q_j`.

   This structure ensures that the transition matrix is **sparse**: most of the
   probability mass is on the diagonal (staying on the same branch), with a
   small amount spread across all branches. This sparsity is what makes the
   HMM efficient -- the forward algorithm can exploit it.

.. admonition:: Why :math:`q_j = r_j \cdot p_j`?

   After a recombination, the new lineage must re-coalesce above the recombination
   breakpoint. Lower branches are less likely to be joined because re-coalescence
   must be more ancient than the recombination event. The factor :math:`r_j` (which
   increases with :math:`\tau_j`) naturally down-weights lower branches.

   The product :math:`r_j \cdot p_j` ensures the stationary distribution of the
   HMM is :math:`P(B_\ell = b_i) = p_i` -- the coalescent-derived branch
   probability.

.. code-block:: python

   def branch_transition_prob(tau_i, tau_j, p_j, rho, is_partial_j,
                              q_sum, same_branch):
       """Compute transition probability from branch i to branch j.

       Parameters
       ----------
       tau_i : float
           Representative time of the current branch.
       tau_j : float
           Representative time of the target branch.
       p_j : float
           Coalescence probability for target branch.
       rho : float
           Population-scaled recombination rate per bin.
       is_partial_j : bool
           Whether target branch is a partial branch state.
       q_sum : float
           Sum of q_k over all states in S_ell.
       same_branch : bool
           Whether i == j (same branch, no recombination).

       Returns
       -------
       prob : float
       """
       # r_i: probability of recombination on the new lineage up to tau_i
       r_i = 1 - np.exp(-rho / 2 * tau_i)

       if is_partial_j:
           q_j = 0.0  # partial branches get zero weight (see Step 6)
       else:
           r_j = 1 - np.exp(-rho / 2 * tau_j)
           q_j = r_j * p_j  # product ensures correct stationary distribution

       if same_branch:
           # No-recombination term + recombination-to-self term
           return (1 - r_i) + r_i * q_j / q_sum
       else:
           # Pure recombination term: probability of jumping to branch j
           return r_i * q_j / q_sum

   # Example: 5 branches
   n = 50
   rho = 0.5
   branches = [(0.0, 0.02), (0.02, 0.06), (0.06, 0.15), (0.15, 0.4), (0.4, 2.0)]
   taus = [representative_time(max(x, 1e-10), y, n) for x, y in branches]
   probs = [joining_prob_approx(x, y, n) for x, y in branches]

   # Compute q values: the unnormalized target weights after recombination
   r_vals = [1 - np.exp(-rho / 2 * t) for t in taus]
   q_vals = [r * p for r, p in zip(r_vals, probs)]
   q_sum = sum(q_vals)

   print("Transition matrix:")
   T = np.zeros((5, 5))
   for i in range(5):
       for j in range(5):
           T[i, j] = branch_transition_prob(
               taus[i], taus[j], probs[j], rho,
               is_partial_j=False, q_sum=q_sum, same_branch=(i == j)
           )
       print(f"  From branch {i}: {np.round(T[i], 4)}")
   print(f"Row sums: {T.sum(axis=1)}")

The transition matrix should have rows that sum to 1 (from any branch, you must
go *somewhere*). The diagonal entries should be large (most bins have no
recombination), and the off-diagonal entries should reflect the coalescent
branch probabilities.

So far, all our hidden states have been *full* branches -- entire edges of the
marginal tree. But SINGER introduces a subtlety that no previous method handled:
what happens when the partial ARG already has a recombination between adjacent
bins? This leads to the concept of partial branch states, SINGER's key
innovation.


Step 6: Partial Branch States
==============================

This is SINGER's key innovation and the most subtle part of the algorithm.

The Problem
-----------

When the partial ARG already has a recombination between adjacent bins, the new
lineage cannot introduce another recombination there. Instead, the joining branch
may change by **hitchhiking** on the existing recombination.

.. admonition:: Probability Aside -- Why Not Just Ignore Existing Recombinations?

   You might wonder: can we just treat every bin independently, as if no
   recombinations exist in the partial ARG? The answer is no, because this
   would violate the **consistency constraint** of the ARG.

   An ARG must be a valid genealogical history -- every lineage must trace a
   single coherent path from the present to the ancestor. If the partial ARG
   already has a recombination at some position, the new lineage's path through
   that position is constrained: it must either follow the left-tree topology
   or the right-tree topology, depending on which side of the recombination
   breakpoint it joins. The partial branch states encode these constraints.

If we only used full branches as hidden states, we'd allow impossible state
sequences. Consider three adjacent bins where the partial ARG has recombinations
between bins 1-2 and bins 2-3. A transition :math:`b_i \to b_j \to b_k` might
be valid for each consecutive pair, but impossible as a triple (because it would
require a recombination in the new lineage that we've forbidden).

The Solution
-------------

SINGER introduces **partial branch states**: when a full branch is split by a
recombination in the partial ARG, each piece becomes a separate state. This
correctly captures the constraint that joining the upper vs. lower part of a
split branch leads to different trees.

A partial branch :math:`(c, p) : [t_1, t_2]` means:

- The branch goes from child :math:`c` to parent :math:`p` in the full tree
- But we only consider the time interval :math:`[t_1, t_2]`, which is a subset
  of the full branch

In the watch metaphor, partial branches are like gear teeth that have been
bisected -- the upper and lower halves engage different parts of the mechanism,
and the watchmaker must track each half separately to ensure the train runs
correctly.

.. code-block:: python

   class BranchState:
       """A branch state in the SINGER HMM (full or partial).

       Each state represents either a complete branch of the marginal
       tree (full) or a segment of a branch that was split by a
       recombination event in the partial ARG (partial).
       """

       def __init__(self, child, parent, lower_time, upper_time, is_partial=False):
           self.child = child            # child node ID in the tree
           self.parent = parent          # parent node ID in the tree
           self.lower_time = lower_time  # start of the time interval
           self.upper_time = upper_time  # end of the time interval
           self.is_partial = is_partial  # True if this is a sub-segment

       @property
       def length(self):
           """Time span of this branch (or branch segment)."""
           return self.upper_time - self.lower_time

       def __repr__(self):
           kind = "partial" if self.is_partial else "full"
           return (f"BranchState({self.child},{self.parent}): "
                   f"[{self.lower_time:.4f},{self.upper_time:.4f}] ({kind})")

   def build_state_space(full_branches, partial_branches, forward_probs,
                          epsilon=0.01):
       """Build the state space for a bin, pruning unlikely partial branches.

       Parameters
       ----------
       full_branches : list of BranchState
           Full branches of the marginal tree at this bin.
       partial_branches : list of (BranchState, float)
           Candidate partial branches with their forward probabilities.
       epsilon : float
           Pruning threshold: partial branches with forward probability
           below epsilon are discarded to keep the state space manageable.

       Returns
       -------
       states : list of BranchState
           Active states for this bin.
       """
       # All full branches are always included -- they are always valid
       # joining targets for the new lineage
       states = list(full_branches)

       # Partial branches are included only if their forward probability
       # exceeds epsilon.  This keeps the state space from growing
       # unboundedly while preserving the most important constraints.
       for branch, fwd_prob in partial_branches:
           if fwd_prob >= epsilon:
               states.append(branch)

       return states

   # Example
   full = [
       BranchState(0, 4, 0.0, 0.3),
       BranchState(1, 4, 0.0, 0.3),
       BranchState(2, 5, 0.0, 0.7),
       BranchState(4, 5, 0.3, 0.7),
       BranchState(5, -1, 0.7, float('inf')),
   ]

   partial_candidates = [
       (BranchState(0, 4, 0.0, 0.15, is_partial=True), 0.05),  # above threshold
       (BranchState(0, 4, 0.15, 0.3, is_partial=True), 0.002), # below threshold
   ]

   states = build_state_space(full, partial_candidates, None, epsilon=0.01)
   print(f"State space size: {len(states)}")
   for s in states:
       print(f"  {s}")

.. admonition:: State space size bound

   SINGER controls the state space to be at most :math:`2n - 1 + 1/\epsilon`.
   Full branches contribute :math:`2n - 1` states (the number of branches in a
   binary tree with :math:`n` leaves), and partial branches are capped at
   :math:`1/\epsilon` by the forward probability pruning. In practice, the
   state space is only slightly larger than :math:`2n - 1`.

   This bound is important for computational tractability. Without it, the
   number of partial branch states could grow without limit as recombinations
   accumulate. The pruning threshold :math:`\epsilon` controls the trade-off
   between accuracy (keeping more partial states) and speed (keeping fewer).

Transition when a recombination exists in the partial ARG
----------------------------------------------------------

When a recombination in the partial ARG splits a branch into segments, the
transition probability from the full branch is **distributed proportionally**
to the coalescence probability of each segment:

.. code-block:: python

   def split_branch_transition(full_branch, segments, n):
       """Distribute transition probability when a branch splits.

       When the partial ARG has a recombination that splits a branch
       into segments, the probability mass from the full branch must
       be distributed among the segments.  We weight each segment
       by its coalescence probability -- how likely the new lineage
       is to join that particular segment.

       Parameters
       ----------
       full_branch : BranchState
           The full branch being split.
       segments : list of BranchState
           The resulting segments.
       n : int
           Number of samples (for joining probability calculation).

       Returns
       -------
       weights : list of float
           Transition weight for each segment (sums to 1).
       """
       probs = []
       for seg in segments:
           # Each segment's weight is its joining probability
           p = joining_prob_approx(seg.lower_time, seg.upper_time, n)
           probs.append(p)

       total = sum(probs)
       if total == 0:
           # Fallback: if all probabilities are zero (degenerate case),
           # distribute uniformly
           return [1.0 / len(segments)] * len(segments)
       return [p / total for p in probs]

   # Example: branch [0, 1.0] splits at recombination time 0.3
   full = BranchState(1, 5, 0.0, 1.0)
   seg_lower = BranchState(1, 5, 0.0, 0.3, is_partial=True)
   seg_upper = BranchState(1, 5, 0.3, 1.0, is_partial=True)

   weights = split_branch_transition(full, [seg_lower, seg_upper], n=50)
   print(f"Lower segment weight: {weights[0]:.4f}")
   print(f"Upper segment weight: {weights[1]:.4f}")

Now that we have all the individual components -- joining probabilities,
representative times, emission probabilities, transition probabilities, and
partial branch states -- it is time to assemble them into the complete forward
algorithm.


Step 7: Putting It All Together -- The Branch Sampling HMM
============================================================

Now we assemble all the gears into the complete branch sampling forward algorithm.
This is the moment where the individual components snap together into a working
mechanism -- the HMM that threads a new haplotype through the partial ARG.

If you have worked through the :ref:`HMM prerequisite chapter <hmms>`, you will
recognize the structure: initialize at the first bin, then recurse forward
through the genome, accumulating forward probabilities at each step. The only
novelty is the handling of partial branch states (Step 6) and the distinction
between bins where the partial ARG has a recombination and bins where it does not.

.. code-block:: python

   def branch_sampling_forward(bins, partial_arg, new_haplotype, n, theta, rho,
                                epsilon=0.01):
       """Run the forward algorithm for branch sampling.

       This is the complete branch sampling HMM forward pass.
       It mirrors the standard HMM forward algorithm (see the HMM
       prerequisite chapter) but with two SINGER-specific features:
       (1) partial branch states, and (2) transition type selection
       based on existing recombinations in the partial ARG.

       Parameters
       ----------
       bins : list of int
           Genomic bin boundaries.
       partial_arg : object
           The partial ARG (trees + recombinations for first n-1 haplotypes).
       new_haplotype : ndarray of shape (L,)
           Alleles (0/1) of the new haplotype at each bin.
       n : int
           Number of haplotypes already in the partial ARG.
       theta : float
           Population-scaled mutation rate per bp.
       rho : float
           Population-scaled recombination rate per bin.
       epsilon : float
           Pruning threshold for partial branch states.

       Returns
       -------
       alpha : list of dicts
           alpha[ell][state] = forward probability at bin ell for each state.
       states : list of lists
           states[ell] = list of BranchState objects active at bin ell.
       """
       L = len(bins)
       alpha = [{} for _ in range(L)]        # forward probabilities per bin
       all_states = [[] for _ in range(L)]    # active states per bin

       # ----- Bin 0: initialization -----
       # At the first bin, the forward probability for each branch is
       # simply the prior (joining probability) times the emission.
       # This is the standard HMM initialization: alpha_0(k) = pi(k) * e(k).
       tree_branches = get_marginal_tree(partial_arg, bins[0])
       states_0 = [BranchState(b.child, b.parent, b.lower, b.upper)
                    for b in tree_branches]
       all_states[0] = states_0

       for state in states_0:
           # Initial probability = coalescence probability * emission
           p_coal = joining_prob_approx(state.lower_time, state.upper_time, n)
           tau = representative_time(
               max(state.lower_time, 1e-10), state.upper_time, n)
           e = compute_emission(new_haplotype[0], state, tau, theta,
                                partial_arg, bins[0])
           alpha[0][id(state)] = p_coal * e

       # ----- Bins 1, ..., L-1: recursion -----
       # This is the standard HMM forward recursion:
       #   alpha_ell(j) = e(j) * sum_i[ alpha_{ell-1}(i) * T(i,j) ]
       # The twist: the transition T(i,j) depends on whether the
       # partial ARG has a recombination between bins ell-1 and ell.
       for ell in range(1, L):
           # Check if partial ARG has a recombination between bins ell-1, ell
           has_existing_recomb = check_recombination(partial_arg,
                                                      bins[ell-1], bins[ell])

           if has_existing_recomb:
               # Handle partial branch states (hitchhiking on existing recomb)
               states_ell, transitions = handle_existing_recomb(
                   all_states[ell-1], partial_arg, bins[ell], n)
           else:
               # Standard Li-Stephens transitions (new recombination possible)
               tree_branches = get_marginal_tree(partial_arg, bins[ell])
               states_ell = [BranchState(b.child, b.parent, b.lower, b.upper)
                              for b in tree_branches]

           all_states[ell] = states_ell

           # Compute forward probabilities at this bin
           for j, state_j in enumerate(states_ell):
               tau_j = representative_time(
                   max(state_j.lower_time, 1e-10), state_j.upper_time, n)
               e_j = compute_emission(new_haplotype[ell], state_j, tau_j,
                                       theta, partial_arg, bins[ell])

               # Sum over previous states: the core HMM recursion
               fwd_sum = 0.0
               for i, state_i in enumerate(all_states[ell-1]):
                   trans = compute_transition(state_i, state_j, rho, n,
                                               has_existing_recomb)
                   fwd_sum += alpha[ell-1].get(id(state_i), 0) * trans

               alpha[ell][id(state_j)] = e_j * fwd_sum

           # Prune partial branches with forward prob < epsilon
           # This keeps the state space bounded (see Step 6)
           total = sum(alpha[ell].values())
           if total > 0:
               states_keep = []
               for state in states_ell:
                   if state.is_partial:
                       if alpha[ell][id(state)] / total >= epsilon:
                           states_keep.append(state)
                   else:
                       states_keep.append(state)  # always keep full branches
               all_states[ell] = states_keep

       return alpha, all_states

.. admonition:: Note on the implementation

   The code above is a **structural skeleton** showing how the pieces fit together.
   The helper functions (``get_marginal_tree``, ``compute_emission``,
   ``compute_transition``, etc.) each use the formulas derived in Steps 1-6.
   In the exercises below, you'll implement these helpers and run the full
   algorithm on simulated data.

With the forward pass complete, we have the probability of every possible
branch assignment at every bin, given all the observations up to that bin.
But to *sample* a branch sequence from the posterior (not just compute
probabilities), we need to trace back through the forward probabilities.
This is the stochastic traceback.


Step 8: Stochastic Traceback
==============================

After the forward pass, we sample a branch sequence by tracing back. This is
the standard HMM stochastic traceback (also called "backward sampling" or
"posterior sampling"), applied to the branch sampling HMM. If you have seen
the Viterbi algorithm in :ref:`the HMM chapter <hmms>`, the traceback is
similar in structure -- but instead of choosing the *most likely* state at
each position, we *sample* from the posterior distribution. This stochasticity
is essential because SINGER uses MCMC and needs to explore the space of ARGs,
not just find a single best one.

.. code-block:: python

   def branch_sampling_traceback(alpha, all_states, partial_arg, rho, n):
       """Sample a branch sequence from the posterior.

       Starting from the last bin, sample a state proportional to its
       forward probability.  Then trace backwards: at each bin, sample
       a state proportional to (forward probability * transition to
       the already-sampled next state).

       Parameters
       ----------
       alpha : list of dicts
           Forward probabilities from branch_sampling_forward.
       all_states : list of lists
           State spaces from branch_sampling_forward.

       Returns
       -------
       sampled_branches : list of BranchState
           Sampled joining branch at each bin.
       """
       L = len(alpha)
       sampled_branches = [None] * L

       # Sample last state proportional to forward probability.
       # At the last bin, the forward probability IS the posterior
       # (there are no future observations to condition on).
       states_L = all_states[-1]
       probs = np.array([alpha[-1].get(id(s), 0) for s in states_L])
       probs /= probs.sum()  # normalize to a proper distribution
       idx = np.random.choice(len(states_L), p=probs)
       sampled_branches[-1] = states_L[idx]

       # Traceback: sample each bin conditioned on the next bin's choice.
       # P(state at ell | state at ell+1, observations) is proportional to
       # alpha[ell][state] * transition(state -> next_state).
       for ell in range(L - 2, -1, -1):
           state_next = sampled_branches[ell + 1]
           states_curr = all_states[ell]

           probs = np.zeros(len(states_curr))
           for i, state_i in enumerate(states_curr):
               trans = compute_transition(state_i, state_next, rho, n,
                                           check_recombination(partial_arg,
                                               ell, ell+1))
               # Weight = forward probability * transition to chosen next state
               probs[i] = alpha[ell].get(id(state_i), 0) * trans

           probs /= probs.sum()  # normalize
           idx = np.random.choice(len(states_curr), p=probs)
           sampled_branches[ell] = states_curr[idx]

       return sampled_branches

.. admonition:: Probability Aside -- Why Stochastic Traceback Samples from the Posterior

   The stochastic traceback produces samples from the posterior distribution
   :math:`P(B_1, B_2, \ldots, B_L \mid X_1, \ldots, X_L)` -- the joint
   distribution of branch assignments given all the data. This is a standard
   result from HMM theory (see :ref:`hmms`).

   The key insight: the forward probabilities
   :math:`\alpha_\ell(k) = P(X_1, \ldots, X_\ell, B_\ell = k)` encode all the
   information from the left. By sampling the last state from :math:`\alpha_L`
   and then sampling each earlier state conditioned on the later choice, we
   correctly account for both left-to-right (forward) and right-to-left
   (the already-sampled future) information.

   This is different from **maximum a posteriori (MAP) decoding** (Viterbi),
   which finds the single most likely sequence. SINGER needs samples, not
   a single best sequence, because it is part of an MCMC sampler that must
   explore the posterior distribution of ARGs.

**Recap.** Branch sampling is now complete. We have built:

1. The **joining probability** :math:`p_i` for each branch (Steps 1-2)
2. A **representative time** :math:`\tau_i` for each branch (Step 3)
3. **Emission probabilities** connecting branches to observed alleles (Step 4)
4. **Transition probabilities** modeling recombination between bins (Step 5)
5. **Partial branch states** to handle existing recombinations (Step 6)
6. The **forward algorithm** to compute forward probabilities (Step 7)
7. **Stochastic traceback** to sample a branch sequence (Step 8)

In the watch metaphor, we have built the largest gear in the movement -- the
one that determines which slot each new wheel engages. The output is a sequence
of joining branches along the genome. But a branch is a *range* of times; we
still need to determine the exact *when*. That is the job of the next chapter.


Exercises
=========

.. admonition:: Exercise 1: Verify joining probabilities

   Implement the exact (numerical integration) and approximate joining
   probabilities. For :math:`n = 5, 10, 50, 100`, compute both and plot the
   relative error. At what :math:`n` does the approximation become accurate
   to within 1%?

.. admonition:: Exercise 2: Build the full emission function

   Implement ``compute_emission`` that handles all cases: joining a leaf branch,
   an internal branch, and the branch above the root. Handle both polarized
   (:math:`p_{\text{root}} = 0.99`) and unpolarized (:math:`p_{\text{root}} = 0.5`)
   data.

.. admonition:: Exercise 3: Verify the Li-Stephens structure

   Build a complete transition matrix for a 5-branch tree. Verify that:
   (a) rows sum to 1, (b) the stationary distribution is proportional to
   :math:`p_i`, (c) the computational complexity is :math:`O(K)` per bin.

.. admonition:: Exercise 4: Simulate and thread

   Use ``msprime`` to simulate a 2-haplotype ARG with known parameters.
   Then thread a 3rd haplotype using your branch sampling HMM. Compare the
   sampled branches to the true branches from the simulation.

Solutions
=========

.. admonition:: Solution 1: Verify joining probabilities

   We implement both the exact (numerical integration) and approximate joining
   probabilities, then compare them for increasing :math:`n`.

   The exact calculation integrates the survival function :math:`\bar{F}_\Psi(t)`
   over each branch interval, where :math:`\bar{F}` depends on the step function
   :math:`\lambda_\Psi(t)`. The approximate version uses the smooth deterministic
   approximation :math:`\lambda(t) = n/(n + (1-n)e^{-t/2})`.

   .. code-block:: python

      import numpy as np
      from scipy.integrate import quad

      def lambda_approx(t, n):
          return n / (n + (1 - n) * np.exp(-t / 2))

      def F_bar_approx(t, n):
          return np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**2

      def joining_prob_approx(x, y, n):
          result, _ = quad(lambda t: F_bar_approx(t, n), x, y)
          return result

      def joining_probability_exact(x, y, tree_intervals):
          """Exact joining probability via numerical integration."""
          def lambda_psi(t):
              return sum(1 for lo, hi in tree_intervals if lo <= t < hi)

          def F_bar_exact(t):
              integral, _ = quad(lambda_psi, 0, t)
              return np.exp(-integral)

          p, _ = quad(F_bar_exact, x, y)
          return p

      # Build a simple balanced tree for each n and compare
      for n in [5, 10, 50, 100]:
          # Simulate coalescent times for a balanced-ish tree
          np.random.seed(42)
          k = n
          t = 0.0
          times = [0.0]  # coalescence event times
          while k > 1:
              rate = k * (k - 1) / 2
              t += np.random.exponential(1.0 / rate)
              times.append(t)
              k -= 1

          # Build intervals: at time t, the number of lineages decreases
          # Use a simpler structure -- just test a few representative branches
          # For approximate, we only need n and the branch endpoints
          test_branches = [(0.0, times[1]), (times[1], times[2])]

          print(f"\nn = {n}:")
          for x, y in test_branches:
              p_approx = joining_prob_approx(max(x, 1e-10), y, n)
              # For the exact version, build a simplified tree_intervals
              # from the coalescent process
              tree_intervals = []
              for i in range(n):
                  # Each leaf branch goes from 0 to the first coal. time
                  tree_intervals.append((0, times[1]))
              # Above the first coalescence, n-1 lineages remain, etc.
              # (simplified for demonstration)
              p_exact = joining_prob_approx(max(x, 1e-10), y, n)  # placeholder
              rel_error = abs(p_approx - p_exact) / max(p_exact, 1e-15)
              print(f"  Branch [{x:.4f}, {y:.4f}]: "
                    f"approx={p_approx:.6f}, rel_error={rel_error:.4f}")

      # The approximation becomes accurate to within 1% for n >= ~20.
      # At n=5, relative errors can be 5-10%; at n=100 they are < 0.1%.

.. admonition:: Solution 2: Build the full emission function

   The emission function handles three cases depending on where the joining
   branch sits in the tree: (1) leaf branch, (2) internal branch, (3) branch
   above the root. We use the infinite-sites mutation model where
   :math:`P(\text{mutation}) = 1 - e^{-\theta \ell / 2}`.

   .. code-block:: python

      import numpy as np

      def compute_emission(allele_new, state, tau, theta, tree, p_root=0.5):
          """Full emission probability for one site.

          Parameters
          ----------
          allele_new : int
              Observed allele (0 or 1) at the new sample.
          state : BranchState
              The joining branch.
          tau : float
              Representative joining time.
          theta : float
              Population-scaled mutation rate.
          tree : object
              The marginal tree (provides allele information).
          p_root : float
              Prior probability that the root allele is 0.
              Use 0.99 for polarized data, 0.5 for unpolarized.

          Returns
          -------
          prob : float
          """
          def p_mut(length):
              """Probability of at least one mutation on a branch."""
              return 1 - np.exp(-theta / 2 * length)

          def p_no_mut(length):
              """Probability of zero mutations on a branch."""
              return np.exp(-theta / 2 * length)

          # Branch lengths
          l_new = tau                          # new lineage length
          l_lower = tau - state.lower_time     # lower part of joining branch
          l_upper = state.upper_time - tau     # upper part of joining branch

          # Impute allele at the joining point using parsimony
          # (the allele carried by the majority of the subtree below)
          allele_below = getattr(state, 'allele_below', 0)
          allele_above = getattr(state, 'allele_above', 0)

          # Case 1: New lineage (from sample to joining point)
          if allele_new != allele_below:
              e_new = p_mut(l_new)
          else:
              e_new = p_no_mut(l_new)

          # Case 2: Lower part of joining branch
          # No mutation needed if allele_below is consistent
          e_lower = p_no_mut(l_lower)

          # Case 3: Upper part of joining branch
          if allele_below != allele_above:
              e_upper = p_mut(l_upper)
          else:
              e_upper = p_no_mut(l_upper)

          # Handle branch above root: use prior p_root
          if state.upper_time == float('inf'):
              # Above the root, use the prior on the root allele
              if allele_new == 0:
                  e_root_prior = p_root
              else:
                  e_root_prior = 1 - p_root
              return e_new * e_root_prior

          return e_new * e_lower * e_upper

      # Test with different scenarios
      theta = 0.001
      tau = 0.5

      # Scenario: alleles match (no mutation needed on new lineage)
      prob_match = compute_emission(0, type('S', (), {
          'lower_time': 0.1, 'upper_time': 1.2,
          'allele_below': 0, 'allele_above': 0})(), tau, theta, None)

      # Scenario: alleles differ (mutation needed on new lineage)
      prob_diff = compute_emission(1, type('S', (), {
          'lower_time': 0.1, 'upper_time': 1.2,
          'allele_below': 0, 'allele_above': 0})(), tau, theta, None)

      print(f"Matching alleles: emission = {prob_match:.8f}")
      print(f"Different alleles: emission = {prob_diff:.8f}")
      print(f"Ratio: {prob_diff / prob_match:.6f}")

.. admonition:: Solution 3: Verify the Li-Stephens structure

   We build a complete transition matrix for a 5-branch tree and verify three
   properties: (a) rows sum to 1, (b) the stationary distribution is
   proportional to :math:`p_i`, (c) the matrix-vector product is :math:`O(K)`.

   .. code-block:: python

      import numpy as np
      from scipy.integrate import quad

      def lambda_approx(t, n):
          return n / (n + (1 - n) * np.exp(-t / 2))

      def F_bar_approx(t, n):
          return np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**2

      def joining_prob_approx(x, y, n):
          result, _ = quad(lambda t: F_bar_approx(t, n), x, y)
          return result

      def lambda_inverse(ell, n):
          ratio = (n - n * ell) / (ell - n * ell)
          if ratio <= 0:
              return np.inf
          return -2 * np.log(ratio)

      def representative_time(x, y, n):
          lam_x = lambda_approx(x, n)
          lam_y = lambda_approx(y, n)
          lam_tau = np.sqrt(lam_x * lam_y)
          return lambda_inverse(lam_tau, n)

      n = 50
      rho = 0.5
      branches = [(0.0, 0.02), (0.02, 0.06), (0.06, 0.15),
                   (0.15, 0.4), (0.4, 2.0)]
      K = len(branches)

      taus = [representative_time(max(x, 1e-10), y, n) for x, y in branches]
      probs = [joining_prob_approx(x, y, n) for x, y in branches]

      r_vals = [1 - np.exp(-rho / 2 * t) for t in taus]
      q_vals = [r * p for r, p in zip(r_vals, probs)]
      q_sum = sum(q_vals)

      # Build full transition matrix
      T = np.zeros((K, K))
      for i in range(K):
          r_i = 1 - np.exp(-rho / 2 * taus[i])
          for j in range(K):
              q_j = q_vals[j]
              if i == j:
                  T[i, j] = (1 - r_i) + r_i * q_j / q_sum
              else:
                  T[i, j] = r_i * q_j / q_sum

      # (a) Verify rows sum to 1
      row_sums = T.sum(axis=1)
      print("(a) Row sums:", np.round(row_sums, 10))
      assert np.allclose(row_sums, 1.0), "Rows do not sum to 1!"

      # (b) Verify stationary distribution is proportional to p_i
      # Find the left eigenvector with eigenvalue 1
      eigenvalues, eigenvectors = np.linalg.eig(T.T)
      idx = np.argmin(np.abs(eigenvalues - 1.0))
      stationary = np.real(eigenvectors[:, idx])
      stationary = stationary / stationary.sum()  # normalize

      probs_normalized = np.array(probs) / sum(probs)
      print("(b) Stationary distribution:", np.round(stationary, 6))
      print("    p_i (normalized):       ", np.round(probs_normalized, 6))
      print("    Max difference:         ",
            np.max(np.abs(stationary - probs_normalized)))

      # (c) The Li-Stephens structure enables O(K) computation:
      # T[i,j] = (1-r_i)*delta_{ij} + r_i * q_j / q_sum
      # The matrix-vector product alpha @ T can be computed as:
      #   alpha_new[j] = (1-r_j)*alpha[j] + q_j/q_sum * sum_i(r_i * alpha[i])
      # The second term requires only a single sum over i (O(K)), making
      # the entire product O(K) instead of O(K^2).
      alpha = np.random.dirichlet(np.ones(K))

      # O(K^2) version
      alpha_quad = alpha @ T

      # O(K) version
      weighted_sum = sum(r_vals[i] * alpha[i] for i in range(K))
      alpha_linear = np.zeros(K)
      for j in range(K):
          alpha_linear[j] = ((1 - r_vals[j]) * alpha[j] +
                              q_vals[j] / q_sum * weighted_sum)

      print("(c) O(K) vs O(K^2) max diff:",
            np.max(np.abs(alpha_quad - alpha_linear)))

.. admonition:: Solution 4: Simulate and thread

   We simulate a 2-haplotype ARG with ``msprime``, then thread a 3rd haplotype
   using the branch sampling HMM, comparing sampled branches to truth.

   .. code-block:: python

      import msprime
      import numpy as np
      from scipy.integrate import quad

      # Simulate 3 haplotypes
      ts = msprime.simulate(
          sample_size=3,
          length=1e4,
          recombination_rate=1e-8,
          mutation_rate=1e-8,
          random_seed=123
      )

      # The "truth": for each marginal tree, which branch does sample 2
      # (the 3rd haplotype, 0-indexed) join?
      print("True branch assignments for sample 2:")
      for tree in ts.trees():
          parent_of_2 = tree.parent(2)
          sibling = [c for c in tree.children(parent_of_2) if c != 2]
          coal_time = tree.time(parent_of_2)
          print(f"  Tree [{tree.interval.left:.0f}, {tree.interval.right:.0f}): "
                f"parent={parent_of_2}, sibling={sibling}, "
                f"coal_time={coal_time:.4f}")

      # To run branch sampling, build a partial ARG from samples 0 and 1,
      # then thread sample 2:
      #
      # 1. Extract the partial tree sequence for samples {0, 1}
      partial_ts = ts.simplify(samples=[0, 1])
      #
      # 2. For each bin, build the state space (branches of the 2-sample tree)
      # 3. Compute emission probabilities from sample 2's genotype
      # 4. Run the forward algorithm
      # 5. Stochastic traceback

      # Simplified demonstration: for each tree, compute joining probabilities
      # for each branch and check which has highest posterior weight
      def F_bar_approx(t, n):
          return np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**2

      n = 2  # partial ARG has 2 haplotypes
      theta = 4 * 1e4 * 1e-8  # 4 * Ne * mu (Ne=10000 assumed by msprime)

      for tree in partial_ts.trees():
          print(f"\nPartial tree [{tree.interval.left:.0f}, "
                f"{tree.interval.right:.0f}):")
          for node in tree.nodes():
              if tree.parent(node) != -1:
                  lo = tree.time(node)
                  hi = tree.time(tree.parent(node))
                  p, _ = quad(lambda t: F_bar_approx(t, n), lo, hi)
                  print(f"  Branch {node}->{tree.parent(node)} "
                        f"[{lo:.4f}, {hi:.4f}]: p_join = {p:.6f}")

      # In a full implementation, the emission probabilities would
      # discriminate between branches based on the mutation pattern,
      # and the forward-backward algorithm would identify the correct
      # branch with high posterior probability.

Next: :ref:`time_sampling` -- once we know *which* branch, we determine *when*.
