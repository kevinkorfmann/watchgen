.. _psmc_discretization:

===================
Discretizing Time
===================

   *To fit continuous gears into a digital mechanism, we need to carve them into teeth.*

In the :ref:`previous chapter <psmc_continuous>`, we built the continuous-time
PSMC model -- the mathematical escapement that describes how coalescence times
change from one genomic position to the next. That model lives in the world of
smooth functions, integrals, and continuous probability densities.

But a Hidden Markov Model, the engine we will use in the :ref:`next chapter
<psmc_hmm>` to actually fit the model to data, needs **discrete states**. An
HMM cannot work with a continuous infinity of possible coalescence times. It
needs a finite list of states, a finite transition matrix, and finite emission
probabilities.

This is where discretization comes in. Think of it as the watchmaker's task of
carving continuous gears into discrete teeth. The continuous escapement produces
a smooth oscillation; the gear train converts it into countable, clickable
increments. Each "tooth" on the gear corresponds to one time interval, and the
gear ratio table -- the transition matrix -- tells us the probability of moving
from any tooth to any other.

In this chapter, we will:

1. **Choose the time intervals** -- decide where to place the boundaries between
   teeth on the gear, using a log-spacing scheme matched to coalescent physics.
2. **Derive the helper quantities** -- build the mathematical machinery
   (survival factors, re-coalescence sums) that make the transition matrix
   computable.
3. **Compute the stationary distribution** -- find the long-run probability of
   being in each interval.
4. **Build the transition matrix** -- the gear ratio table at the heart of the
   discrete HMM.
5. **Compute effective coalescence times** -- the representative times used for
   emission probabilities.
6. **Group parameters** -- reduce the number of free parameters to avoid
   overfitting.

By the end, you will have a complete, tested Python implementation that converts
the continuous PSMC model into a working HMM specification. Every formula will be
verified against numerical integration, so there is no room for doubt about
correctness.


Step 1: Choosing Time Intervals
=================================

The first decision: how do we partition the continuous time axis
:math:`[0, \infty)` into a finite number of intervals?

PSMC divides the time axis into :math:`n + 1` intervals:

.. math::

   [t_0, t_1), \quad [t_1, t_2), \quad \ldots, \quad [t_n, t_{n+1})

where :math:`t_0 = 0` and :math:`t_{n+1} = \infty`. The coalescence time at
any genomic position must fall into exactly one of these intervals. That interval
becomes the HMM's hidden state at that position.

**How should we space the intervals?** This is a design choice with real
consequences, and the answer is: **not uniformly**.

.. admonition:: Probability Aside: Why uniform spacing wastes resolution

   Recall from :ref:`coalescent theory <coalescent_theory>` that under a
   constant population of size :math:`N`, the coalescence time :math:`T` for two
   lineages follows an exponential distribution with rate 1 (in coalescent
   units of :math:`2N` generations). The probability density is
   :math:`f(t) = e^{-t}`, which is heavily concentrated near :math:`t = 0` and
   has a long, thin tail stretching toward infinity.

   If we used 64 uniform intervals from 0 to 15, each would span
   :math:`15/64 \approx 0.23` coalescent units. The first few intervals would
   each contain a substantial fraction of the probability mass -- perhaps 10-20%
   of all coalescence events fall in the first interval alone. But the last
   20 intervals together would contain less than 0.001% of events. We would
   waste most of our resolution on a region where almost nothing happens, while
   cramming all the action into a few crowded intervals near :math:`t = 0`.

   The solution is to use narrow intervals where the density is high (recent
   times) and wide intervals where the density is thin (ancient times). This way,
   each interval carries roughly equal statistical weight -- a principle called
   **equal information content**.

PSMC uses **approximately log-spaced** intervals, defined by:

.. math::

   t_i = \alpha \left(e^{i \beta} - 1\right), \quad \text{where } \beta = \frac{1}{n}\ln\left(1 + \frac{t_{\max}}{\alpha}\right)

The parameter :math:`\alpha` (default: 0.1) controls the spacing near
:math:`t = 0`, and :math:`t_{\max}` is the maximum time we consider. To
understand why this formula works, consider a few cases:

- **At** :math:`i = 0`: :math:`t_0 = \alpha(e^0 - 1) = 0`. The formula
  automatically starts at zero.
- **At small** :math:`i`: the exponential :math:`e^{i\beta}` is close to
  :math:`1 + i\beta`, so :math:`t_i \approx \alpha \cdot i\beta` -- nearly
  linear (uniform) spacing. The intervals are narrow, matching the high density
  of coalescence events in the recent past.
- **At large** :math:`i`: the exponential dominates, and the intervals grow
  geometrically. :math:`t_{i+1} - t_i \approx \alpha \beta \, e^{i\beta}`,
  which increases with :math:`i`. Wide intervals in the deep past, where
  coalescence events are rare.
- **At** :math:`i = n`: :math:`t_n = t_{\max}` exactly, by construction of
  :math:`\beta`.

The parameter :math:`\alpha` controls the crossover between the linear and
exponential regimes. A smaller :math:`\alpha` packs more intervals near
:math:`t = 0`; a larger :math:`\alpha` spreads them more uniformly. The default
of 0.1 was chosen by Li and Durbin to work well for human demographic history.

.. code-block:: python

   import numpy as np

   def compute_time_intervals(n, t_max, alpha=0.1):
       """Compute PSMC time interval boundaries.

       Parameters
       ----------
       n : int
           Number of intervals minus 1 (so there are n+1 intervals).
       t_max : float
           Maximum time (in coalescent units of 2*N_0 generations).
       alpha : float
           Controls spacing near t=0.

       Returns
       -------
       t : ndarray of shape (n + 2,)
           Boundaries [t_0, t_1, ..., t_n, t_{n+1}].
           t_0 = 0, t_{n+1} = infinity (represented as a large number).
       """
       # beta controls the overall rate of exponential growth in the spacing
       beta = np.log(1 + t_max / alpha) / n

       t = np.zeros(n + 2)
       for k in range(n):
           # alpha * (exp(beta*k) - 1) gives near-linear spacing for small k
           # and exponential spacing for large k
           t[k] = alpha * (np.exp(beta * k) - 1)
       t[n] = t_max
       t[n + 1] = 1000.0  # numerical infinity: large enough that survival is ~0

       return t

   # Example: 64 intervals (n=63), t_max = 15
   n = 63
   t = compute_time_intervals(n, t_max=15.0, alpha=0.1)

   print("First 10 interval boundaries:")
   for k in range(10):
       print(f"  t[{k}] = {t[k]:.6f}")
   print(f"  ...")
   print(f"  t[{n}] = {t[n]:.6f}")
   print(f"  t[{n+1}] = {t[n+1]:.6f}")
   print(f"\nInterval widths (first 5): "
         f"{[f'{t[k+1]-t[k]:.4f}' for k in range(5)]}")
   print(f"Interval widths (last 5):  "
         f"{[f'{t[k+1]-t[k]:.4f}' for k in range(n-4, n+1)]}")


.. admonition:: Why this particular spacing?

   The log-spacing ensures roughly equal **information content** per interval.
   Under the coalescent, the probability of coalescence in interval :math:`k` is
   approximately proportional to the interval width in "coalescent-CDF space."
   By making intervals wider in the distant past (where the density is thin),
   each interval contributes a similar amount of statistical power to the
   inference.

Notice what the output reveals: the first few intervals are tiny (widths on the
order of 0.01), while the last interval stretches from :math:`t_{63} = 15.0` to
:math:`t_{64} = 1000.0`. This is exactly the gear-tooth pattern we want -- fine
teeth near the recent past where the escapement ticks rapidly, and coarse teeth
in the distant past where ticks are rare.


Step 2: The Helper Quantities
================================

With the time intervals in hand, we now need to build the mathematical
machinery that connects the continuous PSMC model (from
:ref:`the previous chapter <psmc_continuous>`) to a discrete transition matrix.
This requires several helper quantities, each of which has a clear physical
meaning.

Recall from the continuous model that the population size history is
represented as a **piecewise-constant** function: within each interval
:math:`[t_k, t_{k+1})`, the relative population size is a constant
:math:`\lambda_k`. This is the PSMC assumption -- the gears are flat-topped
teeth, not smooth curves.

.. admonition:: Calculus Aside: What "piecewise constant" means for integration

   A piecewise-constant function is one that takes a constant value on each
   interval of a partition. For PSMC, :math:`\lambda(t) = \lambda_k` whenever
   :math:`t \in [t_k, t_{k+1})`. The power of this assumption is that it
   makes integrals trivial within each interval. For example, if we need
   :math:`\int_{t_k}^{t_{k+1}} \frac{dt}{\lambda(t)}`, we can pull
   :math:`1/\lambda_k` out of the integral to get
   :math:`\frac{t_{k+1} - t_k}{\lambda_k} = \frac{\tau_k}{\lambda_k}`. This
   is the same idea that makes Riemann sums work: approximate a curve by
   rectangles, and each rectangle's area is just width times height. In PSMC,
   the "approximation" is exact by assumption -- we are *choosing* the
   population size to be constant within each interval.

Let us define the helper quantities one by one.

**The interval widths** :math:`\tau_k`:

.. math::

   \tau_k = t_{k+1} - t_k

These are simply the widths of each gear tooth. Nothing subtle here, but they
appear in almost every formula that follows.

**The survival factors** :math:`\alpha_k`:

.. math::

   \alpha_k = \exp\left(-\sum_{i=0}^{k-1} \frac{\tau_i}{\lambda_i}\right)

**Intuition:** :math:`\alpha_k` is the probability that two lineages have NOT
coalesced by time :math:`t_k`. In the continuous model from
:ref:`the previous chapter <psmc_continuous>`, we defined the cumulative
hazard :math:`\Lambda(t) = \int_0^t \frac{du}{\lambda(u)}`, and the survival
probability was :math:`e^{-\Lambda(t)}`. The discrete version is identical,
except that the integral becomes a sum over intervals (because :math:`\lambda`
is piecewise constant):

.. math::

   \Lambda(t_k) = \sum_{i=0}^{k-1} \frac{\tau_i}{\lambda_i}

so :math:`\alpha_k = e^{-\Lambda(t_k)}`.

.. admonition:: Probability Aside: Survival as a product of independent escapes

   There is another way to read the formula for :math:`\alpha_k`. Because the
   exponential of a sum is the product of exponentials:

   .. math::

      \alpha_k = \prod_{i=0}^{k-1} e^{-\tau_i / \lambda_i}

   Each factor :math:`e^{-\tau_i / \lambda_i}` is the probability of surviving
   (not coalescing) through interval :math:`i` alone. The product over all
   intervals up to :math:`k` gives the probability of surviving through *all*
   of them -- like passing through :math:`k` successive gates, each with its
   own probability of stopping you.

Note two boundary conditions: :math:`\alpha_0 = 1` (everyone is alive at time 0
-- the empty sum gives zero, and :math:`e^0 = 1`) and
:math:`\alpha_{n+1} = 0` (everyone has coalesced by :math:`t_{n+1} = \infty`).

**The re-coalescence sums** :math:`\beta_k`:

.. math::

   \beta_k = \sum_{i=0}^{k-1} \lambda_i \left(\frac{1}{\alpha_{i+1}} - \frac{1}{\alpha_i}\right)

**Intuition:** :math:`\beta_k` accumulates information about how "easy" it was to
coalesce in each interval up to :math:`k`. It appears in the transition matrix
because after a recombination event breaks the genealogy, the new lineage must
re-coalesce, and the cumulative coalescence opportunity in each past interval
determines how likely each target interval is. Think of it as measuring how
many teeth the gear has exposed for re-engagement up to position :math:`k`.

**An auxiliary quantity** :math:`q_{\text{aux},k}`:

.. math::

   q_{\text{aux},k} = (\alpha_k - \alpha_{k+1})\left(\beta_k - \frac{\lambda_k}{\alpha_k}\right) + \tau_k

This combines survival and re-coalescence information and appears directly in the
transition matrix formulas. It is not physically intuitive on its own -- it is an
algebraic convenience that simplifies the three-case transition matrix into clean
expressions. We will see its role in Step 4.

.. code-block:: python

   def compute_helpers(n, t, lambdas):
       """Compute the helper quantities for the discrete PSMC.

       Parameters
       ----------
       n : int
       t : ndarray of shape (n + 2,)
           Time boundaries.
       lambdas : ndarray of shape (n + 1,)
           Relative population size in each interval.

       Returns
       -------
       tau : ndarray of shape (n + 1,)
       alpha : ndarray of shape (n + 2,)
       beta : ndarray of shape (n + 1,)
       q_aux : ndarray of shape (n,)
       C_pi : float
       """
       tau = np.zeros(n + 1)
       alpha = np.zeros(n + 2)
       beta = np.zeros(n + 1)
       q_aux = np.zeros(n)

       # tau_k: simple interval widths
       for k in range(n + 1):
           tau[k] = t[k + 1] - t[k]

       # alpha_k: survival probability up to time t_k
       # alpha[0] = 1 because no time has passed, so no coalescence is possible
       alpha[0] = 1.0
       for k in range(1, n + 1):
           # Each step multiplies by the survival factor for one interval:
           # exp(-tau/lambda) is the probability of NOT coalescing in that interval
           alpha[k] = alpha[k - 1] * np.exp(-tau[k - 1] / lambdas[k - 1])
       # By convention, alpha[n+1] = 0 (everything has coalesced by t = infinity)
       alpha[n + 1] = 0.0

       # beta_k: cumulative re-coalescence opportunity
       # beta[0] = 0 because the sum has no terms (empty sum)
       beta[0] = 0.0
       for k in range(1, n + 1):
           # Each term adds the contribution from interval k-1:
           # lambda_{k-1} * (1/alpha_k - 1/alpha_{k-1})
           beta[k] = beta[k - 1] + lambdas[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1])

       # q_aux: the auxiliary quantity used in transition matrix construction
       for k in range(n):
           ak1 = alpha[k] - alpha[k + 1]   # probability of coalescing IN interval k
           q_aux[k] = ak1 * (beta[k] - lambdas[k] / alpha[k]) + tau[k]

       # C_pi: normalization constant for the stationary distribution.
       # This equals the expected coalescence time E[T] under the piecewise model.
       C_pi = 0.0
       for k in range(n + 1):
           # lambda_k * (alpha_k - alpha_{k+1}) is lambda_k times the probability
           # of coalescing in interval k
           C_pi += lambdas[k] * (alpha[k] - alpha[k + 1])

       return tau, alpha, beta, q_aux, C_pi

   # Test with constant population (all lambda = 1)
   n = 10
   t = compute_time_intervals(n, t_max=15.0)
   lambdas = np.ones(n + 1)
   tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)

   print(f"C_pi = {C_pi:.6f} (expected: ~1.0 for constant pop)")
   print(f"\nalpha (survival probabilities):")
   for k in range(min(n + 2, 8)):
       print(f"  alpha[{k}] = {alpha[k]:.6f}")
   print(f"  alpha[{n+1}] = {alpha[n+1]:.6f}")

.. admonition:: Calculus Aside: Why :math:`C_\pi` equals the expected coalescence time

   The quantity :math:`C_\pi = \sum_k \lambda_k (\alpha_k - \alpha_{k+1})`
   is the normalizing constant for the stationary distribution :math:`\pi(t)`,
   and it equals :math:`\mathbb{E}[T]`, the expected coalescence time under
   the piecewise-constant population model. Here is why:

   .. math::

      \mathbb{E}[T] = \int_0^\infty t \cdot f(t) \, dt
      = \int_0^\infty S(t) \, dt

   where :math:`S(t) = e^{-\Lambda(t)}` is the survival function. The second
   equality is a standard result (integration by parts). Under our piecewise
   model, :math:`S(t)` is piecewise exponential, and integrating it interval
   by interval gives exactly :math:`\sum_k \lambda_k (\alpha_k - \alpha_{k+1})`.
   For a constant population (:math:`\lambda_k = 1` for all :math:`k`), this
   should equal 1.0 -- the mean of an Exp(1) distribution. The output above
   confirms this.

With these helpers computed, we have all the gears we need to build the
transition matrix. But first, let us compute the stationary distribution --
the long-run equilibrium of the HMM.


Step 3: The Discrete Stationary Distribution
===============================================

The stationary distribution tells us: in the long run, what fraction of
genomic positions have their coalescence time in interval :math:`k`? This is
the discrete analogue of the continuous stationary distribution :math:`\pi(t)`
derived in :ref:`the previous chapter <psmc_continuous>`.

The probability that the coalescence time falls in interval :math:`[t_k, t_{k+1})`
is obtained by integrating :math:`\pi(t)` over the interval:

.. math::

   \pi_k = \int_{t_k}^{t_{k+1}} \pi(t) \, dt

.. admonition:: Calculus Aside: What this integral means concretely

   The continuous distribution :math:`\pi(t)` gives a probability *density*
   -- the probability per unit time of the coalescence happening at exactly
   time :math:`t`. To find the probability of the coalescence time falling
   in a finite interval :math:`[t_k, t_{k+1})`, we sum up (integrate) the
   density over that interval. This is exactly the area under the
   :math:`\pi(t)` curve between :math:`t_k` and :math:`t_{k+1}`.

   If :math:`\pi(t)` were a complicated function, we might need numerical
   integration (evaluating the function at many points and summing weighted
   values). But because PSMC assumes :math:`\lambda(t)` is piecewise
   constant, the integral has a closed-form solution -- the exponentials
   integrate analytically.

Under piecewise-constant :math:`\lambda(t)` (the PSMC assumption), this integral
can be evaluated analytically:

.. math::

   \pi_k = \frac{1}{C_\pi} \left[(\alpha_k - \alpha_{k+1})\left(\sum_{i=0}^{k-1} \tau_i + \lambda_k\right) - \alpha_{k+1} \tau_k\right]

**Derivation.** Within interval :math:`[t_k, t_{k+1})`, :math:`\lambda(t) = \lambda_k`,
so :math:`\pi(t) = \frac{t}{C_\pi \lambda_k} e^{-\Lambda(t)}`. The integral
becomes:

.. math::

   \pi_k = \frac{\alpha_k}{C_\pi \lambda_k} \int_{t_k}^{t_{k+1}} e^{-(t-t_k)/\lambda_k} \left[\sum_{i=0}^{k-1} \tau_i + (t - t_k)\right] dt

The first term involves :math:`\int x e^{-x/\lambda} dx` and the second involves
:math:`\int e^{-x/\lambda} dx`, both elementary. After evaluating and simplifying
(using :math:`\alpha_{k+1} = \alpha_k e^{-\tau_k/\lambda_k}`), we get the result above.

.. admonition:: Calculus Aside: Working out the integral explicitly

   Let :math:`w = t - t_k` so :math:`w \in [0, \tau_k)`. Within interval :math:`k`,
   :math:`\Lambda(t) = \Lambda(t_k) + w/\lambda_k`, so
   :math:`e^{-\Lambda(t)} = \alpha_k e^{-w/\lambda_k}`. The integral becomes:

   .. math::

      \pi_k = \frac{\alpha_k}{C_\pi \lambda_k} \int_0^{\tau_k}
         \left(S_k + w\right) e^{-w/\lambda_k} \, dw

   where :math:`S_k = \sum_{i=0}^{k-1} \tau_i`. We need two standard integrals:

   1. :math:`\int_0^a e^{-w/\lambda}\,dw = \lambda(1 - e^{-a/\lambda})` -- the
      integral of a decaying exponential, giving a "charging curve."

   2. :math:`\int_0^a w\, e^{-w/\lambda}\,dw = \lambda^2(1 - e^{-a/\lambda}) - a\lambda e^{-a/\lambda}`
      -- found by integration by parts (differentiate :math:`w`, integrate
      :math:`e^{-w/\lambda}`).

   Substituting these, using :math:`\alpha_{k+1}/\alpha_k = e^{-\tau_k/\lambda_k}`,
   and simplifying gives the closed form above.

.. code-block:: python

   # Verify pi_k by numerical integration of the continuous pi(t)
   from scipy.integrate import quad

   def verify_pi_k(k, t, lambdas, alpha, C_pi):
       """Compare analytic pi_k to numerical integration of continuous pi(t)."""
       def pi_continuous(t_val):
           # Compute Lambda(t_val) via piecewise sum: add up tau_i/lambda_i
           # for each interval that t_val has passed through
           Lambda = 0.0
           for i in range(len(lambdas)):
               t_lo, t_hi = t[i], t[i + 1]
               if t_val <= t_lo:
                   break
               # min(t_val, t_hi) - t_lo gives the time spent in interval i
               dt = min(t_val, t_hi) - t_lo
               Lambda += dt / lambdas[i]
               if t_val <= t_hi:
                   break
           # pi(t) = t / (C_pi * lambda(t)) * exp(-Lambda(t))
           # but only if t falls in interval k
           return t_val / (C_pi * lambdas[k]) * np.exp(-Lambda) if t[k] <= t_val < t[k+1] else 0.0

       # quad performs adaptive numerical integration (Gaussian quadrature)
       # It returns (value, error_estimate)
       numerical, _ = quad(pi_continuous, t[k] + 1e-10, t[k + 1] - 1e-10)
       return numerical

   # Quick check for the first few intervals
   n_check = 10
   t_check = compute_time_intervals(n_check, t_max=15.0)
   lam_check = np.ones(n_check + 1)
   tau_c, alpha_c, _, _, C_pi_c = compute_helpers(n_check, t_check, lam_check)
   pi_analytic, _, _ = compute_stationary(n_check, tau_c, alpha_c, lam_check, C_pi_c, 0.001)

   for k in range(min(5, n_check + 1)):
       pi_num = verify_pi_k(k, t_check, lam_check, alpha_c, C_pi_c)
       print(f"  pi[{k}]: analytic={pi_analytic[k]:.6f}, numerical={pi_num:.6f}, "
             f"diff={abs(pi_analytic[k] - pi_num):.2e}")

This numerical check is important. If the analytic and numerical values disagree,
we have a bug. Agreement to machine precision (differences on the order of
:math:`10^{-10}` or smaller) confirms that our closed-form expression is correct.

The **full stationary distribution** :math:`\sigma_k` includes the recombination
factor. Recall from :ref:`the continuous model <psmc_continuous>` that the
stationary distribution of the HMM is not just :math:`\pi_k` (the coalescence
time distribution) but also accounts for how recombination probability depends
on branch length:

.. math::

   \sigma_k = \frac{1}{C_\sigma} \left[\frac{\alpha_k - \alpha_{k+1}}{C_\pi \rho} + \frac{\pi_k}{2} + o(\rho)\right]

The first term dominates when :math:`\rho` is small (rare recombination), and
says that the probability of being in interval :math:`k` is proportional to
:math:`\alpha_k - \alpha_{k+1}` -- the probability of *coalescing* in that
interval. The second term, proportional to :math:`\pi_k`, is a correction
for finite recombination rate.

.. code-block:: python

   def compute_stationary(n, tau, alpha, lambdas, C_pi, rho):
       """Compute discrete stationary distributions pi_k and sigma_k.

       Parameters
       ----------
       n : int
       tau, alpha : from compute_helpers
       lambdas : ndarray of shape (n + 1,)
       C_pi : float
       rho : float

       Returns
       -------
       pi_k : ndarray of shape (n + 1,)
       sigma_k : ndarray of shape (n + 1,)
       C_sigma : float
       """
       pi_k = np.zeros(n + 1)
       sum_tau = 0.0   # running sum of interval widths: sum_{i=0}^{k-1} tau_i

       for k in range(n + 1):
           ak1 = alpha[k] - alpha[k + 1]   # probability of coalescing in interval k
           # The formula combines two parts:
           #   ak1 * (sum_tau + lambda_k): contribution from "reaching" interval k
           #   alpha[k+1] * tau[k]: correction for the probability mass near t_{k+1}
           pi_k[k] = (ak1 * (sum_tau + lambdas[k]) - alpha[k + 1] * tau[k]) / C_pi
           sum_tau += tau[k]

       # C_sigma: normalization for the full stationary distribution
       # 1/(C_pi * rho) + 0.5 is a first-order approximation in rho
       C_sigma = 1.0 / (C_pi * rho) + 0.5

       # sigma_k: the full stationary distribution including recombination
       sigma_k = np.zeros(n + 1)
       for k in range(n + 1):
           ak1 = alpha[k] - alpha[k + 1]
           # Two terms: the dominant 1/(C_pi*rho) term and the pi_k/2 correction
           sigma_k[k] = (ak1 / (C_pi * rho) + pi_k[k] / 2.0) / C_sigma

       return pi_k, sigma_k, C_sigma

   rho = 0.001
   pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)

   print("Discrete stationary distribution pi_k:")
   for k in range(min(n + 1, 8)):
       print(f"  pi[{k}] = {pi_k[k]:.6f}")
   print(f"\nSum of pi_k: {pi_k.sum():.6f} (should be 1.0)")
   print(f"Sum of sigma_k: {sigma_k.sum():.6f} (should be 1.0)")
   print(f"C_sigma = {C_sigma:.4f}")

Both :math:`\pi_k` and :math:`\sigma_k` should sum to 1.0 (they are probability
distributions). If they do not, something is wrong with the helper quantities.
The output above provides a sanity check.

With the stationary distribution in hand, we are ready for the heart of this
chapter: the transition matrix.


Step 4: The Discrete Transition Matrix
=========================================

Now we arrive at the central object of the discretization: the **transition
matrix**. This is the gear ratio table of the PSMC watch -- it tells us, for
every pair of time intervals :math:`(k, l)`, the probability that if the
coalescence time at one genomic position is in interval :math:`k`, the
coalescence time at the next position is in interval :math:`l`.

.. admonition:: Probability Aside: What is a transition matrix?

   A **transition matrix** (also called a stochastic matrix) is a square matrix
   :math:`P` where :math:`P_{kl}` gives the probability of moving from state
   :math:`k` to state :math:`l` in one step. Every row must sum to 1 (from
   any state, you must go *somewhere*), and every entry must be non-negative.

   If you know the current state is :math:`k`, the row :math:`P_{k,:}` gives
   the complete probability distribution over the next state. The matrix
   encodes all the information the HMM needs about state dynamics.

   A probability vector :math:`\sigma` is a **stationary distribution** of
   :math:`P` if :math:`\sigma^T P = \sigma^T` -- applying the transition
   leaves the distribution unchanged. This is the long-run equilibrium: after
   many steps, the probability of being in each state converges to :math:`\sigma`
   regardless of the starting state.

   For more background, see the :ref:`HMM prerequisite chapter <hmms>`.

The discrete transition probability from interval :math:`k` to interval :math:`l`
(conditioned on recombination having occurred) is:

.. math::

   q_{kl} = \frac{1}{\pi_k} \int_{t_k}^{t_{k+1}} ds \int_{t_l}^{t_{l+1}} q(t|s) \pi(s) \, dt

This double integral averages the continuous transition density :math:`q(t|s)`
(derived in :ref:`the previous chapter <psmc_continuous>`) over all source
times :math:`s` in interval :math:`k` and all target times :math:`t` in interval
:math:`l`. The factor :math:`1/\pi_k` normalizes by the probability of starting
in interval :math:`k`.

.. admonition:: Calculus Aside: The double integral as a weighted average

   The formula :math:`q_{kl} = \frac{1}{\pi_k} \int \int q(t|s) \pi(s) \, dt \, ds`
   is a **conditional expectation**. It says: given that the source coalescence
   time :math:`s` is somewhere in interval :math:`k` (with distribution
   :math:`\pi(s) / \pi_k`), what is the probability that the target coalescence
   time :math:`t` lands in interval :math:`l`?

   Under the piecewise-constant assumption, the double integral splits into
   products of single integrals that can each be evaluated analytically. The
   key trick is that :math:`q(t|s)` involves a :math:`\min(s,t)` term (from
   the continuous model), which means the integral behaves differently
   depending on whether :math:`t < s`, :math:`t = s`, or :math:`t > s` -- and
   hence on the relationship between intervals :math:`l` and :math:`k`.

This double integral has different forms depending on whether :math:`l < k`,
:math:`l = k`, or :math:`l > k`. The derivations are lengthy but the results are
clean:

**Case** :math:`l < k` (transition to an earlier interval):

.. math::

   q_{kl} = \frac{\alpha_k - \alpha_{k+1}}{C_\pi \pi_k} \cdot q_{\text{aux},l}

The new coalescence time is *shallower* (more recent) than the old one.
Recombination broke the genealogy, and the new lineage found a partner quickly --
it re-coalesced in an earlier interval. Notice that the :math:`q_{\text{aux},l}`
factor depends only on the *target* interval :math:`l`, not on the source
:math:`k` (except through the leading normalizing factor). This is the
factorization that makes computation efficient.

**Case** :math:`l = k` (staying in the same interval):

.. math::

   q_{kk} = \frac{1}{C_\pi \pi_k} \left[(\alpha_k - \alpha_{k+1})^2 \left(\beta_k - \frac{\lambda_k}{\alpha_k}\right) + 2\lambda_k(\alpha_k - \alpha_{k+1}) - 2\alpha_{k+1}\tau_k\right]

This is the most complex case because the double integral's :math:`\min(s,t)`
term changes behavior within the same interval (sometimes :math:`t < s`,
sometimes :math:`t > s`). Both sub-cases contribute to the result.

**Case** :math:`l > k` (transition to a later interval):

.. math::

   q_{kl} = \frac{\alpha_l - \alpha_{l+1}}{C_\pi \pi_k} \cdot q_{\text{aux},k}

The new coalescence time is *deeper* (more ancient) than the old one. The
recombined lineage drifted into the deeper past before finding a partner. Here,
:math:`q_{\text{aux},k}` depends only on the *source* interval :math:`k`, and
:math:`\alpha_l - \alpha_{l+1}` depends only on the *target*.

**Structural insight:** The factorization across cases is the key to efficiency.
For :math:`l < k`, the contribution to column :math:`l` is the same
:math:`q_{\text{aux},l}` for every row :math:`k > l`. For :math:`l > k`, the
contribution from row :math:`k` is the same :math:`q_{\text{aux},k}` for every
column :math:`l > k`. This means we can fill the entire :math:`(n+1) \times (n+1)`
matrix in :math:`O(n)` work per row, not :math:`O(n^2)`.

.. admonition:: Probability Aside: Why the cases split this way

   The three cases arise from splitting the double integral over the
   :math:`\min(s, t)` in the continuous transition density:

   - **l < k** (:math:`t < s`): the new coalescence time is shallower than
     the old one. Recombination happened somewhere on the branch, and the
     detached lineage re-coalesced earlier. The integral upper limit
     :math:`\min(s,t) = t` removes all dependence on :math:`s` from the
     inner integral.
   - **l > k** (:math:`t > s`): the new coalescence time is deeper.
     Here :math:`\min(s,t) = s`, and the inner integral from :math:`0` to
     :math:`s` captures the full branch length.
   - **l = k**: both cases overlap, producing the most complex expression.

   This factorization means the full :math:`(n+1) \times (n+1)` matrix can be
   computed in :math:`O(n)` time per row (not :math:`O(n^2)`), since the
   :math:`q_{\text{aux}}` values are shared.

The **full transition matrix** (including no-recombination) is:

.. math::

   p_{kl} = \frac{\pi_k}{C_\sigma \sigma_k} q_{kl} + \delta_{kl} \left(1 - \frac{\pi_k}{C_\sigma \sigma_k}\right)

**Why this form?** This is a mixture of two behaviors, weighted by whether
recombination occurred:

- With probability :math:`\frac{\pi_k}{C_\sigma \sigma_k}`, recombination
  happens, and the new coalescence time is drawn from :math:`q_{kl}`. The
  lineage's gear "slips" to a new tooth.
- With probability :math:`1 - \frac{\pi_k}{C_\sigma \sigma_k}`, no
  recombination happens, and the coalescence time stays in the same interval.
  The :math:`\delta_{kl}` (Kronecker delta: 1 if :math:`k = l`, 0 otherwise)
  ensures this contributes only to the diagonal.

In the watch metaphor, this is the difference between a gear tooth engaging a
new position (recombination) and the gear holding steady (no recombination).
Most of the time, adjacent genomic positions share the same coalescence time --
the gear holds. Occasionally, recombination breaks the genealogy and the gear
clicks forward or backward.

.. code-block:: python

   def compute_transition_matrix(n, tau, alpha, beta, q_aux, lambdas,
                                  C_pi, C_sigma, pi_k, sigma_k):
       """Compute the full discrete PSMC transition matrix.

       Parameters
       ----------
       n : int
       tau, alpha, beta, q_aux : from compute_helpers
       lambdas : population sizes
       C_pi, C_sigma : normalization constants
       pi_k, sigma_k : stationary distributions

       Returns
       -------
       p : ndarray of shape (n+1, n+1)
           Transition matrix p_{kl}.
       q : ndarray of shape (n+1, n+1)
           Transition matrix q_{kl} (conditioned on recombination).
       """
       N = n + 1  # number of hidden states (one per time interval)
       q = np.zeros((N, N))  # q[k, l] = transition prob given recombination

       for k in range(N):
           # ak1 = alpha_k - alpha_{k+1}: probability of coalescing in interval k
           ak1 = alpha[k] - alpha[k + 1]

           # cpik = C_pi * pi_k[k]: the unnormalized stationary probability.
           # We compute it directly to avoid dividing by pi_k (which could be ~0).
           cpik = ak1 * (sum(tau[:k]) + lambdas[k]) - alpha[k + 1] * tau[k]

           if cpik < 1e-30:
               # Fallback for negligible-probability intervals: uniform transitions
               q[k, :] = 1.0 / N
               continue

           # Case 1: l < k  (transition to an earlier/shallower interval)
           # q[k,l] = ak1 / cpik * q_aux[l]
           for l in range(k):
               q[k, l] = ak1 / cpik * q_aux[l]

           # Case 2: l = k  (staying in the same interval)
           q[k, k] = (ak1 * ak1 * (beta[k] - lambdas[k] / alpha[k])
                       + 2 * lambdas[k] * ak1
                       - 2 * alpha[k + 1] * tau[k]) / cpik

           # Case 3: l > k  (transition to a later/deeper interval)
           if k < n:
               for l in range(k + 1, N):
                   # (alpha_l - alpha_{l+1}) / cpik * q_aux[k]
                   q[k, l] = (alpha[l] - alpha[l + 1]) / cpik * q_aux[k]

       # Full transition matrix p_{kl}: mixture of recombination and no-recombination
       p = np.zeros((N, N))
       for k in range(N):
           # recomb_prob = pi_k / (C_sigma * sigma_k): probability of recombination
           recomb_prob = pi_k[k] / (C_sigma * sigma_k[k]) if sigma_k[k] > 0 else 0
           for l in range(N):
               p[k, l] = recomb_prob * q[k, l]  # recombination contribution
               if k == l:
                   p[k, l] += (1.0 - recomb_prob)  # no-recombination: stay in place

       return p, q

   p, q = compute_transition_matrix(n, tau, alpha, beta, q_aux, lambdas,
                                      C_pi, C_sigma, pi_k, sigma_k)

   print("Transition matrix q (conditioned on recomb):")
   print(f"  Shape: {q.shape}")
   print(f"  Row sums: min={q.sum(axis=1).min():.6f}, "
         f"max={q.sum(axis=1).max():.6f}")

   print(f"\nFull transition matrix p:")
   print(f"  Shape: {p.shape}")
   print(f"  Row sums: min={p.sum(axis=1).min():.6f}, "
         f"max={p.sum(axis=1).max():.6f}")

   # Check that sigma is the stationary distribution of p:
   # sigma^T @ p should equal sigma^T (left eigenvector with eigenvalue 1)
   sigma_check = sigma_k @ p    # matrix-vector product: sigma * P
   max_diff = np.max(np.abs(sigma_check - sigma_k))
   print(f"\nStationary distribution check: max|sigma*p - sigma| = {max_diff:.2e}")

.. admonition:: Probability Aside: Interpreting the verification

   The row sums of both :math:`q` and :math:`p` should be exactly 1.0 (or
   very close, up to floating-point rounding). A row sum different from 1
   would mean that from some state, the total probability of going *somewhere*
   is not 100% -- a physical impossibility.

   The stationary distribution check :math:`\sigma^T P = \sigma^T` is even
   more stringent. It says that if the probability of being in each state is
   given by :math:`\sigma`, then after one step of the Markov chain, the
   distribution is unchanged. The ``@`` operator in NumPy performs matrix
   multiplication, so ``sigma_k @ p`` computes :math:`\sigma^T P` and the
   result should equal :math:`\sigma^T`. Differences on the order of
   :math:`10^{-10}` or smaller indicate success.

We now have the complete gear ratio table. The transition matrix :math:`p` is
one of the three ingredients the HMM needs (along with emission probabilities
and an initial distribution). Let us compute the second ingredient next.


Step 5: The Average Coalescence Time per Interval
====================================================

For emission probabilities, we need a representative coalescence time for each
interval -- not just the midpoint or the left boundary, but the **effective
time** at which mutations accumulate as if the coalescence happened at that
single point.

Why does this matter? The emission probability in the PSMC HMM is
:math:`P(\text{het} | \text{state } k) = 1 - e^{-\theta \bar{t}_k}`, where
:math:`\bar{t}_k` is the effective coalescence time in interval :math:`k` and
:math:`\theta` is the mutation rate. Using the wrong :math:`\bar{t}_k` would
bias the emission probabilities and corrupt the inference.

PSMC uses a clever trick to define :math:`\bar{t}_k`. From the full transition
matrix, the probability of *not* recombining when in interval :math:`k` is
:math:`p_{kk}^{\text{stay}} = 1 - \pi_k/(C_\sigma \sigma_k)`. On a branch of
length :math:`t`, the no-recombination probability is :math:`e^{-\rho t}`.
Setting these equal and solving:

.. math::

   \bar{t}_k = -\frac{1}{\rho} \ln\left(1 - \frac{\pi_k}{C_\sigma \sigma_k}\right)

.. admonition:: Calculus Aside: The logarithm as an inverse of the exponential

   The equation :math:`e^{-\rho \bar{t}} = 1 - r` (where
   :math:`r = \pi_k / (C_\sigma \sigma_k)` is the recombination probability)
   is solved by taking the natural log of both sides:
   :math:`-\rho \bar{t} = \ln(1 - r)`, hence
   :math:`\bar{t} = -\ln(1 - r) / \rho`.

   This is well-defined as long as :math:`r < 1` (there is some probability
   of *not* recombining). When :math:`r` is small (rare recombination), we
   can use the Taylor expansion :math:`-\ln(1-r) \approx r + r^2/2 + \ldots`,
   so :math:`\bar{t} \approx r/\rho`. When :math:`r` approaches 1 (very long
   branches with near-certain recombination), :math:`\bar{t}` grows without
   bound, as expected.

**Why this particular time?** The no-recombination probability on a branch of
length :math:`t` is :math:`e^{-\rho t}`. The discrete model assigns
:math:`p_{kk}^{\text{stay}} = 1 - \pi_k/(C_\sigma \sigma_k)` as the stay probability.
By setting :math:`e^{-\rho \bar{t}_k} = p_{kk}^{\text{stay}}`, we find the
effective time that would produce this stay probability. This same time is then
used for computing the emission (mutation) probability. The beauty of this
approach is that it ensures consistency: the same effective branch length
governs both recombination and mutation.

.. code-block:: python

   def compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho):
       """Compute the effective coalescence time for each interval.

       Parameters
       ----------
       Returns
       -------
       avg_t : ndarray of shape (n + 1,)
       """
       avg_t = np.zeros(n + 1)
       sum_tau = 0.0   # running sum of interval widths

       for k in range(n + 1):
           ak1 = alpha[k] - alpha[k + 1]
           recomb_prob = pi_k[k] / (C_sigma * sigma_k[k]) if sigma_k[k] > 0 else 0

           if recomb_prob < 1.0:
               # Main formula: -log(1 - recomb_prob) / rho
               # np.log computes the natural logarithm (base e)
               avg_t[k] = -np.log(1.0 - recomb_prob) / rho
           else:
               # Fallback: if recomb_prob = 1 (log would be -inf),
               # use the conditional mean coalescence time within the interval
               lak = lambdas[k]
               avg_t[k] = sum_tau + (lak - tau[k] * alpha[k + 1] / ak1
                                      if ak1 > 0 else tau[k] / 2)

           # Sanity check: avg_t[k] should lie within the interval [t_k, t_{k+1}]
           if np.isnan(avg_t[k]) or avg_t[k] < sum_tau or avg_t[k] > sum_tau + tau[k]:
               lak = lambdas[k]
               ak1 = alpha[k] - alpha[k + 1]
               avg_t[k] = sum_tau + (lak - tau[k] * alpha[k + 1] / ak1
                                      if ak1 > 0 else tau[k] / 2)

           sum_tau += tau[k]

       return avg_t

   avg_t = compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho)

   print("Effective coalescence times per interval:")
   for k in range(min(n + 1, 8)):
       print(f"  Interval [{t[k]:.4f}, {t[k+1]:.4f}): "
             f"avg_t = {avg_t[k]:.4f}")

The effective times should lie within their respective intervals. If you see an
effective time that is outside the interval boundaries, the fallback has been
triggered, which is fine for edge cases but should not happen for typical
parameters.


Step 6: Parameter Grouping (The -p Pattern)
=============================================

We have built all the mathematical machinery for an HMM with :math:`n+1` hidden
states, each with its own population size parameter :math:`\lambda_k`. But there
is a practical problem: with 64 time intervals, estimating 64 independent
:math:`\lambda_k` values leads to **overfitting**.

The issue is statistical power. Each :math:`\lambda_k` is estimated from the
recombination events that fall in interval :math:`k`. For narrow intervals in
the recent past, there may be plenty of such events. But for wide intervals in
the distant past, events are sparse, and the estimate of :math:`\lambda_k`
becomes noisy and unreliable.

In the watch metaphor, this is like trying to read a clock face with 64 tick
marks when you can only reliably distinguish 28 of them. Better to group the
fine tick marks into coarser markings that you can read reliably.

PSMC solves this by **grouping** adjacent intervals to share the same
:math:`\lambda` parameter. The ``-p`` option specifies the grouping pattern:

.. code-block:: text

   -p "4+25*2+4+6"

This means:

- The first group spans 4 atomic intervals (they share one :math:`\lambda`)
- The next 25 groups each span 2 atomic intervals
- Then one group spans 4 intervals
- The last group spans 6 intervals

Total: :math:`4 + 25 \times 2 + 4 + 6 = 64` atomic intervals, but only
:math:`1 + 25 + 1 + 1 = 28` free :math:`\lambda` parameters.

The grouping is denser in the middle of the time range (25 groups of 2, giving
fine resolution) and coarser at the extremes (groups of 4 and 6, where the
data are less informative). This is another example of the same philosophy as
log-spacing: put resolution where the information is.

.. code-block:: python

   def parse_pattern(pattern):
       """Parse a PSMC pattern string into a parameter map.

       Parameters
       ----------
       pattern : str
           Pattern like "4+25*2+4+6".

       Returns
       -------
       par_map : list of int
           par_map[k] = index of the free parameter for interval k.
       n_free : int
           Number of free parameters.
       n_intervals : int
           Total number of atomic intervals (= n + 1).
       """
       par_map = []
       free_idx = 0

       for part in pattern.split('+'):
           if '*' in part:
               # "25*2" means 25 groups, each spanning 2 atomic intervals
               count, width = part.split('*')
               count, width = int(count), int(width)
               for _ in range(count):
                   # Each group of 'width' intervals shares one free parameter
                   for _ in range(width):
                       par_map.append(free_idx)
                   free_idx += 1
           else:
               # "4" means one group spanning 4 atomic intervals
               width = int(part)
               for _ in range(width):
                   par_map.append(free_idx)
               free_idx += 1

       return par_map, free_idx, len(par_map)

   par_map, n_free, n_intervals = parse_pattern("4+25*2+4+6")
   print(f"Pattern: 4+25*2+4+6")
   print(f"Total atomic intervals: {n_intervals}")
   print(f"Free parameters: {n_free}")
   print(f"First 10 of par_map: {par_map[:10]}")
   print(f"Last 10 of par_map: {par_map[-10:]}")

.. admonition:: How to choose the pattern

   The rule of thumb from Li and Durbin: each free parameter should span at
   least ~10 expected recombination events. You can check this with
   :math:`C_\sigma \pi_k` -- if this is less than ~20 for the intervals in a
   group, you should merge them into a larger group. The default pattern
   ``4+25*2+4+6`` has been validated for human whole-genome data.

   In practice, you rarely need to change the pattern unless you are working
   with organisms that have very different genome sizes or recombination rates.
   For short genomes (fewer recombination events), you may need fewer, wider
   groups. For very long genomes or high recombination rates, you could afford
   finer resolution.


Step 7: Putting It All Together
=================================

We have now built every piece of the discrete PSMC model: time intervals,
helper quantities, stationary distribution, transition matrix, effective times,
and parameter grouping. Let us assemble them into a single function that takes
population parameters and produces the complete HMM specification -- the three
ingredients that the :ref:`next chapter's <psmc_hmm>` EM algorithm will use.

In the watch metaphor, this is the moment we snap all the gears into the
movement and close the case. The function below is the complete gear train:
continuous time goes in, discrete HMM parameters come out.

.. code-block:: python

   def build_psmc_hmm(n, t_max, theta, rho, lambdas, par_map=None, alpha_param=0.1):
       """Build the complete discrete PSMC HMM parameters.

       Parameters
       ----------
       n : int
           Number of time intervals minus 1.
       t_max : float
       theta : float
           Mutation rate per bin.
       rho : float
           Recombination rate per bin.
       lambdas : ndarray of shape (n + 1,)
           Relative population sizes (lambda_k for each atomic interval).
       par_map : list, optional
           Parameter grouping. If provided, lambdas should have n_free entries
           and will be expanded.
       alpha_param : float
           Spacing parameter for time intervals.

       Returns
       -------
       transitions : ndarray of shape (n+1, n+1)
       emissions : ndarray of shape (2, n+1)
           emissions[0, k] = P(hom | state k), emissions[1, k] = P(het | state k)
       initial : ndarray of shape (n+1,)
       """
       # Expand lambdas if using parameter grouping:
       # par_map[k] maps atomic interval k to its free parameter index
       if par_map is not None:
           full_lambdas = np.array([lambdas[par_map[k]] for k in range(n + 1)])
       else:
           full_lambdas = lambdas

       # Compute time intervals (the gear teeth positions)
       t = compute_time_intervals(n, t_max, alpha_param)

       # Compute helpers (survival factors, re-coalescence sums, etc.)
       tau, alpha_arr, beta, q_aux, C_pi = compute_helpers(n, t, full_lambdas)

       # Stationary distributions (long-run equilibrium of the HMM)
       pi_k, sigma_k, C_sigma = compute_stationary(
           n, tau, alpha_arr, full_lambdas, C_pi, rho)

       # Transition matrix (the gear ratio table)
       transitions, _ = compute_transition_matrix(
           n, tau, alpha_arr, beta, q_aux, full_lambdas,
           C_pi, C_sigma, pi_k, sigma_k)

       # Average times for emissions (effective branch lengths per interval)
       avg_t = compute_avg_times(
           n, tau, alpha_arr, full_lambdas, pi_k, sigma_k, C_sigma, rho)

       # Emission probabilities:
       # P(hom | state k) = exp(-theta * avg_t[k])   (no mutation on either branch)
       # P(het | state k) = 1 - exp(-theta * avg_t[k])  (at least one mutation)
       emissions = np.zeros((2, n + 1))
       for k in range(n + 1):
           # np.exp(-theta * avg_t[k]): probability of zero mutations on a branch
           # of length avg_t[k] with mutation rate theta (Poisson model)
           emissions[0, k] = np.exp(-theta * avg_t[k])      # P(hom | state k)
           emissions[1, k] = 1 - emissions[0, k]             # P(het | state k)

       # Initial distribution: the HMM starts in the stationary distribution
       initial = sigma_k.copy()

       return transitions, emissions, initial

   # Build the HMM
   n = 10
   theta = 0.001
   rho = theta / 5  # theta/rho ratio of 5
   lambdas = np.ones(n + 1)

   transitions, emissions, initial = build_psmc_hmm(
       n, t_max=15.0, theta=theta, rho=rho, lambdas=lambdas)

   print("HMM parameters:")
   print(f"  States: {n + 1}")
   print(f"  Transition matrix shape: {transitions.shape}")
   print(f"  Row sums: {transitions.sum(axis=1)}")
   print(f"  Emission probabilities (het) for first 5 states: "
         f"{emissions[1, :5]}")
   print(f"  Initial distribution sums to: {initial.sum():.6f}")

The output confirms that the HMM is well-formed: all transition matrix rows
sum to 1, and the initial distribution sums to 1. The emission probabilities
for heterozygosity increase with the state index -- deeper coalescence times
mean longer branches, hence more mutations and more heterozygous sites. This
makes physical sense: the watch reads higher numbers on the dial for deeper
times.

With this function, we have completed the gear train. The continuous-time
PSMC model from the :ref:`previous chapter <psmc_continuous>` has been
discretized into a finite HMM with concrete, computable parameters. In the
:ref:`next chapter <psmc_hmm>`, we will connect this HMM to data using the
forward-backward algorithm and EM, allowing the parameters to learn from
the observed sequence of heterozygous and homozygous sites.


Exercises
=========

.. admonition:: Exercise 1: Verify the transition matrix

   Build the transition matrix for a constant population (:math:`\lambda_k = 1`
   for all :math:`k`) with :math:`n = 20` intervals. Verify that:
   (a) all rows sum to 1,
   (b) the stationary distribution :math:`\sigma_k` satisfies
   :math:`\sigma^T P = \sigma^T`,
   (c) starting from any initial distribution, repeated application of :math:`P`
   converges to :math:`\sigma`.

.. admonition:: Exercise 2: Sensitivity to population size

   Build transition matrices for:
   (a) :math:`\lambda_k = 1` (constant),
   (b) :math:`\lambda_k = 10` for intervals near :math:`t = 1` and 1 elsewhere (expansion),
   (c) :math:`\lambda_k = 0.1` for intervals near :math:`t = 1` and 1 elsewhere (bottleneck).

   Compare the emission probabilities. How does the bottleneck affect heterozygosity?

.. admonition:: Exercise 3: The effect of pattern choice

   Using the same data, run PSMC with patterns ``"64*1"`` (all free),
   ``"4+25*2+4+6"`` (default), and ``"32*2"`` (simple). Compare the inferred
   :math:`\lambda_k` curves. Which shows overfitting? Which is too smooth?

Next: :ref:`psmc_hmm` -- the EM algorithm that makes the parameters talk to the data.


Solutions
=========

.. admonition:: Solution 1: Verify the transition matrix

   We build the transition matrix for a constant population and verify three
   fundamental properties: row normalization, stationarity, and convergence.

   .. code-block:: python

      import numpy as np

      # Setup: n=20 intervals, constant population
      n = 20
      t = compute_time_intervals(n, t_max=15.0)
      lambdas = np.ones(n + 1)
      rho = 0.001

      # Compute all helper quantities
      tau, alpha, beta, q_aux, C_pi = compute_helpers(n, t, lambdas)
      pi_k, sigma_k, C_sigma = compute_stationary(n, tau, alpha, lambdas, C_pi, rho)
      p, q = compute_transition_matrix(n, tau, alpha, beta, q_aux, lambdas,
                                        C_pi, C_sigma, pi_k, sigma_k)

      # (a) Verify all rows sum to 1
      row_sums = p.sum(axis=1)
      print("(a) Row sums of P:")
      print(f"  min = {row_sums.min():.10f}")
      print(f"  max = {row_sums.max():.10f}")
      assert np.allclose(row_sums, 1.0, atol=1e-10), "Row sums are not 1!"

      # (b) Verify sigma^T P = sigma^T
      sigma_after = sigma_k @ p
      max_diff = np.max(np.abs(sigma_after - sigma_k))
      print(f"\n(b) Stationarity check: max|sigma*P - sigma| = {max_diff:.2e}")
      assert max_diff < 1e-10, "Sigma is not stationary!"

      # (c) Convergence from arbitrary initial distribution
      # Start from a point mass on state 0
      dist = np.zeros(n + 1)
      dist[0] = 1.0

      print("\n(c) Convergence from delta(state=0):")
      for iteration in [1, 5, 10, 50, 100, 500]:
          # Apply P repeatedly: dist @ P^iteration
          current = dist.copy()
          for _ in range(iteration):
              current = current @ p
          max_err = np.max(np.abs(current - sigma_k))
          print(f"  After {iteration:4d} steps: max|dist - sigma| = {max_err:.6e}")

   **(a)** All row sums should equal 1.0 to within machine precision
   (:math:`\sim 10^{-15}`), confirming that ``p`` is a valid stochastic matrix.

   **(b)** The difference :math:`|\sigma^T P - \sigma^T|` should be on the order
   of :math:`10^{-12}` or smaller, confirming that :math:`\sigma` is indeed the
   stationary distribution of :math:`P`.

   **(c)** The distribution converges to :math:`\sigma` exponentially fast. After
   ~100 steps, the maximum deviation should be negligible (:math:`< 10^{-6}`).
   This demonstrates that regardless of the starting distribution, the Markov
   chain converges to its unique equilibrium -- the stochastic analogue of a
   watch settling into its natural rhythm.

.. admonition:: Solution 2: Sensitivity to population size

   We build transition matrices and emission probabilities for three population
   scenarios and compare how demography affects heterozygosity.

   .. code-block:: python

      import numpy as np

      n = 20
      t = compute_time_intervals(n, t_max=15.0)
      theta = 0.001
      rho = 0.0002

      # Identify intervals near t=1 in coalescent units
      t_mid = [(t[k] + t[k+1]) / 2.0 for k in range(n + 1)]
      near_t1 = [k for k in range(n + 1) if 0.5 < t_mid[k] < 1.5]

      scenarios = {}

      # (a) Constant population
      lam_a = np.ones(n + 1)
      scenarios["(a) Constant"] = lam_a

      # (b) Expansion near t=1: lambda_k = 10 for intervals near t=1
      lam_b = np.ones(n + 1)
      for k in near_t1:
          lam_b[k] = 10.0
      scenarios["(b) Expansion"] = lam_b

      # (c) Bottleneck near t=1: lambda_k = 0.1 for intervals near t=1
      lam_c = np.ones(n + 1)
      for k in near_t1:
          lam_c[k] = 0.1
      scenarios["(c) Bottleneck"] = lam_c

      for name, lam in scenarios.items():
          trans, emissions, initial = build_psmc_hmm(
              n, t_max=15.0, theta=theta, rho=rho, lambdas=lam)

          # Weighted average heterozygosity under the stationary distribution
          avg_het = np.sum(initial * emissions[1, :])
          print(f"{name}:")
          print(f"  Emission P(het) range: [{emissions[1,:].min():.6f}, "
                f"{emissions[1,:].max():.6f}]")
          print(f"  Weighted avg heterozygosity: {avg_het:.6f}")
          print(f"  Intervals near t=1: lambda = {lam[near_t1[0]]:.1f}")

   **How the bottleneck affects heterozygosity:** The bottleneck
   (:math:`\lambda_k = 0.1`) increases the coalescence rate in those intervals,
   meaning lineages are forced to coalesce quickly. This produces shorter effective
   branch lengths and therefore lower emission probabilities
   :math:`P(\text{het} | k) = 1 - e^{-\theta \bar{t}_k}` for the affected
   intervals. The weighted average heterozygosity decreases compared to the
   constant-population case.

   Conversely, the expansion (:math:`\lambda_k = 10`) reduces the coalescence
   rate, allowing longer branches and higher heterozygosity in those intervals.
   The overall effect depends on the stationary weight assigned to those
   intervals -- a large population means fewer coalescence events there, but
   the ones that do occur have longer branch lengths.

.. admonition:: Solution 3: The effect of pattern choice

   We compare three grouping patterns on the same simulated data to demonstrate
   overfitting vs. over-smoothing. The key insight is that the pattern controls
   the bias-variance tradeoff: more free parameters reduce bias but increase
   variance.

   .. code-block:: python

      import numpy as np

      # Simulate data under a known bottleneck model
      np.random.seed(42)
      n = 63
      N = n + 1  # 64 intervals

      # True population history: bottleneck from t=0.5 to t=1.5
      t = compute_time_intervals(n, t_max=15.0)
      true_lambdas = np.ones(N)
      for k in range(N):
          t_mid = (t[k] + t[k+1]) / 2.0
          if 0.5 < t_mid < 1.5:
              true_lambdas[k] = 0.1

      theta = 0.001
      rho = theta / 5

      # Generate observation sequence
      # seq, _ = simulate_psmc_input(100000, theta, rho,
      #     lambda t: 0.1 if 0.5 < t < 1.5 else 1.0)

      # Three patterns to compare
      patterns = {
          "64*1 (all free)":    "64*1",      # 64 free parameters
          "4+25*2+4+6 (default)": "4+25*2+4+6",  # 28 free parameters
          "32*2 (simple)":      "32*2",      # 32 free parameters
      }

      for name, pattern in patterns.items():
          par_map, n_free, n_intervals = parse_pattern(pattern)
          print(f"{name}: {n_free} free parameters, {n_intervals} intervals")

          # In a full run:
          # results = psmc_inference(seq, n=63, pattern=pattern, n_iters=25)
          # final_lambdas = results[-1]['lambdas']

   **Expected results and interpretation:**

   - **"64*1" (all free):** With 64 independent :math:`\lambda` parameters, the
     inferred curve will show the bottleneck but will also have noisy spikes and
     dips, especially in the recent and ancient past where few recombination
     events provide signal. This is **overfitting**: the model has enough
     flexibility to fit noise in the data. The symptom is a jagged curve that
     changes dramatically between bootstrap replicates.

   - **"4+25*2+4+6" (default):** With 28 free parameters, the recent and ancient
     intervals are grouped (reducing noise), while the intermediate past retains
     fine resolution. The bottleneck should be clearly visible as a smooth dip.
     This is the **sweet spot** chosen by Li and Durbin for human data.

   - **"32*2" (simple):** With 32 free parameters, every pair of adjacent
     intervals shares one :math:`\lambda`. This provides moderate resolution
     everywhere but may be **too smooth** in the intermediate past, slightly
     blurring the edges of the bottleneck. It will miss fine temporal structure
     but will be more robust to noise than "64*1".

   The general rule: check :math:`C_\sigma \sigma_k` for each parameter group.
   If the expected number of recombination events per group is below ~20, the
   group should be made wider (merged with neighbors) to avoid overfitting.
