.. _psmc_continuous:

================================
The Continuous-Time PSMC Model
================================

   *The escapement: the mathematical heartbeat that makes the whole mechanism tick.*

This chapter derives the continuous-time foundations of PSMC. We start from the
simplest case (constant population size) and generalize to variable :math:`N(t)`.
By the end, you'll have derived the transition density, the stationary distribution,
and the recombination probability -- all from first principles.

The transition density is the tick-to-tick mechanism of the PSMC watch -- it
describes how the coalescence time at one genomic position relates to the
coalescence time at the next. The stationary distribution is the long-run
equilibrium where the watch runs steadily. And variable population size is like
a watch whose mainspring tension changes over time, altering the rhythm of the
ticks without breaking the mechanism.

If you've read the :ref:`SMC prerequisite <smc>`, you've already seen the
constant-population version of the PSMC transition density. Here we generalize
it to handle the case PSMC actually cares about: **variable population size**.


Step 1: Coalescence Under Variable Population Size
====================================================

Recall from the :ref:`coalescent theory prerequisite <coalescent_theory>`:
under constant population size :math:`N`, two lineages coalesce at rate 1 (in
units of :math:`2N` generations). The waiting time is :math:`\text{Exp}(1)`.

When the population size varies over time as :math:`N(t) = N_0 \lambda(t)`, the
coalescence rate at time :math:`t` is no longer constant -- it's
:math:`1/\lambda(t)`. A larger population means a lower coalescence rate (harder
to find a common ancestor when there are more individuals), and a smaller
population means a higher rate (fewer individuals, so lineages are forced together
more quickly).

**The survival probability.** The probability that two lineages have *not*
coalesced by time :math:`t` is the product of surviving through each
infinitesimal interval. With a time-varying rate :math:`1/\lambda(t)`, the
probability of surviving each tiny interval :math:`[u, u + du]` is approximately
:math:`1 - du/\lambda(u) \approx e^{-du/\lambda(u)}`. Multiplying these
together over all intervals from 0 to :math:`t` -- and recalling that
multiplying exponentials is the same as adding their exponents -- gives:

.. math::

   P(T > t) = \exp\left(-\int_0^t \frac{du}{\lambda(u)}\right)

This formula belongs to a branch of probability called **survival analysis**,
which studies the time until an event occurs (originally developed for
analyzing lifetimes in medical studies, but applicable whenever you ask "how long
until something happens?").

Let's define a shorthand for the cumulative integral that appears everywhere:

.. math::

   \Lambda(t) = \int_0^t \frac{du}{\lambda(u)}

So :math:`P(T > t) = e^{-\Lambda(t)}`. The function :math:`\Lambda(t)` is
called the **cumulative hazard** of the coalescent process under variable
population size.

.. admonition:: Probability aside: the hazard function and survival analysis

   **Survival analysis** is the study of how long we must wait for an event to
   occur. It was originally developed to study patient lifetimes, but the
   mathematics applies to any waiting time -- including coalescence.

   The central objects are:

   - The **survival function** :math:`S(t) = P(T > t)`: the probability that
     the event has *not yet* happened by time :math:`t`.
   - The **hazard function** (or hazard rate) :math:`h(t)`: the instantaneous
     rate at which the event occurs, given that it has not yet occurred. You can
     think of it as "how dangerous is this moment?" -- at each instant, :math:`h(t)`
     measures the intensity of the risk. Formally,
     :math:`h(t) = -S'(t)/S(t)`.
   - The **cumulative hazard** :math:`H(t) = \int_0^t h(u)\,du`: the total
     accumulated risk up to time :math:`t`. The survival function and
     cumulative hazard are linked by :math:`S(t) = e^{-H(t)}`.

   In our coalescent setting, the hazard rate is :math:`h(t) = 1/\lambda(t)`
   (the instantaneous coalescence rate), and the cumulative hazard is
   :math:`\Lambda(t)`. Everything that follows is an application of the
   identity :math:`S(t) = e^{-H(t)}` -- the survival probability is the
   exponential of the negative cumulative hazard.

.. admonition:: Computing :math:`\Lambda(t)` in practice

   For piecewise-constant :math:`\lambda(t)` (which is the PSMC assumption),
   the integral decomposes into a sum over intervals:

   .. math::

      \Lambda(t) = \sum_{i=0}^{k-1} \frac{\tau_i}{\lambda_i} + \frac{t - t_k}{\lambda_k}
      \quad \text{when } t \in [t_k, t_{k+1})

   where :math:`\tau_i = t_{i+1} - t_i`. No numerical quadrature is needed --
   it's a running sum. In the continuous case (smooth :math:`\lambda(t)`),
   we use numerical integration such as ``scipy.integrate.quad``.

.. code-block:: python

   from scipy.integrate import quad

   def cumulative_hazard(t, lambda_func):
       """Compute Lambda(t) = integral_0^t 1/lambda(u) du.

       This is the cumulative hazard of the coalescent process.

       We use scipy.integrate.quad, which performs adaptive numerical
       integration (also called "quadrature"). It approximates the
       integral by evaluating the function at many points and fitting
       polynomial segments. The function returns two values: the
       estimated integral and an estimate of the numerical error.
       """
       # quad returns (result, error_estimate).
       # The underscore _ is a Python convention meaning "I don't need
       # this value" -- here we discard the error estimate.
       result, _ = quad(
           # "lambda u: ..." is a Python lambda function -- a compact
           # way to define a small, unnamed function inline.
           # This one takes u as input and returns 1/lambda_func(u).
           lambda u: 1.0 / lambda_func(u),
           0,  # lower limit of integration
           t   # upper limit of integration
       )
       return result

   def cumulative_hazard_piecewise(t, t_boundaries, lambdas):
       """Fast Lambda(t) for piecewise-constant lambda.

       No quadrature needed -- just a running sum over intervals.
       For each interval [t_k, t_{k+1}) with population size lambda_k,
       the contribution to the integral is (interval width) / lambda_k.
       """
       Lambda = 0.0
       for k in range(len(lambdas)):
           t_lo = t_boundaries[k]
           t_hi = t_boundaries[k + 1]
           if t <= t_lo:
               break
           dt = min(t, t_hi) - t_lo
           Lambda += dt / lambdas[k]
           if t <= t_hi:
               break
       return Lambda

   # Verify: for constant lambda=1, Lambda(t) should equal t,
   # because integral_0^t 1/1 du = t.
   t_test = 2.5
   print(f"Lambda({t_test}) with lambda=1: "
         f"{cumulative_hazard(t_test, lambda u: 1.0):.6f} (expected: {t_test})")
   # For lambda=2, Lambda(t) = t/2 (half the rate, twice as slow).
   print(f"Lambda({t_test}) with lambda=2: "
         f"{cumulative_hazard(t_test, lambda u: 2.0):.6f} (expected: {t_test/2})")

**The density.** To obtain the probability density of coalescence at *exactly*
time :math:`t`, we differentiate the survival function. The density
:math:`f(t) = -\frac{d}{dt} S(t)` tells us "how much probability mass is
concentrated at time :math:`t`":

.. math::

   f(t) = -\frac{d}{dt} e^{-\Lambda(t)}

By the chain rule (the derivative of :math:`f(g(x))` is :math:`f'(g(x)) \cdot g'(x)`),
with the outer function :math:`e^{-(\cdot)}` and the inner function :math:`\Lambda(t)`:

.. math::

   f(t) = -\left(-\Lambda'(t)\right) e^{-\Lambda(t)} = \Lambda'(t) \cdot e^{-\Lambda(t)}

By the fundamental theorem of calculus, the derivative of
:math:`\Lambda(t) = \int_0^t \frac{du}{\lambda(u)}` with respect to :math:`t`
is simply the integrand evaluated at :math:`t`: :math:`\Lambda'(t) = 1/\lambda(t)`.
Therefore:

.. math::

   f(t) = \frac{1}{\lambda(t)} e^{-\Lambda(t)}

**Intuition:** The density at time :math:`t` is the probability of having
survived to :math:`t` (the :math:`e^{-\Lambda(t)}` term) times the instantaneous
coalescence rate at :math:`t` (the :math:`1/\lambda(t)` term). Same structure as
the constant-population case, just with a time-varying rate.

.. admonition:: Calculus aside: from CDF to density, step by step

   The survival function is :math:`S(t) = P(T > t) = e^{-\Lambda(t)}` and
   the CDF is :math:`F(t) = 1 - S(t)`. The density is obtained by
   differentiating:

   .. math::

      f(t) = F'(t) = -S'(t)
      = -\frac{d}{dt} e^{-\Lambda(t)}

   To differentiate :math:`e^{-\Lambda(t)}`, apply the chain rule: the
   derivative of :math:`e^{g(t)}` is :math:`e^{g(t)} \cdot g'(t)`, where
   :math:`g(t) = -\Lambda(t)`. So :math:`g'(t) = -\Lambda'(t) = -1/\lambda(t)`:

   .. math::

      f(t) = -\left[e^{-\Lambda(t)} \cdot \left(-\frac{1}{\lambda(t)}\right)\right]
      = \frac{1}{\lambda(t)} e^{-\Lambda(t)}

   For constant :math:`\lambda(t) = 1`, we have :math:`\Lambda(t) = t`, so
   :math:`f(t) = e^{-t}` -- the standard exponential density, which is the
   familiar coalescence time distribution for two lineages in a
   constant-size population (as derived in :ref:`coalescent_theory`).

.. code-block:: python

   import numpy as np
   from scipy.integrate import quad

   def coalescent_density(t, lambda_func):
       """Coalescence time density under variable population size.

       Parameters
       ----------
       t : float
           Time.
       lambda_func : callable
           lambda_func(u) returns relative population size at time u.

       Returns
       -------
       f : float
           Density f(t).
       """
       # Compute Lambda(t) = integral_0^t 1/lambda(u) du
       # using numerical integration (quad).
       # The "lambda u:" defines a small inline function that takes u
       # and returns 1/lambda_func(u).
       # quad returns (integral_value, error_estimate); we use _ to
       # discard the error estimate since we only need the integral.
       Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
       return (1.0 / lambda_func(t)) * np.exp(-Lambda_t)

   def coalescent_survival(t, lambda_func):
       """Probability that coalescence has NOT occurred by time t."""
       Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
       return np.exp(-Lambda_t)

   # Example: constant population (lambda = 1)
   t_vals = np.linspace(0.01, 5, 50)
   for lam_val, label in [(1.0, "constant N"), (2.0, "2x larger N")]:
       # "lambda u: lam_val" creates a function that always returns lam_val,
       # regardless of what u is -- i.e., a constant population size.
       densities = [coalescent_density(t, lambda u: lam_val) for t in t_vals]
       mean_t = sum(t * d for t, d in zip(t_vals, densities)) * (t_vals[1] - t_vals[0])
       print(f"{label}: mean coalescence time ~ {mean_t:.2f} "
             f"(expected: {lam_val:.2f})")


Step 2: The Transition Density Under Variable N(t)
=====================================================

Now we arrive at the central equation of PSMC -- the transition density. This
is the tick-to-tick mechanism of the PSMC watch: when a recombination occurs
between adjacent genomic positions :math:`a` and :math:`a+1`, the coalescence
time changes from :math:`s` (at position :math:`a`) to :math:`t` (at position
:math:`a+1`). The transition density :math:`q(t \mid s)` tells us the
probability density of the new time :math:`t`, given that the old time was
:math:`s`.

We already derived this for constant :math:`\lambda` in the :ref:`SMC prerequisite <smc>`.
Here's the generalization to variable :math:`\lambda(t)`.

**The physical process**, step by step:

1. At position :math:`a`, the two lineages coalesce at time :math:`s`.

2. A recombination occurs somewhere on the branch, at time :math:`u`. Under the
   SMC model (see :ref:`smc`), :math:`u` is uniform on :math:`[0, s]`:

   .. math::

      P_1(u \mid s) = \frac{1}{s}

3. From time :math:`u`, the detached lineage floats up and re-coalesces. Under
   variable population size, the re-coalescence density is:

   .. math::

      P_2(t \mid u) = \frac{1}{\lambda(t)} \exp\left(-\int_u^t \frac{dv}{\lambda(v)}\right)

   This is just the coalescence density starting from time :math:`u` instead of 0.
   The exponential term is the probability of surviving from :math:`u` to :math:`t`
   without coalescing, and the :math:`1/\lambda(t)` factor is the instantaneous
   rate of coalescing right at time :math:`t`.

4. The total transition density (conditioned on recombination) is obtained by
   **marginalizing over** :math:`u` -- that is, summing up the contributions
   from all possible recombination times :math:`u`. Physically, this means we
   account for every possible place along the branch where the recombination
   could have struck: near the tips (small :math:`u`), near the coalescence
   point (large :math:`u`), or anywhere in between. Each location gives a
   different "starting point" for the re-coalescence process, and we weight
   them all equally (since :math:`u` is uniform on :math:`[0, s]`) and add
   them up:

.. math::

   q(t \mid s) = \int_0^{\min(s,t)} P_2(t \mid u) \cdot P_1(u \mid s) \, du
   = \frac{1}{\lambda(t)} \int_0^{\min(s,t)} \frac{1}{s} \cdot e^{-\int_u^t \frac{dv}{\lambda(v)}} \, du

This is Equation 5 from Li and Durbin's paper, and Theorem 1 in ``psmc.tex``.

**Why** :math:`\min(s, t)` **?** The recombination time :math:`u` must satisfy
both :math:`u \leq s` (it's on the branch of length :math:`s`) and :math:`u \leq t`
(re-coalescence at :math:`t` must come after recombination at :math:`u`).

.. admonition:: Computing the double integral step by step

   The transition density involves an outer integral over the recombination
   point :math:`u` and an inner integral (inside the exponential) for the
   cumulative hazard from :math:`u` to :math:`t`:

   .. math::

      q(t \mid s) = \frac{1}{\lambda(t)} \int_0^{\min(s,t)}
         \underbrace{\frac{1}{s}}_{\text{uniform recomb.}}
         \cdot \exp\!\left(\underbrace{-\int_u^t \frac{dv}{\lambda(v)}}_{\text{survival from } u \text{ to } t}\right) du

   **For piecewise-constant** :math:`\lambda`: the inner integral
   :math:`\int_u^t 1/\lambda(v)\,dv` again reduces to a sum over intervals.
   This means the exponential term can be expressed as a product of per-interval
   survival factors, avoiding expensive quadrature in the inner loop. The
   outer integral over :math:`u` then splits into sub-integrals within each
   time interval, each of which has a closed-form solution (integrals of the
   form :math:`\int e^{-au}\,du = -\frac{1}{a} e^{-au}`). This is how the
   exact discrete formulas in the next chapter are derived.

   **For general smooth** :math:`\lambda`: we need nested numerical
   quadrature (``quad`` inside ``quad``), which is slower but exact to
   machine precision.

.. admonition:: Probability aside: the law of total probability

   The transition density is a direct application of the **law of total
   probability**. Conditioning on the (unobserved) recombination time :math:`u`:

   .. math::

      q(t \mid s) = \int_0^{\min(s,t)} P_2(t \mid u) \, P_1(u \mid s) \, du

   Each factor has a clear probabilistic interpretation:
   :math:`P_1(u \mid s) = 1/s` is the prior on the recombination breakpoint
   (uniform on the branch), and :math:`P_2(t \mid u)` is the re-coalescence
   density (a coalescent started at time :math:`u`). We **marginalize** over
   all possible :math:`u` to get the marginal density of the new coalescence
   time. "Marginalizing" means integrating out a variable we cannot observe
   -- in this case, we do not know exactly where on the branch the
   recombination struck, so we average over all possibilities, weighted by
   their probability.

.. code-block:: python

   def psmc_transition_density_general(t, s, lambda_func):
       """PSMC transition density q(t|s) under variable population size.

       This is the probability density of the new coalescence time being t,
       given that the old coalescence time was s and a recombination occurred.

       Parameters
       ----------
       t : float
           New coalescence time.
       s : float
           Previous coalescence time.
       lambda_func : callable
           Relative population size function.

       Returns
       -------
       q : float
       """
       upper = min(s, t)

       def integrand(u):
           # For each candidate recombination time u, compute:
           #   (1/s) * exp(-integral_u^t 1/lambda(v) dv)
           # The inner quad call computes the cumulative hazard from u to t.
           # Again, _ discards the error estimate from quad.
           integral, _ = quad(lambda v: 1.0 / lambda_func(v), u, t)
           return (1.0 / s) * np.exp(-integral)

       # The outer quad call integrates over all recombination times u
       # from 0 to min(s, t). result is the integral value; _ discards
       # the error estimate.
       result, _ = quad(integrand, 0, upper)
       return result / lambda_func(t)

   # Verify: for constant lambda, this should match the closed-form
   s = 1.0
   t_vals = np.linspace(0.01, 3.0, 30)

   print("Comparing general formula to closed-form (constant lambda=1):")
   for t in [0.3, 0.7, 1.0, 1.5, 2.0]:
       # "lambda u: 1.0" is a constant function returning 1.0 for any u.
       q_general = psmc_transition_density_general(t, s, lambda u: 1.0)
       # Closed-form for constant lambda=1:
       if t < s:
           q_closed = (1.0 / s) * (1 - np.exp(-t))
       else:
           q_closed = (1.0 / s) * (np.exp(-(t - s)) - np.exp(-t))
       print(f"  t={t:.1f}: general={q_general:.6f}, "
             f"closed={q_closed:.6f}, diff={abs(q_general-q_closed):.2e}")


Step 3: Normalization -- It Integrates to 1
==============================================

**Where we are so far.** In Steps 1 and 2, we derived the coalescence density
under variable population size (the probability of coalescing at a specific time)
and the transition density (the probability of transitioning to a new coalescence
time after a recombination event). Now we must verify a crucial mathematical
property: that the transition density :math:`q(t \mid s)` is a valid probability
density.

For a function to be a valid probability density, it must integrate to 1 over its
entire domain. This means that if we add up the probabilities of all possible
outcomes (here, all possible new coalescence times :math:`t` from 0 to infinity),
the total must be exactly 1 -- because *something* has to happen, and the total
probability of all possibilities is always 100%. If the integral were less than 1,
probability mass would be "leaking" somewhere; if it were greater than 1, we would
be double-counting. Either would make the HMM built on top of this density
mathematically inconsistent.

A crucial property: :math:`q(t \mid s)` integrates to 1 over :math:`t \in [0, \infty)`.
This is not obvious! Let's prove it, because understanding the proof reveals the
structure of the formula.

The trick is a change of variables. Define :math:`\tilde{t}` such that:

.. math::

   \phi'(\tilde{u}) \cdot \frac{1}{\lambda(\phi(\tilde{u}))} = 1, \qquad \phi(0) = 0

In other words, :math:`\tilde{t} = \Lambda(t)` is the "coalescent time measured in
units of constant population size." This is a time-warping transformation: it
stretches intervals where the population is large (slow coalescence) and compresses
intervals where the population is small (fast coalescence). Under this
transformation, the coalescence rate becomes constant (rate 1), and the integral
reduces to the constant-population case, which we already know integrates to 1
from the :ref:`SMC prerequisite <smc>`.

**Why is this important?** It means :math:`q(t|s)` is a valid probability density
-- essential for the HMM to be well-defined. If the transition density did not
integrate to 1, the forward algorithm (which multiplies transition probabilities
at each step) would produce values that drift away from true probabilities,
making all downstream inference meaningless.

.. code-block:: python

   # Numerical verification: q(t|s) integrates to 1
   def verify_normalization(s, lambda_func, t_max=20):
       """Verify that q(t|s) integrates to 1.

       Uses quad to numerically integrate the transition density over
       t from a small positive number (to avoid division by zero at t=0)
       up to t_max (a stand-in for infinity).
       """
       # quad with limit=100 allows up to 100 subdivisions for
       # adaptive integration on difficult integrands.
       result, error = quad(
           # lambda t: ... creates an inline function of t that calls
           # our transition density function.
           lambda t: psmc_transition_density_general(t, s, lambda_func),
           0.001, t_max,
           limit=100
       )
       return result

   # Test with various population size functions
   test_cases = [
       ("Constant N", lambda t: 1.0),
       ("Doubled N", lambda t: 2.0),
       ("Bottleneck", lambda t: 0.1 if 0.5 < t < 1.0 else 1.0),
       ("Growth", lambda t: np.exp(t / 2)),
   ]

   for name, lam in test_cases:
       for s in [0.5, 1.0, 2.0]:
           integral = verify_normalization(s, lam)
           print(f"  {name}, s={s:.1f}: integral = {integral:.6f}")


Step 4: The Stationary Distribution
======================================

The **stationary distribution** :math:`\pi(t)` is the probability density of
the coalescence time at a single position, marginalized over all possible histories.
It answers: "if I pick a random position on the genome, what is the distribution of
its coalescence time?"

Think of it as the long-run equilibrium where the watch runs steadily. If the PSMC
process has been running for a very long time along the genome, and we sample a
position at random, :math:`\pi(t)` tells us how likely we are to find coalescence
time :math:`t`. No matter what coalescence time the process started with, after
enough recombination events it settles into this steady-state distribution.

**Why does the stationary distribution matter for the HMM?** In a Hidden Markov
Model, we need to specify the probability distribution of the *first* hidden
state. The stationary distribution :math:`\pi(t)` serves as this **initial state
distribution**: we assume the first genomic position is drawn from the long-run
equilibrium. This is a natural choice because a random position in the genome
has no special relationship to the beginning of the sequence -- it is just
another position in a long chain of transitions. Using :math:`\pi(t)` as the
initial distribution is equivalent to assuming the process has been running
long enough to reach equilibrium before we start observing.

For the PSMC transition density with variable :math:`\lambda(t)`:

.. math::

   \pi(t) = \frac{t}{C_\pi \lambda(t)} e^{-\Lambda(t)}

where :math:`C_\pi` is the normalization constant:

.. math::

   C_\pi = \int_0^\infty e^{-\Lambda(u)} \, du

**Where does the** :math:`t` **in the numerator come from?** This is the key
insight. The transition :math:`q(t|s)` involves integrating the recombination
position :math:`u` uniformly on :math:`[0, s]`. In the stationary state, we
average over :math:`s` as well. The factor :math:`t` arises because a
recombination on a branch of length :math:`t` is :math:`t` times as likely as on a
branch of length 1 -- longer branches "catch" more recombinations.

More formally, :math:`\pi(t)` is the density such that:

.. math::

   \int_0^\infty q(t \mid s) \pi(s) \, ds = \pi(t)

This is a fixed-point equation: if the coalescence time distribution is
:math:`\pi`, then after one recombination event, the distribution is still
:math:`\pi`. The proof uses Lemma 2 from ``psmc.tex`` (the stationary
distribution lemma), with :math:`g(u) = 1` and :math:`h(u) = 1/\lambda(u)`.

.. admonition:: Probability aside: stationary distributions and Markov chains

   The equation :math:`\int q(t|s)\,\pi(s)\,ds = \pi(t)` is the
   continuous-state analogue of :math:`\boldsymbol{\pi}^T \mathbf{Q} = \boldsymbol{\pi}^T`
   for discrete Markov chains (as introduced in the :ref:`HMM prerequisite <hmms>`).
   The transition kernel :math:`q(t|s)` plays the role of the transition
   matrix, and the integral replaces matrix multiplication. A distribution
   satisfying this equation is an **invariant measure** (or equilibrium) of
   the chain: once the process reaches :math:`\pi`, it stays there forever.

   The extra factor of :math:`t` in the numerator of :math:`\pi(t)` compared
   to the coalescence density :math:`f(t)` reflects a **size-biased sampling**
   effect. Recombination is more likely to hit a longer branch (probability
   proportional to :math:`t`), so the stationary distribution over-represents
   long coalescence times relative to the raw density :math:`f(t)`. This is
   the same phenomenon as the "waiting time paradox" (or inspection paradox)
   in renewal theory: if you arrive at a random time, you are more likely to
   land in a long inter-arrival interval.

.. code-block:: python

   def stationary_distribution(t, lambda_func, C_pi=None):
       """Stationary distribution pi(t) of coalescence time.

       Parameters
       ----------
       t : float
           Time.
       lambda_func : callable
       C_pi : float, optional
           Normalization constant. Computed if not provided.

       Returns
       -------
       pi_t : float
       """
       if C_pi is None:
           C_pi = compute_C_pi(lambda_func)

       # Compute Lambda(t) via numerical integration.
       # _ discards the error estimate returned by quad.
       Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
       return t / (C_pi * lambda_func(t)) * np.exp(-Lambda_t)

   def compute_C_pi(lambda_func, t_max=20):
       """Compute the normalization constant C_pi.

       C_pi = integral_0^inf exp(-Lambda(u)) du

       We approximate the upper limit of infinity with t_max=20, which
       is safe because the integrand decays exponentially.
       """
       def integrand(u):
           # For each u, compute Lambda(u) and then exp(-Lambda(u)).
           Lambda_u, _ = quad(lambda v: 1.0 / lambda_func(v), 0, u)
           return np.exp(-Lambda_u)

       # Integrate the survival function from 0 to t_max.
       C_pi, _ = quad(integrand, 0, t_max)
       return C_pi

   # For constant population lambda=1:
   # pi(t) = t * exp(-t) / C_pi
   # C_pi = integral_0^inf exp(-t) dt = [-exp(-t)]_0^inf = 0 - (-1) = 1
   # So pi(t) = t * exp(-t) -- the Gamma(2,1) distribution!
   C_pi_const = compute_C_pi(lambda t: 1.0)
   print(f"C_pi (constant pop): {C_pi_const:.6f} (expected: 1.0)")

   # The mean coalescence time under constant population:
   # E[T] = integral_0^inf t * pi(t) dt = integral_0^inf t^2 * exp(-t) dt / C_pi
   # = Gamma(3) / C_pi = 2! / 1 = 2
   mean_T, _ = quad(
       # lambda t: ... creates a function that evaluates
       # t * pi(t), which we integrate to get the expected value.
       lambda t: t * stationary_distribution(t, lambda u: 1.0, C_pi_const),
       0, 20
   )
   print(f"Mean coalescence time (constant pop): {mean_T:.4f} (expected: 2.0)")

   # Verify pi(t) is indeed stationary: integral q(t|s)*pi(s)ds = pi(t)
   def verify_stationarity(t_test, lambda_func, C_pi, s_max=15):
       """Check that pi is the fixed point of q.

       If pi is truly stationary, then applying one step of the
       transition density and averaging over all starting states s
       (weighted by pi(s)) should give back pi(t).
       """
       lhs, _ = quad(
           lambda s: psmc_transition_density_general(t_test, s, lambda_func)
                     * stationary_distribution(s, lambda_func, C_pi),
           0.001, s_max)
       rhs = stationary_distribution(t_test, lambda_func, C_pi)
       return lhs, rhs

   for t_val in [0.5, 1.0, 2.0]:
       lhs, rhs = verify_stationarity(t_val, lambda u: 1.0, C_pi_const)
       print(f"t={t_val}: integral q*pi = {lhs:.6f}, pi(t) = {rhs:.6f}, "
             f"ratio = {lhs/rhs:.6f}")


.. admonition:: The Gamma(2,1) distribution

   For constant population size (:math:`\lambda(t) = 1`), the stationary
   distribution is :math:`\pi(t) = t e^{-t}`, which is the :math:`\text{Gamma}(2, 1)`
   distribution. Its mean is 2, matching the well-known result that the expected
   pairwise coalescence time is :math:`2N` generations (or 2 in coalescent units of
   :math:`2N_0`), as established in :ref:`coalescent_theory`.


Step 5: Including No-Recombination (The Full Transition)
==========================================================

**Where we are so far.** We have derived the coalescence density (Step 1), the
transition density conditioned on recombination (Step 2), proved it integrates
to 1 (Step 3), and found the stationary distribution (Step 4). All of these
describe what happens *when* a recombination occurs. But recombination does not
happen at every genomic position -- in fact, for most adjacent positions, no
recombination occurs at all. Now we need to account for both possibilities:
recombination or no recombination.

The density :math:`q(t|s)` describes what happens *given* a recombination. But at
each genomic position, recombination may or may not occur. The probability of
recombination on a branch of length :math:`s` is :math:`1 - e^{-\rho s}` (Poisson
process with rate :math:`\rho` per unit branch length per bin).

The **full transition** includes both possibilities:

.. math::

   p(t \mid s) = (1 - e^{-\rho s}) \, q(t \mid s) + e^{-\rho s} \, \delta(t - s)

- With probability :math:`1 - e^{-\rho s}`: recombination occurs, and the new time
  is drawn from :math:`q(t|s)`.
- With probability :math:`e^{-\rho s}`: no recombination, and the time stays at
  :math:`s` (represented by the Dirac delta :math:`\delta(t - s)`).

The **Dirac delta** :math:`\delta(t - s)` is a mathematical device that represents
a "spike" of probability concentrated at a single point :math:`t = s`. It is not
a function in the ordinary sense -- it is zero everywhere except at :math:`t = s`,
where it is infinitely tall, but its total integral is exactly 1. You can think of
it as the limit of a very narrow, very tall bell curve that shrinks to zero width
while keeping its total area equal to 1. When multiplied by a probability
:math:`e^{-\rho s}`, it means "with probability :math:`e^{-\rho s}`, the new
coalescence time is *exactly* :math:`s` -- not approximately :math:`s`, but
precisely :math:`s`."

The key property of the Dirac delta is its behavior under integration: for any
function :math:`f`,

.. math::

   \int_{-\infty}^{\infty} f(t) \, \delta(t - s) \, dt = f(s)

In other words, integrating against the Dirac delta "picks out" the value of
:math:`f` at the spike location.

.. admonition:: Probability aside: mixture distributions

   The full transition :math:`p(t|s)` is a **mixture** of a continuous density
   and a point mass. This is a common construction in probability:

   .. math::

      T' \sim \begin{cases} q(\cdot \mid s) & \text{with prob } 1 - e^{-\rho s} \\
      \delta_s & \text{with prob } e^{-\rho s} \end{cases}

   When computing expectations under :math:`p(t|s)`, the Dirac delta contributes
   the "no change" case: :math:`E[g(T')] = (1 - e^{-\rho s})\int g(t)\,q(t|s)\,dt + e^{-\rho s}\,g(s)`.

.. code-block:: python

   def full_transition_density(t, s, rho, lambda_func, tol=1e-8):
       """Full transition density p(t|s) including no-recombination.

       For the Dirac delta part (t == s), returns the point mass weight
       separately. For t != s, returns the continuous part.

       Returns
       -------
       continuous : float
           The continuous part of the density at t.
       point_mass : float
           The weight of the point mass at t = s (nonzero only when
           |t - s| < tol).
       """
       recomb_prob = 1.0 - np.exp(-rho * s)
       if abs(t - s) < tol:
           # Point mass contribution: e^{-rho*s} * delta(t-s)
           # Cannot return density at the point mass; return (continuous, point_mass)
           q_ts = psmc_transition_density_general(t, s, lambda_func)
           return recomb_prob * q_ts, np.exp(-rho * s)
       else:
           return recomb_prob * psmc_transition_density_general(t, s, lambda_func), 0.0

   # The probability of recombination depends on branch length s
   for s_val in [0.5, 1.0, 2.0, 5.0]:
       p_rec = 1 - np.exp(-0.001 * s_val)
       print(f"s={s_val}: P(recombination) = {p_rec:.6f}")

**The full stationary distribution** :math:`\sigma(t)` satisfies:

.. math::

   \sigma(t) = \frac{\pi(t)}{C_\sigma(1 - e^{-\rho t})}

where:

.. math::

   C_\sigma = \int_0^\infty \frac{\pi(t)}{1 - e^{-\rho t}} \, dt

**Why does** :math:`\sigma(t)` **differ from** :math:`\pi(t)` **?** The
distribution :math:`\pi(t)` is the stationary distribution of :math:`q(t|s)`,
the transition density *conditioned on recombination*. But at stationarity,
whether or not a recombination occurs also depends on the coalescence time
(through the factor :math:`1 - e^{-\rho s}`). Longer coalescence times have
higher recombination probability, so they "churn" more and their stationary weight
:math:`\sigma(t)` is increased relative to :math:`\pi(t)` by the factor
:math:`1/(1 - e^{-\rho t})`.

.. admonition:: Probability aside: detailed balance and reweighting

   The relationship between :math:`\sigma` and :math:`\pi` is an example of
   **importance reweighting**. The full chain (with skip-probability
   :math:`e^{-\rho s}`) has a different stationary distribution than the
   sub-chain restricted to steps where recombination occurs. The reweighting
   factor :math:`1/(1 - e^{-\rho t})` compensates for the fact that states
   with short coalescence times are "stickier" (less likely to recombine),
   so they need higher stationary weight to account for all the silent
   no-recombination steps.

.. code-block:: python

   def full_stationary(t, lambda_func, rho, C_pi=None, C_sigma=None):
       """Full stationary distribution sigma(t).

       Parameters
       ----------
       t : float
       lambda_func : callable
       rho : float
           Recombination rate per bin.
       C_pi, C_sigma : float, optional
       """
       if C_pi is None:
           C_pi = compute_C_pi(lambda_func)
       pi_t = stationary_distribution(t, lambda_func, C_pi)

       if C_sigma is None:
           C_sigma = compute_C_sigma(lambda_func, rho, C_pi)

       return pi_t / (C_sigma * (1 - np.exp(-rho * t)))

   def compute_C_sigma(lambda_func, rho, C_pi=None, t_max=20):
       """Compute C_sigma = integral pi(t) / (1 - exp(-rho*t)) dt.

       The lower limit is slightly above 0 (1e-6) to avoid division
       by zero, since 1 - exp(-rho*t) -> 0 as t -> 0.
       """
       if C_pi is None:
           C_pi = compute_C_pi(lambda_func)

       def integrand(t):
           return stationary_distribution(t, lambda_func, C_pi) / (1 - np.exp(-rho * t))

       # _ discards the error estimate from quad.
       C_sigma, _ = quad(integrand, 1e-6, t_max)
       return C_sigma

   # Test: the probability of recombination at any site is 1/C_sigma
   rho = 0.001
   C_pi = compute_C_pi(lambda t: 1.0)
   C_sigma = compute_C_sigma(lambda t: 1.0, rho, C_pi)
   print(f"C_sigma = {C_sigma:.4f}")
   print(f"P(recombination) = 1/C_sigma = {1.0/C_sigma:.6f}")


Step 6: Approximating C_sigma
================================

For small :math:`\rho` (which is the typical regime -- recombination rates are
very low per base pair), :math:`C_\sigma` has a clean approximation:

.. math::

   C_\sigma = \frac{1}{C_\pi \rho} + \frac{1}{2} + o(\rho)

The notation :math:`o(\rho)` means "terms that go to zero faster than :math:`\rho`
as :math:`\rho \to 0`" -- in other words, for small :math:`\rho`, these terms are
negligible.

**Derivation.** Starting from the definition and using a Taylor expansion to
approximate :math:`1/(1 - e^{-\rho t})`:

.. math::

   C_\sigma &= \int_0^\infty \frac{\pi(t)}{1 - e^{-\rho t}} \, dt \\
   &= \int_0^\infty \frac{t}{C_\pi \lambda(t)(1 - e^{-\rho t})} e^{-\Lambda(t)} \, dt

A **Taylor expansion** (also called a Taylor series) approximates a function near
a point by a polynomial. The idea is that any smooth function can be written as a
sum of powers of its argument. For :math:`e^{-x} \approx 1 - x + x^2/2 - \cdots`
when :math:`x` is small, we get :math:`1 - e^{-\rho t} \approx \rho t - (\rho t)^2/2 + \cdots \approx \rho t`
for small :math:`\rho t`. Taking the reciprocal:

.. math::

   \frac{1}{1 - e^{-\rho t}} \approx \frac{1}{\rho t}\left(1 + \frac{\rho t}{2} + \cdots\right)

This approximation works because :math:`1 - e^{-x} \approx x` for small :math:`x`,
and we are then expanding :math:`1/(x - x^2/2 + \cdots)` using the geometric
series :math:`1/(1 - r) \approx 1 + r + \cdots`.

Substituting into the integral for :math:`C_\sigma`:

.. math::

   C_\sigma &\approx \frac{1}{C_\pi \rho} \int_0^\infty \frac{1}{\lambda(t)} e^{-\Lambda(t)} \, dt
   + \frac{1}{2} \int_0^\infty \pi(t) \, dt + o(\rho)

Now we use two facts:

- :math:`\int_0^\infty \frac{1}{\lambda(t)} e^{-\Lambda(t)} dt = 1`: this is the
  integral of the coalescence density :math:`f(t)` over all time, which must be 1
  because coalescing at *some* time is certain.
- :math:`\int_0^\infty \pi(t) dt = 1`: this is the normalization of the stationary
  distribution (it is a probability density, so it integrates to 1).

Therefore:

.. math::

   C_\sigma &= \frac{1}{C_\pi \rho} \cdot 1 + \frac{1}{2} \cdot 1 + o(\rho) \\
   &= \frac{1}{C_\pi \rho} + \frac{1}{2} + o(\rho)

.. code-block:: python

   # Verify the approximation C_sigma ~ 1/(C_pi * rho) + 0.5
   C_pi = compute_C_pi(lambda t: 1.0)
   for rho in [0.01, 0.001, 0.0001]:
       C_sigma_exact = compute_C_sigma(lambda t: 1.0, rho, C_pi)
       C_sigma_approx = 1.0 / (C_pi * rho) + 0.5
       print(f"rho={rho:.4f}: exact={C_sigma_exact:.4f}, "
             f"approx={C_sigma_approx:.4f}, "
             f"relative error={abs(C_sigma_exact-C_sigma_approx)/C_sigma_exact:.2e}")


Step 7: The Rate of Pairwise Difference
=========================================

**Where we are so far.** We have built the complete transition machinery of the
PSMC watch: the transition density (how coalescence times change from position
to position), the stationary distribution (the long-run equilibrium), and the
full transition including no-recombination events. Now we need the final piece:
the **emission model** -- how the hidden coalescence time produces the observed
data (heterozygous or homozygous sites).

How many heterozygous sites do we expect? If the coalescence time is :math:`t`, the
probability of at least one mutation in a bin is :math:`1 - e^{-\theta t}` (two
lineages, each of length :math:`t/2` ... wait, let's be careful.

Actually, in coalescent units where :math:`t` measures the **total** time of the
pairwise coalescent (time from present to MRCA), the expected number of mutations
on the two branches is :math:`\theta t` (both lineages have length :math:`t`, but
the mutation rate :math:`\theta` is already scaled to account for both). So the
probability of at least one mutation (heterozygous site) in a bin is:

.. math::

   P(X_a = 1 \mid T_a = t) = 1 - e^{-\theta t}

.. admonition:: Probability aside: the Poisson model of mutations

   Why :math:`1 - e^{-\theta t}` and not just :math:`\theta t`?  Mutations
   along the two lineages of total length :math:`t` follow a Poisson process
   with rate :math:`\theta`. The probability of **at least one** mutation (i.e.,
   a heterozygous site) is :math:`1 - P(\text{zero mutations}) = 1 - e^{-\theta t}`.
   For small :math:`\theta t` this is approximately :math:`\theta t` (first-order
   Taylor expansion: :math:`e^{-x} \approx 1 - x` when :math:`x \ll 1`), but the
   exponential form is exact and avoids overcounting when :math:`\theta t` is not
   small.

   This is the same Poisson mutation model introduced in
   :ref:`coalescent_theory`, where we showed that mutations on a branch of
   length :math:`\ell` follow a Poisson distribution with mean
   :math:`\theta\ell/2`. Here, the total branch length for two lineages
   coalescing at time :math:`t` is :math:`2 \times t = 2t` (each lineage has
   length :math:`t`), giving an expected mutation count of
   :math:`\theta/2 \times 2t = \theta t`. The probability of zero mutations
   is :math:`e^{-\theta t}`, so the probability of at least one is
   :math:`1 - e^{-\theta t}`.

.. code-block:: python

   # Emission probability as a function of coalescence time
   theta = 0.001
   for t_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
       p_het = 1 - np.exp(-theta * t_val)
       # Linear approximation: e^{-x} ~ 1 - x, so 1 - e^{-x} ~ x
       p_het_approx = theta * t_val
       print(f"t={t_val:5.1f}: P(het) = {p_het:.6f}, "
             f"linear approx = {p_het_approx:.6f}, "
             f"error = {abs(p_het - p_het_approx):.2e}")

Averaging over the stationary distribution of :math:`T`:

.. math::

   P(X_a = 1) = \int_0^\infty (1 - e^{-\theta t}) \sigma(t) \, dt

This integral computes the **expected heterozygosity**: we weight the per-site
mutation probability by the probability that a random genomic site has
coalescence time :math:`t`, then integrate over all possible times.

For small :math:`\theta` and :math:`\rho`, this simplifies to:

.. math::

   P(X_a = 1) \approx C_\pi \theta \cdot [1 + o(\rho + \theta)]

**Intuition:** The expected number of heterozygous sites is proportional to
:math:`\theta` (mutation rate) and :math:`C_\pi` (which is the "effective
coalescence time" -- it captures how population size history affects the average
time to MRCA). Larger :math:`C_\pi` means longer average coalescence times, hence
more heterozygosity.

This gives us a way to **estimate** :math:`\theta` from the observed fraction of
heterozygous sites:

.. math::

   \hat{\theta} \approx \frac{\text{fraction of het sites}}{C_\pi}

.. code-block:: python

   # For constant population: C_pi = 1, so P(het) ~ theta
   theta = 0.001  # typical for humans at 100bp bins
   C_pi = 1.0
   p_het_approx = C_pi * theta
   p_het_exact = 1 - np.exp(-theta * 1.0)  # for T = 1 (mean under Exp(1))
   print(f"Approximate P(het): {p_het_approx:.6f}")
   print(f"Using mean T: {p_het_exact:.6f}")

   # Initial estimate of theta from data
   def estimate_theta_initial(seq):
       """Estimate theta_0 from the observed fraction of het sites.

       Uses the exact inversion: if P(het) = 1 - exp(-theta), then
       theta = -log(1 - P(het)).
       """
       frac_het = np.mean(seq)
       # -log(1 - frac_het) is the exact solution of 1 - exp(-theta) = frac_het
       return -np.log(1 - frac_het)

   # Test on simulated data
   theta_hat = estimate_theta_initial(seq)
   print(f"\nTrue theta: {0.001}, Estimated: {theta_hat:.6f}")


Putting It All Together
========================

Here's a summary of the continuous-time PSMC model -- all seven steps in one table:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Quantity
     - Formula
   * - Coalescence rate
     - :math:`1/\lambda(t)`
   * - Cumulative hazard
     - :math:`\Lambda(t) = \int_0^t \frac{du}{\lambda(u)}`
   * - Coalescence density
     - :math:`f(t) = \frac{1}{\lambda(t)} e^{-\Lambda(t)}`
   * - Transition density (given recomb.)
     - :math:`q(t|s) = \frac{1}{\lambda(t)} \int_0^{\min(s,t)} \frac{1}{s} e^{-\int_u^t \frac{dv}{\lambda(v)}} du`
   * - Full transition
     - :math:`p(t|s) = (1 - e^{-\rho s}) q(t|s) + e^{-\rho s} \delta(t-s)`
   * - Stationary dist. (recomb.)
     - :math:`\pi(t) = \frac{t}{C_\pi \lambda(t)} e^{-\Lambda(t)}`
   * - Stationary dist. (full)
     - :math:`\sigma(t) = \frac{\pi(t)}{C_\sigma(1 - e^{-\rho t})}`
   * - Normalization
     - :math:`C_\pi = \int_0^\infty e^{-\Lambda(u)} du`
   * - Approx. normalization
     - :math:`C_\sigma \approx \frac{1}{C_\pi \rho} + \frac{1}{2}`
   * - Emission probability
     - :math:`P(X_a = 1 \mid T_a = t) = 1 - e^{-\theta t}`

These equations form the mathematical foundation. But they're continuous -- and
an HMM needs discrete states. In the next chapter, we discretize time.


Exercises
=========

.. admonition:: Exercise 1: Verify the stationary property

   Numerically verify that :math:`\int_0^\infty q(t|s) \pi(s) ds = \pi(t)` for
   several values of :math:`t` and several population size functions (constant,
   exponential growth, bottleneck).

.. admonition:: Exercise 2: Explore the effect of population size on :math:`\pi(t)`

   Plot :math:`\pi(t)` for:
   (a) constant :math:`\lambda(t) = 1`,
   (b) a bottleneck :math:`\lambda(t) = 0.1` for :math:`t \in [1, 2]` and 1 elsewhere,
   (c) exponential growth :math:`\lambda(t) = e^{t}`.

   How does each demographic event shift the distribution of coalescence times?

.. admonition:: Exercise 3: Estimate :math:`C_\pi` for a bottleneck model

   For :math:`\lambda(t) = 1` for :math:`t < 1`, :math:`\lambda(t) = 0.1` for
   :math:`t \in [1, 2]`, and :math:`\lambda(t) = 1` for :math:`t > 2`:

   (a) Compute :math:`C_\pi` numerically.
   (b) What does a smaller :math:`C_\pi` mean for the expected heterozygosity?
   (c) How would you detect this bottleneck from the data?

Next: :ref:`psmc_discretization` -- turning the continuous model into an HMM.


Solutions
=========

.. admonition:: Solution 1: Verify the stationary property

   We numerically verify that :math:`\int_0^\infty q(t|s) \pi(s) \, ds = \pi(t)` for
   several values of :math:`t` and several population size functions. The key insight is
   that if :math:`\pi(t)` is truly stationary, applying one transition step and averaging
   over all starting states should reproduce :math:`\pi(t)` exactly.

   .. code-block:: python

      from scipy.integrate import quad

      def verify_stationarity_full(t_test, lambda_func, C_pi, s_max=15):
          """Check that pi is the fixed point of q for a given t."""
          lhs, _ = quad(
              lambda s: psmc_transition_density_general(t_test, s, lambda_func)
                        * stationary_distribution(s, lambda_func, C_pi),
              0.001, s_max, limit=100)
          rhs = stationary_distribution(t_test, lambda_func, C_pi)
          return lhs, rhs

      # Test with three demographic scenarios
      scenarios = [
          ("Constant (lambda=1)", lambda t: 1.0),
          ("Exponential growth (lambda=e^t)", lambda t: np.exp(t)),
          ("Bottleneck (lambda=0.1 for t in [1,2])",
           lambda t: 0.1 if 1.0 < t < 2.0 else 1.0),
      ]

      t_values = [0.5, 1.0, 2.0, 3.0]

      for name, lam_func in scenarios:
          C_pi = compute_C_pi(lam_func)
          print(f"\n{name} (C_pi = {C_pi:.4f}):")
          for t_val in t_values:
              lhs, rhs = verify_stationarity_full(t_val, lam_func, C_pi)
              ratio = lhs / rhs if rhs > 0 else float('nan')
              print(f"  t={t_val:.1f}: integral q*pi = {lhs:.6f}, "
                    f"pi(t) = {rhs:.6f}, ratio = {ratio:.6f}")

   For all scenarios and all :math:`t` values, the ratio should be very close to 1.0
   (within numerical integration tolerance, typically :math:`\sim 10^{-6}`). This
   confirms that :math:`\pi(t)` satisfies the fixed-point equation
   :math:`\int q(t|s)\pi(s)\,ds = \pi(t)` regardless of the population size history.

.. admonition:: Solution 2: Explore the effect of population size on :math:`\pi(t)`

   We compute and compare :math:`\pi(t)` for three demographic scenarios. The key insight
   is that :math:`\pi(t) = \frac{t}{C_\pi \lambda(t)} e^{-\Lambda(t)}`, so the shape of
   :math:`\pi(t)` is controlled by both the local population size :math:`\lambda(t)` and
   the cumulative hazard :math:`\Lambda(t)`.

   .. code-block:: python

      import numpy as np

      t_vals = np.linspace(0.01, 6.0, 200)

      scenarios = {
          "(a) Constant": lambda t: 1.0,
          "(b) Bottleneck": lambda t: 0.1 if 1.0 < t < 2.0 else 1.0,
          "(c) Exp. growth": lambda t: np.exp(t),
      }

      for name, lam_func in scenarios.items():
          C_pi = compute_C_pi(lam_func)
          pi_vals = [stationary_distribution(t, lam_func, C_pi) for t in t_vals]
          peak_idx = np.argmax(pi_vals)
          mean_t, _ = quad(
              lambda t: t * stationary_distribution(t, lam_func, C_pi),
              0.001, 20)
          print(f"{name}: C_pi={C_pi:.4f}, peak at t={t_vals[peak_idx]:.2f}, "
                f"mean T={mean_t:.4f}")

   **How each demographic event shifts the distribution:**

   - **(a) Constant:** :math:`\pi(t) = t e^{-t}`, the Gamma(2,1) distribution.
     Peak at :math:`t = 1`, mean :math:`= 2`. This is the baseline.

   - **(b) Bottleneck:** The small :math:`\lambda(t) = 0.1` in :math:`[1, 2]`
     creates a high coalescence rate in that interval, pulling probability mass
     toward :math:`t \approx 1`. The cumulative hazard increases sharply through
     the bottleneck, so survival past :math:`t = 2` becomes very unlikely.
     :math:`C_\pi` decreases (shorter expected coalescence time), and the
     distribution is compressed toward more recent times.

   - **(c) Exponential growth:** The growing :math:`\lambda(t) = e^t` reduces the
     coalescence rate at large :math:`t`, allowing lineages to survive much longer.
     The distribution becomes more spread out with a heavier right tail.
     :math:`C_\pi` increases (longer expected coalescence time), and the peak
     shifts to a larger :math:`t`.

.. admonition:: Solution 3: Estimate :math:`C_\pi` for a bottleneck model

   **(a)** Compute :math:`C_\pi` numerically for the piecewise population size
   :math:`\lambda(t) = 1` for :math:`t < 1`, :math:`\lambda(t) = 0.1` for
   :math:`t \in [1, 2]`, and :math:`\lambda(t) = 1` for :math:`t > 2`.

   .. code-block:: python

      def bottleneck_lambda(t):
          if t < 1.0:
              return 1.0
          elif t < 2.0:
              return 0.1
          else:
              return 1.0

      C_pi_bottleneck = compute_C_pi(bottleneck_lambda)
      C_pi_constant = compute_C_pi(lambda t: 1.0)
      print(f"C_pi (bottleneck): {C_pi_bottleneck:.6f}")
      print(f"C_pi (constant):   {C_pi_constant:.6f}")
      print(f"Ratio: {C_pi_bottleneck / C_pi_constant:.4f}")

   The bottleneck gives :math:`C_\pi \approx 0.64`, compared to :math:`C_\pi = 1.0`
   for a constant population. The bottleneck forces most coalescence events into
   the interval :math:`[1, 2]`, reducing the mean coalescence time.

   **(b)** A smaller :math:`C_\pi` means lower expected heterozygosity, because
   :math:`P(X_a = 1) \approx C_\pi \theta`. With :math:`C_\pi \approx 0.64`, the
   expected heterozygosity is only 64% of what it would be under a constant
   population. The bottleneck reduces the average time to the most recent common
   ancestor, which means fewer mutations accumulate on average.

   .. code-block:: python

      theta = 0.001
      p_het_constant = C_pi_constant * theta
      p_het_bottleneck = C_pi_bottleneck * theta
      print(f"Expected heterozygosity (constant):   {p_het_constant:.6f}")
      print(f"Expected heterozygosity (bottleneck): {p_het_bottleneck:.6f}")
      print(f"Reduction: {(1 - p_het_bottleneck/p_het_constant)*100:.1f}%")

   **(c)** To detect this bottleneck from data, one would:

   1. **Observe reduced heterozygosity** compared to what a constant-population
      model would predict. The initial :math:`\hat{\theta}` estimate from the
      data would be lower than expected.

   2. **Run PSMC** and examine the inferred :math:`\lambda_k` values. The
      bottleneck should appear as :math:`\lambda_k \ll 1` in the time intervals
      corresponding to :math:`t \in [1, 2]` (in coalescent units). The PSMC
      plot would show a sharp dip in :math:`N_e(t)` at that time.

   3. **Check the posterior decoding**: positions with coalescence times in the
      bottleneck interval :math:`[1, 2]` should show an excess of homozygous
      sites (because the high coalescence rate during the bottleneck means those
      lineages have short branch lengths and thus fewer mutations).
