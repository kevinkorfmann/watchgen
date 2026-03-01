.. _gamma_smc_gamma_approximation:

==============================
The Gamma Approximation
==============================

   *The escapement: the mathematical tick that makes the whole mechanism possible.*

This chapter derives the two update rules at the heart of Gamma-SMC: the
**emission step** (exact, via Poisson-gamma conjugacy) and the **transition
step** (approximate, via gamma projection). Together, they allow the posterior
TMRCA distribution to be tracked as a gamma distribution throughout the entire
forward pass.

.. admonition:: Biology Aside -- What TMRCA tells us about population history

   The **time to most recent common ancestor** (TMRCA) at a genomic position
   is the number of generations since the two haplotypes last shared a common
   ancestor at that site. It is a direct window into population size history:
   in a small population, lineages coalesce quickly, so TMRCAs are short; in
   a large population, coalescence is slow and TMRCAs are long. A population
   bottleneck produces a characteristic signature -- a band of similar TMRCAs
   across many positions -- because the small population forced most lineages
   to coalesce during that period. By estimating the TMRCA distribution at
   each position along the genome, Gamma-SMC effectively reads the history
   of population size changes encoded in the pattern of genetic variation
   between two chromosomes.

In a mechanical watch, the escapement is the component that converts the
continuous energy of the mainspring into discrete, regular ticks. In Gamma-SMC,
the gamma approximation plays an analogous role: it converts the continuous
Bayesian updating process into a sequence of simple parameter updates --
:math:`(\alpha, \beta) \to (\alpha', \beta')` -- that can be evaluated in
constant time.


The Emission Step: Exact Conjugacy
====================================

Suppose the forward density at position :math:`i` is approximated as:

.. math::

   X_i \mid Y_{1:i-1} \sim \text{Gamma}(\alpha, \beta)

We observe :math:`Y_i` at position :math:`i`. Using Bayes' rule and the
Markov property:

.. math::

   P(X_i = t \mid Y_{1:i}) &\propto P(Y_i \mid X_i = t) \cdot P(X_i = t \mid Y_{1:i-1})

The emission model is Poisson with rate :math:`\theta t`:

.. math::

   P(Y_i = y \mid X_i = t) = \frac{(\theta t)^y \cdot e^{-\theta t}}{y!}

.. admonition:: Biology Aside -- Why Poisson emissions?

   At each genomic position, we observe either a heterozygous site
   (:math:`Y_i = 1`, meaning the two haplotypes differ) or a homozygous site
   (:math:`Y_i = 0`, meaning they are identical). Mutations accumulate on the
   two lineages at a constant rate :math:`\mu` per base per generation, so
   the number of differences between the two haplotypes at a given position
   is approximately Poisson-distributed with a rate proportional to the
   TMRCA. Intuitively: *the longer two lineages have been separated, the
   more mutations have accumulated between them, and the more likely they
   are to differ at any given site.* This is the molecular clock principle
   applied at the single-nucleotide level.

The prior is :math:`\text{Gamma}(\alpha, \beta)`:

.. math::

   P(X_i = t \mid Y_{1:i-1}) = \frac{\beta^\alpha}{\Gamma(\alpha)} t^{\alpha - 1} e^{-\beta t}

Multiplying these together:

.. math::

   P(X_i = t \mid Y_{1:i}) &\propto (\theta t)^y \cdot e^{-\theta t} \cdot t^{\alpha - 1} \cdot e^{-\beta t} \\
   &\propto t^{\alpha + y - 1} \cdot e^{-(\beta + \theta) t}

This is the kernel of a :math:`\text{Gamma}(\alpha + y, \beta + \theta)`
distribution. Since the posterior must normalize to a proper density:

.. math::

   X_i \mid Y_{1:i} \sim \text{Gamma}(\alpha + Y_i, \; \beta + \theta)

This gives us the three cases:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Observation
     - Posterior
     - Interpretation
   * - :math:`Y_i = 0` (hom)
     - :math:`\text{Gamma}(\alpha, \beta + \theta)`
     - Rate increases (TMRCA shifts shorter). Seeing no
       mutation is evidence of a more recent ancestor.
   * - :math:`Y_i = 1` (het)
     - :math:`\text{Gamma}(\alpha + 1, \beta + \theta)`
     - Shape and rate both increase. Seeing a mutation is
       evidence of a more distant ancestor (shape goes up)
       while also incorporating the observation (rate goes up).
   * - :math:`Y_i = -1` (missing)
     - :math:`\text{Gamma}(\alpha, \beta)`
     - No change. Missing data carries no information.

.. code-block:: python

   import numpy as np
   from scipy.special import digamma, gammaln

   def gamma_emission_update(alpha, beta, y, theta):
       """Apply the Poisson-gamma conjugate emission update.

       Parameters
       ----------
       alpha : float
           Current shape parameter.
       beta : float
           Current rate parameter.
       y : int
           Observation: 1 (het), 0 (hom), or -1 (missing).
       theta : float
           Scaled mutation rate.

       Returns
       -------
       alpha_new, beta_new : float
           Updated gamma parameters.
       """
       if y == -1:  # missing
           return alpha, beta
       return alpha + y, beta + theta

   # Verify: starting from the prior Gamma(1, 1)
   alpha, beta = 1.0, 1.0
   theta = 0.001

   # After a heterozygous observation
   a_het, b_het = gamma_emission_update(alpha, beta, 1, theta)
   print(f"After het: Gamma({a_het}, {b_het:.4f})")
   print(f"  Mean TMRCA = {a_het/b_het:.4f} "
         f"(shifted up -- mutation is evidence of deeper coalescence)")

   # After a homozygous observation
   a_hom, b_hom = gamma_emission_update(alpha, beta, 0, theta)
   print(f"After hom: Gamma({a_hom}, {b_hom:.4f})")
   print(f"  Mean TMRCA = {a_hom/b_hom:.4f} "
         f"(shifted down -- no mutation favours recent coalescence)")

   # After 100 hom and 1 het
   a, b = 1.0, 1.0
   for _ in range(100):
       a, b = gamma_emission_update(a, b, 0, theta)
   a, b = gamma_emission_update(a, b, 1, theta)
   print(f"\nAfter 100 hom + 1 het: Gamma({a:.1f}, {b:.4f}), "
         f"mean = {a/b:.4f}")

.. admonition:: Why is this called "conjugacy"?

   A prior distribution is **conjugate** to a likelihood if the posterior has
   the same functional form as the prior. Here, the gamma prior is conjugate
   to the Poisson likelihood: regardless of the observation, the posterior is
   always gamma. This means we never need to leave the gamma family during the
   emission step -- the posterior is always characterized by just two numbers
   :math:`(\alpha, \beta)`.

   Conjugacy is one of the most powerful tools in Bayesian statistics. It
   turns Bayesian updating from an integration problem (which is generally
   intractable) into a parameter update (which is :math:`O(1)`). The
   Poisson-gamma pair is one of the classical conjugate families, alongside
   normal-normal, beta-binomial, and Dirichlet-multinomial.

.. admonition:: Intuition: what do :math:`\alpha` and :math:`\beta` track?

   Think of :math:`\alpha` as a "mutation count" and :math:`\beta` as a
   "rate accumulator." At the start (the exponential prior
   :math:`\text{Exp}(1) = \text{Gamma}(1, 1)`), we have seen zero mutations
   and accumulated one unit of rate. Each homozygous site adds :math:`\theta`
   to the rate (more evidence of short TMRCA) without adding to the count.
   Each heterozygous site adds :math:`\theta` to the rate *and* 1 to the
   count (evidence of a mutation, hence longer TMRCA). The posterior mean
   :math:`\alpha / \beta` is literally "mutation count divided by rate" --
   an estimate of the TMRCA.


The Transition Step: Gamma Projection
=======================================

The emission step is exact. The transition step is where approximation enters.

After incorporating the observation at position :math:`i`, we have
:math:`X_i \mid Y_{1:i} \sim \text{Gamma}(\alpha, \beta)`. To advance to
position :math:`i + 1`, we must compute:

.. math::

   p_{\alpha,\beta}(t) := \int_0^\infty p(t \mid s) \cdot f_{\alpha,\beta}(s) \, ds

where :math:`p(t \mid s)` is the SMC' transition density (including the
no-recombination case). This compound distribution :math:`p_{\alpha,\beta}(t)`
is the predictive distribution of the TMRCA at the next position, given our
current gamma belief.

.. admonition:: Biology Aside -- What recombination does to TMRCA

   Recombination breaks chromosomes and re-joins segments from different
   parental chromosomes. As a result, the genealogy of two haplotypes
   changes from position to position along the genome: at one site they may
   share an ancestor 10,000 generations ago, but after a recombination event
   a few bases away, they may share a different ancestor 50,000 generations
   ago. The transition step models this: it asks "given our current belief
   about the TMRCA, what is the distribution of the *new* TMRCA at the next
   position, accounting for the possibility of recombination?" The more
   recombination there is (higher :math:`\rho`), the more the TMRCA can
   change between adjacent positions.

**The problem:** :math:`p_{\alpha,\beta}(t)` is *not* exactly a gamma
distribution. The integral mixes over the transition kernel, which introduces
skewness and other deviations from the gamma family.

**The solution:** Empirically, :math:`p_{\alpha,\beta}(t)` is *very close*
to a gamma distribution (Schweiger & Durbin verify this with simulation
studies; see Appendix A of the supplement). So we approximate it by projecting
back onto the gamma family:

.. math::

   p_{\alpha,\beta}(t) \approx f_{\alpha',\beta'}(t) = \text{Gamma}(\alpha', \beta')

The question becomes: how do we find the best :math:`(\alpha', \beta')`?


The PDE Approach
------------------

For small recombination rate :math:`\rho`, the post-transition distribution
:math:`p_{\alpha,\beta}` is a small perturbation of the prior
:math:`f_{\alpha,\beta}`. We can write:

.. math::

   \alpha' &= \alpha + u \cdot \rho \\
   \beta' &= \beta + v \cdot \rho

where :math:`u` and :math:`v` are the "flow" -- the direction and magnitude
of the gamma parameter change caused by one recombination step.

To find :math:`u` and :math:`v`, we express the perturbation
:math:`(p_{\alpha,\beta} - f_{\alpha,\beta}) / \rho` as a linear combination
of the partial derivatives of :math:`f_{\alpha,\beta}`:

.. math::

   \frac{p_{\alpha,\beta}(x) - f_{\alpha,\beta}(x)}{\rho} \approx
   u \cdot \frac{\partial f_{\alpha,\beta}(x)}{\partial \alpha}
   + v \cdot \frac{\partial f_{\alpha,\beta}(x)}{\partial \beta}

The partial derivatives of the gamma PDF are derived by differentiating
:math:`f_{\alpha,\beta}(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}`.
It is easiest to work with the logarithm first:

.. math::

   \log f_{\alpha,\beta}(x) = \alpha \log\beta - \log\Gamma(\alpha)
   + (\alpha - 1)\log x - \beta x

**Derivative with respect to** :math:`\alpha`:

.. math::

   \frac{\partial}{\partial \alpha} \log f = \log\beta - \psi(\alpha) + \log x

where :math:`\psi(\alpha) = \frac{d}{d\alpha} \ln \Gamma(\alpha)` is the
**digamma function** -- the logarithmic derivative of the gamma function.
Since :math:`\frac{\partial f}{\partial \alpha} = f \cdot \frac{\partial \log f}{\partial \alpha}`:

.. math::

   \frac{\partial f_{\alpha,\beta}(x)}{\partial \alpha}
   = f_{\alpha,\beta}(x) \cdot \left(\log\beta - \psi(\alpha) + \log x\right)

**Derivative with respect to** :math:`\beta`:

.. math::

   \frac{\partial}{\partial \beta} \log f = \frac{\alpha}{\beta} - x

since :math:`\frac{\partial}{\partial \beta}(\alpha\log\beta) = \alpha/\beta`
and :math:`\frac{\partial}{\partial \beta}(-\beta x) = -x`. Therefore:

.. math::

   \frac{\partial f_{\alpha,\beta}(x)}{\partial \beta}
   = f_{\alpha,\beta}(x) \cdot \left(\frac{\alpha}{\beta} - x\right)

.. code-block:: python

   from scipy.special import digamma, gammaln
   import numpy as np

   def gamma_pdf_partials(x, alpha, beta):
       """Compute partial derivatives of the gamma PDF analytically.

       Returns df/dalpha and df/dbeta at each x.
       """
       log_f = (alpha * np.log(beta) - gammaln(alpha)
                + (alpha - 1) * np.log(x) - beta * x)
       f = np.exp(log_f)
       df_dalpha = f * (np.log(beta) - digamma(alpha) + np.log(x))
       df_dbeta = f * (alpha / beta - x)
       return df_dalpha, df_dbeta

   # Verify against numerical finite differences
   alpha, beta = 3.0, 2.0
   x = np.linspace(0.01, 5.0, 100)
   eps = 1e-7

   df_da_analytic, df_db_analytic = gamma_pdf_partials(x, alpha, beta)

   # Numerical d/dalpha
   f_plus = np.exp((alpha + eps) * np.log(beta) - gammaln(alpha + eps)
                   + (alpha + eps - 1) * np.log(x) - beta * x)
   f_minus = np.exp((alpha - eps) * np.log(beta) - gammaln(alpha - eps)
                    + (alpha - eps - 1) * np.log(x) - beta * x)
   df_da_numerical = (f_plus - f_minus) / (2 * eps)

   print(f"Max error in df/dalpha: {np.max(np.abs(df_da_analytic - df_da_numerical)):.2e}")

We solve for :math:`u, v` by least squares over a grid of 2,000 values of
:math:`x` covering the main support of :math:`f_{\alpha,\beta}`:

.. math::

   \arg\min_{u,v} \sum_{i=1}^{2000}
   \left(
   u \cdot \frac{\partial f_{\alpha,\beta}(x_i)}{\partial \alpha}
   + v \cdot \frac{\partial f_{\alpha,\beta}(x_i)}{\partial \beta}
   - \frac{p_{\alpha,\beta}(x_i) - f_{\alpha,\beta}(x_i)}{\rho}
   \right)^2

.. admonition:: Why least squares and not moment matching?

   Moment matching (equating the mean and variance of :math:`p_{\alpha,\beta}`
   to those of :math:`\text{Gamma}(\alpha', \beta')`) is another natural
   approach. The PDE/least-squares method is preferred because it minimizes
   the :math:`L^2` distance between the true perturbation and its
   gamma approximation over the entire support of the distribution. This
   gives a better fit in the tails, which matters because the tails of the
   TMRCA distribution carry information about extreme coalescence times.


The Closed-Form Perturbation
-------------------------------

The key quantity :math:`(p_{\alpha,\beta}(t) - f_{\alpha,\beta}(t))/\rho`
can be evaluated analytically. Starting from the SMC' transition density and
integrating against the gamma prior, one obtains (see the full derivation in
Appendix E of the supplement):

.. math::

   \frac{p_{\alpha,\beta}(t) - f_{\alpha,\beta}(t)}{\rho} \approx \;
   & e^{-t} \cdot \frac{(\beta t)^\alpha}{\Gamma(\alpha + 1)}
   \left[ M(\alpha, \alpha+1, (\beta - 1)t) - M(\alpha, \alpha+1, -(\beta+1)t) \right] \\
   & + \frac{2t + e^{-2t} - 1}{2} \cdot f_{\alpha,\beta}(t) \\
   & + (1 - e^{-2t}) \cdot \frac{\Gamma(\alpha, \beta t)}{\Gamma(\alpha)} \\
   & - 2t \cdot f_{\alpha,\beta}(t)

where:

- :math:`M(a, b, z) = {}_1F_1(a, b, z)` is **Kummer's confluent
  hypergeometric function**, a special function that generalizes the
  exponential. It is evaluated using the Arb arbitrary-precision library
  (Johansson, 2017).

- :math:`\Gamma(\alpha, x) = \int_x^\infty t^{\alpha-1} e^{-t} dt` is the
  **upper incomplete gamma function**.

The four terms arise from integrating the SMC' transition density against the
gamma prior, splitting the integral into three regions (:math:`s < t`,
:math:`s = t`, and :math:`s > t`):

1. The first term covers recombination at :math:`s < t` (the detached lineage
   coalesces at a *later* time).
2. The second term covers the self-coalescence (the lineage re-coalesces onto
   the same branch at :math:`t = s`).
3. The third term covers recombination at :math:`s > t` (the detached lineage
   coalesces at an *earlier* time).
4. The fourth term subtracts the no-recombination contribution (which is
   already counted in :math:`f_{\alpha,\beta}`).

.. admonition:: A critical property: parameter independence

   The flow :math:`(u, v)` at each grid point :math:`(\alpha, \beta)` depends
   only on :math:`\alpha` and :math:`\beta` -- not on :math:`\theta`,
   :math:`\rho`, or :math:`N_e`. This is because the perturbation is
   normalized by :math:`\rho`, and the remaining expression involves only the
   gamma distribution parameters and the SMC' transition kernel (which, for
   constant population size, has no free parameters beyond the time units).

   This parameter independence is what makes Gamma-SMC's precomputation
   strategy possible: the flow field is computed **once** and reused for any
   dataset with any :math:`\theta` and :math:`\rho`.


Log-Coordinates
-----------------

Working directly with :math:`(\alpha, \beta)` is numerically inconvenient
because these parameters span several orders of magnitude. Instead, Gamma-SMC
uses **log-coordinates** based on the mean and coefficient of variation:

.. math::

   l_\mu &= \log_{10}\!\left(\frac{\alpha}{\beta}\right) \quad \text{(log-mean)} \\
   l_C &= \log_{10}\!\left(\frac{1}{\sqrt{\alpha}}\right) \quad \text{(log-CV)}

The flow field is evaluated over a grid of these coordinates:

- 51 values of :math:`l_\mu` equally spaced between :math:`-5` and :math:`2`
  (i.e., log-spaced means from :math:`10^{-5}` to :math:`10^2` coalescent
  time units)
- 50 values of :math:`l_C` equally spaced between :math:`-2` and :math:`0`
  (CVs from :math:`10^{-2}` to :math:`1`)

Values with :math:`C_V > 1` (:math:`\alpha < 1`) are excluded because this
would give an infinite gamma density at :math:`x = 0`, which is unphysical
for a TMRCA.

.. admonition:: Why log-coordinates?

   Log-coordinates serve two purposes. First, they allow the grid to cover
   many orders of magnitude uniformly -- a TMRCA could be :math:`10^{-3}`
   coalescent units (very recent) or :math:`10^1` (very ancient), and
   log-spacing gives equal resolution at all scales. Second, the Taylor
   expansion used for the PDE approach is performed in
   :math:`(\log_{10} \alpha, \log_{10} \beta)` coordinates, not in
   :math:`(l_\mu, l_C)` directly. The conversion uses the chain rule:

   .. math::

      \Delta l_\mu &= \Delta \log_{10}(\alpha) - \Delta \log_{10}(\beta) \\
      \Delta l_C &= -0.5 \cdot \Delta \log_{10}(\alpha)

   This avoids issues with the Taylor expansion that would arise from
   working in the :math:`(l_\mu, l_C)` coordinate system directly.


.. code-block:: python

   def to_log_coords(alpha, beta):
       """Convert (alpha, beta) to log-mean / log-CV coordinates.

       Parameters
       ----------
       alpha : float
           Shape parameter (must be > 0).
       beta : float
           Rate parameter (must be > 0).

       Returns
       -------
       l_mu : float
           log10(alpha/beta), the log-mean TMRCA.
       l_C : float
           log10(1/sqrt(alpha)), the log coefficient of variation.
       """
       l_mu = np.log10(alpha / beta)
       l_C = np.log10(1.0 / np.sqrt(alpha))
       return l_mu, l_C

   def from_log_coords(l_mu, l_C):
       """Convert log-coordinates back to (alpha, beta).

       Parameters
       ----------
       l_mu : float
           Log-mean coordinate.
       l_C : float
           Log-CV coordinate.

       Returns
       -------
       alpha, beta : float
       """
       alpha = 10.0 ** (-2 * l_C)
       beta = alpha * 10.0 ** (-l_mu)
       return alpha, beta

   # Verify round-trip conversion
   for alpha, beta in [(1.0, 1.0), (5.0, 2.5), (100.0, 50.0)]:
       l_mu, l_C = to_log_coords(alpha, beta)
       a_back, b_back = from_log_coords(l_mu, l_C)
       print(f"Gamma({alpha}, {beta}) -> (l_mu={l_mu:.4f}, l_C={l_C:.4f}) "
             f"-> Gamma({a_back:.4f}, {b_back:.4f})")

   # The prior Gamma(1, 1) maps to (0, 0) in log-coordinates
   l_mu_prior, l_C_prior = to_log_coords(1.0, 1.0)
   print(f"\nPrior Gamma(1,1): (l_mu, l_C) = ({l_mu_prior:.1f}, {l_C_prior:.1f})")

   # The grid covers:
   l_mu_grid = np.linspace(-5, 2, 51)
   l_C_grid = np.linspace(-2, 0, 50)
   print(f"Grid: {len(l_mu_grid)} x {len(l_C_grid)} = "
         f"{len(l_mu_grid) * len(l_C_grid)} points")


Summary
=========

The gamma approximation rests on two pillars:

1. **Emission step** (exact): :math:`\text{Gamma}(\alpha, \beta) + Y_i
   \to \text{Gamma}(\alpha + Y_i, \beta + \theta)`. This is Poisson-gamma
   conjugacy -- a textbook result that requires no approximation.

2. **Transition step** (approximate): :math:`\text{Gamma}(\alpha, \beta)
   \to \text{Gamma}(\alpha + u \cdot \rho, \beta + v \cdot \rho)`, where
   :math:`(u, v)` are determined by least-squares fitting to the true
   perturbation. This requires a one-time precomputation over a
   two-dimensional grid.

Together, these two rules mean that the entire forward pass of the HMM
reduces to a sequence of arithmetic operations on two numbers
:math:`(\alpha, \beta)`. No matrix multiplications, no numerical integration
during inference, no discretization of time. The cost of one step is
:math:`O(1)`, making the full forward pass :math:`O(N)` in the sequence
length.

Next: :ref:`gamma_smc_flow_field` -- precomputing the transition machinery
into a reusable vector field.
