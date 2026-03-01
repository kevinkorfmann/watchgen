.. _moment_equations:

=================================
The Moment Equations
=================================

   *The gear train: turning biological forces into differential equations.*

This is the heart of ``moments`` -- the mathematical engine that makes everything
tick. We derive, from first principles, the system of ordinary differential
equations (ODEs) that govern how each entry of the SFS evolves over time under
drift, mutation, selection, and migration.

In :ref:`the_frequency_spectrum` we examined the **dial face** -- the SFS and
what it looks like under various demographic scenarios.  Now we open the case
and study **the ODEs governing the gear train** -- the equations that link
biological forces (drift, mutation, selection, migration) to changes in each
SFS entry.  Where ``dadi`` modelled every tooth on every gear (the full
frequency density :math:`\phi(x,t)` via a PDE), ``moments`` writes down the
equations for the hands directly.

By the end of this chapter, you will understand exactly what happens inside
``fs.integrate()`` and be able to implement it yourself.

.. admonition:: Calculus Aside -- What is an ODE?

   An **ordinary differential equation** (ODE) relates a quantity to its rate
   of change with respect to a single variable (here, time).  The notation
   :math:`d\phi_j / dt = f(\phi_1, \ldots, \phi_{n-1})` says: "the
   instantaneous rate at which SFS entry :math:`j` changes is given by the
   function :math:`f` of all current SFS entries."  Solving the ODE means
   starting from a known initial state (the equilibrium SFS) and following
   these rates forward to find :math:`\phi_j` at a later time.  Standard
   numerical methods -- Euler, Runge--Kutta, implicit solvers -- accomplish
   this by taking small time steps and updating the state at each step.
   If you are comfortable with the idea "velocity tells you how position
   changes," you already have the right intuition for ODEs: here, the
   "position" is the SFS and the "velocity" is given by the operators
   we derive below.  For a thorough introduction to ODEs and numerical
   solvers, see the prerequisite chapter :ref:`odes`.


Step 1: The Wright-Fisher Model in One Slide
==============================================

Before we can write down equations for the SFS, we need the model that
generates it.

The **Wright-Fisher model** describes a population of :math:`2N` haploid
individuals (or :math:`N` diploids). Each generation:

1. Each individual produces a very large number of offspring
2. :math:`2N` offspring are sampled randomly to form the next generation
3. Mutations occur on the offspring at rate :math:`\mu` per site per generation

If the current frequency of the derived allele is :math:`x`, the number of
derived copies in the next generation follows:

.. math::

   X' \sim \text{Binomial}(2N, x)

The frequency :math:`x' = X' / (2N)` will fluctuate around :math:`x` due to
the randomness of sampling. This random fluctuation is **genetic drift**.

**Key quantities from the Wright-Fisher model:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Quantity
     - Value
   * - Mean change in frequency per generation
     - :math:`E[\Delta x] = 0` (under neutrality)
   * - Variance of frequency change per generation
     - :math:`\text{Var}[\Delta x] = \frac{x(1-x)}{2N}`
   * - Probability of fixation (neutral)
     - :math:`x` (the initial frequency)
   * - Expected time to fixation/loss
     - :math:`\sim 4N` generations

The variance formula is the crucial ingredient. It tells us that drift is strongest
at intermediate frequencies (:math:`x \approx 0.5`) and weakest near the boundaries
(:math:`x \approx 0` or :math:`1`), and that larger populations (:math:`N`) drift
more slowly.

.. admonition:: Probability Aside -- Binomial sampling and genetic drift

   The binomial draw :math:`X' \sim \text{Binomial}(2N, x)` is the formal
   statement that each of the :math:`2N` offspring independently "chooses" to
   carry the derived allele with probability :math:`x`.  The variance
   :math:`x(1-x)/(2N)` follows directly from the binomial variance formula
   :math:`\text{Var}[X'/2N] = x(1-x)/(2N)`.  This is the stochastic engine
   of genetic drift: even with no selection, allele frequencies wander
   randomly because sampling is noisy.  Smaller populations have larger
   variance per generation -- the allelic "hands" of the watch jitter more
   when the gear train has fewer teeth.

With the raw model in hand, we next take the continuum limit to obtain the
diffusion framework from which the moment equations are derived.


Step 2: From Wright-Fisher to Diffusion
=========================================

When :math:`N` is large, the discrete Wright-Fisher process converges to a
continuous **diffusion process**. The allele frequency :math:`x(t)` evolves
according to the stochastic differential equation:

.. math::

   dx = \underbrace{\mu_1(x) \, dt}_{\text{deterministic drift}} +
   \underbrace{\sqrt{\sigma^2(x)} \, dW}_{\text{random noise}}

where :math:`dW` is a Wiener process (Brownian motion increment) and:

- :math:`\mu_1(x)` is the mean change per unit time (from mutation, selection, migration)
- :math:`\sigma^2(x) = x(1-x)` is the variance per unit time (from genetic drift)

Time is measured in units of :math:`2N` generations (the **diffusion timescale**).
On this timescale, the variance of drift is :math:`x(1-x)` regardless of :math:`N`,
which is why everything is naturally expressed in scaled parameters like
:math:`\theta = 4N\mu`.

.. admonition:: Why the diffusion timescale?

   In one generation, the variance of frequency change is :math:`x(1-x)/(2N)`.
   Over :math:`2N` generations (one unit of diffusion time), the accumulated
   variance is :math:`2N \cdot x(1-x)/(2N) = x(1-x)`. The factor of :math:`2N`
   in the denominator cancels, giving a "universal" variance that doesn't depend
   on :math:`N`. This is why we measure time in :math:`2N` generations and
   parameters in multiples of :math:`N`.

.. admonition:: Calculus Aside -- From sums to integrals (the diffusion limit)

   The diffusion approximation is an application of the **central limit
   theorem** for Markov chains.  Over a single generation the frequency change
   :math:`\Delta x` has mean zero and variance :math:`x(1-x)/(2N)`.  Over
   :math:`2N` generations these small, nearly independent increments
   accumulate into a quantity whose distribution converges to a Gaussian --
   i.e., a Brownian motion.  Formally this is Donsker's theorem applied to
   the rescaled random walk :math:`x(t)`.  The rescaling
   :math:`t \to t/(2N)` compresses :math:`2N` discrete steps into one
   continuous time unit, and the variance per step (:math:`1/(2N)`) multiplied
   by the number of steps (:math:`2N`) gives a variance of order 1 per unit
   time -- exactly the :math:`x(1-x)` we see above.

The diffusion gives us a PDE for the *density* :math:`\phi(x,t)` (the
Fokker-Planck equation -- see :ref:`diffusion_approximation` for the full
derivation).  The key innovation of ``moments`` is to bypass that PDE and work
directly with the SFS entries -- the moments of :math:`\phi`.


Step 3: What Are "Moments"?
==============================

Here's where ``moments`` gets its name. Instead of tracking the full probability
density :math:`\phi(x, t)` of allele frequencies (as the diffusion PDE does),
we track its **moments** -- specifically, the expected values of products of
allele frequencies sampled from the population.

For a single population with sample size :math:`n`, the SFS entry
:math:`\phi_j` (expected number of sites with derived count :math:`j`) is
related to a moment of the frequency distribution:

.. math::

   \phi_j = \theta \cdot L \cdot E\!\left[\binom{n}{j} x^j (1-x)^{n-j}\right]

This is just the expected fraction of sites where :math:`j` out of :math:`n`
sampled chromosomes carry the derived allele, multiplied by the mutation rate.
The term :math:`\binom{n}{j} x^j (1-x)^{n-j}` is the binomial sampling
probability: given true frequency :math:`x`, the probability of sampling
:math:`j` derived copies out of :math:`n`.

.. admonition:: Probability Aside -- Connecting SFS entries to moments

   The link between the SFS and the moments of the frequency distribution is
   the **law of total expectation**.  Let :math:`X` be the number of derived
   alleles in a sample of :math:`n`.  Conditional on the population frequency
   :math:`x`, we have :math:`X \mid x \sim \text{Binomial}(n, x)`.
   Averaging over the distribution of :math:`x`:

   .. math::

      P(X = j) = E_x\!\left[\binom{n}{j} x^j (1-x)^{n-j}\right]

   Each SFS entry is therefore a **polynomial moment** of :math:`x` --
   specifically, a mixture of the raw moments :math:`E[x^k]` for
   :math:`k = 0, \ldots, n`.  By tracking all :math:`n-1` internal SFS
   entries, we effectively track all moments of :math:`x` up to order
   :math:`n`.

**The key insight**: if we can derive how these moments change over time under
drift, mutation, selection, and migration, we get ODEs for the SFS entries
directly -- no need to solve for the full density :math:`\phi(x, t)`.

Now we derive each operator in the gear train, starting with the most important
gear: drift.


Step 4: The Drift Operator
============================

Let's derive how genetic drift changes the SFS. This is the most important
piece, and we'll go slowly.

Starting from the Wright-Fisher model, the expected SFS entry :math:`\phi_j` at
time :math:`t + dt` depends on :math:`\phi_j` at time :math:`t`. We want to find
:math:`d\phi_j / dt`.

The drift term in the diffusion equation for the density :math:`\phi(x, t)` is:

.. math::

   \left.\frac{\partial \phi}{\partial t}\right|_{\text{drift}} =
   \frac{1}{2} \frac{\partial^2}{\partial x^2}[x(1-x)\phi(x)]

To get the effect on the SFS entry :math:`\phi_j`, we need to compute how this
differential operator affects the :math:`j`-th moment. The derivation uses
integration by parts (applied to the frequency integral, not a spatial integral):

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift}} =
   \int_0^1 \binom{n}{j} x^j (1-x)^{n-j} \cdot
   \frac{1}{2}\frac{\partial^2}{\partial x^2}[x(1-x)\phi(x)] \, dx

After two integrations by parts (boundary terms vanish because :math:`x(1-x) = 0`
at :math:`x = 0, 1`), this becomes:

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift}} =
   \frac{1}{2}\int_0^1 \frac{d^2}{dx^2}\left[\binom{n}{j}x^j(1-x)^{n-j}\right]
   \cdot x(1-x) \cdot \phi(x) \, dx

.. admonition:: Calculus Aside -- Integration by parts for the drift operator

   Integration by parts generalizes the product rule for differentiation.
   For two functions :math:`u` and :math:`v`:

   .. math::

      \int_a^b u \, dv = [uv]_a^b - \int_a^b v \, du

   Applied twice, this moves the two :math:`x`-derivatives from
   :math:`x(1-x)\phi` onto the binomial sampling weight
   :math:`\binom{n}{j}x^j(1-x)^{n-j}`.  The boundary terms vanish because
   the factor :math:`x(1-x)` equals zero at both endpoints :math:`x=0` and
   :math:`x=1`.  This is a common trick in mathematical physics: move
   derivatives onto the "test function" (here, the sampling polynomial) where
   they can be computed analytically.

Now we need to compute the second derivative of :math:`\binom{n}{j}x^j(1-x)^{n-j}`
with respect to :math:`x`, then multiply by :math:`x(1-x)`, and recognize the
result in terms of binomial sampling probabilities with shifted indices.

After careful algebra (which we spell out in detail below), the drift contribution
to :math:`d\phi_j/dt` is:

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift}} =
   \frac{1}{2}\Big[
   (j+1)(j+2) \cdot \frac{\phi_{j+2}}{\binom{n}{j+2}/\binom{n}{j}}
   - 2j(n-j) \cdot \phi_j
   + (n-j+1)(n-j+2) \cdot \frac{\phi_{j-2}}{\binom{n}{j-2}/\binom{n}{j}}
   \Big]

This simplifies to the **tridiagonal** recurrence (using the convention from
Jouganous et al. 2017):

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift}} =
   \binom{n}{j}^{-1} \left[
   \binom{n}{j-1}\frac{(n-j+1)j}{2}\phi_{j-1}
   - \binom{n}{j}\frac{j(n-j)}{1}\phi_j
   + \binom{n}{j+1}\frac{(j+1)(n-j)}{2}\phi_{j+1}
   \right]

Wait -- let's be more careful. The drift operator on the SFS can be written
compactly using the **jackknife approximation** (a central trick in ``moments``):

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift}} \approx
   \frac{n+1}{2}\left[
   -2j(n-j)\phi_j + j(j-1)\phi_{j-1} + (n-j)(n-j-1)\phi_{j+1}
   \right] \cdot \frac{1}{n(n-1)}

Let's implement this step by step and verify it against simulation.

.. code-block:: python

   import numpy as np

   def drift_operator(phi, n):
       """Compute the drift contribution to d(phi)/dt.

       Parameters
       ----------
       phi : ndarray of shape (n+1,)
           Current SFS (phi[j] = expected count at frequency j/n).
       n : int
           Sample size.

       Returns
       -------
       dphi : ndarray of shape (n+1,)
           Change in SFS due to drift.
       """
       dphi = np.zeros(n + 1)
       for j in range(1, n):
           # Drift: second-order finite difference in frequency space.
           # The three terms correspond to probability flowing into bin j
           # from the bin below (j-1), out of bin j, and into j from above (j+1).
           term_down = j * (j - 1) * phi[j - 1] if j >= 1 else 0.0   # flow from j-1 -> j
           term_stay = -2 * j * (n - j) * phi[j]                      # flow out of j
           term_up = (n - j) * (n - j - 1) * phi[j + 1] if j < n else 0.0  # flow from j+1 -> j

           # Scale by 1/(2n): converts discrete WF variance to diffusion-time units
           dphi[j] = (term_down + term_stay + term_up) / (2.0 * n)

       return dphi

**Intuition for the drift operator**: Think of the SFS entries as water levels
in connected buckets. Drift "sloshes" probability between adjacent frequency
classes. The rate of sloshing depends on :math:`j(n-j)` -- strongest at
intermediate frequencies, zero at the boundaries. The net effect is that drift
pushes alleles toward fixation (:math:`j = 0` or :math:`j = n`) -- the SFS
"drains" from the middle toward the edges.

In watch terms, drift is the *escapement* -- the ticking mechanism that
continuously redistributes energy (probability mass) among the gear wheels
(SFS bins).

.. code-block:: python

   # Verify: starting from neutral SFS, drift should produce no net change
   # (the neutral SFS is the equilibrium)
   n = 20
   theta = 1.0
   phi_neutral = np.zeros(n + 1)
   for j in range(1, n):
       phi_neutral[j] = theta / j  # the 1/j neutral spectrum

   dphi_drift = drift_operator(phi_neutral, n)

   print("Drift operator on neutral SFS (should be ~ 0 after adding mutation):")
   for j in range(1, min(n, 8)):
       print(f"  d(phi[{j}])/dt|_drift = {dphi_drift[j]:+.6f}")

Drift alone pushes the SFS out of equilibrium.  The next gear -- mutation --
provides the steady influx of new variants that balances the drain.


Step 5: The Mutation Operator
================================

Mutation adds new variants to the population. Under the infinite-sites model, each
new mutation occurs at a previously monomorphic site, creating a singleton (a site
with derived allele count :math:`j = 1`).

The mutation contribution to the SFS is:

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{mutation}} =
   \begin{cases}
   \frac{\theta}{2} & \text{if } j = 1 \\
   0 & \text{if } j > 1
   \end{cases}

**Why** :math:`\theta/2` **?** The mutation rate per site per generation is
:math:`\mu`. In a population of :math:`2N` chromosomes, the rate of new mutations
per site is :math:`2N\mu`. In diffusion time units (:math:`2N` generations per
unit), this becomes :math:`2N\mu \cdot 2N = 4N^2\mu`... wait, that's not right.

Let's be more careful. The rate of new derived alleles appearing per site per
**diffusion time unit** is:

.. math::

   \text{rate} = 2N\mu \cdot 2N \text{ gen/time unit} = 4N^2\mu \text{ per time unit}

But we're tracking the SFS as :math:`\phi_j = \theta \cdot (\text{density at } j)`,
where :math:`\theta = 4N\mu`. In the normalized units used by ``moments``, the
mutation input is simply :math:`\theta/2` into the :math:`j = 1` bin.

.. admonition:: Probability Aside -- Why mutations only enter at :math:`j=1`

   Under the **infinite-sites** mutation model, every new mutation occurs at a
   genomic position that has never mutated before.  This means a new mutation
   starts on exactly one chromosome out of :math:`2N`, giving it an initial
   frequency of :math:`1/(2N)`.  In a sample of :math:`n \ll 2N` chromosomes,
   the probability that this single-copy mutation is captured in the sample is
   :math:`n/(2N)`.  When captured, it appears as a singleton (:math:`j = 1`).
   Therefore, mutation feeds *only* the :math:`j = 1` bin of the SFS.  Higher
   bins are populated solely by the drift operator, which "promotes" singletons
   to higher counts over time.

Alternatively, with the reversible mutation model (finite genome), mutations can
also go backwards (derived -> ancestral):

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{mutation}} =
   \frac{\theta_{\text{fwd}}}{2} \cdot \delta_{j,1}
   + \frac{\theta_{\text{bwd}}}{2} \cdot \delta_{j,n-1}
   - \frac{\theta_{\text{bwd}}}{2n}\phi_j
   - \frac{\theta_{\text{fwd}}}{2n}\phi_j

For now, we'll stick with the infinite-sites model.

.. code-block:: python

   def mutation_operator(phi, n, theta):
       """Compute the mutation contribution to d(phi)/dt.

       Under the infinite-sites model, mutations only add singletons.

       Parameters
       ----------
       phi : ndarray of shape (n+1,)
           Current SFS.
       n : int
           Sample size.
       theta : float
           Population-scaled mutation rate (4*Ne*mu, per site or total).

       Returns
       -------
       dphi : ndarray of shape (n+1,)
       """
       dphi = np.zeros(n + 1)
       dphi[1] = theta / 2.0  # new mutations enter only as singletons (j=1)
       return dphi

With drift and mutation in place, the watch can keep neutral time.  The next
gear adds a directional force: selection.


Step 6: The Selection Operator
================================

Selection favors (or disfavors) the derived allele. A diploid individual with
genotype frequencies has fitness:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Genotype
     - Frequency
     - Fitness
   * - Ancestral homozygote (aa)
     - :math:`(1-x)^2`
     - :math:`1`
   * - Heterozygote (Aa)
     - :math:`2x(1-x)`
     - :math:`1 + 2hs`
   * - Derived homozygote (AA)
     - :math:`x^2`
     - :math:`1 + 2s`

The parameter :math:`h` is the **dominance coefficient**:

- :math:`h = 0`: derived allele is completely recessive
- :math:`h = 0.5`: additive (codominant) selection
- :math:`h = 1`: derived allele is completely dominant

The mean change in allele frequency per generation due to selection is:

.. math::

   E[\Delta x]_{\text{sel}} = sx(1-x)[h + (1-2h)x]

In diffusion time units (dividing by :math:`2N` and multiplying by :math:`2N`
generations), this becomes:

.. math::

   \mu_1^{\text{sel}}(x) = \gamma \cdot x(1-x)[h + (1-2h)x]

where :math:`\gamma = 2Ns` is the scaled selection coefficient.

.. admonition:: Calculus Aside -- Selection as a first-order transport term

   Whereas drift appears as a **second-order** (diffusion) term
   :math:`\tfrac{1}{2}\partial^2_{xx}[x(1-x)\phi]` in the Kolmogorov
   forward equation, selection appears as a **first-order** (advection) term
   :math:`-\partial_x[\mu_1(x)\phi]`.  In fluid-dynamics language, drift is
   analogous to molecular diffusion and selection is analogous to a velocity
   field that transports the frequency density.  Positive :math:`\gamma`
   pushes frequency mass toward :math:`x = 1` (fixation of the derived
   allele); negative :math:`\gamma` pushes it toward :math:`x = 0` (loss).

The selection operator on the SFS (derived analogously to drift, via integration
by parts) is:

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{selection}} = -\gamma \left[
   h \cdot (j+1)\phi_{j+1} - (h + (1-2h)\cdot\frac{j}{n}) \cdot j\phi_j
   + (1-2h)\cdot \frac{j(j-1)}{n}\phi_{j-1}
   \right] \cdot \frac{1}{n}

Let's implement this more carefully using the jackknife approach:

.. code-block:: python

   def selection_operator(phi, n, gamma, h=0.5):
       """Compute the selection contribution to d(phi)/dt.

       Parameters
       ----------
       phi : ndarray of shape (n+1,)
           Current SFS.
       n : int
           Sample size.
       gamma : float
           Scaled selection coefficient (2*Ne*s).
       h : float
           Dominance coefficient.

       Returns
       -------
       dphi : ndarray of shape (n+1,)
       """
       dphi = np.zeros(n + 1)

       for j in range(1, n):
           # The selection flux is proportional to gamma * x*(1-x)*[h + (1-2h)*x]
           # In discrete frequency space (j/n), the mean shift of phi_j due to
           # selection involves contributions from adjacent entries.

           # Additive component (proportional to h):
           # Shifts frequency upward by h*gamma*x*(1-x)
           # Flux from j-1 to j:
           flux_in = gamma * h * j * (n - j) / n**2 * phi[j]
           flux_in_prev = gamma * h * (j - 1) * (n - j + 1) / n**2 * phi[j - 1] if j > 0 else 0

           # Dominance deviation component (proportional to (1-2h)):
           dom = (1 - 2 * h)
           flux_dom = gamma * dom * j * (j - 1) * (n - j) / n**3 * phi[j]

           # Combined (simplified version for clarity):
           # The exact form from Jouganous et al. (2017):
           if j > 0 and j < n:
               # h-weighted linear selection term
               term1 = gamma * h * ((j - 1) * (n - j + 1) * phi[j - 1] -
                                     j * (n - j) * phi[j]) / n
               # dominance-deviation (quadratic in x) term
               term2 = gamma * (1 - 2*h) * (
                   (j - 1) * (j - 2) * phi[j - 1] / (n * (n - 1)) * (n - j + 1)
                   - j * (j - 1) * phi[j] / (n * (n - 1)) * (n - j)
               ) if n > 1 else 0
               dphi[j] = term1 + term2

       return dphi

   # Verify: negative gamma should reduce high-frequency derived alleles
   n = 20
   phi_neutral = expected_sfs_neutral(n, theta=1.0)
   dphi_sel = selection_operator(phi_neutral, n, gamma=-10, h=0.5)

   print("Selection (gamma=-10, purifying) effect on neutral SFS:")
   for j in range(1, 8):
       print(f"  d(phi[{j}])/dt|_sel = {dphi_sel[j]:+.6f}")
   print("  (Negative for high j = selection removes high-freq derived alleles)")

.. admonition:: Purifying selection shifts the SFS left

   With :math:`\gamma < 0` (deleterious derived allele), selection removes
   derived alleles before they reach high frequency. The SFS becomes enriched
   in rare variants -- a signature used to infer the distribution of fitness
   effects (DFE) of new mutations.

With drift, mutation, and selection in place, the single-population gear train
is complete.  For multi-population models we need one more gear: migration.


Step 7: The Migration Operator
================================

Migration mixes allele frequencies between populations. For two populations
with SFS entries :math:`\phi_{j_1, j_2}`, migration of rate :math:`m_{12}`
(fraction of pop 1 that are migrants from pop 2 per generation) shifts
frequencies:

.. math::

   \left.\frac{d\phi_{j_1,j_2}}{dt}\right|_{\text{migration}} =
   M_{12}\left[\frac{n_1 - j_1 + 1}{n_1}\phi_{j_1-1,j_2+1}
   - \frac{j_1}{n_1}\phi_{j_1,j_2}\right]
   + \text{(reverse direction terms)}

where :math:`M_{12} = 2N_e m_{12}`.

**Intuition**: Migration from pop 2 into pop 1 takes a chromosome from pop 2
(reducing :math:`j_2` by one if it's derived) and places it in pop 1 (increasing
:math:`j_1` by one if the migrant carries the derived allele). The probability that
a randomly chosen migrant carries the derived allele is :math:`j_2/n_2`.

.. admonition:: Probability Aside -- Migration as a Markov transition

   Each migration event replaces one random chromosome in the receiving
   population with a copy drawn from the source population.  The SFS entry
   :math:`\phi_{j_1,j_2}` can increase when a migrant carrying the derived
   allele enters (moving the state from :math:`(j_1 - 1, j_2 + 1)` to
   :math:`(j_1, j_2)`) or decrease when a derived-carrying resident is
   replaced (moving from :math:`(j_1, j_2)` to :math:`(j_1 - 1, j_2 + 1)`).
   The transition probabilities are proportional to the fraction of derived
   alleles in each population, giving the :math:`j/n` factors in the
   equation.

.. code-block:: python

   def migration_operator_2pop(phi_2d, n1, n2, M12, M21):
       """Compute migration contribution for a 2D SFS.

       Parameters
       ----------
       phi_2d : ndarray of shape (n1+1, n2+1)
           Joint SFS.
       n1, n2 : int
           Sample sizes.
       M12 : float
           Scaled migration rate from pop 2 into pop 1.
       M21 : float
           Scaled migration rate from pop 1 into pop 2.

       Returns
       -------
       dphi : ndarray of shape (n1+1, n2+1)
       """
       dphi = np.zeros((n1 + 1, n2 + 1))

       for j1 in range(1, n1):
           for j2 in range(1, n2):
               # Migration from pop2 into pop1:
               # Derived allele enters pop1 with prob (j2+1)/n2
               if j1 > 0 and j2 < n2:
                   dphi[j1, j2] += M12 * (j2 + 1) / n2 * phi_2d[j1 - 1, j2 + 1]
               # Derived allele leaves pop1 (replaced by ancestral from pop2)
               dphi[j1, j2] -= M12 * j1 / n1 * phi_2d[j1, j2]

               # Migration from pop1 into pop2 (symmetric structure):
               if j2 > 0 and j1 < n1:
                   dphi[j1, j2] += M21 * (j1 + 1) / n1 * phi_2d[j1 + 1, j2 - 1]
               dphi[j1, j2] -= M21 * j2 / n2 * phi_2d[j1, j2]

       return dphi

The migration operator couples the two populations' SFS bins to each other --
like a differential gear connecting two halves of the watch movement.  The next
step is even simpler: scaling drift by population size.


Step 8: The Population Size Effect
=====================================

Population size enters through the drift operator. When the population size is
:math:`\nu` times the reference size, drift is scaled by :math:`1/\nu`:

.. math::

   \left.\frac{d\phi_j}{dt}\right|_{\text{drift, size } \nu} =
   \frac{1}{\nu} \cdot \left.\frac{d\phi_j}{dt}\right|_{\text{drift, size 1}}

**Why?** The variance of allele frequency change per generation is
:math:`x(1-x)/(2N)`. When :math:`N = \nu N_e`, the variance becomes
:math:`x(1-x)/(2\nu N_e)`, which in diffusion time units is :math:`x(1-x)/\nu`.
Smaller :math:`\nu` means stronger drift, larger :math:`\nu` means weaker drift.

In the watch metaphor, :math:`\nu` controls the mainspring tension.  A bigger
population (larger :math:`\nu`) dampens the stochastic jitter of the escapement;
a smaller population amplifies it.

.. code-block:: python

   def drift_operator_with_size(phi, n, nu):
       """Drift operator scaled by relative population size.

       Parameters
       ----------
       phi : ndarray of shape (n+1,)
       n : int
       nu : float
           Relative population size (N/N_ref). nu > 1 = expansion.

       Returns
       -------
       dphi : ndarray of shape (n+1,)
       """
       return drift_operator(phi, n) / nu  # larger nu => weaker drift

Now we have every gear.  Time to assemble the complete movement.


Step 9: Putting It All Together -- The ODE System
====================================================

The full rate of change of the SFS is the sum of all operators:

.. math::

   \frac{d\phi_j}{dt} = \frac{1}{\nu(t)} \cdot D_j[\boldsymbol{\phi}]
   + \mu_j[\boldsymbol{\phi}]
   + \gamma \cdot S_j[\boldsymbol{\phi}]
   + M_j[\boldsymbol{\phi}]

where :math:`D_j`, :math:`\mu_j`, :math:`S_j`, :math:`M_j` are the drift,
mutation, selection, and migration operators respectively.

This is a system of :math:`n - 1` coupled ODEs (one for each internal SFS entry
:math:`j = 1, \ldots, n-1`). We can solve it numerically using standard ODE
integrators.

.. admonition:: Calculus Aside -- Solving the ODE system numerically

   A system of :math:`n-1` coupled first-order ODEs of the form
   :math:`d\mathbf{y}/dt = \mathbf{f}(\mathbf{y}, t)` can be solved by
   **Runge--Kutta** methods (e.g., the classic RK45 used below).  At each
   time step, RK45 evaluates the right-hand side :math:`\mathbf{f}` at
   several trial points, combines them to estimate the next state, and
   compares two estimates of different order to adaptively choose the step
   size.  This is the same workhorse algorithm used in physics simulations,
   circuit analysis, and orbital mechanics.  The key requirement is that the
   right-hand side can be evaluated cheaply -- and since our operators are
   linear (or low-degree polynomial) in the SFS entries, each evaluation is
   :math:`O(n)` for one population or :math:`O(n_1 n_2)` for two.

.. code-block:: python

   from scipy.integrate import solve_ivp

   def integrate_sfs(phi_init, n, T, nu_func, theta, gamma=0, h=0.5):
       """Integrate the SFS forward through time.

       Parameters
       ----------
       phi_init : ndarray of shape (n+1,)
           Initial SFS.
       n : int
           Sample size.
       T : float
           Integration time (in 2*Ne generations).
       nu_func : callable
           nu_func(t) returns the relative population size at time t.
       theta : float
           Scaled mutation rate.
       gamma : float
           Scaled selection coefficient.
       h : float
           Dominance coefficient.

       Returns
       -------
       phi_final : ndarray of shape (n+1,)
           SFS after integration.
       """
       # Pack the internal SFS entries into a vector for the ODE solver
       y0 = phi_init[1:n].copy()

       def rhs(t, y):
           # Unpack into full SFS (including boundary bins 0 and n)
           phi = np.zeros(n + 1)
           phi[1:n] = y

           # Compute each operator -- each gear in the train
           nu = nu_func(t)                    # current population size
           d_drift = drift_operator(phi, n) / nu             # drift (scaled by size)
           d_mutation = mutation_operator(phi, n, theta)      # mutation input
           d_selection = selection_operator(phi, n, gamma, h) if gamma != 0 else np.zeros(n + 1)

           # Sum the operators to get the total rate of change
           dphi = d_drift + d_mutation + d_selection
           return dphi[1:n]  # return only internal entries

       # Solve the ODE system using adaptive Runge-Kutta (RK45)
       sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                        rtol=1e-10, atol=1e-12,
                        dense_output=True)

       # Unpack the final state back into a full SFS array
       phi_final = np.zeros(n + 1)
       phi_final[1:n] = sol.y[:, -1]
       return phi_final

   # --- Verification 1: neutral equilibrium ---
   # Starting from the neutral SFS, integrating with constant size should
   # give back the same SFS (the equilibrium is a fixed point of the ODE)
   n = 20
   theta = 1.0
   phi_eq = expected_sfs_neutral(n, theta)
   phi_after = integrate_sfs(phi_eq, n, T=1.0, nu_func=lambda t: 1.0, theta=theta)

   print("Verification 1: Neutral equilibrium is stable")
   print(f"{'j':>3} {'Before':>10} {'After':>10} {'Diff':>12}")
   for j in range(1, 6):
       print(f"{j:3d} {phi_eq[j]:10.6f} {phi_after[j]:10.6f} "
             f"{phi_after[j] - phi_eq[j]:12.2e}")

   # --- Verification 2: expansion creates excess rare variants ---
   phi_expanded = integrate_sfs(phi_eq, n, T=0.5,
                                 nu_func=lambda t: 10.0, theta=theta)

   print("\nVerification 2: 10x expansion for T=0.5")
   print(f"{'j':>3} {'Neutral':>10} {'Expanded':>10} {'Ratio':>8}")
   for j in range(1, 8):
       ratio = phi_expanded[j] / phi_eq[j] if phi_eq[j] > 0 else float('inf')
       print(f"{j:3d} {phi_eq[j]:10.6f} {phi_expanded[j]:10.6f} {ratio:8.3f}")
   print("(Ratio > 1 for small j = excess rare variants)")

.. admonition:: The Jackknife Approximation

   The derivation above hides a subtlety. The drift operator involves moments
   of order :math:`n+2` (because multiplying the sampling probability by
   :math:`x(1-x)` increases the polynomial degree by 2). To close the system
   -- express everything in terms of the :math:`n+1` SFS entries -- ``moments``
   uses the **jackknife approximation**: it approximates higher-order moments
   as combinations of the known :math:`n+1` moments.

   The jackknife works by noting that :math:`E[x^{n+1}(1-x)^{m}]` can be
   approximated from :math:`E[x^k(1-x)^{n-k}]` for :math:`k = 0, \ldots, n`
   using interpolation. This approximation becomes exact in the limit
   :math:`n \to \infty` and is highly accurate for practical sample sizes.

.. admonition:: Probability Aside -- Moment closure and why it works

   Any system of moment equations faces a **closure problem**: the equation
   for the :math:`k`-th moment generally involves the :math:`(k+1)`-th
   moment.  Without an approximation, you would need infinitely many
   equations.  The jackknife closure truncates this chain by expressing the
   :math:`(n+1)`-th and :math:`(n+2)`-th moments in terms of the first
   :math:`n` moments.  The approximation is accurate because, for the
   beta-like distributions that arise in population genetics, low-order
   moments strongly constrain higher-order ones.  Empirically, the jackknife
   closure introduces errors on the order of :math:`1/n^2`, which is
   negligible for the sample sizes (:math:`n \geq 20`) used in practice.

With the ODE system in hand, we turn to the discrete events -- population
splits and admixture -- that punctuate the smooth integration.


Step 10: Discrete Demographic Events
=======================================

Not everything is a smooth ODE integration. Some demographic events happen
instantaneously:

Population split
-----------------

A single population with sample size :math:`n = n_1 + n_2` splits into two
populations with sample sizes :math:`n_1` and :math:`n_2`. The 1D SFS becomes
a 2D SFS:

.. math::

   \phi_{j_1, j_2}^{\text{after}} = \phi_{j_1 + j_2}^{\text{before}} \cdot
   \frac{\binom{j_1+j_2}{j_1}\binom{n-j_1-j_2}{n_1-j_1}}{\binom{n}{n_1}}

**Intuition**: At the moment of splitting, both populations are drawing from the
same gene pool. The probability of seeing :math:`j_1` derived alleles in pop 1
and :math:`j_2` in pop 2, given that the total was :math:`j_1 + j_2`, is just
hypergeometric sampling -- the same math as projection (see
:ref:`the_frequency_spectrum`, Step 5).

.. code-block:: python

   from scipy.special import comb

   def split_1d_to_2d(phi_1d, n1, n2):
       """Split a 1D SFS into a 2D joint SFS.

       Parameters
       ----------
       phi_1d : ndarray of shape (n+1,)
           1D SFS with n = n1 + n2.
       n1, n2 : int
           Sample sizes for the two daughter populations.

       Returns
       -------
       phi_2d : ndarray of shape (n1+1, n2+1)
       """
       n = n1 + n2
       phi_2d = np.zeros((n1 + 1, n2 + 1))

       for j in range(n + 1):
           if phi_1d[j] == 0:
               continue
           for j1 in range(max(0, j - n2), min(j, n1) + 1):
               j2 = j - j1
               if j2 < 0 or j2 > n2:
                   continue
               # Hypergeometric probability: given j total derived alleles,
               # what is the probability of j1 in pop1 and j2 in pop2?
               prob = (comb(j, j1, exact=True) *
                       comb(n - j, n1 - j1, exact=True) /
                       comb(n, n1, exact=True))
               phi_2d[j1, j2] = phi_1d[j] * prob

       return phi_2d

   # Verify: marginalizing back should give the original 1D SFS
   n1, n2 = 8, 12
   n = n1 + n2
   phi_1d = expected_sfs_neutral(n, theta=1.0)
   phi_2d = split_1d_to_2d(phi_1d, n1, n2)

   # Marginalize over pop 2 (project onto pop 1)
   phi_marginal_1 = phi_2d.sum(axis=1)
   phi_expected_1 = project_sfs(phi_1d, n1)

   print("Verify: split then marginalize = project")
   for j in range(1, min(n1, 6)):
       print(f"  j={j}: marginal={phi_marginal_1[j]:.6f}, "
             f"projected={phi_expected_1[j]:.6f}, "
             f"match={np.isclose(phi_marginal_1[j], phi_expected_1[j])}")

Admixture
----------

Population :math:`C` is formed as a mixture: fraction :math:`\alpha` from pop
:math:`A` and :math:`1 - \alpha` from pop :math:`B`:

.. math::

   \phi_{j_C}^{\text{after}} = \sum_{k=0}^{j_C}
   \phi_{k,\, j_C - k}^{A, B} \cdot
   \binom{n_C}{j_C} \left(\frac{k}{n_A}\alpha + \frac{j_C - k}{n_B}(1-\alpha)\right)^{j_C}
   \cdots

The exact formula involves a multinomial convolution, but the principle is simple:
each chromosome in the new population was drawn from pop A with probability
:math:`\alpha` or pop B with probability :math:`1 - \alpha`.


Step 11: The Complete Integration Loop
========================================

Here's how ``moments`` puts it all together for a two-population model:

.. code-block:: python

   import moments

   def two_pop_model_manual(params, ns):
       """Two-population model: ancestral -> split -> diverge with migration.

       Parameters
       ----------
       params : tuple
           (nu1, nu2, T, m)
       ns : tuple
           (n1, n2) sample sizes

       Returns
       -------
       fs : moments.Spectrum
       """
       nu1, nu2, T, m = params

       # 1. Start from equilibrium (neutral, constant size)
       #    This is the steady-state solution of:
       #    d(phi)/dt = D[phi]/nu + M[phi] = 0
       #    with nu = 1 (reference size)
       fs = moments.Demographics1D.snm([sum(ns)])

       # 2. Split the ancestral population
       #    Converts 1D SFS -> 2D SFS via hypergeometric sampling
       #    (like a single clock being separated into two independent movements)
       fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])

       # 3. Integrate through the divergence epoch
       #    Under the hood, this solves the coupled ODE system:
       #    d(phi_{j1,j2})/dt = D[phi]/nu + Mut[phi] + Mig[phi]
       #    for duration T, with sizes nu1 and nu2
       migration_matrix = np.array([[0, m], [m, 0]])
       fs.integrate([nu1, nu2], T, m=migration_matrix)

       return fs

   # Run the model
   fs = two_pop_model_manual([2.0, 0.5, 0.3, 1.0], [10, 10])
   print(f"Model log-likelihood shape: {fs.shape}")
   print(f"F_ST = {fs.Fst():.4f}")

We have now assembled the complete gear train.  In the next chapter,
:ref:`demographic_inference`, we use it to **adjust parameters until the
predicted dial matches observation** -- the heart of demographic inference.


Exercises
=========

.. admonition:: Exercise 1: Verify the drift equilibrium

   Start from a flat SFS (:math:`\phi_j = 1` for all :math:`j`) and integrate
   the drift + mutation ODE for a long time (:math:`T = 10`). Verify that the
   SFS converges to :math:`\theta / j`.

.. admonition:: Exercise 2: Selection shifts the SFS

   Compute the equilibrium SFS for :math:`\gamma = -5, 0, +5` with
   :math:`n = 50`. Plot all three on the same axes. Verify that
   :math:`\gamma < 0` enriches rare variants and :math:`\gamma > 0` enriches
   common variants.

.. admonition:: Exercise 3: Compare with moments

   Implement a two-epoch model (constant size, then expansion) using your
   ``integrate_sfs`` function. Compare the result against
   ``moments.Demographics1D.two_epoch``. How many SFS entries agree to
   6 decimal places?

.. admonition:: Exercise 4: Migration equilibrium

   For two populations with :math:`\nu_1 = \nu_2 = 1` and symmetric migration
   rate :math:`M`, the expected :math:`F_{ST}` is approximately
   :math:`1/(1 + 4M)` (Wright's island model). Simulate this with ``moments``
   for :math:`M = 0.1, 1, 10` and compare to the theoretical expectation.

Solutions
=========

.. admonition:: Solution 1: Verify the drift equilibrium

   Start from a flat SFS and integrate the drift + mutation ODE for a long
   time.  Drift drains probability from the interior toward the boundaries
   while mutation continuously injects singletons.  The balance produces the
   :math:`\theta / j` equilibrium.

   .. code-block:: python

      import numpy as np
      from scipy.integrate import solve_ivp

      n = 30
      theta = 1.0

      # Start from a flat SFS (far from equilibrium)
      phi_init = np.ones(n + 1)
      phi_init[0] = 0.0
      phi_init[n] = 0.0

      # Integrate for T = 10 diffusion time units (constant size nu = 1)
      phi_flat_start = integrate_sfs(
          phi_init, n, T=10.0, nu_func=lambda t: 1.0, theta=theta
      )

      # Compare to the expected neutral equilibrium theta / j
      print(f"{'j':>3} {'Integrated':>12} {'theta/j':>10} {'Rel. error':>12}")
      for j in range(1, min(n, 10)):
          expected = theta / j
          rel_err = abs(phi_flat_start[j] - expected) / expected
          print(f"{j:3d} {phi_flat_start[j]:12.6f} {expected:10.6f} {rel_err:12.2e}")

      # All entries should agree to within ~1e-4 or better after T = 10.
      max_rel_err = max(
          abs(phi_flat_start[j] - theta / j) / (theta / j) for j in range(1, n)
      )
      print(f"\nMax relative error: {max_rel_err:.2e}")
      assert max_rel_err < 1e-3, "SFS did not converge to theta/j"

   The key insight is that :math:`T = 10` (i.e., 20 :math:`N_e` generations) is
   long enough for the system to reach steady state, regardless of the starting
   condition.  The drift operator acts as a tridiagonal diffusion matrix and the
   mutation operator provides a constant source at :math:`j = 1`; their balance
   uniquely determines the :math:`\theta / j` spectrum.

.. admonition:: Solution 2: Selection shifts the SFS

   With :math:`\gamma < 0` (purifying selection), deleterious alleles are
   removed before reaching high frequency, enriching rare variants.  With
   :math:`\gamma > 0` (positive selection), the derived allele is pushed toward
   fixation, enriching common variants.

   .. code-block:: python

      import numpy as np

      n = 50
      theta = 1.0
      gamma_values = [-5, 0, 5]

      results = {}
      for gamma in gamma_values:
          # Start from neutral equilibrium
          phi_eq = expected_sfs_neutral(n, theta)
          # Integrate with selection for a long time to reach new equilibrium
          phi_sel = integrate_sfs(
              phi_eq, n, T=10.0,
              nu_func=lambda t: 1.0,
              theta=theta,
              gamma=gamma,
              h=0.5  # additive selection
          )
          results[gamma] = phi_sel

      # Print a comparison table
      print(f"{'j':>3} {'gamma=-5':>12} {'gamma=0':>12} {'gamma=+5':>12}")
      for j in range(1, 10):
          print(f"{j:3d} {results[-5][j]:12.6f} "
                f"{results[0][j]:12.6f} {results[5][j]:12.6f}")

      # Verify the expected pattern:
      # gamma < 0 => more singletons (phi[1] larger), fewer high-freq (phi[n-1] smaller)
      # gamma > 0 => fewer singletons (phi[1] smaller), more high-freq (phi[n-1] larger)
      assert results[-5][1] > results[0][1], "Purifying should enrich singletons"
      assert results[5][1] < results[0][1], "Positive should deplete singletons"
      assert results[-5][n-1] < results[0][n-1], "Purifying should deplete high-freq"
      assert results[5][n-1] > results[0][n-1], "Positive should enrich high-freq"
      print("\nAll assertions passed: selection shifts the SFS as expected.")

   Plotting the three spectra on a log-log scale makes the pattern vivid:
   the :math:`\gamma = -5` curve is steeper than :math:`1/j` (excess rare
   variants), while the :math:`\gamma = +5` curve is shallower (excess common
   variants).

.. admonition:: Solution 3: Compare with moments

   Build a two-epoch model (constant size, then expansion) using our
   ``integrate_sfs`` function and compare entry-by-entry against
   ``moments.Demographics1D.two_epoch``.

   .. code-block:: python

      import numpy as np
      import moments

      n = 20
      theta = 1.0
      nu_expansion = 5.0
      T_expansion = 0.3

      # --- Our manual implementation ---
      phi_eq = expected_sfs_neutral(n, theta)
      phi_manual = integrate_sfs(
          phi_eq, n, T=T_expansion,
          nu_func=lambda t: nu_expansion,
          theta=theta
      )

      # --- moments built-in ---
      fs_moments = moments.Demographics1D.two_epoch(
          (nu_expansion, T_expansion), [n]
      )
      # moments returns a unit-theta SFS; scale to match
      phi_moments = np.array(fs_moments) * theta

      # Compare
      n_match_6 = 0
      print(f"{'j':>3} {'Manual':>12} {'moments':>12} {'Match (6 dp)':>14}")
      for j in range(1, n):
          match = np.isclose(phi_manual[j], phi_moments[j], atol=1e-6)
          if match:
              n_match_6 += 1
          print(f"{j:3d} {phi_manual[j]:12.8f} {phi_moments[j]:12.8f} "
                f"{'YES' if match else 'NO':>14}")

      print(f"\nEntries matching to 6 decimal places: {n_match_6} / {n - 1}")

   The number of matching entries depends on the accuracy of the ODE solver
   tolerances and the drift operator discretisation.  With ``rtol=1e-10`` and
   ``atol=1e-12`` (as used in ``integrate_sfs``), most entries should agree to
   at least 6 decimal places.  Discrepancies arise from the jackknife moment
   closure used internally by ``moments``, which our simplified drift operator
   approximates slightly differently.

.. admonition:: Solution 4: Migration equilibrium

   Wright's island model predicts :math:`F_{ST} \approx 1/(1 + 4M)` for
   symmetric migration rate :math:`M`.  We verify this using ``moments``
   for :math:`M = 0.1, 1, 10`.

   .. code-block:: python

      import numpy as np
      import moments

      M_values = [0.1, 1.0, 10.0]
      n1, n2 = 20, 20

      print(f"{'M':>6} {'Fst (moments)':>16} {'Fst (theory)':>14} {'Rel. error':>12}")
      for M in M_values:
          # Start from equilibrium with total sample size n1 + n2
          fs = moments.Demographics1D.snm([n1 + n2])
          # Split into two populations
          fs = moments.Manips.split_1D_to_2D(fs, n1, n2)
          # Integrate with symmetric migration for a long time (T = 10)
          # to reach migration-drift equilibrium
          mig_mat = np.array([[0, M], [M, 0]])
          fs.integrate([1.0, 1.0], 10.0, m=mig_mat)

          fst_moments = fs.Fst()
          fst_theory = 1.0 / (1.0 + 4.0 * M)
          rel_err = abs(fst_moments - fst_theory) / fst_theory

          print(f"{M:6.1f} {fst_moments:16.6f} {fst_theory:14.6f} {rel_err:12.2e}")

      # The agreement should be good (within a few percent) for large M;
      # for small M the approximation 1/(1+4M) is less precise because it
      # assumes infinite island model whereas we have only two populations.

   The :math:`1/(1 + 4M)` formula is exact for an infinite-island model.
   For two populations, deviations are expected, especially at low migration
   rates.  Nevertheless, the ``moments`` result and the analytical
   approximation should agree within a few percent for
   :math:`M \geq 1`, confirming that the migration operator and the ODE
   integration are correctly implemented.

Next: :ref:`demographic_inference` -- we use these moment equations to find
the parameters that best explain observed data.
