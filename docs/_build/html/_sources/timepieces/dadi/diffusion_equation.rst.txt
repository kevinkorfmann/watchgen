.. _dadi_diffusion_equation:

========================
The Diffusion Equation
========================

   *The escapement -- the fundamental equation whose ticking drives every hand on the dial.*

In this chapter we derive the Wright-Fisher diffusion equation from first
principles, explain each term biologically, and show how ``dadi`` sets up
the equation for one and multiple populations. By the end, you'll understand
the PDE that ``moments`` sought to bypass -- and why ``dadi`` chose to solve
it directly.

From Wright-Fisher to Diffusion
=================================

The starting point is the Wright-Fisher model: a population of :math:`2N`
gene copies at a single locus, with discrete, non-overlapping generations.
Each generation, the new population is formed by sampling :math:`2N` copies
from the previous generation with replacement (possibly biased by selection).

Let :math:`X_t` be the frequency of the derived allele in generation :math:`t`.
Under neutrality, the change in frequency per generation has mean zero and
variance :math:`X_t(1-X_t)/(2N)`. As :math:`N \to \infty` with appropriate
rescaling of time (:math:`\tau = t/(2N)` generations), the discrete process
converges to a continuous diffusion.

The frequency density :math:`\phi(x, \tau)` -- defined so that
:math:`\phi(x, \tau)dx` is the expected number of segregating sites with
derived allele frequency in :math:`[x, x+dx]` at time :math:`\tau` --
satisfies the **Kolmogorov forward equation** (Fokker-Planck equation):

.. math::

   \frac{\partial \phi}{\partial \tau} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\Big[V(x)\,\phi\Big]
   - \frac{\partial}{\partial x}\Big[M(x)\,\phi\Big]

where :math:`V(x)` is the **diffusion coefficient** (variance of frequency
change per unit time) and :math:`M(x)` is the **drift coefficient** (mean
frequency change per unit time, not to be confused with genetic drift).

.. admonition:: Calculus Aside -- Fokker-Planck equations

   The Fokker-Planck (or Kolmogorov forward) equation describes how the
   probability density of a stochastic process evolves in time. It is the
   continuous analog of a Markov chain transition matrix: given the current
   distribution, it tells you the distribution at the next instant. The
   :math:`\partial^2/\partial x^2` term spreads the density (diffusion from
   genetic drift), while the :math:`\partial/\partial x` term shifts it
   (directional forces like selection).

The 1D Diffusion Equation
==========================

For a single population with relative size :math:`\nu` (measured as a ratio
to the reference :math:`N_{\text{ref}}`), the coefficients are:

**Drift (genetic drift):**

.. math::

   V(x) = \frac{x(1-x)}{\nu}

The variance of allele frequency change is :math:`x(1-x)/(2N)` per generation.
After rescaling time by :math:`2N_{\text{ref}}`, this becomes
:math:`x(1-x)/\nu`. Larger populations (:math:`\nu > 1`) have weaker drift.

**Selection (directional force):**

.. math::

   M(x) = \gamma \, x(1-x)\Big[h + (1-2h)x\Big]

where :math:`\gamma = 2N_{\text{ref}}s` is the population-scaled selection
coefficient and :math:`h` is the dominance coefficient. For genic/additive
selection (:math:`h = 0.5`), this simplifies to :math:`M(x) = \gamma x(1-x)/2`.

.. admonition:: Plain-language summary -- Two forces shaping allele frequencies

   The diffusion equation describes two competing forces acting on allele
   frequencies. **Genetic drift** (the :math:`V(x)` term) is the random
   fluctuation caused by finite population size -- like Brownian motion, it
   pushes allele frequencies up and down unpredictably. Drift is strongest
   at intermediate frequencies (where :math:`x(1-x)` is large) and weakest
   near the boundaries. **Selection** (the :math:`M(x)` term) is a
   directional force: beneficial alleles are pushed toward fixation,
   deleterious alleles toward loss. In smaller populations, drift dominates
   and selection is less effective -- this is why nearly neutral mutations
   can fix in small populations but are efficiently purged in large ones.

Putting these together, the 1D diffusion equation that ``dadi`` solves is:

.. math::
   :label: diffusion_1d

   \frac{\partial \phi}{\partial t} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[\frac{x(1-x)}{\nu}\,\phi\right]
   - \frac{\partial}{\partial x}\Big[\gamma \, x(1-x)(h + (1-2h)x)\,\phi\Big]

where :math:`t` is measured in units of :math:`2N_{\text{ref}}` generations
and :math:`\nu` may change with time (modeling population size changes).

In ``dadi``'s code (``Integration.py``), the one-population solver
``one_pop(phi, xx, T, nu, gamma, h, theta0)`` integrates this equation on a
frequency grid ``xx`` for time ``T``.

Boundary Conditions and Mutation
=================================

The diffusion equation needs boundary conditions at :math:`x = 0` (allele
lost) and :math:`x = 1` (allele fixed). In ``dadi``, these are **absorbing
boundaries**: once an allele is lost or fixed, it leaves the segregating pool.
The density :math:`\phi(x, t)` is defined only for :math:`x \in (0, 1)`.

**Mutation injection:**

New mutations enter the population at low frequency. In ``dadi``, this is
implemented as a source term at the lowest interior grid point. Each timestep,
the integration function ``_inject_mutations_1D`` adds:

.. math::

   \phi(x_1, t + dt) \mathrel{+}= \frac{\theta_0 \cdot dt}{2} \cdot \frac{2}{\Delta x}

where :math:`x_1` is the first interior grid point and :math:`\Delta x` is the
local grid spacing.

**Where does this formula come from?** In the infinite-sites model, new
mutations arise at rate :math:`\theta_0/2 = 2N_{\text{ref}}\mu` per site per
unit of diffusion time (the factor of 2 converts from per-generation to
per-:math:`2N` time units). Each new mutation enters at frequency
:math:`1/(2N)`, which is the lowest representable frequency. Since
:math:`\phi(x)` is a *density* (probability per unit frequency), injecting
one new mutation at :math:`x_1` means adding :math:`1/\Delta x` to the density
at that point (the mutation occupies a bin of width :math:`\Delta x`). Over a
timestep :math:`dt`, the total injection is:

.. math::

   \frac{\theta_0}{2} \cdot dt \cdot \frac{1}{\Delta x / 2}
   = \frac{\theta_0 \cdot dt}{2} \cdot \frac{2}{\Delta x}

The extra factor of :math:`2/\Delta x` (rather than :math:`1/\Delta x`) arises
from the trapezoidal integration scheme: the first grid point is at the boundary
of a half-width bin, so the effective bin width is :math:`\Delta x / 2`.

This is an approximation: new mutations arise at frequency :math:`1/(2N)`,
which is below any finite grid resolution. The injection at :math:`x_1` is
accurate when :math:`x_1 \ll 1`, which ``dadi`` ensures by using a grid that
is dense near the boundaries.

Equilibrium Solution
=====================

Under constant population size (:math:`\nu = 1`), neutrality
(:math:`\gamma = 0`), and steady-state mutation, the diffusion equation has
the equilibrium solution:

.. math::

   \phi_{\text{eq}}(x) = \frac{\theta}{x(1-x)}

This is the classical result: rare alleles (small :math:`x`) are much more
common than common alleles, because drift hasn't had time to push them to high
frequency.

.. admonition:: Biology Aside -- The 1/x spectrum and the neutral theory

   The :math:`1/x` shape of the neutral SFS is one of the most fundamental
   results in population genetics. It says that singletons (variants present
   in just one chromosome) are the most abundant class, doubletons the
   second, and so on. This pattern arises because new mutations enter at low
   frequency and most are lost before reaching high frequency. Deviations
   from this shape are informative: an excess of rare variants (steeper than
   :math:`1/x`) indicates recent population growth; a deficit of rare variants
   (shallower than :math:`1/x`) indicates a bottleneck or population
   contraction. These are exactly the signatures that ``dadi`` uses to infer
   demographic history. In ``dadi``, this equilibrium is computed by
   ``PhiManip.phi_1D(xx, nu=1.0, theta0=1.0)``, which evaluates
   :math:`\nu \cdot \theta_0 / x` at each grid point (using the convention that
   the density near :math:`x = 0` dominates).

With genic selection (:math:`h = 0.5`, :math:`\gamma \neq 0`), the
equilibrium density can be derived from the stationary Fokker-Planck equation.
Setting :math:`\partial\phi/\partial t = 0` with :math:`V(x) = x(1-x)` and
:math:`M(x) = (\gamma/2) x(1-x)` gives:

.. math::

   0 = \frac{1}{2}\frac{d^2}{dx^2}[x(1-x)\phi]
   - \frac{d}{dx}\left[\frac{\gamma}{2}x(1-x)\phi\right]

The general stationary solution of the Fokker-Planck equation with drift
:math:`M(x)` and diffusion :math:`V(x) = x(1-x)` is:

.. math::

   \phi_{\text{eq}}(x) = \frac{C}{V(x)} \exp\left(2\int_0^x \frac{M(y)}{V(y)} dy\right)
   = \frac{C}{x(1-x)} e^{\gamma x}

since :math:`2 \int_0^x \frac{\gamma y(1-y)/2}{y(1-y)} dy = \gamma x`. This is
the standard result from the :ref:`diffusion prerequisite <diffusion_approximation>`.
In the infinite-sites framework (mutations entering at :math:`x \approx 0` and
leaving at :math:`x = 0` or :math:`x = 1`), the constant :math:`C` is determined
by matching the mutation injection rate, giving:

.. math::

   \phi_{\text{eq}}(x) \propto \frac{1 - e^{-2\gamma(1-x)}}{x(1-x)(1 - e^{-2\gamma})}

The numerator :math:`1 - e^{-2\gamma(1-x)}` arises from the boundary condition
at :math:`x = 1` (fixation probability under selection). When :math:`\gamma = 0`,
this simplifies to :math:`1/x(1-x)` -- the neutral equilibrium.

This is computed by ``PhiManip.phi_1D_genic(xx, nu, theta0, gamma)``.

.. code-block:: python

   import dadi
   import numpy as np

   # Build a frequency grid
   pts = 60
   xx = dadi.Numerics.default_grid(pts)

   # Equilibrium density under the standard neutral model
   phi = dadi.PhiManip.phi_1D(xx)

   # The equilibrium density is proportional to 1/x
   # (ignoring the x -> 1 boundary)
   print(f"phi at x={xx[1]:.4f}: {phi[1]:.2f}")
   print(f"phi at x={xx[5]:.4f}: {phi[5]:.2f}")
   print(f"Ratio: {phi[1]/phi[5]:.2f}, expected: {xx[5]/xx[1]:.2f}")

The Multi-Population PDE
==========================

For :math:`P` populations, the allele frequency density becomes
:math:`P`-dimensional: :math:`\phi(x_1, x_2, \ldots, x_P, t)`, where
:math:`x_i` is the derived allele frequency in population :math:`i`. The PDE
generalizes to:

.. math::

   \frac{\partial \phi}{\partial t} = \sum_{i=1}^{P}
   \left[
   \frac{1}{2}\frac{\partial^2}{\partial x_i^2}\left[\frac{x_i(1-x_i)}{\nu_i}\,\phi\right]
   - \frac{\partial}{\partial x_i}\Big[\gamma_i \, x_i(1-x_i)(h_i + (1-2h_i)x_i)\,\phi\Big]
   \right]
   + \text{migration terms}

Each population contributes its own drift and selection terms, acting along
its own frequency axis. The populations are coupled through **migration**:

.. math::

   \text{Migration } i \leftarrow j: \quad
   -\frac{\partial}{\partial x_i}\Big[m_{ij}(x_j - x_i)\,\phi\Big]

where :math:`m_{ij}` is the scaled migration rate from population :math:`j`
into population :math:`i`. Migration shifts the frequency in population
:math:`i` toward the frequency in population :math:`j`.

.. admonition:: Biology Aside -- Migration homogenizes allele frequencies

   Gene flow (migration) between populations acts to make their allele
   frequencies more similar. If a variant is at 80% frequency in population A
   and 20% in population B, migration from A to B pushes B's frequency
   upward (and vice versa). Over time, high migration makes two populations
   genetically indistinguishable, while low migration allows them to
   diverge. The migration term in the PDE captures this: it is a directional
   force that pulls each population's frequency toward the other's. The
   scaled migration rate :math:`m_{ij} = 4 N_\text{ref} \cdot m` determines
   the strength of this pull -- values greater than 1 indicate enough gene
   flow to prevent substantial divergence, while values much less than 1
   allow populations to drift apart.

In ``dadi``'s code, ``Integration.two_pops(phi, xx, T, nu1, nu2, m12, m21, ...)``
solves the 2D PDE, and ``Integration.three_pops`` solves the 3D version. The
curse of dimensionality limits ``dadi`` to at most five populations, and in
practice three is the common maximum.

Population Splits via PhiManip
================================

When a population splits, the allele frequency distribution is duplicated: the
new daughter population starts with the same allele frequencies as the parent.
In ``dadi``, this is handled by ``PhiManip.phi_1D_to_2D(xx, phi_1D)`` and its
higher-dimensional analogs.

The operation is conceptually simple: place all the density on the diagonal
:math:`x_1 = x_2` of the 2D grid. In code:

.. code-block:: python

   # A population splits into two identical daughter populations
   phi_1D = dadi.PhiManip.phi_1D(xx)
   phi_2D = dadi.PhiManip.phi_1D_to_2D(xx, phi_1D)
   # phi_2D[i, j] is nonzero only when i == j (the diagonal)

After the split, the two populations evolve independently (if there's no
migration) or remain coupled through migration terms. Each daughter population
can have its own size :math:`\nu_i`, selection coefficient :math:`\gamma_i`,
and mutation rate.

Admixture
==========

Admixture -- the mixing of two populations to form a new one -- is implemented
as a remapping of frequencies. If population 3 is formed as a fraction
:math:`f` from population 1 and :math:`(1-f)` from population 2, then the
frequency in the admixed population is:

.. math::

   x_3 = f \cdot x_1 + (1-f) \cdot x_2

``dadi`` implements this via ``PhiManip.phi_2D_to_3D_admix(phi_2D, f, xx, yy, zz)``,
which creates the 3D density by interpolating the 2D density onto the new
frequency axis at each :math:`(x_1, x_2)` pair. Internally, the function uses
linear interpolation between grid points and careful normalization to preserve
the total density.

A Complete Example
===================

Here's how these pieces connect for a two-epoch model -- the simplest
non-trivial demographic scenario:

.. code-block:: python

   import dadi

   def two_epoch(params, ns, pts):
       """
       A population changes size instantaneously.

       params = (nu, T)
           nu: ratio of contemporary to ancient size
           T:  time of size change (in 2*Nref generations)
       """
       nu, T = params

       # Build frequency grid
       xx = dadi.Numerics.default_grid(pts)

       # Start from equilibrium (standard neutral model)
       phi = dadi.PhiManip.phi_1D(xx)

       # Integrate the diffusion equation for time T
       # with population size nu (relative to Nref)
       phi = dadi.Integration.one_pop(phi, xx, T, nu=nu)

       # Extract the SFS from the frequency density
       fs = dadi.Spectrum.from_phi(phi, ns, (xx,))
       return fs

   # Wrap with Richardson extrapolation
   func_ex = dadi.Numerics.make_extrap_log_func(two_epoch)

   # Compute expected SFS for nu=2 (doubling), T=0.1
   # with sample size 20, using grid sizes [40, 50, 60]
   model = func_ex((2.0, 0.1), ns=[20], pts=[40, 50, 60])

The call chain is:

1. ``phi_1D`` computes the equilibrium density :math:`\phi_{\text{eq}}(x)`
2. ``one_pop`` solves the PDE :eq:`diffusion_1d` for time :math:`T` with
   size :math:`\nu`
3. ``from_phi`` converts the continuous density to a discrete SFS via
   binomial sampling
4. ``make_extrap_log_func`` runs the whole thing at grid sizes 40, 50, and 60,
   then extrapolates to the :math:`n \to \infty` limit

In the next chapter, we'll open the gear train and examine exactly how
``dadi`` discretizes and solves this PDE numerically.
