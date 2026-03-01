.. _dadi_numerical_integration:

=======================
Numerical Integration
=======================

   *The gear train -- the engineering that turns a continuous equation into a computable answer.*

The diffusion equation is a continuous PDE defined on :math:`x \in (0, 1)` and
:math:`t \in [0, T]`. To solve it numerically, ``dadi`` must discretize
both the frequency axis and the time axis. This chapter examines the
engineering choices that make this discretization accurate and efficient:
the nonuniform grid, the finite-difference scheme, the time-stepping strategy,
and Richardson extrapolation.

.. admonition:: Biology Aside -- Why the grid must be fine near the boundaries

   In a real population, most genetic variants are rare -- they exist in only
   one or a few copies out of thousands of chromosomes (the "singletons" and
   "doubletons" that dominate the SFS). These low-frequency variants are the
   most informative about recent population size changes: a population
   expansion produces a flood of new rare variants, while a bottleneck
   depletes them. At the same time, a small number of variants are at very
   high frequency (nearly fixed). The allele frequency density
   :math:`\phi(x)` is therefore strongly peaked near :math:`x = 0` and
   :math:`x = 1`, reflecting the biological reality that rare and near-fixed
   variants are far more numerous than common ones. A grid that fails to
   resolve these peaks will produce inaccurate SFS predictions and,
   consequently, biased demographic estimates.

The Nonuniform Grid
=====================

The allele frequency density :math:`\phi(x)` is concentrated near the
boundaries: under neutrality, :math:`\phi(x) \propto 1/[x(1-x)]`, which
diverges as :math:`x \to 0` and :math:`x \to 1`. A uniform grid would waste
points in the interior and lack resolution where it matters most.

``dadi`` uses an **exponential grid** (``Numerics.default_grid``) that places
more points near the boundaries:

.. math::

   x_i = \frac{1}{1 + e^{-c \cdot u_i}}

where :math:`u_i` are uniformly spaced on :math:`[-1, 1]` and :math:`c`
(the "crowding" parameter, default 8) controls how strongly points cluster
at the boundaries. The grid is then rescaled so that :math:`x_1 > 0` and
:math:`x_n < 1`.

.. code-block:: python

   import dadi
   import numpy as np

   xx = dadi.Numerics.default_grid(40)

   # Grid spacing near boundaries vs. interior
   print(f"First spacing:  {xx[1] - xx[0]:.6f}")
   print(f"Middle spacing: {xx[20] - xx[19]:.6f}")
   print(f"Last spacing:   {xx[-1] - xx[-2]:.6f}")
   # First and last spacings are much smaller than the middle

The crowding parameter :math:`c` can be tuned. ``dadi`` provides
``Numerics.estimate_best_exp_grid_crwd(ns)`` to estimate an optimal crowding
based on the sample size, balancing resolution near the boundaries against
coverage of the interior.

The Finite-Difference Scheme
==============================

With the frequency axis discretized onto grid points :math:`x_1, \ldots, x_n`,
the PDE becomes a system of coupled ODEs -- one for each grid point. ``dadi``
discretizes the spatial derivatives using a finite-difference scheme.

Consider the 1D diffusion equation:

.. math::

   \frac{\partial \phi_i}{\partial t} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[\frac{x(1-x)}{\nu}\phi\right]_i
   - \frac{\partial}{\partial x}\left[\gamma x(1-x)(h + (1-2h)x)\phi\right]_i

The second derivative term (drift) and first derivative term (selection) are
approximated using the values at neighboring grid points. On a nonuniform grid
with spacings :math:`\Delta x_i = x_{i+1} - x_i`, ``dadi`` computes a
**difference factor**:

.. math::

   d_i = \frac{2}{\Delta x_{i-1} + \Delta x_i}

which corrects for the variable grid spacing. The resulting scheme produces a
**tridiagonal system**: each grid point :math:`i` depends only on its neighbors
:math:`i-1`, :math:`i`, and :math:`i+1`.

.. admonition:: Numerical Analysis Aside -- Why tridiagonal?

   The diffusion equation involves at most second-order spatial derivatives.
   A finite-difference approximation of a second derivative uses three points
   (the central point and its two neighbors), producing a tridiagonal matrix.
   Tridiagonal systems can be solved in :math:`O(n)` time using the Thomas
   algorithm, making each timestep very efficient.

Implicit Time-Stepping
========================

After spatial discretization, the system is:

.. math::

   \frac{d\boldsymbol{\phi}}{dt} = A(t) \, \boldsymbol{\phi}

where :math:`A` is the tridiagonal matrix encoding drift, selection, and
mutation, and :math:`\boldsymbol{\phi}` is the vector of density values at
the grid points.

``dadi`` uses an **implicit (backward Euler)** time-stepping scheme:

.. math::

   \boldsymbol{\phi}^{n+1} = (I - \Delta t \, A)^{-1} \boldsymbol{\phi}^n

This requires solving a tridiagonal linear system at each timestep, which
costs :math:`O(n)`. The implicit scheme is **unconditionally stable** -- it
doesn't blow up regardless of the timestep -- unlike an explicit scheme which
would require :math:`\Delta t < O(\Delta x^2)`.

The timestep :math:`\Delta t` is computed adaptively by
``Integration._compute_dt``, which sets:

.. math::

   \Delta t = \frac{\texttt{timescale\_factor}}{\max_i |V(x_i)| + |M(x_i)|}

where ``timescale_factor`` (default :math:`10^{-3}`) controls accuracy. The
timestep is small where the drift and selection coefficients are large (near
the center of the frequency range, where :math:`x(1-x)` is maximal).

1D Integration Walkthrough
============================

Here's the complete sequence for ``Integration.one_pop(phi, xx, T, nu, gamma, h, theta0)``:

1. **Initialize:** receive the density ``phi`` on grid ``xx``, with target
   integration time ``T``

2. **Build coefficients:** compute the drift coefficient
   :math:`V(x_i) = x_i(1-x_i)/\nu` and selection coefficient
   :math:`M(x_i) = \gamma \, x_i(1-x_i)(h + (1-2h)x_i)` at each grid point

3. **Compute timestep:** :math:`\Delta t` from the maximum coefficient value

4. **Time-stepping loop:** for each step :math:`t \to t + \Delta t`:

   a. Build the tridiagonal matrix :math:`I - \Delta t \cdot A`
   b. Solve the tridiagonal system to get :math:`\phi^{n+1}`
   c. Inject mutations: add :math:`\theta_0 \cdot \Delta t / (2 \cdot \Delta x_1)`
      at the first interior grid point
   d. If :math:`\nu` changes with time, update the coefficients

5. **Return:** the density ``phi`` at time :math:`T`

When parameters are constant over the integration interval, ``dadi`` uses an
optimized path (``_one_pop_const_params``) that pre-computes the tridiagonal
matrix once and reuses it for all timesteps.

.. code-block:: python

   import dadi

   pts = 60
   xx = dadi.Numerics.default_grid(pts)

   # Start from equilibrium
   phi = dadi.PhiManip.phi_1D(xx)

   # Integrate for T=0.5 with doubled population size
   phi_evolved = dadi.Integration.one_pop(phi, xx, T=0.5, nu=2.0)

   # The density has spread out (weaker drift in larger population)

Multi-Dimensional Integration
===============================

For two populations, the density :math:`\phi(x, y)` lives on a 2D grid of
size :math:`n \times n`. The PDE has drift and selection terms acting along
each axis independently, plus migration terms that couple the two axes.

``dadi`` uses **alternating direction implicit (ADI)** integration: at each
timestep, it first sweeps along the :math:`x`-axis (treating :math:`y` as
fixed), then along the :math:`y`-axis (treating :math:`x` as fixed). Each
sweep requires solving :math:`n` independent tridiagonal systems -- one for
each row or column of the grid.

.. math::

   \phi^{n+1/2} &= (I - \Delta t \, A_x)^{-1} \, \phi^n \\
   \phi^{n+1} &= (I - \Delta t \, A_y)^{-1} \, \phi^{n+1/2}

For three populations, the density is 3D (:math:`n^3` grid points) and three
ADI sweeps are needed per timestep. The cost per timestep scales as
:math:`O(n^P)` where :math:`P` is the number of populations -- this is the
**curse of dimensionality** that limits ``dadi`` to a few populations.

.. admonition:: Biology Aside -- The curse of dimensionality in population genetics

   Each additional population in a demographic model adds a new axis to the
   allele frequency density, and the computational cost grows exponentially.
   This is why ``dadi`` is typically limited to 3 populations (and sometimes
   up to 5 with heroic effort). For studies involving many populations --
   such as the global human population tree with dozens of branches -- this
   is the fundamental limitation that motivated ``moments`` (which avoids
   the grid altogether by working with moments of the SFS) and ``momi2``
   (which uses tensors that scale more gracefully with the number of
   populations). The choice among these three tools often comes down to how
   many populations are in the model: ``dadi`` excels for 1--3 populations,
   ``moments`` for 3--5, and ``momi2`` for 5 or more.

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Populations
     - Grid size
     - Function
     - Cost per step
   * - 1
     - :math:`n`
     - ``one_pop``
     - :math:`O(n)`
   * - 2
     - :math:`n^2`
     - ``two_pops``
     - :math:`O(n^2)`
   * - 3
     - :math:`n^3`
     - ``three_pops``
     - :math:`O(n^3)`

Richardson Extrapolation
==========================

The finite-difference scheme introduces **discretization error** that decreases
as the grid becomes finer. For a grid with :math:`n` points, the error in the
SFS is approximately :math:`O(1/n^2)` (for a second-order scheme). Rather than
using a very fine grid (expensive), ``dadi`` uses **Richardson extrapolation**
to cancel the leading-order error.

The idea: run the computation at multiple grid sizes (e.g., :math:`n_1 = 40`,
:math:`n_2 = 50`, :math:`n_3 = 60`), then extrapolate to the
:math:`n \to \infty` limit. If the error is polynomial in :math:`1/n`, a
polynomial fit through the results eliminates the leading error terms.

``dadi`` provides this via ``Numerics.make_extrap_func``:

.. code-block:: python

   import dadi

   def my_model(params, ns, pts):
       xx = dadi.Numerics.default_grid(pts)
       phi = dadi.PhiManip.phi_1D(xx)
       nu, T = params
       phi = dadi.Integration.one_pop(phi, xx, T, nu=nu)
       return dadi.Spectrum.from_phi(phi, ns, (xx,))

   # Wrap with extrapolation
   func_ex = dadi.Numerics.make_extrap_log_func(my_model)

   # Run at 3 grid sizes and extrapolate
   model = func_ex((2.0, 0.1), ns=[20], pts=[40, 50, 60])

Internally, ``make_extrap_func`` runs the model function at each grid size in
``pts``, then applies polynomial extrapolation (linear, quadratic, or higher)
in the variable :math:`1/n` to estimate the :math:`n \to \infty` limit. The
``make_extrap_log_func`` variant extrapolates in log-space, which is more
robust when SFS entries span many orders of magnitude.

The extrapolation functions available are:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Points needed
     - Error eliminated
   * - ``linear_extrap``
     - 2
     - :math:`O(1/n^2)`
   * - ``quadratic_extrap``
     - 3
     - :math:`O(1/n^2)` and :math:`O(1/n^4)`
   * - ``cubic_extrap``
     - 4
     - Up to :math:`O(1/n^6)`

The default ``make_extrap_func`` uses ``quadratic_extrap`` with three grid
sizes -- the standard recommendation for ``dadi`` analyses.

.. admonition:: Numerical Analysis Aside -- Richardson extrapolation

   Richardson extrapolation is a general technique: if a quantity :math:`f(h)`
   has an error expansion :math:`f(h) = f_0 + a h^2 + b h^4 + \cdots`, then
   evaluating at two step sizes :math:`h_1, h_2` and taking the appropriate
   linear combination eliminates the :math:`h^2` term. With three evaluations,
   both the :math:`h^2` and :math:`h^4` terms can be eliminated. ``dadi``
   applies this with :math:`h = 1/n` (the grid spacing), effectively
   canceling the finite-difference bias without requiring an impractically
   fine grid.

From :math:`\phi` to SFS
===========================

After solving the PDE, ``dadi`` must convert the continuous density
:math:`\phi(x)` into the discrete SFS. This is done by ``Spectrum.from_phi``.

The expected number of sites where :math:`j` out of :math:`n` sampled
chromosomes carry the derived allele is:

.. math::

   F_j = \int_0^1 \binom{n}{j} x^j (1-x)^{n-j} \, \phi(x) \, dx

This is a weighted integral of the frequency density against the **binomial
sampling probability** -- the probability that a site with population
frequency :math:`x` would show count :math:`j` in a sample of :math:`n`.

``dadi`` evaluates this integral numerically using the trapezoidal rule on the
grid points. For each SFS entry :math:`j`, the integrand is the product of the
density ``phi`` and the binomial coefficient evaluated at each grid point.

.. code-block:: python

   import dadi

   pts = 60
   xx = dadi.Numerics.default_grid(pts)
   phi = dadi.PhiManip.phi_1D(xx)

   # Extract SFS for sample size n=20
   fs = dadi.Spectrum.from_phi(phi, ns=(20,), xxs=(xx,))

   # fs[j] = expected number of sites with j derived alleles
   print(f"Singletons (j=1): {fs[1]:.4f}")
   print(f"Doubletons (j=2): {fs[2]:.4f}")
   print(f"Under neutrality, fs[j] ~ theta/j")

For multi-population spectra, the integral extends over all frequency axes:

.. math::

   F_{j_1, j_2} = \int_0^1 \int_0^1
   \binom{n_1}{j_1} x^{j_1}(1-x)^{n_1-j_1}
   \binom{n_2}{j_2} y^{j_2}(1-y)^{n_2-j_2}
   \, \phi(x, y) \, dx \, dy

The 2D integral is evaluated by the trapezoidal rule on the 2D grid. For
efficient computation, ``dadi`` uses a linear algebra formulation
(``_from_phi_2D_linalg``) that avoids explicit double loops.

Practical Grid Guidelines
===========================

Choosing the right grid size is a trade-off between accuracy and speed:

- **Too few points** (:math:`< 30`): significant discretization error, even
  with extrapolation
- **Standard** (40, 50, 60): the recommended trio for Richardson extrapolation
  in most analyses
- **Large sample sizes** (:math:`n > 100`): may need pts :math:`\geq n + 10`
  to avoid aliasing in the ``from_phi`` integration
- **Selection** (:math:`|\gamma| > 10`): stronger selection creates steeper
  density profiles, requiring finer grids

The rule of thumb: ``pts`` should be at least 10 larger than the largest sample
size, and extrapolation should always be used for publication-quality results.

.. admonition:: Plain-language summary -- What Richardson extrapolation does for you

   Richardson extrapolation is a clever trick: run the computation three
   times on grids of increasing fineness (e.g., 40, 50, and 60 points), then
   mathematically combine the three answers to cancel out most of the error
   from the finite grid. The result is an SFS estimate that is far more
   accurate than any of the three individual calculations -- roughly as
   accurate as if you had used hundreds of grid points, but at a fraction of
   the cost. For biologists, this means you can trust that the predicted SFS
   (and therefore the inferred demographic history) is not an artifact of the
   grid resolution.

In the next chapter, we'll examine how ``dadi`` uses the computed SFS to
perform demographic inference through composite likelihood optimization.
