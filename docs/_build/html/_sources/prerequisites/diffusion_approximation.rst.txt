.. _diffusion_approximation:

============================
The Diffusion Approximation
============================

   *"When the gear teeth are fine enough, the ticking becomes a smooth sweep."*

The Big Idea
============

In the :ref:`coalescent_theory` chapter, we built the Wright-Fisher model: a
discrete population of :math:`2N` gene copies, evolving in discrete generations,
where allele frequencies jump around in integer multiples of :math:`1/(2N)`. This
model is exact, but when :math:`N` is large, the jumps are tiny, the generations
are many, and keeping track of individual ticks becomes both unnecessary and
computationally expensive.

The **diffusion approximation** replaces this discrete gear train with a smooth
sweep. Instead of tracking allele frequencies that hop in discrete steps of
:math:`1/(2N)`, we model frequency as a **continuous** variable :math:`x \in [0,1]`
that drifts and fluctuates according to a **stochastic differential equation** (SDE).
Instead of counting individual generations, we let time flow continuously. The
result is a mathematical framework -- partial differential equations, stationary
distributions, numerical grids -- that is both more elegant and more powerful than
the discrete model it replaces.

Think of a mechanical watch. A quartz watch ticks 32,768 times per second -- far
too fast for the eye to resolve. The second hand appears to sweep smoothly, even
though the underlying mechanism is discrete. The diffusion approximation does the
same thing for population genetics: when :math:`N` is in the thousands or millions,
the discrete ticks of the Wright-Fisher model blur into a continuous flow, and we
gain access to the entire toolkit of calculus -- derivatives, integrals, partial
differential equations -- to describe what happens.

This chapter builds the diffusion machinery from the ground up. We start with the
Wright-Fisher model you already know, derive the continuous limit, and arrive at
the **Fokker-Planck equation** -- the PDE at the heart of tools like
:ref:`dadi <dadi_timepiece>`. We then solve it numerically, connect it to the site
frequency spectrum, and preview how :ref:`moments <moments_timepiece>` sidesteps
the PDE entirely.


From Wright-Fisher to Continuous Frequency
===========================================

Let us begin where the :ref:`coalescent_theory` chapter left off: the Wright-Fisher
model. Consider a single biallelic locus in a diploid population of size :math:`N`.
There are :math:`2N` gene copies, and the allele frequency :math:`X` is the fraction
of copies that carry the derived allele. In one generation, the number of derived
copies in the next generation is drawn from a binomial distribution:

.. math::

   X' \sim \frac{1}{2N}\text{Binom}(2N, x)

where :math:`x` is the current allele frequency. The change in frequency is
:math:`\Delta x = X' - x`.

Mean and variance of :math:`\Delta x`
---------------------------------------

Under neutrality (no selection, no mutation, no migration), the mean change in
allele frequency is zero:

.. math::

   \mathbb{E}[\Delta x] = 0

This follows because :math:`\mathbb{E}[X'] = x` for a binomial with success
probability :math:`x`. Allele frequency has no preferred direction -- it is a
**martingale**.

The variance of the change is:

.. math::

   \text{Var}(\Delta x) = \frac{x(1-x)}{2N}

This comes from the binomial variance: :math:`\text{Var}(\text{Binom}(2N,x)) =
2N \cdot x(1-x)`, and dividing by :math:`(2N)^2` to convert from counts to
frequencies. Notice that the variance is largest when :math:`x = 1/2` (maximum
uncertainty) and vanishes at :math:`x = 0` or :math:`x = 1` (fixed alleles
cannot drift).

.. admonition:: Probability Aside -- Binomial variance

   If :math:`Y \sim \text{Binom}(n, p)`, then :math:`\text{Var}(Y) = np(1-p)`.
   The frequency :math:`X' = Y/(2N)` therefore has variance
   :math:`\text{Var}(X') = \text{Var}(Y)/(2N)^2 = p(1-p)/(2N)`. Since
   :math:`\Delta x = X' - x` and :math:`x` is a constant (conditioning on the
   current generation), :math:`\text{Var}(\Delta x) = \text{Var}(X') = x(1-x)/(2N)`.

The diffusion timescale
------------------------

The variance :math:`x(1-x)/(2N)` is tiny when :math:`N` is large: meaningful
changes in allele frequency accumulate only over many generations. To obtain a
useful continuous-time limit, we rescale time by defining:

.. math::

   \tau = \frac{t}{2N}

where :math:`t` counts discrete generations and :math:`\tau` is **diffusion time**
(the same coalescent time units from the :ref:`coalescent_theory` chapter). In one
unit of diffusion time, :math:`2N` generations pass, and the variance of the
accumulated frequency change is:

.. math::

   \text{Var}\left(\sum_{i=1}^{2N} \Delta x_i\right) \approx 2N \cdot \frac{x(1-x)}{2N} = x(1-x)

The factor of :math:`2N` from summing independent increments cancels the
:math:`1/(2N)` in each increment's variance, leaving a variance of order 1 in
diffusion time -- exactly what we need for a nontrivial continuous limit.

Code: WF trajectories converging to SDE paths
------------------------------------------------

Let us see this convergence in action. As :math:`N` grows, discrete Wright-Fisher
trajectories increasingly resemble continuous diffusion paths.

.. code-block:: python

   import numpy as np

   def wright_fisher_trajectory(two_N, x0, n_generations):
       """Simulate a Wright-Fisher allele frequency trajectory.

       Parameters
       ----------
       two_N : int
           Total number of gene copies (2N).
       x0 : float
           Initial allele frequency.
       n_generations : int
           Number of generations to simulate.

       Returns
       -------
       freqs : ndarray of shape (n_generations + 1,)
           Allele frequency at each generation.
       """
       freqs = np.zeros(n_generations + 1)
       freqs[0] = x0
       for g in range(n_generations):
           # np.random.binomial(n, p) draws one sample from Binom(n, p).
           # We divide by two_N to convert counts to frequency.
           count = np.random.binomial(two_N, freqs[g])
           freqs[g + 1] = count / two_N
       return freqs

   np.random.seed(42)

   # Simulate for different population sizes, all for 1 diffusion time unit
   x0 = 0.3
   for two_N in [20, 200, 2000]:
       n_gen = two_N  # 1 unit of diffusion time = 2N generations
       traj = wright_fisher_trajectory(two_N, x0, n_gen)
       # Convert generations to diffusion time
       diffusion_times = np.arange(n_gen + 1) / two_N
       final_freq = traj[-1]
       step_size = 1.0 / two_N
       print(f"2N={two_N:5d}: final freq={final_freq:.4f}, "
             f"step size={step_size:.6f}, "
             f"num steps={n_gen}")

   # Verification: variance of frequency change per diffusion time unit
   # should approach x(1-x)
   two_N = 10000
   n_replicates = 20000
   changes = np.zeros(n_replicates)
   for rep in range(n_replicates):
       traj = wright_fisher_trajectory(two_N, x0, two_N)
       changes[rep] = traj[-1] - traj[0]
   print(f"\nVariance of Delta x over 1 diffusion time unit:")
   print(f"  Simulated: {changes.var():.4f}")
   print(f"  Theory x(1-x) = {x0 * (1 - x0):.4f}")


Stochastic Differential Equations
===================================

The continuous-time limit of the Wright-Fisher model is a **stochastic
differential equation** (SDE). In the simplest neutral case:

.. math::

   dx = \sqrt{x(1-x)} \, dW

where :math:`dW` is the increment of a **Wiener process** (Brownian motion). This
is the **Wright-Fisher diffusion**. The square root term :math:`\sqrt{x(1-x)}`
ensures that drift vanishes at the boundaries :math:`x = 0` and :math:`x = 1`,
just as in the discrete model.

.. admonition:: Probability Aside -- What is a Wiener process?

   A **Wiener process** (also called **Brownian motion**) :math:`W(t)` is the
   most fundamental continuous-time random process. Its key properties are:

   1. :math:`W(0) = 0`
   2. Increments :math:`W(t+s) - W(t) \sim \text{Normal}(0, s)` -- the change
      over an interval of length :math:`s` is normally distributed with mean 0
      and variance :math:`s`
   3. Increments over non-overlapping intervals are independent
   4. Paths are continuous (no jumps)

   The Wiener process is the continuous analogue of a random walk: at each
   infinitesimal instant, it takes a tiny random step. The cumulative effect of
   infinitely many infinitesimal steps produces a continuous but extremely
   jagged path -- differentiable nowhere, yet continuous everywhere.

   In our SDE, :math:`dW` represents an infinitesimal "kick" of size
   :math:`\sqrt{dt}`. Multiplying by :math:`\sqrt{x(1-x)}` scales this kick
   by the local volatility of allele frequency.

More generally, with selection, mutation, and migration, the SDE takes the form:

.. math::

   dx = \underbrace{\mu(x)}_{\text{drift}} \, dt + \underbrace{\sigma(x)}_{\text{diffusion}} \, dW

where the **drift coefficient** :math:`\mu(x)` and **diffusion coefficient**
:math:`\sigma(x)` encode the different evolutionary forces:

.. math::

   \mu(x) &= \underbrace{\frac{s}{2} x(1-x)}_{\text{selection}} +
              \underbrace{\frac{\theta_1}{2}(1-x) - \frac{\theta_2}{2} x}_{\text{mutation}} +
              \underbrace{m(x_{\text{source}} - x)}_{\text{migration}} \\
   \sigma(x) &= \sqrt{x(1-x)}

Here :math:`s` is the selection coefficient (positive favors the derived allele),
:math:`\theta_1` and :math:`\theta_2` are forward and backward mutation rates
scaled by :math:`4N` (i.e., :math:`\theta = 4N\mu`), :math:`m` is the migration rate, and :math:`x_{\text{source}}`
is the frequency in the source population. The diffusion coefficient
:math:`\sigma^2(x) = x(1-x)` captures **genetic drift** and is always the same
regardless of selection or mutation.

Euler-Maruyama simulation
--------------------------

The **Euler-Maruyama method** is the simplest way to simulate an SDE. It replaces
the continuous increments with discrete steps of size :math:`\Delta t`:

.. math::

   x_{n+1} = x_n + \mu(x_n)\Delta t + \sigma(x_n) \sqrt{\Delta t} \, Z_n

where :math:`Z_n \sim \text{Normal}(0, 1)` are independent standard normal random
variables.

.. code-block:: python

   def euler_maruyama(x0, mu_func, sigma_func, T, n_steps):
       """Simulate an SDE using the Euler-Maruyama method.

       Parameters
       ----------
       x0 : float
           Initial condition.
       mu_func : callable
           Drift coefficient mu(x).
       sigma_func : callable
           Diffusion coefficient sigma(x).
       T : float
           Total time to simulate (in diffusion time units).
       n_steps : int
           Number of discrete time steps.

       Returns
       -------
       times : ndarray of shape (n_steps + 1,)
           Time points.
       x : ndarray of shape (n_steps + 1,)
           Simulated trajectory.
       """
       dt = T / n_steps
       sqrt_dt = np.sqrt(dt)
       times = np.linspace(0, T, n_steps + 1)
       x = np.zeros(n_steps + 1)
       x[0] = x0

       for n in range(n_steps):
           # np.random.randn() draws one sample from Normal(0, 1)
           Z = np.random.randn()
           x[n + 1] = x[n] + mu_func(x[n]) * dt + sigma_func(x[n]) * sqrt_dt * Z
           # Reflect at boundaries to keep x in [0, 1]
           x[n + 1] = np.clip(x[n + 1], 0.0, 1.0)

       return times, x

   # Neutral diffusion: mu(x) = 0, sigma(x) = sqrt(x(1-x))
   mu_neutral = lambda x: 0.0
   sigma_drift = lambda x: np.sqrt(max(x * (1 - x), 0.0))

   # Selection: mu(x) = (s/2)*x*(1-x), sigma(x) = sqrt(x(1-x))
   s_coeff = 5.0  # strong positive selection (scaled by 2N)
   mu_selection = lambda x: (s_coeff / 2) * x * (1 - x)

   np.random.seed(42)

   # Compare neutral vs selected trajectories
   T = 2.0
   n_steps = 5000
   _, x_neutral = euler_maruyama(0.1, mu_neutral, sigma_drift, T, n_steps)
   _, x_selected = euler_maruyama(0.1, mu_selection, sigma_drift, T, n_steps)

   print(f"Neutral:  start={0.1:.2f}, end={x_neutral[-1]:.4f}")
   print(f"Selected: start={0.1:.2f}, end={x_selected[-1]:.4f}")
   print(f"(Selection pushes frequency upward)")


From SDEs to PDEs: The Fokker-Planck Equation
================================================

The SDE describes a single trajectory -- one realization of the random process.
But in population genetics, we rarely care about a single trajectory. We want to
know the **distribution** of allele frequencies across many independent loci or
many replicate populations. This distribution is described by a density function
:math:`\phi(x, t)`, where :math:`\phi(x, t) dx` is the probability of finding an
allele at frequency between :math:`x` and :math:`x + dx` at time :math:`t`.

The equation governing :math:`\phi(x, t)` is the **Fokker-Planck equation** (also
called the **Kolmogorov forward equation**):

.. math::

   \frac{\partial \phi}{\partial t} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[\sigma^2(x)\phi\right]
   - \frac{\partial}{\partial x}\left[\mu(x)\phi\right]

This is a **partial differential equation** (PDE) -- and it is the mathematical
heart of the diffusion approximation.

.. admonition:: Calculus Aside -- What is a partial differential equation?

   An **ordinary differential equation** (ODE) involves a function of a single
   variable and its derivatives. For example, :math:`dy/dt = -ky` describes
   exponential decay, where :math:`y` depends only on :math:`t`.

   A **partial differential equation** (PDE) involves a function of **multiple**
   variables and its partial derivatives. Here, :math:`\phi(x, t)` depends on
   two variables: the allele frequency :math:`x` and time :math:`t`. The symbol
   :math:`\partial \phi / \partial t` means "the rate of change of :math:`\phi`
   with respect to time, holding :math:`x` fixed." Similarly,
   :math:`\partial \phi / \partial x` means "the rate of change with respect to
   frequency, holding :math:`t` fixed."

   PDEs are harder than ODEs because the solution is a surface (a function of
   two variables) rather than a curve (a function of one variable). But they
   arise naturally whenever a quantity depends on both space and time -- as allele
   frequency density :math:`\phi(x, t)` does.

The two terms: diffusion and advection
----------------------------------------

The Fokker-Planck equation has two terms, each with a clear biological meaning:

1. **The diffusion term** (second-order derivative):

   .. math::

      \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[\sigma^2(x)\phi\right]

   This term captures **genetic drift** -- the random fluctuations in allele
   frequency due to finite population size. The second derivative acts as a
   "spreading" operator: it takes a concentration of probability at one frequency
   and spreads it out to neighboring frequencies. The coefficient
   :math:`\sigma^2(x) = x(1-x)` ensures that drift is strongest at intermediate
   frequencies and vanishes at the boundaries.

2. **The advection term** (first-order derivative):

   .. math::

      -\frac{\partial}{\partial x}\left[\mu(x)\phi\right]

   This term captures **directional forces** -- selection, mutation, migration.
   The first derivative acts as a "transport" operator: it moves probability
   density in the direction specified by :math:`\mu(x)`. Under positive selection
   (:math:`\mu(x) > 0` for :math:`x \in (0,1)`), probability mass is transported
   toward higher frequencies. Under mutation pressure, mass flows from the
   boundaries toward the interior.

.. admonition:: Calculus Aside -- Why second-order for drift and first-order for selection?

   This asymmetry reflects a fundamental difference in the nature of the two
   forces:

   **Drift is symmetric**: at any frequency :math:`x`, genetic drift is equally
   likely to push the frequency up or down. Symmetric random perturbations spread
   out a distribution -- they increase its variance without changing its mean.
   Mathematically, spreading is captured by the second derivative. (Think of the
   heat equation :math:`\partial u/\partial t = D \, \partial^2 u / \partial x^2`,
   which describes how heat diffuses from hot regions to cold regions.)

   **Selection is directional**: selection pushes allele frequency systematically
   in one direction (toward fixation for beneficial alleles, toward loss for
   deleterious ones). Directional transport of a density is captured by the first
   derivative. (Think of the transport equation
   :math:`\partial u/\partial t + v \, \partial u / \partial x = 0`, which
   describes how a quantity is carried along by a flow with velocity :math:`v`.)

   The Fokker-Planck equation combines both effects: drift spreads the
   distribution (second-order term) while selection shifts it (first-order term).

For the neutral Wright-Fisher diffusion with :math:`\mu(x) = 0` and
:math:`\sigma^2(x) = x(1-x)`, the Fokker-Planck equation simplifies to:

.. math::

   \frac{\partial \phi}{\partial t} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[x(1-x)\phi\right]

With selection (:math:`\mu(x) = \frac{s}{2}x(1-x)`) and reversible mutation
(:math:`\mu(x) = \frac{\theta_1}{2}(1-x) - \frac{\theta_2}{2}x + \frac{s}{2}x(1-x)`):

.. math::

   \frac{\partial \phi}{\partial t} =
   \frac{1}{2}\frac{\partial^2}{\partial x^2}\left[x(1-x)\phi\right]
   - \frac{\partial}{\partial x}\left[\left(\frac{\theta_1}{2}(1-x)
   - \frac{\theta_2}{2}x + \frac{s}{2}x(1-x)\right)\phi\right]

Each evolutionary force is a separate, additive term in the PDE. This modularity
is one of the great strengths of the diffusion framework: to add a new force,
simply add the corresponding term to :math:`\mu(x)`.


Boundary Conditions
====================

The Fokker-Planck equation describes how the density :math:`\phi(x, t)` evolves
in the interior :math:`x \in (0, 1)`. But what happens at the boundaries
:math:`x = 0` and :math:`x = 1`?

Absorbing boundaries
---------------------

At :math:`x = 0`, the derived allele is **lost** -- all copies have been replaced
by the ancestral allele. At :math:`x = 1`, the derived allele is **fixed** -- it
has replaced all ancestral copies. In the standard Wright-Fisher model (without
mutation), both boundaries are **absorbing**: once frequency reaches 0 or 1, it
stays there forever.

Mathematically, absorbing boundaries mean:

.. math::

   \phi(0, t) = 0 \quad \text{and} \quad \phi(1, t) = 0

No probability density sits at the boundaries -- any probability that reaches
a boundary is permanently removed from the interior. Over time, all probability
mass eventually reaches one boundary or the other: the allele is either lost or
fixed.

Why :math:`x(1-x)` vanishes at boundaries
-------------------------------------------

The diffusion coefficient :math:`\sigma^2(x) = x(1-x)` naturally enforces the
boundary behavior. At :math:`x = 0`:

.. math::

   \sigma^2(0) = 0 \cdot (1 - 0) = 0

and at :math:`x = 1`:

.. math::

   \sigma^2(1) = 1 \cdot (1 - 1) = 0

When the diffusion coefficient vanishes, the random component of the dynamics
disappears. An allele that has been lost (:math:`x = 0`) or fixed (:math:`x = 1`)
experiences no further drift. This is a natural consequence of the Wright-Fisher
model: you cannot have random fluctuations when one allele has zero copies.

The flux condition
-------------------

The **probability flux** at a boundary measures the rate at which probability
density flows out of the interval. For the Fokker-Planck equation, the flux at
:math:`x = 0` is:

.. math::

   J(0, t) = \mu(0)\phi(0, t) - \frac{1}{2}\frac{\partial}{\partial x}\left[\sigma^2(x)\phi\right]\bigg|_{x=0}

Under the absorbing boundary condition, probability that hits the boundary is
removed from the system. The total probability in the interior
:math:`\int_0^1 \phi(x,t)dx` decreases over time as alleles are lost or fixed.

Reflecting boundaries and mutation
------------------------------------

With mutation (:math:`\theta_1, \theta_2 > 0`), the boundaries become
**reflecting** rather than absorbing. Mutation at rate :math:`\theta_1` creates new
derived copies even when :math:`x = 0`, and back-mutation at rate :math:`\theta_2`
creates ancestral copies even when :math:`x = 1`. Probability flux is returned to
the interior, and the density reaches a nondegenerate stationary distribution.

In the **infinite-sites model** -- the standard framework for tools like
:ref:`dadi <dadi_timepiece>` and :ref:`moments <moments_timepiece>` -- each new
mutation arises at a previously monomorphic site. The density :math:`\phi(x, t)`
represents the density of *segregating* sites at frequency :math:`x`. New
mutations enter at :math:`x = 1/(2N) \approx 0` at a rate proportional to
:math:`\theta`, and sites are removed when they reach fixation (:math:`x = 1`)
or loss (:math:`x = 0`).


Stationary Distributions
==========================

When the density :math:`\phi(x, t)` stops changing -- when
:math:`\partial \phi / \partial t = 0` -- we have a **stationary distribution**.
Setting the left-hand side of the Fokker-Planck equation to zero converts the
PDE into an ODE (because the :math:`t` dependence disappears):

.. math::

   0 = \frac{1}{2}\frac{d^2}{dx^2}\left[\sigma^2(x)\phi\right]
   - \frac{d}{dx}\left[\mu(x)\phi\right]

This is a second-order ODE in :math:`x` alone, which is much easier to solve than
the full PDE.

The neutral case
-----------------

With no selection and no mutation (:math:`\mu(x) = 0`), the ODE becomes:

.. math::

   \frac{d^2}{dx^2}\left[x(1-x)\phi(x)\right] = 0

Integrating twice gives :math:`x(1-x)\phi(x) = C_1 x + C_0`, so:

.. math::

   \phi(x) = \frac{C_1 x + C_0}{x(1-x)}

For the infinite-sites model, where new mutations enter near :math:`x = 0` and
we consider the density of segregating sites, the stationary solution is:

.. math::

   \phi(x) \propto \frac{1}{x(1-x)}

This density is **not normalizable** -- it blows up at both boundaries, reflecting
the fact that under pure drift, sites are constantly being lost and fixed, and
the steady-state density of segregating sites is concentrated near the boundaries.

.. admonition:: Probability Aside -- The neutral SFS connection

   The neutral site frequency spectrum (SFS) for a sample of :math:`n` haploids
   has the well-known form :math:`\mathbb{E}[\xi_j] = \theta / j` for
   :math:`j = 1, 2, \ldots, n-1`, where :math:`\xi_j` is the number of sites
   with :math:`j` derived alleles. The :math:`1/j` pattern is a direct
   consequence of the :math:`1/(x(1-x))` stationary density: when you sample
   :math:`n` individuals from a population with allele frequency density
   :math:`\propto 1/x`, the probability of seeing :math:`j` derived alleles
   in a sample of :math:`n` is proportional to :math:`\int_0^1 \binom{n}{j}
   x^j(1-x)^{n-j} \cdot \frac{1}{x} dx`, which evaluates to :math:`1/j`.
   We make this connection precise in the section on the site frequency
   spectrum below.

With mutation: the Beta distribution
--------------------------------------

Adding symmetric mutation with forward rate :math:`\theta_1` and backward rate
:math:`\theta_2`, the drift coefficient becomes
:math:`\mu(x) = \frac{\theta_1}{2}(1-x) - \frac{\theta_2}{2}x`. The stationary
distribution is:

.. math::

   \phi(x) \propto x^{\theta_1 - 1}(1-x)^{\theta_2 - 1}

This is the density of a **Beta distribution** with parameters :math:`\theta_1`
and :math:`\theta_2`. When :math:`\theta_1 = \theta_2 < 1`, the density is
U-shaped (alleles cluster near fixation or loss). When :math:`\theta_1 = \theta_2 > 1`,
it is bell-shaped. When :math:`\theta_1 = \theta_2 = 1`, it is uniform -- mutation
perfectly balances drift.

With selection: exponential tilting
-------------------------------------

Adding genic selection :math:`\mu(x) = \frac{s}{2}x(1-x) + \frac{\theta_1}{2}(1-x)
- \frac{\theta_2}{2}x`, the stationary distribution becomes:

.. math::

   \phi(x) \propto x^{\theta_1 - 1}(1-x)^{\theta_2 - 1} e^{sx}

The exponential factor :math:`e^{sx}` **tilts** the neutral distribution: positive
selection (:math:`s > 0`) upweights high frequencies, shifting probability mass
toward fixation. Negative selection (:math:`s < 0`) upweights low frequencies,
keeping deleterious alleles rare.

.. code-block:: python

   def stationary_density(x, theta1, theta2, s=0.0):
       """Compute the stationary density of the diffusion (unnormalized).

       Parameters
       ----------
       x : float or ndarray
           Allele frequency in (0, 1).
       theta1 : float
           Forward mutation rate (scaled by 2N).
       theta2 : float
           Backward mutation rate (scaled by 2N).
       s : float
           Selection coefficient (scaled by 2N). Positive favors derived.

       Returns
       -------
       density : float or ndarray
           Unnormalized stationary density at each x.
       """
       x = np.asarray(x, dtype=float)
       # Beta distribution kernel times exponential tilting for selection
       log_density = (theta1 - 1) * np.log(x) + (theta2 - 1) * np.log(1 - x) + s * x
       return np.exp(log_density)

   # Evaluate and normalize over a grid
   x_grid = np.linspace(0.001, 0.999, 1000)

   # Case 1: Neutral with symmetric mutation
   phi_neutral = stationary_density(x_grid, theta1=0.5, theta2=0.5)
   phi_neutral /= np.trapz(phi_neutral, x_grid)  # normalize

   # Case 2: With positive selection
   phi_selected = stationary_density(x_grid, theta1=0.5, theta2=0.5, s=10.0)
   phi_selected /= np.trapz(phi_selected, x_grid)

   # Verification: the density should integrate to 1
   print(f"Integral (neutral):  {np.trapz(phi_neutral, x_grid):.6f}")
   print(f"Integral (selected): {np.trapz(phi_selected, x_grid):.6f}")

   # Verification: mean frequency should be higher under positive selection
   mean_neutral = np.trapz(x_grid * phi_neutral, x_grid)
   mean_selected = np.trapz(x_grid * phi_selected, x_grid)
   print(f"Mean frequency (neutral):  {mean_neutral:.4f}")
   print(f"Mean frequency (selected): {mean_selected:.4f}")
   print(f"Selection shifts mean upward: {mean_selected > mean_neutral}")


Numerical Solutions: Finite Differences for PDEs
==================================================

Exact solutions to the Fokker-Planck equation exist only in special cases
(stationary distributions, constant coefficients). For realistic demographic
models -- bottlenecks, exponential growth, population splits -- we must solve
the PDE numerically. This is the heart of :ref:`dadi <dadi_timepiece>`.

Discretizing :math:`x` on a grid
----------------------------------

The first step is to replace the continuous frequency variable :math:`x \in [0,1]`
with a discrete grid of :math:`P` points:

.. math::

   x_0 = 0, \quad x_1 = \Delta x, \quad x_2 = 2\Delta x, \quad \ldots, \quad x_{P-1} = 1

where :math:`\Delta x = 1/(P-1)`. The density :math:`\phi(x, t)` is approximated
by its values at these grid points: :math:`\phi_i(t) \approx \phi(x_i, t)`.

Finite-difference approximations
----------------------------------

We approximate the derivatives in the Fokker-Planck equation using
**finite differences**:

- **First derivative** (central difference):

  .. math::

     \frac{\partial f}{\partial x}\bigg|_{x_i} \approx \frac{f_{i+1} - f_{i-1}}{2\Delta x}

- **Second derivative**:

  .. math::

     \frac{\partial^2 f}{\partial x^2}\bigg|_{x_i} \approx \frac{f_{i+1} - 2f_i + f_{i-1}}{(\Delta x)^2}

These formulas replace derivatives with algebraic operations on neighboring grid
values. The error in each approximation is :math:`O((\Delta x)^2)` -- as the grid
gets finer, the approximation improves quadratically.

The method of lines
---------------------

Substituting the finite-difference approximations into the Fokker-Planck equation
converts the PDE into a **system of ODEs** -- one ODE per grid point:

.. math::

   \frac{d\phi_i}{dt} = F_i(\phi_0, \phi_1, \ldots, \phi_{P-1})

where :math:`F_i` involves :math:`\phi_{i-1}`, :math:`\phi_i`, and
:math:`\phi_{i+1}` (the values at neighboring grid points). This approach is
called the **method of lines**: we discretize in space (the :math:`x` direction)
but leave time continuous, converting a PDE into a system of coupled ODEs that
can be solved with standard ODE integrators.

Crank-Nicolson time stepping
-------------------------------

For time integration, :ref:`dadi <dadi_timepiece>` uses the **Crank-Nicolson**
scheme, which averages the explicit (forward Euler) and implicit (backward Euler)
methods:

.. math::

   \frac{\phi^{n+1}_i - \phi^n_i}{\Delta t} =
   \frac{1}{2}\left[F_i(\phi^n) + F_i(\phi^{n+1})\right]

This scheme is **unconditionally stable** (no restriction on the time step
:math:`\Delta t` relative to :math:`\Delta x`) and **second-order accurate** in
both space and time. The price is that each time step requires solving a linear
system, but the system is **tridiagonal** (each equation involves only three
unknowns), so it can be solved in :math:`O(P)` time using the Thomas algorithm.

The curse of dimensionality
-----------------------------

For a single population, the grid has :math:`P` points and everything is fast.
But for :math:`d` populations, we need a grid in :math:`d`-dimensional frequency
space, with :math:`P^d` grid points. This is the **curse of dimensionality**:

- 1 population: :math:`P` grid points (manageable)
- 2 populations: :math:`P^2` grid points (feasible for moderate :math:`P`)
- 3 populations: :math:`P^3` grid points (expensive but possible)
- 4+ populations: :math:`P^4` or more (often prohibitive)

This scaling is why :ref:`dadi <dadi_timepiece>` becomes expensive for more than
2-3 populations, and why :ref:`moments <moments_timepiece>` was developed as an
alternative that avoids the frequency grid entirely.

Code: 1D diffusion solver
----------------------------

Let us implement a simple 1D diffusion solver using the method of lines with
Crank-Nicolson time stepping. We will evolve a neutral population through a
bottleneck and extract the SFS.

.. code-block:: python

   def solve_diffusion_1d(P, T, n_time_steps, theta, s=0.0, N_func=None):
       """Solve the 1D Wright-Fisher diffusion equation numerically.

       Uses the method of lines with Crank-Nicolson time stepping.
       The PDE is:
           dphi/dt = (1/(2*N(t))) * (1/2) d^2/dx^2 [x(1-x)*phi]
                     - d/dx [mu(x)*phi]
       where the population size N(t) can change over time.

       Parameters
       ----------
       P : int
           Number of grid points (including boundaries).
       T : float
           Total time to integrate (in units of 2*N_ref generations).
       n_time_steps : int
           Number of Crank-Nicolson steps.
       theta : float
           Population-scaled mutation rate 4*N_ref*mu.
       s : float
           Population-scaled selection coefficient 2*N_ref*s.
       N_func : callable or None
           N_func(t) returns N(t)/N_ref at time t. If None, constant size 1.

       Returns
       -------
       x_grid : ndarray of shape (P,)
           Frequency grid points.
       phi : ndarray of shape (P,)
           Final density at each grid point.
       """
       if N_func is None:
           N_func = lambda t: 1.0

       dx = 1.0 / (P - 1)
       dt = T / n_time_steps
       x_grid = np.linspace(0, 1, P)

       # Initialize with the neutral stationary distribution (1/x scaled)
       # Avoid boundary singularities by starting just inside
       phi = np.zeros(P)
       for i in range(1, P - 1):
           x = x_grid[i]
           phi[i] = theta / (x * (1 - x))

       # Normalize so total mass = theta * L (number of segregating sites)
       # For simplicity, normalize to unit mass
       phi /= np.trapz(phi, x_grid)

       # Time-stepping loop
       for step in range(n_time_steps):
           t = step * dt
           N_rel = N_func(t)

           # Build the tridiagonal matrix for Crank-Nicolson
           # We use a simplified version: explicit half-step + implicit half-step

           # Compute the right-hand side F(phi) at interior points
           def rhs(phi_in):
               """Compute the spatial operator F(phi) at interior points."""
               F = np.zeros(P)
               for i in range(1, P - 1):
                   x = x_grid[i]

                   # Diffusion term: (1/2) d^2/dx^2 [x(1-x)*phi]
                   # We compute g(x) = x(1-x)*phi first, then differentiate
                   g_im1 = x_grid[i-1] * (1 - x_grid[i-1]) * phi_in[i-1]
                   g_i   = x * (1 - x) * phi_in[i]
                   g_ip1 = x_grid[i+1] * (1 - x_grid[i+1]) * phi_in[i+1]
                   diffusion = 0.5 * (g_ip1 - 2*g_i + g_im1) / (dx**2)

                   # Advection term: -d/dx[mu(x)*phi]
                   # mu(x) = (s/2)*x*(1-x)  (selection only for simplicity)
                   mu_im1 = (s / 2) * x_grid[i-1] * (1 - x_grid[i-1])
                   mu_ip1 = (s / 2) * x_grid[i+1] * (1 - x_grid[i+1])
                   h_im1 = mu_im1 * phi_in[i-1]
                   h_ip1 = mu_ip1 * phi_in[i+1]
                   advection = -(h_ip1 - h_im1) / (2 * dx)

                   # Scale by 1/N_rel (drift is inversely proportional to N)
                   F[i] = diffusion / N_rel + advection
               return F

           # Crank-Nicolson: phi^{n+1} = phi^n + dt/2 * [F(phi^n) + F(phi^{n+1})]
           # Approximate with two half-steps (predictor-corrector)
           F_n = rhs(phi)
           phi_pred = phi + dt * F_n         # explicit predictor
           phi_pred[0] = 0.0                  # absorbing boundary
           phi_pred[-1] = 0.0
           phi_pred = np.maximum(phi_pred, 0) # enforce non-negativity

           F_pred = rhs(phi_pred)
           phi = phi + 0.5 * dt * (F_n + F_pred)  # corrector
           phi[0] = 0.0
           phi[-1] = 0.0
           phi = np.maximum(phi, 0)

       return x_grid, phi

   # Solve for a constant-size population
   P = 201            # 201 grid points
   T = 0.5            # integrate for 0.5 diffusion time units
   n_steps_t = 2000   # 2000 time steps
   theta = 1.0        # mutation rate

   x_grid, phi_const = solve_diffusion_1d(P, T, n_steps_t, theta)

   # Solve through a bottleneck: N drops to 0.1 for the middle period
   def bottleneck(t):
       """Population size function: bottleneck from t=0.1 to t=0.3."""
       if 0.1 <= t <= 0.3:
           return 0.1  # 10x reduction
       return 1.0

   _, phi_bottle = solve_diffusion_1d(P, T, n_steps_t, theta, N_func=bottleneck)

   # Verification: total probability should be positive
   mass_const = np.trapz(phi_const, x_grid)
   mass_bottle = np.trapz(phi_bottle, x_grid)
   print(f"Total mass (constant N): {mass_const:.4f}")
   print(f"Total mass (bottleneck): {mass_bottle:.4f}")
   print(f"Bottleneck reduces diversity: {mass_bottle < mass_const}")


Connection to the Site Frequency Spectrum
==========================================

The diffusion density :math:`\phi(x, t)` describes the continuous allele frequency
distribution. But what we observe in practice is a **sample** of :math:`n`
individuals, which gives us integer allele counts. The connection between the
continuous density and the discrete **site frequency spectrum** (SFS) is the
**binomial sampling formula**.

The binomial bridge
---------------------

The expected number of segregating sites where :math:`j` out of :math:`n`
sampled chromosomes carry the derived allele is:

.. math::

   \phi_j = \theta L \int_0^1 \binom{n}{j} x^j (1-x)^{n-j} \phi(x, t) \, dx

where :math:`L` is the number of independent loci, :math:`\theta` is the
per-site mutation rate, and :math:`\binom{n}{j} x^j (1-x)^{n-j}` is the
binomial probability of sampling :math:`j` derived alleles from a population
with allele frequency :math:`x`.

.. admonition:: Probability Aside -- Why binomial sampling?

   If the true allele frequency in the population is :math:`x`, and you sample
   :math:`n` chromosomes independently, the number of derived alleles in your
   sample follows :math:`\text{Binom}(n, x)`. This is because each chromosome
   independently carries the derived allele with probability :math:`x`. The
   integral over :math:`x` averages this binomial probability over all possible
   population frequencies, weighted by the diffusion density
   :math:`\phi(x, t)`.

This formula is the link between the continuous world of the diffusion and the
discrete world of observed data. It is used by :ref:`dadi <dadi_timepiece>` to
convert its numerically computed :math:`\phi(x, t)` into a predicted SFS that
can be compared to the observed SFS.

.. code-block:: python

   from scipy.special import comb

   def density_to_sfs(x_grid, phi, n_samples):
       """Convert a diffusion density to a site frequency spectrum.

       Applies the binomial sampling formula:
           sfs[j] = integral of Binom(n, j, x) * phi(x) dx

       Parameters
       ----------
       x_grid : ndarray of shape (P,)
           Frequency grid points.
       phi : ndarray of shape (P,)
           Density values at each grid point.
       n_samples : int
           Number of sampled chromosomes.

       Returns
       -------
       sfs : ndarray of shape (n_samples - 1,)
           Expected SFS entries for j = 1, ..., n_samples - 1.
       """
       sfs = np.zeros(n_samples - 1)
       for j in range(1, n_samples):
           # Binomial probability at each grid point
           # comb(n, j) computes the binomial coefficient "n choose j"
           binom_probs = comb(n_samples, j) * x_grid**j * (1 - x_grid)**(n_samples - j)
           # Integrate phi(x) * Binom(n, j, x) over x using the trapezoidal rule
           sfs[j - 1] = np.trapz(binom_probs * phi, x_grid)
       return sfs

   # Extract SFS from our solved densities
   n_samples = 20  # sample 20 chromosomes

   sfs_const = density_to_sfs(x_grid, phi_const, n_samples)
   sfs_bottle = density_to_sfs(x_grid, phi_bottle, n_samples)

   # Verification: neutral SFS should follow the 1/j pattern
   expected_neutral = np.array([1.0 / j for j in range(1, n_samples)])
   # Normalize both to compare shapes
   sfs_const_norm = sfs_const / sfs_const.sum()
   expected_norm = expected_neutral / expected_neutral.sum()

   print("SFS comparison (constant N vs 1/j theory):")
   print(f"  {'j':>3s}  {'Simulated':>10s}  {'Theory 1/j':>10s}  {'Ratio':>8s}")
   for j in range(min(8, n_samples - 1)):
       ratio = sfs_const_norm[j] / expected_norm[j] if expected_norm[j] > 0 else 0
       print(f"  {j+1:3d}  {sfs_const_norm[j]:10.4f}  {expected_norm[j]:10.4f}  {ratio:8.4f}")

   # The bottleneck SFS should show excess rare and common variants
   print(f"\nBottleneck effect on singletons (j=1):")
   print(f"  Constant N: {sfs_const[0]:.4f}")
   print(f"  Bottleneck: {sfs_bottle[0]:.4f}")

How dadi and moments differ
-----------------------------

The diffusion-to-SFS conversion above is precisely what :ref:`dadi <dadi_timepiece>`
does: it solves the PDE for :math:`\phi(x, t)`, then integrates against binomial
weights to extract the expected SFS.

:ref:`moments <moments_timepiece>` takes a fundamentally different approach. Instead
of solving for the full density :math:`\phi(x, t)` and then sampling it, ``moments``
derives ODEs that govern the SFS entries :math:`\phi_j` **directly**. These ODEs
are obtained by applying the binomial sampling formula to both sides of the
Fokker-Planck equation and integrating. The result is a system of ODEs -- one per
SFS entry -- that skips the frequency grid entirely.

The trade-off:

- **dadi**: Solves the PDE on a grid, then extracts the SFS. The grid introduces
  discretization artifacts, but gives access to the full frequency density
  :math:`\phi(x,t)` -- useful for computing quantities beyond the SFS (e.g.,
  fixation probabilities, frequency densities under selection). Multi-population
  models require :math:`O(P^d)` grid points.

- **moments**: Solves ODEs directly for the SFS entries. No grid artifacts, but
  no access to the full frequency density. Multi-population models require
  :math:`O(\prod n_i)` SFS entries, which can be more efficient than
  :math:`O(P^d)` for large :math:`P`.

Both approaches are built on the same diffusion approximation -- they differ only
in how they extract the SFS from it.


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Concept
     - Key Formula / Idea
   * - WF frequency change
     - :math:`\mathbb{E}[\Delta x] \approx 0` (neutral), :math:`\text{Var}(\Delta x) = x(1-x)/(2N)`
   * - Diffusion timescale
     - :math:`\tau = t/(2N)` (same as coalescent time units)
   * - SDE (Wright-Fisher diffusion)
     - :math:`dx = \mu(x)dt + \sqrt{x(1-x)}\,dW`
   * - Fokker-Planck / Kolmogorov forward
     - :math:`\partial\phi/\partial t = \tfrac{1}{2}\partial^2_{xx}[\sigma^2\phi] - \partial_x[\mu\phi]`
   * - Neutral stationary density
     - :math:`\phi(x) \propto 1/[x(1-x)]`, gives :math:`\xi_j \propto 1/j` SFS
   * - With mutation
     - :math:`\phi(x) \propto x^{\theta_1-1}(1-x)^{\theta_2-1}` (Beta distribution)
   * - With selection
     - :math:`\phi(x) \propto x^{\theta_1-1}(1-x)^{\theta_2-1}e^{sx}` (exponential tilting)
   * - Finite-difference PDE solver
     - Discretize :math:`x` on :math:`P` points, method of lines + Crank-Nicolson
   * - Multi-population scaling
     - :math:`O(P^d)` grid points -- the curse of dimensionality
   * - Diffusion density to SFS
     - :math:`\phi_j = \theta L \int \binom{n}{j}x^j(1-x)^{n-j}\phi(x)dx`

These are the gears that connect the discrete ticking of the Wright-Fisher model
to the smooth sweep of continuous frequency evolution. The diffusion approximation
is the bridge between the coalescent theory you learned in :ref:`coalescent_theory`
and the SFS-based inference tools -- :ref:`dadi <dadi_timepiece>` and
:ref:`moments <moments_timepiece>` -- that use it to read demographic history
from genetic variation.

Next: :ref:`odes` -- the ordinary differential equations that ``moments`` uses to
bypass the diffusion PDE entirely, computing the SFS without a frequency grid.
