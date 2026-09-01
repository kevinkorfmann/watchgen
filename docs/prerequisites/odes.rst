.. _odes:

===============================
Ordinary Differential Equations
===============================

   *"The hands of the clock obey equations you can write down -- and solve."*

The Big Idea
============

A watchmaker does not guess how fast a gear turns. The gear ratio, the spring
tension, and the escapement frequency *determine* the rotation rate exactly.
Given these specifications, the watchmaker can predict the position of every
hand at every moment. The equations governing this motion are ordinary
differential equations (ODEs).

In population genetics, ODEs appear everywhere. The expected number of ancestral
lineages at time :math:`t` satisfies an ODE (see :ref:`coalescent_theory`). Each
entry of the site frequency spectrum evolves under drift and mutation according
to a system of coupled ODEs (see the moments Timepiece,
:ref:`moments_timepiece`). The probability of :math:`j` undistinguished lineages
remaining at time :math:`t` in SMC++ is governed by yet another ODE system (see
:ref:`smcpp_timepiece`). The transition probabilities of momi2's Moran model are
computed via the matrix exponential of a rate matrix -- a linear ODE in disguise
(see :ref:`momi2_timepiece`).

This chapter gives you the mathematical and computational tools to understand,
solve, and implement ODEs. We start from the simplest possible case and build
toward the systems of equations that power multiple Timepieces. By the end, you
will know how to solve an ODE numerically, understand when standard methods fail
and what to do about it, and recognize the matrix exponential as a special case
of a linear ODE solution.


What Is an ODE?
===============

An **ordinary differential equation** relates a quantity to its rate of change
with respect to a single independent variable. In every example in this book,
the independent variable is time :math:`t`. We write:

.. math::

   \frac{dy}{dt} = f(y, t)

This says: "the rate at which :math:`y` changes is given by the function
:math:`f` of the current value :math:`y` and the current time :math:`t`."
If you are comfortable with the idea that velocity tells you how position
changes, you already have the right intuition. Here, :math:`y` is the
"position" and :math:`f(y, t)` is the "velocity."

.. admonition:: Calculus Aside -- Derivatives and rates of change

   The symbol :math:`dy/dt` is the **derivative** of :math:`y` with respect
   to :math:`t`. It measures the *instantaneous rate of change* of :math:`y`
   as :math:`t` increases. If :math:`y(t)` represents the number of lineages
   at time :math:`t`, then :math:`dy/dt` tells you how fast that number is
   changing right now.

   Formally, the derivative is the limit of a difference quotient:

   .. math::

      \frac{dy}{dt} = \lim_{h \to 0} \frac{y(t + h) - y(t)}{h}

   The numerator is the change in :math:`y` over a small interval :math:`h`,
   and dividing by :math:`h` gives the rate per unit time. Taking :math:`h`
   to zero gives the instantaneous rate. If you have never seen calculus,
   think of :math:`dy/dt` as "the slope of the graph of :math:`y` versus
   :math:`t`" -- positive slope means :math:`y` is increasing, negative
   slope means :math:`y` is decreasing.

An **initial value problem** (IVP) is an ODE together with a starting condition:

.. math::

   \frac{dy}{dt} = f(y, t), \qquad y(t_0) = y_0

The goal is to find the function :math:`y(t)` for :math:`t > t_0` that
satisfies both the equation and the initial condition. Think of it as specifying
the gear positions at the moment the watch is wound, then asking where the gears
will be at any future time.

**Connection to the coalescent.** Conditional on currently having an integer
:math:`k` lineages, the next coalescence occurs at rate

.. math::

   c_k=\binom{k}{2}=\frac{k(k-1)}{2}.

Because each event reduces the count by one, the conditional instantaneous
mean change is :math:`-c_k`. Replacing :math:`k` by a continuous
:math:`\lambda` gives the useful mean-field ODE
:math:`d\lambda/dt=-\lambda(\lambda-1)/2`, but it is **not** the exact ODE for
:math:`\mathbb{E}[K_t]`: nonlinearity means
:math:`\mathbb{E}[K_t(K_t-1)]\ne
\mathbb{E}[K_t](\mathbb{E}[K_t]-1)` in general. The exact probability ODE is
developed below.

.. code-block:: python

   import numpy as np

   def coalescence_event_rate(k):
       """Kingman coalescence rate conditional on integer lineage count k."""
       if not isinstance(k, (int, np.integer)) or k < 1:
           raise ValueError("k must be a positive integer")
       return k * (k - 1) / 2

   # Starting with 10 lineages, the initial rate of decrease is:
   lam_0 = 10
   rate = coalescence_event_rate(lam_0)
   print(f"With {lam_0} lineages: event rate = {rate:.1f} per unit time")
   print(f"(There are {lam_0*(lam_0-1)//2} pairs, each coalescing at rate 1)")

Let us warm up with a simpler ODE before tackling numerical methods. The
**logistic equation** is a classic model for growth with saturation:

.. math::

   \frac{dy}{dt} = r \, y \left(1 - \frac{y}{K}\right), \qquad y(0) = y_0

Here :math:`r` is the growth rate and :math:`K` is the carrying capacity. When
:math:`y` is small, growth is approximately exponential (:math:`dy/dt \approx ry`).
As :math:`y` approaches :math:`K`, the factor :math:`(1 - y/K)` drives the growth
rate to zero. This ODE has an exact solution:

.. math::

   y(t) = \frac{K}{1 + \left(\frac{K}{y_0} - 1\right) e^{-rt}}

We will use this to test our numerical methods.

.. code-block:: python

   import numpy as np

   def logistic_exact(t, y0, r, K):
       """Exact solution of the logistic equation.

       Parameters
       ----------
       t : float or ndarray
           Time(s) at which to evaluate the solution.
       y0 : float
           Initial condition y(0).
       r : float
           Growth rate.
       K : float
           Carrying capacity.

       Returns
       -------
       y : float or ndarray
           Solution y(t).
       """
       if y0 <= 0 or K <= 0 or r < 0:
           raise ValueError("require y0 > 0, K > 0, and r >= 0")
       t = np.asarray(t, dtype=float)
       return K / (1 + (K / y0 - 1) * np.exp(-r * t))

   # Verify: y(0) should equal y0, and y(inf) should approach K
   y0, r, K = 0.1, 1.0, 10.0
   print(f"y(0)   = {logistic_exact(0.0, y0, r, K):.4f}  (expected {y0})")
   print(f"y(10)  = {logistic_exact(10.0, y0, r, K):.4f}  (expected ~{K})")
   print(f"y(100) = {logistic_exact(100.0, y0, r, K):.4f}  (expected {K})")


Euler's Method
==============

The simplest numerical method for ODEs is **Euler's method**. The idea is
straightforward: if you know :math:`y(t)` and the rate of change
:math:`f(y(t), t)`, you can *approximate* the value at a slightly later time
:math:`t + h` by assuming the rate is constant over the interval:

.. math::

   y(t + h) \approx y(t) + h \cdot f(y(t), t)

This is the forward Euler formula. The step size :math:`h` controls the trade-off
between speed and accuracy: smaller steps give better approximations but require
more computation.

Like a watchmaker measuring a gear's position by noting its current angular
velocity and projecting forward -- a straight-line approximation that works well
for tiny intervals but accumulates error over longer spans.

**Error analysis.** The error introduced by a single Euler step is called the
**local truncation error**. It comes from the Taylor expansion:

.. math::

   y(t + h) = y(t) + h \, y'(t) + \frac{h^2}{2} y''(t) + O(h^3)

Euler's method keeps only the first two terms, so the local error is
:math:`O(h^2)` -- proportional to :math:`h^2`. Over an interval of total
length :math:`T`, we take :math:`T/h` steps, so the **global error** is:

.. math::

   \text{global error} = O\left(\frac{T}{h} \cdot h^2\right) = O(h)

Euler's method is **first-order**: halving the step size halves the error.
This is adequate for understanding the concept, but too slow for serious
computation. We will improve on it shortly.

.. code-block:: python

   def euler_method(f, y0, t_span, h):
       """Solve dy/dt = f(y, t) using forward Euler with step size h."""
       t0, tf = t_span
       if h <= 0 or tf < t0:
           raise ValueError("require h > 0 and tf >= t0")
       times = [float(t0)]
       values = [np.atleast_1d(np.asarray(y0, dtype=float))]
       while times[-1] < tf:
           step = min(h, tf - times[-1])
           slope = np.atleast_1d(f(values[-1], times[-1]))
           if slope.shape != values[-1].shape:
               raise ValueError("f must return the same shape as y0")
           values.append(values[-1] + step * slope)
           times.append(times[-1] + step)
       return np.array(times), np.array(values).squeeze()

   # Logistic equation: dy/dt = r*y*(1 - y/K)
   def logistic_rhs(y, t):
       """Right-hand side of the logistic equation."""
       r, K = 1.0, 10.0
       return r * y * (1 - y / K)

   # Solve with different step sizes and compare to exact solution
   y0, r, K = 0.1, 1.0, 10.0
   t_final = 5.0

   print("Euler's method: convergence as h decreases")
   print(f"{'h':>10s}  {'y_euler(5)':>12s}  {'y_exact(5)':>12s}  {'error':>12s}")
   y_exact = logistic_exact(t_final, y0, r, K)

   for h in [1.0, 0.5, 0.1, 0.05, 0.01]:
       t_vals, y_vals = euler_method(logistic_rhs, y0, (0.0, t_final), h)
       y_end = y_vals[-1]
       error = abs(y_end - y_exact)
       print(f"{h:10.4f}  {float(y_end):12.6f}  {y_exact:12.6f}  {error:12.2e}")

   # Verify first-order convergence: error should halve when h halves
   _, y_h1 = euler_method(logistic_rhs, y0, (0.0, t_final), 0.1)
   _, y_h2 = euler_method(logistic_rhs, y0, (0.0, t_final), 0.05)
   ratio = abs(float(y_h1[-1]) - y_exact) / abs(float(y_h2[-1]) - y_exact)
   print(f"\nError ratio (h vs h/2): {ratio:.2f}  (expected ~2.0 for first-order)")


The Runge-Kutta Family
======================

Euler's method is first-order: to cut the error in half, you must double the
number of steps. This is wasteful. The **Runge-Kutta family** achieves higher
accuracy by evaluating :math:`f` at multiple intermediate points within each
step, like a watchmaker who checks the gear position not just at the start of
each tick, but at the midpoint and endpoint as well.

RK2: The Midpoint Method
-------------------------

The simplest improvement over Euler is the **midpoint method** (RK2). Instead
of using the slope at the beginning of the interval, it takes a trial Euler
step to the midpoint, evaluates the slope there, and uses that slope for the
full step:

.. math::

   k_1 &= f(y_n, t_n) \\
   k_2 &= f\!\left(y_n + \tfrac{h}{2} k_1, \, t_n + \tfrac{h}{2}\right) \\
   y_{n+1} &= y_n + h \, k_2

The local truncation error is :math:`O(h^3)` and the global error is
:math:`O(h^2)` -- second-order. Halving the step size now reduces the error
by a factor of 4.

RK4: The Classic Method
------------------------

The workhorse of numerical ODEs is the **fourth-order Runge-Kutta method**
(RK4). It evaluates :math:`f` at four points per step and achieves
fourth-order accuracy:

.. math::

   k_1 &= f(y_n, t_n) \\
   k_2 &= f\!\left(y_n + \tfrac{h}{2} k_1, \, t_n + \tfrac{h}{2}\right) \\
   k_3 &= f\!\left(y_n + \tfrac{h}{2} k_2, \, t_n + \tfrac{h}{2}\right) \\
   k_4 &= f(y_n + h \, k_3, \, t_n + h) \\
   y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)

The weights :math:`1/6, 2/6, 2/6, 1/6` are not arbitrary -- they are chosen so
that the Taylor expansion of :math:`y_{n+1}` matches the true solution through
:math:`O(h^4)`. The local error is :math:`O(h^5)` and the global error is
:math:`O(h^4)`. Halving :math:`h` reduces the error by a factor of 16.

The **Butcher tableau** is a compact notation for writing down any Runge-Kutta
method. For RK4:

.. math::

   \begin{array}{c|cccc}
   0 & & & & \\
   \tfrac{1}{2} & \tfrac{1}{2} & & & \\
   \tfrac{1}{2} & 0 & \tfrac{1}{2} & & \\
   1 & 0 & 0 & 1 & \\
   \hline
   & \tfrac{1}{6} & \tfrac{1}{3} & \tfrac{1}{3} & \tfrac{1}{6}
   \end{array}

The left column gives the time offsets for each stage. The lower-triangular
body gives the coefficients used to compute each :math:`k_i` from previous
stages. The bottom row gives the weights for the final combination.

RK45: Adaptive Step Size (Dormand-Prince)
------------------------------------------

A fixed step size is wasteful: some parts of the solution change rapidly
(requiring small :math:`h`) while others change slowly (allowing large
:math:`h`). The **Dormand-Prince method** (RK45) uses two Runge-Kutta
formulas of different orders -- a fourth-order and a fifth-order -- and
compares them to estimate the local error. If the error is too large, the
step is rejected and :math:`h` is reduced. If the error is well below the
tolerance, :math:`h` is increased.

This is the method behind ``scipy.integrate.solve_ivp`` with
``method='RK45'`` (the default), and it is what you should use in practice
for non-stiff problems.

.. code-block:: python

   def rk4_method(f, y0, t_span, h):
       """Solve dy/dt = f(y, t) using the classical fourth-order Runge-Kutta."""
       t0, tf = t_span
       if h <= 0 or tf < t0:
           raise ValueError("require h > 0 and tf >= t0")
       times = [float(t0)]
       values = [np.atleast_1d(np.asarray(y0, dtype=float))]
       while times[-1] < tf:
           step = min(h, tf - times[-1])
           t = times[-1]
           y = values[-1]

           # Four slope evaluations
           k1 = np.atleast_1d(f(y, t))
           k2 = np.atleast_1d(f(y + step / 2 * k1, t + step / 2))
           k3 = np.atleast_1d(f(y + step / 2 * k2, t + step / 2))
           k4 = np.atleast_1d(f(y + step * k3, t + step))
           if not all(k.shape == y.shape for k in (k1, k2, k3, k4)):
               raise ValueError("f must return the same shape as y0")

           # Weighted combination
           values.append(y + (step / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
           times.append(t + step)
       return np.array(times), np.array(values).squeeze()

   # Compare Euler and RK4 with the independent exact solution.

   y0, r, K = 0.1, 1.0, 10.0
   t_final = 5.0
   y_exact = logistic_exact(t_final, y0, r, K)

   # Euler with h=0.1
   _, y_euler = euler_method(logistic_rhs, y0, (0.0, t_final), 0.1)

   # RK4 with h=0.1
   _, y_rk4 = rk4_method(logistic_rhs, y0, (0.0, t_final), 0.1)

   print(f"Exact solution:  {y_exact:.10f}")
   print(f"Euler (h=0.1):   {float(y_euler[-1]):.10f}  error={abs(float(y_euler[-1]) - y_exact):.2e}")
   print(f"RK4   (h=0.1):   {float(y_rk4[-1]):.10f}  error={abs(float(y_rk4[-1]) - y_exact):.2e}")

   # RK4 convergence: error should decrease as h^4
   print("\nRK4 convergence:")
   print(f"{'h':>10s}  {'error':>12s}  {'ratio':>8s}")
   prev_error = None
   for h in [0.5, 0.25, 0.125, 0.0625]:
       _, y_rk = rk4_method(logistic_rhs, y0, (0.0, t_final), h)
       err = abs(float(y_rk[-1]) - y_exact)
       ratio_str = f"{prev_error / err:.1f}" if prev_error is not None else "---"
       print(f"{h:10.4f}  {err:12.2e}  {ratio_str:>8s}")
       prev_error = err
   # Expected ratio ~16 when h halves (fourth-order convergence)


Systems of Coupled ODEs
=======================

So far we have solved scalar ODEs -- a single unknown :math:`y(t)`. But the
ODEs in population genetics almost always involve **systems**: multiple
quantities evolving simultaneously, each one's rate of change depending on the
others. The gears of a watch do not turn in isolation; each gear's motion is
coupled to its neighbors.

.. admonition:: Calculus Aside -- From scalar to vector ODEs

   A system of :math:`m` coupled ODEs can be written as a single vector
   equation. Define :math:`\mathbf{y} = (y_1, y_2, \ldots, y_m)` as the
   **state vector**, and :math:`\mathbf{f} = (f_1, f_2, \ldots, f_m)` as
   the vector of right-hand sides:

   .. math::

      \frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}, t)

   This is the same as writing :math:`m` separate equations:

   .. math::

      \frac{dy_1}{dt} &= f_1(y_1, y_2, \ldots, y_m, t) \\
      \frac{dy_2}{dt} &= f_2(y_1, y_2, \ldots, y_m, t) \\
      &\vdots \\
      \frac{dy_m}{dt} &= f_m(y_1, y_2, \ldots, y_m, t)

   The key point: each :math:`f_i` can depend on *all* components of
   :math:`\mathbf{y}`, not just :math:`y_i`. This coupling is what makes
   systems interesting and what requires us to solve all equations
   simultaneously.

   Every numerical method we have discussed (Euler, RK4, etc.) extends
   directly to vector ODEs. You simply replace scalar :math:`y` with vector
   :math:`\mathbf{y}` and scalar :math:`f` with vector :math:`\mathbf{f}`.
   The formulas are identical; only the data types change from floats to
   arrays.

**Example: exact coalescent lineage-count probabilities.** Let
:math:`p_k(t)=P(K_t=k\mid K_0=n)`. From state :math:`k`, Kingman's coalescent
jumps only to :math:`k-1`, at rate :math:`c_k=\binom{k}{2}`. The Kolmogorov
forward equations are therefore

.. math::

   \frac{dp_k}{dt}=c_{k+1}p_{k+1}-c_kp_k,
   \qquad k=1,\ldots,n,

where :math:`c_1=0` and :math:`p_{n+1}=0`. The gain into state :math:`k`
exactly matches loss from state :math:`k+1`, so summing the equations gives
:math:`d\sum_kp_k/dt=0`. Unlike the earlier mean-field equation, this system is
an exact description of the Kingman lineage-count process :cite:`kingman1982`.

The ``moments`` software also evolves a coupled ODE system, but for expected
sample-frequency-spectrum entries. Its operators and time scaling should be
taken from the original derivation and implementation rather than inferred by
treating adjacent frequency classes as a generic birth--death chain
:cite:`moments`.

.. code-block:: python

   def coalescent_count_ode(probabilities, t):
       """Forward equation for Kingman lineage-count probabilities.

       Parameters
       ----------
       probabilities : ndarray of shape (n,)
           Entry k-1 is P(K_t=k), for k=1,...,n.
       t : float
           Current time (unused for this time-homogeneous process).
       """
       probabilities = np.asarray(probabilities, dtype=float)
       if probabilities.ndim != 1 or len(probabilities) < 1:
           raise ValueError("probabilities must be a nonempty vector")
       n = len(probabilities)
       derivative = np.zeros(n)
       for k in range(1, n + 1):
           loss = k * (k - 1) / 2 * probabilities[k - 1]
           gain = 0.0
           if k < n:
               gain = (k + 1) * k / 2 * probabilities[k]
           derivative[k - 1] = gain - loss
       return derivative

   # For n=3, an independent closed form is available.
   p0 = np.array([0.0, 0.0, 1.0])
   times, probabilities = rk4_method(coalescent_count_ode, p0, (0.0, 2.0), 0.001)
   p_numeric = probabilities[-1]
   t = times[-1]
   p3 = np.exp(-3 * t)
   p2 = 1.5 * (np.exp(-t) - np.exp(-3 * t))
   p_exact = np.array([1 - p2 - p3, p2, p3])
   print("Lineage-count probabilities at t=2:")
   print("  numerical:", p_numeric)
   print("  exact:    ", p_exact)
   print("  total:    ", p_numeric.sum())


Stiffness and Implicit Methods
==============================

Not all ODEs are created equal. Some systems contain processes operating on
**wildly different timescales**. Consider a watch with a seconds hand, a minute
hand, and a tourbillon rotating once per hour -- trying to track all three with
the same tiny time step (sized for the fastest hand) is enormously wasteful.
But using a large time step (sized for the slowest hand) makes the fast
component oscillate and blow up.

This phenomenon is called **stiffness**. A system is stiff when the ratio of
the fastest to slowest timescale is large. Explicit methods like Euler and RK4
are forced to take tiny steps -- not for accuracy, but for **stability**. Even
though the fast component decays quickly and becomes negligible, an explicit
method must resolve it at every step or the numerical solution will oscillate
wildly and diverge.

In population genetics, stiffness arises when:

- **Strong selection meets weak drift**: Selection changes allele frequencies
  on a timescale of :math:`1/s` generations, while drift operates on a
  timescale of :math:`2N` generations. If :math:`2Ns \gg 1`, the system is
  stiff.
- **Migration between populations of very different sizes**: Fast migration
  equilibrates allele frequencies quickly, but population size changes happen
  slowly.
- **Multi-population SFS computation**: The moment equations for large sample
  sizes can have eigenvalues spanning many orders of magnitude.

The **backward (implicit) Euler method** addresses stiffness by evaluating
:math:`f` at the *next* time point rather than the current one:

.. math::

   y_{n+1} = y_n + h \cdot f(y_{n+1}, t_{n+1})

Note that :math:`y_{n+1}` appears on *both* sides of the equation. This means
we must solve an equation (or a system of equations) at each step. For linear
ODEs, this reduces to solving a linear system; for nonlinear ODEs, it requires
Newton's method or a similar iterative procedure.

The trade-off: implicit methods require more work per step. Backward Euler is
**A-stable** for the scalar linear test equation :math:`y'=\lambda y` with
:math:`\operatorname{Re}(\lambda)<0`, so its stability does not impose the
explicit Euler bound. That does not mean every implicit method is stable for
every nonlinear problem or that arbitrarily large steps are accurate; nonlinear
solves can fail and accuracy still constrains the step size.

**BDF methods** (Backward Differentiation Formulas) are a family of implicit
multi-step methods that generalize backward Euler. They are the method behind
``scipy.integrate.solve_ivp`` with ``method='BDF'``, and they are the standard
choice for stiff ODE systems in population genetics.

.. code-block:: python

   def backward_euler_decay(rate, y0, T, h):
       """Solve y'=-rate*y with backward Euler."""
       if rate < 0 or h <= 0 or T < 0:
           raise ValueError("require rate >= 0, h > 0, and T >= 0")
       t = 0.0
       y = float(y0)
       while t < T:
           step = min(h, T - t)
           y /= 1 + rate * step
           t += step
       return y

   rate, y0_stiff, T, h = 1000.0, 1.0, 0.1, 0.01
   _, explicit = euler_method(
       lambda y, t: -rate * y, y0_stiff, (0.0, T), h
   )
   implicit = backward_euler_decay(rate, y0_stiff, T, h)
   exact = np.exp(-rate * T)
   print("Stiff decay y'=-1000y at t=0.1 with h=0.01")
   print(f"  forward Euler:  {float(explicit[-1]):.6e}  (unstable)")
   print(f"  backward Euler: {implicit:.6e}  (stable, but not exact)")
   print(f"  exact:          {exact:.6e}")

.. admonition:: Probability Aside -- Stiffness in the moments Timepiece

   When the moments Timepiece (:ref:`moments_timepiece`) computes the SFS
   under strong selection, the ODE system becomes stiff. The selection
   scaled selection coefficient introduces a short timescale in
   coalescent units, which can be orders of magnitude shorter than the drift
   timescale. The ``moments`` package uses implicit methods (specifically
   a Crank--Nicolson update in its integration code) together with controlled
   time steps :cite:`moments`.


The Matrix Exponential
======================

Many ODE systems in population genetics are **linear**: the right-hand side
is a matrix times the state vector. These linear systems have a beautiful
closed-form solution involving the **matrix exponential**.

A linear ODE system has the form:

.. math::

   \frac{d\mathbf{y}}{dt} = A \, \mathbf{y}

where :math:`A` is an :math:`m \times m` matrix of constant coefficients and
:math:`\mathbf{y}(t)` is the state vector. The solution is:

.. math::

   \mathbf{y}(t) = \exp(At) \, \mathbf{y}(0)

.. admonition:: Calculus Aside -- What is a matrix exponential?

   The matrix exponential :math:`\exp(M)` of a square matrix :math:`M` is
   defined by the same power series as the scalar exponential:

   .. math::

      \exp(M) = I + M + \frac{M^2}{2!} + \frac{M^3}{3!} + \cdots
      = \sum_{k=0}^{\infty} \frac{M^k}{k!}

   where :math:`I` is the identity matrix and :math:`M^k` means :math:`M`
   multiplied by itself :math:`k` times. This series always converges for
   any square matrix.

   To verify that :math:`\mathbf{y}(t) = \exp(At)\mathbf{y}(0)` solves
   :math:`d\mathbf{y}/dt = A\mathbf{y}`, differentiate the series
   term by term:

   .. math::

      \frac{d}{dt} \exp(At) = A + A^2 t + \frac{A^3 t^2}{2!} + \cdots
      = A \left(I + At + \frac{A^2 t^2}{2!} + \cdots\right) = A \exp(At)

   So :math:`d\mathbf{y}/dt = A \exp(At) \mathbf{y}(0) = A \mathbf{y}(t)`,
   confirming the solution.

   If :math:`A` is diagonalizable with **eigendecomposition**
   :math:`A = V \Lambda V^{-1}`
   where :math:`\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_m)` is the
   diagonal matrix of eigenvalues, then:

   .. math::

      \exp(At) = V \, \text{diag}(e^{\lambda_1 t}, \ldots, e^{\lambda_m t}) \, V^{-1}

   This identity is useful for symmetric or otherwise well-conditioned matrices,
   especially at repeated times after one decomposition. It is not a general
   prescription: defective matrices are not diagonalizable, and ill-conditioned
   eigenvectors can destroy accuracy. General ``expm`` routines instead use
   stable algorithms such as scaling and squaring.

**Connection to population genetics.** The matrix exponential appears in
several Timepieces:

- **SMC++** (:ref:`smcpp_timepiece`): The probability of :math:`j`
  undistinguished lineages remaining at time :math:`t` is computed via the
  matrix exponential of the lineage rate matrix. The states track the number
  of remaining lineages, and the rate matrix encodes coalescence events.

- **momi2** (:ref:`momi2_timepiece`): The Moran model transition
  probabilities over an epoch of duration :math:`t` are computed as
  :math:`\exp(Qt)`, where :math:`Q` is the Moran rate matrix. The
  eigendecomposition of :math:`Q` is known in closed form, making this
  computation efficient.

- **Migration models**: When :math:`k` populations exchange migrants at
  constant rates, the allele frequency dynamics within each population form
  a linear ODE, and the matrix exponential gives the exact solution.

Let us implement a concrete example: a simple **two-population migration
model** where allele frequencies in two populations evolve under symmetric
migration.

.. code-block:: python

   def migration_matrix(m_rate, n_pops=2):
       """Rate matrix for symmetric migration between n_pops populations.

       Off-diagonal: A[i,j] = m_rate (gain from pop j).
       Diagonal: A[i,i] = -(n_pops-1)*m_rate. Symmetry makes both rows
       and columns sum to zero, conserving the component sum.
       """
       if m_rate < 0 or not isinstance(n_pops, (int, np.integer)) or n_pops < 2:
           raise ValueError("require m_rate >= 0 and integer n_pops >= 2")
       A = np.full((n_pops, n_pops), m_rate, dtype=float)
       np.fill_diagonal(A, -(n_pops - 1) * m_rate)
       return A

   def symmetric_matrix_exponential(A, t):
       """Compute exp(A*t) for a real symmetric matrix."""
       A = np.asarray(A, dtype=float)
       if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.allclose(A, A.T):
           raise ValueError("A must be a real symmetric square matrix")
       eigenvalues, eigenvectors = np.linalg.eigh(A)
       return (eigenvectors * np.exp(eigenvalues * t)) @ eigenvectors.T

   # Two populations with symmetric migration
   m_rate = 0.5  # migration rate
   A = migration_matrix(m_rate)
   print("Rate matrix A:")
   print(A)

   # Initial frequencies: pop1 has allele at frequency 0.8, pop2 at 0.2
   y0 = np.array([0.8, 0.2])

   # Solve using matrix exponential at several time points
   print("\nAllele frequencies over time (matrix exponential):")
   print(f"{'t':>6s}  {'pop1':>8s}  {'pop2':>8s}")
   for t in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
       y_t = symmetric_matrix_exponential(A, t) @ y0
       print(f"{t:6.1f}  {y_t[0]:8.4f}  {y_t[1]:8.4f}")

   # Verify: at equilibrium, both populations should have the same frequency
   # (the mean of the initial frequencies)
   y_eq = symmetric_matrix_exponential(A, 100.0) @ y0
   expected_eq = np.mean(y0)
   print(f"\nEquilibrium frequency: {y_eq[0]:.6f} (expected {expected_eq:.6f})")

   # Compare with eigendecomposition
   eigenvalues, V = np.linalg.eigh(A)
   print(f"\nEigenvalues of A: {eigenvalues}")
   # For symmetric 2x2 migration: eigenvalues are 0 and -2*m_rate
   # The zero eigenvalue corresponds to the stationary distribution
   # The negative eigenvalue governs the rate of convergence to equilibrium

   # Verify matrix exponential via eigendecomposition
   t_test = 2.0
   # exp(A*t) = V * diag(exp(lambda_i * t)) * V^{-1}
   D = np.diag(np.exp(eigenvalues * t_test))
   y_eigen = V @ D @ V.T @ y0
   y_expm = symmetric_matrix_exponential(A, t_test) @ y0
   print(f"\nAt t={t_test}:")
   print(f"  Via expm:              {y_expm}")
   print(f"  Via eigendecomposition: {np.real(y_eigen)}")
   print(f"  Agree: {np.allclose(y_expm, np.real(y_eigen))}")

The symmetric construction extends directly to three or more populations. For
asymmetric migration, fix a row- or column-vector convention first: with the
column-vector equation used here, conservation of the component sum requires
**columns** of :math:`A` to sum to zero. The matrix can then be non-symmetric, so
the ``eigh`` helper above no longer applies; use a general matrix-exponential
routine. Eigenvalues describe modal timescales when the matrix is diagonalizable,
but non-normal systems can also show transients not captured by eigenvalues alone.


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Concept
     - Key Formula or Idea
   * - ODE (initial value problem)
     - :math:`dy/dt = f(y, t), \quad y(t_0) = y_0`
   * - Euler's method
     - :math:`y_{n+1} = y_n + h \, f(y_n, t_n)` -- first-order, :math:`O(h)`
   * - RK4
     - Four-stage method -- fourth-order, :math:`O(h^4)` global error
   * - RK45 (Dormand-Prince)
     - Adaptive step size, ``scipy.integrate.solve_ivp`` default
   * - Systems of ODEs
     - :math:`d\mathbf{y}/dt = \mathbf{f}(\mathbf{y}, t)` -- vector state
   * - Stiffness
     - Disparate timescales; use implicit methods (BDF, backward Euler)
   * - Matrix exponential
     - :math:`\mathbf{y}(t) = \exp(At)\mathbf{y}(0)` for linear systems
   * - Eigendecomposition
     - :math:`\exp(At) = V \, \text{diag}(e^{\lambda_i t}) \, V^{-1}`

These tools are the gear train that drives the moment equations of the moments
Timepiece, the lineage probability system of SMC++, and the Moran model
transitions of momi2. Whenever you see an ODE system in this book, you now
have the vocabulary and the numerical methods to solve it.

Next: :ref:`mcmc` -- the algorithm that lets us explore parameter spaces too
complex for closed-form solutions.
