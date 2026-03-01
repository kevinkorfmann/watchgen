.. _gamma_smc_flow_field:

==========================
The Flow Field
==========================

   *The gear train: precomputed machinery that transmits the transition dynamics.*

In the previous chapter, we showed that the SMC transition step transforms
a :math:`\text{Gamma}(\alpha, \beta)` distribution into something that is
approximately :math:`\text{Gamma}(\alpha', \beta')`, with the change
:math:`(\alpha' - \alpha, \beta' - \beta) = \rho \cdot (u, v)` determined by
a least-squares projection. The pair :math:`(u, v)` depends on
:math:`(\alpha, \beta)` but not on any model parameters.

This chapter describes how Gamma-SMC precomputes the mapping
:math:`(\alpha, \beta) \mapsto (\alpha', \beta')` over a two-dimensional
grid, creating a **flow field** :math:`\mathcal{F}` that can be queried during
inference via fast interpolation. If the gamma approximation is the
escapement, the flow field is the gear train -- the precision-machined
mechanism that transmits the tick of the escapement through the rest of the
watch.


What Is a Flow Field?
========================

A flow field is a function that assigns a **direction and magnitude** to every
point in a space. In fluid dynamics, a flow field describes the velocity of
a fluid at each point -- where the fluid is going and how fast. In Gamma-SMC,
the "fluid" is the gamma posterior, and the flow field describes how the
posterior parameters change when a recombination event is possible.

.. admonition:: Plain-language summary -- The flow field as a lookup table

   The flow field is, at its core, a precomputed lookup table. For every
   possible state of our belief about the TMRCA (described by the gamma
   parameters :math:`\alpha` and :math:`\beta`), it tells us: *if
   recombination could have occurred at this position, how should we update
   our belief?* The answer depends on the current belief -- if we are very
   certain the TMRCA is recent, recombination shifts our estimate differently
   than if we are uncertain or think the TMRCA is ancient. By precomputing
   this for all possible beliefs on a grid, Gamma-SMC avoids expensive
   calculations during the actual genomic scan.

Concretely, the flow field :math:`\mathcal{F}` is a mapping:

.. math::

   \mathcal{F}: (l_\mu, l_C) \mapsto (l_\mu', l_C')

where :math:`(l_\mu, l_C) = (\log_{10}(\alpha/\beta), \log_{10}(1/\sqrt{\alpha}))`
are the log-mean and log-CV coordinates of the current gamma distribution,
and :math:`(l_\mu', l_C')` are the coordinates after one SMC transition step.


The Grid
==========

The flow field is evaluated over a rectangular grid:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Axis
     - Range
     - Points
     - Physical meaning
   * - :math:`l_\mu`
     - :math:`[-5, 2]`
     - 51
     - Log-mean TMRCA from :math:`10^{-5}` to :math:`10^2` coalescent units
   * - :math:`l_C`
     - :math:`[-2, 0]`
     - 50
     - Log-CV from :math:`10^{-2}` (very certain) to :math:`1` (uncertain)

This gives :math:`51 \times 50 = 2{,}550` grid points, at each of which
:math:`\mathcal{F}` is evaluated. The grid boundaries are chosen to cover
the range of posteriors encountered in practice:

- **Mean TMRCA** from :math:`10^{-5}` (essentially zero -- a very recent
  common ancestor) to :math:`100` coalescent units (:math:`200 N_e`
  generations into the past). Genomic positions with TMRCAs outside this
  range are vanishingly rare.

- **CV** from :math:`0.01` (extremely certain -- a sharp posterior with very
  little uncertainty) to :math:`1.0` (as uncertain as the prior). A CV of
  :math:`1` corresponds to :math:`\alpha = 1`, which is the exponential
  distribution -- the prior itself. The posterior cannot be *more* uncertain
  than the prior (in the absence of pathological approximation errors).

Values with :math:`C_V > 1` (i.e., :math:`\alpha < 1`) are excluded because
they would give an infinite density at :math:`t = 0`, which is unphysical.


Computing Each Grid Point
============================

At each grid point :math:`(l_\mu, l_C)`, the computation proceeds as follows:

1. **Convert to** :math:`(\alpha, \beta)`:

   .. math::

      \alpha &= 10^{-2 l_C} \\
      \beta &= \alpha \cdot 10^{-l_\mu}

2. **Evaluate the perturbation** :math:`(p_{\alpha,\beta}(x) - f_{\alpha,\beta}(x))/\rho`
   at 2,000 values of :math:`x` covering the main support of
   :math:`f_{\alpha,\beta}`. This uses the closed-form expression derived in
   :ref:`gamma_smc_gamma_approximation`, involving Kummer's confluent
   hypergeometric function :math:`M(a, b, z)`.

3. **Evaluate the partial derivatives** :math:`\partial f / \partial \alpha`
   and :math:`\partial f / \partial \beta` (or equivalently,
   :math:`\partial f / \partial \log_{10} \alpha` and
   :math:`\partial f / \partial \log_{10} \beta`) at the same 2,000 points.

4. **Solve the least-squares problem** to find the flow
   :math:`(\Delta \log_{10} \alpha, \Delta \log_{10} \beta)`.

5. **Convert to** :math:`(l_\mu', l_C')` using the linear relationship:

   .. math::

      \Delta l_\mu &= \Delta \log_{10}(\alpha) - \Delta \log_{10}(\beta) \\
      \Delta l_C &= -0.5 \cdot \Delta \log_{10}(\alpha)

.. code-block:: python

   import numpy as np
   from scipy.stats import gamma as gamma_dist
   from scipy.special import digamma

   def gamma_pdf_partials(x, alpha, beta):
       """Evaluate the gamma PDF and its partial derivatives.

       Parameters
       ----------
       x : ndarray
           Points at which to evaluate.
       alpha : float
           Shape parameter.
       beta : float
           Rate parameter.

       Returns
       -------
       f : ndarray
           Gamma PDF values.
       df_dalpha : ndarray
           Partial derivative with respect to alpha.
       df_dbeta : ndarray
           Partial derivative with respect to beta.
       """
       f = gamma_dist.pdf(x, a=alpha, scale=1.0 / beta)
       # d/dalpha [Gamma PDF] = f * (-psi(alpha) + log(beta) + log(x))
       df_dalpha = f * (-digamma(alpha) + np.log(beta) + np.log(x + 1e-300))
       # d/dbeta [Gamma PDF] = f * (alpha/beta - x)
       df_dbeta = f * (alpha / beta - x)
       return f, df_dalpha, df_dbeta

   def compute_flow_at_point(l_mu, l_C, n_eval=2000):
       """Compute the flow displacement at one grid point.

       This is a simplified skeleton showing the least-squares structure.
       The full implementation requires Kummer's hypergeometric function
       M(a, b, z) via the Arb library for the perturbation evaluation.

       Parameters
       ----------
       l_mu, l_C : float
           Log-mean and log-CV coordinates.
       n_eval : int
           Number of evaluation points for least-squares.

       Returns
       -------
       delta_l_mu, delta_l_C : float
           Flow displacement in log-coordinates.
       """
       # Step 1: convert to (alpha, beta)
       alpha = 10.0 ** (-2 * l_C)
       beta = alpha * 10.0 ** (-l_mu)

       # Step 2: set up evaluation grid over the support of Gamma(alpha, beta)
       mean = alpha / beta
       std = np.sqrt(alpha) / beta
       x = np.linspace(max(1e-10, mean - 4*std), mean + 6*std, n_eval)

       # Step 3: evaluate partial derivatives
       f, df_da, df_db = gamma_pdf_partials(x, alpha, beta)

       # Step 4: in the full implementation, evaluate the perturbation
       # (p_{alpha,beta}(x) - f_{alpha,beta}(x)) / rho using Kummer's M.
       # Here we use a placeholder zero perturbation for illustration.
       perturbation = np.zeros_like(x)  # placeholder

       # Step 5: solve least-squares for (u, v) in log10(alpha), log10(beta)
       A = np.column_stack([df_da, df_db])
       result = np.linalg.lstsq(A, perturbation, rcond=None)
       delta_log_a, delta_log_b = result[0]

       # Step 6: convert to (delta_l_mu, delta_l_C)
       delta_l_mu = delta_log_a - delta_log_b
       delta_l_C = -0.5 * delta_log_a
       return delta_l_mu, delta_l_C

   # Demonstrate the structure (flow is zero with placeholder perturbation)
   dl_mu, dl_C = compute_flow_at_point(0.0, -0.5)
   print(f"Flow at (l_mu=0, l_C=-0.5): "
         f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")
   print("(Zero because we used a placeholder perturbation)")

The result is stored as the displacement
:math:`(\Delta l_\mu, \Delta l_C) = (l_\mu' - l_\mu, l_C' - l_C)` at each
grid point.

.. admonition:: Numerical precision: the Arb library

   The confluent hypergeometric function :math:`M(\alpha, \alpha+1, z)` can
   have very large arguments (when :math:`\beta` and :math:`t` are both
   large), making standard floating-point evaluation unreliable. Gamma-SMC
   uses the **Arb library** (Johansson, 2017), which performs
   ball arithmetic -- rigorous interval arithmetic with guaranteed error
   bounds. Each number is represented as a midpoint and a radius, and all
   operations propagate error bounds correctly. This ensures that the flow
   field is computed to full precision even at extreme grid points.


Querying the Flow Field
=========================

During inference, the current gamma posterior :math:`(l_\mu, l_C)` will
generally *not* lie exactly on a grid point. Gamma-SMC uses **bilinear
interpolation** over the four nearest grid points to obtain the flow:

.. math::

   \mathcal{F}(l_\mu, l_C) \approx
   (1-s)(1-t) \cdot \mathcal{F}_{ij}
   + s(1-t) \cdot \mathcal{F}_{(i+1)j}
   + (1-s)t \cdot \mathcal{F}_{i(j+1)}
   + st \cdot \mathcal{F}_{(i+1)(j+1)}

where :math:`s` and :math:`t` are the fractional positions within the grid
cell and :math:`i, j` are the indices of the lower-left corner.

**Clipping.** If either :math:`l_\mu` or :math:`l_C` falls outside the grid
boundaries, it is clipped to the nearest boundary value. This prevents
extrapolation beyond the precomputed grid and provides a conservative fallback
for extreme posteriors.

.. code-block:: python

   import numpy as np

   class FlowField:
       """A precomputed flow field over a (l_mu, l_C) grid.

       Parameters
       ----------
       l_mu_grid : ndarray, shape (n_mu,)
           Log-mean grid values.
       l_C_grid : ndarray, shape (n_C,)
           Log-CV grid values.
       delta_l_mu : ndarray, shape (n_mu, n_C)
           Flow displacement in l_mu at each grid point.
       delta_l_C : ndarray, shape (n_mu, n_C)
           Flow displacement in l_C at each grid point.
       """

       def __init__(self, l_mu_grid, l_C_grid, delta_l_mu, delta_l_C):
           self.l_mu_grid = l_mu_grid
           self.l_C_grid = l_C_grid
           self.delta_l_mu = delta_l_mu
           self.delta_l_C = delta_l_C

       def query(self, l_mu, l_C):
           """Query the flow field via bilinear interpolation with clipping.

           Parameters
           ----------
           l_mu : float
               Log-mean coordinate of the current gamma distribution.
           l_C : float
               Log-CV coordinate of the current gamma distribution.

           Returns
           -------
           dl_mu, dl_C : float
               Interpolated flow displacement.
           """
           # Clip to grid boundaries
           l_mu_c = np.clip(l_mu, self.l_mu_grid[0], self.l_mu_grid[-1])
           l_C_c = np.clip(l_C, self.l_C_grid[0], self.l_C_grid[-1])

           # Find the lower-left grid cell indices
           i = np.searchsorted(self.l_mu_grid, l_mu_c) - 1
           j = np.searchsorted(self.l_C_grid, l_C_c) - 1
           i = np.clip(i, 0, len(self.l_mu_grid) - 2)
           j = np.clip(j, 0, len(self.l_C_grid) - 2)

           # Fractional positions within the cell: s along l_mu, t along l_C
           s = (l_mu_c - self.l_mu_grid[i]) / (self.l_mu_grid[i+1] - self.l_mu_grid[i])
           t = (l_C_c - self.l_C_grid[j]) / (self.l_C_grid[j+1] - self.l_C_grid[j])

           # Bilinear interpolation for each displacement component
           def interp(field):
               return ((1-s)*(1-t) * field[i, j]
                       + s*(1-t) * field[i+1, j]
                       + (1-s)*t * field[i, j+1]
                       + s*t * field[i+1, j+1])

           return interp(self.delta_l_mu), interp(self.delta_l_C)

   # Build a small demonstration flow field (placeholder displacements)
   l_mu_grid = np.linspace(-5, 2, 51)
   l_C_grid = np.linspace(-2, 0, 50)
   # In the real implementation these come from the precomputed PDE solutions;
   # here we use a simple model: flow pushes l_C upward (more uncertainty)
   dl_mu_grid = np.zeros((51, 50))
   dl_C_grid = 0.01 * np.ones((51, 50))  # placeholder: constant upward drift

   ff = FlowField(l_mu_grid, l_C_grid, dl_mu_grid, dl_C_grid)
   dl_mu, dl_C = ff.query(0.5, -0.8)
   print(f"Flow at (l_mu=0.5, l_C=-0.8): "
         f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")
   # Clipping: query outside the grid
   dl_mu, dl_C = ff.query(5.0, -3.0)
   print(f"Flow at (l_mu=5.0, l_C=-3.0) [clipped]: "
         f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")

.. admonition:: Why bilinear interpolation?

   Bilinear interpolation is the simplest scheme that is continuous across
   grid cell boundaries. Higher-order interpolation (e.g., bicubic) would
   give smoother results but would require larger stencils and more
   computation per query. Since the flow field is smooth (it is derived from
   smooth PDEs), bilinear interpolation is sufficient and keeps each
   transition step :math:`O(1)`.


Visualizing the Flow Field
=============================

The flow field can be visualized as a vector plot over the
:math:`(l_\mu, l_C)` grid. At each point, an arrow shows the direction and
magnitude of the change in gamma parameters caused by one transition step.

.. code-block:: text

   l_C (log CV)
    0 +----+----+----+----+----+----+----+
      | \  | \  | \  |  \ |  \ |  \ |  \ |
      |  \ |  \ |  \ |   \|   \|   \|   \|
   -1 +----+----+----+----+----+----+----+
      |  - |  - |  - | -  | -  | -  | -  |
      |  - |  - |  - | -  | -  | -  | -  |
   -2 +----+----+----+----+----+----+----+
     -5   -4   -3   -2   -1    0    1    2
                    l_mu (log mean)

The general pattern:

- At high CV (top of grid, near the prior), the flow has large magnitude:
  the transition step significantly changes the posterior because the current
  belief is weak.
- At low CV (bottom of grid, sharp posterior), the flow is small: the
  posterior is so concentrated that one transition step barely shifts it.
- The flow generally points **upward** (increasing CV): recombination
  introduces uncertainty, broadening the posterior.

.. admonition:: Biology Aside -- Why recombination increases uncertainty

   Without recombination, consecutive genomic positions share the same
   genealogy, so each new observation (het or hom) steadily sharpens our
   estimate of the TMRCA. Recombination disrupts this: it creates the
   possibility that the TMRCA at the next position is *completely different*
   from the current one. This injects uncertainty -- we cannot be as sure
   about the TMRCA after a potential recombination event as we were before.
   The flow field captures exactly how much uncertainty recombination adds,
   depending on how confident we were to begin with. In regions of the genome
   with high recombination rates, the TMRCA changes frequently and the
   posterior stays broad; in recombination deserts, the posterior narrows
   sharply as evidence accumulates.


Parameter Independence
========================

The flow field :math:`\mathcal{F}` is **independent of all model parameters**
:math:`\theta`, :math:`\rho`, and :math:`N_e`. This is because:

- The transition density :math:`p(t \mid s)` under constant population size
  has no free parameters (it is fully determined by the coalescent process).
- The recombination rate :math:`\rho` factors out: the flow is defined as the
  :math:`O(\rho)` perturbation divided by :math:`\rho`, so :math:`\rho`
  cancels.
- The mutation rate :math:`\theta` does not appear in the transition step --
  it only enters through the emission step.

This means the flow field is computed **once** and can be shipped as a
precomputed data file alongside the software. Users never need to recompute
it.


Summary
=========

The flow field :math:`\mathcal{F}` is a 2D vector field over
:math:`(l_\mu, l_C)` space that encodes the effect of one SMC transition
step on gamma-distributed posteriors. It is:

- **Precomputed** over a :math:`51 \times 50` grid using high-precision
  arithmetic
- **Parameter-independent** -- the same field applies to any
  :math:`\theta, \rho, N_e`
- **Queried** during inference via bilinear interpolation with boundary
  clipping
- Evaluated in :math:`O(1)` per query, making it the constant-time
  "gear train" that transmits transition dynamics through the mechanism

With the flow field in hand, we can now assemble the complete forward-backward
algorithm: :ref:`gamma_smc_forward_backward`.
