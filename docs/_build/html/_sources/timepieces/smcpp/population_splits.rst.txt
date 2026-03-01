.. _smcpp_splits:

===================
Population Splits
===================

   *Adding a second dial to the chronograph: cross-population demographic inference.*

From One Population to Two
=============================

Everything we have built so far assumes a single panmictic population. But many
biological questions involve **multiple populations**: When did they diverge? How
did their sizes change before and after the split? Was there gene flow?

SMC++ handles population splits by modifying the ODE system for the undistinguished
lineage counts. The key insight: before the split time :math:`T_{\text{split}}`,
lineages from both populations trace back to the same ancestral population. After
the split (looking forward in time), they are in separate populations and can only
coalesce within their own population.

.. code-block:: text

   Looking backward in time:

   Present -------+------------ t = 0
                  |
   Pop A:   n_A lineages
   Pop B:   n_B lineages
                  |
                  |  (separate populations)
                  |
   Split --------+------------ t = T_split
                  |
                  |  (ancestral population)
                  |  n_A + n_B lineages, size N_anc(t)
                  |
   Past ---------+------------ t -> infinity


The Modified ODE System
=========================

For times :math:`t < T_{\text{split}}` (before the split, looking backward), lineages
from populations A and B are in separate populations with potentially different sizes
:math:`\lambda_A(t)` and :math:`\lambda_B(t)`. They coalesce independently within
their own populations.

Let :math:`p_j^A(t)` and :math:`p_j^B(t)` be the lineage-count probabilities for
populations A and B respectively. Before the split, these evolve independently:

.. math::

   \frac{dp_j^A}{dt} = \frac{1}{\lambda_A(t)} \left[
   -\binom{j}{2} p_j^A(t) + \binom{j+1}{2} p_{j+1}^A(t) \right]

.. math::

   \frac{dp_j^B}{dt} = \frac{1}{\lambda_B(t)} \left[
   -\binom{j}{2} p_j^B(t) + \binom{j+1}{2} p_{j+1}^B(t) \right]

At the split time :math:`T_{\text{split}}`, the two populations merge into one
ancestral population. The lineage counts combine: if population A has :math:`j_A`
lineages and population B has :math:`j_B` lineages at the split, the ancestral
population has :math:`j_A + j_B` lineages. The combined system then evolves under
the ancestral population size :math:`\lambda_{\text{anc}}(t)`.

.. code-block:: python

   import numpy as np
   from scipy.linalg import expm

   def solve_split_ode(n_A, n_B, time_breaks, lambdas_A, lambdas_B,
                        lambdas_anc, t_split):
       """Solve the ODE system for a two-population split model.

       Parameters
       ----------
       n_A : int
           Number of undistinguished haploid lineages from population A.
       n_B : int
           Number of undistinguished haploid lineages from population B.
       time_breaks : array-like
           Time boundaries for piecewise-constant intervals.
       lambdas_A, lambdas_B : array-like
           Population sizes for A and B (pre-split intervals only).
       lambdas_anc : array-like
           Ancestral population sizes (post-split intervals only).
       t_split : float
           Split time in coalescent units.

       Returns
       -------
       h_A : ndarray
           Coalescence rate for a distinguished lineage from pop A.
       h_B : ndarray
           Coalescence rate for a distinguished lineage from pop B.
       """
       Q_A = build_rate_matrix(n_A)
       Q_B = build_rate_matrix(n_B)

       # Initialize: all lineages present
       p_A = np.zeros(n_A)
       p_A[-1] = 1.0
       p_B = np.zeros(n_B)
       p_B[-1] = 1.0

       K = len(time_breaks) - 1
       h_A_values = np.zeros(K)
       h_B_values = np.zeros(K)

       j_A_vals = np.arange(1, n_A + 1)
       j_B_vals = np.arange(1, n_B + 1)

       for k in range(K):
           t_lo = time_breaks[k]
           t_hi = time_breaks[k + 1]
           dt = t_hi - t_lo

           if t_hi <= t_split:
               # Pre-split: populations evolve independently
               lam_A = lambdas_A[k]
               lam_B = lambdas_B[k]

               M_A = expm(dt / lam_A * Q_A)
               p_A = M_A @ p_A

               M_B = expm(dt / lam_B * Q_B)
               p_B = M_B @ p_B

               # h for distinguished from A: only A lineages are partners
               h_A_values[k] = np.dot(j_A_vals, p_A) / lam_A
               # h for distinguished from B: only B lineages are partners
               h_B_values[k] = np.dot(j_B_vals, p_B) / lam_B

           else:
               # Post-split: combined ancestral population
               # Merge lineage counts at the split
               if t_lo < t_split:
                   # This interval spans the split -- handle the boundary
                   pass  # Simplified: assume breaks align with t_split

               # After merging, use combined rate matrix
               n_anc = n_A + n_B
               Q_anc = build_rate_matrix(n_anc)

               # Combine the lineage count distributions
               # (convolution of A and B distributions)
               # For simplicity, use the expected counts
               lam_anc = lambdas_anc[k - len(lambdas_A)]

               p_anc = np.zeros(n_anc)
               # Initial condition for ancestral: convolution of p_A and p_B
               for ja in range(n_A):
                   for jb in range(n_B):
                       j_total = (ja + 1) + (jb + 1)  # 1-indexed counts
                       if j_total <= n_anc:
                           p_anc[j_total - 1] += p_A[ja] * p_B[jb]

               M_anc = expm(dt / lam_anc * Q_anc)
               p_anc = M_anc @ p_anc

               j_anc_vals = np.arange(1, n_anc + 1)
               h_val = np.dot(j_anc_vals, p_anc) / lam_anc
               h_A_values[k] = h_val
               h_B_values[k] = h_val

       return h_A_values, h_B_values

   # Import helper
   from smcpp_ode_helpers import build_rate_matrix  # conceptual import


Cross-Population TMRCA
========================

When the distinguished lineage is from population A and some undistinguished
lineages are from population B, the coalescence of the distinguished lineage with
a B-lineage can only happen **after the split** (looking backward), when both sets
of lineages are in the same ancestral population.

This means the cross-population TMRCA has a minimum value of :math:`T_{\text{split}}`.
The distribution of cross-population coalescence times has a gap between 0 and
:math:`T_{\text{split}}` -- no coalescence can happen during this period because the
lineages are in different populations. This gap is directly informative about the
split time.

.. math::

   P(T_{\text{cross}} > t) = \begin{cases}
   1 & \text{if } t < T_{\text{split}} \\
   \exp\!\left(-\int_{T_{\text{split}}}^{t} h_{\text{anc}}(u) \, du\right) &
   \text{if } t \geq T_{\text{split}}
   \end{cases}

.. code-block:: python

   def cross_population_survival(t, t_split, h_anc_func):
       """Survival function for cross-population TMRCA.

       Parameters
       ----------
       t : float
           Time point.
       t_split : float
           Population split time.
       h_anc_func : callable
           Ancestral coalescence rate function.

       Returns
       -------
       float
           P(T_cross > t).
       """
       if t < t_split:
           return 1.0
       else:
           # Numerical integration of h_anc from t_split to t
           from scipy.integrate import quad
           integral, _ = quad(h_anc_func, t_split, t)
           return np.exp(-integral)


Joint Estimation
==================

For a split model, SMC++ jointly estimates:

- :math:`\lambda_A(t)` -- population size history of population A
- :math:`\lambda_B(t)` -- population size history of population B
- :math:`\lambda_{\text{anc}}(t)` -- ancestral population size history
- :math:`T_{\text{split}}` -- the divergence time

The composite likelihood now sums over distinguished lineages from both populations:

.. math::

   \ell_C = \sum_{i \in A} \log P(\mathbf{X}^{(i)} \mid \boldsymbol{\lambda}_A,
   \boldsymbol{\lambda}_{\text{anc}}, T_{\text{split}}) +
   \sum_{i \in B} \log P(\mathbf{X}^{(i)} \mid \boldsymbol{\lambda}_B,
   \boldsymbol{\lambda}_{\text{anc}}, T_{\text{split}})

where each term is computed using the appropriate modified ODE system.

.. code-block:: python

   def fit_split_model(data_A, data_B, time_breaks, theta, rho):
       """Fit SMC++ split model to two-population data.

       Parameters
       ----------
       data_A : list of ndarray
           Observation sequences from population A samples.
       data_B : list of ndarray
           Observation sequences from population B samples.
       time_breaks : array-like
           Time interval boundaries.
       theta, rho : float
           Scaled mutation and recombination rates.

       Returns
       -------
       dict
           Estimated parameters: lambdas_A, lambdas_B, lambdas_anc, t_split.
       """
       K = len(time_breaks) - 1

       def objective(params):
           # Unpack parameters
           log_lambdas_A = params[:K]
           log_lambdas_B = params[K:2*K]
           log_lambdas_anc = params[2*K:3*K]
           log_t_split = params[3*K]

           lambdas_A = np.exp(log_lambdas_A)
           lambdas_B = np.exp(log_lambdas_B)
           lambdas_anc = np.exp(log_lambdas_anc)
           t_split = np.exp(log_t_split)

           # Compute composite log-likelihood for both populations
           ll = 0.0

           # Distinguished from A, undistinguished from A + B
           # (simplified: separate within-population terms)
           for data_i in data_A:
               # HMM forward with population-A-specific transitions
               ll += forward_log_likelihood(data_i, time_breaks, lambdas_A, theta, rho)

           for data_i in data_B:
               ll += forward_log_likelihood(data_i, time_breaks, lambdas_B, theta, rho)

           return -ll  # Minimize negative log-likelihood

       # Initial guess
       x0 = np.zeros(3 * K + 1)
       x0[3*K] = np.log(0.5)  # Initial split time guess

       from scipy.optimize import minimize
       result = minimize(objective, x0, method='L-BFGS-B')

       return {
           'lambdas_A': np.exp(result.x[:K]),
           'lambdas_B': np.exp(result.x[K:2*K]),
           'lambdas_anc': np.exp(result.x[2*K:3*K]),
           't_split': np.exp(result.x[3*K]),
       }

   def forward_log_likelihood(obs, time_breaks, lambdas, theta, rho):
       """HMM forward algorithm log-likelihood (stub for split model)."""
       # Same as in composite_log_likelihood but for a single sequence
       return 0.0  # Placeholder


Comparison with Other Split-Time Methods
==========================================

Several methods estimate population split times. SMC++ occupies a specific niche:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Method
     - Approach
     - Trade-off
   * - **SMC++**
     - Distinguished lineage + ODE
     - Full :math:`N(t)` history + split time; unphased data; scales to many samples
   * - **MSMC-IM**
     - Pairwise HMM with cross-coalescence
     - Richer migration model; requires phased data; limited samples
   * - **momi2**
     - Coalescent SFS computation
     - Flexible demography; summary statistic (SFS) only; no per-locus signal
   * - **moments**
     - Moment equations for the SFS
     - Similar to momi2; forward-time computation

SMC++ is unique in using per-locus genealogical signal (through the HMM) while
still scaling to many samples (through composite likelihood). It provides
simultaneous estimates of :math:`N(t)` and :math:`T_{\text{split}}`, avoiding
the need to fix one while estimating the other.


Summary
========

Population splits add one more complication to the SMC++ chronograph: a second
dial for the second population. The mathematical extension is natural -- the ODE
system splits into independent equations before the split time and a combined
equation after it. The cross-population TMRCA distribution is directly informative
about the divergence time, and the composite likelihood framework extends
straightforwardly to multiple populations.

With this final gear in place, SMC++ is a complete chronograph: it reads
population size history from multiple unphased diploid genomes with sharper
resolution than PSMC, and it can estimate population divergence times from
cross-population comparisons. All built on the same foundation -- the coalescent
process, discretized into an HMM, decoded by numerical optimization.

*The chronograph is assembled. Every sub-dial reads a different population,
but they are all driven by the same escapement: the coalescent ticking away
beneath the surface of our genomes.*
