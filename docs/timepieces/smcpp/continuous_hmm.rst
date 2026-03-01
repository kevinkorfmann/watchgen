.. _smcpp_hmm:

===========================
The Continuous HMM
===========================

   *From ODE rates to a working inference engine.*

Building the Transition Matrix
================================

In :ref:`Timepiece I <psmc_timepiece>`, PSMC's transition matrix encodes how the
coalescence time :math:`T` changes between adjacent genomic positions. The same
structure applies in SMC++, but with a modified transition density that accounts
for the undistinguished lineages.

Recall the SMC transition mechanism from the :ref:`SMC prerequisite <smc>`:

1. At position :math:`a`, the coalescence time is :math:`T_a = s` (the current state).
2. With probability :math:`1 - e^{-\rho s}`, a recombination occurs on the two
   branches (total length :math:`2s`).
3. If recombination occurs, one lineage detaches at a time :math:`u \in [0, s]` and
   must re-coalesce with the remaining lineage.
4. The new coalescence time :math:`T_{a+1}` depends on when re-coalescence happens.

In PSMC, re-coalescence involves just two lineages: the detached lineage and the
one it was separated from. In SMC++, the detached lineage may re-coalesce with
**any** of the :math:`j` lineages present at time :math:`u` -- the undistinguished
lineages plus the one remaining from the original pair. The re-coalescence rate is
therefore :math:`h(u)` rather than :math:`1/\lambda(u)`.

This modifies the transition density. Where PSMC has:

.. math::

   q_{\text{PSMC}}(t \mid s) = \frac{1}{\lambda(t)} \, \exp\!\left(-\int_s^t
   \frac{du}{\lambda(u)}\right) \quad \text{(for } t > s \text{)}

SMC++ has:

.. math::

   q_{\text{SMC++}}(t \mid s) = h(t) \, \exp\!\left(-\int_s^t h(u) \, du\right)
   \quad \text{(for } t > s \text{)}

The structure is identical -- both are **hazard-function formulas** from
survival analysis: the probability of re-coalescence at time :math:`t` equals
the instantaneous rate at :math:`t` (first factor) times the probability of
*not* having re-coalesced before :math:`t` (exponential factor). The only
difference is the rate function:

- **PSMC** (:math:`1/\lambda(t)`): In the pairwise case, the detached lineage
  has exactly one partner to coalesce with, and the coalescence rate with that
  partner is :math:`1/\lambda(t)` (the inverse population size at time :math:`t`).

- **SMC++** (:math:`h(t)`): The detached lineage can coalesce with any of the
  :math:`j` lineages present at time :math:`t` -- the :math:`j - 1`
  undistinguished lineages plus the one remaining from the original pair. The
  rate is :math:`h(t) = E[J(t)] / \lambda(t)`, where :math:`E[J(t)]` is the
  expected number of available lineages from the ODE system. When :math:`n = 1`
  (no undistinguished lineages), :math:`E[J(t)] = 1` and :math:`h(t)` reduces
  to :math:`1/\lambda(t)` -- recovering PSMC exactly.

.. admonition:: Biology Aside -- Why more lineages help with recent history

   In PSMC, the single pair of lineages coalesces quickly, so the HMM spends
   most of its time in deep coalescence-time states and has little power to
   resolve recent population size changes. In SMC++, the undistinguished
   lineages dramatically increase the coalescence rate in the recent past
   (many lineages means rapid coalescence), giving the HMM much more data
   about recent history. This is why SMC++ can resolve population size changes
   down to ~1,000 years ago while PSMC cannot see below ~20,000 years.


Discretization
================

Following PSMC, we discretize the coalescence time into :math:`K` intervals
:math:`[t_0, t_1), [t_1, t_2), \ldots, [t_{K-1}, t_K)` with log-spacing. The
transition matrix :math:`P_{kl}` gives the probability of transitioning from
interval :math:`k` to interval :math:`l` between adjacent positions.

The discretized transition probability is:

.. math::

   P_{kl} = (1 - r_k) \, \delta_{kl} + r_k \, q_{kl}

where :math:`r_k` is the probability of recombination given coalescence time in
interval :math:`k`, and :math:`q_{kl}` is the probability that re-coalescence
lands in interval :math:`l`.

The key difference from PSMC is that :math:`q_{kl}` is computed using :math:`h(t)`
instead of :math:`1/\lambda(t)`:

.. code-block:: python

   import numpy as np
   from scipy.linalg import expm

   def compute_transition_matrix(time_breaks, lambdas, rho, n_undist):
       """Build the SMC++ transition matrix.

       Parameters
       ----------
       time_breaks : array-like
           K+1 time boundaries [t_0, ..., t_K].
       lambdas : array-like
           Relative population sizes in each time interval.
       rho : float
           Scaled recombination rate per bin.
       n_undist : int
           Number of undistinguished lineages.

       Returns
       -------
       P : ndarray of shape (K, K)
           Transition matrix.
       """
       K = len(time_breaks) - 1

       # First solve the ODE to get h(t) at each time break
       Q_rate = build_rate_matrix(n_undist)
       p0 = np.zeros(n_undist)
       p0[-1] = 1.0

       # Compute h(t) at midpoints of each interval
       h = np.zeros(K)
       p_current = p0.copy()
       for k in range(K):
           dt = time_breaks[k + 1] - time_breaks[k]
           lam = lambdas[k]
           # Expected undistinguished lineages at midpoint
           M = expm(dt / (2 * lam) * Q_rate)
           p_mid = M @ p_current
           j_values = np.arange(1, n_undist + 1)
           h[k] = np.dot(j_values, p_mid) / lam
           # Advance to end of interval
           M_full = expm(dt / lam * Q_rate)
           p_current = M_full @ p_current

       # Build transition matrix
       P = np.zeros((K, K))

       for k in range(K):
           t_mid = (time_breaks[k] + time_breaks[k + 1]) / 2
           lam_k = lambdas[k]

           # Recombination probability: 1 - exp(-rho * t_mid)
           r_k = 1 - np.exp(-rho * t_mid)

           # No recombination: stay in same state
           P[k, k] += 1 - r_k

           # With recombination: transition to new state via h(t)
           for l in range(K):
               dt_l = time_breaks[l + 1] - time_breaks[l]
               # Approximate: probability of landing in interval l
               # proportional to h * exp(-integral of h) * interval width
               q_kl = h[l] * np.exp(-sum(
                   h[m] * (time_breaks[m + 1] - time_breaks[m])
                   for m in range(l)
               )) * dt_l
               P[k, l] += r_k * q_kl

       # Normalize rows
       P = P / P.sum(axis=1, keepdims=True)
       return P

   # Import from ode_system chapter
   from smcpp_ode_helpers import build_rate_matrix  # conceptual import


Emission Probabilities
========================

The emission probabilities are essentially the same as in PSMC. For a bin with
coalescence time in interval :math:`k`, with midpoint time :math:`t_k^*`:

.. math::

   e_k(1) = 1 - e^{-\theta \, t_k^*}, \qquad e_k(0) = e^{-\theta \, t_k^*}

where :math:`\theta` is the scaled mutation rate per bin.

For **unphased diploid** genotypes at a segregating site, the emission probabilities
are modified to account for the ambiguity of which allele belongs to the
distinguished lineage. The emission for a genotype :math:`g \in \{0, 1, 2\}` (count
of derived alleles) at the distinguished individual, given that the derived allele
is present in some undistinguished lineages:

.. code-block:: python

   def emission_probability(genotype, t, theta, allele_count, n_undist):
       """Emission probability for unphased diploid data at the distinguished individual.

       Parameters
       ----------
       genotype : int
           0, 1, or 2 (count of derived alleles at the focal individual).
       t : float
           Coalescence time of the distinguished lineage.
       theta : float
           Scaled mutation rate per bin.
       allele_count : int
           Number of derived alleles observed in the undistinguished panel.
       n_undist : int
           Total number of undistinguished haplotypes.

       Returns
       -------
       float
           P(genotype, allele_count | T = t).
       """
       p_mut = 1 - np.exp(-theta * t)

       # For the distinguished haplotype:
       # If genotype = 0: both alleles ancestral -> (1 - p_mut)
       # If genotype = 1: one derived, one ancestral -> p_mut (approximately)
       # If genotype = 2: both derived -> p_mut (rare case)

       # Simplified emission (full model integrates over allele assignments)
       if genotype == 0:
           return np.exp(-theta * t)
       elif genotype == 1:
           return 1 - np.exp(-theta * t)
       else:
           return (1 - np.exp(-theta * t)) ** 2


Composite Likelihood
======================

SMC++ does not compute a single joint likelihood over all samples. Instead, it
forms a **composite likelihood**: the product of marginal likelihoods computed
for each choice of distinguished lineage.

For :math:`n` diploid samples, let :math:`\mathbf{X}^{(i)}` denote the data when
individual :math:`i` is the distinguished sample and the remaining :math:`n - 1`
individuals form the undistinguished panel. The composite log-likelihood is:

.. math::

   \ell_C(\boldsymbol{\lambda}) = \sum_{i=1}^{n} \ell_i(\boldsymbol{\lambda})
   = \sum_{i=1}^{n} \log P\!\left(\mathbf{X}^{(i)} \mid \boldsymbol{\lambda}\right)

Each term :math:`\ell_i` is computed using a standard HMM forward algorithm, with the
transition matrix and emission probabilities specific to the :math:`(n - 1)`-lineage
background.

.. code-block:: python

   def composite_log_likelihood(data, time_breaks, lambdas, theta, rho):
       """Compute the composite log-likelihood for SMC++.

       Parameters
       ----------
       data : list of ndarray
           data[i] is the observation sequence for the i-th distinguished sample.
       time_breaks : array-like
           Time interval boundaries.
       lambdas : array-like
           Piecewise-constant population sizes.
       theta : float
           Scaled mutation rate.
       rho : float
           Scaled recombination rate.

       Returns
       -------
       float
           Composite log-likelihood.
       """
       n_samples = len(data)
       n_undist = 2 * n_samples - 1  # Haploid lineages minus the distinguished one

       K = len(time_breaks) - 1
       total_ll = 0.0

       for i in range(n_samples):
           # Build HMM for this distinguished sample
           P = compute_transition_matrix(time_breaks, lambdas, rho, n_undist)

           # Initial distribution (stationary)
           pi = np.ones(K) / K  # Simplified; true stationary from h(t)

           # Forward algorithm
           obs = data[i]
           L = len(obs)
           alpha = np.zeros((L, K))

           # Initialize
           for k in range(K):
               t_mid = (time_breaks[k] + time_breaks[k + 1]) / 2
               alpha[0, k] = pi[k] * emission_probability(obs[0], t_mid, theta, 0, n_undist)

           # Scale for numerical stability
           scale = np.zeros(L)
           scale[0] = alpha[0].sum()
           alpha[0] /= scale[0]

           # Forward recursion
           for a in range(1, L):
               for l in range(K):
                   alpha[a, l] = sum(alpha[a - 1, k] * P[k, l] for k in range(K))
                   t_mid = (time_breaks[l] + time_breaks[l + 1]) / 2
                   alpha[a, l] *= emission_probability(obs[a], t_mid, theta, 0, n_undist)
               scale[a] = alpha[a].sum()
               if scale[a] > 0:
                   alpha[a] /= scale[a]

           total_ll += np.sum(np.log(scale[scale > 0]))

       return total_ll


Gradient-Based Optimization
==============================

Unlike PSMC, which uses the EM algorithm, SMC++ uses **gradient-based optimization**
(L-BFGS-B or similar). The gradient of the composite log-likelihood with respect to
the population size parameters :math:`\lambda_k` can be computed efficiently using:

1. **Automatic differentiation** through the matrix exponential (using the
   eigendecomposition from the previous chapter).
2. **The forward-backward algorithm** to compute expected sufficient statistics.

The optimization proceeds as:

.. code-block:: text

   Initialize lambda_k = 1 for all k
                |
                v
   +-------> Solve ODE system for current lambda
   |                |
   |                v
   |         Build HMM (transitions from h(t), emissions from theta)
   |                |
   |                v
   |         Forward algorithm -> composite log-likelihood
   |                |
   |                v
   |         Compute gradient d(ell_C) / d(lambda_k)
   |                |
   |                v
   |         L-BFGS-B update: lambda <- lambda + step * direction
   |                |
   |         Converged?
   |         NO ---+
   |
   YES
   |
   v
   Output: optimized lambda_0, ..., lambda_K

The use of L-BFGS-B instead of EM has two advantages:

- **Faster convergence**: L-BFGS-B typically converges in fewer iterations than EM,
  especially near the optimum where EM's linear convergence is slow.
- **Regularization**: It is straightforward to add penalty terms (e.g., smoothness
  penalties on :math:`\lambda_k`) to the objective function, which helps prevent
  overfitting.

.. code-block:: python

   from scipy.optimize import minimize

   def fit_smcpp(data, time_breaks, theta, rho, max_iter=100):
       """Fit SMC++ model using L-BFGS-B optimization.

       Parameters
       ----------
       data : list of ndarray
           Observation sequences for each distinguished sample.
       time_breaks : array-like
           Time interval boundaries.
       theta : float
           Scaled mutation rate (fixed).
       rho : float
           Scaled recombination rate (fixed).
       max_iter : int
           Maximum optimization iterations.

       Returns
       -------
       lambdas : ndarray
           Estimated piecewise-constant population sizes.
       """
       K = len(time_breaks) - 1

       # Optimize in log-space for positivity
       def objective(log_lambdas):
           lambdas = np.exp(log_lambdas)
           # Negative log-likelihood (minimize)
           return -composite_log_likelihood(data, time_breaks, lambdas, theta, rho)

       # Initial guess: constant population
       x0 = np.zeros(K)

       result = minimize(
           objective,
           x0,
           method='L-BFGS-B',
           options={'maxiter': max_iter, 'disp': True},
       )

       return np.exp(result.x)


Comparison with PSMC
======================

The following table summarizes how SMC++ differs from PSMC:

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Component
     - PSMC
     - SMC++
   * - **Input**
     - 1 diploid genome (phased)
     - 1--hundreds of diploids (unphased)
   * - **Hidden state**
     - Coalescence time :math:`T`
     - Same: coalescence time :math:`T`
   * - **Coalescence rate**
     - :math:`1/\lambda(t)`
     - :math:`h(t) = E[J(t)]/\lambda(t)`
   * - **Extra machinery**
     - None
     - ODE system for :math:`p_j(t)`
   * - **Optimization**
     - EM algorithm
     - L-BFGS-B with gradients
   * - **Recent resolution**
     - Poor (:math:`> 20{,}000` years)
     - Good (down to :math:`\sim 1{,}000` years)
   * - **Phasing required**
     - Yes
     - No

The fundamental architecture is the same: an HMM whose hidden states are discretized
coalescence times. SMC++ adds one new gear (the ODE system) and replaces one
(EM with L-BFGS-B), but the overall structure is recognizably PSMC's.


Summary
========

The continuous HMM is the mainspring of SMC++. It takes the coalescence rate
:math:`h(t)` from the ODE system and builds a complete inference engine:
transition matrix, emission probabilities, composite likelihood across samples,
and gradient-based optimization. The result is a demographic inference method
that inherits PSMC's elegant structure while dramatically improving resolution
in the recent past.

In the final chapter, we extend SMC++ to handle **population splits** --
estimating divergence times and population-specific size histories from
cross-population comparisons.
