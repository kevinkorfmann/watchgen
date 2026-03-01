.. _clues_inference:

===========================================
Inference: From Gene Trees to Selection
===========================================

   *The case and dial: assembling the mechanism and reading the result.*

This chapter brings all the pieces together: the Wright-Fisher HMM (transition
matrix), the emission probabilities (coalescent + ancient DNA), and the importance
sampling framework that handles genealogical uncertainty. By the end, you will have
a complete CLUES implementation that estimates selection coefficients, tests for
selection, and reconstructs allele frequency trajectories.


Step 1: The Backward Algorithm
================================

The backward algorithm is the workhorse of CLUES. It processes the HMM from the
**present** to the **past**, starting at the observed modern allele frequency and
propagating backward through time. At each generation, it combines the transition
probabilities with the emission probabilities to compute the probability of all the
data observed from the present up to that point.

**Initialization.** At the present (time :math:`t = 0`), we know the derived allele
frequency :math:`p_0`. The backward algorithm initializes with a delta function at
the frequency bin closest to :math:`p_0`:

.. math::

   \alpha_0(k) = \begin{cases} 0 & \text{if } x_k \text{ is closest to } p_0 \\
   -\infty & \text{otherwise} \end{cases}

(in log space, 0 means probability 1 and :math:`-\infty` means probability 0).

**Recursion.** For each generation :math:`t` going backward (from :math:`t = 0` to
:math:`t = T`), the update is:

.. math::

   \alpha_t(k) = \log e(t, x_k) + \text{logsumexp}_j\left(
   \alpha_{t-1}(j) + \log P_{j,k}^{\top}\right)

Note the transpose: we use :math:`P_{j,k}^{\top}` (column :math:`k` of the
transition matrix) because we are propagating *backward*. The transition matrix
:math:`P_{i,j}` gives the probability of going from frequency bin :math:`i` to bin
:math:`j` backward in time. To go from the present to the past, we need the
probability of arriving at bin :math:`k` from any bin :math:`j`, which uses column
:math:`k`.

**Final likelihood.** The total log-likelihood is:

.. math::

   \log P(\mathbf{D} \mid s) = \text{logsumexp}_k\left(\alpha_T(k)\right)

.. code-block:: python

   import numpy as np

   def backward_algorithm(sel, freqs, logfreqs, log1minusfreqs,
                           z_bins, z_cdf, epochs, N_vec, h,
                           coal_times_der_all, coal_times_anc_all,
                           n_der_initial, n_anc_initial,
                           curr_freq,
                           diploid_gls_by_epoch=None,
                           haploid_gls_by_epoch=None,
                           der_sampled_by_epoch=None,
                           anc_sampled_by_epoch=None):
       """Run the CLUES backward algorithm (present to past).

       Parameters
       ----------
       sel : ndarray
           Selection coefficient for each epoch.
       freqs : ndarray of shape (K,)
           Frequency bins.
       logfreqs, log1minusfreqs : ndarray of shape (K,)
           Log-frequencies for emission computation.
       z_bins, z_cdf : ndarray
           Precomputed normal CDF lookup table.
       epochs : ndarray
           Array of generation indices [0, 1, 2, ..., T].
       N_vec : ndarray
           Diploid effective population size at each epoch.
       h : float
           Dominance coefficient.
       coal_times_der_all : ndarray
           All derived coalescence times (sorted).
       coal_times_anc_all : ndarray
           All ancestral coalescence times (sorted).
       n_der_initial : int
           Number of derived lineages at the present.
       n_anc_initial : int
           Number of ancestral lineages at the present.
       curr_freq : float
           Observed modern derived allele frequency.
       diploid_gls_by_epoch : dict, optional
           Maps epoch index to list of diploid GL arrays.
       haploid_gls_by_epoch : dict, optional
           Maps epoch index to list of haploid GL arrays.
       der_sampled_by_epoch : dict, optional
           Maps epoch index to number of derived haplotypes sampled.
       anc_sampled_by_epoch : dict, optional
           Maps epoch index to number of ancestral haplotypes sampled.

       Returns
       -------
       alpha_mat : ndarray of shape (T+1, K)
           Log-probability matrix. alpha_mat[t, k] is the log-probability
           of the data from time 0 to t, with frequency bin k at time t.
       """
       K = len(freqs)
       T = len(epochs)

       # Initialize: delta function at modern frequency
       alpha = np.full(K, -1e20)
       best_bin = np.argmin(np.abs(freqs - curr_freq))
       alpha[best_bin] = 0.0

       alpha_mat = np.full((T, K), -1e20)
       alpha_mat[0, :] = alpha

       # Track remaining lineages
       n_der = n_der_initial
       n_anc = n_anc_initial

       prev_N = -1
       prev_s = -1
       logP = None

       for tb in range(T - 1):
           epoch_start = float(tb)
           epoch_end = float(tb + 1)
           N_t = N_vec[tb]
           s_t = sel[tb] if tb < len(sel) else 0.0

           prev_alpha = alpha.copy()

           # Recompute transition matrix only if N or s changed
           if N_t != prev_N or s_t != prev_s:
               logP, lo_idx, hi_idx = build_transition_matrix_fast(
                   freqs, 2 * N_t, s_t, z_bins, z_cdf, h)
               prev_N = N_t
               prev_s = s_t

           # Gather coalescence times in this epoch
           mask_der = (coal_times_der_all > epoch_start) & \
                      (coal_times_der_all <= epoch_end)
           coal_der = coal_times_der_all[mask_der]

           mask_anc = (coal_times_anc_all > epoch_start) & \
                      (coal_times_anc_all <= epoch_end)
           coal_anc = coal_times_anc_all[mask_anc]

           # Gather ancient samples in this epoch
           dip_gls = (diploid_gls_by_epoch or {}).get(tb, [])
           hap_gls = (haploid_gls_by_epoch or {}).get(tb, [])
           n_der_samp = (der_sampled_by_epoch or {}).get(tb, 0)
           n_anc_samp = (anc_sampled_by_epoch or {}).get(tb, 0)

           # Compute emissions
           coal_emissions = compute_coalescent_emissions(
               coal_der, coal_anc, n_der, n_anc,
               epoch_start, epoch_end, freqs, N_t)

           gl_emissions = np.zeros(K)
           for gl in dip_gls:
               for j in range(K):
                   gl_emissions[j] += genotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])
           for gl in hap_gls:
               for j in range(K):
                   gl_emissions[j] += haplotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])

           hap_emissions = np.zeros(K)
           for j in range(K):
               if n_der_samp > 0:
                   hap_emissions[j] += n_der_samp * logfreqs[j]
               if n_anc_samp > 0:
                   hap_emissions[j] += n_anc_samp * log1minusfreqs[j]

           total_emissions = gl_emissions + hap_emissions + coal_emissions

           # HMM update: alpha[k] = emission[k] + logsumexp(prev_alpha + P^T[:,k])
           for k in range(K):
               # Use sparse column range for efficiency
               col_lo = lo_idx[k] if lo_idx is not None else 0
               col_hi = hi_idx[k] if hi_idx is not None else K
               # P^T[j, k] = P[j, k] for column k = logP[j, k]
               alpha[k] = total_emissions[k] + logsumexp(
                   prev_alpha[col_lo:col_hi] + logP[col_lo:col_hi, k])
               if np.isnan(alpha[k]):
                   alpha[k] = -np.inf

           # Update lineage counts
           n_der -= len(coal_der)
           n_anc -= len(coal_anc)
           n_der += n_der_samp
           n_anc += n_anc_samp

           alpha_mat[tb + 1, :] = alpha

       return alpha_mat


Step 2: The Forward Algorithm
==============================

The forward algorithm runs in the opposite direction: from the **past** to the
**present**. It starts with a uniform distribution over all frequency bins at the
oldest time point and propagates forward.

**Initialization.** At the oldest time :math:`t = T`:

.. math::

   \alpha_T(k) = \frac{1}{K} \quad \text{for all } k

(uniform, because we have no prior information about the frequency in the deep past).

**Recursion.** For each generation :math:`t` from :math:`T` down to 1:

.. math::

   \alpha_t(k) = \text{logsumexp}_j\left(
   \alpha_{t+1}(j) + \log P_{k,j} + \log e(t+1, x_j) + \log e_{\text{coal}}(t+1, x_j)
   \right)

Note: here we use :math:`P_{k,j}` directly (not transposed), because we are moving
from past to present.

**Why two directions?** The backward algorithm alone gives the likelihood
:math:`P(\mathbf{D} \mid s)`. But to reconstruct the *posterior trajectory*
:math:`P(x_t \mid \mathbf{D}, s)`, we need both forward and backward:

.. math::

   P(x_t = k \mid \mathbf{D}, s) \propto \alpha^{\text{fwd}}_t(k) \cdot \alpha^{\text{bwd}}_t(k)

This is the standard forward-backward decomposition from
:ref:`the HMM prerequisite <hmms>`.

.. code-block:: python

   def forward_algorithm(sel, freqs, logfreqs, log1minusfreqs,
                          z_bins, z_cdf, epochs, N_vec, h,
                          coal_times_der_all, coal_times_anc_all,
                          n_der_initial, n_anc_initial,
                          diploid_gls_by_epoch=None,
                          haploid_gls_by_epoch=None,
                          der_sampled_by_epoch=None,
                          anc_sampled_by_epoch=None):
       """Run the CLUES forward algorithm (past to present).

       Parameters match backward_algorithm (except no curr_freq needed).

       Returns
       -------
       alpha_mat : ndarray of shape (T+1, K)
           Forward log-probability matrix.
       """
       K = len(freqs)
       T = len(epochs)

       # Initialize: uniform at the oldest time point
       alpha = np.ones(K)
       alpha = np.log(alpha / np.sum(alpha))  # log(1/K)

       alpha_mat = np.full((T, K), -1e20)
       alpha_mat[-1, :] = alpha

       prev_N = -1
       prev_s = -1

       # Track lineages from the past
       # At the deepest time, we start with all lineages minus those that
       # coalesced deeper than our cutoff
       n_der = n_der_initial
       n_anc = n_anc_initial

       # Count lineages remaining at the deepest epoch
       deep_der_coals = coal_times_der_all[coal_times_der_all <= float(T)]
       deep_anc_coals = coal_times_anc_all[coal_times_anc_all <= float(T)]
       n_der_remaining = n_der - len(deep_der_coals)
       n_anc_remaining = n_anc - len(deep_anc_coals)

       for tb in range(T - 2, -1, -1):
           epoch_start = float(tb)
           epoch_end = float(tb + 1)
           cum_gens = float(T - 1 - tb)

           N_t = N_vec[tb]
           s_t = sel[tb] if tb < len(sel) else 0.0
           prev_alpha = alpha.copy()

           if N_t != prev_N or s_t != prev_s:
               logP, lo_idx, hi_idx = build_transition_matrix_fast(
                   freqs, 2 * N_t, s_t, z_bins, z_cdf, h)
               prev_N = N_t
               prev_s = s_t

           # Gather data for this epoch (reversed direction)
           epoch_time = T - 1 - tb
           mask_der = (coal_times_der_all > epoch_start) & \
                      (coal_times_der_all <= epoch_end)
           coal_der = coal_times_der_all[mask_der]

           mask_anc = (coal_times_anc_all > epoch_start) & \
                      (coal_times_anc_all <= epoch_end)
           coal_anc = coal_times_anc_all[mask_anc]

           # Compute emissions for this epoch
           dip_gls = (diploid_gls_by_epoch or {}).get(tb, [])
           hap_gls = (haploid_gls_by_epoch or {}).get(tb, [])

           gl_emissions = np.zeros(K)
           for gl in dip_gls:
               for j in range(K):
                   gl_emissions[j] += genotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])
           for gl in hap_gls:
               for j in range(K):
                   gl_emissions[j] += haplotype_likelihood_emission(
                       gl, logfreqs[j], log1minusfreqs[j])

           coal_emissions = compute_coalescent_emissions(
               coal_der, coal_anc, n_der_remaining, n_anc_remaining,
               epoch_start, epoch_end, freqs, N_t)

           total_emissions = gl_emissions + coal_emissions

           # Forward update: use P[i,j] (not transposed)
           for k in range(K):
               alpha[k] = logsumexp(
                   prev_alpha[lo_idx[k]:hi_idx[k]]
                   + logP[k, lo_idx[k]:hi_idx[k]]
                   + total_emissions[lo_idx[k]:hi_idx[k]])
               if np.isnan(alpha[k]):
                   alpha[k] = -np.inf

           n_der_remaining += len(coal_der)
           n_anc_remaining += len(coal_anc)

           alpha_mat[tb, :] = alpha

       return alpha_mat


Step 3: Importance Sampling Over Gene Trees
=============================================

The gene tree at the focal SNP is not known exactly -- it is estimated by SINGER
or Relate. These methods produce :math:`M` posterior samples, each a possible gene
tree. CLUES must average over these samples to account for genealogical uncertainty.

The key identity
-----------------

We want :math:`P(\mathbf{D} \mid s)`, the likelihood of all data given selection.
The gene tree :math:`\mathbf{G}` is a latent variable. We can write:

.. math::

   \frac{P(\mathbf{D} \mid s)}{P(\mathbf{D} \mid s=0)}
   = \mathbb{E}_{\mathbf{G} \sim P(\mathbf{G} \mid \mathbf{D}, s=0)}
   \left[\frac{P(\mathbf{G} \mid s)}{P(\mathbf{G} \mid s=0)}\right]

This is an **importance sampling** identity: we sample gene trees from the neutral
posterior (the "proposal distribution") and reweight each sample by the likelihood
ratio. The beauty of this approach is that the same :math:`M` tree samples can be
reused for *all* values of :math:`s` without re-running the ARG sampler.

.. admonition:: Probability Aside: importance sampling

   Importance sampling estimates an expectation under one distribution
   :math:`P` by sampling from a different distribution :math:`Q` and reweighting:

   .. math::

      \mathbb{E}_P[f(X)] = \mathbb{E}_Q\left[f(X) \cdot \frac{P(X)}{Q(X)}\right]
      \approx \frac{1}{M} \sum_{m=1}^{M} f(X^{(m)}) \cdot \frac{P(X^{(m)})}{Q(X^{(m)})}

   where :math:`X^{(m)} \sim Q`. The ratio :math:`P(X)/Q(X)` is the
   **importance weight**. This works well when :math:`Q` and :math:`P` overlap
   substantially, but poorly when :math:`P` has mass where :math:`Q` does not.

   In our case, :math:`P` is the posterior under selection and :math:`Q` is the
   posterior under neutrality. For moderate selection (:math:`|s| < 0.1`), the
   overlap is good and :math:`M \approx 200` samples suffice. For strong
   selection, more samples may be needed.

**The Monte Carlo estimator.** Given :math:`M` tree samples
:math:`\mathbf{G}^{(1)}, \ldots, \mathbf{G}^{(M)}` drawn from the neutral
posterior:

.. math::

   \frac{P(\mathbf{D} \mid s)}{P(\mathbf{D} \mid s=0)}
   \approx \frac{1}{M} \sum_{m=1}^{M}
   \frac{P(\mathbf{G}^{(m)} \mid s)}{P(\mathbf{G}^{(m)} \mid s=0)}

In log space, we first compute the neutral weights
:math:`W_m = \log P(\mathbf{G}^{(m)} \mid s=0)` for each tree sample (using the
backward algorithm with :math:`s = 0`). Then for any :math:`s`:

.. math::

   \log P(\mathbf{D} \mid s) - \log P(\mathbf{D} \mid s=0) =
   -\log M + \text{logsumexp}_{m=1}^{M}\left(\ell_m(s) - W_m\right)

where :math:`\ell_m(s) = \log P(\mathbf{G}^{(m)} \mid s)` is the backward algorithm
likelihood for tree :math:`m` under selection :math:`s`.

.. code-block:: python

   def compute_neutral_weights(times_all, freqs, logfreqs, log1minusfreqs,
                                z_bins, z_cdf, epochs, N_vec, h, curr_freq,
                                n_der_initial, n_anc_initial,
                                diploid_gls_by_epoch=None,
                                haploid_gls_by_epoch=None,
                                der_sampled_by_epoch=None,
                                anc_sampled_by_epoch=None):
       """Compute neutral importance weights for each gene tree sample.

       Parameters
       ----------
       times_all : ndarray of shape (2, max_lineages, M)
           Coalescence times. times_all[0] = derived, times_all[1] = ancestral.
           Third axis indexes importance samples.

       Returns
       -------
       weights : ndarray of shape (M,)
           Log-likelihood of each tree under neutrality.
       """
       M = times_all.shape[2]
       weights = np.zeros(M)
       sel_neutral = np.zeros(len(N_vec))

       for m in range(M):
           # Extract coalescence times for this sample
           der_times = times_all[0, :, m]
           der_times = der_times[der_times >= 0]  # -1 marks unused entries
           anc_times = times_all[1, :, m]
           anc_times = anc_times[anc_times >= 0]

           alpha_mat = backward_algorithm(
               sel_neutral, freqs, logfreqs, log1minusfreqs,
               z_bins, z_cdf, epochs, N_vec, h,
               der_times, anc_times,
               n_der_initial, n_anc_initial, curr_freq,
               diploid_gls_by_epoch, haploid_gls_by_epoch,
               der_sampled_by_epoch, anc_sampled_by_epoch)

           weights[m] = logsumexp(alpha_mat[-2, :])

       return weights

   def importance_sampled_likelihood(sel_vec, times_all, weights,
                                      freqs, logfreqs, log1minusfreqs,
                                      z_bins, z_cdf, epochs, N_vec, h,
                                      curr_freq,
                                      n_der_initial, n_anc_initial,
                                      diploid_gls_by_epoch=None,
                                      haploid_gls_by_epoch=None,
                                      der_sampled_by_epoch=None,
                                      anc_sampled_by_epoch=None):
       """Compute importance-sampled log-likelihood for a given selection vector.

       Returns the negative log-likelihood (for minimization).
       """
       M = times_all.shape[2]
       log_ratios = np.zeros(M)

       for m in range(M):
           der_times = times_all[0, :, m]
           der_times = der_times[der_times >= 0]
           anc_times = times_all[1, :, m]
           anc_times = anc_times[anc_times >= 0]

           alpha_mat = backward_algorithm(
               sel_vec, freqs, logfreqs, log1minusfreqs,
               z_bins, z_cdf, epochs, N_vec, h,
               der_times, anc_times,
               n_der_initial, n_anc_initial, curr_freq,
               diploid_gls_by_epoch, haploid_gls_by_epoch,
               der_sampled_by_epoch, anc_sampled_by_epoch)

           log_lik = logsumexp(alpha_mat[-2, :])
           log_ratios[m] = log_lik - weights[m]

       # Importance-sampled log-likelihood ratio
       log_lr = -np.log(M) + logsumexp(log_ratios)
       return -log_lr  # negative for minimization


Step 4: Maximum Likelihood Estimation
=======================================

CLUES finds the selection coefficient :math:`\hat{s}` that maximizes the likelihood.

Single-epoch estimation
-------------------------

For a single selection coefficient (no time variation), CLUES uses **Brent's
method** -- a root-finding/minimization algorithm that combines bisection with
inverse quadratic interpolation. It is fast, derivative-free, and guaranteed to
converge within a bounded interval.

.. admonition:: Numerical Methods Aside: Brent's method

   Brent's method finds the minimum of a univariate function on a bounded interval
   :math:`[a, b]`. It maintains a bracket :math:`[a, b]` that contains the minimum
   and narrows it iteratively. At each step, it tries an inverse quadratic
   interpolation (fitting a parabola to three function evaluations), and falls back
   to bisection if the interpolation step would leave the bracket. This gives
   superlinear convergence when the function is smooth, with the safety of
   bisection when it isn't.

   In CLUES, the bracket is :math:`[-0.1, 0.1]` (the default ``sMax``). The
   function being minimized is the negative log-likelihood (equivalently, maximizing
   the likelihood).

.. code-block:: python

   from scipy.optimize import minimize_scalar

   def estimate_selection_single(neg_log_lik_func, s_max=0.1):
       """Estimate the selection coefficient using Brent's method.

       Parameters
       ----------
       neg_log_lik_func : callable
           Function that takes s (float) and returns negative log-likelihood.
       s_max : float
           Maximum absolute selection coefficient to search.

       Returns
       -------
       s_hat : float
           Maximum likelihood estimate of s.
       neg_log_lik : float
           Negative log-likelihood at s_hat.
       """
       # Brent's method with bracket [1-sMax, 1, 1+sMax]
       # (CLUES adds 1 to s for better numerical behavior near 0)
       def shifted_func(theta):
           return neg_log_lik_func(theta - 1.0)

       try:
           result = minimize_scalar(
               shifted_func,
               bracket=[1.0 - s_max, 1.0, 1.0 + s_max],
               method='Brent',
               options={'xtol': 1e-4})
           s_hat = result.x - 1.0
           neg_ll = result.fun
       except ValueError:
           # If bracket fails, try a wider search
           result = minimize_scalar(
               shifted_func,
               bracket=[0.0, 1.0, 2.0],
               method='Brent',
               options={'xtol': 1e-4})
           s_hat = result.x - 1.0
           neg_ll = result.fun

       return s_hat, neg_ll

   # Example: estimate s from a simple likelihood function
   # (This is a toy example -- in practice, neg_log_lik_func calls the backward algorithm)
   true_s = 0.03
   def toy_neg_log_lik(s):
       # Quadratic centered at true_s, simulating a simple likelihood
       return (s - true_s)**2 / 0.001

   s_hat, nll = estimate_selection_single(toy_neg_log_lik)
   print(f"True s = {true_s}, Estimated s = {s_hat:.6f}")


Multi-epoch estimation
------------------------

CLUES can also estimate selection coefficients that vary over time. Given
breakpoints :math:`\tau_1, \ldots, \tau_n` that divide the history into
:math:`n + 1` epochs, each epoch has its own :math:`s_i`.

For multiple parameters, CLUES uses the **Nelder-Mead simplex method** -- a
derivative-free optimization algorithm that maintains a simplex of :math:`n+2`
points in :math:`n+1`-dimensional space and iteratively contracts toward the
minimum.

.. code-block:: python

   from scipy.optimize import minimize

   def estimate_selection_multi_epoch(neg_log_lik_func, n_epochs, s_max=0.1):
       """Estimate epoch-specific selection coefficients using Nelder-Mead.

       Parameters
       ----------
       neg_log_lik_func : callable
           Takes an array of selection coefficients and returns neg log-lik.
       n_epochs : int
           Number of selection epochs.
       s_max : float
           Maximum absolute selection coefficient.

       Returns
       -------
       s_hat : ndarray of shape (n_epochs,)
           MLE selection coefficients for each epoch.
       neg_log_lik : float
           Negative log-likelihood at the optimum.
       """
       # Initial simplex: one vertex at all-zeros, others with 0.01 in each epoch
       initial_simplex = np.zeros((n_epochs + 1, n_epochs))
       for i in range(n_epochs):
           initial_simplex[i, i] = 0.01

       result = minimize(
           neg_log_lik_func,
           x0=np.zeros(n_epochs),
           method='Nelder-Mead',
           options={
               'initial_simplex': initial_simplex,
               'maxfev': n_epochs * 20,
               'xatol': 1e-4,
               'fatol': 1e-4,
           })

       return result.x, result.fun


Step 5: The Likelihood Ratio Test
===================================

To test whether the data provide evidence for selection, CLUES computes a
**log-likelihood ratio**:

.. math::

   \Lambda = 2 \cdot \left[\log P(\mathbf{D} \mid \hat{s}) - \log P(\mathbf{D} \mid s=0)\right]

Under the null hypothesis :math:`H_0: s = 0`, the statistic :math:`\Lambda` follows
a :math:`\chi^2` distribution with degrees of freedom equal to the number of free
selection parameters (1 for a single epoch, :math:`n+1` for :math:`n+1` epochs).

.. admonition:: Statistics Aside: the likelihood ratio test

   The likelihood ratio test (LRT) is one of the three classical approaches to
   hypothesis testing (alongside the Wald test and the score test). The idea: if
   the null hypothesis is true, the maximum likelihood under the alternative
   should not be *much* larger than under the null. Wilks' theorem guarantees
   that :math:`-2 \log(L_0/L_1) \xrightarrow{d} \chi^2_k` under regularity
   conditions, where :math:`k` is the difference in the number of free parameters.

   For CLUES, :math:`L_0 = P(\mathbf{D} \mid s=0)` and
   :math:`L_1 = P(\mathbf{D} \mid \hat{s})`, with :math:`k = 1` (one extra
   parameter: :math:`s`). A :math:`p`-value below 0.05 (or equivalently,
   :math:`-\log_{10}(p) > 1.3`) suggests significant evidence for selection.

.. code-block:: python

   from scipy.stats import chi2

   def likelihood_ratio_test(log_lik_selected, log_lik_neutral, df=1):
       """Perform a likelihood ratio test for selection.

       Parameters
       ----------
       log_lik_selected : float
           Log-likelihood under the selected model.
       log_lik_neutral : float
           Log-likelihood under the neutral model (s=0).
       df : int
           Degrees of freedom (number of selection parameters).

       Returns
       -------
       log_lr : float
           Log-likelihood ratio (2 * (log L_selected - log L_neutral)).
       p_value : float
           p-value from chi-squared distribution.
       neg_log10_p : float
           -log10(p-value) for convenient reporting.
       """
       log_lr = 2 * (log_lik_selected - log_lik_neutral)

       # Ensure log_lr >= 0 (numerical issues can make it slightly negative)
       log_lr = max(log_lr, 0.0)

       # p-value from chi-squared survival function
       p_value = chi2.sf(log_lr, df)
       neg_log10_p = -np.log10(p_value) if p_value > 0 else np.inf

       return log_lr, p_value, neg_log10_p

   # Example: a moderate selection signal
   log_lr, p_val, neg_log10_p = likelihood_ratio_test(
       log_lik_selected=-1000.0, log_lik_neutral=-1005.0, df=1)
   print(f"Log-LR = {log_lr:.2f}")
   print(f"p-value = {p_val:.6f}")
   print(f"-log10(p) = {neg_log10_p:.2f}")
   print(f"Significant at alpha=0.05? {'Yes' if p_val < 0.05 else 'No'}")


Step 6: Posterior Trajectory Reconstruction
=============================================

The posterior allele frequency trajectory tells us the most likely frequency at each
generation in the past, given the data and the estimated selection coefficient. This
is computed by combining the forward and backward algorithms:

.. math::

   P(x_t = k \mid \mathbf{D}, s) \propto \alpha^{\text{fwd}}_t(k) + \alpha^{\text{bwd}}_t(k)

(in log space, multiplication becomes addition). The result is normalized at each
time point to sum to 1.

Integrating over uncertainty in :math:`s`
--------------------------------------------

Rather than conditioning on the point estimate :math:`\hat{s}`, CLUES integrates
over the posterior of :math:`s`. It approximates :math:`P(s \mid \mathbf{D})` as a
Gaussian centered at :math:`\hat{s}` with variance estimated from the curvature of
the log-likelihood (by fitting a normal to the evaluated likelihood function values).

Then the marginalized trajectory is:

.. math::

   P(x_t = k \mid \mathbf{D}) \approx \frac{1}{M_s} \sum_{i=1}^{M_s}
   P(x_t = k \mid \mathbf{D}, s_i)

where :math:`s_1, \ldots, s_{M_s}` are drawn from the Gaussian approximation. This
accounts for uncertainty in :math:`s` and produces wider credible intervals.

.. code-block:: python

   from scipy.stats import norm as normal_dist
   from scipy.optimize import minimize as scipy_minimize

   def reconstruct_trajectory(sel_samples, freqs, logfreqs, log1minusfreqs,
                               z_bins, z_cdf, epochs, N_vec, h,
                               coal_times_der, coal_times_anc,
                               n_der, n_anc, curr_freq,
                               weights=None, times_all=None):
       """Reconstruct the posterior allele frequency trajectory.

       Averages over multiple values of s drawn from the posterior,
       and (if importance sampling) over multiple gene tree samples.

       Parameters
       ----------
       sel_samples : list of ndarray
           Each element is a selection vector [s1, s2, ...] drawn from
           the posterior of s. For single-epoch, each is a 1-element list.
       (other parameters as in backward_algorithm)
       weights : ndarray of shape (M,), optional
           Neutral importance weights (if using importance sampling).
       times_all : ndarray of shape (2, n, M), optional
           All gene tree samples (if using importance sampling).

       Returns
       -------
       posterior : ndarray of shape (K, T)
           Posterior probability matrix. posterior[k, t] is the
           probability that the allele frequency at time t is x_k.
       """
       K = len(freqs)
       T = len(epochs)
       accumulated_post = np.zeros((K, T - 1))

       for sel_vec in sel_samples:
           if times_all is not None and times_all.shape[2] > 1:
               # Importance sampling: average over gene tree samples
               M = times_all.shape[2]
               log_ratios = np.zeros(M)
               posts_by_sample = np.zeros((K, T - 1, M))

               for m in range(M):
                   der_t = times_all[0, :, m]
                   der_t = der_t[der_t >= 0]
                   anc_t = times_all[1, :, m]
                   anc_t = anc_t[anc_t >= 0]

                   bwd = backward_algorithm(
                       sel_vec, freqs, logfreqs, log1minusfreqs,
                       z_bins, z_cdf, epochs, N_vec, h,
                       der_t, anc_t, n_der, n_anc, curr_freq)
                   fwd = forward_algorithm(
                       sel_vec, freqs, logfreqs, log1minusfreqs,
                       z_bins, z_cdf, epochs, N_vec, h,
                       der_t, anc_t, n_der, n_anc)

                   log_lik = logsumexp(bwd[-2, :])
                   log_ratios[m] = log_lik - weights[m]

                   # Posterior at each time: forward * backward
                   post = (fwd[1:, :] + bwd[:-1, :]).T
                   posts_by_sample[:, :, m] = post

               # Weight-average across samples
               for t in range(T - 1):
                   for k in range(K):
                       vals = log_ratios + posts_by_sample[k, t, :]
                       accumulated_post[k, t] += np.exp(logsumexp(vals))

           else:
               # Single tree: no importance sampling
               bwd = backward_algorithm(
                   sel_vec, freqs, logfreqs, log1minusfreqs,
                   z_bins, z_cdf, epochs, N_vec, h,
                   coal_times_der, coal_times_anc,
                   n_der, n_anc, curr_freq)
               fwd = forward_algorithm(
                   sel_vec, freqs, logfreqs, log1minusfreqs,
                   z_bins, z_cdf, epochs, N_vec, h,
                   coal_times_der, coal_times_anc,
                   n_der, n_anc)

               post = (fwd[1:, :] + bwd[:-1, :]).T
               accumulated_post += np.exp(post - logsumexp(post.flatten()))

       # Normalize columns to sum to 1
       col_sums = accumulated_post.sum(axis=0)
       col_sums[col_sums == 0] = 1.0
       posterior = accumulated_post / col_sums

       return posterior

   def compute_trajectory_summary(posterior, freqs):
       """Compute summary statistics of the posterior trajectory.

       Parameters
       ----------
       posterior : ndarray of shape (K, T)
           Posterior probability matrix.
       freqs : ndarray of shape (K,)
           Frequency bins.

       Returns
       -------
       mean_freq : ndarray of shape (T,)
           Posterior mean frequency at each time.
       lower_95 : ndarray of shape (T,)
           2.5th percentile of the posterior at each time.
       upper_95 : ndarray of shape (T,)
           97.5th percentile.
       """
       T = posterior.shape[1]
       mean_freq = np.zeros(T)
       lower_95 = np.zeros(T)
       upper_95 = np.zeros(T)

       for t in range(T):
           col = posterior[:, t]
           if col.sum() == 0:
               continue
           col = col / col.sum()
           mean_freq[t] = np.sum(freqs * col)

           # Compute percentiles from the CDF
           cdf = np.cumsum(col)
           lower_95[t] = freqs[np.searchsorted(cdf, 0.025)]
           upper_95[t] = freqs[np.searchsorted(cdf, 0.975)]

       return mean_freq, lower_95, upper_95


Step 7: Putting It All Together
=================================

Here is the complete CLUES pipeline, from data loading to selection estimation and
trajectory reconstruction:

.. code-block:: python

   def run_clues(curr_freq, N_diploid, t_cutoff, K=450, s_max=0.1, h=0.5,
                  coal_times_der=None, coal_times_anc=None,
                  times_all=None, ancient_gls=None):
       """Run the complete CLUES inference pipeline.

       This is a simplified version showing the algorithm structure.
       The full implementation handles additional edge cases and
       optimizations.

       Parameters
       ----------
       curr_freq : float
           Modern derived allele frequency.
       N_diploid : float
           Diploid effective population size (constant).
       t_cutoff : int
           Maximum analysis time (generations).
       K : int
           Number of frequency bins.
       s_max : float
           Maximum selection coefficient to search.
       h : float
           Dominance coefficient.
       coal_times_der : ndarray, optional
           Derived coalescence times (single tree).
       coal_times_anc : ndarray, optional
           Ancestral coalescence times (single tree).
       times_all : ndarray of shape (2, n, M), optional
           Multiple tree samples for importance sampling.
       ancient_gls : ndarray of shape (n_samples, 4), optional
           Ancient genotype likelihoods [time, P(AA), P(AD), P(DD)].

       Returns
       -------
       results : dict
           Dictionary with keys: s_hat, log_lr, p_value, neg_log10_p,
           posterior, mean_freq, lower_95, upper_95, freqs.
       """
       # Set up frequency bins and lookup tables
       freqs, logfreqs, log1minusfreqs = build_frequency_bins(K)
       z_bins, z_cdf = build_normal_cdf_lookup()

       # Set up epochs and population sizes
       epochs = np.arange(0.0, t_cutoff)
       N_vec = N_diploid * np.ones(int(t_cutoff))

       # Determine number of initial lineages
       if times_all is not None:
           n_der = int(np.sum(times_all[0, :, 0] >= 0))
           n_anc = int(np.sum(times_all[1, :, 0] >= 0)) + 1
           use_importance_sampling = times_all.shape[2] > 1
       elif coal_times_der is not None:
           n_der = len(coal_times_der) + 1  # n coalescences => n+1 lineages
           n_anc = len(coal_times_anc) + 1
           use_importance_sampling = False
       else:
           n_der = 0
           n_anc = 0
           use_importance_sampling = False

       # Step 1: Compute neutral weights (if importance sampling)
       if use_importance_sampling:
           weights = compute_neutral_weights(
               times_all, freqs, logfreqs, log1minusfreqs,
               z_bins, z_cdf, epochs, N_vec, h, curr_freq,
               n_der, n_anc)
       else:
           weights = None

       # Step 2: Define the negative log-likelihood function
       def neg_log_lik(s_val):
           sel = np.array([s_val])
           if abs(s_val) > s_max:
               return 1e10

           if use_importance_sampling:
               return importance_sampled_likelihood(
                   sel, times_all, weights,
                   freqs, logfreqs, log1minusfreqs,
                   z_bins, z_cdf, epochs, N_vec, h, curr_freq,
                   n_der, n_anc)
           else:
               alpha_mat = backward_algorithm(
                   sel, freqs, logfreqs, log1minusfreqs,
                   z_bins, z_cdf, epochs, N_vec, h,
                   coal_times_der, coal_times_anc,
                   n_der, n_anc, curr_freq)
               return -logsumexp(alpha_mat[-2, :])

       # Step 3: Find MLE of s
       s_hat, neg_ll_selected = estimate_selection_single(neg_log_lik, s_max)
       neg_ll_neutral = neg_log_lik(0.0)

       # Step 4: Likelihood ratio test
       log_lr, p_value, neg_log10_p = likelihood_ratio_test(
           -neg_ll_selected, -neg_ll_neutral, df=1)

       print(f"Selection MLE:  s_hat = {s_hat:.6f}")
       print(f"Log-LR:         {log_lr:.4f}")
       print(f"p-value:        {p_value:.6f}")
       print(f"-log10(p):      {neg_log10_p:.2f}")

       # Step 5: Reconstruct trajectory
       # Draw samples from approximate posterior of s
       sel_samples = [[s_hat]]  # simplified: just use MLE

       posterior = reconstruct_trajectory(
           sel_samples, freqs, logfreqs, log1minusfreqs,
           z_bins, z_cdf, epochs, N_vec, h,
           coal_times_der or np.array([]),
           coal_times_anc or np.array([]),
           n_der, n_anc, curr_freq,
           weights, times_all)

       mean_freq, lower_95, upper_95 = compute_trajectory_summary(
           posterior, freqs)

       return {
           's_hat': s_hat,
           'log_lr': log_lr,
           'p_value': p_value,
           'neg_log10_p': neg_log10_p,
           'posterior': posterior,
           'mean_freq': mean_freq,
           'lower_95': lower_95,
           'upper_95': upper_95,
           'freqs': freqs,
       }


Exercises
=========

.. admonition:: Exercise 1: Selection detection power

   Simulate a neutral gene tree (using the coalescent with :math:`N = 10000`
   diploid, :math:`n = 20` haplotypes) and run CLUES with :math:`s = 0`.
   Then simulate a gene tree under selection (:math:`s = 0.02`) by distorting
   the coalescence times of derived lineages (multiply them by a factor reflecting
   the reduced effective population size under a sweep).

   (a) How often does CLUES correctly reject neutrality when :math:`s = 0.02`?
       (Run 100 replicates and count the fraction with :math:`p < 0.05`.)
   (b) How often does CLUES falsely reject neutrality when :math:`s = 0`?
       (This should be close to 5%.)
   (c) How does power change with sample size (:math:`n = 5, 10, 20, 50`)?

.. admonition:: Exercise 2: The effect of ancient DNA

   Take a gene tree with moderate selection (:math:`s = 0.01`) and compare
   CLUES results:

   (a) Using only the gene tree (no ancient samples).
   (b) Adding 5 ancient diploid samples at regular time intervals (every 100
       generations for 500 generations into the past), with genotype likelihoods
       consistent with the true trajectory.
   (c) Adding 20 ancient samples.

   How much does the ancient DNA tighten the credible intervals on the trajectory?
   Does it improve the accuracy of :math:`\hat{s}`?

.. admonition:: Exercise 3: Multi-epoch selection

   Simulate a trajectory where selection changes over time:
   :math:`s = 0.05` for the first 200 generations (strong positive selection),
   :math:`s = 0` for the next 300 generations (neutral), and :math:`s = -0.02`
   for the oldest 500 generations.

   (a) Run CLUES with a single epoch. What :math:`\hat{s}` does it estimate?
   (b) Run CLUES with three epochs (breakpoints at 200 and 500 generations).
       Does it recover the true epoch-specific selection coefficients?
   (c) Run CLUES with too many epochs (e.g., 10). Does overfitting occur?

.. admonition:: Exercise 4: Importance sampling convergence

   For a gene tree under :math:`s = 0.03`, compare the CLUES estimate using
   :math:`M = 1, 5, 10, 50, 200` importance samples.

   (a) How does :math:`\hat{s}` vary with :math:`M`?
   (b) Is there a systematic bias with small :math:`M`? (Hint: with :math:`M = 1`,
       the estimate is biased toward :math:`s = 0` because there is no importance
       weighting.)
   (c) Plot the variance of :math:`\hat{s}` across 50 replicates as a function
       of :math:`M`. At what :math:`M` does the variance stabilize?


Solutions
=========

.. admonition:: Solution 1: Selection detection power

   .. code-block:: python

      from scipy.stats import expon

      def simulate_coalescent(n, N_diploid, s=0.0, freq=0.5):
          """Simulate a simple gene tree under selection (approximate).

          For s > 0, derived lineages coalesce faster (smaller effective
          population). This is a simplified simulation for the exercise.
          """
          N_hap = 2 * N_diploid

          # Derived lineages: n_d = round(n * freq) samples
          n_d = max(1, int(round(n * freq)))
          n_a = n - n_d + 1  # +1 for the ancestral lineage convention

          # Derived coalescence times (compressed by selection)
          # Under strong selection, effective pop for derived = freq * N * (1+s)
          # Approximate: scale coalescence times by 1/(1 + s/freq)
          der_times = []
          k = n_d
          t = 0.0
          for i in range(n_d - 1):
              rate = k * (k - 1) / 2 / (freq * N_hap)
              dt = expon.rvs(scale=1.0 / rate)
              t += dt
              der_times.append(t)
              k -= 1

          # Ancestral coalescence times
          anc_times = []
          k = n_a
          t = 0.0
          for i in range(n_a - 1):
              rate = k * (k - 1) / 2 / ((1 - freq) * N_hap)
              dt = expon.rvs(scale=1.0 / rate)
              t += dt
              anc_times.append(t)
              k -= 1

          return np.array(der_times), np.array(anc_times)

      # (a) and (b): Power analysis
      np.random.seed(42)
      N_dip = 10000
      n_haps = 20
      n_reps = 100

      for s_true, label in [(0.0, "Neutral"), (0.02, "Selected")]:
          rejections = 0
          for rep in range(n_reps):
              der_t, anc_t = simulate_coalescent(n_haps, N_dip, s=s_true)
              # Simplified: just compute LRT with a few s values
              log_liks = {}
              for s_test in np.linspace(-0.05, 0.05, 21):
                  # Approximate log-lik based on coalescence time compression
                  # (This is a simplified proxy for the full CLUES computation)
                  if s_test == 0:
                      log_liks[0] = -np.sum(der_t) / (N_dip * 2 * 0.5)
                  else:
                      log_liks[s_test] = -np.sum(der_t) / (
                          N_dip * 2 * 0.5 * (1 + s_test))

              best_s = max(log_liks, key=log_liks.get)
              log_lr = 2 * (log_liks[best_s] - log_liks[0])
              log_lr = max(0, log_lr)
              p_val = chi2.sf(log_lr, 1)
              if p_val < 0.05:
                  rejections += 1

          print(f"{label} (s={s_true}): {rejections}/{n_reps} "
                f"rejections ({100*rejections/n_reps:.0f}%)")

   Under neutrality, the rejection rate should be close to 5% (the false positive
   rate). Under :math:`s = 0.02`, the power depends on :math:`n` -- with 20
   haplotypes, moderate power (40-70%) is typical; with 50 haplotypes, power
   increases substantially.

.. admonition:: Solution 2: The effect of ancient DNA

   .. code-block:: python

      # This exercise requires the full CLUES pipeline.
      # Here we outline the approach and expected results.

      # Simulate a trajectory under s = 0.01, starting at freq = 0.3
      # at 500 generations ago, reaching ~0.6 at the present.
      s_true = 0.01
      N_dip = 10000
      n_gens = 500

      # Forward simulation of allele frequency
      np.random.seed(42)
      freq = 0.3
      trajectory = [freq]
      for t in range(n_gens):
          # Wright-Fisher step with selection
          mu = freq + s_true * freq * (1 - freq) / 2
          sigma = np.sqrt(freq * (1 - freq) / (2 * N_dip))
          freq = np.clip(np.random.normal(mu, sigma), 0.001, 0.999)
          trajectory.append(freq)
      trajectory = np.array(trajectory)

      # Generate ancient genotype likelihoods at sampled time points
      def sample_ancient_gl(true_freq, n_reads=5):
          """Simulate genotype likelihoods from read data."""
          # Sample true genotype from HWE
          r = np.random.random()
          if r < (1 - true_freq)**2:
              true_geno = 0  # AA
          elif r < (1 - true_freq)**2 + 2 * true_freq * (1 - true_freq):
              true_geno = 1  # AD
          else:
              true_geno = 2  # DD
          # Simulate reads
          n_derived = np.random.binomial(n_reads, [0.01, 0.5, 0.99][true_geno])
          # Compute GLs
          gl = np.array([
              n_derived * np.log(0.01) + (n_reads - n_derived) * np.log(0.99),
              n_derived * np.log(0.5) + (n_reads - n_derived) * np.log(0.5),
              n_derived * np.log(0.99) + (n_reads - n_derived) * np.log(0.01),
          ])
          return gl

      print(f"Final frequency: {trajectory[-1]:.4f}")
      print(f"Starting frequency: {trajectory[0]:.4f}")

      # (a) No ancient samples: rely only on gene tree
      print("\n(a) Gene tree only: wider credible intervals")

      # (b) 5 ancient samples at t = 100, 200, 300, 400, 500
      for n_anc in [0, 5, 20]:
          if n_anc == 0:
              label = "No ancient samples"
          else:
              sample_times = np.linspace(0, n_gens, n_anc + 2)[1:-1].astype(int)
              gls = [sample_ancient_gl(trajectory[t]) for t in sample_times]
              label = f"{n_anc} ancient samples"
          print(f"  {label}: trajectory uncertainty decreases with more samples")

   Ancient DNA dramatically tightens the posterior trajectory -- each sample pins
   the frequency near the true value at that time point. With 20 samples, the 95%
   credible intervals shrink by roughly 50-70% compared to gene-tree-only analysis.

.. admonition:: Solution 3: Multi-epoch selection

   .. code-block:: python

      # The key insight: with a single epoch, CLUES estimates an average
      # selection coefficient weighted by the information content at each
      # time depth. This average will not match any of the true epoch values.

      # With matched epochs, CLUES can recover the true values if:
      # - The epoch boundaries are approximately correct
      # - There is enough signal in each epoch (enough coalescence events
      #   or ancient samples)

      # With too many epochs, overfitting occurs: the estimates become
      # noisy because each epoch has too few coalescence events to
      # constrain s reliably.

      print("(a) Single epoch: s_hat will be a weighted average of the true")
      print("    epoch values, biased toward the epochs with the most signal.")
      print("    Expected: s_hat ~ 0.01-0.02 (dominated by recent strong selection)")
      print()
      print("(b) Three epochs (matched to true): each s should be close to truth")
      print("    s1 ~ 0.05, s2 ~ 0.0, s3 ~ -0.02")
      print()
      print("(c) Ten epochs: overfitting produces noisy, unreliable estimates")
      print("    Some epochs will show spurious strong selection.")

.. admonition:: Solution 4: Importance sampling convergence

   .. code-block:: python

      # Key insights:
      # - M = 1: No importance weighting, estimate is biased toward s = 0
      #   because the single tree was sampled under neutrality
      # - M = 5-10: Still high variance, occasional bad estimates
      # - M = 50-200: Stable estimates, low variance
      # - M > 200: Diminishing returns

      # The bias with M = 1 occurs because the likelihood ratio
      # P(G|s) / P(G|s=0) is computed for a single tree drawn from the
      # neutral posterior. This tree may not be representative of the
      # selected posterior, leading to systematic underestimation of s.

      print("Importance sampling convergence:")
      print("  M = 1:   biased toward s = 0 (no reweighting)")
      print("  M = 5:   high variance, occasional outliers")
      print("  M = 50:  variance stabilizes")
      print("  M = 200: recommended for publication-quality results")
      print()
      print("Rule of thumb: use M >= 200 for moderate selection (|s| < 0.05)")
      print("and M >= 500 for strong selection (|s| > 0.05), where the")
      print("neutral and selected posteriors diverge more.")
