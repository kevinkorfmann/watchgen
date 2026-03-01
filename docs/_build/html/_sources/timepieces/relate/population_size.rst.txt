.. _relate_population_size:

================================================
Gear 4: Population Size Estimation
================================================

   *A watch tells you the time, but only if it's been calibrated. The
   coalescent prior is only as good as the population size that feeds it.*

The coalescent prior in :ref:`Gear 3 <relate_branch_lengths>` assumes a known
effective population size :math:`N_e`. In practice, :math:`N_e` is unknown and
may have changed over time (bottlenecks, expansions, population splits). Relate
estimates a **piecewise-constant** population size history :math:`N_e(t)` using
an **Expectation-Maximization (EM)** algorithm that alternates between sampling
branch lengths and updating population size estimates.

This is the regulator of the movement: it adjusts the coalescent prior so that
the inferred coalescence times are consistent with the population size that
generated them. Without this feedback loop, a misspecified :math:`N_e` would
systematically bias all coalescence time estimates.

.. admonition:: Prerequisites

   - :ref:`relate_branch_lengths` (Gear 3): the MCMC sampler for branch
     lengths, the coalescent prior, and the Poisson likelihood
   - :ref:`Coalescent Theory <coalescent_theory>` -- how population size
     affects coalescence rates


The Piecewise-Constant Model
==============================

Relate models population size as constant within pre-defined time epochs:

.. math::

   N_e(t) = N_j \quad \text{for } t \in [t_j, t_{j+1})

where :math:`t_0 = 0 < t_1 < \cdots < t_M` are epoch boundaries and
:math:`N_0, N_1, \ldots, N_{M-1}` are the population sizes in each epoch.
The epoch boundaries are fixed in advance (logarithmically spaced is common);
only the :math:`N_j` values are estimated.

Within epoch :math:`j`, the coalescence rate for :math:`k` lineages is
constant:

.. math::

   \lambda_{k,j} = \frac{k(k-1)}{2 N_j}

.. code-block:: python

   import numpy as np

   def make_epochs(max_time, n_epochs):
       """Create logarithmically spaced epoch boundaries.

       Parameters
       ----------
       max_time : float
           Maximum time (most ancient epoch boundary).
       n_epochs : int
           Number of epochs.

       Returns
       -------
       boundaries : ndarray of shape (n_epochs + 1,)
           Epoch boundaries [0, t_1, t_2, ..., max_time].
       """
       # Log-space between a small positive value and max_time
       boundaries = np.zeros(n_epochs + 1)
       boundaries[1:] = np.logspace(
           np.log10(max_time / n_epochs),
           np.log10(max_time),
           n_epochs
       )
       return boundaries

   # Example: 20 epochs up to 100,000 generations
   boundaries = make_epochs(100_000, n_epochs=20)
   print("Epoch boundaries (generations):")
   for i in range(len(boundaries) - 1):
       print(f"  Epoch {i}: [{boundaries[i]:.0f}, {boundaries[i+1]:.0f})")


The Coalescent Prior with Variable Population Size
=====================================================

When population size varies over time, the coalescence rate is no longer
constant. The probability that :math:`k` lineages have not yet coalesced by
time :math:`t`, given they had :math:`k` lineages at time :math:`t_0`, is:

.. math::

   S_k(t) = \exp\left(-\binom{k}{2} \int_{t_0}^{t} \frac{1}{N_e(s)} \, ds\right)

For piecewise-constant :math:`N_e`, the integral becomes a sum:

.. math::

   \int_{t_0}^{t} \frac{1}{N_e(s)} \, ds = \sum_{j} \frac{\Delta t_j}{N_j}

where :math:`\Delta t_j` is the time spent in epoch :math:`j` during the
interval :math:`[t_0, t]`.

The density of the coalescence time is:

.. math::

   f_k(t) = \frac{k(k-1)}{2 N_e(t)} \cdot S_k(t)

.. code-block:: python

   def integrated_rate(t_start, t_end, boundaries, N_e_values):
       """Compute the integrated inverse population size.

       Parameters
       ----------
       t_start, t_end : float
           Time interval [t_start, t_end).
       boundaries : ndarray
           Epoch boundaries.
       N_e_values : ndarray
           Population size in each epoch.

       Returns
       -------
       float
           Integral of 1/N_e(t) from t_start to t_end.
       """
       result = 0.0
       n_epochs = len(N_e_values)

       for j in range(n_epochs):
           epoch_start = boundaries[j]
           epoch_end = boundaries[j + 1]

           # Overlap of [t_start, t_end) with [epoch_start, epoch_end)
           overlap_start = max(t_start, epoch_start)
           overlap_end = min(t_end, epoch_end)

           if overlap_start < overlap_end:
               result += (overlap_end - overlap_start) / N_e_values[j]

       # Handle time beyond the last epoch boundary
       if t_end > boundaries[-1]:
           overlap_start = max(t_start, boundaries[-1])
           result += (t_end - overlap_start) / N_e_values[-1]

       return result

   def log_coalescent_prior_variable(coalescence_times, boundaries, N_e_values):
       """Log coalescent prior with piecewise-constant population size.

       Parameters
       ----------
       coalescence_times : list of float
           Coalescence times sorted youngest to oldest.
       boundaries : ndarray
           Epoch boundaries.
       N_e_values : ndarray
           Population size in each epoch.

       Returns
       -------
       float
           Log prior probability.
       """
       n_coal = len(coalescence_times)
       N = n_coal + 1  # number of leaves

       log_prior = 0.0
       prev_time = 0.0

       for idx, t in enumerate(coalescence_times):
           k = N - idx  # number of lineages before this coalescence
           if k < 2:
               break

           coal_rate_k = k * (k - 1) / 2.0

           # Find which epoch t falls in to get instantaneous N_e
           epoch_idx = np.searchsorted(boundaries[1:], t)
           epoch_idx = min(epoch_idx, len(N_e_values) - 1)
           N_e_at_t = N_e_values[epoch_idx]

           # Instantaneous rate
           rate = coal_rate_k / N_e_at_t

           # Survival: integral of coal_rate_k / N_e(s) from prev_time to t
           integral = coal_rate_k * integrated_rate(
               prev_time, t, boundaries, N_e_values)

           # Log density: log(rate) - integral
           log_prior += np.log(rate) - integral
           prev_time = t

       return log_prior

   # Example
   N_e_values = np.full(20, 10000.0)  # constant N_e for comparison
   coal_times = [100, 300, 500]
   lp_variable = log_coalescent_prior_variable(coal_times, boundaries,
                                                N_e_values)
   print(f"Log prior (variable model, constant N_e): {lp_variable:.2f}")


The EM Algorithm
==================

The EM algorithm alternates between two steps:

**E-step**: Given current population size estimates :math:`\hat{N}_e(t)`,
sample coalescence times from the posterior using the MCMC from Gear 3.
Specifically, run ``SampleBranchLengths`` with the current :math:`\hat{N}_e(t)`
as the coalescent prior.

**M-step**: Given the sampled coalescence times, estimate new population
sizes that maximize the coalescent likelihood. For each epoch :math:`j`,
count the total "lineage time" (sum of time intervals where :math:`k`
lineages were active) and the total number of coalescence events:

.. math::

   \hat{N}_j = \frac{\sum_{\text{trees}} \sum_{\text{intervals in epoch } j}
   \binom{k}{2} \cdot \Delta t}
   {\sum_{\text{trees}} C_j}

where :math:`C_j` is the number of coalescence events in epoch :math:`j`
and the numerator sums :math:`\binom{k}{2} \Delta t` over all time intervals
within epoch :math:`j` across all trees.

.. admonition:: Probability Aside -- Why This M-step Works

   For a constant-rate Poisson process with rate :math:`\lambda`, the
   maximum likelihood estimate of :math:`\lambda` given :math:`n` events
   in time :math:`T` is :math:`\hat{\lambda} = n / T`. The coalescent
   with :math:`k` lineages and population size :math:`N` is exactly such
   a process with rate :math:`\binom{k}{2} / N`. Rearranging:

   .. math::

      \hat{N}_j = \frac{\text{total exposure (lineage-time)}}
      {\text{number of coalescences}}

   This is the moment estimator for the population size: the ratio of
   total time "at risk" of coalescence to the number of events observed.

.. code-block:: python

   def m_step(coalescence_times_per_tree, num_leaves_per_tree,
              boundaries, span_per_tree):
       """M-step: estimate population sizes from coalescence times.

       Parameters
       ----------
       coalescence_times_per_tree : list of list of float
           For each tree, the sorted coalescence times.
       num_leaves_per_tree : list of int
           Number of leaves in each tree.
       boundaries : ndarray
           Epoch boundaries.
       span_per_tree : list of float
           Genomic span of each tree (for weighting).

       Returns
       -------
       N_e_estimates : ndarray
           Estimated N_e for each epoch.
       """
       n_epochs = len(boundaries) - 1
       total_exposure = np.zeros(n_epochs)  # lineage-time at risk
       total_events = np.zeros(n_epochs)    # coalescence events

       for tree_idx, coal_times in enumerate(coalescence_times_per_tree):
           N = num_leaves_per_tree[tree_idx]
           weight = span_per_tree[tree_idx]
           prev_time = 0.0

           for idx, t in enumerate(coal_times):
               k = N - idx  # lineages before this coalescence
               if k < 2:
                   break

               # Distribute exposure across epochs
               for j in range(n_epochs):
                   ep_start = boundaries[j]
                   ep_end = boundaries[j + 1]

                   overlap_start = max(prev_time, ep_start)
                   overlap_end = min(t, ep_end)

                   if overlap_start < overlap_end:
                       dt = overlap_end - overlap_start
                       exposure = k * (k - 1) / 2.0 * dt * weight
                       total_exposure[j] += exposure

               # Record the coalescence event in the appropriate epoch
               event_epoch = np.searchsorted(boundaries[1:], t)
               event_epoch = min(event_epoch, n_epochs - 1)
               total_events[event_epoch] += weight

               prev_time = t

       # Estimate N_e: exposure / events (avoid division by zero)
       N_e_estimates = np.zeros(n_epochs)
       for j in range(n_epochs):
           if total_events[j] > 0:
               N_e_estimates[j] = total_exposure[j] / total_events[j]
           else:
               # No coalescence events in this epoch -- use neighbor
               N_e_estimates[j] = np.nan

       # Fill NaN epochs by interpolation
       valid = ~np.isnan(N_e_estimates)
       if valid.any():
           epoch_mids = (boundaries[:-1] + boundaries[1:]) / 2
           N_e_estimates[~valid] = np.interp(
               epoch_mids[~valid], epoch_mids[valid], N_e_estimates[valid])

       return N_e_estimates


   # Example: constant population size recovery
   np.random.seed(42)
   true_N_e = 10000
   n_trees = 100
   n_leaves = 10

   # Simulate coalescence times under constant N_e
   coal_times_all = []
   for _ in range(n_trees):
       times = []
       prev_t = 0.0
       for k in range(n_leaves, 1, -1):
           rate = k * (k - 1) / (2.0 * true_N_e)
           dt = np.random.exponential(1.0 / rate)
           prev_t += dt
           times.append(prev_t)
       coal_times_all.append(times)

   boundaries_em = make_epochs(50_000, n_epochs=10)
   spans = np.full(n_trees, 1e4)

   N_e_est = m_step(coal_times_all, [n_leaves] * n_trees,
                      boundaries_em, spans)

   print("Population size estimates:")
   for j in range(len(N_e_est)):
       print(f"  Epoch {j} [{boundaries_em[j]:.0f}, {boundaries_em[j+1]:.0f}): "
             f"N_e = {N_e_est[j]:.0f}")
   print(f"\nTrue N_e: {true_N_e}")
   print(f"Mean estimated N_e: {np.nanmean(N_e_est):.0f}")


The Complete EM Loop
=====================

The full EM algorithm iterates between sampling branch lengths (E-step)
and updating population sizes (M-step):

.. code-block:: python

   def em_population_size(trees, haplotypes, mu, initial_N_e,
                           boundaries, n_em_iterations=10,
                           n_mcmc_samples=200):
       """Estimate population size history via EM.

       Parameters
       ----------
       trees : list of dict
           Local trees with topologies.
       haplotypes : ndarray
           Haplotype matrix.
       mu : float
           Mutation rate.
       initial_N_e : float
           Initial (constant) population size guess.
       boundaries : ndarray
           Epoch boundaries.
       n_em_iterations : int
           Number of EM iterations.
       n_mcmc_samples : int
           MCMC samples per tree per E-step.

       Returns
       -------
       N_e_history : list of ndarray
           Population size estimates at each EM iteration.
       """
       n_epochs = len(boundaries) - 1
       N_e_values = np.full(n_epochs, initial_N_e)
       N_e_history = [N_e_values.copy()]

       for em_iter in range(n_em_iterations):
           print(f"EM iteration {em_iter + 1}/{n_em_iterations}")

           # E-step: sample branch lengths using current N_e
           all_coal_times = []
           all_n_leaves = []
           all_spans = []

           for tree_info in trees:
               root = tree_info['root']
               span = tree_info['end'] - tree_info['start']
               site_indices = tree_info.get('site_indices', [])

               # Map mutations
               branch_muts, _ = map_mutations(root, haplotypes, site_indices)

               # Run MCMC with current population size
               # (simplified: using constant N_e equal to the mean)
               mean_N_e = np.mean(N_e_values)
               samples, _ = mcmc_branch_lengths(
                   root, branch_muts, mu, span, mean_N_e,
                   n_samples=n_mcmc_samples, burn_in=50)

               # Extract coalescence times from the last sample
               if samples:
                   last_sample = samples[-1]
                   # Get internal node times, sorted
                   internal_times = sorted([
                       t for nid, t in last_sample.items()
                       if t > 0  # exclude leaves
                   ])
                   all_coal_times.append(internal_times)
                   n_leaves = sum(1 for t in last_sample.values() if t == 0)
                   all_n_leaves.append(n_leaves)
                   all_spans.append(span)

           # M-step: update population sizes
           N_e_values = m_step(all_coal_times, all_n_leaves,
                                boundaries, all_spans)
           N_e_history.append(N_e_values.copy())

           print(f"  Mean N_e: {np.nanmean(N_e_values):.0f}")

       return N_e_history


Interpreting the Results
=========================

The EM algorithm produces a piecewise-constant population size trajectory
:math:`\hat{N}_e(t)`. This can reveal:

- **Population bottlenecks**: Epochs with small :math:`\hat{N}_e`
- **Population expansions**: Epochs with increasing :math:`\hat{N}_e`
- **Recent growth**: Very large :math:`\hat{N}_e` in the most recent epoch

.. code-block:: python

   def plot_population_size(boundaries, N_e_values, true_N_e=None):
       """Plot the estimated population size history.

       Parameters
       ----------
       boundaries : ndarray
           Epoch boundaries.
       N_e_values : ndarray
           Estimated N_e per epoch.
       true_N_e : callable or None
           True N_e(t) function for comparison.
       """
       import matplotlib.pyplot as plt

       # Step plot for piecewise-constant N_e
       fig, ax = plt.subplots(figsize=(8, 5))

       for j in range(len(N_e_values)):
           ax.plot([boundaries[j], boundaries[j + 1]],
                   [N_e_values[j], N_e_values[j]],
                   'b-', lw=2, label='Estimated' if j == 0 else '')

       if true_N_e is not None:
           t_grid = np.linspace(boundaries[0], boundaries[-1], 500)
           ax.plot(t_grid, [true_N_e(t) for t in t_grid],
                   'r--', lw=1.5, label='True')

       ax.set_xlabel('Time (generations ago)')
       ax.set_ylabel('Effective population size')
       ax.set_xscale('log')
       ax.set_yscale('log')
       ax.legend()
       ax.set_title('Population Size History')
       plt.tight_layout()
       plt.show()


Verification
=============

We verify the EM on a scenario with known population size history:

.. code-block:: python

   def verify_em():
       """Verify the M-step recovers a known constant population size."""
       np.random.seed(42)
       true_N_e = 10000
       n_trees = 200
       n_leaves = 8

       # Simulate coalescence times under constant N_e
       coal_times_all = []
       for _ in range(n_trees):
           times = []
           prev_t = 0.0
           for k in range(n_leaves, 1, -1):
               rate = k * (k - 1) / (2.0 * true_N_e)
               dt = np.random.exponential(1.0 / rate)
               prev_t += dt
               times.append(prev_t)
           coal_times_all.append(times)

       # Estimate using M-step
       boundaries = make_epochs(100_000, n_epochs=15)
       spans = np.full(n_trees, 1e4)
       N_e_est = m_step(coal_times_all, [n_leaves] * n_trees,
                         boundaries, spans)

       # Check: mean estimated N_e should be close to true
       mean_est = np.nanmean(N_e_est)
       rel_error = abs(mean_est - true_N_e) / true_N_e

       print("EM verification (constant N_e):")
       print(f"  True N_e:     {true_N_e}")
       print(f"  Mean est N_e: {mean_est:.0f}")
       print(f"  Relative error: {rel_error:.1%}")
       print(f"  [{'ok' if rel_error < 0.2 else 'FAIL'}] "
             f"Mean within 20% of true value")

       # Per-epoch estimates
       print("\n  Per-epoch estimates:")
       for j in range(len(N_e_est)):
           status = "ok" if abs(N_e_est[j] - true_N_e) / true_N_e < 0.5 \
                    else "imprecise"
           print(f"    Epoch {j}: N_e = {N_e_est[j]:.0f} [{status}]")

   verify_em()


Summary
========

The complete Relate pipeline, from haplotype data to dated genealogies with
population size estimates:

.. code-block:: text

   1. PAINTING (Gear 1)
      Modified Li & Stephens forward-backward
      -> Asymmetric distance matrices
              |
              v
   2. TREE BUILDING (Gear 2)
      Agglomerative clustering on asymmetric distances
      -> Local tree topologies
              |
              v
   3. MUTATION MAPPING
      Infinite-sites: each derived allele -> one branch
              |
              v
   4. BRANCH LENGTHS (Gear 3)
      MCMC: Poisson likelihood + coalescent prior
      -> Posterior samples of coalescence times
              |
              v
   5. POPULATION SIZE (Gear 4)
      EM: E-step = sample branch lengths
          M-step = update N_e(t)
      -> Piecewise-constant N_e(t)
              |
              v
   Iterate steps 4-5 until convergence

The key equations:

.. math::

   d(i,j) = -\log p_{ij}(s) \quad \text{(asymmetric distance)}

.. math::

   m_b \sim \text{Poisson}(\mu \cdot \ell_b \cdot \Delta t_b) \quad \text{(mutation likelihood)}

.. math::

   P(\mathbf{t} \mid \mathbf{m}) \propto \prod_b \text{Poisson}(m_b \mid \mu \ell_b \Delta t_b) \cdot \prod_k \lambda_k e^{-\lambda_k \Delta t_k} \quad \text{(posterior)}

.. math::

   \hat{N}_j = \frac{\sum \binom{k}{2} \Delta t_j}{\sum C_j} \quad \text{(M-step)}

By building all four gears, you now have a complete understanding of Relate's
mechanism -- from raw haplotype data to dated genealogies with uncertainty
quantification and population size estimation. Like a well-calibrated automatic
watch, every gear meshes with the others to produce a self-consistent picture
of the population's evolutionary history.


Exercises
==========

.. admonition:: Exercise 1: Bottleneck recovery

   Simulate coalescence times under a bottleneck model: :math:`N_e = 10{,}000`
   for :math:`t < 1{,}000`, :math:`N_e = 500` for :math:`1{,}000 \leq t <
   2{,}000`, :math:`N_e = 10{,}000` for :math:`t \geq 2{,}000`. Run the
   M-step and check whether the bottleneck is recovered. How does the number
   of trees affect accuracy?

.. admonition:: Exercise 2: EM convergence

   Run the full EM for 20 iterations, starting from a deliberately wrong
   initial :math:`N_e`. Plot :math:`\hat{N}_e` at each iteration. How many
   iterations are needed for convergence? Does the starting value matter?

.. admonition:: Exercise 3: Epoch boundary sensitivity

   Run the M-step with different numbers of epochs (5, 10, 20, 50). How does
   the resolution of the inferred population size history change? What
   happens when there are too many epochs (overfitting)?
