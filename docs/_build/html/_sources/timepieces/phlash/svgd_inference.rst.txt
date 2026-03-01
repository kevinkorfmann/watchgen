.. _phlash_svgd:

============================================
Stein Variational Gradient Descent (SVGD)
============================================

   *The winding mechanism: converting gradient energy into the organized motion of the posterior.*

In a mechanical watch, the **winding mechanism** converts the energy you
apply (turning the crown) into the stored tension of the mainspring, which
then powers the entire movement. The winding process must be controlled: too
much tension and the spring breaks; too little and the watch stops.

SVGD plays the same role in PHLASH. It takes the gradient of the log-posterior
(computed by the score function algorithm) and converts it into motion of the
particles -- each particle is a candidate demographic history, and SVGD
steers them all toward the posterior distribution. Like a winding mechanism,
SVGD must balance two forces: **attraction** toward high-probability regions
(the data pull) and **repulsion** between particles (the diversity push). The
result is a collection of particles that collectively represent the posterior
distribution over population size histories.


.. admonition:: Biology Aside -- Why posterior distributions matter for evolutionary biology

   A single best-fit demographic history (the "point estimate") can be
   misleading. Real genomic data often admit many plausible histories:
   perhaps a strong bottleneck 50,000 years ago fits equally well as a
   moderate decline starting 80,000 years ago. The **posterior distribution**
   captures this ambiguity -- it gives the probability of every possible
   demographic history given the data. From the posterior, biologists can
   ask: *Is the bottleneck real, or could the data be explained without one?
   How precisely can we date the out-of-Africa migration? Are two competing
   models distinguishable?* These questions require not just a single answer
   but a measure of uncertainty -- which is exactly what SVGD provides.

Why Not MCMC?
==============

Traditional Bayesian inference uses Markov chain Monte Carlo (MCMC) to sample
from the posterior. MCMC generates a single chain of correlated samples by
proposing random perturbations and accepting or rejecting them. It has two
well-known limitations for problems like PHLASH's:

1. **Sequential by nature.** Each MCMC sample depends on the previous one.
   You cannot easily parallelize MCMC across multiple chains on a GPU (each
   chain must wait for its predecessor).

2. **Mixing in high dimensions.** PHLASH's parameter space has :math:`M`
   dimensions (one population size per time interval, with :math:`M = 32` to
   64). MCMC methods can mix slowly in such spaces, requiring many iterations
   to explore the full posterior.

SVGD addresses both limitations:

- It maintains :math:`J` particles **in parallel**, updating all of them
  simultaneously. On a GPU, this parallelism translates directly into
  speedup.

- It uses **gradient information** to guide the particles, avoiding the
  random-walk behavior that slows MCMC in high dimensions.


The SVGD Algorithm
===================

SVGD maintains a set of :math:`J` particles
:math:`\{\boldsymbol{h}^{(1)}, \ldots, \boldsymbol{h}^{(J)}\}`, where each
:math:`\boldsymbol{h}^{(j)} = \log \boldsymbol{\eta}^{(j)}` is a log-space
demographic history. At each iteration, every particle is updated by:

.. math::

   \boldsymbol{h}^{(j)} \leftarrow \boldsymbol{h}^{(j)}
   + \epsilon \, \boldsymbol{\phi}^*(\boldsymbol{h}^{(j)})

where :math:`\epsilon` is a step size and :math:`\boldsymbol{\phi}^*` is the
**optimal perturbation direction**:

.. math::

   \boldsymbol{\phi}^*(\boldsymbol{h})
   = \frac{1}{J} \sum_{j=1}^{J} \left[
   k(\boldsymbol{h}^{(j)}, \boldsymbol{h}) \,
   \nabla_{\boldsymbol{h}^{(j)}} \log p(\boldsymbol{h}^{(j)} \mid \text{data})
   + \nabla_{\boldsymbol{h}^{(j)}} k(\boldsymbol{h}^{(j)}, \boldsymbol{h})
   \right]

This update has two terms:

1. **The attraction term**:
   :math:`k(\boldsymbol{h}^{(j)}, \boldsymbol{h}) \, \nabla \log p(\boldsymbol{h}^{(j)} \mid \text{data})`.
   This pushes particle :math:`\boldsymbol{h}` in the direction that nearby
   particles :math:`\boldsymbol{h}^{(j)}` want to move (toward higher
   posterior probability). The kernel :math:`k` weights the influence by
   proximity: particles close to :math:`\boldsymbol{h}` have more influence.

2. **The repulsion term**:
   :math:`\nabla_{\boldsymbol{h}^{(j)}} k(\boldsymbol{h}^{(j)}, \boldsymbol{h})`.
   This pushes particles *apart*, preventing them from collapsing to a single
   point. Without repulsion, all particles would converge to the MAP estimate
   (the posterior mode), giving a point estimate rather than a distribution.
   The repulsion ensures the particles spread out to cover the posterior's
   support.

.. admonition:: Plain-language summary -- Attraction and repulsion as forces

   Imagine the particles as a swarm of explorers searching a landscape for
   the highest peaks (the most probable demographic histories). **Attraction**
   makes each explorer move uphill, guided by what nearby explorers see --
   if your neighbor is on a steep slope, you are pulled in that direction.
   **Repulsion** prevents all explorers from piling onto the same peak -- it
   pushes them apart so they collectively map out the full shape of the
   mountain range, not just its highest point. After enough iterations, the
   explorers settle into a pattern where their density matches the shape of
   the landscape: many explorers on tall peaks (probable histories), few in
   valleys (improbable histories). This spatial distribution of particles
   *is* the posterior.


The Kernel
===========

SVGD typically uses a **radial basis function (RBF) kernel**:

.. math::

   k(\boldsymbol{h}, \boldsymbol{h}')
   = \exp\left(
   -\frac{\|\boldsymbol{h} - \boldsymbol{h}'\|^2}{2\sigma^2}
   \right)

where :math:`\sigma` is the kernel bandwidth. The bandwidth controls the
range of interaction between particles:

- **Large** :math:`\sigma`: particles interact over long distances. The
  repulsion is gentle but far-reaching, keeping particles well-separated.
  Good for exploring the posterior broadly.

- **Small** :math:`\sigma`: particles only interact with close neighbors.
  The repulsion is strong but local, allowing particles to cluster in
  high-probability regions.

A common heuristic is the **median trick**: set :math:`\sigma` to the median
of all pairwise distances between current particles, divided by
:math:`\sqrt{2 \log J}`. This adapts the bandwidth to the current spread of
the particles: as they converge, :math:`\sigma` shrinks; if they spread out,
:math:`\sigma` grows.

.. code-block:: python

   import numpy as np
   from scipy.spatial.distance import pdist, squareform

   def rbf_kernel(particles, bandwidth=None):
       """Compute the RBF kernel matrix and its gradients.

       Parameters
       ----------
       particles : ndarray, shape (J, M)
           J particles, each of dimension M.
       bandwidth : float or None
           Kernel bandwidth. If None, uses the median heuristic.

       Returns
       -------
       K : ndarray, shape (J, J)
           Kernel matrix K_{ij} = exp(-||h_i - h_j||^2 / (2 sigma^2)).
       grad_K : ndarray, shape (J, J, M)
           grad_K[i, j] = gradient of K_{ij} with respect to h_i.
       bandwidth : float
           The bandwidth used.
       """
       J, M = particles.shape
       dists = squareform(pdist(particles, 'sqeuclidean'))

       # Median heuristic for bandwidth
       if bandwidth is None:
           median_dist = np.median(pdist(particles, 'sqeuclidean'))
           bandwidth = np.sqrt(median_dist / (2 * np.log(J + 1)))
           bandwidth = max(bandwidth, 1e-5)

       K = np.exp(-dists / (2 * bandwidth**2))

       # Gradient: dK_{ij}/dh_i = K_{ij} * (h_j - h_i) / sigma^2
       diff = particles[None, :, :] - particles[:, None, :]  # (J, J, M)
       grad_K = K[:, :, None] * diff / bandwidth**2

       return K, grad_K, bandwidth

   def svgd_update(particles, grad_log_posterior, epsilon=0.01):
       """Perform one SVGD update step.

       Parameters
       ----------
       particles : ndarray, shape (J, M)
           Current particle positions.
       grad_log_posterior : ndarray, shape (J, M)
           Gradient of log-posterior at each particle.
       epsilon : float
           Step size.

       Returns
       -------
       particles_new : ndarray, shape (J, M)
           Updated particle positions.
       """
       J = particles.shape[0]
       K, grad_K, bw = rbf_kernel(particles)

       # phi*(h) = (1/J) * sum_j [ K(h_j, h) * grad_j + grad_K(h_j, h) ]
       # Attraction: K @ grad_log_posterior
       attraction = K @ grad_log_posterior / J      # (J, M)
       # Repulsion: sum_j grad_K[j, :, :]
       repulsion = grad_K.sum(axis=0) / J           # (J, M)

       phi = attraction + repulsion
       return particles + epsilon * phi

   # Demonstrate SVGD on a simple 2D target
   J = 16   # particles
   M = 2    # dimensions (for visualization clarity)
   rng = np.random.default_rng(42)
   particles = rng.normal(0, 2, size=(J, M))

   # Target: standard normal (grad log p = -h)
   for step in range(50):
       grad_lp = -particles  # gradient of log N(0, I)
       particles = svgd_update(particles, grad_lp, epsilon=0.1)

   print(f"After 50 SVGD steps ({J} particles, {M}D):")
   print(f"  Particle mean: {particles.mean(axis=0).round(3)}")
   print(f"  Particle std:  {particles.std(axis=0).round(3)}")
   print(f"  (Target: mean ~ 0, std ~ 1)")


GPU Parallelism
================

SVGD is naturally parallel in two ways:

1. **Across particles.** Each particle's gradient
   :math:`\nabla \log p(\boldsymbol{h}^{(j)} \mid \text{data})` can be
   computed independently. With :math:`J` particles on a GPU, all gradients
   are computed in a single batched operation.

2. **Across pairs.** The coalescent HMM gradient for each diploid individual
   is independent. If there are :math:`P` pairs, the :math:`J \times P`
   gradient computations can be parallelized.

PHLASH is implemented in **JAX**, which provides:

- **JIT compilation**: the forward-backward algorithm and score function
  are compiled to GPU machine code, eliminating Python interpreter overhead.

- **Vectorization** (``vmap``): the same compiled kernel is automatically
  applied across particles and pairs without writing explicit loops.

- **Automatic batching**: JAX handles memory management, distributing the
  computation across GPU cores.

The result is that SVGD with :math:`J = 32` particles, each evaluating the
composite likelihood over :math:`P` pairs with :math:`L \sim 10^7` positions
and :math:`M = 64` time intervals, runs in minutes on a modern GPU. The same
computation would take hours with sequential MCMC on a CPU.


The Full PHLASH Loop
=====================

Putting all four gears together, one iteration of PHLASH looks like this:

.. code-block:: text

   For each SVGD iteration t = 1, 2, ..., T:

     1. Sample a random time discretization G_t
        (tourbillon: different grid each iteration)

     2. For each particle j = 1, ..., J (in parallel on GPU):
        a. Build HMM transition matrix and emissions on grid G_t
        b. For each pair p = 1, ..., P (in parallel):
           - Forward pass: compute log-likelihood
           - Backward pass: compute posterior marginals
           - Score function: compute gradient of HMM log-likelihood
        c. Compute SFS log-likelihood and its gradient
        d. Add prior gradient (smoothness penalty)
        e. Total: gradient of log-posterior for particle j

     3. Compute kernel matrix K_{ij} = k(h^(i), h^(j))
        and kernel gradients

     4. SVGD update: for each particle j,
        h^(j) <- h^(j) + epsilon * phi*(h^(j))
        (attraction + repulsion)

     5. Adapt step size epsilon (e.g., Adam optimizer)

   After T iterations:
     Particles {h^(1), ..., h^(J)} approximate the posterior.
     Compute posterior mean, credible intervals, etc.

.. code-block:: python

   def phlash_loop(n_particles, M, n_iterations, observed_sfs,
                   sigma_prior=1.0, epsilon=0.01, rng=None):
       """Simplified PHLASH inference loop.

       Demonstrates the full pipeline: random discretization, score
       function, and SVGD update. Uses placeholder likelihoods.

       Parameters
       ----------
       n_particles : int
           Number of SVGD particles (J).
       M : int
           Number of time intervals.
       n_iterations : int
           Number of SVGD iterations.
       observed_sfs : ndarray
           Observed SFS.
       sigma_prior : float
           Smoothness prior scale.
       epsilon : float
           SVGD step size.
       rng : numpy.random.Generator or None

       Returns
       -------
       particles : ndarray, shape (J, M)
           Final particle positions (log population sizes).
       """
       if rng is None:
           rng = np.random.default_rng()

       # Initialize particles near the prior mean
       particles = rng.normal(0, 0.5, size=(n_particles, M))

       for t in range(n_iterations):
           # Step 1: sample a random grid (tourbillon)
           grid = sample_random_grid(M, rng=rng)

           # Step 2: compute gradient for each particle
           grads = np.zeros_like(particles)
           for j in range(n_particles):
               h = particles[j]
               # Prior gradient
               grad_prior = np.zeros(M)
               for k in range(1, M):
                   grad_prior[k] += (h[k-1] - h[k]) / sigma_prior**2
                   grad_prior[k-1] += (h[k] - h[k-1]) / sigma_prior**2

               # Placeholder likelihood gradient (pulls toward 0 = constant)
               grad_lik = -0.05 * h + rng.normal(0, 0.02, size=M)
               grads[j] = grad_lik + grad_prior

           # Steps 3-4: SVGD update (kernel + attraction + repulsion)
           particles = svgd_update(particles, grads, epsilon=epsilon)

       return particles

   # Run a small demonstration
   rng = np.random.default_rng(42)
   particles = phlash_loop(
       n_particles=8, M=16, n_iterations=100,
       observed_sfs=D_observed, epsilon=0.05, rng=rng
   )
   eta_particles = np.exp(particles)  # convert to population size
   posterior_mean = eta_particles.mean(axis=0)
   posterior_std = eta_particles.std(axis=0)

   print(f"PHLASH result ({8} particles, {16} intervals, 100 iterations):")
   print(f"  Posterior mean N_e (first 5): "
         f"{posterior_mean[:5].round(3)}")
   print(f"  Posterior std  N_e (first 5): "
         f"{posterior_std[:5].round(3)}")
   print(f"  (All particles provide uncertainty quantification)")


Convergence and Diagnostics
=============================

How do we know when SVGD has converged? Unlike MCMC, SVGD does not have a
well-established convergence diagnostic like the Gelman-Rubin
:math:`\hat{R}` statistic. In practice, convergence is monitored by:

- **Stabilization of particle positions.** When the particles stop moving
  appreciably between iterations, the algorithm has converged.

- **Stabilization of the posterior mean.** The average demographic history
  across particles should stabilize.

- **Kernel bandwidth.** If the median pairwise distance between particles
  stabilizes, the spread of the posterior approximation has converged.

- **Multiple runs.** Running SVGD from different initializations and checking
  that the resulting posterior approximations agree.


From Particles to Inference
=============================

After SVGD converges, the :math:`J` particles provide a discrete
approximation to the posterior distribution over demographic histories. From
these particles, we can compute:

- **Posterior mean**: :math:`\bar{\eta}(t) = \frac{1}{J} \sum_{j=1}^{J} \eta^{(j)}(t)`.
  This is the average demographic history, analogous to PSMC's point estimate
  but derived from the full posterior.

- **Credible intervals**: For each time point :math:`t`, sort the particle
  values :math:`\eta^{(1)}(t), \ldots, \eta^{(J)}(t)` and take the 2.5th
  and 97.5th percentiles for a 95% credible interval.

- **Posterior uncertainty bands**: Plot all :math:`J` trajectories together
  to visualize the full spread of plausible histories.

- **Model comparison**: The average log-likelihood across particles can
  serve as a proxy for the marginal likelihood, enabling comparison between
  different demographic model classes.

These outputs go beyond what PSMC can provide. Where PSMC gives a single line
on a plot and requires bootstrapping to estimate uncertainty, PHLASH gives a
cloud of lines whose spread directly reflects the posterior uncertainty. The
credible intervals are Bayesian: they say "there is a 95% probability that the
true history lies within this band," rather than the frequentist bootstrap's
"if we repeated the experiment many times, 95% of the intervals would contain
the truth."

.. admonition:: Biology Aside -- What posterior uncertainty looks like in practice

   In a PHLASH analysis, the output is a bundle of :math:`J` demographic
   history curves. Where the curves are tightly clustered, the data strongly
   constrain the population size -- for example, the out-of-Africa bottleneck
   in human data consistently appears as a narrow band of low :math:`N_e`
   around 50-70 kya. Where the curves spread apart, the data are ambiguous
   -- for instance, very ancient population sizes (>1 Mya) are poorly
   constrained because few lineages survive that far back. This visual
   representation makes it immediately clear which features of the inferred
   history are robust and which are uncertain -- information that is critical
   for drawing evolutionary conclusions and impossible to obtain from a
   single point estimate.


Summary
========

SVGD completes the PHLASH mechanism. The four gears -- composite likelihood,
random discretization, score function algorithm, and SVGD -- mesh together
into a Bayesian inference machine that infers population size history with
principled uncertainty quantification:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Gear
     - Watch analogy
     - Function
   * - Composite likelihood
     - Mainspring
     - Stores the information from SFS + coalescent HMM data
   * - Random discretization
     - Tourbillon
     - Cancels systematic discretization bias by rotating through grids
   * - Score function algorithm
     - Gear train
     - Transmits likelihood gradients to SVGD at :math:`O(LM^2)` cost
   * - SVGD
     - Winding mechanism
     - Converts gradients into posterior samples on GPU

The result is a watch that not only tells time but also tells you how
precisely it is keeping time -- a grand complication built atop the
foundations of PSMC.
