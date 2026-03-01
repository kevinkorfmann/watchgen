.. _phlash_overview:

====================
Overview of PHLASH
====================

   *Before assembling the watch, lay out every part and understand what it does.*

.. figure:: /_static/figures/fig_mini_phlash.png
   :width: 100%
   :align: center

   **PHLASH at a glance.** The four pillars of the PHLASH inference pipeline:
   composite SFS likelihood connecting data to demography, the RBF kernel
   measuring inter-particle similarity for SVGD, particle evolution under
   Stein Variational Gradient Descent, and random time discretisation for
   debiased gradient estimation.

What Does PHLASH Do?
=====================

Given whole-genome sequencing data from one or more diploid individuals,
PHLASH infers the **posterior distribution** over the population size history
:math:`N(t)`.

.. math::

   \text{Input: } \text{Whole-genome data from } n \text{ diploid individuals}

.. math::

   \text{Output: } p\bigl(\eta \mid \text{data}\bigr), \quad \eta(t) = \text{population size history}

Unlike PSMC, which returns a single best estimate of :math:`N(t)`, PHLASH
returns a **distribution** over possible histories -- a cloud of plausible
trajectories rather than a single line. Each trajectory in the cloud is
consistent with the data; the spread of the cloud tells you how certain or
uncertain the inference is.

Think of it this way. PSMC is a watch that shows you a single reading: "It is
3:17." PHLASH is a watch that shows you a distribution: "It is 3:17 give or
take two minutes, and here is a plot showing the probability density over all
possible times." Both tell time; PHLASH also tells you how well it is keeping
time.


Why a Successor to PSMC?
==========================

PSMC (:ref:`Timepiece I <psmc_timepiece>`) demonstrated that a single diploid
genome encodes deep demographic history. But it has three well-known
limitations:

1. **No uncertainty quantification.** PSMC uses the EM algorithm, which finds
   a point estimate (a single maximum-likelihood history). The standard
   approach to uncertainty -- bootstrapping over genomic blocks -- is
   computationally expensive and can underestimate true uncertainty because
   blocks are not truly independent.

2. **Discretization bias.** PSMC divides time into fixed intervals and
   estimates a constant population size within each interval. The choice of
   intervals affects the result: too few intervals miss real demographic
   events, while the spacing pattern (typically log-spaced) introduces
   systematic bias because the integral approximation errors do not cancel
   across a fixed grid.

3. **Limited data use.** PSMC analyzes one diploid genome at a time. When
   multiple genomes are available, it cannot easily combine their information.
   The pairwise approach (PSMC') extends to pairs of individuals but does not
   scale to many samples.

PHLASH addresses all three:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Limitation
     - PSMC
     - PHLASH
   * - Uncertainty
     - Point estimate + bootstrap
     - Full posterior via SVGD
   * - Discretization
     - Fixed grid, systematic bias
     - Random grids, bias cancels on average
   * - Data sources
     - One diploid (pairwise HMM only)
     - SFS (many individuals) + pairwise HMM
   * - Optimization
     - EM (local maximum, slow convergence)
     - SVGD (posterior sampling, GPU-parallel)
   * - Gradients
     - Not needed (EM uses closed-form updates)
     - Score function algorithm, :math:`O(LM^2)`


How PHLASH Extends the Coalescent HMM
=======================================

At its core, PHLASH still uses the same coalescent HMM as PSMC. Recall from
:ref:`Timepiece I <psmc_timepiece>`: the hidden states are discretized
coalescence time intervals, the observations are heterozygous/homozygous bins,
and the transition matrix encodes how the TMRCA changes between adjacent
positions under the SMC approximation.

PHLASH inherits this entire machinery but wraps it in a larger framework:

- The **coalescent HMM likelihood** is computed for each pair of haplotypes
  (or each diploid individual), exactly as in PSMC. This gives a likelihood
  :math:`\ell_{\text{HMM}}(\eta)` for a demographic history :math:`\eta`.

- The **SFS likelihood** is computed from the allele frequency distribution
  across all sampled individuals. This gives a second likelihood
  :math:`\ell_{\text{SFS}}(\eta)` that captures information from many
  individuals simultaneously.

- The two likelihoods are combined into a **composite likelihood**:

  .. math::

     \ell_{\text{comp}}(\eta) = \ell_{\text{SFS}}(\eta) + \ell_{\text{HMM}}(\eta)

  (on the log scale, so the composite log-likelihood is a sum). This composite
  likelihood is then used as the basis for Bayesian inference, combined with a
  smoothness prior on :math:`\eta`.

The composite likelihood is the **mainspring** of PHLASH -- the energy source
that drives the entire mechanism. But turning the mainspring requires computing
gradients efficiently, which is where the score function algorithm comes in.
And the gradients must be computed on a discretized time grid, which is where
random discretization cancels the bias. And the gradients drive SVGD, which
produces the posterior. Every gear depends on the one before it.


The Demographic Model
======================

PHLASH parameterizes the population size history as a **piecewise-constant
function** on a set of time intervals, just as PSMC does:

.. math::

   \eta(t) = \eta_k \quad \text{for } t \in [t_k, t_{k+1})

But unlike PSMC, the time breakpoints :math:`t_0 < t_1 < \cdots < t_M` are
not fixed across the analysis. Instead, PHLASH **randomly samples** the
breakpoints for each gradient computation. Different random grids yield
different discretization errors, but these errors have zero mean: they cancel
when averaged across draws.

The vector of population sizes :math:`\boldsymbol{\eta} = (\eta_0, \ldots,
\eta_{M-1})` lives in a high-dimensional space. PHLASH places a **smoothness
prior** on :math:`\boldsymbol{\eta}` -- a Gaussian process prior that
penalizes histories with large jumps between adjacent epochs. This prior acts
as a regularizer, preventing overfitting to noise while allowing the data to
drive the inference.


Parameters
===========

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Symbol
     - Physical meaning
   * - Population sizes
     - :math:`\eta_0, \ldots, \eta_{M-1}`
     - Effective population size in each time interval. The primary quantities
       of interest -- the posterior distribution over these values is the output.
   * - Mutation rate
     - :math:`\theta`
     - Population-scaled mutation rate, inherited from PSMC's parameterization.
       Controls the density of heterozygous sites per unit coalescence time.
   * - Recombination rate
     - :math:`\rho`
     - Population-scaled recombination rate. Controls how frequently the
       genealogy changes between adjacent positions.
   * - Time breakpoints
     - :math:`t_0, \ldots, t_M`
     - Boundaries of the piecewise-constant intervals. Randomly sampled in
       each gradient computation step; not treated as fixed parameters.
   * - Number of particles
     - :math:`J`
     - The number of candidate histories maintained by SVGD. More particles
       give a better approximation to the posterior but cost more computation.
   * - Smoothness prior
     - :math:`\Sigma`
     - The covariance structure of the Gaussian process prior on
       :math:`\log \boldsymbol{\eta}`. Controls how much the inferred history
       is allowed to fluctuate between adjacent time intervals.


The Flow in Detail
===================

.. code-block:: text

   Step 1: Data preparation
       From whole-genome data, compute:
       (a) SFS from all individuals
       (b) Heterozygosity sequences for each diploid individual
           (same binning as PSMC)

   Step 2: Initialize particles
       Draw J initial demographic histories from the prior.
       Each particle is a vector eta = (eta_0, ..., eta_{M-1}).

   Step 3: For each SVGD iteration:
       (a) Sample a random time discretization t_0 < t_1 < ... < t_M
       (b) For each particle j:
           - Compute SFS log-likelihood given eta^(j)
           - Compute coalescent HMM log-likelihood given eta^(j)
           - Sum to get composite log-likelihood
           - Compute gradient via score function algorithm
           - Add gradient of log-prior (smoothness penalty)
       (c) Compute SVGD update:
           - Attraction: gradient of log-posterior pushes particles
             toward high-probability regions
           - Repulsion: kernel gradient pushes particles apart
             to maintain diversity
       (d) Update all particles in parallel (on GPU)

   Step 4: After convergence:
       The J particles approximate the posterior distribution.
       Compute credible intervals, posterior mean, uncertainty bands.


Comparison to Related Methods
==============================

.. list-table::
   :header-rows: 1
   :widths: 18 18 18 18 28

   * - Property
     - PSMC
     - SMC++
     - MSMC2
     - PHLASH
   * - Data
     - 1 diploid
     - SFS + distinguished pair
     - Up to 8 haplotypes
     - SFS + diploid pairs
   * - Inference
     - EM (point estimate)
     - EM (point estimate)
     - EM (point estimate)
     - SVGD (posterior)
   * - Discretization
     - Fixed grid
     - Fixed grid
     - Fixed grid
     - Random grid (bias cancels)
   * - Gradients
     - Not needed
     - Autodiff
     - Not needed
     - Score function (:math:`O(LM^2)`)
   * - GPU support
     - No
     - No
     - No
     - Yes (JAX)
   * - Uncertainty
     - Bootstrap
     - Bootstrap
     - Bootstrap
     - Posterior samples


Ready to Build
===============

The next four chapters disassemble PHLASH gear by gear:

- :ref:`phlash_composite_likelihood` -- **The mainspring**: how the SFS
  likelihood and the coalescent HMM likelihood are computed and combined
  into a single composite objective.

- :ref:`phlash_random_discretization` -- **The tourbillon**: how random time
  breakpoints cancel discretization bias, turning a systematic error into
  zero-mean noise that vanishes under averaging.

- :ref:`phlash_score_function` -- **The gear train**: the :math:`O(LM^2)`
  algorithm that computes the gradient of the log-likelihood 30--90x faster
  than autodiff, making SVGD practical at genomic scale.

- :ref:`phlash_svgd` -- **The winding mechanism**: how SVGD uses gradients
  and a repulsive kernel to push particles toward the posterior distribution,
  producing uncertainty estimates on a GPU.

Each chapter derives the math, explains the intuition, and connects back to
the PSMC foundations from :ref:`Timepiece I <psmc_timepiece>`.

Let us start with the foundation: the composite likelihood.
