.. _phlash_timepiece:

====================================
Timepiece XIV: PHLASH
====================================

   *Population History Learning by Averaging Sampled Histories*

The Mechanism at a Glance
==========================

PHLASH is a **Bayesian** method for inferring population size history
:math:`N(t)` from whole-genome sequencing data. It is the successor to PSMC
(:ref:`Timepiece I <psmc_timepiece>`), inheriting PSMC's coalescent HMM
framework but extending it in four fundamental ways:

1. It combines **two sources of information** -- the site frequency spectrum
   (SFS) from many individuals *and* the pairwise coalescent HMM from diploid
   genomes -- into a single composite likelihood.

2. It uses **random time discretizations** whose biases cancel when averaged,
   eliminating the systematic error that plagues fixed-grid approaches.

3. It computes gradients with a novel **score function algorithm** that is
   30--90x faster than automatic differentiation.

4. It samples from the **posterior distribution** over demographic histories
   using Stein Variational Gradient Descent (SVGD), a GPU-parallel algorithm
   that produces uncertainty estimates rather than point estimates.

.. math::

   \text{Input: } \text{Whole-genome data from one or more diploid individuals}

.. math::

   \text{Output: } \text{Posterior distribution over } N(t) \text{ (population size as a function of time)}

Where PSMC is a two-hand watch -- two haplotypes, one HMM, a point estimate --
PHLASH is a **grand complication**: multiple data sources, randomized internal
calibration, a custom-built gear train for fast gradient computation, and a
Bayesian mechanism that reports not just the time but the *uncertainty* in the
time. If PSMC tells you "the population was 50,000 at that epoch," PHLASH
tells you "the population was 50,000 with a 95% credible interval of
30,000--80,000."

.. admonition:: Primary Reference

   :cite:`phlash`

The four gears of PHLASH:

1. **The Composite Likelihood** (the mainspring) -- A two-part likelihood that
   combines the site frequency spectrum (SFS) from many individuals with the
   coalescent HMM likelihood from diploid pairs (inherited from PSMC). Two
   independent views of the same demographic history, fused into a single
   objective.

2. **Random Time Discretization** (the tourbillon) -- Randomized time
   breakpoints whose discretization biases cancel when averaged across many
   draws, like a tourbillon rotating the escapement to cancel positional
   errors in a mechanical watch.

3. **The Score Function Algorithm** (the gear train) -- An :math:`O(LM^2)`
   algorithm for computing the gradient of the log-likelihood, 30--90x faster
   than reverse-mode automatic differentiation. This is what makes SVGD
   practical at genomic scale.

4. **Stein Variational Gradient Descent** (the winding mechanism) -- A
   GPU-parallel posterior sampling algorithm that maintains a set of
   "particles" (candidate demographic histories) and iteratively pushes them
   toward the posterior distribution using gradient information and a
   repulsive kernel that maintains diversity.

These gears mesh together into a complete Bayesian inference machine:

.. code-block:: text

   Whole-genome data (one or more diploid individuals)
                      |
                      v
            +-----------------------+
            |  COMPUTE COMPOSITE    |
            |  LIKELIHOOD           |
            |                       |
            |  SFS likelihood       |
            |  (many individuals)   |
            |        +              |
            |  Coalescent HMM       |
            |  (diploid pairs)      |
            +-----------------------+
                      |
                      v
            +-----------------------+
            |  RANDOM DISCRETIZE    |
            |                       |
            |  Sample M breakpoints |
            |  Average over draws   |
            |  to cancel bias       |
            +-----------------------+
                      |
                      v
            +-----------------------+
            |  SCORE FUNCTION       |
            |  ALGORITHM            |
            |                       |
            |  O(LM^2) gradient     |
            |  (30-90x faster       |
            |   than autodiff)      |
            +-----------------------+
                      |
                      v
   +--------> SVGD UPDATE              |
   |          Push particles toward    |
   |          posterior using:          |
   |          - gradient (attraction)  |
   |          - kernel (repulsion)     |
   |                    |              |
   |              converged? NO        |
   |                    |              |
   +--------------------+              |
                        |              |
                       YES             |
                        |              |
                        v              |
              +-----------------------+
              |  POSTERIOR OVER N(t)  |
              |                       |
              |  Credible intervals   |
              |  Uncertainty bands    |
              |  Multiple histories   |
              +-----------------------+
                        |
                        v
                Posterior distribution
                over population size history

.. admonition:: Prerequisites for this Timepiece

   Before starting PHLASH, you should have worked through:

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times and
     their dependence on population size
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm
   - :ref:`The SMC <smc>` -- the Markov approximation to the coalescent with
     recombination
   - :ref:`PSMC (Timepiece I) <psmc_timepiece>` -- the coalescent HMM that
     PHLASH inherits and extends
   - :ref:`The Frequency Spectrum <the_frequency_spectrum>` -- the SFS and its
     relationship to demographic history (from the moments Timepiece)
   - Basic Bayesian inference (prior, likelihood, posterior) -- introduced
     inline as needed

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   composite_likelihood
   random_discretization
   score_function
   svgd_inference
   demo
