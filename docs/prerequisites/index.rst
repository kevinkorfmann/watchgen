.. _prerequisites:

================================
The Workbench (Prerequisites)
================================

Before you can build a Timepiece, you need the right tools on your workbench.

A watchmaker's bench holds files, pliers, tweezers, loupes -- each one essential for
specific tasks, each one mastered individually before being used in combination. Our
workbench holds mathematical and biological concepts: the foundational ideas that every
algorithm in this book is built upon.

This section covers eight topics. Each one is self-contained and builds from the ground
up -- we don't assume you've seen these ideas before. If you *have* encountered them,
the treatment here will sharpen your understanding and connect the concepts directly to
the algorithms we'll build later.

How to Use the Workbench
=========================

You do not need to memorize every derivation before opening a Timepiece. A useful
way to read this section is to make two passes:

1. On the first pass, learn the vocabulary, the generative story, and what each
   equation computes. Run the smallest code example and check that you can explain
   its inputs and output.
2. On the second pass, return to a derivation when a Timepiece uses it. At that
   point the symbols have a concrete job, so details such as scaling conventions
   and boundary conditions are easier to retain.

The examples assume only basic Python familiarity: variables, functions, loops,
and NumPy arrays. Calculus, linear algebra, probability notation, and population-
genetic terminology are introduced when they first become necessary. Code blocks
are pedagogical implementations rather than replacements for the production
libraries discussed in the Timepiece chapters.

.. admonition:: A quick comprehension check

   For each worked example, ask three questions before moving on: **What is
   observed? What is hidden or unknown? What probability model connects them?**
   For a simulator, replace the last two questions with: **What is sampled, and
   from which distribution?** These questions expose most misunderstandings before
   they become buried under notation.

Suggested Reading Order
========================

The suggested reading order is:

1. **Likelihood-Based Probabilistic Inference** -- The inferential logic that unifies
   the inference Timepieces. Before diving into any specific inference algorithm, understand
   the framework they all share: the likelihood function, maximum likelihood estimation,
   and Bayesian inference. This chapter explains how we go from observed genetic data to
   conclusions about evolutionary history, and situates the likelihood-based approach
   alongside the emerging neural-network paradigm. *(Start here -- it frames everything
   that follows.)*

2. **Coalescent Theory** -- How to think about ancestry backwards in time. This is the
   biological foundation: the idea that the genealogical history of a sample can be
   described by a branching tree, and that the shape of this tree is governed by
   simple probabilistic rules. We'll introduce the exponential distribution, the
   Poisson process, and the fundamental connection between population size and
   coalescence time. *(This is our most important tool -- it goes into every Timepiece.)*

3. **Ancestral Recombination Graphs** -- When a single tree isn't enough. Recombination
   means that different parts of the genome can have different genealogical histories.
   An ARG captures this full picture: the complete history of a sample, including all
   the places where the tree changes. We'll explain what recombination is, why it
   complicates things, and how the tree sequence data structure makes the complexity
   manageable.

4. **Hidden Markov Models** -- The computational engine behind most methods. An HMM is
   a mathematical framework for inferring hidden information from noisy observations.
   We'll build one from scratch, implement the forward algorithm, and show how a
   clever trick (the Li-Stephens structure) makes it fast enough for genomic data. If
   you've never seen an HMM before, this chapter will give you everything you need.

5. **The Sequentially Markov Coalescent** -- Making the impossible tractable. The full
   coalescent with recombination is not Markov, which means we can't use HMMs directly.
   The SMC is an approximation that restores the Markov property by restricting which
   ancestral events are retained. We'll explain exactly what is changed and why the
   consequences depend on the inferential setting.

6. **The Diffusion Approximation** -- When :math:`N` is large, the discrete Wright-Fisher
   model converges to a continuous diffusion process governed by a partial differential
   equation (the Fokker-Planck equation). This chapter develops the diffusion limit,
   stochastic differential equations, boundary conditions, stationary distributions, and
   finite-difference numerical methods. *(Essential for moments and dadi.)*

7. **Ordinary Differential Equations** -- Many Timepieces reduce their core computation
   to solving a system of ODEs. This chapter covers ODE fundamentals, Euler's method,
   Runge-Kutta solvers, coupled systems, stiffness, and the matrix exponential.
   *(Essential for moments; useful for SMC++ and momi2.)*

8. **Markov Chain Monte Carlo** -- When the posterior distribution over genealogies is too
   complex to compute directly, MCMC provides a principled way to sample from it. This
   chapter covers Bayesian inference, the Metropolis-Hastings algorithm, Gibbs sampling,
   and convergence diagnostics. *(Essential for ARGweaver and SINGER.)*

The first five tools -- probabilistic inference, the coalescent, the ARG, the HMM, and
the SMC -- form the inferential, biological, and computational backbone of the book. The
last three -- diffusion theory, ODEs, and MCMC -- provide the mathematical machinery that
specific Timepieces rely on. Together, these eight instruments equip you to build any
Timepiece in the collection.

Choose a Shorter Route
=======================

If you already know which family of methods interests you, use the routes below.
The arrows mean "read before," not that every later chapter uses every equation in
the earlier one.

.. list-table:: Goal-oriented routes through the prerequisites
   :header-rows: 1
   :widths: 28 45 27

   * - Goal
     - Recommended route
     - Example Timepieces
   * - Simulate genealogies and genomes
     - Coalescent Theory :math:`\rightarrow` ARGs
     - msprime, discoal, SLiM
   * - Copying models and tree-sequence reconstruction
     - ARGs :math:`\rightarrow` HMMs
     - LSHMM, tsinfer, threads, Relate
   * - Infer history along a recombining genome
     - Probabilistic Inference :math:`\rightarrow` Coalescent Theory
       :math:`\rightarrow` ARGs :math:`\rightarrow` HMMs
       :math:`\rightarrow` SMC
     - PSMC, SMC++, Gamma-SMC, SINGER, ARGweaver
   * - Infer demography from an allele-frequency spectrum
     - Probabilistic Inference :math:`\rightarrow` Diffusion Approximation;
       add ODEs for moment-based methods
     - dadi, moments
   * - Compute an SFS from genealogical transitions
     - Probabilistic Inference :math:`\rightarrow` Coalescent Theory
       :math:`\rightarrow` ODEs
     - momi2
   * - Understand posterior sampling over genealogies
     - Probabilistic Inference :math:`\rightarrow` Coalescent Theory
       :math:`\rightarrow` ARGs :math:`\rightarrow` HMMs
       :math:`\rightarrow` SMC :math:`\rightarrow` MCMC
     - ARGweaver, SINGER

These are minimum conceptual routes, not software installation instructions. A
Timepiece's overview identifies any additional machinery unique to that method.

Notation and Scaling: Read the Labels
=======================================

Population-genetic formulas often look contradictory because authors measure
time and rates in different units. This book always states the convention near a
derivation, but the following translations are worth keeping visible:

.. list-table:: Common quantities in biological and scaled units
   :header-rows: 1
   :widths: 20 35 45

   * - Quantity
     - Biological units
     - Common scaled form in a diploid model
   * - Time
     - :math:`g` generations
     - :math:`t=g/(2N_e)` coalescent or diffusion time units
   * - Mutation
     - :math:`\mu` per generation per base
     - :math:`\theta=4N_e\mu`
   * - Recombination
     - :math:`r` per generation per base
     - :math:`\rho=4N_e r`
   * - Genic selection
     - :math:`s` per generation
     - Often :math:`\gamma=2N_e s`

The factor of two changes for haploid models, and some papers absorb factors of
two into :math:`s`, :math:`\theta`, :math:`\rho`, or time. Never substitute a
rate into a formula on the strength of its symbol alone: check its units and the
definition used in that chapter. Likewise, a probability mass, a probability
density, and an expected site-frequency intensity can all be written with a
symbol such as :math:`p` or :math:`\phi`, but they have different units and
normalization rules.

.. admonition:: Worked unit conversion

   Suppose :math:`N_e=10{,}000`, :math:`\mu=10^{-8}` mutations per generation
   per base, and :math:`r=10^{-8}` recombinations per generation per base. Then
   the per-base scaled rates are
   :math:`\theta=4N_e\mu=4\times10^{-4}` and
   :math:`\rho=4N_e r=4\times10^{-4}`. Across a 100 kb region, the corresponding
   region-wide values are :math:`\theta L=40` and :math:`\rho L=40`. Confusing
   a per-base rate with a region-wide rate therefore changes this example by a
   factor of 100,000. Whenever a function also accepts ``sequence_length``, check
   whether it expects the per-base rate or the already-multiplied total.

.. admonition:: Do I need to read all of these?

   It depends on which Timepiece you want to build. **Probabilistic Inference** and
   **Coalescent Theory** provide the broadest foundation, but a reader interested
   only in simulation can begin directly with the coalescent. **HMMs** are needed
   for copying, threading, and sequential-coalescent methods. **ARGs** explain the
   genealogical object reconstructed by tree-sequence methods, while the **SMC** is
   specifically needed by sequential-coalescent approximations such as PSMC and
   SINGER. **The Diffusion Approximation** is central to dadi and moments; **ODEs**
   are central to moments, SMC++, and momi2. **MCMC** is most useful when studying
   posterior samplers such as ARGweaver and SINGER. If you are unsure, read the
   first five chapters in order and add the last three when a Timepiece points to
   them.

.. toctree::
   :maxdepth: 2

   probabilistic_inference
   coalescent_theory
   args
   hmms
   smc
   diffusion_approximation
   odes
   mcmc
