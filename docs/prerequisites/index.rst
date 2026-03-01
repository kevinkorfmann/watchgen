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

The suggested reading order is:

1. **Likelihood-Based Probabilistic Inference** -- The inferential logic that unifies
   every Timepiece. Before diving into any specific algorithm, you need to understand
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
   The SMC is an approximation that restores the Markov property at the cost of a tiny
   amount of accuracy. We'll explain exactly what's lost and why it doesn't matter much.

6. **The Diffusion Approximation** -- When :math:`N` is large, the discrete Wright-Fisher
   model converges to a continuous diffusion process governed by a partial differential
   equation (the Fokker-Planck equation). This chapter develops the diffusion limit,
   stochastic differential equations, boundary conditions, stationary distributions, and
   finite-difference numerical methods. *(Essential for moments, dadi, and momi2.)*

7. **Ordinary Differential Equations** -- Many Timepieces reduce their core computation
   to solving a system of ODEs. This chapter covers ODE fundamentals, Euler's method,
   Runge-Kutta solvers, coupled systems, stiffness, and the matrix exponential.
   *(Essential for moments; useful for SMC++ and momi2.)*

8. **Markov Chain Monte Carlo** -- When the posterior distribution over genealogies is too
   complex to compute directly, MCMC provides a principled way to sample from it. This
   chapter covers Bayesian inference, the Metropolis-Hastings algorithm, Gibbs sampling,
   and convergence diagnostics. *(Essential for ARGweaver, SINGER, and PHLASH.)*

The first five tools -- probabilistic inference, the coalescent, the ARG, the HMM, and
the SMC -- form the inferential, biological, and computational backbone of the book. The
last three -- diffusion theory, ODEs, and MCMC -- provide the mathematical machinery that
specific Timepieces rely on. Together, these eight instruments equip you to build any
Timepiece in the collection.

.. admonition:: Do I need to read all of these?

   It depends on which Timepiece you want to build. **Probabilistic Inference** and
   **Coalescent Theory** are essential for everything -- start there. **HMMs** are
   needed for PSMC, SINGER, and ARGweaver. **ARGs** and the **SMC** are needed for
   SINGER and tsinfer. **The Diffusion Approximation** and **ODEs** are needed for
   moments and dadi. **MCMC** is needed for ARGweaver, SINGER, and PHLASH. If you're
   not sure, start with Probabilistic Inference and Coalescent Theory, then work
   through the first five in order -- they build naturally on each other. The last
   three can be read independently as needed.

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
