.. _dadi_timepiece:

====================================
Timepiece XI: dadi
====================================

   *Inferring demographic history by solving the Wright-Fisher diffusion -- the PDE that moments sought to avoid.*

The Mechanism at a Glance
==========================

``dadi`` (Diffusion Approximations for Demographic Inference) is a method for
**demographic inference** -- learning a population's history from patterns in
its DNA variation. Like ``moments`` (:ref:`Timepiece X <moments_timepiece>`),
it works with the **site frequency spectrum** (SFS) as its summary statistic
and uses composite likelihood optimization to fit demographic models.

But the internal mechanism is different. Where ``moments`` derives ordinary
differential equations that govern the SFS entries directly, ``dadi`` takes the
classical approach: it **solves the Wright-Fisher diffusion equation** -- a
partial differential equation (PDE) governing the continuous allele frequency
density :math:`\phi(x, t)`. The expected SFS is then extracted from this density
by integrating against binomial sampling probabilities.

Think of it this way: ``moments`` tracks how each hand on the dial advances
(one ODE per SFS entry). ``dadi`` instead models the full shape of every gear
tooth -- the continuous density of allele frequencies -- and reads the hand
positions from the gear profile. More work per tick, but the gear-level view
reveals details (like the full frequency density under selection) that the
hand-level view abstracts away.

``dadi`` was the first tool to use the diffusion approximation for
multi-population demographic inference from the joint SFS
(Gutenkunst et al. 2009), and it remains the predecessor against which
``moments`` and ``momi2`` define themselves. Understanding ``dadi`` is
understanding the foundation on which SFS-based inference was built.

.. admonition:: Primary Reference

   :cite:`dadi`

The four gears of ``dadi``:

1. **The Frequency Spectrum** (the dial) -- The same SFS as ``moments``, but
   ``dadi`` works with the continuous frequency density :math:`\phi(x, t)` from
   which the discrete SFS is sampled. The density is the gear; the SFS is the
   hand position read from it.

2. **The Diffusion Equation** (the escapement) -- The Wright-Fisher diffusion
   PDE governing the evolution of :math:`\phi(x, t)` under drift, selection,
   mutation, and migration. This is the fundamental equation of population
   genetics -- the continuous-time limit of the Wright-Fisher model. Each
   evolutionary force contributes a term to the PDE.

3. **Numerical Integration** (the gear train) -- A finite-difference scheme on
   a nonuniform frequency grid, with implicit time-stepping and Richardson
   extrapolation to correct grid bias. This is the computational engine that
   transforms the continuous PDE into a discrete computation. The grid design
   and extrapolation scheme are ``dadi``'s key engineering innovations.

4. **Demographic Inference** (the mainspring) -- Poisson composite likelihood
   optimization via BFGS, with Godambe Information Matrix for uncertainty
   quantification. The same statistical framework as ``moments``, applied to
   ``dadi``'s PDE-computed expected SFS.

These gears mesh together into a complete inference framework:

.. code-block:: text

   Observed DNA variation
           |
           v
   +-----------------------+
   |  FREQUENCY SPECTRUM   |
   |                       |
   |  Summarize variation  |
   |  as allele frequency  |
   |  histogram (SFS)      |
   +-----------------------+
           |
           v
   +-----------------------+
   |  DIFFUSION EQUATION   |
   |                       |
   |  Solve the Wright-    |
   |  Fisher PDE on a      |
   |  frequency grid to    |
   |  get phi(x, t)        |
   +-----------------------+
           |
           v
   +-----------------------+
   |  NUMERICAL            |
   |  INTEGRATION          |
   |                       |
   |  Finite differences,  |
   |  nonuniform grid,     |
   |  Richardson extrap.   |
   |  -> expected SFS      |
   +-----------------------+
           |
           v
   +-----------------------+
   |  DEMOGRAPHIC          |
   |  INFERENCE            |
   |                       |
   |  Poisson composite    |
   |  likelihood + BFGS    |
   |  optimization         |
   +-----------------------+
           |
           v
   Demographic history:
   population sizes, split
   times, migration rates,
   selection coefficients

dadi vs. moments
==================

``dadi`` and ``moments`` solve the same problem with different internal
mechanisms. Understanding the trade-offs clarifies when to use each:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - dadi
     - moments
   * - **Core equation**
     - Wright-Fisher diffusion PDE
     - Moment ODEs for SFS entries
   * - **What it tracks**
     - Continuous density :math:`\phi(x, t)`
     - Discrete SFS entries :math:`\phi_j`
   * - **Frequency discretization**
     - Grid of :math:`n` points in :math:`[0, 1]`
     - None (SFS entries are the state)
   * - **Multi-population scaling**
     - :math:`O(n^P)` grid points
     - :math:`O(\prod n_i)` SFS entries
   * - **Grid artifacts**
     - Yes (mitigated by extrapolation)
     - None
   * - **Selection handling**
     - Natural (term in PDE)
     - Natural (term in ODEs)
   * - **Key innovation**
     - First multi-pop diffusion tool
     - Grid-free moment equations

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- the relationship between
     population size and genetic diversity
   - Familiarity with the site frequency spectrum is helpful but not required --
     see :ref:`The Frequency Spectrum <the_frequency_spectrum>` in the moments
     Timepiece for a ground-up introduction
   - Basic partial differential equations are helpful (we derive everything
     we need)

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   diffusion_equation
   numerical_integration
   demographic_inference
   demo
