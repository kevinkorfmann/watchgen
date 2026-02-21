.. _momi2_overview:

====================
Overview of momi2
====================

   *Before assembling the watch, lay out every part and understand what it does.*

What Does momi2 Do?
=====================

Given DNA sequence data from one or more populations, ``momi2`` infers the
**demographic history** that produced the observed patterns of genetic variation.

.. math::

   \text{Input: } \mathbf{D} = \text{observed SFS from } k \text{ populations}

.. math::

   \text{Output: } \hat{\boldsymbol{\Theta}} = \text{best-fit demographic parameters (sizes, times, admixture fractions)}

Concretely, ``momi2`` asks: *What history of population size changes, splits,
and admixture events would produce a frequency spectrum that looks like the one
we observe?*

Think of a fine mechanical watch. A watchmaker can deduce every gear ratio
inside the case by studying the positions of the hands on the dial. ``momi2``
works the same way: it computes the expected hand positions (the expected SFS)
for a given gear configuration (a demographic model), then adjusts the gears
until the hands match what is observed. But unlike ``moments``, which turns the
gears forward in time to see where the hands end up, ``momi2`` **traces the
hands backward** through the coalescent to figure out what gear configuration
must have produced them.

From the inferred history, you learn:

- **Population sizes** through time -- expansions, contractions, bottlenecks
- **Divergence times** -- when populations split from each other
- **Admixture fractions** -- the proportion of ancestry contributed by each
  source during a pulse admixture event
- **Growth rates** -- exponential growth or decline within epochs

How momi2 Differs from moments and dadi
=========================================

All three tools infer demography from the SFS. The difference is *how* they
compute the expected SFS under a given model.

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Property
     - dadi
     - moments
     - momi2
   * - Direction
     - Forward (diffusion PDE)
     - Forward (moment ODEs)
     - Backward (coalescent)
   * - Core equation
     - Wright-Fisher diffusion PDE on a frequency grid
     - System of ODEs for SFS entries
     - Tensor products over a demographic event tree
   * - Population model
     - Continuous diffusion
     - Continuous diffusion (via moments)
     - Moran model (discrete)
   * - Gradient computation
     - Finite differences
     - Analytic or finite differences
     - Automatic differentiation (autograd)
   * - Multi-population scaling
     - Grid grows as :math:`O(n^D)` -- expensive beyond 3 populations
     - ODE system grows polynomially -- practical for 4--5 populations
     - Tensor operations along event tree -- scales well for many populations
   * - Selection
     - Yes
     - Yes
     - No (neutral models only)

.. admonition:: The key trade-off

   ``momi2`` gives up selection modeling in exchange for two advantages:
   exact gradients via automatic differentiation (enabling faster, more
   reliable optimization) and a coalescent formulation that scales more
   naturally to many populations. If you need selection, use ``moments``
   or ``dadi``. If you have many populations and neutral models, ``momi2``
   is often the better choice.

The Key Innovation: Tensors + Autograd
=======================================

The central insight of ``momi2`` is that the expected SFS under any demographic
model can be written as a sequence of **tensor operations** -- matrix
multiplications, convolutions, and antidiagonal summations -- applied to
likelihood tensors that track allele configurations across populations.

Each operation corresponds to a demographic event:

- **Moran transition** (a period of constant or exponential population size):
  multiply the likelihood tensor along the relevant axis by the Moran
  transition matrix :math:`P(t) = V e^{\Lambda t} V^{-1}`
- **Population merge** (a split event, viewed backward in time): convolve two
  population axes into one
- **Admixture pulse**: contract the likelihood tensor with a 3-tensor encoding
  the binomial mixing of lineages

Because every one of these operations is differentiable, the Python library
``autograd`` can compute exact gradients of the log-likelihood with respect to
all demographic parameters by backpropagating through the entire computation
graph. No hand-coded Jacobians, no finite-difference approximations, no
truncation error.

The Moran Model Connection
===========================

Within each epoch (a period between demographic events), ``momi2`` needs to
compute how the distribution of lineage counts evolves over time. It uses the
**Moran model** -- a continuous-time Markov chain on the number of derived
alleles in a sample of size :math:`n`.

The Moran model has a tridiagonal rate matrix with a known eigendecomposition.
This means the transition matrix for *any* time :math:`t` can be computed in
closed form:

.. math::

   P(t) = V \, e^{\Lambda t} \, V^{-1}

where :math:`V` is the matrix of eigenvectors and :math:`\Lambda` is the diagonal
matrix of eigenvalues. No numerical ODE integration is needed -- just a single
matrix operation per epoch.

For epochs with exponential growth, the time is rescaled by integrating the
inverse population size: :math:`\tau = \int_0^T 2/N(s)\, ds`, and the expected
coalescence times are computed via closed-form expressions involving exponential
integrals.

Parameters
===========

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Meaning
     - Typical units
   * - :math:`N_e`
     - Effective population size (reference)
     - Individuals
   * - :math:`N_i(t)`
     - Population size of deme :math:`i` at time :math:`t`
     - Individuals (or scaled by :math:`N_e`)
   * - :math:`T_{\text{split}}`
     - Time of population split (backward from present)
     - Generations (or scaled: :math:`T / 4N_e`)
   * - :math:`f`
     - Admixture fraction in a pulse event
     - Proportion (0 to 1)
   * - :math:`g`
     - Exponential growth rate within an epoch
     - Per generation
   * - :math:`\mu`
     - Per-site per-generation mutation rate
     - :math:`\sim 10^{-8}`

The Flow in Detail
===================

.. code-block:: text

   Step 1: Observed SFS
       Count derived allele frequencies across k populations
       from SNP data. Result: a k-dimensional histogram.

   Step 2: Demographic model
       Specify a population tree: which populations split when,
       with what sizes, growth rates, and admixture events.

   Step 3: Expected SFS via tensor computation
       Traverse the event tree bottom-up (leaves to root):
       - At each leaf: initialize a likelihood tensor
       - At each epoch boundary: apply Moran transition matrix
       - At each split: convolve two population tensors
       - At each pulse: apply admixture 3-tensor
       Accumulate the expected SFS along the way.

   Step 4: Likelihood computation
       Compare observed and expected SFS via composite
       log-likelihood (multinomial or Poisson).

   Step 5: Optimization
       Compute gradients via autograd, then optimize
       parameters using TNC, L-BFGS-B, ADAM, or SVRG.

   Step 6: Uncertainty quantification
       Use the autograd Hessian or bootstrap to estimate
       confidence intervals on inferred parameters.

Ready to Build
===============

The next four chapters disassemble ``momi2`` gear by gear:

- :ref:`coalescent_sfs` -- **The dial**: how expected branch lengths in the
  coalescent translate into expected SFS entries, and how demographic events
  modify these expectations.

- :ref:`moran_model` -- **The escapement**: the Moran model as a continuous-time
  Markov chain, its eigendecomposition, and how it governs lineage dynamics
  within each epoch.

- :ref:`tensor_machinery` -- **The gear train**: likelihood tensors, convolution
  for population merges, the junction tree algorithm, and how the full SFS is
  assembled from parts.

- :ref:`momi2_inference` -- **The mainspring**: autograd-powered optimization,
  composite likelihoods, stochastic gradient methods, and uncertainty
  quantification.
