.. _momi2_timepiece:

====================================
Timepiece X: momi2
====================================

   *Computing the expected frequency spectrum backward through the coalescent -- one tensor at a time.*

.. epigraph::

   "We introduce a new method for computing the expected sample frequency
   spectrum (SFS) for a tree of populations with arbitrary size histories."

   -- Kamm, Terhorst, Song, and Durbin (2017)

The Mechanism at a Glance
==========================

``momi2`` is a method for **demographic inference** -- learning a population's
history (size changes, splits, admixture events) from patterns in its DNA
variation. Like ``moments`` (:ref:`Timepiece IX <moments_timepiece>`), it works
with the **site frequency spectrum** (SFS) as its summary statistic. But the
internal mechanism is fundamentally different.

Where ``moments`` works *forward in time* -- deriving ODEs that push allele
frequencies through the diffusion process -- ``momi2`` works *backward in time*
through the **coalescent**. It asks: given a demographic model, what are the
expected branch lengths in the genealogy? And from those branch lengths, what
SFS do we expect to observe?

The key innovation is how ``momi2`` performs this computation. Rather than
simulating genealogies or solving differential equations, it uses **tensor
algebra**. The expected SFS is assembled as a product of tensors, one for each
demographic event, processed along a junction tree. Each tensor encodes how
lineage counts change at a split, an admixture pulse, or during a period of
constant (or exponential) population size. The population dynamics within each
epoch are governed by the **Moran model**, a continuous-time Markov chain whose
eigendecomposition allows efficient computation of transition probabilities for
arbitrary time spans.

The final ingredient is **automatic differentiation** (autograd). Because every
operation in the SFS computation is differentiable, ``momi2`` obtains exact
gradients of the likelihood with respect to all demographic parameters -- no
hand-coded derivatives, no finite differences. This enables efficient
gradient-based optimization and even Hessian computation for uncertainty
quantification.

The four gears of ``momi2``:

1. **The Coalescent SFS** (the dial) -- The expected frequency spectrum derived
   from coalescent theory: how expected branch lengths in the genealogy translate
   directly into expected SFS entries. This is the quantity ``momi2`` computes
   and compares against data.

2. **The Moran Model** (the escapement) -- The discrete population model that
   governs lineage dynamics within each epoch. Its eigendecomposition lets us
   compute transition probabilities for any time span in a single matrix
   operation, replacing numerical ODE integration.

3. **Tensor Machinery** (the gear train) -- The computational engine: likelihood
   tensors that track allele configurations across populations, assembled via
   convolution (for population merges), matrix multiplication (for Moran
   transitions), and antidiagonal summation (for coalescence). A junction tree
   algorithm processes demographic events in the correct order.

4. **Automatic Differentiation & Inference** (the mainspring) -- Autograd-powered
   optimization: exact gradients flow backward through the entire tensor
   computation, enabling efficient maximum-likelihood estimation with TNC,
   L-BFGS-B, or stochastic methods (ADAM, SVRG).

These gears mesh together into a complete inference framework:

.. code-block:: text

   Observed SFS
           |
           v
   +-----------------------+
   |  DEMOGRAPHIC MODEL    |
   |                       |
   |  Population tree with |
   |  sizes, split times,  |
   |  admixture fractions  |
   +-----------------------+
           |
           v
   +-----------------------+
   |  COALESCENT SFS       |
   |  COMPUTATION          |
   |                       |
   |  Tensor products over |
   |  the event tree:      |
   |  Moran transitions,   |
   |  population merges,   |
   |  admixture pulses     |
   +-----------------------+
           |
           v
   +-----------------------+
   |  LIKELIHOOD &         |
   |  OPTIMIZATION         |
   |                       |
   |  Composite log-lik +  |
   |  autograd gradients   |
   |  -> maximize via TNC  |
   |  or stochastic SGD    |
   +-----------------------+
           |
           v
   Demographic history:
   population sizes, split
   times, admixture fractions

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- understanding the relationship
     between genealogies, branch lengths, and genetic diversity
   - Basic linear algebra -- eigenvalues, matrix exponentials, tensor products
   - Familiarity with the site frequency spectrum is helpful but not required --
     see :ref:`The Frequency Spectrum <the_frequency_spectrum>` in the moments
     Timepiece for a ground-up introduction

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   coalescent_sfs
   moran_model
   tensor_machinery
   inference
