.. _phlash_timepiece:

=========================================
Timepiece XIV: PHLASH — the Tourbillon
=========================================

PHLASH (population history learning by averaging sampled histories) is a
Bayesian descendant of PSMC.  It uses multiple diploid genomes, an allele
frequency spectrum (AFS) term, and a particle approximation to a posterior over
piecewise-constant coalescent-rate histories :cite:`phlash`.

The tourbillon metaphor is useful only at the ensemble level: individual
posterior draws have different endpoint-defined time grids, and summaries are
formed by evaluating the draws on a common time axis.  It is not a claim that
PHLASH repeatedly jitters every breakpoint or that averaging makes discretization
bias vanish.

.. admonition:: Verified scope

   The code in ``watchgen.mini_phlash`` is a small parity model of selected
   kernels, not a reimplementation of PHLASH.  Production PHLASH includes VCF and
   tree-sequence input, data chunking, JAX/CUDA likelihood kernels, automatic
   differentiation around the model parameters, optimization, diagnostics, and
   rescaling.  Use the official package for inference.

.. toctree::
   :maxdepth: 2

   overview
   composite_likelihood
   random_discretization
   score_function
   svgd_inference
   demo
