.. _phlash_overview:

===============
What PHLASH Is
===============

The inferential object is the diploid coalescent-rate function
:math:`\eta(t)`.  The effective population size is

.. math::

   N_e(t) = \frac{1}{2\eta(t)}.

PHLASH represents :math:`\eta` by 16 positive, piecewise-constant rates.  A
PSMC-like hidden Markov model (HMM) relates the hidden local time to most recent
common ancestor to binned homozygous, heterozygous, or missing observations.
The released sequence-data model combines a prior, a product of per-diploid PSMC
likelihoods, and an AFS score.  It then uses Stein variational gradient descent
(SVGD) to maintain an ensemble of model parameters :cite:`phlash`.

Ground truth used here
======================

This chapter was checked against:

* Terhorst's published 2025 paper and its Supplementary Note :cite:`phlash`;
* the manuscript archive, PHLASH 1.0.5 (Zenodo DOI
  ``10.5281/zenodo.16414354``); and
* the official repository at commit ``96a6e3f`` (package version 1.0.6).

The current source matters because it narrows several descriptions in the
paper.  In ``phlash.model.log_density``, the implemented sequence-data density
has three terms: prior, PSMC chunks, and transformed AFS.  Although the paper's
general equation also describes an LD-decay penalty, version 1.0.6 does not add
an LD term in that function.  Likewise, the optimized ``PSMCParams`` conversion
requires 16 intervals, and ``fit`` uses the tied pattern ``14*1+1*2`` by default.

Input and output
================

The public Python API uses ``phlash.contig`` to describe VCF/BCF or tree-sequence
input and ``phlash.fit`` for estimation.  It does not currently expose a command
line interface.  ``fit`` returns a list of ``DemographicModel`` posterior
particles.  Passing ``mutation_rate`` rescales coalescent units to generations;
otherwise the model remains in its internal scaling.

A held-out contig can be supplied as ``test_data``.  The implementation monitors
expected log-predictive density and can stop after the configured patience
period.  GPU execution is recommended, but the source falls back to a pure-JAX
kernel when CUDA is unavailable.

Mini-implementation boundary
============================

The mini code verifies exact identities and small recursions, but it is **not a replacement**
for PHLASH.  In particular it does not infer a history from a VCF,
claim calibrated credible intervals, reproduce the GPU score kernel, or turn the
composite posterior into an ordinary fully specified Bayesian posterior.
