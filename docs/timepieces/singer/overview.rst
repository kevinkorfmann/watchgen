.. _singer_overview:

========
Overview
========

Input and target
================

SINGER takes phased whole-genome variation and targets an approximate
posterior over ARGs,

.. math::

   \Pr(G\mid D) \propto \Pr(D\mid G)\Pr(G),

under a constant-size, panmictic SMC model in its threading HMMs.  Posterior
sampling matters because local topology and coalescence time can remain
uncertain even when the mutation and recombination rates are known.  The
published implementation accepts a mutation rate or mutation map, a
recombination-to-mutation ratio or recombination map, a reference diploid
effective size, and a polarization probability.

The algorithm
=============

Initialization starts from an empty ARG and threads samples successively.  For
the next sample, branch sampling chooses a joining branch in each genomic bin;
conditional on that path, time sampling chooses a joining time.  This split is
an approximation: ARGweaver instead represents complete joining points in one
larger HMM.

After initialization, SINGER rescales node times.  Each MCMC proposal then:

#. introduces a cut in one marginal tree and propagates the equivalent cut
   through neighboring trees until recombination breaks equivalence;
#. removes the ancestral material above those cuts;
#. rethreads the cut-defined sub-ARG with the same two-stage HMM; and
#. accepts or rejects using the SGPR Metropolis-Hastings rule.

After each thinning interval the implementation rescales again and writes an
ARG sample.  Rescaling is therefore neither an operation after every newly
threaded haplotype nor an extra state inside the HMM.

.. code-block:: text

   phased variants
        |
        v
   iterative branch + time threading
        |
        v
   rescale initialized ARG
        |
        +--> cut-defined SGPR proposals --+
        |                                |
        +<------- repeat to thinning ----+
        |
        v
   rescale and write posterior sample

What the miniature covers
==========================

:mod:`watchgen.mini_singer` makes the following source-level mechanisms
executable:

* the branch joining mass and deterministic lineage-count approximation;
* the three-edge Poisson emission calculation;
* the Li-Stephens-like branch transition;
* the modified PSMC kernel and interval transition matrix;
* equal-ARG-length time windows and their piecewise-linear rescaling; and
* the SGPR tree-height acceptance approximation.

It omits VCF parsing, mutation mapping, stochastic traceback, the persistent
ARG representation, partial-branch propagation across actual recombinations,
and complete SGPR proposal densities.  Results from the miniature must not be
presented as inferred ARGs.

Practical limits
================

The official software requires phased, high-quality contemporary genomes.
Non-polymorphic and multiallelic sites are not used for inference, and sites
with missingness are excluded by the current implementation.  Data-poor
regions can cause numerical underflow; the authors recommend analyzing long
chromosomes in segments and merging output.  A single ARG is not a Bayesian
summary: users should inspect convergence diagnostics and retain multiple
thinned samples.
