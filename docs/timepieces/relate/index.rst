.. _relate_timepiece:

=======================
Timepiece XVII: Relate
=======================

   *Scalable local-genealogy estimation from phased haplotypes*

Relate infers a sequence of rooted local trees, maps mutations to those trees,
and estimates coalescence times under a coalescent prior. Its central scaling
choice is to infer topology first and then condition on those topologies while
dating them. The stages are connected rather than statistically independent:
mutation mapping decides when a new topology is built, equivalent branches share
dating information, and an estimated population history can be fed back into the
branch-length sampler :cite:`relate`.

This Timepiece separates the production program from the accompanying teaching
module. Production Relate implements windowed chromosome painting, robust tree
construction, approximate and multi-branch mutation mapping, association of
branches across neighbouring trees, event-order and interval MCMC moves, and
iterative coalescence-rate estimation. ``watchgen.mini_relate`` demonstrates only
small, source-guided kernels. It does not read ``.haps`` files, build a genomic tree
sequence, or reproduce Relate's posterior; it is not a reimplementation of Relate.

Source and review target
========================

The primary methodological source is the program paper and its Supplementary Note
:cite:`relate`. The software source is the upstream Myers Group repository,
``https://github.com/MyersGroup/relate``. It was audited at commit
``b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74``. In particular, the review follows
``fast_painting.cpp``, ``tree_builder.cpp``, ``anc_builder.cpp``, and
``branch_length_estimator.cpp`` rather than inferring behavior from command names.

Production pipeline
===================

For each target haplotype, Relate uses a modified Li--Stephens HMM to obtain local
copying scores against the other haplotypes. It transforms these into a
non-symmetric distance matrix and applies a directional hierarchical clustering
rule. Moving from the 5' end, it retains the current tree until a mutation cannot
be mapped uniquely or appears to require an ancestral/derived flip. It then
associates equivalent branches between neighbouring trees, pools their mutation
information, and samples coalescence times. A separate iterative procedure can
alternate pairwise coalescence-rate estimation with branch-length re-estimation.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   asymmetric_painting
   tree_building
   branch_lengths
   population_size
   demo
