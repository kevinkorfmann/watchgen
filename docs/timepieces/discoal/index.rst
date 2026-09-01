.. _discoal_timepiece:

===========================
Timepiece XVIII: discoal
===========================

   *Selective-sweep simulation with a discrete ancestral recombination graph*

``discoal`` is a C simulator for coalescent histories with crossover, gene
conversion, population structure, demographic events, and selective sweeps. Its
name contracts *discrete* and *coalescent*: ancestry is tracked across a discrete
number of sites. Selection is represented by first generating an allele-frequency
trajectory and then running a structured coalescent conditional on that trajectory
:cite:`discoal,braverman_1995,coop_griffiths_2004`.

This Timepiece separates three levels that the earlier draft conflated. The
production program constructs a chromosome-scale ancestral recombination graph.
The mathematical sweep kernel assigns ancestral material to beneficial and
wild-type backgrounds. The accompanying ``watchgen.mini_discoal`` module implements
only a single-locus version of that kernel. It is useful for inspecting rates and
limiting cases, but it is not a chromosome-scale ARG simulator and must not be used
as a replacement for ``discoal``.

Source and review target
========================

The primary software reference for this review is the upstream repository at
``https://github.com/kern-lab/discoal``. The current source was audited at commit
``7d0955f4107053c135d2086790b0426457147a8e``; executable parity was also checked
against the paper-era commit ``82971bf``. In particular, the trajectory rules come
from ``alleleTraj.c`` and the sweep event loop from ``discoalFunctions.c``.
The primary methodological references are the original program paper
:cite:`discoal`, the hitchhiking coalescent of Braverman and colleagues
:cite:`braverman_1995`, and the conditional diffusion construction of Coop and
Griffiths :cite:`coop_griffiths_2004`.

What the mini implementation covers
===================================

The function ``mini_discoal.deterministic_trajectory`` reproduces the deterministic
curve and discretization used by the C source. The function
``mini_discoal.stochastic_trajectory`` implements the two-point conditional jump
law, including the neutral phase used for standing variation. The function
``mini_discoal.structured_coalescent_sweep`` then applies the two-background event
rates to one linked neutral locus. Finally,
``mini_discoal.simulate_linked_locus_genealogy`` adds the neutral phases before and
after the sweep.

The mini does not split ancestral segments, place mutations on an ARG, emit
``ms``-format haplotypes, model gene conversion, or reproduce discoal's multipopulation
event scheduler. Those are properties of the original software, not of this teaching
kernel.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   allele_trajectory
   structured_coalescent
   sweep_types
   msprime_comparison
   demo
