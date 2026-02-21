.. _singer_timepiece:

====================================
Timepiece VI: SINGER
====================================

   *Sampling and Inference of Genealogies with Recombination*

.. epigraph::

   "Robust and accurate Bayesian inference of genome-wide genealogies for large samples"

   -- Deng, Nielsen, and Song (2024)

The Mechanism at a Glance
==========================

SINGER is a Bayesian method for sampling **Ancestral Recombination Graphs (ARGs)**
from their posterior distribution, given observed genetic variation data. It uses
an iterative threading algorithm: one haplotype at a time is "threaded" onto a
growing partial ARG using Hidden Markov Models.

If PSMC is a two-hand watch (two lineages, one coalescence time), then SINGER is
a grand complication -- a mechanism of extraordinary complexity that tracks the
complete genealogical history of many individuals simultaneously. Where PSMC reads
population size from two haplotypes, SINGER reconstructs the full ancestral
recombination graph: every coalescence event, every recombination, every marginal
tree, for as many samples as you can provide.

The four gears of SINGER:

1. **Branch Sampling** (the first gear train) -- An HMM that determines *which branch*
   each new lineage joins at each genomic position. This solves the topological question:
   where in the existing tree does the new haplotype attach?

2. **Time Sampling** (the second gear train) -- A second HMM that determines *when*
   (at what time) the lineage joins, conditioned on the branch choice. This uses the
   PSMC transition density from :ref:`Timepiece I <psmc_timepiece>` -- the simpler
   mechanism reappears as a component in the more complex one.

3. **ARG Rescaling** (the regulator) -- A post-processing step that adjusts coalescence
   times to better match the mutation clock, like a watchmaker calibrating the beat
   rate against a reference frequency.

4. **SGPR** (the winding mechanism) -- Sub-Graph Pruning and Re-grafting: the MCMC
   update mechanism that explores the space of ARGs by removing and re-threading
   subsets of the genealogy.

These gears mesh together into a complete MCMC sampler:

.. code-block:: text

   Initialize ARG by threading haplotypes 1, 2, ..., n
                          |
                          v
            +---> Pick a sub-graph to prune (SGPR)
            |              |
            |              v
            |     Re-thread using Branch + Time Sampling
            |              |
            |              v
            |     Accept/reject (Metropolis-Hastings)
            |              |
            |              v
            |     Rescale the ARG
            |              |
            +--------------+
                   (repeat)

.. admonition:: Prerequisites for this Timepiece

   SINGER draws on all the prerequisite chapters and builds on earlier Timepieces:

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times and rates
   - :ref:`Ancestral Recombination Graphs <args>` -- the data structure SINGER infers
   - :ref:`Hidden Markov Models <hmms>` -- forward algorithm, stochastic traceback,
     Li-Stephens trick
   - :ref:`The SMC <smc>` -- the Markov approximation enabling HMM inference
   - :ref:`PSMC <psmc_timepiece>` -- the transition density reused in time sampling

   If you've built those earlier mechanisms, you have every tool you need. SINGER
   is where all the gears finally mesh together into the most complex Timepiece in
   our collection.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   branch_sampling
   time_sampling
   arg_rescaling
   sgpr
