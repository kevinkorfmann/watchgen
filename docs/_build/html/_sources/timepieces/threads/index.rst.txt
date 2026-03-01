.. _threads_timepiece:

====================================
Timepiece VII: Threads
====================================

   *Threading Instructions for Ancestral Recombination Graphs*

.. epigraph::

   "Threads takes as input a set of phased genotypes and outputs a set of
   threading instructions for each sample."

   -- Brandt, Chiang, Guo *et al.* (2024)

The Mechanism at a Glance
==========================

Threads is a **deterministic** method for inferring Ancestral Recombination
Graphs from phased genotype data. Where SINGER (:ref:`singer_timepiece`)
samples ARGs from a posterior distribution using Bayesian MCMC, Threads finds
the single most likely threading path for each haplotype using a three-step
pipeline: pre-filter candidate matches, run a memory-efficient Viterbi
algorithm, and date the resulting segments.

.. admonition:: Primary Reference

   :cite:`threads`

The three gears of Threads:

1. **PBWT Haplotype Matching** (the pre-filter) -- Uses the positional
   Burrows-Wheeler transform to identify a small set of candidate haplotype
   matches for each sample, reducing the search space from :math:`O(N)` to
   :math:`O(L)` candidates per sample (:math:`L \ll N`).

2. **Memory-Efficient Viterbi** (the inference engine) -- A branch-and-bound
   implementation of the Viterbi algorithm under the Li-Stephens model that
   finds the optimal threading path in :math:`O(NM)` time and :math:`O(N)`
   average memory, avoiding the classical :math:`O(NM)` memory requirement.

3. **Segment Dating** (the calibration) -- Assigns coalescence times to each
   Viterbi segment using likelihood-based and Bayesian estimators that model
   segments as pairwise IBD regions.

These gears connect in a simple linear pipeline:

.. code-block:: text

   Phased genotypes (N haplotypes, M sites)
                    |
                    v
   +-------------------------------+
   |   PBWT HAPLOTYPE MATCHING     |
   |                               |
   |  Chunk genome into 0.5 cM     |
   |  segments, sort with PBWT,    |
   |  query L-neighbourhood,       |
   |  filter by match count        |
   |                               |
   |  -> L candidates per sample   |
   +-------------------------------+
                    |
                    v
   +-------------------------------+
   |   MEMORY-EFFICIENT VITERBI    |
   |                               |
   |  For each sample n = 1..N:    |
   |    Run Li-Stephens Viterbi    |
   |    on L x M candidate panel   |
   |    using branch-and-bound     |
   |                               |
   |  -> Threading targets per     |
   |     sample at each site       |
   +-------------------------------+
                    |
                    v
   +-------------------------------+
   |   SEGMENT DATING              |
   |                               |
   |  For each Viterbi segment:    |
   |    Model as IBD region        |
   |    Estimate age from length   |
   |    + mutations + demography   |
   |                               |
   |  -> Coalescence times         |
   +-------------------------------+
                    |
                    v
         Threading instructions
         (= the inferred ARG)

.. admonition:: Prerequisites for this Timepiece

   Threads builds on several earlier concepts and Timepieces:

   - :ref:`Li & Stephens HMM <lshmm_timepiece>` -- the copying model that
     Threads optimizes with its memory-efficient Viterbi
   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times and rates
     used in the dating step
   - :ref:`The SMC <smc>` -- the sequentially Markov coalescent model underlying
     the segment length distribution

   If you have worked through the Li & Stephens Timepiece, you already have
   the main conceptual foundation. Threads extends that foundation to
   biobank-scale data through algorithmic innovation rather than model
   complexity.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   pbwt_matching
   viterbi
   dating
   demo
