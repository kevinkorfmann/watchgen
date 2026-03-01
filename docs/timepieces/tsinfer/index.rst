.. _tsinfer_timepiece:

====================================
Timepiece VI: tsinfer
====================================

   *Tree Sequence Inference from Genetic Variation Data*

The Mechanism at a Glance
==========================

**tsinfer** infers a tree sequence from observed genetic variation data. Unlike
MCMC-based methods (such as SINGER and ARGweaver), tsinfer is a *deterministic
algorithm* that scales to biobank-sized datasets -- hundreds of thousands of samples
and millions of sites -- in hours rather than weeks.

The core idea is breathtakingly simple: every sample's genome is a **mosaic** of
ancestral haplotypes, glued together by recombination. If we can figure out *what*
those ancestral pieces are and *how* they were assembled, we've reconstructed the
genealogy.

If SINGER and ARGweaver are precision mechanical watches -- statistically optimal but
complex and slow -- tsinfer is a quartz movement: simpler, faster, and designed for
scale. What it sacrifices in statistical sophistication (no Bayesian posterior, no
uncertainty quantification), it gains in raw throughput. And like a good quartz
movement, it's remarkably accurate for most practical purposes.

.. admonition:: Primary Reference

   :cite:`tsinfer`

The four gears of tsinfer:

1. **Ancestor Generation** (the escapement) -- Infer putative ancestral haplotypes from
   the patterns of derived alleles in the data. Older ancestors carry
   higher-frequency derived alleles. This is where the biological signal is first
   extracted.

2. **The Copying Model** (the gear train) -- A Li & Stephens HMM engine (from
   :ref:`Timepiece III <lshmm_timepiece>`) that finds the best way to express one
   haplotype as a mosaic of others. This is the workhorse shared by both the ancestor
   matching and sample matching phases.

3. **Ancestor Matching** (the first assembly) -- Match each ancestor against older
   ancestors using the copying model, building a tree sequence of ancestors from the
   root down. Like assembling the base caliber of a movement.

4. **Sample Matching** (the final assembly) -- Thread each sample through the ancestor
   tree using the same copying model, then post-process to produce the final tree
   sequence. Like fitting the dial and hands onto the finished movement.

These gears mesh together into a three-phase pipeline:

.. code-block:: text

   Variant data D (n samples x m sites)
               |
               v
   +----------------------------+
   |   PHASE 1: Generate        |
   |   Ancestral Haplotypes     |
   |                            |
   |   For each frequency tier: |
   |     Build consensus from   |
   |     samples carrying the   |
   |     derived allele         |
   +----------------------------+
               |
               | A putative ancestors
               v
   +----------------------------+
   |   PHASE 2: Match Ancestors |
   |                            |
   |   For each ancestor        |
   |   (oldest first):          |
   |     Express it as a mosaic |
   |     of older ancestors     |
   |     (Li & Stephens HMM)    |
   |                            |
   |   -> Build ancestor tree   |
   +----------------------------+
               |
               | Ancestor tree sequence
               v
   +----------------------------+
   |   PHASE 3: Match Samples   |
   |                            |
   |   For each sample:         |
   |     Express it as a mosaic |
   |     of ancestors           |
   |     (Li & Stephens HMM)    |
   |                            |
   |   -> Post-processing:      |
   |     - Parsimony mutations  |
   |     - Simplification       |
   +----------------------------+
               |
               v
       Final tree sequence T
       (tskit TreeSequence)

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- the biological framework
   - :ref:`Ancestral Recombination Graphs <args>` -- tree sequences and marginal trees
   - :ref:`Hidden Markov Models <hmms>` -- the forward algorithm and the
     Li-Stephens :math:`O(K)` trick
   - :ref:`Li & Stephens HMM <lshmm_timepiece>` -- the copying model (Timepiece III)

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   ancestor_generation
   copying_model
   ancestor_matching
   sample_matching
   demo

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you'll have built a complete tree sequence
inference engine from scratch -- and you'll understand every gear that makes it tick.
