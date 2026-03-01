.. _relate_timepiece:

====================================
Timepiece XVII: Relate
====================================

   *Genome-Wide Genealogy Estimation for Thousands of Samples*

The Mechanism at a Glance
==========================

**Relate** estimates genome-wide genealogies -- local trees along the chromosome
-- from phased haplotype data. It occupies a sweet spot in the landscape of ARG
inference: far more scalable than full Bayesian methods (ARGweaver, SINGER), yet
more statistically principled than purely deterministic ones (tsinfer). Where
tsinfer is a quartz movement and SINGER is a grand complication, Relate is a
**robust automatic movement**: it uses heuristic topology inference for speed, then
applies rigorous MCMC sampling for branch-length accuracy.

The key insight is a clean **separation of concerns**: topology and timing are
inferred in two independent phases. Phase 1 identifies *who coalesces with whom*
using a modified Li & Stephens painting and a bespoke tree-building heuristic.
Phase 2 determines *when* those coalescence events happened, using Metropolis-
Hastings MCMC under a coalescent prior. By decoupling these problems, Relate
avoids the combinatorial explosion that plagues joint inference, achieving
:math:`O(N^2 L)` scaling (quadratic in sample count, linear in genome length)
and handling tens of thousands of haplotypes.

.. admonition:: Primary Reference

   :cite:`relate`

The five gears of Relate:

1. **Asymmetric Painting** (the oscillator) -- A modified Li & Stephens HMM that
   incorporates ancestral vs. derived allele status. The forward-backward
   algorithm produces posterior copying probabilities, which are converted into a
   position-specific **asymmetric distance matrix**. The asymmetry encodes the
   direction of mutation: how many derived alleles *i* carries that *j* does not.

2. **Tree Building** (the gear train) -- A bespoke agglomerative (bottom-up)
   clustering algorithm that operates on the asymmetric distance matrix. Unlike
   standard hierarchical clustering (which requires symmetric distances), this
   algorithm exploits the directional information to correctly identify which
   pairs of lineages coalesce first. It produces a rooted binary tree at each
   genomic position.

3. **Mutation Mapping** (the dial) -- Under the infinite-sites model, each
   derived allele maps to a unique branch of the local tree -- the branch that
   separates carriers from non-carriers. This deterministic step connects the
   observed data to the tree topology.

4. **Branch Length MCMC** (the escapement) -- With topologies fixed and
   mutations mapped, a Metropolis-Hastings sampler estimates coalescence times.
   The likelihood is Poisson (mutations accumulate proportionally to branch
   length), the prior is coalescent (exponential waiting times between
   coalescence events). The MCMC explores the posterior over node times,
   producing either point estimates or full posterior samples.

5. **Population Size Estimation** (the regulator) -- An EM algorithm that
   alternates between (E-step) sampling branch lengths given current population
   sizes, and (M-step) updating population sizes given current branch lengths.
   The population size is modeled as a piecewise-constant function of time.

These gears mesh together into a two-phase pipeline:

.. code-block:: text

   Phased haplotype data H (N haplotypes x L sites)
                  |
                  v
   +---------------------------------+
   |   PHASE 1: Topology Inference   |
   |                                 |
   |   For each focal SNP:           |
   |     1. Li & Stephens painting   |
   |        (forward-backward)       |
   |     2. Asymmetric distance      |
   |        matrix d(i,j)            |
   |     3. Agglomerative tree       |
   |        building                 |
   +---------------------------------+
                  |
                  | Local tree topologies (one per SNP interval)
                  v
   +---------------------------------+
   |   Mutation Mapping              |
   |   (infinite sites: each derived |
   |    allele -> unique branch)     |
   +---------------------------------+
                  |
                  v
   +---------------------------------+
   |   PHASE 2: Branch Lengths       |
   |                                 |
   |   Metropolis-Hastings MCMC:     |
   |     - Poisson mutation          |
   |       likelihood                |
   |     - Coalescent prior          |
   |     - Propose new node times    |
   |     - Accept/reject             |
   +---------------------------------+
                  |
                  v
   +---------------------------------+
   |   Population Size Estimation    |
   |   (optional, iterative EM)      |
   |                                 |
   |   E: sample branch lengths      |
   |   M: update N_e(t)              |
   +---------------------------------+
                  |
                  v
       .anc + .mut files
       (or tree sequence via tskit)


Where tsinfer and SINGER End and Relate Begins
===============================================

Relate occupies a distinct niche:

- **tsinfer** (Timepiece VI) is purely deterministic: it builds topology via
  Viterbi (no posterior), assigns frequency-proxy times, and scales to millions
  of samples. No uncertainty quantification; no MCMC. Paired with tsdate for
  dating.

- **SINGER** (Timepiece VII) is fully Bayesian: it jointly samples topology and
  branch lengths from the posterior, producing the most accurate ARGs -- but
  only for hundreds of samples.

- **Relate** splits the difference. Phase 1 is heuristic (like tsinfer), but
  uses forward-backward (not Viterbi) to extract richer information. Phase 2 is
  Bayesian (like SINGER), but only over branch lengths, not topology. The
  result: accurate dated genealogies for thousands of samples, with posterior
  uncertainty on coalescence times.

In the watch metaphor: tsinfer is the quartz movement (fast, no springs),
SINGER is the tourbillon (precise, complex, slow), and Relate is the **robust
automatic** -- self-winding, accurate, and built for daily wear.

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times and the
     exponential waiting time distribution
   - :ref:`Ancestral Recombination Graphs <args>` -- local trees and how they
     change along the genome
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm
   - :ref:`Li & Stephens HMM <lshmm_timepiece>` -- the copying model and the
     :math:`O(K)` trick (Timepiece III)
   - :ref:`Markov Chain Monte Carlo <mcmc>` -- Metropolis-Hastings sampling

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

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you'll have built a complete genealogy
estimation engine from scratch -- and you'll understand every gear that makes
it tick.
