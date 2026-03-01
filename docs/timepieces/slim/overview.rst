.. _slim_overview:

==================
Overview of SLiM
==================

   *Before firing up the forge, understand what it can build.*

What Does SLiM Do?
===================

**Input:** A population model -- population size :math:`N`, genome length
:math:`L`, mutation rate :math:`\mu`, recombination rate :math:`r`, and a
**fitness model** that maps genotypes to reproductive success.

**Output:** A population of :math:`N` individuals, each carrying two
haplosomes (chromosome copies) with mutations accumulated over :math:`T`
generations of evolution. Optionally, SLiM records the complete **tree
sequence** -- the full genealogical history of every base pair in every
individual -- which can be analyzed with tskit.

The key difference from msprime:

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Property
     - msprime (Timepiece IV)
     - SLiM
   * - Direction
     - Backward in time (coalescent)
     - Forward in time (Wright-Fisher)
   * - Selection
     - Neutral only (no fitness)
     - Full selection models
   * - Speed
     - Very fast (:math:`O(n)` in sample size)
     - Slower (:math:`O(N \cdot T)` in population size and generations)
   * - Output
     - Tree sequence
     - Population state (+ optional tree sequence)
   * - Best for
     - Neutral demography, ground truth
     - Selection, complex ecology, spatial models

SLiM is necessary whenever you need **natural selection**. The coalescent
does not model selection well -- it assumes all lineages are exchangeable,
which breaks down when some alleles have higher fitness than others. SLiM
tracks every individual, every mutation, and every fitness effect, so
selection falls out naturally from the simulation.


Why Forward Simulation?
========================

The coalescent is elegant because it only tracks the :math:`n` sampled
lineages, ignoring the vast majority of the population. But this elegance
comes at a cost: it assumes **neutrality**. When selection acts, the
genealogy depends on which alleles individuals carry, which depends on the
genealogy -- a chicken-and-egg problem that the backward-time framework
cannot easily resolve.

Forward simulation breaks this circularity by brute force: simulate every
individual in every generation. Selection is trivial in the forward
direction -- individuals with higher fitness leave more offspring. The
price is computational: we must simulate all :math:`N` individuals for
all :math:`T` generations, even though we may only care about a small
sample at the end.

.. admonition:: When to use SLiM vs. msprime

   **Use msprime** when your model is neutral (no selection), or when
   selection is weak enough to ignore. msprime is orders of magnitude
   faster for neutral simulations.

   **Use SLiM** when selection matters: selective sweeps, background
   selection, balancing selection, frequency-dependent selection, local
   adaptation, or anything where fitness varies among individuals.

   **Use both together**: simulate neutral ancestry with msprime, then
   "replay" it through SLiM to add selection. Or use SLiM's tree-sequence
   recording to get msprime-compatible output. The tools are designed to
   interoperate.


Terminology
============

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - Definition
   * - Haplosome
     - One copy of the chromosome (SLiM's term for what is often called a
       "haplotype" or "gamete"). Each diploid individual carries two
       haplosomes.
   * - Mutation type
     - A class of mutations sharing a distribution of fitness effects (DFE).
       For example, "neutral mutations" (:math:`s = 0`) and "deleterious
       mutations" (:math:`s \sim \text{Gamma}`) might be two different types.
   * - Selection coefficient :math:`s`
     - The fitness effect of a mutation. :math:`s > 0` is beneficial,
       :math:`s < 0` is deleterious, :math:`s = 0` is neutral.
   * - Dominance coefficient :math:`h`
     - How the mutation's effect manifests in heterozygotes. :math:`h = 0.5`
       is codominant (additive), :math:`h = 0` is fully recessive, :math:`h
       = 1` is fully dominant.
   * - Fitness :math:`w`
     - An individual's total reproductive fitness: the product of the
       effects of all mutations it carries. Determines the probability of
       being chosen as a parent.
   * - DFE
     - Distribution of Fitness Effects. The probability distribution from
       which selection coefficients are drawn for new mutations.
   * - Tick
     - One generation in the Wright-Fisher model.
   * - Tree-sequence recording
     - SLiM's ability to record the complete genealogical history of the
       simulation, producing a tskit-compatible tree sequence without
       storing every intermediate state.


Parameters
===========

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Symbol
     - Typical value
     - Meaning
   * - :math:`N`
     - 1,000 -- 100,000
     - Diploid population size
   * - :math:`L`
     - :math:`10^5` -- :math:`10^8` bp
     - Genome (chromosome) length
   * - :math:`\mu`
     - :math:`10^{-8}` -- :math:`10^{-7}`
     - Per-bp, per-generation mutation rate
   * - :math:`r`
     - :math:`10^{-8}` -- :math:`10^{-7}`
     - Per-bp, per-generation recombination rate
   * - :math:`s`
     - :math:`-0.1` -- :math:`0.1`
     - Selection coefficient (per mutation)
   * - :math:`h`
     - 0 -- 1
     - Dominance coefficient
   * - :math:`T`
     - :math:`10 N` -- :math:`20 N`
     - Number of generations to simulate (burn-in + observation)


The Flow in Detail
===================

.. code-block:: text

   INITIALIZATION
   ==============
   Create N individuals, each with 2 empty haplosomes
   Burn in for ~10N generations to reach mutation-drift equilibrium
        |
        v
   FOR EACH GENERATION (tick):
   ===========================
        |
        v
   1. RECALCULATE FITNESS
      For each individual i:
        w_i = 1.0
        For each mutation m on haplosome 1:
          If m also on haplosome 2 (homozygous):
            w_i *= (1 + s_m)              <-- full effect
          Else (heterozygous):
            w_i *= (1 + h_m * s_m)        <-- dominance-modulated
        For each mutation m on haplosome 2 only:
          w_i *= (1 + h_m * s_m)          <-- heterozygous
        |
        v
   2. GENERATE N OFFSPRING
      For each child:
        a. Draw parent 1 with P(parent=i) ~ w_i
        b. Draw parent 2 with P(parent=j) ~ w_j
        c. From parent 1: recombine haplosomes -> child haplosome 1
        d. From parent 2: recombine haplosomes -> child haplosome 2
        e. Add new mutations to child haplosome 1 (Poisson)
        f. Add new mutations to child haplosome 2 (Poisson)
        |
        v
   3. OFFSPRING REPLACE PARENTS
      The N children become the new population
      (non-overlapping generations)
        |
        v
   4. BOOKKEEPING
      Remove mutations that have fixed (frequency = 1.0)
      Remove mutations that have been lost (frequency = 0)
      (Optionally: record tree-sequence edges)
        |
        v
   Repeat from step 1


Ready to Build
===============

We have laid out the parts. The mechanism is conceptually simple: a
Wright-Fisher population with mutations, recombination, and selection.
The complexity lies in doing it efficiently -- and SLiM's source code is
a masterwork of C++ engineering -- but the *algorithm* fits on a napkin.

In the following chapters, we build each gear from scratch:

1. :ref:`slim_wright_fisher` -- The core generation cycle: parent selection,
   recombination, mutation, and fitness calculation. We implement a minimal
   Wright-Fisher simulator in Python.

2. :ref:`slim_recipes` -- Practical recipes: a selective sweep, background
   selection, and tree-sequence recording. These show the mechanism in action.

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works.

Let us start with the escapement: the Wright-Fisher cycle.
