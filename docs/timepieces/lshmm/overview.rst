.. _lshmm_overview:

==================================
Overview of the Li & Stephens HMM
==================================

   *Before assembling the watch, lay out every part and understand what it does.*

The Li & Stephens Hidden Markov Model is a versatile gear that appears in many
movements across population genetics. Just as a single escapement mechanism can
drive a simple desk clock or a grand complication, the LS-HMM underlies tools
as diverse as genotype imputation, haplotype phasing, ancestry painting, and
ancestral recombination graph inference. In this chapter we lay out the
blueprint -- what the model does, why it matters, and how the remaining chapters
fit together.

.. note::

   **Prerequisites.** This chapter assumes you have read the
   :ref:`HMM chapter <hmms>`, where we introduced hidden states, observations,
   transition matrices, emission probabilities, and the forward, backward, and
   Viterbi algorithms in their general form. We also assume familiarity with the
   basics of **coalescent theory** -- the idea that a sample of sequences shares
   a common ancestor, and that recombination causes different genomic positions
   to have different genealogical trees. If these concepts are new, a good
   starting point is Wakeley's *Introduction to Coalescent Theory* or the
   coalescent chapter of this book (if available).


What Does the Li & Stephens HMM Do?
=====================================

Given a **reference panel** of :math:`n` haplotypes and a **query** haplotype,
the Li & Stephens (LS) model finds the most likely way the query was assembled
as a **mosaic** of the reference haplotypes.

.. math::

   \text{Input: } H = \{h_1, h_2, \ldots, h_n\} \quad \text{(reference haplotypes)}

.. math::

   \text{Input: } s = (s_1, s_2, \ldots, s_m) \quad \text{(query haplotype, } m \text{ sites)}

.. math::

   \text{Output: } Z^* = (Z_1^*, Z_2^*, \ldots, Z_m^*) \quad \text{(copying path through the panel)}

Each :math:`Z_\ell^*` tells you which reference haplotype was being "copied" at
site :math:`\ell`. Where :math:`Z` changes value, a recombination breakpoint
occurred. Where the query allele differs from the copied haplotype's allele,
a mutation occurred.

.. admonition:: Terminology: Haplotype vs. Genotype

   A **haplotype** is the sequence of alleles on a single chromosome. A
   **genotype** is the combination of alleles across both chromosomes at a
   given site (for diploid organisms). In the haploid LS model, we work
   directly with haplotypes. In the diploid extension (Chapter 4), we work
   with genotypes -- allele dosages of 0, 1, or 2.

From this model you can compute:

- **The most likely copying path** (Viterbi algorithm) -- used in haplotype
  imputation and phasing
- **Posterior copying probabilities** (forward-backward algorithm) -- the
  probability that any given reference haplotype is the copying source at any
  given site
- **The data likelihood** -- how well the model explains the query, useful for
  model comparison and parameter estimation
- **Local ancestry** -- which population the query haplotype was copied from
  at each position, when reference haplotypes come from labeled populations

Think of the Viterbi algorithm as finding the most likely gear sequence -- the
single best explanation for how the mechanism ticked through each position along
the genome. The forward-backward algorithm, by contrast, gives you a
probability distribution over all possible gear configurations at each position.

With this high-level picture in mind, let us turn to the biological motivation
that makes this model so natural.


The Mosaic Model
=================

The biological intuition behind the LS model comes from the **coalescent**.
When you trace the ancestry of a set of haplotypes backwards in time, each
genomic position has a genealogical tree. Due to recombination, the tree changes
along the genome. At each site, the query haplotype is most closely related to
some reference haplotype -- and that closest relative changes at recombination
breakpoints.

.. admonition:: Coalescent Theory in Brief

   The **coalescent** is a stochastic model of how a sample of gene copies
   traces back to a common ancestor. Two key processes shape these ancestral
   histories:

   - **Coalescence**: two lineages merge into a common ancestor (looking
     backwards in time). The rate depends on the effective population size
     :math:`N_e`.
   - **Recombination**: a single ancestral chromosome is assembled from two
     parental chromosomes, so the left and right portions of the genome may
     have different genealogical trees.

   The result is an **ancestral recombination graph (ARG)** -- a collection
   of local trees that change along the genome at recombination breakpoints.
   The LS model is a practical approximation of this ARG structure: it
   captures the idea that the query's closest relative in the reference panel
   changes at recombination events, without explicitly building the full ARG.

Li and Stephens (2003) turned this into a practical model:

   **A new haplotype is an imperfect mosaic of existing haplotypes.**

"Imperfect" because mutations introduce differences between the query and its
copying source. "Mosaic" because recombination switches the source. This is the
template mechanism at the heart of the model -- the query is stamped out from
a sequence of templates drawn from the reference panel, with occasional
imperfections (mutations) in the stamping process.

.. code-block:: text

   Reference panel (n=4 haplotypes, m=12 sites):

   h_1: 0 0 1 0 0 1 1 0 0 1 0 0
   h_2: 0 1 0 0 1 1 0 0 1 0 0 1
   h_3: 1 0 0 1 0 0 1 1 0 0 1 0
   h_4: 0 0 1 0 1 0 0 1 1 0 0 0

   Query haplotype:
   s:   0 0 1 0 1 1 0 0 1 0 0 1
              ^--- mutation        ^---------- mutation?
        |--------|---------|------|----------|
        copy h_1   copy h_4  h_2    copy h_2
              recomb    recomb         (match)

The query copies from :math:`h_1` for the first few sites, then a recombination
switches the source to :math:`h_4`, then to :math:`h_2`. At one site, the query
differs from its source -- that's a mutation.

Now that we have the biological picture, let us formalize it as an HMM.


The HMM Formulation
=====================

This copying process maps directly to a Hidden Markov Model. If you have
worked through the :ref:`HMM chapter <hmms>`, the following table will feel
familiar -- it is the same framework, with the states and observations filled in
by the biology:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - HMM Component
     - LS Model Meaning
     - Notation
   * - **Hidden states**
     - Which reference haplotype is being copied
     - :math:`Z_\ell \in \{1, 2, \ldots, n\}`
   * - **Observations**
     - The query allele at each site
     - :math:`s_\ell \in \{0, 1, \ldots\}`
   * - **Initial distribution**
     - Uniform: any haplotype equally likely
     - :math:`\pi_j = 1/n`
   * - **Transitions**
     - Recombination switches the copying source
     - :math:`A_{ij}` (derived in :ref:`copying_model`)
   * - **Emissions**
     - Mutation changes the allele relative to the source
     - :math:`e_j(s_\ell)` (derived in :ref:`copying_model`)

.. admonition:: Probability Aside: Why Uniform Initial Distribution?

   Under the coalescent, exchangeability of lineages means that, before
   observing any data, the query is equally likely to be most closely related
   to any reference haplotype. This justifies :math:`\pi_j = 1/n`. A
   non-uniform prior could incorporate population structure (e.g., giving
   higher weight to reference haplotypes from the same population as the
   query), but the uniform prior is standard and works well in practice.

The key parameters are:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Symbol
     - Meaning
   * - Recombination probability
     - :math:`r`
     - Probability of switching copying source between adjacent sites
   * - Mutation probability
     - :math:`\mu`
     - Probability that the query allele differs from the copied allele
   * - Number of ref. haplotypes
     - :math:`n`
     - Size of the reference panel
   * - Number of sites
     - :math:`m`
     - Number of genomic positions

.. admonition:: Terminology: Site vs. Variant vs. Locus

   In this book, a **site** (or **locus**) refers to a specific genomic
   position included in the analysis. In practice, many implementations
   consider only **variant sites** (positions where at least one individual
   in the reference panel differs from the others), but the LS model is
   defined over any set of positions. The recombination probability
   :math:`r_\ell` between adjacent sites depends on their physical or genetic
   distance, so the spacing of sites matters.

With the HMM formulation in hand, we can now appreciate why the LS model is so
widely used.


Why the LS Model Matters
=========================

The LS model is not just an academic exercise. It's a foundational piece of
machinery in modern genetics -- a versatile gear that appears in many
movements:

1. **Haplotype imputation** (e.g., IMPUTE, Beagle, Minimac): Fill in missing
   genotypes by finding which reference haplotypes the query is copying from.
   The posterior copying probabilities weight the "votes" of different reference
   alleles.

2. **Phasing** (e.g., SHAPEIT, Eagle): Resolve which alleles are on the same
   chromosome. The diploid LS model finds the most likely pair of copying paths.

3. **Ancestry painting** (e.g., RFMix, MOSAIC): When reference haplotypes come
   from labeled populations, the copying path reveals local ancestry.

4. **ARG inference** (e.g., tsinfer, SINGER): The LS model forms the backbone
   of threading algorithms that build ancestral recombination graphs.

5. **Selection scans**: Unusually long copying tracts suggest reduced
   recombination or recent shared ancestry -- potential signatures of selection.

.. admonition:: Probability Aside: Data Likelihood and Model Comparison

   The forward algorithm (introduced in the :ref:`HMM chapter <hmms>` and
   detailed in :ref:`haploid_algorithms`) produces the **data likelihood**
   :math:`P(X_1, \ldots, X_m)` as a byproduct. This quantity tells you how
   well the model explains the observed query, given the reference panel and
   model parameters. It is essential for:

   - **Parameter estimation**: finding the :math:`r` and :math:`\mu` that
     maximize the likelihood (or using EM to iterate).
   - **Model comparison**: comparing different reference panels or different
     model configurations via their likelihoods.
   - **Composite likelihoods**: products of per-individual likelihoods are used
     in population-level inference.

Having seen why the model matters, let us now map out how the remaining chapters
build it up piece by piece.


The Pieces of the Mechanism
=============================

Here's how the chapters connect. Think of this as an exploded diagram of a
watch movement -- each box is a sub-assembly, and the arrows show how they fit
together:

.. code-block:: text

   Reference panel H + Query s
           |
           v
   +---------------------------+
   |   THE COPYING MODEL       |
   |   (template mechanism)    |
   |                           |
   |  Transition: P(switch     |
   |    copying source)        |
   |  Emission: P(query allele |
   |    | copying source)      |
   |  Mutation rate estimation  |
   |                           |
   |  -> The O(K) trick        |
   +---------------------------+
           |
           v
   +---------------------------+
   |   HAPLOID ALGORITHMS      |
   |   (the gear train)        |
   |                           |
   |  Forward algorithm        |
   |    -> P(data, state)      |
   |  Backward algorithm       |
   |    -> posterior decoding   |
   |  Viterbi algorithm        |
   |    -> best path (most     |
   |       likely gear         |
   |       sequence)           |
   |  Scaling + memory tricks  |
   +---------------------------+
           |
           v
   +---------------------------+
   |   DIPLOID EXTENSION       |
   |   (two watches ticking    |
   |    together)              |
   |                           |
   |  State space: n x n       |
   |  Four transition types:   |
   |    no switch, single,     |
   |    double                 |
   |  Genotype emissions       |
   |  Forward-backward         |
   |  Viterbi                  |
   +---------------------------+
           |
           v
       Copying path / posteriors / likelihood

Each chapter builds on the previous one. The copying model defines the
individual gears (transition and emission probabilities). The haploid algorithms
assemble those gears into a working movement. The diploid extension doubles the
mechanism to handle the reality that most organisms carry two copies of each
chromosome.


Ready to Build
===============

You now have the high-level blueprint. In the following chapters, we'll build
each gear from scratch:

1. :ref:`copying_model` -- The transition and emission probabilities (the
   template mechanism that defines how haplotypes copy from each other)
2. :ref:`haploid_algorithms` -- Forward, backward, and Viterbi for haploid data
   (the gear train that turns the model into answers)
3. :ref:`diploid` -- Extending to diploid genotypes (two watches ticking
   together)

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you will own every gear in this mechanism
completely -- able to inspect it, modify it, or slot it into a larger
complication.

Let's start with the core of the model: the copying process.
