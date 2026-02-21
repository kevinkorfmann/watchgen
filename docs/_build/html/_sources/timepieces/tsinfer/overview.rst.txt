.. _tsinfer_overview:

====================
Overview of tsinfer
====================

   *Before assembling the watch, lay out every part and understand what it does.*

If the MCMC-based methods we explored earlier -- SINGER and ARGweaver -- are
fine mechanical watches, hand-assembled with painstaking precision, then
**tsinfer is a quartz movement**: simpler, faster, designed for scale. A
quartz watch sacrifices the romance of a balance wheel for the reliability
of an oscillating crystal. tsinfer makes an analogous trade: it gives up
full posterior inference for a single best-fit genealogy, and in return
it scales to millions of samples where MCMC methods would take months.

This chapter lays out every component of the tsinfer movement before we
begin assembly. If you have not yet read the :ref:`tree sequence
fundamentals <args>`, do so now -- tsinfer's output is a tree
sequence, and understanding that data structure is essential. You will also
want familiarity with the :ref:`Li & Stephens HMM <lshmm_timepiece>`,
since tsinfer uses the Viterbi algorithm from that model as its core engine.


What Does tsinfer Do?
======================

Given a set of :math:`n` aligned DNA sequences (haplotypes) at :math:`m` variable
sites, tsinfer produces a **tree sequence**: a compact, lossless representation of
the correlated genealogies along the genome.

.. math::

   \text{Input: } \mathbf{D} \in \{0, 1\}^{n \times m} \quad \text{(variant matrix: samples} \times \text{sites)}

.. math::

   \text{Output: } \mathcal{T} = (\mathcal{N}, \mathcal{E}, \mathcal{M}) \quad \text{(tree sequence: nodes, edges, mutations)}

Each column of :math:`\mathbf{D}` is a **site**: the allelic states of all
:math:`n` samples at one genomic position. Each row is a **sample haplotype**.
By convention, ``0`` denotes the ancestral allele and ``1`` denotes the derived
allele.

.. admonition:: Confusion Buster -- Ancestral vs. Derived Alleles

   The **ancestral allele** is the allele present in the most recent common
   ancestor of the sample. It is the "original" state at a given genomic
   position. The **derived allele** is the "new" state that arose by mutation
   at some point in the past. In the variant matrix :math:`\mathbf{D}`,
   ancestral is encoded as ``0`` and derived as ``1``. Knowing which is
   which (the *polarity* of the site) is critical for tsinfer, because the
   frequency of the derived allele is used as a proxy for the mutation's age.
   Outgroup species (e.g., chimpanzee for human data) are typically used to
   determine polarity.

The output tree sequence :math:`\mathcal{T}` encodes:

- **Nodes** :math:`\mathcal{N}`: the samples, their ancestors, and a virtual
  root
- **Edges** :math:`\mathcal{E}`: parent-child relationships over genomic
  intervals
- **Mutations** :math:`\mathcal{M}`: allele changes placed on edges

From this compact structure, you can extract:

- **Local trees** at any genomic position
- **Pairwise coalescence times** between any pair of samples
- **Allele ages** (when mutations arose)
- **Ancestral recombination events** (where trees change)


The Mosaic Insight
===================

tsinfer's central idea is that every genome is a **mosaic** of ancestral
segments. Your chromosome 1 might carry a segment from an ancestor who lived
500 generations ago, then switch (via recombination) to a segment from a
different ancestor at 200 generations, then switch again.

Think of it this way: in a mechanical watch, every gear turns because another
gear drives it. In tsinfer, every haplotype exists because it was "driven"
-- copied, with occasional errors -- from ancestral haplotypes. The
algorithm's job is to figure out which ancestral gear drove which segment
of each present-day haplotype.

**Why is this useful?** If we can identify the ancestral segments and figure
out which ancestors they came from, we've effectively reconstructed the
genealogy. And there's a well-studied statistical framework for exactly this
problem: the **Li & Stephens copying model** (see the
:ref:`Li & Stephens Timepiece <lshmm_timepiece>` for the full derivation).

The copying model says: express a query haplotype as a mosaic of reference
haplotypes, allowing occasional switches (recombination) and mismatches
(mutation). The hidden state is "which reference am I copying right now?",
and the Viterbi algorithm finds the best mosaic.

tsinfer applies this idea twice:

1. **Ancestor matching**: Express each ancestor as a mosaic of *older*
   ancestors. This builds the upper (ancient) part of the tree sequence.

2. **Sample matching**: Express each sample as a mosaic of ancestors.
   This connects the samples to the ancestor tree.

With this two-pass approach, tsinfer assembles the complete movement --
first the mainplate and bridges (ancestor tree), then the hands and dial
(sample connections).


The Three Phases
=================

Let's walk through the algorithm at a high level. Each phase corresponds
to a chapter that follows, so treat this section as a roadmap.

Phase 1: Generate Ancestors
-----------------------------

The first challenge: where do the ancestral haplotypes come from? We don't
observe them directly -- only the present-day samples are in our data.

tsinfer's key insight is to use **allele frequency as a proxy for age**.
Under neutral evolution, a derived allele carried by 90% of samples is
(on average) much older than one carried by 5%. This follows from
coalescent theory: older mutations have had more time to spread through
the population.

.. admonition:: Confusion Buster -- What is Allele Frequency?

   The **allele frequency** (or **sample frequency**) of a derived allele
   is the proportion of sampled haplotypes that carry it. If 18 out of 20
   haplotypes carry allele ``1`` at a site, the derived allele frequency is
   :math:`18/20 = 0.9`. This is sometimes called the **derived allele
   frequency (DAF)**. In population genetics, frequencies are central: they
   summarize how common or rare a variant is, and under neutral models they
   carry information about the variant's age.

For each site, tsinfer:

1. Computes the **frequency** of the derived allele
2. Groups sites into **time tiers** by frequency
3. Constructs an **ancestral haplotype** for each tier by taking the
   consensus of samples that carry the derived allele

The result is a set of :math:`A` putative ancestors, each with an assigned
time (based on frequency) and an allelic state at each site within its
genomic interval. In our watch metaphor, this phase is **extracting the
template gears** -- identifying the ancestral components from which the
present-day mechanism was assembled.

Phase 2: Match Ancestors
--------------------------

Now we have ancestors. The next step is to figure out how they relate to
each other -- **assembling the movement**.

tsinfer processes ancestors **from oldest to youngest**. For each ancestor,
it runs the Li & Stephens copying model against all *older* ancestors that
have already been placed in the tree. The Viterbi path tells us: this
ancestor is a mosaic of these older ancestors, with recombination breakpoints
here and here.

Each segment of the mosaic becomes an **edge** in the tree sequence (a
parent-child relationship over a genomic interval). By the time all ancestors
are processed, we have an **ancestor tree sequence** -- a partial genealogy
that doesn't yet include the samples.

If you have not yet read the :ref:`copying model chapter
<tsinfer_copying_model>`, you may wish to skim it now for context on how
the Viterbi algorithm works. The ancestor matching chapter
(:ref:`tsinfer_ancestor_matching`) will build on both Gear 1 and Gear 2.

Phase 3: Match Samples
------------------------

The final phase threads the actual samples through the ancestor tree.
For each sample, tsinfer runs the same Li & Stephens engine against the
complete set of ancestors. The Viterbi path tells us which ancestor each
sample copies at each position.

After matching, tsinfer applies **post-processing**:

- Places mutations at **non-inference sites** using parsimony
- Removes the virtual root and cleans up flanking edges
- Runs **simplification** to remove redundant nodes and edges

.. admonition:: Confusion Buster -- What Does Simplification Do?

   **Simplification** is a tskit operation that removes everything from
   a tree sequence that is not ancestral to the designated samples. Imagine
   a tree where some internal nodes have only one child (so-called *unary
   nodes*) -- these represent lineages that passed through without any
   coalescence and can be removed without changing the trees as seen from
   the samples. Simplification also removes nodes that have no sample
   descendants at all, and merges adjacent edges that can be combined.
   The result is a leaner, equivalent tree sequence.

The result is a complete ``tskit.TreeSequence``.


Terminology
============

Before diving into the gears, let's nail down the terminology precisely.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - Definition
   * - **Variant matrix**
     - The :math:`n \times m` matrix :math:`\mathbf{D}` of allelic states
   * - **Ancestral allele**
     - The allele present in the common ancestor (encoded as ``0``)
   * - **Derived allele**
     - The mutant allele (encoded as ``1``)
   * - **Inference site**
     - A site used for tree building (biallelic, ancestral known, non-singleton)
   * - **Non-inference site**
     - A site excluded from inference (mutations placed by parsimony later)
   * - **Focal site**
     - The site that defines an ancestor's time tier
   * - **Ancestor**
     - A putative ancestral haplotype inferred from frequency patterns
   * - **Time**
     - A proxy for age, computed as frequency of the derived allele
   * - **Copying model**
     - The Li & Stephens HMM used to express one haplotype as a mosaic of others
   * - **Viterbi path**
     - The most likely sequence of hidden states (which haplotype is being copied)
   * - **Path compression**
     - Merging redundant edges via synthetic "path compression" nodes
   * - **Simplification**
     - Removing nodes and edges not ancestral to any sample


Parameters
===========

tsinfer takes the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Symbol
     - Meaning
   * - Recombination rate
     - :math:`\rho`
     - Expected number of recombinations per unit of genetic distance
   * - Mismatch ratio
     - :math:`\mu / \rho`
     - Ratio of mismatch to recombination probability in the LS model
   * - Number of samples
     - :math:`n`
     - Number of input haplotypes
   * - Number of sites
     - :math:`m`
     - Number of variable sites in the input data
   * - Precision
     - :math:`p`
     - Number of decimal digits for edge coordinates (default: 22)
   * - Simplify
     - --
     - Whether to run simplification on the output (default: True)

.. admonition:: Defaults are usually fine

   In practice, tsinfer's defaults work well for most datasets. The
   recombination and mismatch rates are estimated from the data when not
   provided. The main parameter to watch is the mismatch ratio, which
   controls how aggressively the model allows copying errors vs.
   recombination switches.


The Flow in Detail
===================

Here's a more detailed view of how the pieces connect. Follow this diagram
as a reference map throughout the subsequent chapters:

.. code-block:: text

   Variant data D
         |
         v
   +---------------------------+
   |  Site filtering           |
   |  - Biallelic?             |
   |  - Ancestral allele known?|
   |  - >= 2 derived copies?   |
   |  - >= 1 ancestral copy?   |
   +---------------------------+
         |
         | inference sites + non-inference sites
         v
   +---------------------------+
   |  ANCESTOR GENERATION      |   <-- "Extracting the template gears"
   |                           |
   |  For each inference site: |
   |    time = freq(derived)   |
   |    focal samples =        |
   |      {i : D[i,j] = 1}    |
   |    Extend left/right by   |
   |    consensus voting       |
   +---------------------------+
         |
         | A ancestors with [start, end] intervals
         v
   +---------------------------+
   |  ANCESTOR MATCHING        |   <-- "Assembling the movement"
   |                           |
   |  Sort ancestors by time   |
   |  (oldest first)           |
   |                           |
   |  For each ancestor a:     |
   |    panel = older ancestors |
   |    path = Viterbi(a, panel)|
   |    Add edges to tree seq  |
   +---------------------------+
         |
         | Ancestor tree sequence
         v
   +---------------------------+
   |  SAMPLE MATCHING          |   <-- "Fitting the hands to the dial"
   |                           |
   |  For each sample s:       |
   |    panel = all ancestors  |
   |    path = Viterbi(s, panel)|
   |    Add edges to tree seq  |
   +---------------------------+
         |
         v
   +---------------------------+
   |  POST-PROCESSING          |
   |                           |
   |  1. Place mutations at    |
   |     non-inference sites   |
   |     (parsimony)           |
   |  2. Remove virtual root   |
   |  3. Erase flanking edges  |
   |  4. Simplify              |
   +---------------------------+
         |
         v
   Final tree sequence T


Why Not MCMC?
==============

tsinfer deliberately sacrifices statistical optimality for **scalability**.
MCMC methods like SINGER and ARGweaver sample from the posterior distribution
of genealogies, but they scale as :math:`O(n^2)` or worse per iteration.
For biobank-scale data (:math:`n > 100{,}000`), this is impractical.

tsinfer instead finds a **single best-fit** genealogy using the Viterbi
algorithm. The cost:

- No posterior uncertainty estimates
- Branch lengths (times) are approximate (frequency-based proxy)
- Topology may be suboptimal in regions with low information

The benefit:

- Scales to :math:`n = 10^6` samples
- Runs in hours, not months
- Output is a standard ``tskit.TreeSequence`` that integrates with the
  entire tskit ecosystem

In practice, the topology inferred by tsinfer is remarkably accurate, and the
approximate times can be refined downstream using ``tsdate``.

.. admonition:: Probability Aside -- Point Estimates vs. Posterior Distributions

   The distinction between tsinfer and MCMC methods mirrors a fundamental
   divide in statistics. MCMC methods approximate the full **posterior
   distribution** :math:`P(\mathcal{T} \mid \mathbf{D})`, giving you not
   just one genealogy but a distribution over genealogies weighted by their
   probability. The Viterbi algorithm used by tsinfer finds the **maximum
   a posteriori (MAP) estimate** -- the single most probable genealogy. This
   is analogous to reporting a point estimate (the mean or mode) instead of
   a full confidence interval. The MAP estimate can be excellent when data is
   abundant (many sites, many samples), but it discards information about
   uncertainty. For downstream analyses that require uncertainty
   quantification, consider pairing tsinfer with tsdate (for branch length
   uncertainty) or using MCMC methods on smaller subsets of the data.


Ready to Build
===============

You now have the high-level blueprint. In the following chapters, we'll build
each gear from scratch:

1. :ref:`tsinfer_ancestor_generation` -- Inferring ancestral haplotypes from allele frequencies
2. :ref:`tsinfer_copying_model` -- The Li & Stephens HMM engine
3. :ref:`tsinfer_ancestor_matching` -- Building the ancestor tree
4. :ref:`tsinfer_sample_matching` -- Threading samples and post-processing

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. The chapters build on each other: Gear 1 produces
the ancestors that Gear 2's engine will process, Gear 3 uses that engine
to assemble the ancestor tree, and Gear 4 completes the movement by
threading in the samples.

Let's start with the first gear: Ancestor Generation.
