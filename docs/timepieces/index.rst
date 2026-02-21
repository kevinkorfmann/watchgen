.. _timepieces:

==========
Timepieces
==========

   *Every great watch begins as a pile of parts on the bench.*

Each Timepiece is a complete algorithm from population genetics, disassembled and
rebuilt from scratch. Like a watchmaker laying out parts on the bench, we examine
every gear before assembling the whole mechanism. Nothing is taken on faith. Nothing
is hidden behind "see supplementary materials."

The Timepieces are ordered roughly by complexity, though each is self-contained. We
recommend starting with PSMC (the simplest complete inference machine) and msprime
(the simulator that generates the ground truth), then working through the others as
your interests guide you.

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - #
     - Timepiece
     - What it does
   * - I
     - :ref:`PSMC <psmc_timepiece>`
     - Infers population size history from a single diploid genome. The simplest
       complete inference Timepiece -- a two-hand watch.
   * - II
     - :ref:`Li & Stephens HMM <lshmm_timepiece>`
     - The copying model that describes a haplotype as a mosaic of reference
       haplotypes. A versatile gear that appears inside many other Timepieces.
   * - III
     - :ref:`msprime <msprime_timepiece>`
     - Simulates ancestral histories under the coalescent with recombination. The
       clockwork that generates ground truth.
   * - IV
     - :ref:`ARGweaver <argweaver_timepiece>`
     - Bayesian ARG sampling with discretized time. SINGER's predecessor -- a
       single-HMM approach with exact forward-backward computation.
   * - V
     - :ref:`tsinfer <tsinfer_timepiece>`
     - Deterministic tree sequence inference that scales to biobank-sized data.
       Speed over statistical optimality.
   * - VI
     - :ref:`SINGER <singer_timepiece>`
     - Bayesian ARG sampling with continuous time and two-HMM architecture.
       The most complex Timepiece in the collection.
   * - VII
     - :ref:`Threads <threads_timepiece>`
     - Deterministic ARG inference at biobank scale. PBWT pre-filtering,
       memory-efficient Viterbi, and segment dating.
   * - VIII
     - :ref:`tsdate <tsdate_timepiece>`
     - Dates the nodes in a tree sequence using the molecular clock. Turns
       tsinfer's skeleton into a fully calibrated genealogy.
   * - IX
     - :ref:`moments <moments_timepiece>`
     - Demographic inference from the site frequency spectrum using moment
       equations. A different lens on the same evolutionary history.
   * - X
     - :ref:`momi2 <momi2_timepiece>`
     - Demographic inference from the SFS via the coalescent, using tensor
       algebra and automatic differentiation. The backward-time counterpart
       to moments.

.. toctree::
   :maxdepth: 2
   :hidden:

   psmc/index
   lshmm/index
   msprime/index
   argweaver/index
   tsinfer/index
   tsdate/index
   singer/index
   threads/index
   moments/index
   momi2/index
