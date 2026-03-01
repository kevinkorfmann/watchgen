.. _timepieces:

==========
Timepieces
==========

   *Every great watch begins as a pile of parts on the bench.*

Each Timepiece is a complete algorithm from population genetics, disassembled and
rebuilt from scratch. Like a watchmaker laying out parts on the bench, we examine
every gear before assembling the whole mechanism. Nothing is taken on faith. Nothing
is hidden behind "see supplementary materials."


Verification Status
====================

.. admonition:: Disclaimer

   The code examples in each Timepiece are verified by automated unit tests that
   re-implement the documented functions and check their mathematical properties.
   **No Timepiece has been independently verified by a domain expert.** If you find
   an error -- mathematical, pedagogical, or computational -- please open an issue.
   The table below shows the current verification status.

The Timepieces are grouped by what they do, with verification status for each.

**Simulators** -- tools that generate ground truth

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 10 55

   * - #
     - Timepiece
     - Tests
     - Verified
     - What it does
   * - IV
     - :ref:`msprime <msprime_timepiece>`
     - 190
     - --
     - Neutral coalescent with recombination. The clockwork that generates ground truth.
   * - XVI
     - :ref:`SLiM <slim_timepiece>`
     - 67
     - --
     - Forward-time simulation with natural selection. The forge that builds what the
       coalescent cannot.
   * - XVIII
     - :ref:`discoal <discoal_timepiece>`
     - 132
     - --
     - Coalescent simulation with selective sweeps via trajectory + structured coalescent.

**Demographic inference** -- estimating population size history

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 10 55

   * - #
     - Timepiece
     - Tests
     - Verified
     - What it does
   * - I
     - :ref:`PSMC <psmc_timepiece>`
     - 186
     - --
     - Population size history from a single diploid genome. The simplest inference
       Timepiece.
   * - II
     - :ref:`SMC++ <smcpp_timepiece>`
     - 112
     - --
     - Extends PSMC to multiple unphased genomes with a distinguished lineage approach.
   * - XIII
     - :ref:`Gamma-SMC <gamma_smc_timepiece>`
     - 107
     - --
     - Ultrafast pairwise TMRCA inference with gamma-distributed posteriors.
   * - XIV
     - :ref:`PHLASH <phlash_timepiece>`
     - 130
     - --
     - GPU-accelerated Bayesian inference of population size history via SVGD.

**SFS-based demographic inference** -- using the site frequency spectrum

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 10 55

   * - #
     - Timepiece
     - Tests
     - Verified
     - What it does
   * - X
     - :ref:`moments <moments_timepiece>`
     - 162
     - --
     - Demographic inference from the SFS using moment equations.
   * - XI
     - :ref:`dadi <dadi_timepiece>`
     - 84
     - --
     - Demographic inference from the SFS by solving the Wright-Fisher diffusion PDE.
   * - XII
     - :ref:`momi2 <momi2_timepiece>`
     - 140
     - --
     - Demographic inference from the SFS via coalescent tensor algebra.

**Genealogy and ARG inference** -- reconstructing ancestral histories

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 10 55

   * - #
     - Timepiece
     - Tests
     - Verified
     - What it does
   * - III
     - :ref:`Li & Stephens HMM <lshmm_timepiece>`
     - 158
     - --
     - The copying model: a haplotype as a mosaic of references. A gear inside many
       Timepieces.
   * - V
     - :ref:`ARGweaver <argweaver_timepiece>`
     - 120
     - --
     - Bayesian ARG sampling with discretized time. SINGER's predecessor.
   * - VI
     - :ref:`tsinfer <tsinfer_timepiece>`
     - 142
     - --
     - Deterministic tree sequence inference at biobank scale.
   * - VII
     - :ref:`SINGER <singer_timepiece>`
     - 172
     - --
     - Bayesian ARG sampling with continuous time and two-HMM architecture.
   * - VIII
     - :ref:`Threads <threads_timepiece>`
     - 56
     - --
     - Deterministic ARG inference at biobank scale with PBWT pre-filtering.
   * - XVII
     - :ref:`Relate <relate_timepiece>`
     - 66
     - --
     - Genome-wide genealogy estimation via asymmetric painting + MCMC branch lengths.

**Dating and selection** -- calibrating genealogies and detecting selection

.. list-table::
   :header-rows: 1
   :widths: 5 20 10 10 55

   * - #
     - Timepiece
     - Tests
     - Verified
     - What it does
   * - IX
     - :ref:`tsdate <tsdate_timepiece>`
     - 139
     - --
     - Dates tree sequence nodes using the molecular clock.
   * - XV
     - :ref:`CLUES <clues_timepiece>`
     - 93
     - --
     - Full-likelihood estimation of selection coefficients from gene trees and ancient
       DNA.

.. toctree::
   :maxdepth: 2
   :hidden:

   psmc/index
   smcpp/index
   lshmm/index
   msprime/index
   argweaver/index
   tsinfer/index
   singer/index
   threads/index
   tsdate/index
   moments/index
   dadi/index
   momi2/index
   gamma_smc/index
   phlash/index
   clues/index
   slim/index
   relate/index
   discoal/index
