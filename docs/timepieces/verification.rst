.. _verification_status:

===================
Verification Status
===================

.. admonition:: Disclaimer

   The code examples in each Timepiece are checked by chapter-specific automated
   tests, including mathematical invariants and upstream numerical fixtures where
   available.
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
     - 209
     - 2026-09-01
     - Hudson simulation and mutation generation, including gap recombination and
       discrete-site recurrent/back mutations checked against upstream source behavior.
   * - XVI
     - :ref:`SLiM <slim_timepiece>`
     - 67
     - --
     - Forward-time simulation with natural selection. The forge that builds what the
       coalescent cannot.
   * - XVIII
     - :ref:`discoal <discoal_timepiece>`
     - 32
     - 2026-09-01
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
     - 205
     - 2026-09-01
     - Population size history from a single diploid genome; continuous-chain
       normalization checked against analytic infinite-tail oracles.
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
     - 179
     - 2026-09-01
     - PAC likelihood and copying HMM, checked against Appendix A, lshmm 0.0.8,
       and brute-force haploid and diploid recursions.
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
     - 35
     - 2026-09-01
     - Published branch/time HMM equations, mutation-clock rescaling, and SGPR
       acceptance checked against the paper supplement and official C++ source;
       teaching code explicitly bounded below full ARG-sampler parity.
   * - VIII
     - :ref:`Threads <threads_timepiece>`
     - 56
     - --
     - Deterministic ARG inference at biobank scale with PBWT pre-filtering.
   * - XVII
     - :ref:`Relate <relate_timepiece>`
     - 37
     - 2026-09-01
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
