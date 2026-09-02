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
     - 28
     - 2026-09-01
     - SLiM 5.2 WF mutation identity, default fitness, parent sampling, tick order,
       and uniform-map recombination checked against source/manual semantics; three
       current Eidos recipes execute with the tagged 5.2 binary.
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
     - 114
     - 2026-09-01
     - Canonical flow perturbation, conjugate emissions, backward-message indexing,
       and segmented trailing blocks checked against the primary paper and official
       Gamma-SMC source (``61a4d046``).
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
     - 118
     - 2026-09-01
     - SFS neutral drift/mutation scaling and population splitting checked against
       moments-popgen 1.6.1 source fixtures; selection, migration, and LD teaching
       approximations are explicitly bounded.
   * - XI
     - :ref:`dadi <dadi_timepiece>`
     - 54
     - 2026-09-01
     - Grid, equilibrium density, split, implicit integration, likelihood, and
       scaling checked against dadi 2.4.5 source/executable behavior (``8db007f``).
   * - XII
     - :ref:`momi2 <momi2_timepiece>`
     - 143
     - 2026-09-01
     - Coalescent recurrence/sojourn times, Moran transitions, pulse tensor,
       hypergeometric pseudoinverse, likelihood, and finite-sample f-statistic
       weights checked against papers and official momi2 source (``b038d43``).

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
     - 125
     - 2026-09-01
     - Exact Jukes--Cantor emissions, diploid coalescent scaling, discretized
       recoalescence, and bounded teaching approximations checked against the
       primary paper and official C++ source (``fee3d32``); the upstream
       executable was built and completed a finite two-iteration sampling run.
   * - VI
     - :ref:`tsinfer <tsinfer_timepiece>`
     - 21
     - 2026-09-01
     - Ancestor grouping, carrier-filtered extension, two-root construction,
       probability transforms, dense Viterbi, path coordinates, compression
       eligibility, parsimony, and flank handling checked against the primary
       paper and official 0.1.4/0.4.1 Python implementations; the teaching code
       is explicitly bounded below production tree-sequence inference.
   * - VII
     - :ref:`SINGER <singer_timepiece>`
     - 35
     - 2026-09-01
     - Published branch/time HMM equations, mutation-clock rescaling, and SGPR
       acceptance checked against the paper supplement and official C++ source;
       teaching code explicitly bounded below full ARG-sampler parity.
   * - VIII
     - :ref:`Threads <threads_timepiece>`
     - 65
     - 2026-09-01
     - Dating equations, production dispatch, PBWT/Viterbi boundaries, and
       numerical tail stability checked against the thesis/paper and official
       source (``25c6c0d``); ``threads-arg`` 0.2.1 completed its 500-haplotype
       example with 55,765 finite, positive dated segments.
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
     - 21
     - 2026-09-01
     - CLUES/CLUES2 version and parameter boundaries audited against both primary
       papers and source ``b20dc5d``; transition/emission fixtures, brute-force HMM
       recursion, importance ratios, and the official ancient-haplotype CLI run passed.
