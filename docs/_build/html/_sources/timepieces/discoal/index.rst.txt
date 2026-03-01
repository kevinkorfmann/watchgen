.. _discoal_timepiece:

====================================
Timepiece XVIII: discoal
====================================

   *Simulating selective sweeps in the coalescent with recombination*

The Mechanism at a Glance
==========================

discoal is a **coalescent simulator with selection**: given a sample size, genome
length, recombination rate, and a selection coefficient, it produces random genealogies
that bear the scars of a selective sweep. While msprime (Timepiece IV) generates
ground truth under *neutrality*, discoal generates ground truth under *selection* --
the same coalescent process, but warped by the passage of a beneficial allele
through the population.

If msprime is the master clockmaker's bench for neutral evolution, discoal is the
**stress-testing rig** -- the apparatus that subjects the neutral clockwork to an
external force (selection) and records how the mechanism deforms. The resulting
genealogies show the characteristic signatures that tools like CLUES (Timepiece XV)
try to detect: reduced diversity near the selected site, distorted frequency
spectra, and extended haplotype homozygosity.

The key insight behind discoal is elegant: a selective sweep at one locus creates a
**time-varying population structure** at linked neutral loci. During the sweep, the
population is split into two "backgrounds" -- chromosomes carrying the beneficial
allele and chromosomes carrying the wild type -- and the coalescent runs differently
within each. This **structured coalescent under selection** is the heart of the
mechanism.

The algorithm proceeds in two steps, like a watch with two barrels:

1. **Generate an allele frequency trajectory** for the beneficial allele (forward in
   time).
2. **Run the structured coalescent** conditioned on that trajectory (backward in
   time).

.. admonition:: Primary Reference

   :cite:`discoal`

The five gears of discoal:

1. **The Allele Frequency Trajectory** (the mainspring) -- The time course of the
   beneficial allele from its origin to fixation (or present frequency). This is
   the driving force: it determines *how the population is partitioned* at every
   moment during the sweep. Generated either deterministically (logistic growth) or
   stochastically (conditioned Wright-Fisher diffusion).

2. **The Structured Coalescent** (the escapement) -- During the sweep, lineages are
   assigned to one of two backgrounds: beneficial (:math:`B`) or wild-type
   (:math:`b`). Coalescence rates within each background depend on the background's
   size, which changes with the trajectory. The critical bottleneck -- rapid
   coalescence in the shrinking :math:`B` class -- is what destroys diversity and
   creates the sweep signature.

3. **Recombination as Migration** (the gear train) -- Recombination between the
   neutral locus and the selected site moves lineages between backgrounds. A
   lineage on a beneficial chromosome that recombines may find itself on a wild-type
   chromosome in the next generation (going backward). The ratio :math:`r/s`
   governs escape: tightly linked loci are dragged along; loosely linked ones
   recombine free.

4. **Sweep Varieties** (the complications) -- Hard sweeps from a single new
   mutation. Soft sweeps from standing variation. Soft sweeps from recurrent
   mutation. Partial sweeps where the beneficial allele has not yet fixed. Each
   variety reshapes the genealogy differently, producing distinct statistical
   signatures.

5. **Neutral Bookends** (the case and dial) -- Before the sweep begins and after it
   ends, the coalescent runs under standard neutrality. Mutations are scattered on
   branches at the end, producing the haplotype data that researchers actually
   observe.

These gears mesh together into a complete sweep simulator:

.. code-block:: text

   Parameters (n, L, theta, rho, alpha=2Ns, tau, sweep type)
                      |
                      v
   STEP 1: Generate allele frequency trajectory x(t)
             deterministic (logistic)
                  or
             stochastic (conditioned diffusion)
                      |
                      v
             x(t): [1/2N ... ... ... 1.0]
                      |
                      v
   STEP 2: Run coalescent backward in time
                      |
        +-------------+-------------+
        |             |             |
        v             v             v
   Neutral phase   Sweep phase   Ancestral neutral
   (present to     (structured   (before sweep
    tau)            coalescent    to MRCA)
                    with two
                    backgrounds)
        |             |             |
        +-------------+-------------+
                      |
                      v
             Scatter mutations on branches (Poisson)
                      |
                      v
             Output: haplotype matrix (ms-compatible)


.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- exponential waiting times,
     coalescence rates, the standard neutral coalescent
   - :ref:`The msprime Timepiece <msprime_timepiece>` -- Hudson's algorithm and the
     neutral coalescent with recombination. discoal extends this machinery.
   - Familiarity with the Wright-Fisher model and genetic drift is helpful but not
     strictly required -- we build the selection extension from scratch.

.. admonition:: How discoal relates to other Timepieces

   - **msprime** (IV) simulates neutral genealogies. discoal adds selection.
     msprime 1.0 now includes sweep support using the same two-step algorithm.
   - **SLiM** (XVI) simulates selection forward in time, tracking every individual.
     discoal works backward, using the coalescent -- far more efficient for moderate
     sample sizes.
   - **CLUES** (XV) *detects* selection from gene trees. discoal *generates* the gene
     trees under selection -- they are natural complements.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   allele_trajectory
   structured_coalescent
   sweep_types
   msprime_comparison
   demo
