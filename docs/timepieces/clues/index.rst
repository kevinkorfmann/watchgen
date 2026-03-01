.. _clues_timepiece:

====================================
Timepiece XV: CLUES
====================================

   *Coalescent Likelihood Under Effects of Selection*

The Mechanism at a Glance
==========================

CLUES is a full-likelihood method for estimating the **selection coefficient**
:math:`s` acting on a biallelic SNP, using modern and ancient DNA. It answers the
most direct question in molecular evolution: *is this allele being favored or
disfavored by natural selection, and by how much?*

.. math::

   \text{Input: } \text{Gene trees at a locus (from Relate or SINGER) + optional ancient genotypes}

.. math::

   \text{Output: } \hat{s} \text{ (selection coefficient MLE)}, \quad \text{posterior allele frequency trajectory } P(x_t | \mathbf{D})

The key insight: an allele's frequency trajectory through time is shaped by
selection. Under neutrality (:math:`s = 0`), allele frequencies drift randomly --
the Wright-Fisher diffusion. Under selection (:math:`s \neq 0`), the trajectory is
biased: a beneficial allele drifts *upward* on average, a deleterious allele
drifts *downward*. By modeling the frequency trajectory as a Hidden Markov Model
and conditioning on the observed genealogy, CLUES computes the likelihood of the
data for each value of :math:`s` and finds the maximum.

- **Strong positive selection** (:math:`s > 0`) = the allele frequency rises faster
  than drift alone can explain. Coalescence times among derived-allele carriers
  are shorter (a selective sweep compresses the genealogy).
- **Neutrality** (:math:`s = 0`) = the frequency trajectory is consistent with
  random drift. Coalescence times follow the standard coalescent.
- **Negative selection** (:math:`s < 0`) = the allele frequency is pushed downward.
  Rare alleles stay rare; common alleles become rarer.

CLUES reads these patterns with a Hidden Markov Model whose hidden states are
discretized allele frequencies and whose emissions come from the coalescent
structure of the gene tree.

If PSMC is a two-hand watch that reads population size from a single genome, then
CLUES is a **chronometer with a compass** -- it reads both the *tempo* (coalescence
times) and the *direction* (frequency change) to detect the invisible hand of
natural selection.

.. admonition:: Primary Reference

   :cite:`clues`

The four gears of CLUES:

1. **The Wright-Fisher HMM** (the escapement) -- A discretized Wright-Fisher
   diffusion with selection, modeling how the allele frequency changes from one
   generation to the next. The transition matrix encodes the strength and direction
   of selection. This is where population genetics meets probability: the mean
   frequency shift depends on :math:`s`, and the variance depends on :math:`N`.

2. **Emission Probabilities** (the gear train) -- Two types of evidence constrain
   the allele frequency at each time point: (a) coalescent events in the gene tree
   (which lineages merge, and when) and (b) ancient genotype likelihoods (direct
   observations of the allele in the past). These are the data that the HMM
   "observes."

3. **Importance Sampling** (the mainspring) -- The genealogy at a locus is not known
   exactly -- it is estimated from an ARG sampler like Relate or SINGER. CLUES
   averages over multiple sampled genealogies using importance weights, properly
   accounting for genealogical uncertainty. This is where the output of
   :ref:`SINGER <singer_timepiece>` feeds directly into CLUES.

4. **Inference and Testing** (the case and dial) -- Maximum likelihood estimation of
   :math:`s` via Brent's method, a likelihood ratio test calibrated against
   :math:`\chi^2`, multi-epoch selection estimation, and posterior trajectory
   reconstruction. The final step: reading the watch.

These gears mesh together into a complete selection inference machine:

.. code-block:: text

   Gene tree samples from Relate/SINGER + optional ancient genotypes
                      |
                      v
            +-----------------------+
            |  BUILD FREQUENCY BINS |
            |  Beta(1/2,1/2)        |
            |  quantile spacing     |
            +-----------------------+
                      |
                      v
   +--------> BUILD TRANSITION MATRIX    |
   |          (Wright-Fisher + selection) |
   |                    |                |
   |                    v                |
   |          COMPUTE EMISSIONS          |
   |          (coalescent + ancient GL)  |
   |                    |                |
   |                    v                |
   |          BACKWARD ALGORITHM         |
   |          (for each gene tree sample)|
   |                    |                |
   |                    v                |
   |          IMPORTANCE-WEIGHTED        |
   |          LIKELIHOOD                 |
   |                    |                |
   |                    v                |
   |          OPTIMIZE s (Brent/NM)      |
   |                    |                |
   +-------- update s, recompute?        |
                        |                |
                       DONE              |
                        |                |
                        v                |
              +-------------------------+
              |  TEST & RECONSTRUCT     |
              |  LRT vs chi-squared     |
              |  Posterior trajectory    |
              +-------------------------+
                        |
                        v
                Selection coefficient + allele frequency history

Why Detect Selection?
======================

Population genetics has two great engines: **drift** and **selection**. Most of the
Timepieces in this collection focus on drift -- using the coalescent to infer
demographic history, date ancestors, or reconstruct genealogies. CLUES flips the
lens: given a genealogy (from SINGER or Relate) and a demographic model (from PSMC
or similar), it asks whether the *pattern of coalescence times at a specific locus*
departs from what neutral drift would predict.

This question matters because:

1. **Medical genetics**: Identifying loci under selection reveals genes that affect
   fitness -- often the same genes that affect disease susceptibility. The alleles
   that natural selection has targeted (lactase persistence, malaria resistance,
   skin pigmentation) are often medically relevant.

2. **Evolutionary biology**: Measuring the strength of selection across the genome
   tells us how evolution actually works in practice -- how strong are selective
   pressures? How often do they change? Do they differ between populations?

3. **Ancient DNA integration**: With the explosion of ancient DNA data, we can now
   *directly observe* allele frequencies in the past. CLUES integrates this direct
   evidence with the genealogical signal, providing a uniquely powerful framework
   for studying selection through time.

.. admonition:: Prerequisites for this Timepiece

   Before starting CLUES, you should have worked through:

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence times, exponential
     waiting times, and how population size affects the coalescent
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm and
     numerical stability in log space
   - :ref:`The SMC <smc>` -- the sequential Markov coalescent approximation
   - Familiarity with :ref:`SINGER <singer_timepiece>` or Relate is helpful, since
     CLUES uses their gene tree samples as input
   - :ref:`PSMC <psmc_timepiece>` -- to understand coalescent time distributions
     under variable population size (CLUES reuses the concept of a piecewise-constant
     :math:`N(t)`)

   If you've built those earlier mechanisms, you have every tool you need. CLUES
   adds one new ingredient -- selection -- and shows how a single parameter
   :math:`s` reshapes the entire frequency trajectory.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   wright_fisher_hmm
   emission_probabilities
   inference
   demo
