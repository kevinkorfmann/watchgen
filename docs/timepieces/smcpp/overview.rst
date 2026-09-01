.. _smcpp_overview:

===================
Overview of SMC++
===================

   *The hidden genealogy stays small; the observation becomes richer.*

What SMC++ changes
==================

PSMC follows the TMRCA of two haplotypes along a genome. Its binary emission
records whether those haplotypes differ. This uses linkage information but
discards variants carried only by other sampled genomes. A conventional
frequency-spectrum analysis uses all samples but discards linkage by treating
sites as independent.

SMC++ joins those two views. Choose two haplotypes as a distinguished pair and
let :math:`T_\ell` be their TMRCA at locus :math:`\ell`. At the same locus,
record

.. math::

   X_\ell=(a_\ell,b_\ell),

where :math:`a_\ell\in\{0,1,2\}` is the derived count in the pair and
:math:`b_\ell\in\{0,\ldots,n\}` is the count among :math:`n`
undistinguished haplotypes. The hidden state is still one time interval, but
the emission is the CSFS

.. math::

   P(X_\ell=(a,b)\mid T_\ell\in I_m,\eta),

which depends on the population-size history :math:`\eta`.

.. admonition:: The essential correction

   Extra samples affect the emissions. They do **not** replace the pairwise
   TMRCA with the first coalescence among all samples, and they do not create a
   sample-size-dependent hazard in the transition matrix. The original paper
   explicitly states that PSMC and SMC++ track the same hidden information;
   their emissions differ.

Coalescent units
================

For :math:`n` haploid lineages in a diploid population of size :math:`N`, each
pair coalesces at rate :math:`1/(2N)` per generation. The mean time to the first
event is therefore

.. math::

   E[T_{\mathrm{first}}]
   =\frac{2N}{\binom n2}
   =\frac{4N}{n(n-1)}.

This quantity explains why a large sample contains recent frequency-spectrum
information. It is not the SMC++ hidden state: the latter remains the TMRCA of
the chosen pair, whose marginal mean is :math:`2N` under a constant diploid
population.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: expected_first_coalescence

Why unphased data are sufficient
================================

The data identify a diploid genotype count, not the parental chromosome on
which each allele lies. SMC++ chooses the two haplotypes of one diploid as the
usual distinguished pair. Exchanging those haplotypes leaves
:math:`a\in\{0,1,2\}` unchanged, and the undistinguished haplotypes enter only
through their total derived count :math:`b`. Consequently, phase switches do
not change the SMC++ observation sequence. If the distinguished haplotypes are
chosen from different individuals, their phase is not identifiable from
unphased data; the command-line documentation therefore recommends choosing a
single individual in that setting.

Composite likelihood
====================

One SMC++ input file contains one choice of distinguished pair. Users may make
several files from the same chromosome by changing that choice. The program
multiplies their HMM likelihoods, or equivalently sums their log likelihoods.
Because reused chromosomes and samples make those terms dependent, this is a
**composite likelihood**, not an ordinary likelihood with independent
replicates. The production documentation recommends only a modest number of
distinguished individuals because computational cost grows with the total
analyzed sequence length and excessive reuse can make the composite likelihood
degenerate.

What creates recent resolution
==============================

At a fixed pair TMRCA, the CSFS links that pair to branch lengths in the full
sample genealogy. A large undistinguished panel provides many recent branches,
so its allele counts contain information about recent population size even
when the distinguished pair has not yet coalesced. This is the source of the
method's recent-time resolution. It is not a shift of the hidden pair's TMRCA
toward the first event among all sampled lineages.

Limits of the mini implementation
=================================

The reference implementation computes the CSFS for hundreds of samples with
specialized spectral and combinatorial formulas. Our small implementation
enumerates set partitions, which is exact for the stated constant-population
model but grows as a Bell number. It is designed to expose the mechanism and
test identities for small :math:`n`; production analyses should use the
original SMC++ software.
