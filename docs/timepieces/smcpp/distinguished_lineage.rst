.. _smcpp_distinguished:

========================
The Distinguished Pair
========================

   *Condition a full sample genealogy on one pairwise coalescence time.*

The hidden variable
===================

Let haplotypes 1 and 2 be distinguished and let
:math:`C_{12}=\tau` be their TMRCA. The other :math:`n` haplotypes are
undistinguished. At a site, the observable symbol is :math:`(a,b)`, where
:math:`a` counts derived alleles among haplotypes 1 and 2 and :math:`b` counts
them among the other :math:`n` haplotypes.

The CSFS entry

.. math::

   \operatorname{CSFS}_{a,b}(\tau)

is the expected total branch length subtending exactly :math:`a`
distinguished and :math:`b` undistinguished leaves, conditional on
:math:`C_{12}=\tau`. Under the infinite-sites model, a mutation on such a
branch produces the observation :math:`(a,b)`.

The conditioned coalescent
==========================

The paper gives a useful backward construction. Before time :math:`\tau`, the
two blocks containing distinguished leaves 1 and 2 may each coalesce with
other blocks, but they may not coalesce with each other. Every other pair of
ancestral blocks coalesces at the ordinary pairwise rate. At
:math:`\tau`, the two distinguished ancestral blocks are forced to merge.
Above :math:`\tau`, the process is an ordinary coalescent.

If :math:`k` ancestral blocks exist below :math:`\tau`, there are
:math:`\binom{k}{2}-1` allowed mergers: all pairs except the unique pair that
would join the two distinguished ancestors. This is the precise conditioning
used by SMC++; it is not a pure-death process that remembers only a lineage
count. Descendant labels matter because they determine the emitted
:math:`(a,b)` category.

Small-sample exact calculation
==============================

The mini implementation represents a genealogy state as a set partition of
the sampled leaves. A continuous-time Markov generator assigns unit rate to
each allowed pair of blocks. A reward vector counts, in every state, how many
current branches belong to each :math:`(a,b)` category. Matrix-exponential
occupation integrals give expected branch lengths below :math:`\tau`; an
absorbing-chain calculation gives the lengths above it.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: conditioned_sfs

For two distinguished haplotypes and no undistinguished sample, the result has
the closed form

.. math::

   \operatorname{CSFS}_{1,0}(\tau)=2\tau,

with every other finite-branch category equal to zero. The two tip branches
each have length :math:`\tau`. This identity is one of the unit tests.

From branch lengths to emissions
================================

Let

.. math::

   L_m=\sum_{a,b}\operatorname{CSFS}^{(m)}_{a,b}

be the total expected tree length after averaging over TMRCA interval
:math:`I_m`. The original source converts branch lengths into emission
probabilities using

.. math::

   P_m(a,b)=\operatorname{CSFS}^{(m)}_{a,b}
   \frac{1-e^{-\theta L_m}}{L_m}

for polymorphic outcomes. The residual probability
:math:`e^{-\theta L_m}` is assigned to the monomorphic ancestral outcome
:math:`(0,0)`. This preserves the relative CSFS branch-length weights while
using a Poisson probability for at least one mutation. It is more precise than
the first-order expression :math:`\theta\operatorname{CSFS}_{a,b}` and is the
formula implemented in ``src/conditioned_sfs.cpp``.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: incorporate_theta

There is no independent-binomial emission for the two alleles of a diploid.
Their allelic state is correlated through the conditioned genealogy, and the
undistinguished allele count is an essential part of the emission.

Conditioning on an interval
===========================

The hidden state is an interval rather than an exact time. For a constant
relative population size :math:`\lambda`, the pairwise density is

.. math::

   f(\tau)=\lambda^{-1}e^{-\tau/\lambda}.

Within :math:`I_m=[t_m,t_{m+1})`, SMC++ averages the CSFS against the
conditional density

.. math::

   f_m(\tau)=
   \frac{\lambda^{-1}e^{-\tau/\lambda}}
        {e^{-t_m/\lambda}-e^{-t_{m+1}/\lambda}}.

The mini implementation uses Gaussian quadrature, including a Gauss-Laguerre
rule for the final interval ending at infinity.

.. literalinclude:: ../../../watchgen/mini_smcpp.py
   :language: python
   :pyobject: interval_conditioned_sfs

Production scaling
==================

Set-partition enumeration grows too quickly for the hundreds of genomes that
motivated SMC++. The original program instead decomposes the CSFS above and
below :math:`\tau`, evaluates expected first-coalescence-time integrals, and
uses exact Moran-model spectral decompositions to distribute descendants among
the :math:`(a,b)` categories. Those formulas compute the same conditional
branch-length object without enumerating labeled genealogical partitions.
