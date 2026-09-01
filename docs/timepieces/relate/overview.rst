.. _relate_overview:

========================
What Relate estimates
========================

Relate takes phased, biallelic haplotypes together with physical positions, a
recombination map, ancestral-allele labels, a mutation rate, and a population-size
scale. Its core outputs are ``.anc`` files describing local trees and ``.mut``
files describing mapped mutations. Conversion utilities can translate those
files to other formats, but tskit is not the native inference representation.

The inferred tree sequence is an approximation to an ancestral recombination
graph. Relate estimates marginal trees locally without enforcing a single global
recombination history. This choice makes the calculation parallelizable and gives
the reported linear scaling in sequence length and quadratic scaling in sample
size :cite:`relate`.

Three coupled stages
====================

Topology inference
------------------

Relate paints each haplotype as a mosaic of all other haplotypes. Its modified
emission distinguishes a mutation derived in the target and ancestral in the
reference from the other three allele pairs. Forward--backward scores give a
position-specific ordering of likely relatives. After a log rescaling and a
row-minimum subtraction, these scores form a directional matrix.

The tree builder repeatedly merges clusters that are mutual row minima, allowing
a documented tolerance for inconsistent data. Cluster-to-cluster distances are
means over all ordered pairs of member haplotypes. A smallest-symmetrized-distance
rule resolves multiple feasible pairs and supplies a fallback when none are
feasible.

Mutation mapping and tree changes
---------------------------------

Relate starts with a tree near the 5' end and attempts to map each subsequent
mutation to it. Under infinite sites, the carriers should equal the descendants
of one branch. The production mapper permits small discrepancies, can flip
ancestral and alternative labels, and can distribute a mutation across a minimal
set of branches. A new topology is built when a mutation cannot be mapped uniquely
or appears flipped. Thus production Relate does not construct an unrelated tree
at every SNP.

Dating and demographic refinement
----------------------------------

Equivalent branches in adjacent trees are associated so their mutation counts
and genomic exposures can be pooled. Relate samples ranked coalescence events and
the intervals :math:`\tau_k` during which :math:`k` lineages exist. The target
combines Poisson mutation probabilities with a standard- or variable-size
coalescent prior.

Population history is then estimated from pairwise TMRCA events and survival
exposure within time epochs. Relate alternates this calculation with branch-length
re-estimation, using five iterations in the paper's implementation. Calling this
whole procedure a conventional EM algorithm obscures the actual estimator and its
heuristic mutation-rate rescaling.

Scope of the mini
=================

``watchgen.mini_relate`` includes a dense small-panel painting recursion, relative
distance conversion, the documented mutual-minimum tree rule, exact mutation
mapping, a fixed-event-order interval sampler, and the pairwise epoch-rate MLE.
It omits production windowing, topology reuse along a chromosome, approximate
mapping, branch association, event-order swaps, variable-size dating, input/output
formats, and parallel execution.

.. figure:: /_static/figures/fig_mini_relate.png
   :width: 100%
   :align: center

   Source-guided kernels used in this Timepiece; the figure is not an output from
   the Relate executable.
