.. _relate_tree_building:

===================================
Directional hierarchical clustering
===================================

Relate begins with one cluster per haplotype. For clusters :math:`A` and
:math:`B`, it defines the directional distance

.. math::

   d(A,B)=\frac{1}{|A||B|}\sum_{i\in A}\sum_{j\in B}d(i,j).

This is a cardinality-weighted mean over original haplotype pairs. It is not the
minimum of the two directions and not an unweighted average of previously merged
cluster scores.

The merge condition
===================

Clusters :math:`A` and :math:`B` are eligible when each lies at the minimum of the
other's directional row:

.. math::

   d(A,B)\leq \min_{C\ne A}d(A,C)+\epsilon,
   \qquad
   d(B,A)\leq \min_{C\ne B}d(B,C)+\epsilon.

The Supplementary Note reports :math:`\epsilon=0.2` on the mutation-distance
scale. The C++ implementation stores an equivalent tolerance on its log-score
scale. If several pairs are eligible, Relate chooses the one minimizing
:math:`d(A,B)+d(B,A)`. If inconsistent data leave no eligible pair, it falls back
to the smallest symmetrized score. The earlier mini instead selected the pair
with the smallest value in either direction; that rule can merge clusters that
are not mutual neighbours.

Under infinite sites and no recombination, the directional mutation-count matrix
guarantees a tree consistent with every mutation-supported clade. It does not
guarantee recovery of binary resolutions unsupported by mutations. The paper
also proves consistency for limiting maps made of zero- and infinite-recombination
segments, not for arbitrary finite-recombination data :cite:`relate`.

Mapping mutations while moving along the genome
================================================

For an exact infinite-sites mutation, the carrier set equals the descendants of
one branch. Production Relate adds robust behavior:

* a candidate branch must recover more than 70% of carriers and non-carriers;
* the final misclassification fraction must be below 0.03;
* ancestral and alternative labels are flipped only when that improves the fit;
* if no unique branch works, a smallest exact collection of branches is used and
  the mutation count is divided across those branches.

Relate builds a new local topology when the next mutation cannot be mapped
uniquely or appears potentially flipped. Otherwise it reuses the current tree.
This is why a function that independently clusters every SNP is not a production
Relate implementation.

The teaching boundary
=====================

``mini_relate.build_tree`` implements the mutual-minimum rule and recomputes the
cardinality-weighted means directly. ``mini_relate.map_mutation_exact`` implements
only exact single-branch mapping and exact flip detection. It intentionally omits
the production thresholds, fractional multi-branch mapping, random tie handling,
sample-age constraints, topology templates, and streaming chromosome logic.

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: find_mutual_minimum_pair

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: map_mutation_exact
