.. _argweaver_emissions:

=======================
Emission Probabilities
=======================

ARGweaver's threading HMM scores a candidate attachment by the probability of
the observed bases on the local tree that would result from that attachment.
The production implementation does this with probabilistic partial likelihoods,
not with a parsimony score. This distinction matters: a parsimony assignment
keeps only one reconstruction of the ancestral bases, whereas a likelihood sums
over every possible ancestral base assignment.

The description here follows the original ARGweaver paper
:cite:`argweaver` and the current ``mdrasmus/argweaver`` implementation. In
particular, ``src/argweaver/emit.cpp`` computes inner and outer likelihood
tables and evaluates candidate attachment states with Jukes--Cantor branch
probabilities.

The Jukes--Cantor branch model
==============================

For a branch of length :math:`t` and mutation rate :math:`\mu`, ARGweaver uses

.. math::

   P_{aa}(t) = \frac{1}{4}\left(1 + 3e^{-4\mu t/3}\right),
   \qquad
   P_{ab}(t) = \frac{1}{4}\left(1 - e^{-4\mu t/3}\right),\quad a\ne b.

The second expression is the probability of one *specified* alternative base,
not the probability of any change. Both expressions approach :math:`1/4` on a
very long branch. Replacing them by :math:`e^{-\mu t}` and
:math:`(1-e^{-\mu t})/3` is a low-rate approximation, but it is not the branch
kernel used by the production C++ emission calculation.

Conditional likelihoods on a tree
==================================

Let :math:`L_u(a)` be the likelihood of all observations below node :math:`u`
conditional on base :math:`a` at that node. At a leaf with observed base
:math:`x_u`,

.. math::

   L_u(a) = \mathbf{1}[a=x_u].

For an internal binary node with children :math:`v` and :math:`w`, pruning gives

.. math::

   L_u(a) =
   \left(\sum_b P_{ab}(t_v)L_v(b)\right)
   \left(\sum_c P_{ac}(t_w)L_w(c)\right).

At the root, the stationary Jukes--Cantor distribution is uniform, so the tree
likelihood is :math:`\tfrac14\sum_a L_r(a)`. Missing ``N`` observations are
represented by likelihood one for every base rather than being forced to an
arbitrary nucleotide.

The miniature implementation exposes the same recursion in log space. It
returns the four root-conditional log likelihoods so that tests can compare
each state directly with the closed-form Jukes--Cantor probabilities.

.. literalinclude:: ../../../watchgen/mini_argweaver.py
   :language: python
   :pyobject: felsenstein_pruning

Scoring a threading state
=========================

A state :math:`(b,i)` attaches the threaded lineage, or threaded subtree, to
branch :math:`b` at grid time :math:`t_i`. This creates three relevant branch
pieces: the branch from the threaded lineage to the new coalescence, the lower
piece of branch :math:`b`, and, unless the attachment is above the root, the
upper piece of branch :math:`b`. Their lengths determine three Jukes--Cantor
transition kernels.

Re-running pruning independently for every state would repeat nearly all work.
ARGweaver therefore caches an ``inner`` table for the likelihood below each
node and an ``outer`` table for the likelihood outside each node's subtree.
For a candidate state, ``calc_emit`` combines the cached inner and outer terms
with the three new branch kernels and sums over the base at the new attachment
point. This is algebraically the likelihood of the augmented local tree.

The same construction handles internal-branch threading. In that case one
cached partial represents the removed subtree and another represents the
remaining tree. The likelihood calculation joins those two partials at each
candidate state. Sites affected by unphased heterozygotes are evaluated under
both phase assignments and averaged, while recording conditional phase
probabilities for sampling.

Where parsimony is actually used
================================

The source tree still contains parsimony routines, but they do not replace the
finite-sites likelihood above. They support the optional infinite-sites
compatibility filter and related diagnostics. When that option is active,
states incompatible with a one-mutation explanation are down-weighted by the
configured penalty. The normal finite-sites emission path continues to use
inner and outer likelihoods.

Invariant sites have a separate optimized path because every observed leaf has
the same base. Fully masked sites emit probability one for every state. These
shortcuts are computational optimizations with explicit source branches; they
should not be generalized into the claim that all emissions are parsimony
scores.

Numerical checks
================

For a single observed ``A`` separated from the root by branch length :math:`t`,
the root-conditional probabilities provide a direct oracle:

.. code-block:: python

   import numpy as np
   from watchgen.mini_argweaver import felsenstein_pruning

   mu = 0.2
   t = 2.5
   result = felsenstein_pruning(
       {"root": ["sample"]},
       {"sample": "root"},
       {"sample": "A"},
       mu,
       {"sample": t},
   )

   decay = np.exp(-4 * mu * t / 3)
   assert np.isclose(np.exp(result["A"]), (1 + 3 * decay) / 4)
   assert np.isclose(np.exp(result["C"]), (1 - decay) / 4)

This check distinguishes the exact kernel from the low-rate approximation and
also verifies that the pruning direction is correct. Larger tests compare the
recursion with direct enumeration over internal-node bases.

Scope of the teaching implementation
====================================

The miniature function demonstrates the exact branch kernel and pruning
recursion, but it is not a replacement for ``calc_emissions``. It does not
build inner and outer tables across an alignment, perform invariant-site
compression, average phase configurations, attach a removed subtree, or apply
the optional infinite-sites penalty. Those features remain explicit boundaries
between the teaching kernel and ARGweaver's production engine.

The central point is therefore precise: ARGweaver uses probabilistic
Jukes--Cantor emissions for posterior threading. Parsimony is an auxiliary
compatibility tool, not the emission likelihood itself.
