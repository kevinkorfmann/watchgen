.. _relate_painting:

===============================
Directional chromosome painting
===============================

Relate's HMM is directional because it models mutations on the target lineage
since its most recent common ancestor with a candidate copying haplotype. Let
:math:`D^{(i)}_\ell` be the target allele, :math:`D^{(j)}_\ell` the reference
allele, and :math:`p` the mismatch parameter. The emission is

.. math::

   P_m(D^{(i)}_\ell\mid H_\ell=j)=
   \begin{cases}
   p, & D^{(i)}_\ell=1,\ D^{(j)}_\ell=0,\\
   1-p, & \text{otherwise}.
   \end{cases}

Only the target-derived/reference-ancestral case is penalized. In
particular, ancestral-target/derived-reference is *not* assigned a second,
user-chosen mismatch weight. The old mini's ``w_d`` and ``w_a`` parameters did
not describe Relate :cite:`relate`.

Dense teaching recursion
========================

For a small panel of :math:`K` references, let :math:`r_\ell` be the probability
of redrawing a copying state between sites :math:`\ell-1` and :math:`\ell`.
The forward prediction for state :math:`j` is

.. math::

   \widetilde\alpha_\ell(j)
   = (1-r_\ell)\alpha_{\ell-1}(j)+\frac{r_\ell}{K}
     \sum_h\alpha_{\ell-1}(h).

Multiplication by the emission and normalization gives the forward message; the
backward recursion is analogous. ``mini_relate.copying_posterior`` implements
this dense form. Production Relate exploits the one-sided emission by painting
only sites derived in the target, stores stepping-stone boundary messages, caps
extreme transition probabilities, and repaints short sections when a complete
matrix is needed.

From posterior weight to directional score
==========================================

With no recombination, define :math:`d(i,j)` as the number of sites derived in
target :math:`i` and ancestral in reference :math:`j`. The Supplementary Note
derives

.. math::

   d(i,j)=
   \frac{\log P_m(D^{(i)}\mid H=j)-L\log(1-p)}
        {\log[p/(1-p)]}.

The same rescaling of the local forward--backward score defines
:math:`d(i,j;\ell)`. Relate subtracts :math:`\min_{j\ne i}d(i,j;\ell)` from each
row. This removes an additive tip-branch residual and matters for the fallback
symmetrized score. Consequently a normalized posterior is sufficient for the
mini's *relative* row, but it must not be presented as an absolute mutation count.

Why asymmetry matters
=====================

Under infinite sites and no recombination,

.. math::

   d(i,j)=\#\{\ell:D^{(i)}_\ell=1,\ D^{(j)}_\ell=0\}.

This matrix need not equal its transpose. Symmetrizing it produces the ordinary
number of pairwise differences and discards which lineage carries each derived
mutation. The directional row order is the information used by Relate's tree
builder.

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: modified_emission

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: relative_distance_row
