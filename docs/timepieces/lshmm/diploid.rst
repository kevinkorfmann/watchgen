.. _diploid:

====================
A Diploid Extension
====================

This section describes a later extension, not a derivation from Li and Stephens
(2003). For an unphased biallelic query genotype, use an ordered pair of copying
states :math:`(Z^{(1)}_\ell,Z^{(2)}_\ell)`. There are :math:`k^2` states.
The two chromosomes transition independently, so

.. math::

   P((i_1,i_2)\rightarrow(j_1,j_2))=A_{i_1j_1}A_{i_2j_2}.

The rows sum to one because each factor :math:`A` is stochastic. Although the
state pair is ordered computationally, unphased genotype emissions are
symmetric under swapping the two labels; the labels are therefore not
identifiable from these observations alone.

For a copied genotype and an observed genotype, the emission probabilities are
obtained from two independent allele-error events. For example, matching
homozygotes have probability :math:`(1-\mu)^2`, opposite homozygotes have
probability :math:`\mu^2`, and a copied homozygote emitting a heterozygote has
probability :math:`2\mu(1-\mu)`. A copied heterozygote emitting either
homozygote has probability :math:`\mu(1-\mu)`.

.. literalinclude:: ../../../watchgen/mini_lshmm.py
   :language: python
   :pyobject: emission_matrix_diploid

A naive forward update costs :math:`O(k^4)` per site. Separating the cases in
which neither, one, or both copying states redraw reduces it to
:math:`O(k^2)`. If :math:`F` is the previous :math:`k\times k` forward
matrix, the unscaled pre-emission value for target :math:`(a,b)` is

.. math::

   (1-r)^2F_{ab}
   +(1-r)\frac{r}{k}\left(\sum_jF_{aj}+\sum_iF_{ib}\right)
   +\left(\frac{r}{k}\right)^2\sum_{ij}F_{ij}.

.. literalinclude:: ../../../watchgen/mini_lshmm.py
   :language: python
   :pyobject: forward_diploid

The optimized recursion is tested against a direct :math:`O(k^4)` enumeration
for both scaled and unscaled calculations. This is important because a vector
broadcasting error can preserve plausible row sums while assigning probability
to the wrong destination states.
