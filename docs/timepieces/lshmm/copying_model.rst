.. _copying_model:

=================
The Copying Model
=================

At the first site the copying state is uniform,

.. math::

   P(Z_1=j)=\frac{1}{k}.

For interval :math:`\ell`, let :math:`\rho_\ell d_\ell` denote the
population-scaled recombination intensity used in Appendix A of Li and
Stephens. Define the probability of a template redraw as

.. math::

   r_\ell=1-\exp\!\left(-\frac{\rho_\ell d_\ell}{k}\right).

The transition law is then

.. math::

   P(Z_\ell=j\mid Z_{\ell-1}=i)
   =(1-r_\ell)\mathbf{1}_{i=j}+\frac{r_\ell}{k}.

The diagonal therefore equals :math:`1-r_\ell+r_\ell/k`; an event can redraw
the current template. The probability of observing a *different* state after
the interval is :math:`r_\ell(k-1)/k`, not :math:`r_\ell` itself. In code,
the rho array must contain interval intensities, not probabilities that have
already been transformed.

.. literalinclude:: ../../../watchgen/mini_lshmm.py
   :language: python
   :pyobject: compute_recombination_probs

Emissions
=========

For the biallelic model in the paper, Appendix A defines

.. math::

   \widetilde\theta=\left(\sum_{a=1}^{n-1}\frac{1}{a}\right)^{-1},

where :math:`n` is the total number of haplotypes in the PAC sample. When the
conditional factor has :math:`k` templates, its mismatch probability is

.. math::

   \mu_k=\frac{1}{2}\frac{\widetilde\theta}{k+\widetilde\theta},

and the match probability is :math:`1-\mu_k`. The modern lshmm package
exposes a convenience estimator parameterized by reference-panel size. Its
indexing convention is reproduced exactly below but should not be confused
with substituting a new symbol into the paper's PAC formula.

.. literalinclude:: ../../../watchgen/mini_lshmm.py
   :language: python
   :pyobject: estimate_mutation_probability

For :math:`a>2` alleles, the teaching implementation treats mu as total
mismatch probability and distributes it uniformly over the :math:`a-1`
alternatives. Missing query alleles emit with probability one. A NONCOPY
panel entry emits with probability zero, as in the reference package.

.. literalinclude:: ../../../watchgen/mini_lshmm.py
   :language: python
   :pyobject: emission_matrix_haploid

The structured transition matrix is a diagonal matrix plus a rank-one matrix.
Consequently, if :math:`S_{\ell-1}=\sum_i\alpha_i(\ell-1)`, the forward
update is

.. math::

   \alpha_j(\ell)=e_j(s_\ell)\left[
   (1-r_\ell)\alpha_j(\ell-1)+\frac{r_\ell}{k}S_{\ell-1}\right].

Computing :math:`S_{\ell-1}` once makes the update :math:`O(k)`, rather than
forming an :math:`O(k^2)` transition matrix.
