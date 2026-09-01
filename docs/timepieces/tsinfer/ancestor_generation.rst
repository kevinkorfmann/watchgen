Ancestor generation
===================

Site ordering is not dating
---------------------------

Let :math:`G_{js}\in\{0,1\}` be the allele carried by sample haplotype
:math:`s` at inference site :math:`j`, and let

.. math::

   f_j = \sum_s G_{js}

be the derived-allele count. The paper implementation used this count to order
sites. In 0.1.4, distinct counts were mapped to ordinal ranks: the smallest
distinct count received rank 1, the next rank 2, and so on. These ranks are a
time proxy and are **not generations**. Two counts can share no calibrated time
interpretation even when their order is correct.

Multiple focal sites
--------------------

The original implementation did not blindly make one ancestor per site. Sites with
the same count and exactly the same carrier vector were grouped, so one ancestor
could have **multiple focal sites**. Suppose focal sites :math:`a` and :math:`b`
share their carriers. An intervening site :math:`j` splits them when
:math:`f_j>f_a` and its alleles are polymorphic among the focal carriers. This
prevents one ancestor from spanning evidence inconsistent with a single lineage.

Extending a partial ancestor
----------------------------

At every focal site the ancestor carries allele 1. Between multiple focal sites,
an older intervening site's state is the majority among focal carriers; ties choose
1. The ancestor is then extended left and right.

For one direction, begin with carrier set :math:`S` at the focal site and threshold
:math:`\lfloor f/2\rfloor`. Each traversed site is initially assigned 0. At a site
with count greater than the focal count:

#. take the majority state among the current :math:`S` (ties choose 1);
#. apply the implementation's delayed removal buffer;
#. stop when at most :math:`\lfloor f/2\rfloor` carriers remain;
#. otherwise buffer carriers that disagree with the current consensus.

The delayed buffer is important source-level detail. The earlier chapter instead
kept the original carrier set fixed and stopped at the first older site; both
choices changed ancestor spans and states.

Two all-zero ancestors
----------------------

Version 0.1.4 prepended two full-length all-zero haplotypes. The older one was a
matching **virtual root** used to align identifiers; the younger was the all-zero
**ultimate ancestor**. They are distinct implementation objects, not two biological
ancestors inferred from independent evidence. Stable 0.4.1 retained this two-node
pattern but changed details of assigning their proxy times.

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: ancestor_descriptors

.. literalinclude:: ../../../watchgen/mini_tsinfer.py
   :language: python
   :pyobject: build_ancestor
