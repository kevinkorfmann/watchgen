.. _slim_wright_fisher:

Four Wright--Fisher Mechanisms
==============================

The accompanying ``watchgen.mini_slim`` code isolates four mechanisms that
can be tested without pretending to reproduce the whole program.

Mutation identity
=================

A SLiM mutation is a lineage object.  If the **same mutation object** is present
on both haplosomes, the individual is homozygous for that mutation.  Two
independent recurrent mutations can share a position, selection coefficient,
and dominance coefficient and still be distinct heterozygous lineages.  Keying
mutations only by position therefore changes fitness and allele frequencies.

For a mutation with selection coefficient :math:`s` and dominance
:math:`h`, the default contribution is

.. math::

   w_m = \begin{cases}
      1+s, & \text{same lineage on both haplosomes},\\
      1+hs, & \text{lineage on one haplosome}.
   \end{cases}

Default mutation contributions multiply.  SLiM then prevents a nonpositive
computed fitness from becoming a negative reproductive weight.  Mutation and
fitness callbacks can replace this default calculation.

Relative-fitness parent sampling
================================

For cached nonnegative fitnesses :math:`w_1,\ldots,w_N`, the default WF parent
probability is

.. math::

   P(i)=\frac{w_i}{\sum_j w_j}.

The two default hermaphroditic parent choices are independent.  This is a
fixed-size reproduction rule, not the nonWF survival rule described in the
overview.

Recombination probability
=========================

For a uniform map, SLiM's value :math:`p` is the desired probability of a
breakpoint between adjacent bases and must lie in :math:`[0,0.5]`.  The source
reparameterizes each interval to a Poisson intensity

.. math::

   \lambda = -\log(1 - p),

In code, this is ``lambda = -log(1 - p)`` (implemented stably as
``-log1p(-p)``).

so :math:`1-e^{-\lambda}=p`.  A chromosome with integer positions
``0, ..., L - 1`` has ``L - 1`` possible breakpoint intervals.  Raw events are
located on the map, then sorted and duplicate breakpoint positions are
collapsed.  A breakpoint coordinate ``j`` lies immediately left of base
``j``.

The common approximation ``Poisson(r * L)`` is close only for small rates; it
uses the wrong number of intervals and is not exact at larger :math:`p`.

Mutation input
==============

For the teaching model's uniform map, the number of mutation events on one
haplosome is Poisson with mean :math:`\mu L`, and positions are sampled from
``0, ..., L - 1``.  Events at one position remain distinct lineages.  Real
SLiM additionally applies genomic-element types, mutation-type weights, and a
mutation stacking policy.

Bounded Python example
======================

.. code-block:: python

   from watchgen.mini_slim import Individual, Mutation, calculate_fitness

   lineage = Mutation(position=12, s=0.1, h=0.5)
   homozygote = Individual([lineage], [lineage])
   assert calculate_fitness(homozygote) == 1.1

   recurrent = Mutation(position=12, s=0.1, h=0.5)
   compound = Individual([lineage], [recurrent])
   assert calculate_fitness(compound) == 1.05 ** 2

The second genotype contains two independent mutations at the same base; it is
not homozygous for either lineage.
