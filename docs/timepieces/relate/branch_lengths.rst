.. _relate_branch_lengths:

===============================
Estimating coalescence times
===============================

Relate dates fixed local-tree topologies, but it does not date every tree in
isolation. A branch is defined by its descendant sets at its lower and upper
coalescence events. Across adjacent trees, production Relate first associates
exactly equivalent branches and then accepts approximate associations when the
descendant-vector correlations at both ends exceed 0.9. Greedy matching ensures
that each branch is associated at most once. Mutation counts and genomic exposure
are pooled across an associated run :cite:`relate`.

Likelihood and prior
====================

Let :math:`m_b` be the pooled mutation count on branch :math:`b`, :math:`t_b` its
length in coalescent time, and :math:`\theta_b/2` the mutation rate summed over the
bases for which it persists. Relate uses

.. math::

   m_b\mid t_b\sim\operatorname{Poisson}(\theta_b t_b/2).

For a constant population size, write :math:`\tau_k` for the interval during
which :math:`k` lineages remain. In standard coalescent units,

.. math::

   \tau_k\sim\operatorname{Exponential}\!\left({k\choose2}\right),
   \qquad k=N,N-1,\ldots,2.

The ranked order of internal events and the vector of intervals determine all
node times and branch lengths. Relate's posterior is the product of these
coalescent densities and the branchwise Poisson probabilities.

Production MCMC moves
=====================

Production Relate samples more than independent node ages. With probability 0.8
it proposes swapping the rank of two topologically compatible coalescence events.
This changes six branches while retaining the interval vector. Otherwise it
chooses one :math:`\tau_k` and proposes a positive replacement from an exponential
distribution with mean equal to its current value; the Metropolis--Hastings ratio
includes the asymmetric proposal correction. Changing :math:`\tau_k` changes the
lengths of the :math:`k` branches then alive.

The paper implementation initializes a compatible event order with
:math:`N^2` swap proposals and initializes intervals by an EM calculation with the
event order fixed. For production sampling it uses at least
:math:`\max(10N,1000)` burn-in proposals and continues until every interval has
received at least 20 proposals and the resulting branch lengths are positive.
Posterior mean event ages are differenced to obtain an ultrametric tree.

What the mini samples
=====================

``mini_relate.sample_ranked_branch_lengths`` fixes one compatible event order and
updates only the interval vector. It evaluates the same constant-size
coalescent/Poisson target for that restricted state space and includes the exact
Hastings correction for its exponential proposal. It does not associate branches
between trees, swap event ranks, implement sample ages, or reproduce Relate's
adaptive stopping rule. Its output is therefore a teaching diagnostic, not a
Relate posterior sample.

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: log_branch_mutation_likelihood

.. literalinclude:: ../../../watchgen/mini_relate.py
   :language: python
   :pyobject: sample_ranked_branch_lengths
