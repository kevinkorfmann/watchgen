.. _argweaver_mcmc:

==============
MCMC Sampling
==============

ARGweaver explores posterior uncertainty by repeatedly removing part of an ARG
and threading it back through the remainder. The hidden Markov model supplies a
distribution over complete attachment paths along the genome. A normalized
forward pass followed by stochastic traceback samples one such path; the ARG is
then rebuilt with that thread.

The phrase "ARGweaver uses Gibbs sampling" needs a qualification. Removing and
re-threading one external chromosome is a full-conditional Gibbs update and is
accepted automatically. The software also implements internal-branch and
subtree proposals. Those more general moves are used in a
Metropolis--Hastings sampler and require an acceptance correction, as described
in the original paper :cite:`argweaver` and implemented in
``src/argweaver/sample_arg.cpp``.

Sequential initialization
=========================

The program constructs an initial ARG sequentially. It begins with a small
coalescent genealogy and adds chromosomes one at a time. At each addition, the
current partial ARG fixes the HMM state spaces, transition objects, and emission
tables. Forward sampling generates the new chromosome's path, including any
recombination events required to move between attachment states.

Sequential construction is an initializer, not a claim that the first ARG is a
posterior draw from the full data set. Subsequent MCMC iterations revisit its
threads. Burn-in and convergence checks remain necessary because exact sampling
of one conditional path does not make successive complete ARGs independent.

External chromosome threading
==============================

For an external update, ARGweaver removes one sampled chromosome from every
local tree while preserving the induced partial ARG. Conditional on that
partial ARG and the sequence data, the removed chromosome's attachment path is
an HMM. Sampling exactly from this HMM is a Gibbs step:

.. math::

   z_k' \sim P(z_k \mid \mathcal{G}_{-k},D,\theta).

Here :math:`z_k` is the chromosome's genome-wide thread,
:math:`\mathcal{G}_{-k}` is the partial ARG, :math:`D` is the alignment, and
:math:`\theta` contains the population size, mutation rate, recombination rate,
and time grid. There is no accept/reject decision for this move because its
proposal distribution is the full conditional.

Forward sampling
================

Within a non-recombining block of the partial ARG, the forward recursion is

.. math::

   \alpha_x(s') \propto e_x(s')\sum_s \alpha_{x-1}(s)T_x(s,s').

ARGweaver normalizes each forward column for numerical stability. The
transition multiplication uses the compressed, time-grouped calculation from
:ref:`argweaver_transitions`. At a breakpoint in the partial ARG, it substitutes
a switch transition object that maps states across the known SPR.

Stochastic traceback first samples the final state from the last normalized
forward column. Moving backward, it samples each preceding state with weight

.. math::

   \alpha_{x-1}(s)T_x(s,s_x).

This produces a joint path sample. Independently sampling each site's marginal
posterior would not preserve the transition dependence and would not be a valid
thread.

Internal and subtree moves
==========================

Single-chromosome updates can mix slowly when several neighboring branches need
to move together. ARGweaver therefore supports threading an internal branch or
subtree. The state space is constrained by the removed subtree's age, the
emission calculation joins subtree and main-tree partial likelihoods, and the
transition object receives a minimum allowed attachment age.

These proposals do not all have the Gibbs guarantee of an external chromosome
update. The general sampler computes the target-density and proposal-density
terms needed for a Metropolis--Hastings decision. Consequently, statements that
all ARGweaver moves have 100 percent acceptance are false; that property applies
only to full-conditional threading moves.

Coalescent and recombination helpers
====================================

The miniature module contains small stochastic helpers for checking parameter
conventions. ``sample_tree`` simulates continuous coalescent waiting times under
a piecewise-constant diploid effective population size. With :math:`k` extant
lineages, its event rate is

.. math::

   \binom{k}{2}\frac{1}{2N_e}.

The result can subsequently be snapped to a discrete time grid, but the helper
itself is not ARGweaver's full initialization procedure.

.. literalinclude:: ../../../watchgen/mini_argweaver.py
   :language: python
   :pyobject: sample_tree

For a valid local tree of length :math:`L>0`, the distance to the next
recombination event is exponential with rate :math:`\rho L`. A zero-length tree
cannot generate such an event and is rejected rather than silently assigned an
invented rate.

.. literalinclude:: ../../../watchgen/mini_argweaver.py
   :language: python
   :pyobject: sample_next_recomb

What the toy loop is not
========================

``simplified_mcmc`` in the miniature module perturbs a list of coalescence times
and records a tree-length trajectory for plotting. It has no alignment, ARG,
threading HMM, posterior density, or valid full-conditional update. It is not an
ARGweaver sampler and cannot be used to assess acceptance rates, effective sample
size, posterior calibration, or convergence of the real method.

.. literalinclude:: ../../../watchgen/mini_argweaver.py
   :language: python
   :pyobject: simplified_mcmc

Validation boundaries
=====================

Appropriate tests for the small helpers include the :math:`2N_e` mean for a
pairwise coalescence, the :math:`1/(\rho L)` mean recombination distance, exact
Jukes--Cantor likelihood oracles, stochastic transition rows, and direct time-grid
parity with the upstream source. Validation of the complete sampler additionally
requires the official C++ transition, emission, traceback, and ARG-consistency
tests. A plausible toy trace is not evidence of posterior parity.

The resulting interpretation is deliberately bounded. The book explains how
ARGweaver's exact conditional HMM updates fit inside its MCMC machinery, while
keeping general subtree Metropolis--Hastings moves and production ARG bookkeeping
distinct from the miniature demonstrations.
