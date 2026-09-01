.. _discoal_structured_coalescent:

==========================================
The structured coalescent during a sweep
==========================================

Conditional on a selected-allele trajectory, ancestry is divided into the
beneficial background :math:`B` and the wild-type background :math:`b`. At
frequency :math:`x`, these backgrounds contain fractions :math:`x` and
:math:`1-x` of the population. This construction follows the hitchhiking
coalescent :cite:`braverman_1995` and is the central sweep kernel in discoal
:cite:`discoal`.

Coalescence rates
=================

For a diploid population of size :math:`N`, a pair of lineages on background
:math:`B` coalesces at rate :math:`1/(2Nx)` per generation. With :math:`n_B`
lineages, the total rate is

.. math::

   \lambda_B = \frac{\binom{n_B}{2}}{2Nx}.

Similarly,

.. math::

   \lambda_b = \frac{\binom{n_b}{2}}{2N(1-x)}.

Lineages on different backgrounds cannot coalesce at the selected site. As the
backward trajectory approaches the single-copy origin, the beneficial background
becomes small and its coalescence rate becomes large. For a hard sweep, any
beneficial-background lineages still present at the single mutational origin must
share that origin.

The function ``mini_discoal.coalescence_rate`` evaluates the per-generation rate
with an explicit background frequency. Its argument is a frequency, not a number
of chromosomes. This prevents the factor-of-:math:`2N` ambiguity present in the
old API.

A single linked locus
=====================

For one neutral locus at recombination fraction :math:`r` from the selected site,
background switching has total rates

.. math::

   M_{B\rightarrow b}=n_Br(1-x), \qquad
   M_{b\rightarrow B}=n_brx.

The factor :math:`1-x` appears because a recombination involving a
:math:`B` lineage changes its background only if the other parental chromosome is
wild type. The reverse event has probability :math:`x`.

``mini_discoal.migration_rates`` implements these expressions. The term
*migration* is useful for the two-background structured coalescent, but no
geographic movement is implied.

Why chromosome-scale recombination is different
================================================

The single-locus equations do not describe an in-locus crossover in the complete
simulator. A crossover can split one active ancestor into two ancestors covering
different site intervals. The segment containing the selected site retains the
parental selected background, while the other segment receives a background
according to :math:`x`. Both pieces remain in the ancestral recombination graph.
Gene conversion similarly changes ancestry over a tract.

The old mini replaced every crossover by a whole-lineage background switch and
then simulated sites independently. That loses correlations among marginal trees,
cannot produce an ARG, and is not parity with discoal. The revised mini therefore
labels its scope as a fixed-distance single-locus kernel.

Discrete event probabilities
============================

During a short interval :math:`\Delta g` generations, the four single-locus event
probabilities are the rates above multiplied by :math:`\Delta g`. In the C code,
the same calculation is expressed on the internal :math:`2N` clock. For example,
the beneficial-background coalescence contribution is

.. math::

   \frac{\binom{n_B}{2}}{x}\,\Delta t,

where :math:`\Delta t=\Delta g/(2N)`. The raw-generation and internal-time
expressions are identical after this substitution.

Production discoal accumulates no-event probability over the grid until an event
is accepted, then chooses the event in proportion to its current contribution.
The teaching function ``mini_discoal.structured_event_probabilities`` uses the
equivalent one-step probabilities. It refuses a grid on which their sum reaches
one, because silently allowing such a grid would violate the at-most-one-event
approximation.

``mini_discoal.structured_coalescent_sweep`` recomputes all rates after every
accepted event. The previous implementation carried an exponential residual
across changing rates, allowed several events while retaining stale lineage counts,
and then subtracted the full interval hazard again. Those operations did not
simulate either a continuous-time inhomogeneous chain or discoal's rejection loop.

Neutral bookends
================

If a sweep ended :math:`\tau` generations ago, the ordinary coalescent runs from
the present to :math:`\tau`. The structured sweep then runs backward through its
trajectory. Any remaining lineages continue in the older neutral population.
``mini_discoal.simulate_linked_locus_genealogy`` implements exactly these three
phases for one marginal locus and returns :math:`n-1` coalescence times.

The distinction between tree height and diversity is important. For two haploid
samples, neutral TMRCA has mean :math:`2N`, so mean pairwise diversity is
proportional to mean TMRCA. For larger samples, the sample MRCA and total branch
length are different quantities. The old chapter summed coalescence event times
and called the result diversity; the revised
``mini_discoal.pairwise_diversity_profile`` uses two samples and normalizes mean
TMRCA by the correct neutral expectation :math:`2N`.
