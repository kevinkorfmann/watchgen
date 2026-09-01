.. _discoal_sweep_types:

=====================
Sweep models
=====================

The labels *hard*, *soft*, and *partial* describe different histories at the
selected locus. They cannot be reduced to different initial values of the same
logistic curve. discoal changes both the trajectory and the ancestral events
available during the sweep :cite:`discoal`.

Hard sweeps
===========

A hard sweep traces the adaptive class to one mutational origin. ``-wd`` uses the
deterministic production curve, while ``-ws`` uses the conditional stochastic
trajectory. ``-wn`` conditions a neutral allele on fixation. The sweep endpoint is
set by the time argument to these flags, the selected coordinate by ``-x``, and
the strength of genic selection by ``-a`` with :math:`\alpha=2Ns`.

At complete fixation, every sampled lineage begins the sweep phase on background
:math:`B`. Backward recombination may move neutral ancestry to :math:`b`. At the
single-copy origin, all remaining :math:`B` ancestry shares one ancestor. This
last statement applies to a single-origin hard sweep; it must not be reused as a
generic rule for recurrent adaptive mutation.

Standing variation
==================

With ``-f f0``, the allele was neutral below :math:`f_0` and became beneficial at
that frequency. Backward simulation therefore has a selected conditional phase
from the endpoint to :math:`f_0`, followed by a neutral conditional phase from
:math:`f_0` to the mutational origin. Recombination during the standing phase can
preserve several linked haplotypes even though the selected allele itself has a
single older origin. This is the soft-sweep mechanism from standing variation
:cite:`hermisson_pennings_2005`.

The old mini ended the structured phase at :math:`f_0` and merged all remaining
lineages in an ordinary neutral coalescent. That omitted the background-conditioned
standing phase. The ``selection_start_frequency`` option in
``mini_discoal.stochastic_trajectory`` now represents it explicitly.

Recurrent adaptive mutation
===========================

The ``-uA`` option allows lineages to change selective class through recurrent
mutation during the sweep. This can produce several independent adaptive origins,
the recurrent-mutation soft sweep described by Pennings and Hermisson
:cite:`pennings_hermisson_2006`. In the production event loop, the recurrent
mutation contribution depends on the number of beneficial-background ancestors
and on :math:`1/x`.

The previous chapter supplied a formula called an expected number of independent
origins, :math:`2N\mu_a\log(2Ns)/s`. That expression was neither implemented by
discoal nor a correctly scoped result from the cited soft-sweep theory, so it has
been removed. The probability and number of origins depend on the population
model, mutation scaling, conditioning, and sample. Production discoal should be
used when recurrent-origin ancestry is required; the revised single-locus mini
does not implement ``-uA``.

Partial sweeps
==============

``-c c`` specifies a sweep whose selected phase ended at frequency
:math:`0<c<1`. At the recent endpoint, active ancestors are assigned to
:math:`B` with probability :math:`c` and to :math:`b` otherwise. The program then
runs backward through the sweep trajectory. With ``-f``, discoal requires
:math:`f_0<c` and represents a partial sweep from standing variation.

A partial sweep is not necessarily a sweep that is still occurring at the moment
of sampling. The sweep event has its own time, and :math:`c` is the frequency at
which selection ended in the modeled event. Neutral evolution can occur between
that endpoint and the samples.

Recurrent hitchhiking and off-locus sweeps
==========================================

The ``-R`` option places recurrent sweeps within the simulated locus, while ``-L``
places recurrent sweeps to its left. Single off-locus events use ``-ld``, ``-ls``,
or ``-ln`` for deterministic, stochastic, or neutral fixation. For an off-locus
sweep, only the recombination distance to the region is required; this is the case
most closely represented by the mini's fixed-distance background-switch kernel.

The full feature surface also includes gene conversion, demographic size changes,
admixture, migration outside the active sweep population, and ancient samples.
These features interact through the production event scheduler. They should not
be inferred from the much smaller teaching functions.
