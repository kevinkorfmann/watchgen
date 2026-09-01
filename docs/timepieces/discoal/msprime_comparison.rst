.. _discoal_msprime_comparison:

====================
discoal and msprime
====================

msprime's ``SweepGenicSelection`` and discoal use the same broad construction: a
conditional selected-allele trajectory drives a two-background structured
coalescent :cite:`msprime2,discoal,coop_griffiths_2004`. They do not expose the same
model surface, and matching a few scaled parameters does not make two simulations
identical.

Parameter translation
=====================

For a locus of span :math:`L` with per-base mutation and crossover rates
:math:`\mu` and :math:`r`, the standard diploid scaling is

.. math::

   \theta=4N\mu L, \qquad \rho=4NrL, \qquad \alpha=2Ns.

``mini_discoal.msprime_to_discoal`` applies these conversions, and
``mini_discoal.discoal_to_msprime`` reverses them. The conversion is a units
translation, not a statistical-equivalence guarantee. In particular, discoal's
sample-size argument counts haploid chromosomes. ``msprime.sim_ancestry`` treats
an integer sample count as diploid individuals unless told otherwise, so a direct
comparison must use ``ploidy=1``. The revised converter returns ``samples=n`` and
``ploidy=1`` explicitly; the earlier version silently doubled the sample.

For a hard sweep, a matching msprime model also requires

.. code-block:: python

   sweep = msprime.SweepGenicSelection(
       position=selected_position,
       start_frequency=1 / (2 * N),
       end_frequency=1 - 1 / (2 * N),
       s=alpha / (2 * N),
       dt=1 / (40 * N),
   )

   ts = msprime.sim_ancestry(
       samples=n,
       ploidy=1,
       population_size=N,
       sequence_length=L,
       recombination_rate=r,
       model=[sweep, msprime.StandardCoalescent()],
       random_seed=123,
   )

The neutral coalescent after the sweep model is necessary because a sweep phase can
end before all ancestral material has coalesced. Mutations are added afterward
with ``msprime.sim_mutations``.

Shared mechanics
================

Both implementations use a conditional diffusion jump process and compete
coalescence with recombination on the two selected backgrounds. Both represent
ancestral material along a recombining sequence, rather than simulating marginal
sites independently. The msprime 1.0 paper reports its sweep implementation and
benchmarks against discoal :cite:`msprime2`.

Important differences
=====================

discoal supports deterministic sweeps (``-wd``), stochastic sweeps (``-ws``),
neutral fixation (``-wn``), standing variation (``-f``), recurrent adaptive
mutation (``-uA``), partial sweeps (``-c``), off-locus sweeps, recurrent
hitchhiking, and gene conversion during the sweep. It can combine its models with
stepwise size changes, multiple populations, population merges, admixture, and
ancient samples, subject to the restriction that selection occurs in population 0
and migration involving that population is disabled during the sweep.

msprime emits a tree sequence and composes its sweep phase with other ancestry
models. Its public sweep API allows an arbitrary selected position and start and
end frequencies; it is not restricted to the midpoint. Current official
documentation warns that sweeps with more than one population and population-size
changes during the sweep are not implemented. An elevated start frequency alone
does not reproduce discoal's standing-variation model, because discoal also follows
the preceding neutral conditional phase.

Representation
==============

Production discoal maintains active ancestral nodes and dynamically allocated
ancestry segments, then emits an ``ms``-compatible haplotype matrix or marginal
Newick trees. It is inaccurate to describe its internal state as merely an
:math:`n\times S` haplotype array. msprime records nodes and edges directly in
tskit tables and emits a tree sequence. Both programs build genealogical ancestry;
their storage layouts and output formats differ.

What parity means here
======================

Useful parity checks compare units, haploid sample counts, trajectory endpoints,
time-step scaling, and qualitative or statistical summaries over many replicates.
Seed-for-seed output equality is not expected because the simulators use different
random-number generators and event implementations. The teaching module is held
to a smaller target: its deterministic curve, conditional jump moments,
single-locus event rates, lineage conservation, and neutral limits must match the
source equations. It is not presented as a substitute executable.
