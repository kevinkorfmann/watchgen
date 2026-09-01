.. _slim_overview:

========
Overview
========

SLiM evolves a population from past to present.  An Eidos script defines the
genome, mutation and recombination models, populations, life cycle, and event
callbacks.  The simulator supplies the optimized state engine.  This division
is important: a small Python Wright--Fisher loop cannot reproduce SLiM's
callback ordering, mutation-run representation, spatial interactions, or tree
sequence tables.

WF and nonWF are different models
=================================

In a WF model, generations do not overlap and subpopulation size is imposed.
Cached fitness is **relative reproductive success**: it weights which parents
produce the fixed number of offspring.  In the default hermaphroditic case,
the two parents are drawn independently, so incidental selfing is possible.

In a nonWF model, individuals may persist across ticks and population size is
emergent.  Fitness is **absolute survival** by default, although callbacks can
change both survival and reproduction.  Treating nonWF fitness as merely the
WF parent-sampling weight gives the wrong process.

The default WF tick
===================

The SLiM 5.2 manual gives this high-level order:

#. ``first()`` events;
#. ``early()`` events;
#. offspring generation, including migration choice, mate choice,
   recombination, mutation, and ``modifyChild()`` callbacks;
#. offspring replace the parental generation;
#. fixed mutations are processed (normally converted to substitutions);
#. ``late()`` events;
#. fitness values for the next tick are calculated; and
#. the tick/cycle counter advances.

Consequently, offspring generation uses fitness cached at the end of the
preceding tick.  A teaching function may calculate that cache immediately
before a transition, but should not claim that this is SLiM's literal event
order.

What tree-sequence recording retains
====================================

``initializeTreeSeq()`` records ancestry needed by extant haplosomes and
simplifies it periodically.  It is retained ancestry, not an ever-growing
record of every individual and every pedigree relationship.  Neutral mutations
may be overlaid afterward with compatible tskit/msprime/pyslim versions;
selected mutations must be present during the forward simulation when they
affect fitness.
