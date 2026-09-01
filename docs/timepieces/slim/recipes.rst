.. _slim_recipes:

=============================
Versioned SLiM 5.2 Recipes
=============================

These recipes are executable Eidos inputs, not Python substitutes.  They use
the current SLiM 5.2 API and are exercised by the chapter test suite.

Neutral WF population
=====================

.. literalinclude:: scripts/neutral.slim
   :language: c
   :caption: A small neutral WF simulation.

``initializeMutationRate()`` sets the mutation map;
``initializeRecombinationRate()`` sets adjacent-base breakpoint probabilities.
The genomic element covers positions 0 through 9999, inclusive.

Introduce a selected lineage
============================

.. literalinclude:: scripts/selected.slim
   :language: c
   :caption: Introduce one beneficial mutation lineage in tick 10.

SLiM 5 uses ``haplosomes`` where older recipes may show ``genomes``.  The
selected mutation is inserted in ``late()`` as a real mutation object.  It is
therefore present when fitness is cached for the next tick; inserting it in WF
``early()`` would be too late to affect that tick's offspring generation.  It
is not overlaid after simulation.

Enable tree-sequence recording
==============================

.. literalinclude:: scripts/tree_sequence.slim
   :language: c
   :caption: Enable succinct ancestry recording.

For an actual analysis, call ``sim.treeSeqOutput()`` with an output path, then
load the result with versions of pyslim and tskit compatible with that SLiM
release.  Recording tracks retained ancestry and SLiM state; it does not make a
hand-built list of nodes and edges a valid tree sequence.

Production checklist
====================

#. Pin the SLiM version and record ``slim -v``.
#. State whether the model is WF or nonWF.
#. Verify mutation, recombination, and genomic-element coordinate endpoints.
#. Treat mutation identity separately from position.
#. Run short recipes before scaling population size or duration.
#. If using tree sequences, pin compatible SLiM, pyslim, tskit, and msprime
   versions and validate the output tables.
