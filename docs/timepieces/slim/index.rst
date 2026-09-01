.. _slim_timepiece:

====================
Timepiece XVI: SLiM
====================

   *Forward simulation with explicit individuals, genomes, and selection*

SLiM is a programmable forward-time evolutionary simulator.  It supports
Wright--Fisher (WF) and non-Wright--Fisher (nonWF) life cycles, selection,
spatial and ecological interactions, and succinct tree-sequence recording.
The primary SLiM 3 paper introduced the general nonWF framework
:cite:`slim`; this chapter is checked against the **SLiM 5.2** source, manual,
and command-line program.

.. important:: Version and scope

   SLiM 5 changed several Eidos names used by older SLiM 3/4 recipes.  The
   executable recipes here use 5.2 names such as ``haplosomes``.  The Python
   module is a bounded **teaching model**, not a reimplementation of SLiM.  It
   covers four default WF mechanisms and deliberately leaves callbacks,
   nonWF survival, genomic elements, stacking policies, migration, and
   tree-sequence bookkeeping to the original software.

The four mechanisms are:

#. mutation **lineage identity**, which is not genomic-position identity;
#. default multiplicative mutation fitness;
#. independent, relative-fitness parent draws in a WF population; and
#. uniform-map breakpoint generation using SLiM's probability convention.

For production simulations, write and run an Eidos recipe with the versioned
SLiM executable.  Use the Python model only to inspect these mechanisms.

.. toctree::
   :maxdepth: 2

   overview
   wright_fisher
   recipes
   demo
