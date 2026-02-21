.. _msprime_timepiece:

====================================
Timepiece III: msprime
====================================

   *Simulating ancestral histories under the coalescent with recombination*

.. epigraph::

   "Efficient coalescent simulation and genealogical analysis for large sample sizes"

   -- Kelleher, Etheridge, and McVean (2016)

The Mechanism at a Glance
==========================

msprime is a **coalescent simulator**: given a sample size, genome length,
mutation rate, and recombination rate, it produces random ancestral histories
(genealogies) that are consistent with the evolutionary process. It works
**backwards in time**, starting from a set of sampled genomes in the present
and tracing their ancestry back until all lineages have found common ancestors.

While SINGER (Timepiece VI) *infers* an ARG from observed data, msprime
*generates* an ARG from a specified model. They are complementary tools, like a
watch and a watch-testing machine: msprime creates the ground truth that tools
like SINGER try to recover. Understanding how the simulator works gives you deep
insight into the coalescent process itself, and provides you with a reliable way
to test every other Timepiece in this book.

If PSMC is the simplest watch (two hands, one gear train), msprime is the
**master clockmaker's bench** -- the machine that produces the movements for
every other watch. Its output (tree sequences) is what inference methods consume,
and its internals (the coalescent with recombination, implemented with clever data
structures) reveal how nature's own clockwork operates.

The four gears of msprime:

1. **The Coalescent Process** (the escapement) -- The mathematical engine: how
   lineages find common ancestors backwards in time, and how recombination fragments
   the genome into independently-evolving segments. This is the coalescent from
   :ref:`The Workbench <coalescent_theory>` in action.

2. **Segments & the Fenwick Tree** (the mainspring) -- The data structures that make
   it fast: linked-list segments track which parts of the genome each lineage carries,
   and Fenwick trees enable :math:`O(\log n)` event scheduling. Clever engineering
   turns an elegant algorithm into a practical tool.

3. **Hudson's Algorithm** (the gear train) -- The main simulation loop: an event-driven
   machine that races recombination, coalescence, and migration against each other,
   always executing whichever happens first. This is where the coalescent process
   comes alive as executable code.

4. **Demographics & Mutations** (the case and dial) -- The outer layers: population
   size changes, migration, growth, and the final step of painting mutations onto the
   genealogy. These layers transform a simple simulator into a rich model of
   population history.

These gears mesh together into a complete simulator:

.. code-block:: text

   Parameters (n, L, mu, rho, demography)
                    |
                    v
   Initialize n lineages, each carrying [0, L)
                    |
                    v
         +---> Compute event rates
         |         |
         |         v
         |    Sample next event time (exponential)
         |         |
         |         v
         |    Execute event:
         |      Recombination? --> split a lineage
         |      Coalescence?   --> merge two lineages
         |      Migration?     --> move a lineage
         |      Demographic?   --> change population params
         |         |
         |         v
         |    Update data structures
         |         |
         +---------+
              (repeat until all positions coalesced)
                    |
                    v
         Output: tree sequence (tskit format)
                    |
                    v
         Add mutations (Poisson process on branches)
                    |
                    v
         Output: tree sequence with mutations

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- exponential waiting times,
     coalescence rates, the Poisson mutation model
   - :ref:`Ancestral Recombination Graphs <args>` -- the data structure msprime
     produces (marginal trees, tree sequences)

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   coalescent
   segments_and_fenwick
   hudson_algorithm
   demographics
   mutations
