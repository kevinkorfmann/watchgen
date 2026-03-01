.. _slim_timepiece:

====================================
Timepiece XVI: SLiM
====================================

   *Forward-time population genetics simulation with natural selection*

The Mechanism at a Glance
==========================

SLiM is a **forward-time population genetics simulator**: given a population
size, genome length, mutation rate, recombination rate, and -- crucially -- a
model of **natural selection**, it evolves a population generation by generation
from past to present. Where msprime (Timepiece IV) works *backwards* from
sampled genomes to their common ancestors, SLiM works *forwards*, tracking
every individual, every mutation, and every fitness effect as they unfold.

If msprime is the master clockmaker's bench -- a machine that produces
genealogies by tracing ancestry backwards -- then SLiM is the **forge**:
a furnace that heats raw alloy, hammers it through selection, and produces
the finished timepiece by brute force. It is slower than the bench (forward
simulation is inherently more expensive than coalescent simulation), but it
can build watches that the bench cannot: watches with **natural selection**,
**complex fitness landscapes**, **spatial structure**, and **ecological
interactions**. The coalescent cannot model selection easily. SLiM can model
almost anything.

SLiM is a massive project (Haller & Messer, 2019) with its own scripting
language (Eidos), a graphical interface, and support for both Wright-Fisher
and non-Wright-Fisher life cycles. We will not attempt to cover all of it.
Instead, we disassemble only the **core mechanism** -- the Wright-Fisher
generation cycle -- and then build a few **recipes** that demonstrate the
key ideas: a selective sweep, background selection, and tree-sequence recording.

The three gears of our simplified SLiM:

1. **The Wright-Fisher Cycle** (the escapement) -- The discrete-generation
   engine: in each tick, parents are selected with probability proportional
   to fitness, offspring are generated through recombination, and new mutations
   are added. This is the heartbeat of the simulation.

2. **Fitness and Selection** (the mainspring) -- The force that drives
   evolution: each mutation carries a selection coefficient :math:`s` and a
   dominance coefficient :math:`h`. An individual's fitness is the product
   of the effects of all its mutations. Parents are drawn in proportion to
   their fitness -- this is how selection acts.

3. **Recipes** (the complications) -- Practical applications: a selective
   sweep, background selection, and tree-sequence recording. These show how
   the core mechanism produces the phenomena that population geneticists
   study.

These gears mesh together into a complete forward simulator:

.. code-block:: text

   Parameters (N, L, mu, rho, fitness model)
                    |
                    v
   Initialize N individuals, each with two haplosomes
                    |
                    v
         +---> RECALCULATE FITNESS
         |      w_i = product of (1 + h*s) or (1 + s) for all mutations
         |         |
         |         v
         |    SELECT PARENTS (proportional to fitness)
         |         |
         |         v
         |    GENERATE OFFSPRING:
         |      - Recombine parental haplosomes (Poisson breakpoints)
         |      - Add new mutations (Poisson along genome)
         |         |
         |         v
         |    OFFSPRING REPLACE PARENTS (non-overlapping generations)
         |         |
         +---------+
              (repeat for T generations)
                    |
                    v
         Output: final population
         (optionally: tree sequence recording the full genealogy)

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- to understand what SLiM's
     output looks like from the backward perspective (and to appreciate why
     forward simulation is necessary when selection is involved)
   - :ref:`msprime <msprime_timepiece>` -- the backward-time counterpart; SLiM
     and msprime are complementary tools, and SLiM can even output tree
     sequences that msprime/tskit can read

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   wright_fisher
   recipes
