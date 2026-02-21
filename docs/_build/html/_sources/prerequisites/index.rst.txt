.. _prerequisites:

================================
The Workbench (Prerequisites)
================================

Before you can build a Timepiece, you need the right tools on your workbench.

A watchmaker's bench holds files, pliers, tweezers, loupes -- each one essential for
specific tasks, each one mastered individually before being used in combination. Our
workbench holds mathematical and biological concepts: the foundational ideas that every
algorithm in this book is built upon.

This section covers four topics. Each one is self-contained and builds from the ground
up -- we don't assume you've seen these ideas before. If you *have* encountered them,
the treatment here will sharpen your understanding and connect the concepts directly to
the algorithms we'll build later.

The suggested reading order is:

1. **Coalescent Theory** -- How to think about ancestry backwards in time. This is the
   biological foundation: the idea that the genealogical history of a sample can be
   described by a branching tree, and that the shape of this tree is governed by
   simple probabilistic rules. We'll introduce the exponential distribution, the
   Poisson process, and the fundamental connection between population size and
   coalescence time. *(This is our most important tool -- it goes into every Timepiece.)*

2. **Ancestral Recombination Graphs** -- When a single tree isn't enough. Recombination
   means that different parts of the genome can have different genealogical histories.
   An ARG captures this full picture: the complete history of a sample, including all
   the places where the tree changes. We'll explain what recombination is, why it
   complicates things, and how the tree sequence data structure makes the complexity
   manageable.

3. **Hidden Markov Models** -- The computational engine behind most methods. An HMM is
   a mathematical framework for inferring hidden information from noisy observations.
   We'll build one from scratch, implement the forward algorithm, and show how a
   clever trick (the Li-Stephens structure) makes it fast enough for genomic data. If
   you've never seen an HMM before, this chapter will give you everything you need.

4. **The Sequentially Markov Coalescent** -- Making the impossible tractable. The full
   coalescent with recombination is not Markov, which means we can't use HMMs directly.
   The SMC is an approximation that restores the Markov property at the cost of a tiny
   amount of accuracy. We'll explain exactly what's lost and why it doesn't matter much.

These four tools -- the coalescent, the ARG, the HMM, and the SMC -- are the essential
instruments on our workbench. Once you've mastered them, you'll be ready to build any
Timepiece in the collection.

.. admonition:: Do I need to read all of these?

   It depends on which Timepiece you want to build. **Coalescent Theory** is essential
   for everything. **HMMs** are needed for PSMC, SINGER, and ARGweaver. **ARGs** and
   the **SMC** are needed for SINGER and tsinfer. If you're not sure, start with
   Coalescent Theory and work through them in order -- each one builds naturally on
   the previous.

.. toctree::
   :maxdepth: 2

   coalescent_theory
   args
   hmms
   smc
