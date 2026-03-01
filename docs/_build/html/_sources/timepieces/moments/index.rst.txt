.. _moments_timepiece:

=====================================
Timepiece X: moments
=====================================

   *Inferring demographic history from the frequency spectrum -- without solving a single PDE.*

The Mechanism at a Glance
==========================

``moments`` is a method for **demographic inference** -- learning a population's
history (size changes, splits, migrations) from patterns in its DNA variation.
It works by computing how the **site frequency spectrum** (SFS) evolves through
time under different demographic scenarios, then finding the scenario that best
matches observed data.

If the other Timepieces in this book work by reconstructing genealogies (the tree
structure connecting samples to their ancestors), ``moments`` takes a fundamentally
different approach. It summarizes the genetic data into a compact statistic -- the
frequency spectrum -- and works entirely in this summary space. Think of it as
reading the time from the position of the hands without ever opening the case to
see the gears. The mechanism is different, but the information comes from the same
source: the coalescent process shaped by population history.

The key innovation: while its predecessor ``dadi`` solves a partial differential
equation (the Wright-Fisher diffusion) on a frequency grid, ``moments`` bypasses
this entirely. It derives **ordinary differential equations** that govern the SFS
entries directly. No grid, no PDE, no numerical diffusion artifacts. Just a clean
system of ODEs that you can solve with standard numerical methods.

.. admonition:: Primary Reference

   :cite:`moments`

The four gears of ``moments``:

1. **The Frequency Spectrum** (the dial) -- The summary statistic at the heart of
   everything: a histogram of allele frequencies across your sample. Different
   demographic histories produce different spectral signatures, and this is what
   ``moments`` reads.

2. **Moment Equations** (the escapement and gear train) -- The mathematical engine:
   a system of ODEs describing how each SFS entry changes under drift, mutation,
   selection, and migration. This is where coalescent theory and population genetics
   theory come together.

3. **Demographic Inference** (the mainspring) -- The optimization machinery: finding
   the demographic parameters that maximize the likelihood of the observed data.
   This is where the model meets reality.

4. **Linkage Disequilibrium** (a complication) -- A second lens: two-locus statistics
   that capture information invisible to the SFS alone, especially about recent
   admixture and recombination rate variation.

These gears mesh together into a complete inference framework:

.. code-block:: text

   Observed DNA variation
           |
           v
   +-----------------------+
   |  FREQUENCY SPECTRUM   |
   |                       |
   |  Summarize variation  |
   |  as allele frequency  |
   |  histogram (SFS)      |
   +-----------------------+
           |
           v
   +-----------------------+
   |  MOMENT EQUATIONS     |
   |                       |
   |  Compute expected SFS |
   |  under a demographic  |
   |  model (forward in    |
   |  time via ODEs)       |
   +-----------------------+
           |
           v
   +-----------------------+
   |  DEMOGRAPHIC          |
   |  INFERENCE            |
   |                       |
   |  Find parameters that |
   |  maximize P(data |    |
   |  model) via numerical |
   |  optimization         |
   +-----------------------+
           |
           v
   Demographic history:
   population sizes, split
   times, migration rates

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- the relationship between
     population size and genetic diversity
   - Basic familiarity with differential equations is helpful (we'll explain as we go)
   - No prior knowledge of the site frequency spectrum is needed -- we build it
     from scratch

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   the_frequency_spectrum
   moment_equations
   demographic_inference
   linkage_disequilibrium
   demo
