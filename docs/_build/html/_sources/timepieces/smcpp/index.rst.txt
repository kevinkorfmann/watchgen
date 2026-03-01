.. _smcpp_timepiece:

====================================
Timepiece II: SMC++
====================================

   *From two sequences to many -- demographic inference with the distinguished lineage.*

The Mechanism at a Glance
==========================

**SMC++** (Terhorst, Kamm & Song, 2017) extends PSMC from a single diploid genome
to **multiple unphased diploid genomes**. Where PSMC reads population size history
from two haplotypes -- one simple watch with two hands -- SMC++ adds more hands to
the dial without requiring phased data or full ARG inference. The result is sharper
resolution in the recent past, exactly where PSMC's two-sequence approach runs out
of steam.

The key insight is the **distinguished lineage**. Rather than tracking the full
genealogy of all samples (which would require exponentially many states), SMC++ singles
out one lineage and tracks how it relates to a *demographic background* of :math:`n - 1`
undistinguished lineages. The coalescence time of the distinguished lineage is hidden;
the presence or absence of the other lineages provides additional signal about population
size. This trick keeps the state space manageable while extracting far more information
than PSMC's two-haplotype approach.

If PSMC is a two-hand watch, SMC++ is a **chronograph** -- a complication that adds
sub-dials tracking multiple time measurements simultaneously. Each additional sample
genome is another sub-dial, providing independent readings of the same demographic
history. The distinguished lineage is the central seconds hand, and the undistinguished
lineages sweep around their own sub-dials, all driven by the same escapement (the
coalescent process under variable population size).

.. admonition:: Primary Reference

   :cite:`smcpp`

The four gears of SMC++:

1. **The Distinguished Lineage** (the escapement) -- The setup: one lineage is
   singled out, and its coalescence time :math:`T` is tracked as a hidden variable.
   The remaining :math:`n - 1` lineages form a demographic background that modifies the
   coalescence rate. This is where PSMC's two-lineage framework generalizes to many.

2. **The ODE System** (the gear train) -- A system of ordinary differential equations
   that tracks the probability :math:`p_j(t)` that :math:`j` undistinguished lineages
   remain at time :math:`t`. The matrix exponential of the rate matrix gives exact
   transition probabilities. This replaces PSMC's simple exponential coalescence with
   a richer model.

3. **The Continuous HMM** (the mainspring) -- A modified transition matrix built from
   the ODE rates, combined via composite likelihood across pairs of sites. Gradient-based
   optimization (L-BFGS or EM) estimates the piecewise-constant population size function
   :math:`\lambda(t)`. This is the inference engine.

4. **Population Splits** (a complication) -- Cross-population analysis: modified ODEs
   that track lineage counts before and after a population split, enabling joint
   estimation of :math:`N_A(t)`, :math:`N_B(t)`, and the split time :math:`T_{\text{split}}`.

These gears mesh together into a complete inference machine:

.. code-block:: text

   Multiple unphased diploid genomes
                    |
                    v
          +-------------------------+
          |  CHOOSE DISTINGUISHED   |
          |  LINEAGE                |
          |                         |
          |  Pair it with each of   |
          |  the n-1 undistinguished|
          |  lineages               |
          +-------------------------+
                    |
                    v
   +-------> SOLVE ODE SYSTEM
   |         p_j(t): probability j
   |         undistinguished lineages
   |         remain at time t
   |                  |
   |                  v
   |         BUILD HMM
   |         States: discretized T
   |         Emissions: P(data | T)
   |         Transitions: from ODE
   |                  |
   |                  v
   |         COMPOSITE LIKELIHOOD
   |         across all pairs
   |                  |
   |                  v
   |         OPTIMIZE (L-BFGS)
   |         update lambda_k
   |                  |
   |         Converged?
   |         NO ---+
   |
   YES
   |
   v
   Output: lambda_0, ..., lambda_n
   |
   v
   Scale to real units: N(t)

.. admonition:: Prerequisites for this Timepiece

   SMC++ builds directly on PSMC. Before starting, you should have worked through:

   - :ref:`PSMC <psmc_timepiece>` -- the transition density, discretization, and
     HMM inference for two sequences. SMC++ generalizes every gear in PSMC.
   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence rates with multiple
     lineages, the relationship between population size and coalescence time
   - :ref:`The SMC <smc>` -- the sequential Markov coalescent approximation

   If you have built PSMC, you have most of the tools you need. SMC++ adds the
   multi-lineage generalization, but the underlying mathematical framework is the same.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   distinguished_lineage
   ode_system
   continuous_hmm
   population_splits
   demo

Each chapter derives the math, explains the intuition, implements the code,
and verifies it works. By the end, you'll have built a complete multi-sample
demographic inference engine -- and you'll see how PSMC's simple watch grows
into a chronograph.
