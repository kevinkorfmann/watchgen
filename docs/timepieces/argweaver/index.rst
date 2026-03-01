.. _argweaver_timepiece:

====================================
Timepiece V: ARGweaver
====================================

   *Bayesian Sampling of Ancestral Recombination Graphs via the Discrete SMC*

The Mechanism at a Glance
==========================

ARGweaver is a Bayesian method for sampling **Ancestral Recombination Graphs (ARGs)**
from their posterior distribution given observed sequence data. Like SINGER, it uses
an iterative threading algorithm -- but with a crucial difference: ARGweaver
**discretizes time** into a finite grid, making the HMM state space finite and enabling
exact forward-backward computation.

Where SINGER splits the problem into two HMMs (one for branches, one for times),
ARGweaver uses a **single HMM** whose states are (node, time-index) pairs.
This is more direct: each state specifies both *where* and *when* the new lineage
joins the partial tree.

If SINGER is a grand complication with a continuous-time escapement, ARGweaver is its
predecessor: a simpler design that discretizes the time axis into a finite number of
positions, like a digital watch that displays hours and minutes but not the continuous
sweep of a second hand. What it loses in time resolution, it gains in computational
simplicity -- the exact forward-backward algorithm works because the state space is
finite. Understanding ARGweaver is valuable both as an algorithm in its own right and
as context for appreciating SINGER's continuous-time innovations.

.. admonition:: Primary Reference

   :cite:`argweaver`

The five gears of ARGweaver:

1. **Time Discretization** (the escapement) -- A log-spaced time grid that makes the
   state space finite while concentrating resolution near the present, where most
   coalescent events happen.

2. **Transition Probabilities** (the gear train) -- The HMM transition matrix encoding
   recombination and re-coalescence under the discrete SMC. This is where the
   :ref:`SMC approximation <smc>` becomes a concrete computation.

3. **Emission Probabilities** (the mainspring) -- A parsimony-based likelihood that
   scores how well each state explains the observed data. Simpler than a full
   probabilistic model, but effective.

4. **MCMC Sampling** (the winding mechanism) -- Subtree re-threading with Gibbs
   updates: remove one chromosome, re-sample its path through the ARG using
   forward-backward, repeat. No Metropolis-Hastings acceptance step needed.

5. **Switch Transitions** -- Special transition matrices at recombination breakpoints
   in the partial ARG, where the local tree topology changes. These handle the
   complexity of trees that differ between adjacent positions.

These gears mesh together into a complete MCMC sampler:

.. code-block:: text

   Initialize ARG by threading haplotypes 1, 2, ..., n
                          |
                          v
            +---> Pick a chromosome to remove
            |              |
            |              v
            |     Remove its thread (prune from ARG)
            |              |
            |              v
            |     Re-thread using single HMM
            |     (forward algorithm + stochastic traceback)
            |              |
            |              v
            |     The new thread is automatically accepted
            |     (Gibbs sampling -- no MH step needed)
            |              |
            +--------------+
                   (repeat)

.. admonition:: Prerequisites for this Timepiece

   - :ref:`Coalescent Theory <coalescent_theory>` -- coalescence rates and times
   - :ref:`Ancestral Recombination Graphs <args>` -- the ARG data structure
   - :ref:`Hidden Markov Models <hmms>` -- forward-backward algorithm and
     stochastic traceback
   - :ref:`The SMC <smc>` -- the Markov approximation that makes HMM inference possible

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   time_discretization
   transition_probabilities
   emission_probabilities
   mcmc_sampling
   demo
