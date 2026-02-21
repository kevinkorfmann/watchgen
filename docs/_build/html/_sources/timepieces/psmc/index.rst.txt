.. _psmc_timepiece:

====================================
Timepiece I: PSMC
====================================

   *The Pairwise Sequentially Markovian Coalescent*

.. epigraph::

   "Inference of human population history from individual whole-genome sequences"

   -- Li and Durbin (2011)

The Mechanism at a Glance
==========================

PSMC is an algorithm that infers **population size history** :math:`N(t)` from a
**single diploid genome**. It looks at the pattern of heterozygous and homozygous
sites along the genome and asks: what demographic history best explains this pattern?

.. math::

   \text{Input: } \text{One diploid genome} \quad \rightarrow \quad \text{a sequence of 0s and 1s (hom/het)}

.. math::

   \text{Output: } \hat{N}(t) \quad \text{(effective population size as a function of time)}

The key insight: at each position along the genome, the two copies of the chromosome
share a **most recent common ancestor (MRCA)**. The time to this MRCA -- the
**coalescence time** -- varies along the genome because of recombination. And the
distribution of coalescence times is shaped by the population size history.

- **More heterozygous sites** = the two copies diverged long ago = large :math:`N(t)`
  in the past (large populations take longer to coalesce)
- **Fewer heterozygous sites** = the two copies diverged recently = small :math:`N(t)`
  (bottleneck forced rapid coalescence)

PSMC reads these patterns with a Hidden Markov Model.

If the coalescent is the escapement -- the fundamental ticking mechanism -- then PSMC
is the simplest complete watch you can build around it: just two hands (two haplotypes),
a single gear train (the HMM), and a dial that reads out population size through time.

The four gears of PSMC:

1. **The Continuous-Time Model** (the escapement) -- The transition density :math:`q(t|s)` that
   describes how the coalescence time changes between adjacent positions, under
   variable population size :math:`N(t)`. This is where the coalescent theory from
   :ref:`The Workbench <prerequisites>` comes alive.

2. **Discretization** (the gear train) -- Converting the continuous coalescence time into discrete
   time intervals, with a transition matrix :math:`p_{kl}` suitable for an HMM.
   Continuous math becomes finite computation.

3. **The HMM and EM** (the mainspring) -- The complete Hidden Markov Model: hidden states are time
   intervals, observations are het/hom, and the Expectation-Maximization algorithm
   estimates :math:`N(t)`. This is the engine that iteratively refines our estimate.

4. **Decoding the Clock** (the case and dial) -- Posterior decoding, scaling to real units, bootstrapping,
   and interpreting the population size history. The final step: reading the watch.

These gears mesh together into a complete inference machine:

.. code-block:: text

   Diploid consensus sequence (het/hom along genome)
                      |
                      v
            +-----------------------+
            |  DISCRETIZE TIME      |
            |                       |
            |  Choose intervals     |
            |  [t_k, t_{k+1})      |
            |  Initialize lambda_k  |
            +-----------------------+
                      |
                      v
   +--------> BUILD HMM               |
   |          States: time intervals   |
   |          Emissions: P(het | t_k)  |
   |          Transitions: p_{kl}      |
   |                    |              |
   |                    v              |
   |          FORWARD-BACKWARD         |
   |                    |              |
   |                    v              |
   |          MAXIMIZE Q (update       |
   |            theta, rho, lambda_k)  |
   |                    |              |
   +-------- converged? NO             |
                        |              |
                       YES             |
                        |              |
                        v              |
              +-----------------------+
              |  DECODE & INTERPRET   |
              |                       |
              |  Posterior decoding    |
              |  Scale to real units  |
              |  Plot N(t) vs time    |
              +-----------------------+
                        |
                        v
                Population size history

Why Just Two Sequences?
========================

You might wonder: why would anyone analyze just two sequences when we have methods
like SINGER that handle many? Three reasons:

1. **Data efficiency**: A single high-coverage diploid genome contains millions of
   informative sites. Two haplotypes recombine enough times along a full chromosome
   to trace population size changes over hundreds of thousands of years.

2. **Computational simplicity**: With only two sequences, the marginal tree at each
   position is trivial -- just a single coalescence time. No tree topology to infer,
   no branching structure. Just a number: *when did these two copies last share an
   ancestor?* This makes PSMC the ideal first Timepiece -- the simplest watch you
   can build that actually tells useful time.

3. **Historical importance**: PSMC was published in 2011, when whole-genome data
   from single individuals was becoming available. It showed that a single genome
   contains a remarkable amount of information about the species' past.

The simplicity of PSMC makes it an ideal first Timepiece to understand,
because the PSMC transition density reappears as a gear inside SINGER's time sampling
step (Timepiece VI). We start here and build the complete watch around it.

.. admonition:: Prerequisites for this Timepiece

   Before starting PSMC, you should have worked through:

   - :ref:`Coalescent Theory <coalescent_theory>` -- the exponential distribution of
     coalescence times and the concept of coalescent time units
   - :ref:`Hidden Markov Models <hmms>` -- the forward-backward algorithm and scaling
   - :ref:`The SMC <smc>` -- the Markov approximation and the basic PSMC transition
     density for constant population size

   If you've read those chapters, you have all the tools you need. If some concepts
   are rusty, we'll point you back to the relevant sections as they come up.

Chapters
========

.. toctree::
   :maxdepth: 2

   overview
   continuous_model
   discretization
   hmm_inference
   decoding
