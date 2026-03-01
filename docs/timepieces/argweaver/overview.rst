.. _argweaver_overview:

======================
Overview of ARGweaver
======================

   *Before you can reassemble the watch, you must understand what every gear does ---
   and why this particular watchmaker chose to cut them the way he did.*

.. figure:: /_static/figures/fig_mini_argweaver.png
   :width: 100%
   :align: center

   **ARGweaver at a glance.** The four core components of the discrete SMC
   machinery: time discretisation converting continuous coalescent time into
   a finite grid, transition probabilities between time intervals at
   recombination breakpoints, re-coalescence distributions governing where
   a detached lineage re-attaches, and MCMC tree sampling behaviour showing
   convergence of the Gibbs sampler.

If PSMC (see :ref:`coalescent_theory`) is an analog wristwatch --- elegant, continuous,
but limited to reading the time for a single pair of chromosomes --- then ARGweaver is
a **digital watch**. It chops continuous time into discrete ticks, displays everything
on a finite readout, and in exchange gains the ability to track *all* your chromosomes
simultaneously. The time grid is the tick marks on the dial; the HMM is the quartz
oscillator; and Gibbs sampling is the mechanism that keeps it all running.

This chapter gives you the complete blueprint before we manufacture any individual gear.
If you have not yet read the prerequisite chapters on the coalescent
(:ref:`coalescent_theory`), Hidden Markov Models (:ref:`hmms`), the Sequentially Markov
Coalescent (:ref:`smc`), and Ancestral Recombination Graphs (:ref:`args`), now is the
time. ARGweaver builds on all four.

What Does ARGweaver Do?
========================

Given a set of :math:`n` aligned DNA sequences (haplotypes) and a set of model
parameters, ARGweaver produces **samples from the posterior distribution** of
Ancestral Recombination Graphs:

.. math::

   \text{Input: } \mathbf{D} = \{d_1, d_2, \ldots, d_n\} \quad \text{(observed haplotypes)}

.. math::

   \text{Output: } \mathcal{G}^{(1)}, \mathcal{G}^{(2)}, \ldots, \mathcal{G}^{(M)} \sim P(\mathcal{G} \mid \mathbf{D})

Each sampled ARG :math:`\mathcal{G}^{(m)}` is a full genealogical history: a sequence
of local trees along the genome connected by SPR (Subtree Pruning and Regrafting)
operations at recombination breakpoints, with coalescence times on every internal node.

From these posterior samples, you can compute:

- **Pairwise coalescence times** between any two samples at any genomic position
- **Allele ages** --- when mutations first arose
- **Local effective population size** :math:`N_e(t)` through time
- **Recombination rate estimates** across the genome
- **Signatures of natural selection** via distortions in the local genealogies

.. admonition:: Probability Aside --- What is a posterior distribution?

   If you are new to Bayesian inference, the posterior :math:`P(\mathcal{G} \mid \mathbf{D})`
   is the probability of a genealogy *given the data we observed*. It combines two
   ingredients via Bayes' theorem:

   .. math::

      P(\mathcal{G} \mid \mathbf{D}) \propto
      \underbrace{P(\mathbf{D} \mid \mathcal{G})}_{\text{likelihood: how well does this ARG explain the mutations?}}
      \;\cdot\;
      \underbrace{P(\mathcal{G})}_{\text{prior: how likely is this ARG under the coalescent?}}

   ARGweaver does not compute this probability directly (the space of ARGs is far too
   vast). Instead, it *samples* from this distribution using MCMC, which we will meet
   in :ref:`argweaver_mcmc`. Each sample is one plausible genealogical history; by
   collecting many samples, we can estimate any quantity of interest (e.g., the mean
   coalescence time at a locus) by simply averaging over the samples.

The Threading Idea
===================

ARGweaver's core algorithm is the same as SINGER's: build the ARG **one haplotype at
a time**. This is called **threading** (or **sequential sampling**).

Imagine you have already built an ARG for the first :math:`n-1` haplotypes. To add
the :math:`n`-th haplotype, you need to decide, at each position along the genome:

1. **Which branch** of the current local tree does the new lineage coalesce with?
2. **At what time** does it coalesce?

The key insight is that these decisions form a **Hidden Markov Model** along the genome:
the hidden state at each position specifies the branch and time of coalescence, and
the state transitions are governed by the SMC (Sequentially Markov Coalescent) process.

If the HMM machinery feels unfamiliar, revisit :ref:`hmms` --- the forward algorithm,
stochastic traceback, and the relationship between hidden states and observations are
all essential here. If the SMC process is new, see :ref:`smc` for how recombination
creates a Markov chain of local trees along the genome.

.. code-block:: python

   # Pseudocode for ARGweaver initialization
   def initialize_arg(haplotypes, times, Ne, rho, mu):
       """Build an initial ARG by threading haplotypes one at a time."""
       n = len(haplotypes)

       # Start with the first two haplotypes (sample a pairwise coalescence)
       arg = create_pairwise_arg(haplotypes[0], haplotypes[1], times, Ne)

       # Thread remaining haplotypes one by one
       for i in range(2, n):
           # Remove haplotype i's thread (it has no thread yet; this is init)
           partial_arg = arg  # ARG for haplotypes 0..i-1

           # Build the single HMM:
           #   States: (node, time_index) pairs at each genomic position
           #   Transitions: DSMC recombination + re-coalescence
           #   Emissions: parsimony-based mutation likelihood
           hmm = build_hmm(partial_arg, haplotypes[i], times, Ne, rho, mu)

           # Run forward algorithm, then stochastic traceback
           # (see :ref:`hmms` for how forward-traceback sampling works)
           thread = sample_thread(hmm)

           # Graft the new haplotype into the ARG along this thread
           arg = add_thread_to_arg(partial_arg, haplotypes[i], thread)

       return arg

*Recap so far:* ARGweaver takes aligned sequences and produces sampled ARGs. It does
this by threading one haplotype at a time into a growing ARG using an HMM whose states
encode where and when the new lineage joins the existing genealogy.

The Discrete SMC (DSMC)
========================

ARGweaver models genealogical history using the **Discrete Sequentially Markov
Coalescent** (DSMC). This is the standard SMC (Wiuf and Hein, 1999; McVean and
Cardin, 2005) but with time discretized onto a finite grid.

In the continuous SMC, a recombination event at position :math:`s` breaks the
genealogy: the ancestral lineage above the recombination point detaches and must
re-coalesce with the remaining tree. The SMC approximation says that re-coalescence
can happen with any lineage in the local tree (not just those actually ancestral at
that point), which makes the process Markov along the genome.

The DSMC adds one more approximation: coalescence times are restricted to a finite
set of time points :math:`t_0 < t_1 < \cdots < t_{n_t}`. This makes the HMM state
space finite:

.. math::

   \text{States} = \{(b, i) : b \text{ is a branch in the local tree}, \;
   t_i \in [t_{\text{bottom}(b)}, \, t_{\text{top}(b)})\}

where :math:`t_{\text{bottom}(b)}` is the age of the child node of branch :math:`b`
and :math:`t_{\text{top}(b)}` is the age of the parent node.

.. admonition:: Why discretize?

   Continuous-time models (like SINGER's) require separate treatment of topology
   and timing, or adaptive quadrature over the time axis. Discretization makes
   everything finite: the transition matrix has a fixed size per tree, and the
   forward--backward algorithm runs in :math:`O(S^2)` per site, where :math:`S`
   is the number of states. The cost is a small approximation error controlled by
   the number of time points.

.. admonition:: Calculus Aside --- Why does discretization make the problem finite?

   In continuous time, the coalescence time :math:`T` is a real number in
   :math:`(0, \infty)`. The HMM hidden state is then :math:`(b, T)`, where :math:`b`
   is one of finitely many branches but :math:`T` takes uncountably many values. You
   cannot write down a transition *matrix* for an uncountable state space --- you need
   a transition *kernel* (a function :math:`K((b,T) \to (b',T'))` that you must
   integrate over). The forward algorithm becomes an integral equation rather than a
   matrix--vector product.

   By snapping :math:`T` to a finite grid :math:`\{t_0, t_1, \ldots, t_{n_t}\}`, the
   state space shrinks to (number of branches) :math:`\times` (number of time points),
   which is finite. The transition kernel becomes an ordinary matrix. The forward
   algorithm becomes a sequence of matrix--vector multiplications, which any computer
   can execute exactly (up to floating-point error). This is the fundamental tradeoff
   of the digital watch: you lose the smooth sweep of continuous time, but you gain a
   mechanism that can be computed exactly and efficiently.

Think of the time grid as **the tick marks on the watch dial**. A continuous-time model
lets the hands point anywhere; a discrete-time model forces them to snap to the nearest
tick. With enough ticks (ARGweaver defaults to 20), the approximation error is small,
and the computational machinery becomes dramatically simpler.

Terminology
============

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Term
     - Definition
   * - **Partial ARG**
     - The ARG for :math:`n-1` haplotypes, before threading the :math:`n`-th
   * - **Thread**
     - The path of the new lineage through the partial ARG: a sequence of (branch, time) states along the genome
   * - **Local tree**
     - The marginal genealogical tree at a single genomic position
   * - **SPR**
     - Subtree Pruning and Regrafting: the topological operation connecting adjacent local trees at a recombination breakpoint
   * - **State** :math:`(b, i)`
     - A branch :math:`b` in the local tree and a time index :math:`i` specifying where the new lineage coalesces
   * - **Time grid**
     - The set of discretized time points :math:`t_0, t_1, \ldots, t_{n_t}`; log-spaced to concentrate near the present
   * - **Coal time**
     - The geometric-mean midpoint :math:`t_{\text{mid},i} = \sqrt{(t_i + 1)(t_{i+1} + 1)} - 1` used as the representative time within interval :math:`i`
   * - **nbranches[i]**
     - Number of lineages (branches) passing through time interval :math:`[t_i, t_{i+1})`
   * - **nrecombs[i]**
     - Number of points where recombination can occur in interval :math:`i`
   * - **ncoals[i]**
     - Number of points where re-coalescence can occur at time :math:`t_i`

The Single-HMM Architecture
=============================

Unlike SINGER, which uses two HMMs (one for branches, one for times), ARGweaver
uses a **single HMM** with joint (branch, time) states. This is possible because
time discretization makes the joint state space finite.

**The single HMM:**

- **Hidden states**: :math:`(b, i)` = branch :math:`b` at time index :math:`i`
- **Observations**: allelic states (A/C/G/T) at each genomic position
- **Transitions**: probability of moving from state :math:`(b, i)` at position :math:`s`
  to state :math:`(b', j)` at position :math:`s+1`, determined by the DSMC
- **Emissions**: probability of the observed base given that the new lineage joins
  at state :math:`(b, i)`, computed using parsimony

At each position, the number of states is:

.. math::

   S = \sum_{b \in \text{branches}} |\{i : t_i \in [t_{\text{bottom}(b)}, \, t_{\text{top}(b)})\}|

For :math:`k` lineages and :math:`n_t` time points, this is roughly :math:`O(k \cdot n_t)`.

.. admonition:: One HMM vs. two HMMs

   ARGweaver's single-HMM approach is **exact** given the DSMC model (no decoupling
   approximation), but requires :math:`O(S^2)` work per site for transitions.
   SINGER's two-HMM approach is an approximation but scales better to large sample
   sizes because each HMM has fewer states. The tradeoff: ARGweaver is more accurate
   for small :math:`n`, SINGER scales to thousands of haplotypes.

.. admonition:: Probability Aside --- Why does a joint state space need :math:`O(S^2)` per site?

   The forward algorithm at each position computes
   :math:`\alpha_s(j) = \sum_i \alpha_{s-1}(i) \cdot T(i,j)` for every destination
   state :math:`j`. If there are :math:`S` states, this is a matrix--vector product
   costing :math:`O(S^2)`. Splitting the state into two independent HMMs (as SINGER
   does) reduces each HMM's state count: if one has :math:`S_1` states and the other
   :math:`S_2`, the cost is :math:`O(S_1^2 + S_2^2)` instead of
   :math:`O((S_1 \cdot S_2)^2)`. The catch is that the two HMMs are not truly
   independent, so the split introduces an approximation. ARGweaver avoids this
   approximation at the cost of a larger state space.

.. admonition:: Switch transitions

   When the local tree changes (at an existing recombination breakpoint in the partial
   ARG), the set of branches changes. ARGweaver uses special **switch transition
   matrices** at these positions, which map states in the old tree to states in the
   new tree. These are mostly deterministic (a branch in the old tree maps to the
   same branch in the new tree) with probabilistic components only for the branches
   directly involved in the SPR. See :ref:`argweaver_transitions` for details.

Parameters
===========

ARGweaver takes the following parameters:

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Parameter
     - Symbol
     - Meaning
   * - Mutation rate
     - :math:`\mu`
     - Per-base, per-generation mutation rate
   * - Recombination rate
     - :math:`\rho`
     - Per-base, per-generation recombination rate
   * - Effective pop. size
     - :math:`N_e`
     - Effective population size (can vary over time: :math:`N_e(t)`)
   * - Number of times
     - :math:`n_t`
     - Number of discrete time points (default: 20)
   * - Max time
     - :math:`T_{\max}`
     - Maximum time in generations (default: 160,000)
   * - Delta
     - :math:`\delta`
     - Controls log-spacing of the time grid (default: 0.01)

*Recap:* The model parameters are familiar from PSMC and SINGER --- mutation rate,
recombination rate, and population size. The new parameters (:math:`n_t`,
:math:`T_{\max}`, :math:`\delta`) control the time discretization, which is unique
to ARGweaver's digital-watch design. We will see exactly how these shape the time grid
in :ref:`argweaver_time_discretization`.

The Flow in Detail
===================

Here is a detailed view of how the pieces connect during one MCMC iteration:

.. code-block:: text

   Current ARG (n haplotypes)
           |
           v
   Choose chromosome k to remove
           |
           v
   +--------------------------+
   |  REMOVE THREAD           |
   |                          |
   |  Delete k's lineage from |
   |  the ARG, yielding a     |
   |  partial ARG for n-1     |
   |  haplotypes              |
   +--------------------------+
           |
           | Partial ARG + sequence data for k
           v
   +--------------------------+
   |  BUILD HMM               |
   |                          |
   |  For each position s:    |
   |    States: (branch, time)|
   |    Transition: DSMC      |
   |      (normal or switch)  |
   |    Emission: parsimony   |
   |      likelihood          |
   +--------------------------+
           |
           v
   +--------------------------+
   |  FORWARD ALGORITHM       |
   |                          |
   |  Compute forward probs   |
   |  alpha[s][(b,i)] for     |
   |  all positions            |
   +--------------------------+
           |
           v
   +--------------------------+
   |  STOCHASTIC TRACEBACK    |
   |                          |
   |  Sample a path of states |
   |  from right to left,     |
   |  proportional to forward |
   |  * backward probs        |
   +--------------------------+
           |
           | New thread: sequence of (branch, time) states
           v
   +--------------------------+
   |  ADD THREAD TO ARG       |
   |                          |
   |  Graft haplotype k back  |
   |  into the ARG along the  |
   |  sampled thread           |
   +--------------------------+
           |
           v
       Updated ARG

.. admonition:: Closing the confusion gap --- What is Gibbs sampling?

   The diagram above describes one iteration of **Gibbs sampling**, a particular
   MCMC strategy. The idea is deceptively simple: instead of trying to sample the
   entire ARG at once (impossible --- the space is too large), we update one piece
   at a time while holding everything else fixed.

   Think of it as **systematically removing and replacing each gear** in a watch.
   You take out gear :math:`k`, examine the space it occupied (the partial ARG), and
   then manufacture a new gear that fits perfectly into that space (by sampling from
   the conditional posterior using the HMM). Because you are sampling from the *exact*
   conditional distribution, the new gear is guaranteed to be a valid replacement ---
   no accept/reject step is needed.

   After cycling through all gears (all chromosomes), the watch is in a new
   configuration that is a valid sample from the posterior. Repeating this process
   many times yields a collection of posterior samples. For more on Gibbs sampling
   and why it converges to the correct distribution, see :ref:`argweaver_mcmc`.

Ready to Build
===============

You now have the high-level blueprint. In the following chapters, we build each gear
from scratch:

1. :ref:`argweaver_time_discretization` --- The time grid that makes it all finite
2. :ref:`argweaver_transitions` --- The HMM transition matrix
3. :ref:`argweaver_emissions` --- Scoring states against the data
4. :ref:`argweaver_mcmc` --- The MCMC sampling loop

Each chapter derives the math, explains the intuition, implements the code, and
verifies it works.

*Where we stand:* You now know *what* ARGweaver does (samples ARGs from the posterior),
*how* it does it at a high level (threading + Gibbs sampling over an HMM), and *why*
it discretizes time (to make the state space finite and the HMM exact). In the next
chapter, we will forge the first gear: the time grid itself --- the tick marks on the
digital watch's dial.

Let's start with the foundational gear: Time Discretization.
