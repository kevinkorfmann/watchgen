.. _argweaver_mcmc:

==============
MCMC Sampling
==============

   *The mainspring: wind it up, let it tick, and each tick produces a new
   sample from the posterior --- a complete genealogical history of your data.*

This chapter describes how ARGweaver's gears --- time discretization, transitions,
and emissions --- mesh together into a complete MCMC (Markov Chain Monte Carlo)
sampler. The MCMC explores the space of ARGs by repeatedly removing one chromosome's
thread and re-sampling it from the conditional posterior.

We have now forged every individual gear:

- :ref:`argweaver_time_discretization` --- the tick marks on the dial
- :ref:`argweaver_transitions` --- the gear train that advances the hidden state
- :ref:`argweaver_emissions` --- the escapement that couples the mechanism to data

In this chapter, we assemble them into a working watch and wind the mainspring.

The Gibbs Sampling Strategy
=============================

ARGweaver uses **Gibbs sampling** (a special case of MCMC) rather than the
Metropolis-Hastings framework used by SINGER's SGPR. The difference is fundamental:

- **Metropolis-Hastings** (SINGER): propose a new state, compute an acceptance ratio,
  accept or reject.
- **Gibbs sampling** (ARGweaver): sample the new state *exactly* from the conditional
  posterior. No accept/reject step needed --- every proposal is accepted.

This is possible because the discrete-time HMM allows exact computation of the
conditional posterior :math:`P(\text{thread}_k \mid \text{ARG}_{-k}, \mathbf{D})`
via the forward--backward algorithm.

.. admonition:: Closing the confusion gap --- What is Gibbs sampling?

   Gibbs sampling is a strategy for sampling from a joint probability distribution
   :math:`P(x_1, x_2, \ldots, x_n)` when the joint distribution is too complex to
   sample from directly, but the *conditional* distributions
   :math:`P(x_k \mid x_1, \ldots, x_{k-1}, x_{k+1}, \ldots, x_n)` are tractable.

   The algorithm is:

   1. Start with some initial values :math:`(x_1^{(0)}, x_2^{(0)}, \ldots, x_n^{(0)})`
   2. For each iteration :math:`t`:

      a. Sample :math:`x_1^{(t)} \sim P(x_1 \mid x_2^{(t-1)}, x_3^{(t-1)}, \ldots, x_n^{(t-1)})`
      b. Sample :math:`x_2^{(t)} \sim P(x_2 \mid x_1^{(t)}, x_3^{(t-1)}, \ldots, x_n^{(t-1)})`
      c. ...and so on for all variables.

   3. After many iterations, the samples :math:`(x_1^{(t)}, \ldots, x_n^{(t)})` are
      drawn from (approximately) the joint distribution :math:`P(x_1, \ldots, x_n)`.

   In ARGweaver, the "variables" are the threads of the :math:`n` chromosomes. At
   each iteration, one thread :math:`x_k` is removed and re-sampled from its
   conditional posterior :math:`P(\text{thread}_k \mid \text{all other threads}, \text{data})`.
   This conditional is exactly the posterior of an HMM --- the forward algorithm
   computes it, and stochastic traceback produces a sample.

   The watch metaphor captures this perfectly: Gibbs sampling is **systematically
   removing and replacing each gear**. You pull out one gear (remove a chromosome's
   thread), examine the space it left (the partial ARG), manufacture a new gear that
   fits exactly (sample from the conditional posterior via the HMM), and insert it.
   After cycling through all gears, the watch is in a new valid configuration.

.. admonition:: Why Gibbs works here

   The conditional distribution of one chromosome's thread, given the rest of the ARG
   and the data, is exactly the posterior of an HMM with known parameters. The forward
   algorithm computes this posterior in :math:`O(L \cdot S^2)` time, and stochastic
   traceback produces an exact sample. This is the same as "sampling from the full
   conditional" in Gibbs sampling theory.

   The guarantee: if you cycle through all chromosomes and re-sample each one's thread,
   the stationary distribution of the Markov chain is the joint posterior
   :math:`P(\mathcal{G} \mid \mathbf{D})`.

.. admonition:: Probability Aside --- Why Gibbs converges to the correct distribution

   Gibbs sampling satisfies **detailed balance** with respect to the target
   distribution :math:`\pi(\mathbf{x}) = P(\mathcal{G} \mid \mathbf{D})`. For a
   single-variable update of :math:`x_k`, the transition probability from
   :math:`\mathbf{x}` to :math:`\mathbf{x}'` (which differs only in coordinate
   :math:`k`) is :math:`T(\mathbf{x} \to \mathbf{x}') = P(x_k' \mid \mathbf{x}_{-k})`.
   Detailed balance requires :math:`\pi(\mathbf{x}) T(\mathbf{x} \to \mathbf{x}') = \pi(\mathbf{x}') T(\mathbf{x}' \to \mathbf{x})`.
   Since :math:`T(\mathbf{x} \to \mathbf{x}') = P(x_k' \mid \mathbf{x}_{-k}) = \pi(\mathbf{x}') / \pi(\mathbf{x}_{-k})`
   and :math:`T(\mathbf{x}' \to \mathbf{x}) = P(x_k \mid \mathbf{x}_{-k}) = \pi(\mathbf{x}) / \pi(\mathbf{x}_{-k})`,
   both sides equal :math:`\pi(\mathbf{x}) \pi(\mathbf{x}') / \pi(\mathbf{x}_{-k})`.
   Detailed balance holds, so :math:`\pi` is stationary.

Sampling the Initial Tree
===========================

Before the MCMC begins, ARGweaver needs an initial ARG. It builds one by threading
haplotypes sequentially, starting from a coalescent tree for the first pair.

The initial tree is sampled from a **coalescent with variable population sizes**:

.. code-block:: python

   import random
   from math import exp

   def sample_tree(k, popsizes, times):
       """
       Sample a coalescent tree using a discrete-time coalescent.

       Starting with k lineages at time 0, simulate coalescence events
       through the time grid. At each time interval, the coalescence rate
       depends on the number of lineages and the local population size.

       Parameters
       ----------
       k : int
           Number of lineages (chromosomes).
       popsizes : list of float
           Effective population size at each time interval.
       times : list of float
           Discretized time points.

       Returns
       -------
       list of float
           Coalescence times (one per coalescence event).
       """
       ntimes = len(times)
       coal_times = []

       timei = 0
       n = popsizes[timei]
       t = 0.0
       k2 = k  # current number of lineages

       while k2 > 1:
           # Coalescent rate: k2 choose 2 pairs, each coalescing at rate 1/(2N).
           # This is the standard coalescent rate from :ref:`coalescent_theory`.
           coal_rate = (k2 * (k2 - 1) / 2) / float(n)

           # Sample waiting time to next coalescence (exponential distribution)
           t2 = random.expovariate(coal_rate)

           if timei < ntimes - 2 and t + t2 > times[timei + 1]:
               # Crossed into next time interval; update population size.
               # Do NOT record a coalescence --- the lineage survived this
               # interval and moves on to the next one with a new N_e.
               timei += 1
               t = times[timei]
               n = popsizes[timei]
               continue

           t += t2
           coal_times.append(t)
           k2 -= 1  # one fewer lineage after coalescence

       return coal_times

After discretizing the initial tree to the time grid, the algorithm threads additional
haplotypes one at a time using the HMM.

.. admonition:: Closing the confusion gap --- Why start with a pairwise coalescence?

   The simplest possible ARG has two haplotypes: just a single tree with one
   coalescence event. ARGweaver starts here because (a) sampling a two-haplotype
   coalescence requires no HMM at all (just draw a coalescence time from the
   coalescent prior), and (b) once you have an ARG for two haplotypes, you can
   thread a third using the full HMM machinery. The ARG grows from 2 to 3 to 4
   to :math:`n` haplotypes, each step using the HMM to find where the new lineage
   best fits. This initial ARG is *not* a posterior sample --- it is just a starting
   point for the MCMC. The burn-in phase (below) lets the chain forget this initial
   configuration.

Sampling SPRs from the DSMC
=============================

Once the initial tree is built, the ARG is extended along the genome by sampling
recombination events (SPRs) from the Discrete SMC. At each position, there's a
chance of recombination; if it occurs, the tree changes via an SPR.

The five steps of sampling an SPR mirror the transition probability derivation in
:ref:`argweaver_transitions`: first determine *whether* a recombination occurs,
then *where* on the tree and *when* in time, and finally *where* the lineage
re-coalesces.

Step 1: Sample Recombination Position
---------------------------------------

Recombination events are a Poisson process along the genome with rate
:math:`\rho \cdot L` per base pair, where :math:`L` is the total tree length.

.. code-block:: python

   def sample_next_recomb(treelen, rho):
       """
       Sample the distance to the next recombination event.

       The waiting time (in base pairs) is exponential with rate
       rho * treelen.

       Parameters
       ----------
       treelen : float
           Total tree length.
       rho : float
           Per-base recombination rate.

       Returns
       -------
       float
           Distance in base pairs to the next recombination.
       """
       rate = max(treelen * rho, rho)  # guard against zero tree length
       # Exponential waiting time: the mean distance between recombination
       # events is 1 / (rho * treelen) base pairs.
       return random.expovariate(rate)

Step 2: Sample Recombination Time
-----------------------------------

Given that recombination occurs, the time is weighted by the amount of branch
material at each time interval:

.. math::

   P(\text{recomb at time } k) \propto n_{\text{branches}}[k] \cdot \Delta t_k

.. code-block:: python

   def sample_recomb_time(nbranches, time_steps, root_age_index):
       """
       Sample which time interval the recombination falls in.

       Probability is proportional to the amount of branch material
       at each time interval: nbranches[k] * time_steps[k].

       Parameters
       ----------
       nbranches : list of int
           Number of branches at each time interval.
       time_steps : list of float
           Time step sizes.
       root_age_index : int
           Time index of the root (recombination can only happen below).

       Returns
       -------
       int
           Time index of the recombination.
       """
       # Weight each interval by total branch material = count * duration.
       # More branches and longer intervals mean more opportunity for
       # recombination to land in that interval.
       weights = [nbranches[i] * time_steps[i]
                  for i in range(root_age_index + 1)]
       total = sum(weights)
       probs = [w / total for w in weights]

       # Sample from categorical distribution using inverse CDF.
       r = random.random()
       cumsum = 0.0
       for i, p in enumerate(probs):
           cumsum += p
           if r < cumsum:
               return i
       return len(probs) - 1

Step 3: Sample Recombination Node
-----------------------------------

Given the time, the recombination branch is chosen **uniformly** among all branches
that exist at that time index (excluding the root):

.. code-block:: python

   def sample_recomb_node(states, recomb_time_index, root_name):
       """
       Sample which branch the recombination falls on.

       Parameters
       ----------
       states : set of (str, int)
           Valid coalescent states.
       recomb_time_index : int
           Time index of the recombination.
       root_name : str
           Name of the root node (excluded from recombination).

       Returns
       -------
       str
           Name of the node below the recombination point.
       """
       # Filter to branches that exist at this time, excluding the root
       # (recombination above the root has no effect since there is only
       # one lineage there).
       branches = [name for name, timei in states
                   if timei == recomb_time_index and name != root_name]
       return random.choice(branches)

Step 4: Sample Coalescence Time
----------------------------------

After the recombination, the detached lineage must re-coalesce. It floats upward
through the time grid, with a chance to coalesce at each interval --- the same
discrete coalescent process used in the transition probabilities.

.. admonition:: Probability Aside --- The coalescent race

   Re-coalescence is a "race" between the detached lineage and the existing tree.
   At each time interval, the detached lineage has :math:`n_{\text{branches}}[j]`
   potential partners. The probability of coalescing with any one of them in a small
   time interval :math:`\Delta t` is approximately
   :math:`n_{\text{branches}}[j] \cdot \Delta t / (2N_j)`. If the lineage fails to
   coalesce, it moves to the next interval and tries again.

   This is the same discrete-coalescent process that generates the re-coalescence
   distribution in the transition probability derivation
   (:ref:`argweaver_transitions`). The only difference is that here we are *sampling*
   from the distribution (using random numbers) rather than *computing* it
   (enumerating all possibilities).

.. code-block:: python

   def sample_coal_time(recomb_time_index, nbranches, popsizes,
                        coal_times, ntimes, recomb_node, states):
       """
       Sample the re-coalescence time after a recombination.

       The lineage starts at recomb_time_index and moves upward,
       with a hazard of coalescing at each time interval proportional
       to the number of available lineages / (2 * Ne).

       Parameters
       ----------
       recomb_time_index : int
           Time index where recombination occurred.
       nbranches : list of int
           Number of branches at each time interval.
       popsizes : list of float
           Population sizes at each time interval.
       coal_times : list of float
           Interleaved time points and midpoints.
       ntimes : int
           Number of time intervals.
       recomb_node : object
           The recombination node (used to adjust lineage count).
       states : set of (str, int)
           Valid coalescent states.

       Returns
       -------
       int
           Time index of re-coalescence.
       """
       j = recomb_time_index
       last_kj = nbranches[max(j - 1, 0)]

       while j < ntimes - 1:
           kj = nbranches[j]

           # Adjust: if the recomb node passes through this interval,
           # it shouldn't count as an available coalescence partner.
           # (A lineage cannot coalesce with itself.)
           if ((recomb_node.name, j) in states and
                   recomb_node.parents[0].age > times[j]):
               kj -= 1

           assert kj > 0

           # Compute exposure in this interval using the interleaved
           # coal_times structure (see :ref:`argweaver_time_discretization`).
           A = (coal_times[2*j + 1] - coal_times[2*j]) * kj
           if j > recomb_time_index:
               A += (coal_times[2*j] - coal_times[2*j - 1]) * last_kj

           # Trial: coalesce in this interval?
           # Draw a Bernoulli trial with success probability = coal_prob.
           coal_prob = 1.0 - exp(-A / float(popsizes[j]))
           if random.random() < coal_prob:
               break

           # Survived this interval; move to the next tick mark on the dial.
           j += 1
           last_kj = kj

       return j

Step 5: Sample Coalescence Node
----------------------------------

Given the coalescence time, the branch is chosen uniformly among valid branches
at that time, excluding the recombination node itself and certain relatives:

.. code-block:: python

   def sample_coal_node(states, coal_time_index, recomb_node, tree, times):
       """
       Sample which branch the re-coalescing lineage joins.

       Parameters
       ----------
       states : set of (str, int)
           Valid coalescent states.
       coal_time_index : int
           Time index of the re-coalescence.
       recomb_node : object
           The node where recombination occurred.
       tree : tree object
           The local tree.
       times : list of float
           Discretized time points.

       Returns
       -------
       str
           Name of the node below the coalescence point.
       """
       coal_time = times[coal_time_index]

       # Build exclusion set: the recomb node and its descendants
       # at the same time (since coal points collapse).
       # The lineage cannot re-coalesce with the subtree it just
       # detached from --- that would create a trivial recombination.
       exclude = set()

       def walk(node):
           exclude.add(node.name)
           if node.age == coal_time:
               for child in node.children:
                   walk(child)

       walk(recomb_node)

       # Also exclude the recomb node's parent at its time
       exclude_parent = (recomb_node.parents[0].name,
                         times.index(recomb_node.parents[0].age))

       # Filter valid branches: must be at the right time index
       # and not in the exclusion set.
       branches = [(name, timei) for name, timei in states
                   if timei == coal_time_index
                   and name not in exclude
                   and (name, timei) != exclude_parent]

       chosen = random.choice(branches)
       return chosen[0]

*Recap of the five sampling steps:* We sample (1) the genomic position of the
recombination, (2) the time interval, (3) the specific branch, (4) the re-coalescence
time, and (5) the re-coalescence branch. Together, these define one SPR that transforms
the local tree into a new local tree at the next recombination breakpoint.

The Full MCMC Loop
===================

With all the pieces in place, the full MCMC loop is:

.. code-block:: python

   def argweaver_mcmc(sequences, times, popsizes, rho, mu,
                      num_iters=1000, burn_in=200):
       """
       Run the ARGweaver MCMC sampler.

       Parameters
       ----------
       sequences : dict of {str: str}
           Aligned haplotype sequences.
       times : list of float
           Discretized time points.
       popsizes : list of float
           Population sizes at each time interval.
       rho : float
           Per-base recombination rate.
       mu : float
           Per-base mutation rate.
       num_iters : int
           Number of MCMC iterations.
       burn_in : int
           Number of burn-in iterations to discard.

       Yields
       ------
       arg : ARG object
           Sampled ARG (one per iteration after burn-in).
       """
       names = list(sequences.keys())
       n = len(names)

       # ---- Step 1: Build initial ARG ----
       # Thread haplotypes one at a time, starting from a pairwise
       # coalescence. This is NOT a posterior sample --- just a
       # starting point for the Markov chain.
       arg = build_initial_arg(sequences, times, popsizes, rho, mu)

       # ---- Step 2: MCMC iterations ----
       for iteration in range(num_iters):
           # Choose a random chromosome to re-thread.
           # This is the "remove a gear" step in the watch metaphor.
           remove_idx = random.randint(0, n - 1)
           remove_name = names[remove_idx]

           # Remove this chromosome's thread from the ARG.
           # This yields a partial ARG for n-1 haplotypes ---
           # the "space left by the removed gear."
           partial_arg = remove_thread(arg, remove_name)

           # Build the HMM for re-threading.
           # States: (branch, time_index) at each genomic position
           # Transitions: normal (within same tree) or switch (at breakpoints)
           # Emissions: parsimony-based likelihood
           # This assembles the time grid, transitions, and emissions
           # from the previous three chapters.
           hmm = build_threading_hmm(
               partial_arg, sequences[remove_name],
               times, popsizes, rho, mu
           )

           # Run forward algorithm (see :ref:`argweaver_transitions`).
           forward_probs = forward_algorithm(hmm)

           # Stochastic traceback: sample a thread from the posterior.
           # This is the "manufacture a new gear" step --- the new
           # thread is drawn from the exact conditional distribution.
           new_thread = stochastic_traceback(forward_probs, hmm)

           # Add the new thread back into the ARG.
           # The "insert the new gear" step.
           arg = add_thread(partial_arg, remove_name,
                            sequences[remove_name], new_thread)

           # Yield sample (after burn-in).
           # The first burn_in iterations are discarded because the
           # chain has not yet converged to the stationary distribution.
           if iteration >= burn_in:
               yield arg

.. code-block:: text

   MCMC Loop Diagram:

   Iteration i:
   +---------+     +---------+     +---------+     +---------+
   | Current |     | Remove  |     | Build   |     | Sample  |
   | ARG     | --> | thread  | --> | HMM for | --> | new     |
   | (n haps)|     | for k   |     | thread k|     | thread  |
   +---------+     +---------+     +---------+     +---------+
                                                        |
                        +-------------------------------+
                        |
                        v
                   +---------+
                   | Add new |
                   | thread  | --> ARG for iteration i+1
                   | to ARG  |
                   +---------+

Comparison with SINGER's SGPR
================================

ARGweaver and SINGER both use iterative re-threading, but their MCMC strategies
differ in important ways:

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Property
     - ARGweaver
     - SINGER (SGPR)
   * - **Update unit**
     - Single chromosome (leaf)
     - Sub-graph (can include internal nodes)
   * - **Proposal mechanism**
     - Exact conditional (Gibbs)
     - Data-informed proposal (MH)
   * - **Accept/reject**
     - Always accepted
     - Metropolis-Hastings ratio
   * - **Acceptance rate**
     - 100% (by construction)
     - High (~90%+) due to informed proposal
   * - **Time model**
     - Discrete (finite grid)
     - Continuous
   * - **HMM structure**
     - Single HMM (branch + time)
     - Two HMMs (branch, then time)
   * - **Per-iteration cost**
     - :math:`O(L \cdot S^2)` where :math:`S \sim k \cdot n_t`
     - :math:`O(L \cdot k)` per HMM
   * - **Scaling with** :math:`k`
     - :math:`O(k^2 n_t^2)` per site
     - :math:`O(k)` per site (approximate)
   * - **Mixing**
     - Slower (one leaf at a time)
     - Faster (sub-graphs span multiple nodes)

.. admonition:: The mixing tradeoff

   ARGweaver re-threads one chromosome at a time, which means it takes :math:`n`
   iterations to give every chromosome a chance to move. SINGER's SGPR can modify
   multiple nodes simultaneously, potentially exploring the posterior more efficiently.

   However, ARGweaver's exact conditional sampling means every update is statistically
   "perfect" given the rest of the ARG --- there's no wasted computation from rejected
   proposals. SINGER compensates for its approximate proposals with very high
   acceptance rates from data-informed sampling.

   In practice, both methods produce good posterior samples. ARGweaver is better
   suited for smaller sample sizes (:math:`n < 50`) with high accuracy requirements.
   SINGER scales to much larger sample sizes.

.. admonition:: Probability Aside --- Metropolis-Hastings vs. Gibbs acceptance rates

   In Metropolis-Hastings, the acceptance probability for a proposal :math:`x' \sim q(x' \mid x)` is

   .. math::

      \alpha(x \to x') = \min\!\left(1, \frac{\pi(x') \, q(x \mid x')}{\pi(x) \, q(x' \mid x)}\right)

   If the proposal distribution :math:`q` equals the conditional posterior :math:`\pi(x' \mid x_{-k})`,
   then :math:`\alpha = 1` always --- this is exactly Gibbs sampling. The Gibbs sampler
   is the special case of MH where the proposal is so good that nothing is ever rejected.

   SINGER's MH proposals are data-informed but not exactly equal to the conditional
   posterior (because SINGER uses continuous time and approximate two-HMM decoupling).
   The resulting acceptance rates are high (~90%+) but not 100%, meaning some
   computation is "wasted" on rejected proposals. ARGweaver wastes no computation on
   rejections, but each proposal is more expensive to compute (larger state space).

Convergence and Diagnostics
=============================

Like any MCMC, ARGweaver needs monitoring to ensure convergence:

**Burn-in**: The initial ARG (built by sequential threading) may not be representative
of the posterior. Discard the first several hundred iterations.

**Thinning**: Consecutive samples are correlated (they differ by only one chromosome's
thread). Thin by keeping every :math:`n`-th sample (where :math:`n` is the number
of haplotypes) to reduce autocorrelation.

**Diagnostics**:

- **Joint log-likelihood** :math:`\log P(\mathbf{D} \mid \mathcal{G})`: should stabilize
- **Total tree length**: should fluctuate around an equilibrium
- **Pairwise TMRCA** between specific pairs: check for convergence at specific loci

.. admonition:: Closing the confusion gap --- Why burn-in and thinning?

   **Burn-in** addresses the fact that the MCMC starts from an arbitrary initial ARG
   (the one built by sequential threading). This initial ARG may be far from the
   high-probability region of the posterior. The chain needs time to "forget" its
   starting point and reach the stationary distribution. Discarding early samples
   (the burn-in period) avoids contaminating your estimates with these unrepresentative
   samples.

   **Thinning** addresses autocorrelation: consecutive MCMC samples differ by only
   one chromosome's thread, so they are highly correlated. If you keep every sample,
   your effective sample size (the number of independent samples) is much smaller than
   the total number of samples. By keeping only every :math:`n`-th sample (one per
   "sweep" through all chromosomes), you reduce this correlation. A sweep is like
   one full rotation of the watch's second hand --- every gear has been replaced once.

.. code-block:: python

   def compute_log_likelihood(arg, sequences, mu, times):
       """
       Compute the joint log-likelihood of the data given the ARG.

       This sums the log emission probability at every site for the
       actual topology encoded in the ARG. Useful as an MCMC diagnostic.

       Parameters
       ----------
       arg : ARG object
           The current ARG sample.
       sequences : dict of {str: str}
           Observed sequences.
       mu : float
           Mutation rate.
       times : list of float
           Discretized time points.

       Returns
       -------
       float
           Joint log-likelihood.
       """
       total_ll = 0.0
       for (start, end), tree in iter_local_trees(arg):
           for pos in range(start, end):
               for node in tree:
                   if not node.parents:
                       continue
                   # Branch length: distance from node to parent.
                   # Floored to avoid log(0).
                   blen = max(node.get_dist(), times[1] * 0.1)
                   # Under Jukes-Cantor: P(no mut) = exp(-mu*blen)
                   # P(specific mut) = 1/3 * (1 - exp(-mu*blen))
                   parent_base = sequences.get(node.parents[0].name,
                                               'N')  # internal
                   child_base = sequences.get(node.name, 'N')
                   if parent_base == child_base:
                       total_ll += -mu * blen
                   else:
                       total_ll += log(1.0/3 * (1 - exp(-mu * blen)))
       return total_ll

*Final recap:* ARGweaver is a digital watch. Its time grid
(:ref:`argweaver_time_discretization`) provides the tick marks; its transition
probabilities (:ref:`argweaver_transitions`) are the gear train; its emission
probabilities (:ref:`argweaver_emissions`) are the escapement; and its Gibbs
sampler (this chapter) is the mainspring. Each MCMC iteration removes one gear
(thread), manufactures an exact replacement via the HMM, and inserts it. After
many ticks, the watch reads the correct time --- posterior samples of the ARG.

For the continuous-time alternative, see SINGER. For the simpler case of a single
diploid genome (no ARG, just a sequence of coalescence times), see PSMC in
:ref:`coalescent_theory`. For the shared theoretical foundations, see :ref:`smc`
and :ref:`args`.

Exercises
==========

.. admonition:: Exercise 1: Gibbs vs. Metropolis-Hastings

   Prove that Gibbs sampling satisfies detailed balance. That is, show that if
   the update for variable :math:`x_k` samples from :math:`P(x_k \mid x_{-k})`,
   then the joint distribution :math:`P(x_1, \ldots, x_n)` is a stationary
   distribution of the resulting Markov chain.

.. admonition:: Exercise 2: Mixing time analysis

   Consider an ARG with :math:`n = 10` chromosomes and :math:`L = 10{,}000` sites.
   How many MCMC iterations are needed for every chromosome to be re-threaded at
   least once (in expectation)? This is the coupon collector problem ---
   the expected number is :math:`n \cdot H_n` where :math:`H_n` is the :math:`n`-th
   harmonic number.

.. admonition:: Exercise 3: Implement the full loop

   Using the building blocks from previous chapters (time discretization, transitions,
   emissions), implement a simplified version of the MCMC loop for a small example
   (4 haplotypes, 100 sites). Run 1000 iterations and plot the total tree length
   at a specific site over iterations. Does it converge?

.. admonition:: Exercise 4: SINGER comparison

   For the same dataset, run both an ARGweaver-style Gibbs sampler and a
   SINGER-style MH sampler (using simplified transition/emission models). Compare:
   (a) wallclock time per iteration, (b) effective sample size after 1000 iterations,
   (c) autocorrelation of pairwise TMRCA at a fixed site. Which sampler is more
   efficient per iteration? Per unit of wallclock time?
