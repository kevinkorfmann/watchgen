.. _smc:

=====================================
The Sequentially Markov Coalescent
=====================================

   *To make the impossible tractable, we give up a little exactness.*

In the :ref:`coalescent_theory` chapter, we learned how lineages coalesce
backwards in time. In the :ref:`ARG chapter <args>`, we saw how recombination
creates a *sequence* of marginal trees along the genome. And in the
:ref:`HMM chapter <hmms>`, we set up the machinery to do inference on hidden
state sequences -- provided those states form a Markov chain.

There is a problem. The sequence of marginal trees produced by the full
coalescent with recombination (CwR) is **not** Markov. This chapter explains
why, introduces the Sequentially Markov Coalescent (SMC) approximation that
fixes the problem, and derives the transition densities that will power our
Timepieces.

.. admonition:: The Watch Metaphor

   The SMC approximation is like simplifying a complex mechanism by ignoring
   gear interactions that almost never occur. A master watchmaker knows that
   two gears deep inside the movement *could* brush against each other under
   extreme conditions, but for all practical purposes they never do. By
   designing the movement as if those interactions cannot happen, the watch
   becomes far simpler to build and maintain -- and it still keeps nearly
   perfect time.


The Problem with the Full Coalescent
======================================

The coalescent with recombination (CwR) is the correct model for ancestry along
a recombining chromosome (see :ref:`args`). But it has a fatal flaw for
computation: the process of marginal trees along the genome is **not Markov**.

In the CwR, the marginal tree at position :math:`x` depends on the trees at
*all* previous positions, not just the immediately preceding one. This means we
cannot directly plug the tree sequence into a Hidden Markov Model -- and without
HMMs, efficient inference is out of reach.

The **Sequentially Markov Coalescent (SMC)** (McVean and Cardin, 2005; Marjoram
and Wall, 2006) is an approximation that restores the Markov property by
excluding a class of rare events. The rest of this chapter develops this idea
step by step.


What Does "Markov" Mean, and Why Does It Matter?
===================================================

Before diving into the technical details, let us build intuition for what the
Markov property is and why computational methods depend on it.

Intuitive Explanation
-----------------------

Imagine you are watching a clock. At each tick, the configuration of gears
determines exactly what happens at the next tick. You do not need to know the
entire history of every tick since the clock was wound -- the current gear
positions are sufficient. That is the Markov property: **the future depends on
the past only through the present.**

.. admonition:: The Watch Metaphor

   The Markov property is like a mechanism where the next tick depends only on
   the current state of the gears, not on how they got there. A watchmaker
   can predict what will happen next by examining the movement right now,
   without consulting a logbook of every previous tick.

In the context of genealogical trees along a genome: if the tree sequence were
Markov, knowing the tree at position :math:`x` would be all we need to compute
the probability of the tree at position :math:`x + 1`. We would not need to
remember the trees at positions :math:`1, 2, \ldots, x - 1`.

Formal Definition
-------------------

A sequence of random variables :math:`(Z_1, Z_2, \ldots)` is **Markov** if for
every :math:`\ell`:

.. math::

   P(Z_{\ell+1} \mid Z_\ell, Z_{\ell-1}, \ldots, Z_1) = P(Z_{\ell+1} \mid Z_\ell)

.. admonition:: Probability Aside -- Conditional Probability Notation

   The expression :math:`P(A \mid B)` is read "the probability of :math:`A`
   given :math:`B`". It means: if we know :math:`B` has happened, what is the
   probability that :math:`A` also happens? The Markov property says that
   conditioning on the *entire history* :math:`(Z_1, \ldots, Z_\ell)` gives the
   same result as conditioning on just the most recent value :math:`Z_\ell`.
   In other words, the history provides no additional predictive power beyond
   what the present state already tells us.

Why Markov Matters for Computation
------------------------------------

The Markov property is not just a mathematical nicety -- it is a **computational
necessity**. Recall from the :ref:`HMM chapter <hmms>` that the forward
algorithm computes the likelihood of an observed sequence in :math:`O(L K^2)`
time, where :math:`L` is the sequence length and :math:`K` is the number of
hidden states. This efficiency relies entirely on the Markov property: at each
step, the forward variable :math:`\alpha_\ell(j) = P(X_1, \ldots, X_\ell,
Z_\ell = j)` can be updated using only :math:`\alpha_{\ell-1}`, not the full
history.

Without the Markov property, we would need to track all possible histories
:math:`(Z_1, \ldots, Z_\ell)` to compute the probability of
:math:`Z_{\ell+1}`. The number of such histories grows exponentially with
:math:`\ell`, making exact computation infeasible for genomes with millions of
positions.


What Makes CwR Non-Markov?
============================

If the marginal trees along the genome were Markov, we could use HMMs. But
in the full CwR, they are **not**. The culprit is a phenomenon called
**ghost lineages**.

The Mechanism
--------------

When a recombination happens in the CwR:

1. A lineage splits at a breakpoint
2. The piece to the right of the breakpoint floats up and can coalesce with
   *any* lineage in the full ancestral process, including ones that are
   **not** in the current marginal tree

This means a lineage from the "past" -- one that already coalesced and left the
current tree -- can reappear. These returning lineages are called **ghost
lineages**.

What Are Ghost Lineages? A Concrete Example
----------------------------------------------

To understand ghost lineages, consider a concrete example with three samples
:math:`A`, :math:`B`, and :math:`C`.

Suppose the marginal tree at genomic position :math:`x` has this shape:

.. code-block:: text

   Position x:

   Time
    |
    |        MRCA          <-- most recent common ancestor of all three
    |       /    \
    |      /      \
    |    (AB)      |       <-- A and B coalesce first
    |    / \       |
    |   A   B      C
    |
    0 ──────────────────

In this tree, the internal node where :math:`A` and :math:`B` coalesce
(call it node :math:`AB`) sends a single lineage upward. That lineage eventually
coalesces with :math:`C`'s lineage at the MRCA.

Now suppose a recombination occurs between positions :math:`x` and :math:`x+1`.
The recombination cuts :math:`A`'s lineage at some time below :math:`AB`. The
piece of :math:`A`'s ancestry to the right of the breakpoint detaches and floats
upward, looking for a lineage to coalesce with.

In the **full CwR**, this detached lineage can coalesce with *any* lineage that
exists at that time in the full ancestral recombination graph -- including
lineages that are not visible in the marginal tree at position :math:`x`. For
instance, perhaps at an earlier position there was a lineage :math:`D` that
coalesced with :math:`C`'s lineage but then disappeared from the marginal tree.
Lineage :math:`D` is a **ghost lineage**: it is not part of the current tree,
but it still exists in the full ARG and could reappear.

**Why this breaks the Markov property**: Knowing the current tree
:math:`\Psi_x` is not enough to predict which ghost lineages might return. You
would need to know the full history of trees :math:`\Psi_1, \ldots,
\Psi_{x-1}` to track all potentially returning lineages.

.. code-block:: text

   The ghost lineage problem:

   Position x tree:      Full ARG (invisible from position x alone):

       MRCA                     MRCA
      /    \                   /    \
    (AB)    |                (AB)    |
    / \     |                / \    (CD)   <-- ghost: D coalesced with C
   A   B    C              A   B   / \        at some earlier position
                                  C   D       but left the marginal tree


The SMC Approximation
======================

The SMC makes one simple restriction: after a recombination, the detached
lineage can only re-coalesce with branches that are **present in the current
marginal tree**. Ghost lineages are forbidden.

.. math::

   \text{CwR: re-coalesce with any ancestral lineage (including "ghost" lineages)} \\
   \text{SMC: re-coalesce only with branches in the current marginal tree}

Why Does This Restore the Markov Property?
--------------------------------------------

Under the SMC, the next tree :math:`\Psi_{x+1}` is determined entirely by:

1. The current tree :math:`\Psi_x` (which branches exist)
2. Whether a recombination occurs between :math:`x` and :math:`x+1`
3. If so, which branch is cut and which branch is re-joined

All three of these depend only on :math:`\Psi_x`, not on any earlier tree.
No ghost lineages can appear. Therefore
:math:`P(\Psi_{x+1} \mid \Psi_x, \Psi_{x-1}, \ldots) = P(\Psi_{x+1} \mid \Psi_x)`.

The Markov property is restored, and we can build an HMM.

How Good Is the Approximation?
---------------------------------

The events excluded by the SMC (re-coalescence with a lineage not in the
current tree) are rare: they require a specific lineage to leave the tree and
then return. For large sample sizes, the current tree has many branches, so
re-coalescence with an existing branch is overwhelmingly likely anyway.
Empirically, the SMC matches the CwR very closely for most statistics of
interest.

Think of it this way: the approximation removes a set of paths through the ARG
that are both rare and difficult to track. The resulting model captures the vast
majority of the probability mass while gaining the enormous computational
advantage of the Markov property.

.. code-block:: python

   import numpy as np

   def smc_transition(tree_branches, recomb_rate, coal_rates):
       """Compute SMC transition: given a marginal tree, produce the next one.

       Under the SMC, recombination picks a branch (proportional to length),
       snips above the recombination point, and the detached lineage re-coalesces
       with one of the remaining branches -- never with a 'ghost' lineage outside
       the current tree.

       Parameters
       ----------
       tree_branches : list of (child, parent, lower_time, upper_time)
           Each tuple represents one branch of the marginal tree.
           - child: node index of the child end of the branch
           - parent: node index of the parent end of the branch
           - lower_time: time at the child (bottom of branch)
           - upper_time: time at the parent (top of branch)
       recomb_rate : float
           rho/2 per unit branch length.
       coal_rates : callable
           coal_rates(t) returns the coalescence rate at time t.

       Returns
       -------
       dict or None
           Description of the transition (which branch was cut, where,
           and which branch was rejoined), or None if no valid
           re-coalescence target exists.
       """
       # Compute the total branch length of the tree.
       # The expression (u - l) computes the length of one branch;
       # the generator iterates over all branches, and sum() adds them up.
       # The syntax "_, _, l, u" is called tuple unpacking: it extracts
       # the third and fourth elements (lower_time, upper_time) from each
       # 4-tuple, ignoring the first two (child, parent) with underscores.
       total_length = sum(u - l for _, _, l, u in tree_branches)

       # Probability of recombination on each branch (proportional to length).
       # This is a list comprehension: it creates a new list by computing
       # (u - l) for each branch tuple in tree_branches.
       branch_lengths = [(u - l) for _, _, l, u in tree_branches]

       # Convert to a NumPy array and normalize to get probabilities.
       # Longer branches are more likely to be hit by recombination because
       # they represent more "exposure time" to the recombination process.
       probs = np.array(branch_lengths) / total_length

       # Randomly pick which branch the recombination falls on,
       # weighted by branch length.
       idx = np.random.choice(len(tree_branches), p=probs)

       # Unpack the chosen branch into its four components.
       # This is tuple unpacking again: the single element
       # tree_branches[idx] is a 4-tuple, and we assign each element
       # to a separate variable.
       child, parent, lower, upper = tree_branches[idx]

       # Pick a recombination time uniformly on the chosen branch.
       # The recombination can happen anywhere between the bottom (lower)
       # and top (upper) of this branch.
       recomb_time = np.random.uniform(lower, upper)

       # Under the SMC: re-coalesce with a branch in the CURRENT tree
       # above the recombination time. This is the key SMC restriction --
       # in the full CwR, we could also coalesce with ghost lineages.
       #
       # This list comprehension filters tree_branches to keep only those
       # branches that:
       #   (a) extend above the recombination time (u > recomb_time), AND
       #   (b) are not the same branch that was cut.
       # The "!=" comparison checks tuple inequality element by element.
       available = [(c, p, l, u) for c, p, l, u in tree_branches
                    if u > recomb_time and (c, p, l, u) != tree_branches[idx]]

       if not available:
           return None  # No valid re-coalescence (rare edge case)

       # Choose re-coalescence branch (simplified: uniform random choice).
       # In reality, the coalescent process determines this with rates
       # that depend on the number of lineages at each time, but for
       # illustration we use a uniform distribution.
       rejoin_idx = np.random.randint(len(available))
       rejoin_branch = available[rejoin_idx]

       return {
           'recomb_branch': tree_branches[idx],
           'recomb_time': recomb_time,
           'rejoin_branch': rejoin_branch,
       }

   # Example tree: ((0,1):4, (2,3):5, (4,5):6)
   # This is a tree with 4 leaf nodes (0, 1, 2, 3) and 2 internal nodes
   # (4, 5), plus a root (6).
   # Times: leaves at 0.0; node 4 at 0.3; node 5 at 0.7; root 6 at 1.5
   tree_branches = [
       (0, 4, 0.0, 0.3),  # leaf 0 to internal node 4
       (1, 4, 0.0, 0.3),  # leaf 1 to internal node 4
       (2, 5, 0.0, 0.7),  # leaf 2 to internal node 5
       (3, 5, 0.0, 0.7),  # leaf 3 to internal node 5
       (4, 6, 0.3, 1.5),  # internal node 4 to root 6
       (5, 6, 0.7, 1.5),  # internal node 5 to root 6
   ]

   np.random.seed(42)
   result = smc_transition(tree_branches, 0.01, None)
   if result:
       print(f"Recombination on branch: {result['recomb_branch']}")
       print(f"Recombination time: {result['recomb_time']:.4f}")
       print(f"Re-join branch: {result['rejoin_branch']}")

Now that we have established the SMC approximation and its key consequence --
the Markov property -- we can derive the transition probabilities that an HMM
needs.


The SMC Transition Probability
================================

Under the SMC, when a recombination occurs in a new lineage being threaded onto
the tree, the probability of the lineage moving from branch :math:`b_i` to
branch :math:`b_j` is:

.. math::

   P(B_\ell = b_j \mid B_{\ell-1} = b_i) =
   (1 - r_i)\delta_{ij} + r_i \frac{q_j}{\sum_k q_k}

where:

- :math:`r_i = 1 - \exp(-\frac{\rho}{2}\tau_i)` is the probability of
  recombination on branch :math:`b_i`, with :math:`\tau_i` being the
  representative time for that branch
- :math:`q_j = r_j \cdot p_j` weights the re-joining probability
- :math:`\delta_{ij}` is the Kronecker delta (1 if :math:`i = j`, 0 otherwise)

This transition has the classic **stay-or-switch** structure: with probability
:math:`1 - r_i`, nothing happens and we stay on the same branch; with
probability :math:`r_i`, a recombination occurs and we jump to a new branch
chosen according to the weights :math:`q_j`. This is exactly the Li-Stephens
transition structure from the :ref:`HMM chapter <hmms>`.

Deriving :math:`r_i`: The Recombination Probability
------------------------------------------------------

**Where does** :math:`r_i` **come from?** A lineage at representative time
:math:`\tau_i` has been evolving for :math:`\tau_i` coalescent time units. In
each infinitesimal interval :math:`dt`, a recombination occurs with probability
:math:`(\rho/2)dt`. The probability of *no* recombination over the entire
interval :math:`[0, \tau_i]` is :math:`e^{-\rho\tau_i/2}` (survival probability
of a Poisson process). So the probability of *at least one* recombination is
:math:`r_i = 1 - e^{-\rho\tau_i/2}`.

.. admonition:: Probability Aside -- Survival Probability of a Poisson Process

   If events happen at a constant rate :math:`\lambda` per unit time, the
   probability of *no* event in an interval of length :math:`t` is
   :math:`e^{-\lambda t}`. This is the survival function of the exponential
   distribution. Here, the "event" is recombination, the rate is
   :math:`\rho/2`, and the interval is :math:`\tau_i`, giving survival
   probability :math:`e^{-\rho \tau_i / 2}`.

Deriving :math:`q_j`: The Re-joining Weights
------------------------------------------------

**Why** :math:`q_j = r_j \cdot p_j` **?** This choice ensures the **stationary
distribution** of the Markov chain is :math:`\pi_i = p_i`. To verify, a
stationary distribution satisfies :math:`\pi_j = \sum_i \pi_i A_{ij}`:

.. math::

   \sum_i p_i A_{ij} &= \sum_i p_i \left[(1-r_i)\delta_{ij} + r_i \frac{q_j}{\sum_k q_k}\right] \\
   &= p_j(1-r_j) + \frac{q_j}{\sum_k q_k} \sum_i p_i r_i \\
   &= p_j(1-r_j) + \frac{r_j p_j}{\sum_k r_k p_k} \sum_i r_i p_i \\
   &= p_j(1-r_j) + r_j p_j = p_j \quad \checkmark

.. admonition:: Probability Aside -- Stationary Distributions

   A **stationary distribution** :math:`\pi` of a Markov chain with transition
   matrix :math:`A` is a probability vector such that :math:`\pi A = \pi`. If the
   chain starts in distribution :math:`\pi`, it stays in distribution :math:`\pi`
   forever. Requiring that the SMC chain have stationary distribution
   :math:`p_i` (the coalescence probability for branch :math:`i`) ensures that
   the chain is consistent with the coalescent in equilibrium.

The product :math:`r_j \cdot p_j` also has a physical interpretation: after a
recombination at time :math:`u`, the lineage must re-coalesce at some time
:math:`t > u`. Branches with higher representative times (larger :math:`\tau_j`)
are more likely to be "available" for re-coalescence, and :math:`r_j` captures
this effect. The coalescence probability :math:`p_j` captures the intrinsic
likelihood of joining branch :math:`j`.

.. code-block:: python

   def smc_branch_transition(tau, p, rho, n_branches):
       """Compute SMC transition probabilities between branches.

       This implements the stay-or-switch transition matrix:
           A[i,j] = (1 - r_i) * delta(i,j)  +  r_i * q_j / sum(q)
       where r_i is the recombination probability on branch i and
       q_j = r_j * p_j is the re-joining weight for branch j.

       Parameters
       ----------
       tau : ndarray of shape (K,)
           Representative joining time for each branch.
       p : ndarray of shape (K,)
           Coalescence probability for each branch.
       rho : float
           Population-scaled recombination rate (4*Ne*r*m per bin).
       n_branches : int
           Number of branches (K).

       Returns
       -------
       T : ndarray of shape (K, K)
           Transition matrix where T[i, j] = P(next branch = j | current = i).
       """
       K = n_branches

       # Recombination probability for each branch:
       # r_i = 1 - exp(-rho/2 * tau_i)
       # This is 1 minus the "survival probability" (no recombination).
       r = 1 - np.exp(-rho / 2 * tau)

       # Re-joining weights: q_j = r_j * p_j
       # These ensure the stationary distribution is p_i.
       q = r * p

       # Sum of all re-joining weights (used to normalize).
       q_sum = q.sum()

       # Build the K x K transition matrix.
       # T[i, j] is the probability of transitioning from branch i to branch j.
       T = np.zeros((K, K))
       for i in range(K):           # iterate over source branches
           for j in range(K):       # iterate over destination branches
               if i == j:
                   # Stay on the same branch: no recombination + recombination
                   # that happens to land back on the same branch.
                   T[i, j] = (1 - r[i]) + r[i] * q[j] / q_sum
               else:
                   # Switch to a different branch: requires recombination.
                   T[i, j] = r[i] * q[j] / q_sum

       # Verify rows sum to 1 (each row is a probability distribution).
       assert np.allclose(T.sum(axis=1), 1.0), "Rows must sum to 1"
       return T

   # Example: 5 branches with different representative times and
   # coalescence probabilities.
   K = 5
   tau = np.array([0.1, 0.3, 0.5, 0.8, 1.2])   # representative times
   p = np.array([0.3, 0.25, 0.2, 0.15, 0.1])     # coalescence probabilities
   rho = 0.5                                       # recombination rate

   T = smc_branch_transition(tau, p, rho, K)
   print("Transition matrix:")
   print(np.round(T, 4))
   print(f"\nRow sums: {T.sum(axis=1)}")

With the general SMC transition structure in hand, we now turn to the simplest
and most important special case: the pairwise (two-sequence) version.


PSMC: The Pairwise Case
=========================

The **Pairwise Sequentially Markov Coalescent (PSMC)** (Li and Durbin, 2011) is
the simplest application of the SMC: it uses just two sequences.

.. admonition:: The Watch Metaphor

   The PSMC transition is the simplest timepiece movement: just two hands
   (lineages) whose relative position changes at recombination breakpoints.
   With only two hands, the "tree" at each position is trivially described by
   a single number -- the coalescence time -- making the movement elegant
   in its simplicity.

With two sequences, the marginal tree at each position is trivial -- just a
single coalescence time :math:`T`. There is one branch going up from each
sample to the coalescence point, and that is it. As you move along the genome,
:math:`T` changes at recombination breakpoints.

The hidden state is the coalescence time :math:`T`, which is now a continuous
variable rather than a discrete branch index. The transition density
:math:`q_\rho(t \mid s)` gives the probability density of the new coalescence
time :math:`t` given that the previous coalescence time was :math:`s`.

Let us derive this transition density from first principles.

**Step 1: Does a recombination happen?**

The two lineages form a branch of total length :math:`2s` (two lineages, each
of length :math:`s`, but since we measure the rate :math:`\rho/2` per lineage
the effective rate along both lineages together is :math:`\rho s`). The
probability of *at least one* recombination is:

.. math::

   P(\text{recombination}) = 1 - e^{-\rho s}

.. admonition:: Calculus Aside -- Why :math:`\rho s` and Not :math:`\rho \cdot 2s`?

   There are two lineages, each of length :math:`s`, and each experiences
   recombination at rate :math:`\rho/2`. The total rate is
   :math:`2 \times (\rho/2) \times s = \rho s`. Different references use
   different conventions for whether :math:`\rho` already includes the factor
   of 2 or not. Here we follow the convention where the total rate on the pair
   is :math:`\rho s`.

With probability :math:`e^{-\rho s}`, no recombination occurs and the
coalescence time stays at :math:`s`. This is a **point mass** at :math:`t = s`:

.. math::

   q_\rho(t \mid s) = e^{-\rho s} \quad \text{at } t = s

.. admonition:: Probability Aside -- What Is a Point Mass (Dirac Delta)?

   When we say the distribution has a "point mass" at :math:`t = s`, we mean
   that there is a nonzero probability :math:`e^{-\rho s}` concentrated at the
   single point :math:`t = s`. This is different from a continuous density,
   which assigns zero probability to any individual point and only assigns
   nonzero probability to intervals.

   Mathematically, a point mass is often written using the **Dirac delta
   function** :math:`\delta(t - s)`, which is not a function in the ordinary
   sense but rather a "generalized function" (or distribution) defined by its
   behavior under integration:

   .. math::

      \int_{-\infty}^{\infty} f(t) \, \delta(t - s) \, dt = f(s)

   The Dirac delta is zero everywhere except at :math:`t = s`, where it is
   "infinitely tall" in such a way that its total integral is 1. When we write
   that the transition "density" has a component :math:`e^{-\rho s}` at
   :math:`t = s`, we are implicitly using this notation:
   :math:`e^{-\rho s} \cdot \delta(t - s)`.

   For our purposes, the key intuition is simple: with probability
   :math:`e^{-\rho s}`, no recombination happens and the coalescence time is
   *exactly* :math:`s` -- not "near :math:`s`" or "approximately :math:`s`",
   but precisely :math:`s`.

**Step 2: Where does the recombination happen?**

Conditioned on a recombination occurring, let :math:`u` be the recombination time.
Under the SMC, :math:`u` is **uniform** on :math:`[0, s]` (the branch extends
from the present at 0 to the coalescence at :math:`s`). So:

.. math::

   p(u \mid \text{recomb}) = \frac{1}{s}, \quad 0 \leq u \leq s

The recombination can strike anywhere along the branch with equal probability.
This is because, given that at least one recombination event occurred on a
branch of length :math:`s` (under a Poisson process), the location of the
*first* event is approximately uniform for the rates we consider.

**Step 3: Where does the lineage re-coalesce?**

After recombination at time :math:`u`, the detached lineage floats up and
re-coalesces with the other lineage. Under the coalescent (with 2 lineages and
pairwise coalescence rate 1 in coalescent time units), the waiting time from
:math:`u` is :math:`\text{Exp}(1)` -- an exponential random variable with
rate 1.

So the re-coalescence time is :math:`t = u + W` where :math:`W \sim
\text{Exp}(1)`. The density of :math:`t` given :math:`u` is:

.. math::

   p(t \mid u) = e^{-(t - u)}, \quad t \geq u

.. admonition:: Probability Aside -- The Exponential Distribution

   A random variable :math:`W \sim \text{Exp}(\lambda)` has density
   :math:`f(w) = \lambda e^{-\lambda w}` for :math:`w \geq 0`. Its mean is
   :math:`1/\lambda`. With :math:`\lambda = 1`, the density simplifies to
   :math:`e^{-w}` and the mean waiting time is 1 coalescent time unit. This
   is the standard coalescent waiting time for two lineages, as derived in the
   :ref:`coalescent_theory` chapter.

**Step 4: Combine by integrating over** :math:`u`.

We now have all the ingredients. Given that a recombination occurred:

- The recombination time :math:`u` is uniform on :math:`[0, s]`, with density
  :math:`1/s`
- Given :math:`u`, the new coalescence time :math:`t` has density
  :math:`e^{-(t-u)}` for :math:`t \geq u`

To get the density of :math:`t` (marginalizing over :math:`u`), we integrate:

.. math::

   q_0(t \mid s) = \int_0^{s \wedge t} \frac{1}{s} e^{-(t-u)} \, du

.. admonition:: Calculus Aside -- Why :math:`s \wedge t` as the Upper Limit?

   The notation :math:`s \wedge t` means :math:`\min(s, t)`. There are two
   constraints on :math:`u`: it must satisfy :math:`u \leq s` (the
   recombination must occur on the branch, which extends from 0 to :math:`s`)
   and :math:`u \leq t` (the re-coalescence time :math:`t` must come after the
   recombination time :math:`u`). The binding constraint is whichever is
   smaller, hence :math:`\min(s, t)`.

This integral splits into two cases depending on whether :math:`t < s` or
:math:`t \geq s`.

**Case 1:** :math:`t < s` **(new coalescence time is earlier than the old one).**

The integral runs from 0 to :math:`t`:

.. math::

   q_0(t \mid s) = \frac{1}{s} \int_0^{t} e^{-(t-u)} \, du

To evaluate this, we use the substitution :math:`v = t - u`. Then
:math:`dv = -du`, and the limits change:

- When :math:`u = 0`: :math:`v = t`
- When :math:`u = t`: :math:`v = 0`

.. math::

   \int_0^t e^{-(t-u)} \, du = \int_t^0 e^{-v}(-dv) = \int_0^t e^{-v} \, dv

Now we compute the antiderivative of :math:`e^{-v}`:

.. math::

   \int_0^t e^{-v} \, dv = \left[-e^{-v}\right]_0^t = -e^{-t} - (-e^{0}) = 1 - e^{-t}

.. admonition:: Calculus Aside -- Evaluating Definite Integrals

   The notation :math:`\left[F(v)\right]_a^b` means :math:`F(b) - F(a)`. Here,
   :math:`F(v) = -e^{-v}`, so :math:`F(t) - F(0) = -e^{-t} - (-1) = 1 - e^{-t}`.
   This is the CDF of the standard exponential distribution evaluated at
   :math:`t`.

Therefore:

.. math::

   q_0(t \mid s) = \frac{1}{s}(1 - e^{-t}), \quad t < s

**Case 2:** :math:`t \geq s` **(new coalescence time is later than or equal to the old one).**

The integral runs from 0 to :math:`s`:

.. math::

   q_0(t \mid s) = \frac{1}{s} \int_0^{s} e^{-(t-u)} \, du

Using the same substitution :math:`v = t - u`, with :math:`dv = -du`:

- When :math:`u = 0`: :math:`v = t`
- When :math:`u = s`: :math:`v = t - s`

.. math::

   \int_0^s e^{-(t-u)} \, du = \int_t^{t-s} e^{-v}(-dv) = \int_{t-s}^{t} e^{-v} \, dv

Computing the antiderivative:

.. math::

   \int_{t-s}^{t} e^{-v} \, dv = \left[-e^{-v}\right]_{t-s}^{t} = -e^{-t} - (-e^{-(t-s)}) = e^{-(t-s)} - e^{-t}

Therefore:

.. math::

   q_0(t \mid s) = \frac{1}{s}\left(e^{-(t-s)} - e^{-t}\right), \quad t \geq s

**Step 5: Include the no-recombination probability.**

Combining the recombination probability :math:`(1 - e^{-\rho s})` with the
conditional density :math:`q_0(t \mid s)`, and adding the point mass for the
no-recombination case:

.. math::

   q_\rho(t \mid s) = \begin{cases}
   \frac{1 - e^{-\rho s}}{s}[1 - e^{-t}] & t < s \\[6pt]
   e^{-\rho s} & t = s \text{ (point mass: no recombination)} \\[6pt]
   \frac{1 - e^{-\rho s}}{s}[e^{-(t-s)} - e^{-t}] & t > s
   \end{cases}

**Verification: Does the density integrate to 1?**

The total probability must be 1: the continuous part should integrate to
:math:`1 - e^{-\rho s}` (the probability that recombination occurs), and the
point mass contributes :math:`e^{-\rho s}`.

We need to verify that :math:`\int_0^\infty q_0(t \mid s) \, dt = 1` (the
conditional density given recombination integrates to 1):

.. math::

   \int_0^\infty q_0(t|s) \, dt &= \frac{1}{s}\int_0^s (1-e^{-t})dt + \frac{1}{s}\int_s^\infty (e^{-(t-s)} - e^{-t})dt

**First integral** (:math:`t` from 0 to :math:`s`):

.. math::

   \int_0^s (1 - e^{-t}) \, dt = \left[t + e^{-t}\right]_0^s = (s + e^{-s}) - (0 + 1) = s + e^{-s} - 1

**Second integral** (:math:`t` from :math:`s` to :math:`\infty`):

.. math::

   \int_s^\infty (e^{-(t-s)} - e^{-t}) \, dt = \left[-e^{-(t-s)} + e^{-t}\right]_s^\infty

At :math:`t \to \infty`: both :math:`e^{-(t-s)}` and :math:`e^{-t}` go to 0,
so the upper limit contributes 0.

At :math:`t = s`: :math:`-e^{0} + e^{-s} = -1 + e^{-s}`.

Therefore: :math:`0 - (-1 + e^{-s}) = 1 - e^{-s}`.

**Putting it together:**

.. math::

   \int_0^\infty q_0(t|s) \, dt = \frac{1}{s}[(s + e^{-s} - 1) + (1 - e^{-s})] = \frac{1}{s} \cdot s = 1 \quad \checkmark

So the conditional density integrates to 1 (given recombination). Multiplying
by :math:`(1 - e^{-\rho s})` and adding the point mass :math:`e^{-\rho s}`
gives total probability :math:`(1 - e^{-\rho s}) + e^{-\rho s} = 1`.

.. code-block:: python

   def psmc_transition_density(t, s, rho):
       """PSMC transition density q_rho(t | s).

       Computes the probability density of the new coalescence time t,
       given that the previous coalescence time was s.

       The density has three components:
       1. For t < s: (p_recomb / s) * (1 - exp(-t))
       2. At t = s:  a point mass of weight exp(-rho * s)  [no recombination]
       3. For t > s: (p_recomb / s) * (exp(-(t-s)) - exp(-t))

       This function returns only the continuous part (items 1 and 3).
       The point mass at t = s must be handled separately.

       Parameters
       ----------
       t : float or ndarray
           New coalescence time(s) to evaluate the density at.
       s : float
           Previous coalescence time.
       rho : float
           Recombination rate for the bin.

       Returns
       -------
       density : float or ndarray
           The continuous part of the transition density at each t.
       """
       # Probability of no recombination on the pair of branches.
       p_no_recomb = np.exp(-rho * s)

       # Probability that at least one recombination occurs.
       p_recomb = 1 - p_no_recomb

       # Convert t to a NumPy array so we can use boolean indexing.
       # The dtype=float ensures we get floating-point arithmetic.
       t = np.asarray(t, dtype=float)

       # Initialize output array to zero (same shape as t).
       density = np.zeros_like(t)

       # Case 1: t < s (new coalescence time is earlier than old).
       # mask_lt is a boolean array: True where t < s, False elsewhere.
       mask_lt = t < s
       density[mask_lt] = (p_recomb / s) * (1 - np.exp(-t[mask_lt]))

       # Case 2: t >= s (new coalescence time is later than old).
       # Note: at the single point t = s, the continuous density is
       # well-defined (both formulas give the same value there), but the
       # point mass must be added separately.
       mask_ge = t >= s
       density[mask_ge] = (p_recomb / s) * (
           np.exp(-(t[mask_ge] - s)) - np.exp(-t[mask_ge])
       )

       return density

   # Visualize the transition density
   s = 1.0        # previous coalescence time
   rho = 0.5      # recombination rate
   t_values = np.linspace(0.01, 4.0, 200)  # grid of new coalescence times
   densities = psmc_transition_density(t_values, s, rho)

   print(f"Transition density from s={s}, rho={rho}:")
   print(f"  Peak near t={t_values[np.argmax(densities)]:.2f}")

   # np.trapz approximates the integral using the trapezoidal rule.
   # This should be close to (1 - exp(-rho * s)), the recombination probability.
   print(f"  Integral (approx): {np.trapz(densities, t_values):.4f}")

   # The "missing mass" is the point mass at t = s (no recombination).
   print(f"  Missing mass (at t=s): {np.exp(-rho * s):.4f}")

   # Together they should sum to 1.
   print(f"  Total: {np.trapz(densities, t_values) + np.exp(-rho * s):.4f}")

The PSMC transition density is the foundation of the PSMC's time-discretized
HMM (see :ref:`psmc_overview`) and of SINGER's **time sampling** step.


The Cumulative Distribution Function
=======================================

To use the transition density in practice -- for instance, to sample from it
or to compute probabilities over intervals -- we need the **cumulative
distribution function** (CDF).

.. admonition:: Probability Aside -- What Is a CDF?

   The **cumulative distribution function** (CDF) of a random variable
   :math:`T` is defined as :math:`F(t) = P(T \leq t)`: the probability that
   :math:`T` takes a value less than or equal to :math:`t`. The CDF is obtained
   by integrating the density:

   .. math::

      F(t) = \int_{-\infty}^{t} f(x) \, dx

   CDFs are monotonically increasing from 0 to 1. They are essential for:

   - Computing the probability that :math:`T` falls in an interval
     :math:`[a, b]`: :math:`P(a \leq T \leq b) = F(b) - F(a)`
   - Sampling from the distribution via **inverse transform sampling**:
     generate :math:`U \sim \text{Uniform}(0, 1)` and set :math:`T = F^{-1}(U)`
   - Discretizing a continuous distribution by computing probabilities for
     each bin: :math:`P(a_k \leq T < a_{k+1}) = F(a_{k+1}) - F(a_k)`

   In the PSMC, we discretize coalescence time into bins and need the
   probability mass in each bin, which requires the CDF.

The CDF is obtained by integrating the density. Let us derive each case.

**Case 1:** :math:`t < s`.

We integrate the density :math:`\frac{1-e^{-\rho s}}{s}(1 - e^{-x})` from 0
to :math:`t`:

.. math::

   Q_\rho(t \mid s) &= \frac{1-e^{-\rho s}}{s} \int_0^t (1 - e^{-x}) \, dx

The integrand :math:`1 - e^{-x}` has antiderivative :math:`x + e^{-x}` (since
the derivative of :math:`x` is 1 and the derivative of :math:`e^{-x}` is
:math:`-e^{-x}`, so :math:`\frac{d}{dx}(x + e^{-x}) = 1 - e^{-x}`).

.. math::

   Q_\rho(t \mid s) &= \frac{1-e^{-\rho s}}{s} \left[x + e^{-x}\right]_0^t \\
   &= \frac{1-e^{-\rho s}}{s} \left[(t + e^{-t}) - (0 + 1)\right] \\
   &= \frac{1-e^{-\rho s}}{s}[t + e^{-t} - 1]

**Case 2:** :math:`t \geq s`.

We build the CDF in three pieces:

1. The integral of the density from 0 to :math:`s` (the Case 1 density,
   evaluated at its upper limit :math:`s`)
2. The point mass :math:`e^{-\rho s}` at :math:`t = s` (no recombination)
3. The integral of the density from :math:`s` to :math:`t` (the Case 2 density)

.. math::

   Q_\rho(t \mid s) &= \underbrace{\frac{1-e^{-\rho s}}{s}[s + e^{-s} - 1]}_{\text{integral up to } s}
   + \underbrace{e^{-\rho s}}_{\text{point mass}}
   + \underbrace{\frac{1-e^{-\rho s}}{s} \int_s^t (e^{-(x-s)} - e^{-x}) \, dx}_{\text{integral from } s \text{ to } t}

Let us compute the remaining integral. The integrand has two terms:

- :math:`e^{-(x-s)}` has antiderivative :math:`-e^{-(x-s)}`
- :math:`-e^{-x}` has antiderivative :math:`e^{-x}`

So the antiderivative of the integrand is :math:`-e^{-(x-s)} + e^{-x}`.
Evaluating:

.. math::

   \int_s^t (e^{-(x-s)} - e^{-x}) dx &= \left[-e^{-(x-s)} + e^{-x}\right]_s^t \\
   &= \left(-e^{-(t-s)} + e^{-t}\right) - \left(-e^{0} + e^{-s}\right) \\
   &= -e^{-(t-s)} + e^{-t} + 1 - e^{-s} \\
   &= 1 - e^{-s} - e^{-(t-s)} + e^{-t}

Adding it all up:

.. math::

   Q_\rho(t \mid s) &= \frac{1-e^{-\rho s}}{s}[\underbrace{s + e^{-s} - 1 + 1 - e^{-s}}_{= s} - e^{-(t-s)} + e^{-t}] + e^{-\rho s} \\
   &= \frac{1-e^{-\rho s}}{s}[s - e^{-(t-s)} + e^{-t}] + e^{-\rho s}

**Verification**: As :math:`t \to \infty`, both :math:`e^{-(t-s)} \to 0` and
:math:`e^{-t} \to 0`, so:

.. math::

   Q_\rho(\infty \mid s) = \frac{1-e^{-\rho s}}{s} \cdot s + e^{-\rho s} = 1 - e^{-\rho s} + e^{-\rho s} = 1 \quad \checkmark

In summary:

.. math::

   Q_\rho(t \mid s) = \begin{cases}
   \frac{1 - e^{-\rho s}}{s}[t + e^{-t} - 1] & t < s \\[6pt]
   \frac{1 - e^{-\rho s}}{s}[s - e^{-(t-s)} + e^{-t}] + e^{-\rho s} & t \geq s
   \end{cases}

.. code-block:: python

   def psmc_transition_cdf(t, s, rho):
       """PSMC transition CDF Q_rho(t | s).

       Computes the cumulative distribution function of the new coalescence
       time t, given the previous coalescence time s.

       This CDF includes the point mass at t = s (no recombination).
       For t >= s, the CDF jumps by exp(-rho * s) at t = s.

       Parameters
       ----------
       t : float or ndarray
           New coalescence time(s) to evaluate the CDF at.
       s : float
           Previous coalescence time.
       rho : float
           Recombination rate for the bin.

       Returns
       -------
       cdf : float or ndarray
           CDF values at each t.
       """
       # Probability of no recombination (the point mass weight).
       p_no_recomb = np.exp(-rho * s)

       # Probability of recombination (weight of the continuous part).
       p_recomb = 1 - p_no_recomb

       # Convert t to a NumPy array for vectorized computation.
       t = np.asarray(t, dtype=float)
       cdf = np.zeros_like(t)

       # Case 1: t < s
       # CDF is the integral of the density from 0 to t.
       mask_lt = t < s
       cdf[mask_lt] = (p_recomb / s) * (
           t[mask_lt] + np.exp(-t[mask_lt]) - 1
       )

       # Case 2: t >= s
       # CDF includes the continuous integral plus the point mass.
       mask_ge = t >= s
       cdf[mask_ge] = (p_recomb / s) * (
           s - np.exp(-(t[mask_ge] - s)) + np.exp(-t[mask_ge])
       ) + p_no_recomb

       return cdf

   # Verify: CDF should approach 1 as t -> infinity.
   print(f"CDF at t=10: {psmc_transition_cdf(np.array([10.0]), s, rho)[0]:.6f}")
   print(f"CDF at t=100: {psmc_transition_cdf(np.array([100.0]), s, rho)[0]:.6f}")


Why SMC Enables HMM Inference
================================

Let us now make explicit the connection between everything we have derived and
the HMM framework from the :ref:`HMM chapter <hmms>`.

An HMM requires three ingredients:

1. **Hidden states**: The possible values of the hidden variable at each
   genomic position.

2. **Transition probabilities**: The probability of moving from one hidden state
   to another between adjacent positions. These must depend only on the current
   state (Markov property).

3. **Emission probabilities**: The probability of the observed data at each
   position, given the hidden state.

The SMC provides ingredients 1 and 2. Here is the mapping:

1. **Markov property**: The SMC ensures that the tree at position :math:`\ell`
   depends only on the tree at :math:`\ell - 1` and the recombination event
   between them. This is exactly the Markov property that HMMs require.
   Without the SMC approximation (in the full CwR), we would need the entire
   tree history, and the HMM framework would not apply.

2. **Conditional independence**: Given the tree at position :math:`\ell`, the
   mutations at that position are independent of mutations at other positions.
   This gives us emission probabilities :math:`P(X_\ell \mid Z_\ell)` that
   depend only on the current hidden state, as HMMs require.

3. **Li-Stephens structure**: The SMC transition probability has the
   stay-or-switch form :math:`(1 - r_i)\delta_{ij} + r_i \cdot (\text{weights})`,
   which enables :math:`O(K)` forward steps instead of :math:`O(K^2)`.
   This is the same structure we studied in the :ref:`HMM chapter <hmms>`.

Together: **SMC + HMM = tractable ARG inference**. The SMC gives us a Markov
model of tree evolution along the genome, the HMM machinery lets us do efficient
inference in that model, and the PSMC transition density derived above provides
the specific transition probabilities for the pairwise case.


Summary
=======

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Concept
     - Key Idea
   * - CwR limitation
     - Tree sequence is not Markov (ghost lineages can return)
   * - Markov property
     - Future depends on past only through present; enables HMM inference
   * - Ghost lineages
     - Ancestral lineages not in the current tree that could reappear in the CwR
   * - SMC approximation
     - Re-coalescence only with current tree branches; eliminates ghost lineages
   * - SMC transitions
     - Li-Stephens structure: :math:`(1-r_i)\delta_{ij} + r_i \frac{q_j}{\sum q}`
   * - PSMC transitions
     - Continuous-time density :math:`q_\rho(t|s)` and CDF :math:`Q_\rho(t|s)` for pairwise coalescence
   * - SMC + HMM
     - The SMC provides the Markov property; the HMM provides the inference algorithm

You now have all the prerequisites. Time to build a Timepiece.

Next:

- :ref:`singer_overview` -- disassembling the SINGER algorithm (ARG inference for multiple samples).
- :ref:`psmc_overview` -- building the PSMC (population size history from a single diploid genome).
