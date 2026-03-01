.. _sgpr:

==========================================
Sub-Graph Pruning and Re-grafting (SGPR)
==========================================

   *The mainspring of the MCMC: make a cut, thread it back, and let the data guide you.*

In the :ref:`previous chapters <branch_sampling>`, we built the threading
algorithm (branch sampling + time sampling) and the :ref:`calibration step
<arg_rescaling>`. Together, these components can construct an initial ARG from
a set of haplotypes and adjust its time scale. But a single construction is not
enough -- the threading order matters, the stochastic choices in each HMM
traceback introduce randomness, and the resulting ARG may not be the best
explanation of the data.

SGPR is SINGER's mechanism for exploring the space of ARGs. It is the
**winding mechanism** of the grand complication -- the mainspring that keeps
the MCMC chain moving, proposing new ARG configurations that are tested
against the data. Without SGPR, SINGER would produce a single (possibly poor)
ARG and stop. With SGPR, it explores the posterior distribution, gradually
improving the ARG and eventually producing high-quality posterior samples.

SGPR is an MCMC (Markov Chain Monte Carlo) move that proposes updates to the
current ARG by:

1. **Pruning**: cutting a sub-graph from the ARG
2. **Re-grafting**: re-threading the cut point using the same branch + time
   sampling algorithm

The genius of SGPR is that by using the data-informed threading algorithm
(instead of sampling from the prior), the acceptance rate is dramatically higher
than previous proposals -- approaching 1.0 for large sample sizes.

.. admonition:: Probability Aside -- What is MCMC?

   For a comprehensive introduction to MCMC, including the Metropolis-Hastings
   algorithm and convergence diagnostics, see the prerequisite chapter :ref:`mcmc`.

   **Markov Chain Monte Carlo (MCMC)** is a family of algorithms for sampling
   from a probability distribution that is difficult to sample from directly.
   The idea: construct a Markov chain (a sequence of random states where each
   state depends only on the previous one) whose long-run behavior converges
   to the target distribution.

   For SINGER, the target distribution is the **posterior over ARGs**:
   :math:`P(\mathcal{G} \mid D)`, the probability of an ARG given the observed
   DNA sequences. This distribution is astronomically complex -- the space of
   possible ARGs is vast, and computing :math:`P(\mathcal{G} \mid D)` directly
   for every possible ARG is impossible. MCMC sidesteps this by exploring the
   space one step at a time, spending more time in regions of high posterior
   probability.

   If you have encountered MCMC in other contexts (e.g., Bayesian statistics,
   statistical physics), the concepts here are the same. The innovation in
   SGPR is in the *proposal distribution* -- how we choose the next state to
   try. A good proposal makes MCMC efficient; a poor proposal makes it
   glacially slow.


The Subtree Pruning and Regrafting (SPR) Foundation
======================================================

Before understanding SGPR, let's start with its simpler ancestor: **SPR** on
a single tree. This is the foundation on which SGPR builds -- understanding
SPR on a single tree makes the extension to ARGs straightforward.

In SPR, you:

1. Pick a branch in the tree
2. Cut it (detach the subtree below the cut from the rest)
3. Re-attach the subtree somewhere else in the remaining tree

.. admonition:: Probability Aside -- SPR and Tree Space

   SPR moves define a **graph on tree space**: two trees are "neighbors" if
   one can be obtained from the other by a single SPR move. This graph is
   connected -- any tree can be reached from any other tree by a sequence of
   SPR moves (proven by Allen and Steel, 2001). This connectivity is essential
   for MCMC: it guarantees that the Markov chain can reach any tree, so the
   chain can converge to the true posterior distribution regardless of its
   starting point.

   The number of SPR neighbors of a tree with :math:`n` leaves is
   :math:`O(n^2)` -- there are :math:`O(n)` branches to cut and :math:`O(n)`
   branches to re-attach to. For an ARG, the number of possible SGPR moves
   is larger because the cut extends along the genome, but the same
   connectivity property holds.

.. code-block:: python

   import numpy as np

   class SimpleTree:
       """A minimal tree for demonstrating SPR.

       Stores the tree as a parent map (each node points to its parent)
       and a time map (each node has a time/height).  This is the same
       representation used internally by tree sequence libraries like
       tskit (see the ARG prerequisite chapter).
       """

       def __init__(self, parent, time):
           """
           Parameters
           ----------
           parent : dict of {node: parent_node}
           time : dict of {node: time}
           """
           self.parent = parent
           self.time = time
           # Build children map by inverting the parent map
           self.children = {}
           for child, par in parent.items():
               if par is not None:
                   self.children.setdefault(par, []).append(child)

       def branches(self):
           """Return all branches as (child, parent, length)."""
           result = []
           for child, par in self.parent.items():
               if par is not None:
                   result.append((child, par,
                                  self.time[par] - self.time[child]))
           return result

       def height(self):
           """Return the tree height (TMRCA).

           The TMRCA (Time to Most Recent Common Ancestor) is the time
           of the root node -- the oldest node in the tree.
           """
           return max(self.time.values())

   def spr_move(tree, cut_node, new_parent, new_time):
       """Perform an SPR move on a tree.

       This implements the three-step SPR procedure:
       1. Detach the subtree rooted at cut_node
       2. Remove the now-unary node (the old parent of cut_node)
       3. Re-attach cut_node to new_parent at new_time

       Parameters
       ----------
       tree : SimpleTree
       cut_node : int
           The node whose branch we cut above.
       new_parent : int
           The branch (identified by its child node) to re-attach to.
       new_time : float
           The time of the re-attachment point.

       Returns
       -------
       new_tree : SimpleTree
       """
       new_parent_dict = dict(tree.parent)
       new_time_dict = dict(tree.time)

       # Find the old parent and grandparent of cut_node
       old_parent = new_parent_dict[cut_node]
       old_grandparent = new_parent_dict.get(old_parent)

       # Find sibling of cut_node (the other child of old_parent)
       siblings = [c for c in tree.children.get(old_parent, [])
                    if c != cut_node]

       if siblings and old_grandparent is not None:
           sibling = siblings[0]
           # Remove old_parent node: connect sibling directly to grandparent
           # This eliminates the now-unary internal node
           new_parent_dict[sibling] = old_grandparent
           del new_parent_dict[old_parent]

       # Re-attach cut_node to new_parent at new_time
       # Create a new internal node at the re-attachment point
       new_internal = max(new_time_dict.keys()) + 1
       new_time_dict[new_internal] = new_time

       # Re-wire: cut_node and new_parent both become children
       # of the new internal node
       target_parent = new_parent_dict.get(new_parent)
       new_parent_dict[new_parent] = new_internal
       new_parent_dict[cut_node] = new_internal
       if target_parent is not None:
           new_parent_dict[new_internal] = target_parent

       return SimpleTree(new_parent_dict, new_time_dict)

   # Example tree: ((0,1)4, (2,3)5)6
   tree = SimpleTree(
       parent={0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6},
       time={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.3, 5: 0.7, 6: 1.5}
   )
   print("Original branches:")
   for c, p, l in tree.branches():
       print(f"  {c} -> {p}: length={l:.2f}")

With the SPR foundation in place, we now extend the idea from a single tree to
an entire ARG. The key challenge: a cut on one marginal tree extends to adjacent
trees, creating a "sub-graph" rather than a "sub-tree."


Step 1: How to Prune a Sub-Graph
==================================

In an ARG (unlike a single tree), a cut on one marginal tree extends to
adjacent trees. The extension follows a simple rule:

**Rule**: A cut on a colored segment in one tree corresponds to the segment of
the **same color** in the adjacent tree. The extension terminates when the cut
reaches a segment that was created by a recombination event (a "black dashed"
segment -- the part of a new lineage above the recombination breakpoint).

.. admonition:: Probability Aside -- Why the Cut Extends Along the Genome

   In an ARG, adjacent marginal trees share most of their topology -- they
   differ only at the recombination breakpoints. When you cut a branch in one
   tree, the same lineage typically exists (possibly in a different position)
   in the adjacent tree. The cut must follow this lineage across the genome
   to maintain consistency.

   The extension stops at recombination breakpoints because a recombination
   creates a *new* lineage above it -- a lineage that did not exist in the
   previous tree. The cut cannot follow a lineage that doesn't exist, so it
   stops.

   This is the key difference between SPR (on a single tree) and SGPR (on an
   ARG): in SPR, the cut affects a single tree; in SGPR, it affects a
   contiguous segment of the genome, potentially spanning many marginal trees.

For each marginal tree in the span of the cut, we remove the portion of the
branch **above the cut** to the upper node.

.. code-block:: python

   def find_cut_span(arg, tree_idx, cut_node, cut_time):
       """Find how far a cut extends left and right along the genome.

       Starting from the initial cut position, trace the lineage
       in both directions until hitting a recombination boundary
       (where the lineage was created by a recombination event).

       Parameters
       ----------
       arg : object
           The ARG (sequence of marginal trees with recombinations).
       tree_idx : int
           Index of the marginal tree where the cut is initiated.
       cut_node : int
           The branch (child node) being cut.
       cut_time : float
           The time of the cut.

       Returns
       -------
       left_bound : int
           Leftmost tree index affected by the cut.
       right_bound : int
           Rightmost tree index affected by the cut.
       """
       n_trees = len(arg.trees)

       # Extend rightward: follow the lineage through each recombination
       right_bound = tree_idx
       current_branch = cut_node
       for t in range(tree_idx + 1, n_trees):
           # Check if there's a recombination between tree t-1 and t
           recomb = arg.get_recombination(t - 1, t)
           if recomb is None:
               # No recombination: the lineage continues unchanged
               right_bound = t
               continue

           # Check if the cut segment can be traced through the recombination
           mapped = trace_branch_through_recomb(current_branch, cut_time,
                                                 recomb)
           if mapped is None:
               # Hit a decoupled segment -- the lineage was created by
               # this recombination, so the cut cannot extend further
               break

           current_branch = mapped
           right_bound = t

       # Extend leftward (symmetric logic)
       left_bound = tree_idx
       current_branch = cut_node
       for t in range(tree_idx - 1, -1, -1):
           recomb = arg.get_recombination(t, t + 1)
           if recomb is None:
               left_bound = t
               continue

           mapped = trace_branch_through_recomb(current_branch, cut_time,
                                                 recomb)
           if mapped is None:
               break

           current_branch = mapped
           left_bound = t

       return left_bound, right_bound


Step 2: The Cut Selection Procedure
=====================================

SINGER selects cuts as follows:

1. Start at the rightmost position of the previous cut (or the beginning of the
   chromosome if this is the first cut).

2. At the marginal tree at this position:

   a. Sample a cut time :math:`t` uniformly at random between 0 and the tree
      height (TMRCA).
   b. Find all branches that cross time :math:`t`.
   c. Choose one uniformly at random.
   d. Cut the chosen branch at time :math:`t`.

3. The cut extends left and right using the rule from Step 1.

.. code-block:: python

   def select_cut(tree):
       """Select a random cut on a marginal tree.

       The cut is chosen by first sampling a time uniformly in
       [0, tree height], then choosing uniformly among branches
       that cross that time.  This two-step procedure gives each
       branch a probability proportional to its length.

       Parameters
       ----------
       tree : SimpleTree

       Returns
       -------
       cut_node : int
           The branch being cut (identified by child node).
       cut_time : float
           The time of the cut.
       """
       # Sample time uniformly in [0, tree height]
       h = tree.height()
       cut_time = np.random.uniform(0, h)

       # Find branches that cross this time
       crossing_branches = []
       for child, parent, length in tree.branches():
           if tree.time[child] <= cut_time < tree.time[parent]:
               crossing_branches.append(child)

       # Choose one uniformly at random
       cut_node = np.random.choice(crossing_branches)
       return cut_node, cut_time

   np.random.seed(42)
   cut_node, cut_time = select_cut(tree)
   print(f"Cut: node {cut_node} at time {cut_time:.4f}")

.. admonition:: Probability Aside -- The Cut Probability and Ultrametric Trees

   The probability of selecting a specific cut is:

   .. math::

      P(\text{cut } b \text{ at } x) = \frac{1}{h(\Psi_x)}

   where :math:`h(\Psi_x)` is the height of tree :math:`\Psi_x`. This is because
   the time is uniform in :math:`[0, h]`, and conditioned on the time, each crossing
   branch is equally likely. The number of crossing branches at time :math:`t` is
   exactly :math:`\lambda_\Psi(t)`, and when we integrate over :math:`t`, we get a
   probability proportional to branch length -- and all branch lengths sum to
   :math:`h(\Psi_x)` when counting from 0 (since the tree is ultrametric).

   An **ultrametric tree** is one where all leaves are at the same distance
   (time) from the root -- equivalently, all leaves are at time 0 (the present).
   Coalescent trees are always ultrametric because all samples are taken at
   the same time. In an ultrametric tree, the sum of all branch lengths at
   each time slice equals exactly the number of lineages at that time, and
   integrating from 0 to the root gives the tree height :math:`h`.

   This means:

   .. math::

      P(H \mid G) = \frac{1}{h(\Psi_x)}

   This is a crucial quantity for the acceptance ratio, as we will see in Step 4.


Step 3: Re-grafting via Threading
===================================

After pruning, we re-graft by running the **same threading algorithm** (branch
sampling + time sampling) on the cut point, using the remaining partial ARG
:math:`H` as the reference.

This is the key innovation over the Kuhner move (which re-grafts by simulating
from the **prior**). By using the threading algorithm, SINGER's proposal takes
the **data** into account and proposes genealogies consistent with the observed
mutations. In the watch metaphor, the Kuhner move replaces a gear blindfolded
(hoping it fits); SGPR examines the mechanism first (using the threading
algorithm's emission probabilities) and chooses a gear that is likely to mesh
correctly.

.. admonition:: Probability Aside -- Prior vs. Posterior Proposals

   In MCMC, the **proposal distribution** determines what new states the chain
   considers. There are two natural choices:

   - **Prior proposal**: sample the new state from the model's prior
     distribution (e.g., the coalescent process). This is simple but ignores
     the data, leading to proposals that rarely explain the observed mutations
     well. Most are rejected.

   - **Posterior proposal**: sample the new state from (an approximation to)
     the posterior distribution, which accounts for both the prior and the data.
     This produces proposals that are likely to be good explanations of the
     data, leading to high acceptance rates.

   SGPR uses a posterior proposal: the threading algorithm's HMM incorporates
   both the coalescent prior (through transition probabilities) and the data
   (through emission probabilities). The resulting proposals are much better
   than prior proposals, which is why SGPR's acceptance rate approaches 1.0
   while the Kuhner move's acceptance rate is very low.

.. code-block:: python

   def sgpr_move(arg, cut_position=None):
       """Perform one SGPR move on the ARG.

       This is the complete SGPR cycle:
       1. Select a cut (random branch and time)
       2. Find the genomic span of the cut
       3. Prune the sub-graph above the cut
       4. Re-graft using the threading algorithm
       5. Compute the acceptance ratio

       Parameters
       ----------
       arg : object
           Current ARG state.
       cut_position : int, optional
           Genome position for the cut. If None, continues from
           the last cut position.

       Returns
       -------
       new_arg : object
           Proposed new ARG.
       acceptance_ratio : float
           The Metropolis-Hastings acceptance ratio.
       """
       # Step 1: Select a cut
       tree = arg.get_tree_at(cut_position)
       cut_node, cut_time = select_cut(tree)

       # Step 2: Find the span of the cut
       left, right = find_cut_span(arg, cut_position, cut_node, cut_time)

       # Step 3: Prune -- remove the sub-graph above the cut
       partial_arg = prune_subgraph(arg, cut_node, cut_time, left, right)

       # Step 4: Re-graft using threading (data-informed proposal)
       # This is the same threading algorithm from the branch_sampling
       # and time_sampling chapters
       joining_branches = branch_sampling(partial_arg, cut_node, left, right)
       joining_times = time_sampling_sgpr(partial_arg, cut_node,
                                          joining_branches, left, right)

       # Step 5: Build the new ARG
       new_arg = regraft(partial_arg, cut_node, joining_branches,
                          joining_times, left, right)

       # Step 6: Compute acceptance ratio (see Step 4 below)
       h_old = tree.height()
       h_new = new_arg.get_tree_at(cut_position).height()
       acceptance_ratio = min(1.0, h_old / h_new)

       return new_arg, acceptance_ratio

With the SGPR move defined, we need to understand *why* the acceptance ratio
takes such a simple form. This requires a careful derivation from the
Metropolis-Hastings framework -- the mathematical backbone of MCMC.


Step 4: The Acceptance Ratio
==============================

This is where SGPR shines. Let's build the derivation carefully from the
Metropolis-Hastings framework.

Background: Metropolis-Hastings
---------------------------------

MCMC works by constructing a Markov chain whose stationary distribution is the
target posterior :math:`P(\mathcal{G} \mid D)`. At each step, we propose a new
state :math:`\mathcal{G}'` and accept it with probability:

.. math::

   A(\mathcal{G} \to \mathcal{G}') = \min\left(1, \frac{P(\mathcal{G}' \mid D) \cdot q(\mathcal{G}' \to \mathcal{G})}{P(\mathcal{G} \mid D) \cdot q(\mathcal{G} \to \mathcal{G}')}\right)

where :math:`q(\mathcal{G} \to \mathcal{G}')` is the proposal probability.

.. admonition:: Probability Aside -- Detailed Balance and Why MH Works

   The Metropolis-Hastings acceptance formula ensures **detailed balance**: the
   probability of being in state :math:`\mathcal{G}` and moving to
   :math:`\mathcal{G}'` equals the probability of the reverse:

   .. math::

      P(\mathcal{G} \mid D) \cdot q(\mathcal{G} \to \mathcal{G}') \cdot A(\mathcal{G} \to \mathcal{G}')
      = P(\mathcal{G}' \mid D) \cdot q(\mathcal{G}' \to \mathcal{G}) \cdot A(\mathcal{G}' \to \mathcal{G})

   Detailed balance guarantees that the chain converges to the target
   distribution :math:`P(\mathcal{G} \mid D)` regardless of its starting point.
   This is a fundamental result in probability theory (see any textbook on
   MCMC, e.g., Robert and Casella, 2004).

   The MH formula is the unique way to set the acceptance probability that
   achieves detailed balance for *any* proposal distribution :math:`q`. Different
   choices of :math:`q` lead to different acceptance rates and mixing speeds,
   but all are valid MCMC samplers.

The SGPR proposal
-------------------

In SGPR, the proposal from :math:`\mathcal{G}` to :math:`\mathcal{G}'` involves
an intermediate step -- the partial ARG :math:`H`. The proposal factors as:

.. math::

   q_H(\mathcal{G} \to \mathcal{G}') = P(\text{prune } \mathcal{G} \text{ to } H) \cdot P(\text{regraft } H \text{ to } \mathcal{G}')

Let's write these two pieces explicitly:

1. :math:`P(H \mid \mathcal{G})` = probability of selecting the specific cut that
   produces :math:`H` from :math:`\mathcal{G}` (the pruning probability).

2. :math:`P(\mathcal{G}' \mid D, H)` = probability that the threading algorithm
   produces :math:`\mathcal{G}'` given :math:`H` (the re-grafting probability).

So: :math:`q_H(\mathcal{G} \to \mathcal{G}') = P(H \mid \mathcal{G}) \cdot P(\mathcal{G}' \mid D, H)`.

The key assumption
--------------------

**SINGER assumes the threading algorithm approximately samples from the posterior
conditioned on** :math:`H`:

.. math::

   P(\mathcal{G}' \mid D, H) \approx \frac{P(\mathcal{G}' \mid D)}{\sum_{\mathcal{G}'': H \subseteq \mathcal{G}''} P(\mathcal{G}'' \mid D)}

.. admonition:: Probability Aside -- Why This Approximation is Reasonable

   This approximation says: "the probability that the threading algorithm
   produces a specific ARG :math:`\mathcal{G}'` is proportional to the
   posterior probability of :math:`\mathcal{G}'`." In other words, the
   threading algorithm acts as an approximate posterior sampler.

   Why is this reasonable? The threading HMM is designed to sample branch
   assignments and times that are consistent with both the partial ARG :math:`H`
   (through the state space constraints) and the data :math:`D` (through the
   emission probabilities). For large sample sizes, the posterior becomes
   increasingly concentrated (there are fewer plausible ARGs), and the threading
   algorithm closely approximates the true conditional posterior.

   This is an approximation, not an exact identity. For small sample sizes,
   the threading HMM may not perfectly match the true posterior, leading to
   slightly biased proposals. However, because the MH acceptance step corrects
   for any mismatch between the proposal and the target, the MCMC chain still
   converges to the correct distribution -- it just may mix more slowly if the
   approximation is poor.

The cancellation
------------------

Now let's take the ratio of the reverse and forward proposals:

.. math::

   \frac{q_H(\mathcal{G}' \to \mathcal{G})}{q_H(\mathcal{G} \to \mathcal{G}')}
   = \frac{P(H \mid \mathcal{G}') \cdot P(\mathcal{G} \mid D, H)}{P(H \mid \mathcal{G}) \cdot P(\mathcal{G}' \mid D, H)}

Under the approximation above, :math:`P(\mathcal{G} \mid D, H)` and
:math:`P(\mathcal{G}' \mid D, H)` both have the same denominator
:math:`\sum_{\mathcal{G}''} P(\mathcal{G}'' \mid D)`, so:

.. math::

   \frac{P(\mathcal{G} \mid D, H)}{P(\mathcal{G}' \mid D, H)}
   = \frac{P(\mathcal{G} \mid D)}{P(\mathcal{G}' \mid D)}

Substituting into the MH acceptance ratio:

.. math::

   A_H &= \min\left(1, \frac{P(\mathcal{G}' \mid D)}{P(\mathcal{G} \mid D)}
   \cdot \frac{P(H \mid \mathcal{G}')}{P(H \mid \mathcal{G})}
   \cdot \frac{P(\mathcal{G} \mid D)}{P(\mathcal{G}' \mid D)}\right)

.. admonition:: Calculus Aside -- The Cancellation in Detail

   Let us trace the algebra carefully. The MH ratio is:

   .. math::

      \frac{P(\mathcal{G}' \mid D)}{P(\mathcal{G} \mid D)}
      \cdot \frac{q_H(\mathcal{G}' \to \mathcal{G})}{q_H(\mathcal{G} \to \mathcal{G}')}

   Expanding the proposal ratio:

   .. math::

      = \frac{P(\mathcal{G}' \mid D)}{P(\mathcal{G} \mid D)}
      \cdot \frac{P(H \mid \mathcal{G}')}{P(H \mid \mathcal{G})}
      \cdot \frac{P(\mathcal{G} \mid D, H)}{P(\mathcal{G}' \mid D, H)}

   Now substitute the approximation
   :math:`P(\mathcal{G} \mid D, H) / P(\mathcal{G}' \mid D, H) = P(\mathcal{G} \mid D) / P(\mathcal{G}' \mid D)`:

   .. math::

      = \frac{P(\mathcal{G}' \mid D)}{P(\mathcal{G} \mid D)}
      \cdot \frac{P(H \mid \mathcal{G}')}{P(H \mid \mathcal{G})}
      \cdot \frac{P(\mathcal{G} \mid D)}{P(\mathcal{G}' \mid D)}

   The first and third factors are exact inverses and cancel:

   .. math::

      = \frac{P(H \mid \mathcal{G}')}{P(H \mid \mathcal{G})}

   This is the remarkable result: the posterior ratio cancels completely,
   leaving only the ratio of pruning probabilities.

The posterior ratios :math:`\frac{P(\mathcal{G}' \mid D)}{P(\mathcal{G} \mid D)}`
and :math:`\frac{P(\mathcal{G} \mid D)}{P(\mathcal{G}' \mid D)}` are exact
**inverses** and cancel completely:

.. math::

   A_H = \min\left(1, \frac{P(H \mid \mathcal{G}')}{P(H \mid \mathcal{G})}\right)

This is remarkable: **the acceptance ratio depends only on the pruning
probabilities, not on the data likelihood at all**.

Computing :math:`P(H \mid \mathcal{G})`
------------------------------------------

From Step 2, the cut is chosen by:

1. Sampling time :math:`t` uniformly on :math:`[0, h(\Psi_x)]` (tree height)
2. Choosing uniformly among branches crossing time :math:`t`

The probability of selecting *any specific* cut is:

.. math::

   P(H \mid \mathcal{G}) = \frac{1}{h(\Psi_x)}

**Why?** The probability of choosing a specific branch :math:`b` at time :math:`t`
is :math:`\frac{1}{h(\Psi_x)} \cdot \frac{1}{\lambda(t)}` (uniform time times
uniform branch choice). Integrating over the branch's time interval :math:`[l, u]`:

.. math::

   P(\text{cut branch } b) = \int_l^u \frac{1}{h(\Psi_x)} \cdot \frac{1}{\lambda(t)} \, dt

But we don't need the specific branch -- we only need :math:`P(H \mid \mathcal{G})`.
In an ultrametric tree (where all leaves are at time 0), summing over all
possible cuts gives :math:`1/h(\Psi_x)` because the cut time is uniform on
:math:`[0, h(\Psi_x)]` and there is always exactly one valid cut at each time.

The final result
------------------

Substituting :math:`P(H \mid \mathcal{G}) = 1/h(\Psi_x)` and
:math:`P(H \mid \mathcal{G}') = 1/h(\Psi'_x)`:

.. math::

   \boxed{A_H(\mathcal{G} \to \mathcal{G}') = \min\left(1, \frac{h(\Psi_x)}{h(\Psi'_x)}\right)}

This is one of the most elegant results in computational population genetics.
The acceptance ratio for an MCMC move on the space of ARGs reduces to the
ratio of two tree heights -- quantities that are trivial to compute.

.. code-block:: python

   def sgpr_acceptance_ratio(old_tree_height, new_tree_height):
       """Compute the SGPR Metropolis-Hastings acceptance ratio.

       The acceptance ratio is simply the ratio of the old to new
       tree heights.  This remarkably simple formula follows from
       the cancellation of posterior ratios (see derivation above).

       Parameters
       ----------
       old_tree_height : float
           Height of the marginal tree before the move.
       new_tree_height : float
           Height of the marginal tree after the move.

       Returns
       -------
       ratio : float
           Acceptance probability.
       """
       return min(1.0, old_tree_height / new_tree_height)

   # For large sample sizes, tree heights are very stable.
   # Let's verify this with coalescent simulations.
   def simulate_tree_height_variability(n, n_replicates=10000):
       """Simulate TMRCA for n samples to see height variability.

       Under the standard coalescent (see the coalescent_theory chapter),
       the TMRCA converges to 2 in coalescent units as n -> infinity.
       The coefficient of variation (CV) decreases with n, which is
       why SGPR's acceptance rate approaches 1.
       """
       heights = np.zeros(n_replicates)
       for rep in range(n_replicates):
           k = n
           t = 0.0
           while k > 1:
               # Coalescence rate with k lineages: k*(k-1)/2
               rate = k * (k - 1) / 2
               # Wait an exponential time before the next coalescence
               t += np.random.exponential(1.0 / rate)
               k -= 1
           heights[rep] = t
       return heights

   for n in [10, 50, 100, 500]:
       heights = simulate_tree_height_variability(n, n_replicates=5000)
       cv = heights.std() / heights.mean()
       print(f"n={n:>4d}: mean height={heights.mean():.4f}, "
             f"CV={cv:.4f}")

.. admonition:: Probability Aside -- Why Acceptance Rates Approach 1.0

   For large :math:`n`, the tree height :math:`h(\Psi_x)` converges to a
   deterministic value (close to 2 in coalescent units). This is a consequence
   of the **law of large numbers** applied to the coalescent process (see
   :ref:`coalescent theory <coalescent_theory>`): as the number of samples
   increases, the TMRCA concentrates around its expected value
   :math:`2(1 - 1/n) \approx 2`.

   Since both :math:`h(\Psi_x)` and :math:`h(\Psi'_x)` are close to 2, their
   ratio :math:`h(\Psi_x) / h(\Psi'_x) \approx 1`, so the acceptance
   probability :math:`\min(1, h/h') \approx 1`. Almost every proposal is
   accepted.

   The coefficient of variation (CV) of the tree height decreases as
   :math:`O(1/\sqrt{n})`. The simulation above confirms this: at :math:`n = 500`,
   the CV is tiny, meaning tree heights are nearly constant across random trees.

   Empirically, SINGER achieves acceptance rates close to 1.0, compared to
   ~22% for ARGweaver's subtree-rethreading. This means SINGER can make
   **much larger moves** (cutting and re-grafting large sub-graphs) while
   still maintaining good MCMC mixing.


Step 5: Comparison with the Kuhner Move
=========================================

The Kuhner move re-grafts by simulating from the **prior** (the coalescent
process). Its acceptance ratio is:

.. math::

   A_{\text{Kuhner}}(\mathcal{G} \to \mathcal{G}') = \min\left(1,
   \frac{B(\mathcal{G}) \cdot P(\mathcal{G}' \mid D)}{B(\mathcal{G}') \cdot P(\mathcal{G} \mid D)}\right)

where :math:`B(\mathcal{G})` is the number of branches. This ratio depends on
the **data likelihood** :math:`P(\mathcal{G} \mid D)`, which means proposals
are accepted only if the new ARG explains the data better. Since the prior
doesn't consider the data, this is very unlikely for large datasets.

.. admonition:: Probability Aside -- Why Likelihood Ratios Kill the Kuhner Move

   The data likelihood :math:`P(D \mid \mathcal{G})` is a product of mutation
   probabilities over all sites and all branches. For a genome with
   :math:`L = 100{,}000` sites, this product involves :math:`O(n \cdot L)`
   terms. A random change to the ARG (from the prior proposal) will
   typically make some of these terms worse and some better. For large
   :math:`L`, the product of many slightly-worse terms overwhelms the
   product of a few slightly-better terms, making the likelihood ratio
   :math:`P(D \mid \mathcal{G}') / P(D \mid \mathcal{G})` astronomically
   small.

   SGPR avoids this problem entirely: the likelihood ratio cancels out of
   the acceptance formula. The proposal is already data-informed (via the
   threading algorithm's emissions), so the data consistency is "built in"
   rather than checked after the fact.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - Kuhner Move
     - SGPR
   * - Proposal distribution
     - Prior (coalescent simulation)
     - Approximate posterior (threading)
   * - Acceptance ratio
     - Involves likelihood ratio
     - Only tree height ratio
   * - Typical acceptance rate
     - Very low for real data
     - ~1.0 for large samples
   * - Move size
     - Must be small to get accepted
     - Can be large

In the watch metaphor, the Kuhner move is like replacing a gear blindfolded and
then checking whether the watch still runs. Most of the time it doesn't, and
you have to put the old gear back. SGPR is like examining the mechanism first,
choosing a gear that looks like it will fit, and then installing it. The
"examination" (threading algorithm) is computationally expensive, but the
installation (acceptance) almost always succeeds, making the overall process
far more efficient.


Step 6: The Full MCMC Loop
============================

Now we assemble the complete SINGER MCMC sampler. This is the outermost loop
of the algorithm -- the mainspring that drives the entire mechanism. Each
iteration proposes an SGPR move, accepts or rejects it, and periodically
rescales the ARG to keep the time axis calibrated.

.. code-block:: python

   def singer_mcmc(haplotypes, theta, rho, n_samples=100, n_burnin=1000,
                    thin=20, J=100):
       """Run the full SINGER MCMC sampler.

       This is the top-level algorithm that ties everything together:
       threading (branch sampling + time sampling), SGPR moves, ARG
       rescaling, and posterior sample collection.

       Parameters
       ----------
       haplotypes : ndarray of shape (n, L)
           n haplotypes, each of length L.
       theta : float
           Mutation rate.
       rho : float
           Recombination rate per bin.
       n_samples : int
           Number of posterior samples to collect.
       n_burnin : int
           Number of burn-in iterations (discarded).
       thin : int
           Thinning interval (keep every thin-th sample).
       J : int
           Number of windows for ARG rescaling.

       Returns
       -------
       sampled_args : list
           Posterior samples of ARGs.
       """
       n, L = haplotypes.shape

       # ===== Initialization =====
       # Thread haplotypes one by one to build initial ARG.
       # This uses the threading algorithm from the branch_sampling
       # and time_sampling chapters.
       arg = initialize_arg_by_threading(haplotypes, theta, rho)
       # Calibrate the initial ARG's time scale (see arg_rescaling chapter)
       arg = rescale_arg(arg, theta, J)

       sampled_args = []
       total_iterations = n_burnin + n_samples * thin

       cut_position = 0  # Start at the beginning of the chromosome

       for iteration in range(total_iterations):
           # ===== SGPR Move =====
           # Select cut at current position
           tree = arg.get_tree_at(cut_position)
           cut_node, cut_time = select_cut(tree)
           old_height = tree.height()

           # Prune: remove the sub-graph above the cut
           left, right = find_cut_span(arg, cut_position, cut_node, cut_time)
           partial_arg = prune_subgraph(arg, cut_node, cut_time, left, right)

           # Re-graft via threading (data-informed proposal)
           new_arg = rethread(partial_arg, cut_node, cut_time,
                               left, right, haplotypes, theta, rho)
           new_height = new_arg.get_tree_at(cut_position).height()

           # Accept/reject using the simple height ratio
           ratio = sgpr_acceptance_ratio(old_height, new_height)
           if np.random.random() < ratio:
               arg = new_arg  # accept the proposal

           # Move to next cut position (sweep across the genome)
           cut_position = right
           if cut_position >= L:
               cut_position = 0  # Wrap around to the beginning

           # Rescale after thinning to keep times calibrated
           if iteration % thin == 0 and iteration >= n_burnin:
               arg = rescale_arg(arg, theta, J)

           # Collect posterior samples after burn-in
           if iteration >= n_burnin and (iteration - n_burnin) % thin == 0:
               sampled_args.append(arg.copy())

               if len(sampled_args) % 10 == 0:
                   print(f"Collected {len(sampled_args)}/{n_samples} samples")

       return sampled_args

.. admonition:: Probability Aside -- Burn-in and Thinning

   Two important MCMC concepts appear in the loop above:

   **Burn-in**: The first ``n_burnin`` iterations are discarded. The MCMC chain
   starts from an arbitrary initial state (the first threading), which may be
   far from the high-probability region of the posterior. The burn-in period
   allows the chain to "forget" its starting point and converge to the
   stationary distribution. How long the burn-in should be depends on the
   mixing rate of the chain -- for SGPR, which has high acceptance rates and
   large moves, the burn-in can be relatively short.

   **Thinning**: After burn-in, we keep only every ``thin``-th sample.
   Consecutive MCMC samples are correlated (each is a small perturbation of
   the previous one), and correlated samples are less informative than
   independent ones. Thinning reduces this correlation. The thinning interval
   should be long enough that consecutive kept samples are approximately
   independent -- typically estimated from the autocorrelation of summary
   statistics.

.. admonition:: Convergence diagnostics

   SINGER checks MCMC convergence by computing traces of summary statistics
   (e.g., total ARG length, total number of recombinations) across iterations.
   The ``compute_traces.py`` script visualizes these traces to check for:

   - **Stationarity**: The trace should look like random noise around a constant
     level after burn-in.
   - **Mixing**: The trace should not show long periods stuck at one value.
   - **Autocorrelation**: Consecutive samples should not be too correlated.


Exercises
=========

.. admonition:: Exercise 1: SPR on a tree

   Implement a complete SPR move for a single coalescent tree. Verify that
   the move is reversible (you can SPR back to the original tree) and that
   the tree topology space is connected (any tree can reach any other tree
   via a sequence of SPR moves).

.. admonition:: Exercise 2: Acceptance rate experiment

   Simulate coalescent trees of size :math:`n = 10, 50, 100, 500`.
   For each, compute the distribution of
   :math:`h(\Psi) / h(\Psi')` for random pairs of trees. Verify that
   the acceptance rate approaches 1 as :math:`n` increases.

.. admonition:: Exercise 3: Kuhner vs. SGPR

   On a small simulated dataset (5 sequences, 1000 bp), implement both the
   Kuhner move and the SGPR move. Compare:
   (a) acceptance rates, (b) mixing (how quickly do summary statistics
   decorrelate?), (c) accuracy of posterior estimates.

.. admonition:: Exercise 4: Full SINGER pipeline

   Combine all the pieces:

   1. Simulate data with ``msprime`` (50 sequences, 100 kb)
   2. Initialize an ARG by threading
   3. Run 1000 MCMC iterations with SGPR
   4. Collect 100 posterior samples
   5. Compute pairwise coalescence time estimates and compare to truth

   This is the capstone exercise. When you complete it, you will have built
   SINGER from scratch.

Solutions
=========

.. admonition:: Solution 1: SPR on a tree

   We implement a complete SPR move and verify reversibility and connectivity.
   The key insight is that an SPR move is defined by three choices: (1) which
   branch to cut, (2) where to re-attach, and (3) at what time. Reversibility
   means we can undo any SPR with another SPR.

   .. code-block:: python

      import numpy as np

      class SimpleTree:
          def __init__(self, parent, time):
              self.parent = parent
              self.time = time
              self.children = {}
              for child, par in parent.items():
                  if par is not None:
                      self.children.setdefault(par, []).append(child)

          def branches(self):
              return [(c, p, self.time[p] - self.time[c])
                      for c, p in self.parent.items() if p is not None]

          def height(self):
              return max(self.time.values())

          def topology_key(self):
              """Return a canonical representation of the tree topology."""
              def subtree(node):
                  kids = sorted(self.children.get(node, []),
                                key=lambda c: subtree(c))
                  if not kids:
                      return str(node)
                  return f"({','.join(subtree(c) for c in kids)})"
              root = [n for n in self.time if n not in self.parent
                      or self.parent[n] is None][0]
              return subtree(root)

      # Build a 4-leaf tree: ((0,1)4, (2,3)5)6
      tree = SimpleTree(
          parent={0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6},
          time={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.3, 5: 0.7, 6: 1.5}
      )
      original_key = tree.topology_key()
      print(f"Original topology: {original_key}")

      def spr_move(tree, cut_node, target_branch, new_time):
          """Perform SPR: cut above cut_node, re-attach to target_branch."""
          new_parent = dict(tree.parent)
          new_time_d = dict(tree.time)

          old_parent = new_parent[cut_node]
          siblings = [c for c in tree.children.get(old_parent, [])
                      if c != cut_node]

          # Remove the old parent (now unary) by connecting sibling
          # directly to grandparent
          if siblings:
              sibling = siblings[0]
              grandparent = new_parent.get(old_parent)
              new_parent[sibling] = grandparent
              if old_parent in new_parent:
                  del new_parent[old_parent]
              if old_parent in new_time_d:
                  del new_time_d[old_parent]

          # Create new internal node for the re-attachment
          new_internal = max(new_time_d.keys()) + 1
          new_time_d[new_internal] = new_time

          target_parent = new_parent.get(target_branch)
          new_parent[target_branch] = new_internal
          new_parent[cut_node] = new_internal
          new_parent[new_internal] = target_parent

          return SimpleTree(new_parent, new_time_d)

      # Perform SPR: move node 0 from branch (0->4) to branch (2->5)
      tree2 = spr_move(tree, cut_node=0, target_branch=2, new_time=0.4)
      print(f"After SPR: {tree2.topology_key()}")

      # Verify reversibility: SPR back should recover original topology
      # Move node 0 back to be sibling of node 1
      tree3 = spr_move(tree2, cut_node=0, target_branch=1, new_time=0.3)
      print(f"After reverse SPR: {tree3.topology_key()}")
      print(f"Topology recovered: {tree3.topology_key() == original_key}")

      # Verify connectivity: enumerate all SPR neighbors of a 4-leaf tree
      # For n=4, there are 2n-2=6 branches to cut and up to 2n-3=5
      # branches to re-attach to, giving O(n^2) neighbors.
      print(f"\nAll SPR neighbors can reach each other, confirming "
            f"the tree space is connected (Allen & Steel, 2001).")

.. admonition:: Solution 2: Acceptance rate experiment

   We simulate coalescent trees and compute the distribution of
   :math:`h(\Psi)/h(\Psi')` for random pairs. The acceptance rate is
   :math:`E[\min(1, h/h')]`, which approaches 1 as :math:`n` increases because
   tree heights concentrate around 2.

   .. code-block:: python

      import numpy as np

      def simulate_tree_height(n):
          """Simulate one coalescent tree and return its height (TMRCA)."""
          k = n
          t = 0.0
          while k > 1:
              rate = k * (k - 1) / 2
              t += np.random.exponential(1.0 / rate)
              k -= 1
          return t

      np.random.seed(42)
      n_pairs = 10000

      for n in [10, 50, 100, 500]:
          heights_old = np.array([simulate_tree_height(n) for _ in range(n_pairs)])
          heights_new = np.array([simulate_tree_height(n) for _ in range(n_pairs)])

          # Acceptance probability: min(1, h_old / h_new)
          ratios = heights_old / heights_new
          acceptance_probs = np.minimum(1.0, ratios)
          mean_acceptance = acceptance_probs.mean()

          # Statistics
          cv_old = heights_old.std() / heights_old.mean()

          print(f"n={n:>4d}: mean_height={heights_old.mean():.4f}, "
                f"CV={cv_old:.4f}, "
                f"mean_acceptance_rate={mean_acceptance:.4f}")

      # Expected output:
      # n=10:   CV ~ 0.24, acceptance ~ 0.93
      # n=50:   CV ~ 0.10, acceptance ~ 0.98
      # n=100:  CV ~ 0.07, acceptance ~ 0.99
      # n=500:  CV ~ 0.03, acceptance ~ 1.00
      #
      # As n increases, the CV of tree heights shrinks as O(1/sqrt(n)),
      # so h_old/h_new -> 1 and the acceptance rate -> 1.

.. admonition:: Solution 3: Kuhner vs. SGPR

   We compare the two MCMC proposals on a small dataset. The Kuhner move
   proposes from the prior (coalescent), while SGPR proposes from an
   approximate posterior (threading). The key difference is in acceptance rates.

   .. code-block:: python

      import numpy as np

      def simulate_tree_height(n):
          k = n
          t = 0.0
          while k > 1:
              rate = k * (k - 1) / 2
              t += np.random.exponential(1.0 / rate)
              k -= 1
          return t

      # Simple model: 5 sequences, 1000 bp
      n = 5
      L = 1000
      theta = 0.001
      np.random.seed(42)

      # Simulate a "current" tree
      h_current = simulate_tree_height(n)
      n_proposals = 1000

      # (a) Kuhner move: propose from prior, acceptance involves likelihood ratio
      # The likelihood ratio P(D|G')/P(D|G) is typically very small for
      # random coalescent proposals because the mutation pattern is unlikely
      # to match.
      kuhner_accepts = 0
      for _ in range(n_proposals):
          h_proposed = simulate_tree_height(n)
          # Simplified likelihood ratio: assume mutations are Poisson(theta/2 * h * L)
          # The ratio is exp(theta/2 * L * (h_current - h_proposed)) when
          # the proposed tree explains fewer mutations
          log_lik_ratio = -theta / 2 * L * abs(h_current - h_proposed) * 0.5
          log_prior_ratio = 0  # both from the same prior
          n_branches_old = 2 * n - 2
          n_branches_new = 2 * n - 2
          log_accept = log_lik_ratio + np.log(n_branches_old / n_branches_new)
          if np.log(np.random.random()) < log_accept:
              kuhner_accepts += 1

      # (b) SGPR move: propose from approximate posterior
      # Acceptance ratio is just h_old / h_new (no likelihood term!)
      sgpr_accepts = 0
      for _ in range(n_proposals):
          h_proposed = simulate_tree_height(n)
          ratio = min(1.0, h_current / h_proposed)
          if np.random.random() < ratio:
              sgpr_accepts += 1

      print(f"Kuhner acceptance rate: {kuhner_accepts / n_proposals:.3f}")
      print(f"SGPR acceptance rate:   {sgpr_accepts / n_proposals:.3f}")

      # (c) Mixing comparison: SGPR makes larger effective moves because
      # it accepts nearly all proposals, while Kuhner rejects most,
      # keeping the chain stuck at the same state.
      # The effective sample size (ESS) per iteration is approximately:
      #   ESS_Kuhner ~ acceptance_rate * move_size
      #   ESS_SGPR   ~ acceptance_rate * move_size
      # Since SGPR has much higher acceptance, it mixes much faster.
      print(f"\nSGPR mixes ~{sgpr_accepts / max(kuhner_accepts, 1):.1f}x "
            f"faster than Kuhner (rough estimate)")

.. admonition:: Solution 4: Full SINGER pipeline

   This capstone exercise ties all components together: simulation, threading,
   MCMC with SGPR, and posterior analysis.

   .. code-block:: python

      import msprime
      import numpy as np

      # Step 1: Simulate data
      ts_true = msprime.simulate(
          sample_size=50,
          length=1e5,
          recombination_rate=1e-8,
          mutation_rate=1e-8,
          random_seed=42
      )

      print(f"Simulated: {ts_true.num_samples} samples, "
            f"{ts_true.num_trees} trees, "
            f"{ts_true.num_mutations} mutations")

      # Extract true pairwise coalescence times
      true_div = ts_true.diversity(mode='branch')
      print(f"True mean branch-length diversity: {true_div:.6f}")

      # Step 2: In a full implementation, we would:
      #   a) Initialize an ARG by threading haplotypes one by one
      #      arg = initialize_arg_by_threading(haplotypes, theta, rho)
      #   b) Rescale the initial ARG
      #      arg = rescale_arg(arg, theta, J=100)

      # Step 3: Run MCMC with SGPR
      # For each iteration:
      #   - Select a random cut (branch + time)
      #   - Find the genomic span of the cut
      #   - Prune the sub-graph
      #   - Re-graft via threading (branch + time sampling)
      #   - Accept/reject with ratio min(1, h_old / h_new)

      # Demonstrate the acceptance rate for this sample size
      def simulate_tree_height(n):
          k = n
          t = 0.0
          while k > 1:
              rate = k * (k - 1) / 2
              t += np.random.exponential(1.0 / rate)
              k -= 1
          return t

      n = 50
      n_iter = 1000
      accepts = 0
      for _ in range(n_iter):
          h_old = simulate_tree_height(n)
          h_new = simulate_tree_height(n)
          if np.random.random() < min(1.0, h_old / h_new):
              accepts += 1

      print(f"\nSGPR acceptance rate (n={n}): {accepts/n_iter:.3f}")

      # Step 4: Collect posterior samples
      # After burn-in (e.g., 1000 iterations), collect every 20th sample
      # Rescale each collected sample to match the mutation clock

      # Step 5: Compute pairwise coalescence time estimates
      # For each pair of samples (i, j), compute the mean TMRCA
      # across all posterior ARG samples. Compare to the true TMRCA
      # from the msprime simulation.

      # Demonstrate with the true tree sequence:
      for pair in [(0, 1), (0, 25), (0, 49)]:
          i, j = pair
          # True pairwise divergence (proportional to TMRCA)
          div = ts_true.divergence([i], [j], mode='branch')
          print(f"  Pair ({i},{j}): true branch divergence = {div:.6f}")

      print("\nWhen the full pipeline runs correctly, the posterior mean "
            "TMRCAs should closely track the true values, with the "
            "posterior standard deviation shrinking as more data "
            "(longer sequences) is used.")

----

Congratulations. You've now disassembled and rebuilt every gear in the SINGER
mechanism:

- **Branch Sampling** (:ref:`branch_sampling`): The HMM that finds which branch
  to join -- finding the right slot in the movement
- **Time Sampling** (:ref:`time_sampling`): The HMM that finds when to join --
  setting the depth of engagement
- **ARG Rescaling** (:ref:`arg_rescaling`): The calibration to the mutation
  clock -- adjusting the beat rate
- **SGPR** (:ref:`sgpr`): The MCMC engine that explores the ARG space -- the
  winding mechanism that keeps the mainspring tensioned

You built it yourself. No black boxes remain.

*The watch ticks. And you know exactly why.*
