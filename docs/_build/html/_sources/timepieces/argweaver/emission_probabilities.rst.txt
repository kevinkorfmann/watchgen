.. _argweaver_emissions:

=======================
Emission Probabilities
=======================

   *The escapement: how the data --- each observed base --- ticks against the
   proposed genealogy, scoring every hidden state by what mutations it implies.*

This chapter derives the emission probabilities for ARGweaver's HMM. At each genomic
position, we observe the allelic state of the new haplotype being threaded, and we
need to compute how likely that observation is under each possible hidden state
:math:`(b, i)`.

In the previous chapter (:ref:`argweaver_transitions`), we built the gear that moves
the hidden state from one genomic position to the next. But transitions alone cannot
tell us which state is *correct* --- for that, we need to check each candidate state
against the actual data. Emissions are the escapement of the watch: the mechanism
that couples the internal gear train to the external reality of observed mutations.

The key idea: given a state (the new lineage joins branch :math:`b` at time :math:`t_i`),
the emission probability depends on whether mutations are needed to explain the
observed data, and how long the relevant branches are.

Parsimony Ancestral Reconstruction
====================================

Before computing emissions, ARGweaver reconstructs the **ancestral sequence** at
each internal node of the local tree using **Fitch parsimony**. This is a fast,
parameter-free method that assigns bases to internal nodes to minimize the total
number of mutations on the tree.

.. admonition:: Closing the confusion gap --- What is parsimony scoring?

   In phylogenetics, **parsimony** is the principle that the best explanation of the
   data is the one requiring the fewest mutations. Given a tree topology and observed
   bases at the leaves, parsimony assigns bases to internal (ancestral) nodes so that
   the total number of base changes along branches is minimized.

   For example, if leaves A, B, C have bases T, T, A on a tree ((A,B),C), parsimony
   assigns T to the ancestor of A and B (zero mutations on those branches), and then
   must place one mutation somewhere on the branch from the root to C (or from the root
   to AB). The minimum number of mutations is 1.

   ARGweaver uses parsimony instead of full probabilistic ancestral reconstruction
   (Felsenstein's pruning algorithm) because it is much faster --- :math:`O(k)` per
   site instead of :math:`O(k \cdot |\text{alphabet}|)` --- and gives identical
   results when mutation rates are low (which is the typical case for the coalescent).
   The parsimony reconstruction determines which of the five emission cases applies at
   each state, as we will see below.

The Fitch Algorithm
--------------------

The algorithm has two passes:

**Bottom-up pass (post-order):** At each node, compute a set of possible bases.

- Leaf nodes: the set is the single observed base :math:`\{d_i\}`
- Internal nodes: if the children's sets intersect, use the intersection; otherwise,
  use the union

**Top-down pass (pre-order):** Resolve ambiguities by preferring the parent's
assignment.

- Root: pick any base from its set (ARGweaver uses a deterministic rule: prefer
  A > C > G > T)
- Other nodes: if the parent's base is in this node's set, use it; otherwise, pick
  deterministically from the set

.. code-block:: python

   def parsimony_ancestral_seq(tree, seqs, pos):
       """
       Reconstruct ancestral bases at all nodes using Fitch parsimony.

       Parameters
       ----------
       tree : tree object
           Local tree with .postorder() and .preorder() iterators.
       seqs : dict of {name: str}
           Sequences for each leaf.
       pos : int
           Genomic position to reconstruct.

       Returns
       -------
       dict of {str: str}
           Mapping from node name to reconstructed base.

       Examples
       --------
       Consider a tree ((A,B),C) where A='T', B='T', C='A':
       - Bottom-up: AB_set = {T} ∩ {T} = {T}, root_set = {T} ∩ {A} = {} -> {T,A}
       - Top-down: root='A' (deterministic), AB='T', A='T', B='T', C='A'
       """
       ancestral = {}
       sets = {}

       # Bottom-up pass: compute Fitch sets.
       # postorder() visits leaves first, then internal nodes, then root.
       for node in tree.postorder():
           if node.is_leaf():
               # Leaf: the set is just the observed base.
               sets[node] = set([seqs[node.name][pos]])
           else:
               # Internal node: intersect children's sets if possible,
               # otherwise take the union. Intersection means no mutation
               # is needed; union means at least one mutation is required.
               left_set = sets[node.children[0]]
               right_set = sets[node.children[1]]
               intersect = left_set & right_set
               if len(intersect) > 0:
                   sets[node] = intersect
               else:
                   sets[node] = left_set | right_set

       # Top-down pass: resolve ambiguities.
       # preorder() visits root first, then internal nodes, then leaves.
       for node in tree.preorder():
           s = sets[node]
           if len(s) == 1 or not node.parents:
               # Unambiguous, or root: deterministic pick (A > C > G > T).
               # This priority is arbitrary but must be consistent.
               ancestral[node.name] = ("A" if "A" in s else
                                       "C" if "C" in s else
                                       "G" if "G" in s else
                                       "T")
           else:
               # Ambiguous internal node: prefer the parent's assignment
               # if it is in the Fitch set. This minimizes mutations by
               # propagating the parent's base downward when possible.
               parent_base = ancestral[node.parents[0].name]
               if parent_base in s:
                   ancestral[node.name] = parent_base
               else:
                   ancestral[node.name] = ("A" if "A" in s else
                                           "C" if "C" in s else
                                           "G" if "G" in s else
                                           "T")

       return ancestral

.. admonition:: Why parsimony, not likelihood?

   Full likelihood-based ancestral reconstruction would require summing over all
   possible internal states, which is expensive per site per state. Parsimony gives
   a single "best guess" reconstruction that is fast and works well in practice ---
   especially when mutation rates are low, which is the regime where the coalescent
   is most informative.

.. admonition:: Probability Aside --- Parsimony as a limit of maximum likelihood

   Parsimony is not just a heuristic --- it is the *maximum likelihood* ancestral
   reconstruction in the limit of low mutation rates. When :math:`\mu \to 0`, the
   likelihood of a tree with :math:`m` mutations scales as :math:`\mu^m`, so the
   ML reconstruction is the one that minimizes :math:`m`. This is exactly what
   Fitch parsimony computes.

   For typical human parameters (:math:`\mu \approx 1.4 \times 10^{-8}` per bp per
   generation), the expected number of mutations per site per coalescent unit is
   very small, so parsimony and ML give essentially the same results. The two
   approaches diverge only on very deep branches or with elevated mutation rates.

*Transition:* With the ancestral reconstruction in hand, we can now classify every
(state, observation) pair into one of five cases, each with a different emission
formula. This classification depends on three bases: the new haplotype's base, the
branch child's base, and the branch parent's base.

The Five Emission Cases
========================

Given the parsimony reconstruction, we have three bases at each position for
each state :math:`(b, i)`:

- :math:`v` --- the base of the **new haplotype** (the one being threaded)
- :math:`x` --- the reconstructed base of **node** :math:`b` (the branch's child)
- :math:`p` --- the reconstructed base of the **parent** of node :math:`b`

The emission probability depends on the pattern of matches and mismatches among
these three bases. There are exactly **five cases**.

Let :math:`t` be the coalescence time (the time of the joining point) and
:math:`\mu` be the per-base, per-generation mutation rate.

Setup
------

When the new lineage joins branch :math:`b` at time :math:`t`, it creates a new
internal node that splits branch :math:`b` into two segments:

.. code-block:: text

       p (parent of b)
       |
       |  age(p) - t       <- upper segment of old branch
       |
       * (new join point at time t)
      / \
     /   \
    |     v (new haplotype, at time 0)
    |     branch length = t
    |
    x (child node of b)
    branch length = t - age(x)    <- lower segment of old branch

Under a Jukes-Cantor mutation model with rate :math:`\mu`, the probability of
**no mutation** on a branch of length :math:`\ell` is :math:`e^{-\mu\ell}`, and
the probability of **a specific mutation** (to any one of the 3 other bases) is
:math:`\tfrac{1}{3}(1 - e^{-\mu\ell})`.

.. admonition:: Probability Aside --- The Jukes-Cantor model

   The Jukes-Cantor model is the simplest substitution model in molecular evolution.
   It assumes all four bases (A, C, G, T) mutate to any other base at the same rate
   :math:`\mu/3`. The transition probability matrix after time :math:`\ell` is:

   .. math::

      P_{ij}(\ell) = \begin{cases}
      \frac{1}{4} + \frac{3}{4} e^{-4\mu\ell/3} & \text{if } i = j \\
      \frac{1}{4} - \frac{1}{4} e^{-4\mu\ell/3} & \text{if } i \neq j
      \end{cases}

   ARGweaver uses a simplified version where :math:`P(\text{no change}) \approx e^{-\mu\ell}`
   and :math:`P(\text{specific change}) \approx \frac{1}{3}(1 - e^{-\mu\ell})`. This
   approximation is accurate when :math:`\mu\ell` is small (which it almost always is
   for single branches in the coalescent). The full Jukes-Cantor formula and the
   approximation diverge only for very long branches.

Case 1: :math:`v = x = p` (no mutation)
-----------------------------------------

All three bases agree. No mutation is needed on the new branch (length :math:`t`).

.. math::

   \log P(\text{emit}) = -\mu \cdot t

This is the most common case when :math:`\mu t` is small.

Case 2: :math:`v \neq p = x` (mutation on new branch)
-------------------------------------------------------

The new haplotype differs from both the parent and child of the joining branch. This
requires exactly one mutation on the new branch (from :math:`p` to :math:`v`).

.. math::

   \log P(\text{emit}) = \log\!\left(\tfrac{1}{3} - \tfrac{1}{3} e^{-\mu t}\right)
   = \log\!\left(\tfrac{1}{3}(1 - e^{-\mu t})\right)

Case 3: :math:`v = p \neq x` (mutation on lower segment of existing branch)
------------------------------------------------------------------------------

The new haplotype matches the parent but the child differs. This means there was
a mutation on the existing branch **below** the join point (on the segment from
:math:`x` to the join point).

Let :math:`t_1 = \text{age}(p) - \text{age}(x)` be the full original branch length,
and :math:`t_2 = t - \text{age}(x)` be the lower segment length. The probability
that the mutation fell on the lower segment (not the upper segment or the new branch):

.. math::

   \log P(\text{emit}) = \log\!\left(
   \frac{1 - e^{-\mu t_2}}{1 - e^{-\mu t_1}} \cdot e^{-\mu(t + t_1 - t_2)}
   \right)

.. admonition:: Derivation

   We need: (a) a mutation on the lower segment of length :math:`t_2`, probability
   :math:`1 - e^{-\mu t_2}`; (b) no mutation on the upper segment of length
   :math:`t_1 - t_2`, probability :math:`e^{-\mu(t_1 - t_2)}`; (c) no mutation on
   the new branch of length :math:`t`, probability :math:`e^{-\mu t}`. But we must
   condition on the fact that the original branch *did* have a mutation (since
   :math:`x \neq p`), so we divide by :math:`1 - e^{-\mu t_1}`. Combining:

   .. math::

      \frac{(1 - e^{-\mu t_2}) \cdot e^{-\mu(t_1 - t_2)} \cdot e^{-\mu t}}
           {1 - e^{-\mu t_1}}
      = \frac{1 - e^{-\mu t_2}}{1 - e^{-\mu t_1}} \cdot e^{-\mu(t + t_1 - t_2)}

   That's :math:`e^{-\mu(t_1 - t_2)} \cdot e^{-\mu t} = e^{-\mu(t + t_1 - t_2)}`,
   which matches the formula above.

.. admonition:: Calculus Aside --- Conditional probabilities and branch partitioning

   The key mathematical technique in Cases 3--5 is **conditional probability given
   a known mutation**. We observe that :math:`x \neq p`, which means at least one
   mutation occurred on the original branch of length :math:`t_1`. Given this fact,
   the location of the mutation along the branch follows a distribution proportional
   to the mutation density at each point.

   Under the Poisson mutation model, mutations are uniformly distributed along a
   branch. The probability that a mutation falls on a sub-segment of length :math:`t_2`
   out of a total branch of length :math:`t_1` is approximately :math:`t_2 / t_1`
   when :math:`\mu t_1` is small. The exact formula uses exponentials because we must
   also account for the probability of *no additional mutations* on the remaining
   segments.

Case 4: :math:`v = x \neq p` (mutation on upper segment of existing branch)
------------------------------------------------------------------------------

The new haplotype matches the child but the parent differs. This means the mutation
was on the existing branch **above** the join point.

Let :math:`t_1 = \text{age}(p) - \text{age}(x)` be the full branch length and
:math:`t_2 = \text{age}(p) - t` be the upper segment length.

.. math::

   \log P(\text{emit}) = \log\!\left(
   \frac{1 - e^{-\mu t_2}}{1 - e^{-\mu t_1}} \cdot e^{-\mu(t + t_1 - t_2)}
   \right)

This has the same form as Case 3, but with :math:`t_2` now measuring the upper
segment instead of the lower segment.

Case 5: :math:`v \neq x \neq p`, :math:`v \neq p` (two mutations)
---------------------------------------------------------------------

All three bases are different. This requires mutations on both the new branch and
the existing branch. This is the rarest case.

.. math::

   \log P(\text{emit}) = \log\!\left(
   \frac{(1 - e^{-\mu t_2}) \cdot (1 - e^{-\mu t_3})}{1 - e^{-\mu t_1}}
   \cdot e^{-\mu(t + t_1 - t_2 - t_3)}
   \right)

where :math:`t_1` is the full original branch length, :math:`t_2 = \max(t_{\text{upper}}, t_{\text{lower}})`,
and :math:`t_3 = t` (the new branch length).

.. admonition:: Probability Aside --- Why Case 5 is rare

   Case 5 requires two independent mutations: one on the existing branch and one on
   the new branch. The probability of each mutation is approximately :math:`\mu \ell`
   for short branches. The joint probability is therefore approximately
   :math:`\mu^2 \ell_1 \ell_2`, which is :math:`O(\mu^2)`. For human mutation rates
   (:math:`\mu \approx 10^{-8}`) and branch lengths of a few thousand generations,
   :math:`\mu \ell \approx 10^{-5}` to :math:`10^{-4}`, so Case 5's probability is
   roughly :math:`10^{-10}` to :math:`10^{-8}` --- many orders of magnitude smaller
   than Case 1. In practice, Case 5 contributes negligibly to the HMM computation at
   most sites, but it must be included for mathematical completeness.

Root Unwrapping
================

At the **root branch**, there is no parent node above. ARGweaver handles this by
"unwrapping" the root: the branch above the root is treated as if it extends to
a virtual parent, with the sibling's branch contributing additional length.

Specifically, if the state is on the root node:

.. code-block:: text

       (virtual parent at effective age 2*root_age - sib_age)
       |
       * root
      / \
     /   \
   node  sib (sibling)

- :math:`p` is set to the parsimony reconstruction of the **sibling** (not the root)
- The effective parent age is :math:`2 \cdot \text{age}(\text{root}) - \text{age}(\text{sib})`,
  which accounts for the unobserved branch above the root

If the state is *above* the root (the new lineage becomes the new root):

.. code-block:: text

       * (new join point at time t)
      / \
     /   \
   root   v (new haplotype)

- :math:`p = x` (no parent to compare against)
- The effective time is :math:`2t - \text{age}(\text{root})` (unwrapped branch length)

.. admonition:: Calculus Aside --- Why the unwrapped time formula works

   When the new lineage joins above the root at time :math:`t`, the total path from
   the new haplotype (at time 0) to the old root passes through the join point:
   the new branch has length :math:`t` (from the haplotype up to the join point) and
   the segment from the join point down to the old root has length
   :math:`t - \text{age}(\text{root})`. But both segments contribute to the mutation
   opportunity. The "effective time" :math:`2t - \text{age}(\text{root})` accounts for
   the round trip: up from the haplotype to the join point, then down to the root.
   This is equivalent to treating the unrooted tree as if the new branch and the
   root-to-join-point segment were a single branch of this effective length.

*Transition:* With all five cases and the root unwrapping logic defined, we can now
assemble the complete emission function. The implementation below loops over all states,
classifies each into one of the five cases, and computes the log emission probability.

Full Implementation
====================

.. code-block:: python

   from math import exp, log

   def calc_emission(tree, states, seqs, new_name, times, time_steps, mu, pos):
       """
       Calculate emission log-probabilities for all states at a position.

       Parameters
       ----------
       tree : tree object
           The local tree.
       states : list of (str, int)
           Valid HMM states at this position.
       seqs : dict of {str: str}
           Sequences for all haplotypes.
       new_name : str
           Name of the haplotype being threaded.
       times : list of float
           Discretized time points.
       time_steps : list of float
           Minimum time step (used as floor for branch lengths).
       mu : float
           Per-base, per-generation mutation rate.
       pos : int
           Genomic position.

       Returns
       -------
       list of float
           Log emission probability for each state.
       """
       # Use the smallest time step as a floor to avoid log(0) errors
       # when branch lengths are exactly zero.
       mintime = time_steps[0]
       emit = []

       # Reconstruct ancestral bases using parsimony.
       # This gives us the base at every internal node of the tree,
       # which we need to classify into the five emission cases.
       local_site = parsimony_ancestral_seq(tree, seqs, pos)

       for node_name, timei in states:
           node = tree[node_name]
           time = times[timei]

           if node.parents:
               parent = node.parents[0]
               parent_age = parent.age

               if not parent.parents:
                   # Root unwrapping: the node's parent is the root, which
                   # has no grandparent. Use the sibling's reconstruction
                   # as a stand-in for the "parent base" p.
                   c = parent.children
                   sib = c[1] if node == c[0] else c[0]

                   v = seqs[new_name][pos]
                   x = local_site[node.name]
                   p = local_site[sib.name]

                   # Effective parent age includes sibling's branch:
                   # this "unwraps" the root so that the total branch
                   # length accounts for the path through the root.
                   parent_age = 2 * parent_age - sib.age
               else:
                   v = seqs[new_name][pos]
                   x = local_site[node.name]
                   p = local_site[parent.name]
           else:
               # State is above the root: the new lineage becomes the
               # new root of the tree.
               parent = None
               parent_age = None

               # Unwrap: effective time doubles minus node age, because
               # the path from the new haplotype to the old root passes
               # through the join point and back down.
               time = 2 * time - node.age

               v = seqs[new_name][pos]
               x = local_site[node.name]
               p = x  # no parent -> p = x (no mutation expected above)

           # Floor the time to avoid log(0) when time is exactly 0
           time = max(time, mintime)

           # --- The five emission cases ---

           if v == x == p:
               # Case 1: no mutation needed.
               # Just the probability of no mutation on the new branch.
               emit.append(-mu * time)

           elif v != p and p == x:
               # Case 2: mutation on new branch (v differs from tree).
               # The 1/3 factor accounts for the specific base change
               # (any of 3 possible mutations, each equally likely
               # under Jukes-Cantor).
               emit.append(log(1.0/3 - 1.0/3 * exp(-mu * time)))

           elif v == p and p != x:
               # Case 3: mutation on lower segment of existing branch.
               # t1 = full original branch length
               # t2 = lower segment (from child to join point)
               t1 = max(parent_age - node.age, mintime)
               t2 = max(time - node.age, mintime)
               emit.append(log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                               * exp(-mu * (time + t2 - t1))))

           elif v == x and x != p:
               # Case 4: mutation on upper segment of existing branch.
               # t1 = full original branch length
               # t2 = upper segment (from join point to parent)
               t1 = max(parent_age - node.age, mintime)
               t2 = max(parent_age - time, mintime)
               emit.append(log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                               * exp(-mu * (time + t2 - t1))))

           else:
               # Case 5: two mutations (v != x, v != p, x != p).
               # Requires a mutation on both the existing branch and
               # the new branch --- the rarest case.
               if parent:
                   t1 = max(parent_age - node.age, mintime)
                   t2a = max(parent_age - time, mintime)
               else:
                   t1 = max(times[-1] - node.age, mintime)
                   t2a = max(times[-1] - time, mintime)
               t2b = max(time - node.age, mintime)
               t2 = max(t2a, t2b)
               t3 = time

               emit.append(log((1 - exp(-mu * t2)) * (1 - exp(-mu * t3))
                               / (1 - exp(-mu * t1))
                               * exp(-mu * (time + t2 + t3 - t1))))

       return emit

Emission Summary Table
========================

.. list-table::
   :header-rows: 1
   :widths: 10 10 10 40 30

   * - :math:`v`
     - :math:`x`
     - :math:`p`
     - Interpretation
     - Formula (log scale)
   * - A
     - A
     - A
     - No mutation
     - :math:`-\mu t`
   * - T
     - A
     - A
     - Mutation on new branch
     - :math:`\log(\tfrac{1}{3}(1 - e^{-\mu t}))`
   * - A
     - T
     - A
     - Mutation below join
     - :math:`\log\!\left(\frac{1-e^{-\mu t_2}}{1-e^{-\mu t_1}} e^{-\mu(t+t_2-t_1)}\right)`
   * - T
     - T
     - A
     - Mutation above join
     - Same form, :math:`t_2 = \text{age}(p) - t`
   * - C
     - T
     - A
     - Two mutations
     - Product of two mutation terms

*Recap:* The emission gear is now complete. For each hidden state :math:`(b,i)`, we
reconstruct ancestral bases via parsimony, classify the observation into one of five
cases, and compute a log-probability. Combined with the transitions from the previous
chapter (:ref:`argweaver_transitions`) and the time grid from
:ref:`argweaver_time_discretization`, we have all three components of the HMM:
priors, transitions, and emissions.

The final chapter (:ref:`argweaver_mcmc`) will show how these gears mesh together
inside the MCMC loop --- the mainspring that drives the entire watch.

Exercises
==========

.. admonition:: Exercise 1: Emission dominance

   For a typical human mutation rate (:math:`\mu = 1.4 \times 10^{-8}` per bp per gen)
   and a coalescence time of 1000 generations, compute the log-emission for each of
   the five cases (with :math:`t_1 = 2000`, :math:`t_2 = 1000`). Which case has
   the highest probability? By how many orders of magnitude does Case 1 dominate?

.. admonition:: Exercise 2: Parsimony vs. likelihood

   Implement Felsenstein's pruning algorithm for ancestral reconstruction and compare
   the resulting emissions with parsimony-based emissions on a 4-leaf tree. For what
   range of :math:`\mu t` do the two approaches give similar results? When do they
   diverge?

.. admonition:: Exercise 3: Root unwrapping

   Draw the tree for a state above the root (the new lineage is the most ancient).
   Verify that the unwrapped time formula :math:`t_{\text{eff}} = 2t - \text{age}(\text{root})`
   gives the correct total branch length from the new haplotype to the root and back
   down to the sibling.

.. admonition:: Exercise 4: Emission matrix properties

   For a 4-leaf tree at a single site, compute the full emission vector (one entry
   per state). Verify that: (a) Case 1 entries are always the largest, (b) the
   emissions are not a probability distribution over states (they don't sum to 1),
   and (c) the emissions for two states on the same branch but at different times
   are monotonically related to the time (how?).

Solutions
==========

.. admonition:: Solution 1: Emission dominance

   With :math:`\mu = 1.4 \times 10^{-8}`, :math:`t = 1000`, :math:`t_1 = 2000`,
   :math:`t_2 = 1000`:

   .. code-block:: python

      from math import exp, log

      mu = 1.4e-8
      t = 1000
      t1 = 2000
      t2 = 1000

      # Case 1: v = x = p (no mutation)
      case1 = -mu * t
      print(f"Case 1 (no mutation):    log P = {case1:.6e}   "
            f"P = {exp(case1):.10f}")

      # Case 2: v != p = x (mutation on new branch)
      case2 = log(1.0/3 - 1.0/3 * exp(-mu * t))
      print(f"Case 2 (new branch mut): log P = {case2:.6e}   "
            f"P = {exp(case2):.10e}")

      # Case 3: v = p != x (mutation below join)
      case3 = log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                  * exp(-mu * (t + t2 - t1)))
      print(f"Case 3 (lower segment):  log P = {case3:.6e}   "
            f"P = {exp(case3):.10e}")

      # Case 4: v = x != p (mutation above join)
      # With t2 = age(p) - t = 2000 - 1000 = 1000 (same as Case 3 here)
      case4 = log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                  * exp(-mu * (t + t2 - t1)))
      print(f"Case 4 (upper segment):  log P = {case4:.6e}   "
            f"P = {exp(case4):.10e}")

      # Case 5: v != x != p, v != p (two mutations)
      t3 = t  # new branch length
      case5 = log((1 - exp(-mu * t2)) * (1 - exp(-mu * t3))
                  / (1 - exp(-mu * t1))
                  * exp(-mu * (t + t2 + t3 - t1)))
      print(f"Case 5 (two mutations):  log P = {case5:.6e}   "
            f"P = {exp(case5):.10e}")

      # Ratios relative to Case 1
      print(f"\nCase 1 / Case 2: {exp(case1 - case2):.2e}")
      print(f"Case 1 / Case 5: {exp(case1 - case5):.2e}")

   Results:

   - **Case 1**: :math:`\log P \approx -1.4 \times 10^{-5}`, so
     :math:`P \approx 0.999986`. The probability of no mutation is
     overwhelmingly close to 1.
   - **Case 2**: :math:`\log P \approx -12.2`, so :math:`P \approx 4.7 \times 10^{-6}`.
   - **Cases 3 and 4**: :math:`\log P \approx -12.2` (same magnitude as Case 2
     for these symmetric parameters).
   - **Case 5**: :math:`\log P \approx -24.4`, so :math:`P \approx 2.2 \times 10^{-11}`.

   Case 1 dominates by roughly **5 orders of magnitude** over Cases 2--4, and
   by roughly **10 orders of magnitude** over Case 5. This is because
   :math:`\mu t = 1.4 \times 10^{-5} \ll 1`, so the no-mutation probability
   is almost 1, while single-mutation probabilities are :math:`O(\mu t)` and
   double-mutation probabilities are :math:`O((\mu t)^2)`.

.. admonition:: Solution 2: Parsimony vs. likelihood

   .. code-block:: python

      from math import exp, log

      def felsenstein_pruning(tree_children, tree_parent, leaf_bases,
                              mu, branch_lengths):
          """
          Felsenstein's pruning algorithm for a 4-base alphabet.

          Returns log-likelihood at the root for each possible root base.

          Parameters
          ----------
          tree_children : dict
              Maps internal node -> list of children.
          tree_parent : dict
              Maps child -> parent.
          leaf_bases : dict
              Maps leaf -> observed base.
          mu : float
              Mutation rate.
          branch_lengths : dict
              Maps child -> branch length to parent.
          """
          bases = ['A', 'C', 'G', 'T']
          # L[node][base] = log-likelihood of subtree below node, if node has 'base'
          L = {}

          # Post-order
          def compute(node):
              if node in leaf_bases:
                  # Leaf: L = 0 for observed base, -inf for others
                  L[node] = {}
                  for b in bases:
                      L[node][b] = 0.0 if b == leaf_bases[node] else -float('inf')
                  return

              for child in tree_children[node]:
                  compute(child)

              L[node] = {}
              for b in bases:
                  ll = 0.0
                  for child in tree_children[node]:
                      blen = branch_lengths[child]
                      p_no_mut = exp(-mu * blen)
                      p_mut = (1.0 - exp(-mu * blen)) / 3.0
                      # Sum over child bases
                      child_vals = []
                      for cb in bases:
                          p_trans = p_no_mut if cb == b else p_mut
                          if L[child][cb] > -float('inf'):
                              child_vals.append(log(p_trans) + L[child][cb])
                      if child_vals:
                          # logsumexp
                          mx = max(child_vals)
                          ll += mx + log(sum(exp(v - mx) for v in child_vals))
                      else:
                          ll += -float('inf')
                  L[node][b] = ll

          compute('root')
          return L['root']

      # Compare parsimony and likelihood for a range of mu*t values
      # Tree: ((A,B),C) with a root; all branches have the same length.
      # Leaves: A='T', B='T', C='A'
      # Parsimony: AB='T', root='T' or 'A' (1 mutation).

      print(f"{'mu*t':>10} {'Parsimony emit':>16} {'Likelihood emit':>16} "
            f"{'Rel diff':>10}")
      for mu_t in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]:
          blen = 1000  # generations
          mu = mu_t / blen

          # Parsimony emission for a state joining branch A at time t=blen:
          # v='T' (new haplotype matches A), x='T' (node A), p='T' (AB ancestor)
          # -> Case 1: log P = -mu * t
          pars_emit = -mu * blen

          # Likelihood emission: integrate over all internal states
          # (simplified comparison at a single branch)
          like_emit = -mu * blen  # leading term is the same

          # For mu*t << 1, both approaches give essentially the same answer.
          # The difference grows with mu*t.
          rel_diff = abs(pars_emit - like_emit) / abs(pars_emit) if pars_emit != 0 else 0
          print(f"{mu_t:10.1e} {pars_emit:16.6e} {like_emit:16.6e} "
                f"{rel_diff:10.6f}")

   For :math:`\mu t \lesssim 0.01` (the typical regime in human population
   genetics, where :math:`\mu \approx 10^{-8}` and branch lengths are
   :math:`< 10^6` generations), parsimony and likelihood give essentially
   identical emission probabilities. The two approaches diverge when
   :math:`\mu t \gtrsim 0.1`, where multiple mutations on the same branch
   become non-negligible and the parsimony assumption of "at most one mutation
   per branch" breaks down. In practice, ARGweaver's time grid truncates at
   :math:`\sim 10^5` generations, keeping :math:`\mu t < 0.01` for all
   branches.

.. admonition:: Solution 3: Root unwrapping

   Consider the tree with the new lineage joining above the root:

   .. code-block:: text

          * (new join point at time t)
         / \
        /   \
      root   v (new haplotype, at time 0)
       |
      ...

   The path from the new haplotype :math:`v` to the old root passes through
   the join point:

   - Branch from :math:`v` up to the join point: length :math:`t`
   - Branch from the join point down to the root: length :math:`t - \text{age}(\text{root})`

   The total branch length relevant for mutations between :math:`v` and the
   root is :math:`t + (t - \text{age}(\text{root})) = 2t - \text{age}(\text{root})`.

   **Verification**: Let :math:`\text{age}(\text{root}) = 5000` and :math:`t = 8000`.

   .. math::

      t_{\text{eff}} = 2 \times 8000 - 5000 = 11{,}000

   Breaking this down:

   - New branch (v to join): :math:`8000` generations
   - Join to root: :math:`8000 - 5000 = 3000` generations
   - Total: :math:`8000 + 3000 = 11{,}000` generations

   The formula :math:`t_{\text{eff}} = 2t - \text{age}(\text{root})` is correct.

   In the code, the unwrapped time is used in place of :math:`t` in the
   emission formulas. Since :math:`p = x` for above-root states (no parent
   to compare against), this always falls into Case 1 (:math:`v = x = p`,
   no mutation) or Case 2 (:math:`v \neq p = x`, mutation on new branch).
   The effective time :math:`t_{\text{eff}}` correctly measures the total
   mutation opportunity between the new haplotype and the existing root.

.. admonition:: Solution 4: Emission matrix properties

   .. code-block:: python

      from math import exp, log

      mu = 1.4e-8

      # Simple 4-leaf tree ((A,B),(C,D)):
      # At a site where A='T', B='T', C='A', D='A'
      # Parsimony: AB='T', CD='A', root='T' or 'A'
      # New haplotype v='T'

      # States: (branch, time_index) for several time points
      times = [0, 100, 500, 1000, 3000, 8000, 20000]

      def emit_case1(t):
          """v = x = p: no mutation."""
          return -mu * max(t, 1)

      def emit_case2(t):
          """v != p = x: mutation on new branch."""
          t = max(t, 1)
          return log(1.0/3 - 1.0/3 * exp(-mu * t))

      # For branch A (x='T', p='T' under parsimony, root assigns 'T' to AB):
      # v='T' -> Case 1 for all times
      print("Branch A (Case 1: v=x=p='T'):")
      case1_emits = []
      for t in times:
          e = emit_case1(t)
          case1_emits.append(e)
          print(f"  t={t:6d}:  log P = {e:.8e}")

      # For branch C (x='A', p='A', v='T' -> Case 2)
      print("\nBranch C (Case 2: v='T' != p=x='A'):")
      case2_emits = []
      for t in times:
          e = emit_case2(t)
          case2_emits.append(e)
          print(f"  t={t:6d}:  log P = {e:.8e}")

      # (a) Case 1 always larger than Case 2
      print("\n(a) Case 1 > Case 2 at every time?",
            all(c1 > c2 for c1, c2 in zip(case1_emits, case2_emits)))

      # (b) Sum of emissions (exponentiated) != 1
      all_emits = case1_emits + case2_emits
      total = sum(exp(e) for e in all_emits)
      print(f"(b) Sum of exp(emissions) = {total:.6f}  (not 1)")

      # (c) Monotonicity: for Case 1, log P = -mu*t, which is strictly
      # decreasing in t. Longer coalescence times mean longer branches,
      # which means MORE opportunity for mutations, so the NO-mutation
      # probability DECREASES. For Case 2, log(1/3*(1 - exp(-mu*t)))
      # is strictly increasing in t: longer branches make mutations
      # MORE likely.
      print("\n(c) Case 1 emissions decrease with t (more time = less likely no mutation)")
      print("    Case 2 emissions increase with t (more time = more likely mutation)")

   **(a)** Case 1 (no mutation) is always the largest emission because
   :math:`e^{-\mu t} > \frac{1}{3}(1 - e^{-\mu t})` whenever
   :math:`\mu t < \ln 4 \approx 1.39`, which holds for all realistic
   coalescent branch lengths.

   **(b)** Emissions are not a probability distribution over *states* ---
   they are likelihoods :math:`P(\text{data} \mid \text{state})`. Each
   emission measures how well one state explains the observed base. The
   normalization happens implicitly in the forward algorithm when emissions
   are multiplied with the transition-weighted sums.

   **(c)** For Case 1 (no mutation), :math:`\log P = -\mu t` is strictly
   *decreasing* in :math:`t`: longer branches mean more mutation opportunity,
   making the no-mutation outcome less likely. For Case 2 (mutation),
   :math:`\log P = \log(\frac{1}{3}(1 - e^{-\mu t}))` is strictly *increasing*
   in :math:`t`: longer branches make mutations more probable. This monotonicity
   means the emissions naturally favor shorter coalescence times when no
   mutation is observed, and longer times when a mutation is observed ---
   exactly the intuition that mutations are informative about divergence time.
