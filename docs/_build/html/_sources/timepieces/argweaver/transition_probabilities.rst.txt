.. _argweaver_transitions:

========================
Transition Probabilities
========================

   *The largest gear in the mechanism: how the hidden state changes from one
   genomic position to the next, governed by recombination and re-coalescence.*

This chapter derives the HMM transition probabilities --- the heart of ARGweaver's
threading algorithm. At each step along the genome, the hidden state :math:`(b, i)`
(branch and time index) can either stay the same (no recombination) or change
(recombination occurs, lineage detaches and re-coalesces elsewhere).

In the previous chapter (:ref:`argweaver_time_discretization`), we forged the tick
marks on the watch dial --- the time grid, midpoints, and lineage counts. Now we
build the gear that *uses* those tick marks: the mechanism that turns one genomic
position's state into the next position's state. If the time grid is the dial, the
transition matrix is the gear train that advances the hands.

We cover three types of transitions:

1. **Normal transitions** --- between positions within the same local tree
2. **Switch transitions** --- at positions where the partial ARG has a recombination
   breakpoint (the local tree changes)
3. **State priors** --- the initial distribution at the first position

If you need a refresher on HMM transition matrices and how they drive the forward
algorithm, see :ref:`hmms`. If the SMC process (recombination breaking a genealogy
and re-coalescence repairing it) is unclear, revisit :ref:`smc`.

Normal Transitions
===================

Between two adjacent positions in the same local tree, the state can change only
if a recombination event occurs on the threading lineage. There are two cases:

1. **No recombination**: the state stays the same
2. **Recombination**: the lineage detaches and re-coalesces, possibly at a different
   (branch, time) state

No-Recombination Probability
------------------------------

The probability that no recombination occurs between two adjacent positions is:

.. math::

   P(\text{no recomb}) = \exp(-\rho \cdot L)

where :math:`\rho` is the per-base recombination rate and :math:`L` is the total
tree length (sum of all branch lengths, including the basal branch above the root).

.. admonition:: Why tree length?

   Under the SMC, recombination is a Poisson process along the genome, with rate
   proportional to the total tree length. Longer trees have more material on which
   recombination can occur. The probability of *at least one* recombination in a
   single base pair is :math:`1 - e^{-\rho L}`.

.. admonition:: Probability Aside --- The Poisson process for recombination

   A Poisson process is the canonical model for "random events scattered along an
   axis." If events occur at rate :math:`\lambda` per unit, then the number of events
   in an interval of length :math:`d` is Poisson-distributed with mean
   :math:`\lambda d`. The probability of *zero* events is :math:`e^{-\lambda d}`.

   Here, the "axis" is the genome (measured in base pairs), and the rate is
   :math:`\lambda = \rho L` --- the per-base recombination rate :math:`\rho` times
   the total tree length :math:`L`. A recombination event can occur anywhere on any
   branch of the tree, so the total "opportunity" for recombination at a single site
   is proportional to :math:`L`.

   For adjacent sites (distance :math:`d = 1` bp), the probability of no recombination
   is :math:`e^{-\rho L}`. For typical human parameters
   (:math:`\rho \approx 10^{-8}`, :math:`L \approx 10^4` generations for 10
   haplotypes), this gives :math:`e^{-10^{-4}} \approx 0.9999` --- recombination
   between adjacent sites is very rare, which is why the diagonal of the transition
   matrix dominates.

Recombination Probability at Time :math:`k`
---------------------------------------------

Given that a recombination occurs, where does it happen? The recombination point
is uniformly distributed along the tree's branches (weighted by branch length).
In the discrete model, we sum over time intervals:

.. math::

   P(\text{recomb at time } k) = \frac{n_{\text{branches}}[k] \cdot \Delta t_k}{L}
   \cdot (1 - e^{-\rho L})

where:

- :math:`n_{\text{branches}}[k]` is the number of branches passing through time
  interval :math:`k`
- :math:`\Delta t_k = t_{k+1} - t_k` is the time step size
- :math:`L` is the total tree length
- The factor :math:`(1 - e^{-\rho L})` is the total recombination probability

.. admonition:: Derivation

   The total "branch material" in interval :math:`k` is
   :math:`n_{\text{branches}}[k] \cdot \Delta t_k`. The fraction of the tree in this
   interval is :math:`n_{\text{branches}}[k] \cdot \Delta t_k / L`. Given that a
   recombination occurs somewhere on the tree, the probability it falls in interval
   :math:`k` is proportional to this fraction.

   Note that we sum over all branches at time :math:`k`, not specific branches. The
   specific branch is selected uniformly among the :math:`n_{\text{recombs}}[k]`
   valid recombination points.

*Transition:* Once a recombination happens at some time :math:`k`, the lineage is now
"floating" --- detached from the tree and looking for somewhere to re-attach. The
re-coalescence process determines where it lands, and it follows the same coalescent
physics that governs the original tree (see :ref:`coalescent_theory`).

Re-coalescence Probability
----------------------------

After recombination at time :math:`k`, the detached lineage floats upward and must
re-coalesce with the remaining tree. This follows a discrete coalescent process:
at each time interval :math:`m \geq k`, the lineage has a chance to coalesce with
one of the :math:`n_{\text{branches}}[m]` existing lineages.

The probability of **surviving** through intervals :math:`k, k+1, \ldots, m-1`
without coalescing, then coalescing at time :math:`m`, is:

.. math::

   P(\text{recoal at } m \mid \text{recomb at } k) =
   \underbrace{\prod_{j=k}^{m-1} \exp\!\left(-\frac{\Delta t^*_j \cdot n_{\text{branches}}[j]}{2 N_j}\right)}_{\text{survival through intervals } k \text{ to } m-1}
   \cdot
   \underbrace{\left(1 - \exp\!\left(-\frac{\Delta t^*_m \cdot n_{\text{branches}}[m]}{2 N_m}\right)\right)}_{\text{coalescence in interval } m}
   \cdot
   \underbrace{\frac{1}{n_{\text{coals}}[m]}}_{\text{choose a specific branch}}

where:

- :math:`\Delta t^*_j` is the **coal time step** at index :math:`j`
  (the sub-interval width from the midpoint structure, see
  :ref:`argweaver_time_discretization`)
- :math:`N_j` is the effective population size at time :math:`j`
- :math:`n_{\text{coals}}[m]` normalizes over the possible coalescence points at
  time :math:`m`

.. admonition:: Probability Aside --- Survival times and the geometric distribution

   The re-coalescence process is a discrete analog of the exponential waiting time.
   In continuous time, the probability of *not* coalescing for duration :math:`t` is
   :math:`e^{-\lambda t}` (the survival function of an exponential). In our discrete
   model, each time interval is like a Bernoulli trial: the lineage either coalesces
   (with probability :math:`1 - e^{-\lambda_m \Delta t^*_m}`) or survives (with the
   complementary probability). Stringing these trials together gives a **geometric-like
   distribution** over the interval index :math:`m` --- analogous to flipping a
   biased coin at each tick mark on the dial until you get heads.

   The product of survival terms :math:`\prod_{j=k}^{m-1} e^{-(\cdots)}` is exactly
   the probability of "tails" on all trials from :math:`k` to :math:`m-1`, and the
   factor :math:`1 - e^{-(\cdots)}` at index :math:`m` is the probability of finally
   getting "heads."

.. admonition:: Step-by-step derivation

   1. **Coalescent rate at time** :math:`j`: In a population of size :math:`N_j`,
      the rate at which one specific lineage coalesces with any of
      :math:`n_{\text{branches}}[j]` others is :math:`n_{\text{branches}}[j] / (2N_j)`.

   2. **Survival probability**: The probability of *not* coalescing during the time
      step :math:`\Delta t^*_j` is :math:`\exp(-\Delta t^*_j \cdot n_{\text{branches}}[j] / (2N_j))`.

   3. **Coalescence probability**: The probability of coalescing during
      :math:`\Delta t^*_m` is :math:`1 - \exp(-\Delta t^*_m \cdot n_{\text{branches}}[m] / (2N_m))`.

   4. **Branch choice**: Given coalescence at time :math:`m`, the specific branch
      is chosen uniformly among the :math:`n_{\text{coals}}[m]` valid points.

   The :math:`\Delta t^*` values use the coal-time midpoint structure rather than
   the raw time steps. This accounts for the fact that recombination at time :math:`k`
   means the lineage starts partway through interval :math:`k`, and the exposure in
   the first and last intervals may be partial.

The Full Transition Probability
---------------------------------

Putting it together, the transition from state :math:`(a, i)` to state :math:`(b, j)` is:

.. math::

   P\big((a, i) \to (b, j)\big) =
   \underbrace{\delta_{(a,i),(b,j)} \cdot e^{-\rho L}}_{\text{no recombination}}
   + \sum_{k=0}^{k_{\max}} P(\text{recomb at } k) \cdot P(\text{recoal at } (b, j) \mid \text{recomb at } k)

where:

- :math:`\delta_{(a,i),(b,j)}` is 1 if :math:`(a,i) = (b,j)`, 0 otherwise
- :math:`k_{\max}` is the time index of the root (recombination can only happen below the root)
- The sum accounts for all possible recombination times

.. admonition:: Key insight

   The transition probability *does not depend on the source state* :math:`(a, i)`
   for the recombination component! The recombination can happen anywhere on the tree,
   regardless of where the threading lineage currently sits. The source state only
   matters for the no-recombination term (the Kronecker delta).

   This means the transition matrix has a special structure: it is a **rank-1 update**
   of the identity (times the no-recomb probability). This structure can be exploited
   for computational efficiency.

.. admonition:: Calculus Aside --- Rank-1 structure and computational savings

   A rank-1 matrix is one that can be written as :math:`\mathbf{u} \mathbf{v}^\top`
   for vectors :math:`\mathbf{u}` and :math:`\mathbf{v}`. The transition matrix has
   the form:

   .. math::

      T = e^{-\rho L} \cdot I + \mathbf{1} \cdot \mathbf{q}^\top

   where :math:`\mathbf{q}` is the vector of destination-state probabilities (the
   column sums of the recombination component) and :math:`\mathbf{1}` is the all-ones
   vector.

   A matrix--vector product :math:`T \mathbf{x} = e^{-\rho L} \mathbf{x} + \mathbf{1}(\mathbf{q}^\top \mathbf{x})`
   costs only :math:`O(S)` instead of :math:`O(S^2)`, because
   :math:`\mathbf{q}^\top \mathbf{x}` is a single dot product and
   :math:`\mathbf{1} \cdot (\text{scalar})` is a scalar broadcast. This optimization
   reduces the per-site cost of the forward algorithm from :math:`O(S^2)` to
   :math:`O(S)`, which is critical for making ARGweaver practical on real genomes.

Implementation
---------------

.. code-block:: python

   import numpy as np
   from math import exp, log

   def calc_transition_probs(tree, states, times, time_steps,
                             nbranches, nrecombs, ncoals,
                             popsizes, rho, treelen):
       """
       Compute the normal transition probability matrix.

       Parameters
       ----------
       tree : tree object
           The local tree.
       states : list of (str, int)
           Valid HMM states as (node_name, time_index) pairs.
       times : list of float
           Discretized time points.
       time_steps : list of float
           Time step sizes.
       nbranches : list of int
           Branch counts at each time index.
       nrecombs : list of int
           Recombination point counts at each time index.
       ncoals : list of int
           Coalescence point counts at each time index.
       popsizes : list of float
           Population sizes at each time index.
       rho : float
           Per-base recombination rate.
       treelen : float
           Total tree length.

       Returns
       -------
       numpy.ndarray
           Transition probability matrix of shape (nstates, nstates).
       """
       nstates = len(states)
       root_age_index = times.index(tree.root.age)

       # No-recombination probability: the chance that neither site
       # experiences a recombination on any branch of the tree.
       no_recomb = exp(-rho * treelen)

       # Recombination probability at each time index k.
       # This is the fraction of the tree in interval k, times the
       # total recombination probability.
       recomb_probs = []
       for k in range(root_age_index + 1):
           p = (nbranches[k] * time_steps[k] / treelen
                * (1 - no_recomb))
           recomb_probs.append(p)

       # Re-coalescence probabilities: P(recoal at j | recomb at k)
       # Precompute for all (k, j) pairs
       coal_times = get_coal_times(times)
       ntimes = len(times) - 1

       def recoal_prob(k, j):
           """P(recoal at time j | recomb at time k), unnormalized by branch."""
           # Survival from k to j-1: accumulate the product of
           # exp(-exposure / popsize) through each intermediate interval.
           survival = 1.0
           last_nbr = nbranches[max(k - 1, 0)]
           for m in range(k, j):
               nbr = nbranches[m]
               # A is the total "coalescent exposure" in interval m:
               # the upper half-interval (from time point to midpoint above)
               # times the number of branches, plus the lower half-interval
               # times the previous interval's branch count.
               A = (coal_times[2*m + 1] - coal_times[2*m]) * nbr
               if m > k:
                   A += (coal_times[2*m] - coal_times[2*m - 1]) * last_nbr
               survival *= exp(-A / popsizes[m])
               last_nbr = nbr

           # Coalescence in interval j: same exposure calculation,
           # but now we compute 1 - exp(-exposure / popsize).
           nbr = nbranches[j]
           A = (coal_times[2*j + 1] - coal_times[2*j]) * nbr
           if j > k:
               A += (coal_times[2*j] - coal_times[2*j - 1]) * last_nbr
           coal_prob = 1.0 - exp(-A / popsizes[j])

           # Note: survival already accumulated; for j == k, survival = 1
           return survival * coal_prob

       # Build state-to-time lookup
       state_to_idx = {s: idx for idx, s in enumerate(states)}

       # Build transition matrix.
       # Because of the rank-1 structure, each column j gets the same
       # value from all source states (for the recombination component).
       trans = np.zeros((nstates, nstates))

       for j_idx, (node_j, time_j) in enumerate(states):
           # Sum over recombination times: for each possible recomb time k,
           # accumulate the probability of recombining at k and then
           # re-coalescing at (node_j, time_j).
           total_recomb_to_j = 0.0
           for k in range(root_age_index + 1):
               if time_j >= k:
                   rc = recoal_prob(k, time_j)
                   if ncoals[time_j] > 0:
                       total_recomb_to_j += (recomb_probs[k]
                                             * rc / ncoals[time_j])

           # Fill column: all source states can transition to (node_j, time_j)
           # with the same recombination-driven probability (rank-1 structure).
           for i_idx in range(nstates):
               trans[i_idx, j_idx] = total_recomb_to_j

       # Add no-recombination diagonal: if the state does not change,
       # it means no recombination occurred.
       for i_idx in range(nstates):
           trans[i_idx, i_idx] += no_recomb

       return trans

*Recap so far:* Normal transitions combine two scenarios --- no recombination (stay
in the same state, probability :math:`e^{-\rho L}`) and recombination (detach and
re-coalesce, with the destination independent of the source). The resulting matrix
has a rank-1 structure that enables :math:`O(S)` forward passes. Next, we handle
the positions where the tree itself changes.

Switch Transitions
===================

At positions where the partial ARG has a recombination breakpoint, the local tree
changes via an SPR operation. The state space changes too: branches in the old tree
may not exist in the new tree, and vice versa.

ARGweaver handles these positions with **switch transition matrices** that map states
in the old tree to states in the new tree.

The SPR and Its Effect on States
----------------------------------

When the partial ARG has a recombination at position :math:`s`, the local tree
changes from :math:`T_{\text{old}}` to :math:`T_{\text{new}}` via an SPR defined by:

- **Recombination node** :math:`r` at time :math:`t_r`: a branch is cut above this node
- **Coalescence node** :math:`c` at time :math:`t_c`: the subtree re-attaches here

The SPR operation:

1. Detaches the subtree rooted at :math:`r` from its parent
2. Removes the resulting degree-2 node (the "broken" node)
3. Creates a new node at time :math:`t_c` above :math:`c`
4. Attaches the subtree below this new node

Most branches survive this operation unchanged. The only branches affected are:

- The **recombination branch** (the one that was cut)
- The **coalescence branch** (where re-attachment occurs)
- The **broken branch** (the sibling of the recomb branch, which gets a new parent)

If you need a refresher on SPR operations and how they connect adjacent local trees
in an ARG, see :ref:`args`.

Deterministic Mapping
----------------------

For states :math:`(b, i)` in the old tree where branch :math:`b` is not directly
affected by the SPR, the mapping to the new tree is **deterministic**: the same
branch exists in the new tree (possibly with a different name if nodes were
renumbered), and the time index stays the same.

.. math::

   (b_{\text{old}}, i) \to (b_{\text{new}}, i) \quad \text{(deterministic, for unaffected branches)}

Probabilistic Mapping
-----------------------

For the branches directly involved in the SPR (the recombination branch and coal
branch), the mapping is **probabilistic**. The new thread's state at the affected
branches depends on where it was in the old tree and how the SPR rearranges the
topology.

Specifically:

- If the threading lineage was on the **recombination branch** above the recomb point:
  it now needs to be mapped to one of the branches in the new tree at the appropriate
  time, weighted by the re-coalescence probability.

- If the threading lineage was on the **coalescence branch**: it maps to the
  corresponding branch in the new tree, but the branch may have been split by the
  new coalescence node.

.. admonition:: Closing the confusion gap --- Why switch transitions are needed

   At most genomic positions, the local tree does not change between adjacent sites,
   and the normal transition matrix applies. But at positions where the *partial ARG*
   (the ARG for :math:`n-1` haplotypes, before threading the :math:`n`-th) has a
   recombination breakpoint, the tree topology changes. The set of branches is
   different, so the state space is different.

   You cannot use the normal transition matrix here because the source states (in the
   old tree) and destination states (in the new tree) live in different state spaces.
   The switch matrix bridges this gap. Think of it as swapping one gear for a slightly
   different gear mid-rotation --- most teeth mesh perfectly (deterministic mapping),
   but a few teeth need to find new partners (probabilistic mapping).

.. code-block:: python

   def calc_switch_transition_probs(old_tree, new_tree, old_states, new_states,
                                    recomb_node, recomb_time,
                                    coal_node, coal_time,
                                    times, time_steps,
                                    nbranches, nrecombs, ncoals,
                                    popsizes, rho, old_treelen, new_treelen):
       """
       Compute the switch transition matrix at a recombination breakpoint.

       At positions where the partial ARG has a recombination, the local
       tree changes via an SPR. This matrix maps states in the old tree
       to states in the new tree.

       Most mappings are deterministic (1-to-1). Only the states on the
       recombination and coalescence branches have probabilistic mappings.

       Parameters
       ----------
       old_tree, new_tree : tree objects
           Trees before and after the SPR.
       old_states, new_states : list of (str, int)
           States in the old and new trees.
       recomb_node : str
           Name of the node where recombination occurs.
       recomb_time : int
           Time index of the recombination.
       coal_node : str
           Name of the node where re-coalescence occurs.
       coal_time : int
           Time index of the re-coalescence.
       times, time_steps : list of float
           Time grid and step sizes.
       nbranches, nrecombs, ncoals : list of int
           Lineage counts for the new tree.
       popsizes : list of float
           Population sizes.
       rho : float
           Recombination rate.
       old_treelen, new_treelen : float
           Tree lengths before and after.

       Returns
       -------
       numpy.ndarray
           Switch transition matrix of shape (n_old_states, n_new_states).
       """
       n_old = len(old_states)
       n_new = len(new_states)
       trans = np.zeros((n_old, n_new))

       # Build mapping from old branches to new branches
       # For most branches, this is deterministic
       new_state_lookup = {s: idx for idx, s in enumerate(new_states)}

       for i, (old_node, old_time) in enumerate(old_states):
           if old_node == recomb_node and old_time >= recomb_time:
               # This state is on the recombination branch above the cut.
               # It must be re-mapped probabilistically to new tree states.
               # The mapping follows a coalescent process from recomb_time.
               for j, (new_node, new_time) in enumerate(new_states):
                   if new_time >= recomb_time:
                       # Probability of re-coalescing at this new state
                       # (simplified; full implementation uses coal process)
                       trans[i, j] = 1.0 / ncoals[new_time] if ncoals[new_time] > 0 else 0.0
           else:
               # Deterministic mapping: find the same branch in new tree.
               # The branch name and time index carry over unchanged
               # because the SPR did not affect this branch.
               new_state = (old_node, old_time)
               if new_state in new_state_lookup:
                   trans[i, new_state_lookup[new_state]] = 1.0

       # Normalize rows to ensure each is a valid probability distribution.
       for i in range(n_old):
           row_sum = trans[i].sum()
           if row_sum > 0:
               trans[i] /= row_sum

       return trans

.. admonition:: Efficiency of switch transitions

   Switch transitions are sparse: most rows have a single 1.0 entry (deterministic
   mapping). Only :math:`O(n_t)` rows (those on the recomb/coal branches) have
   probabilistic entries. This sparsity can be exploited to avoid full matrix
   multiplications at switch positions.

State Priors
=============

At the first genomic position (or after a long gap), we need a prior distribution
over states. ARGweaver uses the **coalescent prior**: the probability that the new
lineage coalesces at state :math:`(b, j)` is proportional to the probability of
surviving without coalescence to time :math:`j`, then coalescing in interval :math:`j`,
on branch :math:`b`.

.. math::

   P\big((b, j)\big) = \prod_{m=0}^{j-1} \exp\!\left(-\frac{\Delta t^*_m \cdot n_{\text{branches}}[m]}{2 N_m}\right) \cdot \left(1 - \exp\!\left(-\frac{\Delta t^*_j \cdot n_{\text{branches}}[j]}{2 N_j}\right)\right) \cdot \frac{1}{n_{\text{coals}}[j]}

This is the same formula as the re-coalescence probability, but starting from
:math:`k = 0` (the present).

.. admonition:: Probability Aside --- The prior as a special case of re-coalescence

   Notice that the state prior is mathematically identical to the re-coalescence
   distribution with :math:`k = 0`. This makes perfect sense: at the first genomic
   position, there is no "previous state" to transition from. The new lineage starts
   at the present and coalesces with the existing tree exactly as if it had just been
   "born" by a recombination event at :math:`t = 0`. The coalescent prior encodes the
   ancestral belief that recent coalescence is more likely than ancient coalescence
   (because there are more lineages near the present), which is the standard coalescent
   theory from :ref:`coalescent_theory`.

.. code-block:: python

   def calc_state_priors(states, times, nbranches, ncoals,
                         popsizes):
       """
       Compute prior probabilities for each HMM state.

       The prior is the coalescent probability: the chance that a new
       lineage, starting at the present, coalesces at each (branch, time)
       state.

       Parameters
       ----------
       states : list of (str, int)
           Valid HMM states.
       times : list of float
           Discretized time points.
       nbranches : list of int
           Branch counts at each time index.
       ncoals : list of int
           Coalescence point counts at each time index.
       popsizes : list of float
           Population sizes at each time index.

       Returns
       -------
       list of float
           Prior probability for each state (log scale).
       """
       coal_times = get_coal_times(times)
       ntimes = len(times) - 1

       # Precompute survival and coalescence probabilities at each time.
       # These are shared across all states at the same time index.
       survival = [0.0] * ntimes
       coal_prob = [0.0] * ntimes

       cum_survival = 0.0  # cumulative log survival (starts at 0 = log(1))
       for j in range(ntimes):
           nbr = nbranches[j]
           # A is the effective coalescent exposure in interval j,
           # combining the upper and lower half-intervals weighted
           # by their respective lineage counts.
           A = (coal_times[2*j + 1] - coal_times[2*j]) * nbr
           if j > 0:
               A += (coal_times[2*j] - coal_times[2*j - 1]) * nbranches[j-1]

           survival[j] = cum_survival  # log survival to reach j
           coal_prob[j] = log(max(1e-300,
                                  1.0 - exp(-A / popsizes[j])))
           cum_survival += -A / popsizes[j]

       # Compute prior for each state: survival * coalescence * 1/ncoals
       # All in log space for numerical stability.
       priors = []
       for node_name, timei in states:
           if ncoals[timei] > 0:
               p = survival[timei] + coal_prob[timei] - log(ncoals[timei])
           else:
               p = -float('inf')
           priors.append(p)

       return priors


Putting It Together: The Forward Algorithm
============================================

With transitions, switch transitions, and priors defined, the forward algorithm
proceeds along the genome:

.. code-block:: text

   Position:  1       2       3       ...     L
   Tree:      T_1     T_1     T_2     ...     T_m
                |       |       |               |
   Transition: prior   normal  switch  ...     normal
                |       |       |               |
   Forward:   a[1]    a[2]    a[3]    ...     a[L]

At each position:

1. If it's the first position: :math:`\alpha_1(s) = \pi(s) \cdot e(s, d_1)`
2. If the tree hasn't changed: :math:`\alpha_s(j) = \sum_i \alpha_{s-1}(i) \cdot T_{\text{normal}}(i,j) \cdot e(j, d_s)`
3. If the tree changed (switch): :math:`\alpha_s(j) = \sum_i \alpha_{s-1}(i) \cdot T_{\text{switch}}(i,j) \cdot e(j, d_s)`

where :math:`e(s, d)` is the emission probability (see :ref:`argweaver_emissions`).

If this forward recursion looks unfamiliar, see :ref:`hmms` for a detailed derivation
of the forward algorithm and its role in HMM inference.

.. code-block:: python

   def forward_algorithm(states_by_pos, transitions, switch_transitions,
                         emissions, priors):
       """
       Run the forward algorithm along the genome.

       Parameters
       ----------
       states_by_pos : list of list of (str, int)
           States at each genomic position.
       transitions : list of numpy.ndarray or None
           Normal transition matrices (None at switch positions).
       switch_transitions : list of numpy.ndarray or None
           Switch transition matrices (None at normal positions).
       emissions : list of list of float
           Log emission probabilities at each position.
       priors : list of float
           Log prior probabilities.

       Returns
       -------
       list of numpy.ndarray
           Forward probability vectors (log scale) at each position.
       """
       L = len(states_by_pos)
       forward = []

       for s in range(L):
           nstates = len(states_by_pos[s])

           if s == 0:
               # Initialize with prior * emission (log space: addition)
               alpha = np.array([priors[i] + emissions[s][i]
                                 for i in range(nstates)])
           else:
               alpha_prev = forward[-1]

               if switch_transitions[s] is not None:
                   # Switch position: use switch transition matrix
                   trans = switch_transitions[s]
               else:
                   # Normal position: use normal transition matrix
                   trans = transitions[s]

               # Matrix-vector multiply in log space.
               # For each destination state j, compute:
               #   alpha[j] = logsumexp_i(alpha_prev[i] + log(T[i,j])) + emit[j]
               # The logsumexp avoids underflow from very small probabilities.
               alpha = np.full(nstates, -np.inf)
               for j in range(nstates):
                   vals = alpha_prev + np.log(trans[:, j] + 1e-300)
                   alpha[j] = logsumexp(vals) + emissions[s][j]

           forward.append(alpha)

       return forward


   def logsumexp(x):
       """Numerically stable log-sum-exp.

       Computes log(sum(exp(x))) by factoring out the maximum value:
       log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
       This prevents overflow (if max(x) is large) and underflow (if
       all x values are very negative).
       """
       m = np.max(x)
       if m == -np.inf:
           return -np.inf
       return m + np.log(np.sum(np.exp(x - m)))

*Recap:* We have now assembled the complete transition gear. Normal transitions handle
the common case (same tree, recombination or not). Switch transitions handle tree
changes at partial-ARG breakpoints. The state prior initializes the forward algorithm.
Together with the emission probabilities (next chapter, :ref:`argweaver_emissions`),
these form the complete HMM that drives ARGweaver's threading.

Verification
=============

A useful sanity check: the rows of the normal transition matrix should sum to 1
(or very close to 1, modulo floating-point precision).

.. code-block:: python

   def verify_transition_matrix(trans, tol=1e-6):
       """
       Verify that transition matrix rows sum to 1.

       Parameters
       ----------
       trans : numpy.ndarray
           Transition probability matrix.
       tol : float
           Tolerance for row sums.

       Returns
       -------
       bool
           True if all rows sum to 1 within tolerance.
       """
       row_sums = trans.sum(axis=1)
       ok = np.allclose(row_sums, 1.0, atol=tol)
       if not ok:
           print(f"Row sums range: [{row_sums.min():.8f}, {row_sums.max():.8f}]")
       return ok

Exercises
==========

.. admonition:: Exercise 1: Rank-1 structure

   Show algebraically that the normal transition matrix can be written as
   :math:`T = e^{-\rho L} \cdot I + (1 - e^{-\rho L}) \cdot \mathbf{1} \cdot \mathbf{q}^\top`,
   where :math:`\mathbf{q}` is a probability vector over destination states.
   What is :math:`\mathbf{q}`? How does this structure allow :math:`O(S)` instead
   of :math:`O(S^2)` matrix-vector products?

.. admonition:: Exercise 2: Re-coalescence distribution

   For a tree with :math:`k = 10` lineages and constant :math:`N_e = 10{,}000`,
   compute the re-coalescence distribution after a recombination at time :math:`t_0 = 0`.
   Plot the probability mass function across the 20 default time points. Where
   is the mode?

.. admonition:: Exercise 3: Switch matrix sparsity

   For a tree with :math:`k = 8` lineages and :math:`n_t = 20` time points, how many
   states are there? How many rows of the switch transition matrix are deterministic
   (single 1.0 entry)? Express the fraction as a function of :math:`k` and :math:`n_t`.

.. admonition:: Exercise 4: Transition matrix verification

   Implement the full normal transition matrix for a simple 4-leaf tree. Verify that:
   (a) all entries are non-negative, (b) rows sum to 1, (c) the no-recombination
   diagonal dominates when :math:`\rho` is small, and (d) the matrix approaches
   a uniform row-stochastic matrix when :math:`\rho` is large.

Solutions
==========

.. admonition:: Solution 1: Rank-1 structure

   From the full transition formula, the entry :math:`T_{(a,i) \to (b,j)}` is:

   .. math::

      T_{(a,i),(b,j)} = \delta_{(a,i),(b,j)} \, e^{-\rho L}
      + \underbrace{\sum_{k=0}^{k_{\max}} P(\text{recomb at } k) \,
        P(\text{recoal at } (b,j) \mid \text{recomb at } k)}_{q_{(b,j)}}

   The second term depends on the *destination* state :math:`(b,j)` but **not**
   on the source state :math:`(a,i)`. Call this term :math:`q_{(b,j)}`. Then:

   .. math::

      T = e^{-\rho L} \, I + \mathbf{1} \, \mathbf{q}^\top

   where :math:`\mathbf{q} \in \mathbb{R}^S` is the vector with entries
   :math:`q_{(b,j)}` and :math:`\mathbf{1}` is the all-ones vector.

   **What is** :math:`\mathbf{q}`? Each component is the total probability of
   arriving at destination :math:`(b,j)` via recombination anywhere on the tree
   followed by re-coalescence at :math:`(b,j)`:

   .. math::

      q_{(b,j)} = \sum_{k=0}^{k_{\max}}
      \frac{n_{\text{branches}}[k] \, \Delta t_k}{L}
      (1 - e^{-\rho L}) \;
      \frac{P(\text{recoal at time } j \mid \text{recomb at } k)}{n_{\text{coals}}[j]}

   Since the rows of :math:`T` must sum to 1 and the diagonal contribution is
   :math:`e^{-\rho L}`, the vector :math:`\mathbf{q}` sums to
   :math:`1 - e^{-\rho L}`, making :math:`\mathbf{q}/(1 - e^{-\rho L})` a
   proper probability distribution over destination states.

   **The** :math:`O(S)` **trick**: For a matrix-vector product
   :math:`\mathbf{y} = T \mathbf{x}`:

   .. math::

      \mathbf{y} = e^{-\rho L} \, \mathbf{x}
                 + \mathbf{1} \, (\mathbf{q}^\top \mathbf{x})

   Step 1: compute the scalar :math:`c = \mathbf{q}^\top \mathbf{x} = \sum_s q_s x_s`
   in :math:`O(S)`. Step 2: set :math:`y_s = e^{-\rho L} x_s + c` for each :math:`s`
   in :math:`O(S)`. Total: :math:`O(S)` instead of :math:`O(S^2)`.

.. admonition:: Solution 2: Re-coalescence distribution

   With :math:`k = 10`, constant :math:`N_e = 10{,}000`, and recombination at
   :math:`t_0 = 0`, the re-coalescence distribution is the coalescent prior
   (the state prior formula with :math:`k=0`).

   .. code-block:: python

      from math import exp, log
      import numpy as np

      def recoal_distribution(nbranches_const, Ne, times):
          """
          Compute the re-coalescence PMF across time indices,
          starting from time 0, for a constant population size
          and constant lineage count.
          """
          coal_times_list = get_coal_times(times)
          ntimes = len(times) - 1
          pmf = []
          cum_log_surv = 0.0

          for j in range(ntimes):
              nbr = nbranches_const
              A = (coal_times_list[2*j + 1] - coal_times_list[2*j]) * nbr
              if j > 0:
                  A += (coal_times_list[2*j] - coal_times_list[2*j - 1]) * nbr
              coal_prob = 1.0 - exp(-A / Ne)
              pmf.append(exp(cum_log_surv) * coal_prob)
              cum_log_surv += -A / Ne

          return pmf

      times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
      Ne = 10000
      # With k=10 lineages, nbranches = 10 at every time index
      # (simplified: ignoring that lineages decrease after coalescence).
      # For the threading HMM, nbranches is computed from the existing tree.
      # Here we use a constant 10 for illustration.
      pmf = recoal_distribution(10, Ne, times)

      print("Re-coalescence distribution (recomb at t_0 = 0, k=10, Ne=10000):")
      mode_idx = int(np.argmax(pmf))
      for j, p in enumerate(pmf):
          marker = " <-- MODE" if j == mode_idx else ""
          print(f"  j={j:2d}  t={times[j]:10.1f}  P={p:.6f}{marker}")

      # The mode is at j=0 (the first interval), because with 10 lineages
      # the coalescent rate is high: lambda = 10*9/(2*10000) = 0.0045/gen.
      # Most re-coalescence happens near the present.

   The mode is at :math:`j = 0` (the earliest interval). With 10 lineages,
   the coalescent rate :math:`\lambda = \binom{10}{2}/(2 \times 10{,}000) = 0.00225`
   per generation is high enough that re-coalescence overwhelmingly occurs in
   the first few intervals. The PMF decays roughly geometrically, consistent
   with the discrete-geometric interpretation described in the chapter.

.. admonition:: Solution 3: Switch matrix sparsity

   For a tree with :math:`k = 8` leaves: the tree has :math:`2k - 2 = 14`
   branches. With :math:`n_t = 20` time points, the total number of states is
   :math:`O(k \cdot n_t)`. For a balanced tree the exact count depends on
   coalescence times, but a rough upper bound is :math:`14 \times 20 = 280`.

   An SPR affects at most 3 branches (the recomb branch, the coal branch, and
   the broken/sibling branch). Each affected branch spans at most :math:`n_t`
   time intervals. So the number of **probabilistic rows** is at most
   :math:`3 \cdot n_t = 60`.

   The remaining :math:`S - 3n_t` rows (at least :math:`11 \times n_t = 220`
   in the worst case) are **deterministic** (a single 1.0 entry).

   The fraction of deterministic rows is:

   .. math::

      f_{\text{det}} = 1 - \frac{3 n_t}{S}
      \geq 1 - \frac{3 n_t}{(2k-5) n_t}
      = 1 - \frac{3}{2k - 5}

   For :math:`k = 8`: :math:`f_{\text{det}} \geq 1 - 3/11 \approx 73\%`.
   As :math:`k` grows, the fraction approaches 1. This means the switch
   matrix is very sparse, and the matrix-vector product at switch positions
   can be computed in :math:`O(S + 3 n_t^2)` rather than :math:`O(S^2)` by
   handling the deterministic rows as simple copies and only doing full
   computation for the :math:`O(n_t)` probabilistic rows.

.. admonition:: Solution 4: Transition matrix verification

   .. code-block:: python

      import numpy as np
      from math import exp, log

      def build_simple_transition_matrix(ntimes, nbranches, ncoals,
                                         popsizes, rho, treelen, times):
          """
          Build the normal transition matrix for a simplified model
          where all states share the same lineage counts.

          Returns a matrix indexed by time index (ignoring branch identity
          for simplicity, since all branches at the same time contribute
          equally in the rank-1 structure).
          """
          time_steps = get_time_steps(times)
          coal_times_list = get_coal_times(times)
          root_idx = ntimes - 1
          no_recomb = exp(-rho * treelen)

          # Total states = sum of ncoals across time indices
          nstates = sum(ncoals[j] for j in range(ntimes))

          # Build destination probability vector q
          # q[j] = sum_k P(recomb at k) * P(recoal at j | recomb at k) / ncoals[j]
          q = np.zeros(nstates)
          state_map = []  # (time_index, branch_within_time)
          idx = 0
          for j in range(ntimes):
              for b in range(ncoals[j]):
                  state_map.append(j)
                  idx += 1

          for s_idx in range(nstates):
              j = state_map[s_idx]
              total = 0.0
              for k in range(root_idx + 1):
                  if j < k:
                      continue
                  recomb_p = (nbranches[k] * time_steps[k] / treelen
                              * (1 - no_recomb))
                  # Survival from k to j-1
                  surv = 1.0
                  for m in range(k, j):
                      A = (coal_times_list[2*m+1] - coal_times_list[2*m]) * nbranches[m]
                      if m > k:
                          A += (coal_times_list[2*m] - coal_times_list[2*m-1]) * nbranches[max(m-1,0)]
                      surv *= exp(-A / popsizes[m])
                  # Coalescence at j
                  A = (coal_times_list[2*j+1] - coal_times_list[2*j]) * nbranches[j]
                  if j > k:
                      A += (coal_times_list[2*j] - coal_times_list[2*j-1]) * nbranches[max(j-1,0)]
                  cp = 1.0 - exp(-A / popsizes[j])
                  total += recomb_p * surv * cp / max(ncoals[j], 1)
              q[s_idx] = total

          # Build T = no_recomb * I + 1 * q^T
          T = np.outer(np.ones(nstates), q) + no_recomb * np.eye(nstates)
          return T

      # --- Verification ---
      times = get_time_points(ntimes=10, maxtime=100000, delta=0.01)
      ntimes = len(times) - 1  # 9
      Ne = 10000
      popsizes = [Ne] * ntimes
      # Simplified 4-leaf tree: nbranches = [4,4,3,3,2,2,1,1,1]
      nbranches = [4, 4, 3, 3, 2, 2, 1, 1, 1]
      ncoals = [4, 4, 4, 4, 3, 3, 2, 2, 2]
      treelen = sum(nbranches[i] * (times[i+1] - times[i])
                    for i in range(ntimes))
      rho_small = 1e-9
      rho_large = 1e-2

      T_small = build_simple_transition_matrix(
          ntimes, nbranches, ncoals, popsizes, rho_small, treelen, times)
      T_large = build_simple_transition_matrix(
          ntimes, nbranches, ncoals, popsizes, rho_large, treelen, times)

      nstates = T_small.shape[0]

      # (a) Non-negativity
      assert np.all(T_small >= -1e-15), "Negative entries found"
      assert np.all(T_large >= -1e-15), "Negative entries found"
      print("(a) All entries non-negative: PASS")

      # (b) Row sums
      print(f"(b) Row sums (small rho): "
            f"[{T_small.sum(axis=1).min():.8f}, "
            f"{T_small.sum(axis=1).max():.8f}]")

      # (c) Diagonal dominance at small rho
      diag_frac = np.diag(T_small).sum() / T_small.sum()
      print(f"(c) Diagonal fraction (small rho): {diag_frac:.6f}")
      # Should be close to 1 (almost all probability mass on diagonal)

      # (d) Approaches uniform at large rho
      # When rho is large, e^{-rho*L} -> 0, so T -> 1 * q^T.
      # Each row becomes the same vector q (uniform row-stochastic).
      row_std = np.std(T_large, axis=0)  # std across rows for each column
      print(f"(d) Cross-row std of columns (large rho): "
            f"max = {row_std.max():.6e}")
      # Should be near 0: all rows are (nearly) the same.

   **(a)** All entries are non-negative because they are sums/products of
   probabilities and the exponential function. **(b)** Rows sum to 1 because
   the no-recombination probability plus the total recombination-and-recoal
   probability accounts for all events. **(c)** When :math:`\rho` is small,
   :math:`e^{-\rho L} \approx 1`, so the diagonal dominates --- the state
   almost never changes. **(d)** When :math:`\rho` is large,
   :math:`e^{-\rho L} \to 0`, and every row converges to the same vector
   :math:`\mathbf{q}^\top`, making the matrix rank-1 and row-uniform.
