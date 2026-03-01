.. _relate_branch_lengths:

================================================
Gear 3: Branch Length Estimation (MCMC)
================================================

   *The topology tells you who is related to whom. The branch lengths tell you
   when -- and that changes everything.*

With tree topologies in hand from :ref:`Gear 2 <relate_tree_building>`, we now
estimate **when** each coalescence event occurred. This is Phase 2 of Relate:
a Metropolis-Hastings MCMC sampler that explores the posterior distribution
over coalescence times, combining a Poisson mutation likelihood with a
coalescent prior.

This chapter covers three tightly linked steps: mapping mutations to branches,
defining the posterior, and sampling from it.

.. admonition:: Prerequisites

   - :ref:`relate_tree_building` (Gear 2): the local tree topologies
   - :ref:`Coalescent Theory <coalescent_theory>` -- exponential waiting times
     between coalescence events
   - :ref:`Markov Chain Monte Carlo <mcmc>` -- Metropolis-Hastings algorithm


Step 1: Mutation Mapping
=========================

Under the **infinite-sites model**, every derived allele arose by a unique
mutation at a unique genomic position. Each mutation can be placed on exactly
one branch of the local tree: the branch that separates all carriers of the
derived allele from all non-carriers.

Given a tree :math:`\mathcal{T}` and a biallelic site where the derived allele
is carried by a subset :math:`S` of the :math:`N` haplotypes, the mutation maps
to the branch :math:`b` such that the descendants of the child node of
:math:`b` are exactly :math:`S`.

.. math::

   \text{branch}(S) = b \text{ such that } \text{descendants}(b.\text{child}) = S

If no such branch exists (the allele pattern is incompatible with the tree),
the mutation is flagged as **non-mapping** -- a signal that the tree topology
may be incorrect at this site.

.. code-block:: python

   import numpy as np

   def get_descendants(node):
       """Get the set of leaf IDs descended from a node."""
       if node.is_leaf:
           return {node.id}
       return get_descendants(node.left) | get_descendants(node.right)

   def map_mutations(root, haplotypes, site_indices):
       """Map mutations to branches of the tree.

       Parameters
       ----------
       root : TreeNode
           Root of the local tree.
       haplotypes : ndarray of shape (N, L)
           Haplotype matrix.
       site_indices : list of int
           Indices of sites that fall within this tree's genomic interval.

       Returns
       -------
       branch_mutations : dict
           {(parent_id, child_id): count} -- number of mutations on each branch.
       unmapped : int
           Number of mutations that don't map to any branch.
       """
       # Pre-compute descendant sets for each internal node
       def collect_branches(node):
           """Collect all branches as (parent_id, child_id, descendant_set)."""
           branches = []
           if not node.is_leaf:
               left_desc = get_descendants(node.left)
               right_desc = get_descendants(node.right)
               branches.append((node.id, node.left.id, left_desc))
               branches.append((node.id, node.right.id, right_desc))
               branches.extend(collect_branches(node.left))
               branches.extend(collect_branches(node.right))
           return branches

       branches = collect_branches(root)
       N = haplotypes.shape[0]

       branch_mutations = {}
       for parent_id, child_id, _ in branches:
           branch_mutations[(parent_id, child_id)] = 0

       unmapped = 0

       for site in site_indices:
           # Which haplotypes carry the derived allele?
           carriers = {i for i in range(N) if haplotypes[i, site] == 1}

           if len(carriers) == 0 or len(carriers) == N:
               continue  # monomorphic -- skip

           # Find the branch whose descendants exactly match the carriers
           matched = False
           for parent_id, child_id, desc_set in branches:
               if desc_set == carriers:
                   branch_mutations[(parent_id, child_id)] += 1
                   matched = True
                   break

           if not matched:
               unmapped += 1

       return branch_mutations, unmapped

   # Example with a known tree
   from tree_building_module import TreeNode  # (use the TreeNode from Gear 2)

   # Build a simple tree: ((0,1),2),3)
   leaf0 = TreeNode(0)
   leaf1 = TreeNode(1)
   leaf2 = TreeNode(2)
   leaf3 = TreeNode(3)
   node4 = TreeNode(4, left=leaf0, right=leaf1, is_leaf=False)
   node4.leaf_ids = {0, 1}
   node5 = TreeNode(5, left=node4, right=leaf2, is_leaf=False)
   node5.leaf_ids = {0, 1, 2}
   root = TreeNode(6, left=node5, right=leaf3, is_leaf=False)
   root.leaf_ids = {0, 1, 2, 3}

   # Haplotypes with known mutations
   haps = np.array([
       [1, 1, 1, 0],  # 0: carries mutations at sites 0, 1, 2
       [1, 1, 0, 0],  # 1: carries mutations at sites 0, 1
       [0, 1, 0, 0],  # 2: carries mutation at site 1
       [0, 0, 0, 1],  # 3: carries mutation at site 3
   ])

   branch_muts, n_unmapped = map_mutations(root, haps, list(range(4)))
   print("Mutation mapping:")
   for (p, c), count in sorted(branch_muts.items()):
       if count > 0:
           print(f"  Branch ({p} -> {c}): {count} mutation(s)")
   print(f"  Unmapped: {n_unmapped}")

.. admonition:: Confusion Buster -- What About Non-Mapping Mutations?

   In real data, some mutations won't map perfectly to any branch. This
   happens when: (a) the inferred topology is incorrect at that position,
   (b) recurrent mutation has occurred (violating infinite sites), or
   (c) genotyping errors are present. Relate flags these as ``is_not_mapping``
   in its ``.mut`` output file. Non-mapping mutations are excluded from the
   branch length likelihood but retained for downstream analysis. A high
   non-mapping rate at a particular tree may indicate topology error.


Step 2: The Mutation Likelihood
================================

With mutations mapped to branches, we define the likelihood of the data given
the coalescence times. Under the infinite-sites Poisson mutation model:

**The number of mutations on branch** :math:`b` **is Poisson-distributed**:

.. math::

   m_b \sim \text{Poisson}(\mu \cdot \ell_b \cdot \Delta t_b)

where:

- :math:`\mu` is the per-base, per-generation mutation rate
- :math:`\ell_b` is the genomic span (in base pairs) covered by the tree
  containing branch :math:`b`
- :math:`\Delta t_b = t_{\text{parent}} - t_{\text{child}}` is the branch
  length in generations

The **likelihood of all mutations** on a single tree is:

.. math::

   P(\mathbf{m} \mid \mathbf{t}) = \prod_b
   \frac{(\mu \ell_b \Delta t_b)^{m_b}}{m_b!}
   e^{-\mu \ell_b \Delta t_b}

Taking the log:

.. math::

   \log P(\mathbf{m} \mid \mathbf{t}) = \sum_b \left[
   m_b \log(\mu \ell_b \Delta t_b) - \mu \ell_b \Delta t_b - \log(m_b!)
   \right]

.. code-block:: python

   from scipy.special import gammaln

   def log_mutation_likelihood(branch_mutations, node_times, mu, span):
       """Compute the log Poisson mutation likelihood.

       Parameters
       ----------
       branch_mutations : dict
           {(parent_id, child_id): mutation_count}.
       node_times : dict
           {node_id: coalescence_time}.
       mu : float
           Mutation rate per base per generation.
       span : float
           Genomic span of this tree (in base pairs).

       Returns
       -------
       float
           Log likelihood.
       """
       log_lik = 0.0
       for (parent, child), m_b in branch_mutations.items():
           dt = node_times[parent] - node_times[child]
           if dt <= 0:
               return -np.inf  # invalid: parent must be older than child

           rate = mu * span * dt
           # Poisson log-probability: m*log(rate) - rate - log(m!)
           log_lik += m_b * np.log(rate) - rate - gammaln(m_b + 1)

       return log_lik

   # Example
   node_times = {0: 0, 1: 0, 2: 0, 3: 0,
                 4: 100, 5: 300, 6: 500}
   log_lik = log_mutation_likelihood(branch_muts, node_times,
                                      mu=1.25e-8, span=1e4)
   print(f"Log likelihood: {log_lik:.2f}")

.. admonition:: Biology Aside -- The Molecular Clock

   The Poisson model is the mathematical expression of the **molecular
   clock**: mutations accumulate at a roughly constant rate per generation.
   A branch of length :math:`\Delta t` generations spanning :math:`\ell`
   base pairs is expected to carry :math:`\mu \ell \Delta t` mutations.
   The actual count is random (Poisson), but longer branches accumulate
   more mutations on average. This is the signal Relate uses to estimate
   branch lengths: more mutations on a branch = longer branch = greater
   time separation between parent and child.


Step 3: The Coalescent Prior
==============================

The coalescent prior specifies the expected distribution of coalescence times,
given the effective population size :math:`N_e`. For :math:`k` lineages, the
rate of coalescence is:

.. math::

   \lambda_k = \binom{k}{2} \cdot \frac{1}{N_e} = \frac{k(k-1)}{2 N_e}

The time until the next coalescence event (reducing :math:`k` to :math:`k-1`
lineages) is exponentially distributed:

.. math::

   t_k \sim \text{Exponential}(\lambda_k)

For a tree with :math:`N` leaves, coalescence events happen at times
:math:`t_N > t_{N-1} > \cdots > t_2 > 0` (going backward in time). The prior
probability of these times is:

.. math::

   P(\mathbf{t}) = \prod_{k=2}^{N} \lambda_k \cdot
   e^{-\lambda_k \cdot (t_k - t_{k-1})}

where :math:`t_1 = 0` (the present).

For a **piecewise-constant** population size :math:`N_e(t)`, the coalescence
rate changes at epoch boundaries, and the exponential waiting time must be
computed as a piecewise integral (see :ref:`relate_population_size`).

.. code-block:: python

   def log_coalescent_prior(coalescence_times, N_e):
       """Compute the log coalescent prior for a set of coalescence times.

       Parameters
       ----------
       coalescence_times : list of float
           Coalescence times sorted in increasing order (t_N, t_{N-1}, ..., t_2).
           These are the times of the N-1 internal nodes, sorted youngest first.
       N_e : float
           Effective population size (constant).

       Returns
       -------
       float
           Log prior probability.
       """
       n_coal = len(coalescence_times)
       N = n_coal + 1  # number of leaves (lineages start at N)

       log_prior = 0.0
       prev_time = 0.0  # most recent time (present)

       for idx, t in enumerate(coalescence_times):
           # Number of lineages just before this coalescence
           k = N - idx
           if k < 2:
               break

           # Coalescence rate
           rate = k * (k - 1) / (2.0 * N_e)
           # Waiting time
           dt = t - prev_time
           if dt < 0:
               return -np.inf

           # Exponential log-density: log(rate) - rate * dt
           log_prior += np.log(rate) - rate * dt
           prev_time = t

       return log_prior

   # Example: 4 leaves, 3 coalescence events
   coal_times = [100, 300, 500]  # youngest to oldest
   log_prior = log_coalescent_prior(coal_times, N_e=10000)
   print(f"Log coalescent prior: {log_prior:.2f}")

   # Verify: expected time to first coalescence with k=4, N_e=10000
   # rate = 4*3/(2*10000) = 6e-4
   # E[t] = 1/rate = 1666.67 generations
   print(f"Expected first coalescence: {2*10000/(4*3):.0f} generations")


Step 4: The Posterior
======================

The posterior over coalescence times combines the likelihood and the prior:

.. math::

   P(\mathbf{t} \mid \mathbf{m}, \mathcal{T}) \propto
   P(\mathbf{m} \mid \mathbf{t}, \mathcal{T}) \cdot P(\mathbf{t} \mid N_e)

The log-posterior is:

.. math::

   \log P(\mathbf{t} \mid \mathbf{m}, \mathcal{T}) =
   \log P(\mathbf{m} \mid \mathbf{t}) + \log P(\mathbf{t}) + \text{const}

.. code-block:: python

   def log_posterior(node_times, branch_mutations, mu, span, N_e,
                     internal_ids, leaf_ids):
       """Compute the log posterior over coalescence times.

       Parameters
       ----------
       node_times : dict
           {node_id: time} for all nodes.
       branch_mutations : dict
           {(parent, child): count}.
       mu : float
           Mutation rate.
       span : float
           Genomic span.
       N_e : float
           Effective population size.
       internal_ids : list of int
           IDs of internal nodes, sorted by time (youngest first).
       leaf_ids : list of int
           IDs of leaf nodes.

       Returns
       -------
       float
           Log posterior (up to a constant).
       """
       # Likelihood
       ll = log_mutation_likelihood(branch_mutations, node_times, mu, span)
       if ll == -np.inf:
           return -np.inf

       # Prior: extract coalescence times in order
       coal_times = sorted([node_times[n] for n in internal_ids])
       lp = log_coalescent_prior(coal_times, N_e)

       return ll + lp


Step 5: Metropolis-Hastings MCMC
==================================

Relate samples from the posterior using the **Metropolis-Hastings algorithm**.
At each step, it proposes a change to one internal node's time and
accepts or rejects based on the posterior ratio.

The proposal distribution is:

.. math::

   t_k^* = t_k + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)

where :math:`\sigma` is a tuning parameter that controls the step size. The
proposed time must satisfy constraints: the node must be younger than its
parent and older than both its children.

The acceptance probability is:

.. math::

   \alpha = \min\left(1, \;
   \frac{P(\mathbf{t}^* \mid \mathbf{m}, \mathcal{T})}
        {P(\mathbf{t} \mid \mathbf{m}, \mathcal{T})}
   \right)

Since the proposal is symmetric (:math:`q(t^* \mid t) = q(t \mid t^*)`),
the Hastings ratio cancels.

.. admonition:: Probability Aside -- Why Metropolis-Hastings Works

   The Metropolis-Hastings algorithm generates a Markov chain whose
   stationary distribution is the posterior :math:`P(\mathbf{t} \mid \mathbf{m})`.
   The key insight is the **detailed balance** condition: the probability
   of being in state :math:`\mathbf{t}` and transitioning to
   :math:`\mathbf{t}^*` equals the probability of the reverse transition.
   The acceptance ratio :math:`\alpha` is designed to satisfy this condition.
   After a burn-in period, the samples from the chain are (approximately)
   draws from the posterior.

   For a full derivation, see the :ref:`MCMC prerequisite <mcmc>`.

.. code-block:: python

   def mcmc_branch_lengths(root, branch_mutations, mu, span, N_e,
                            n_samples=1000, burn_in=200, sigma=50.0,
                            seed=42):
       """Estimate branch lengths via Metropolis-Hastings MCMC.

       Parameters
       ----------
       root : TreeNode
           Root of the local tree.
       branch_mutations : dict
           {(parent, child): count}.
       mu : float
           Mutation rate.
       span : float
           Genomic span.
       N_e : float
           Effective population size.
       n_samples : int
           Number of MCMC samples (after burn-in).
       burn_in : int
           Number of burn-in steps.
       sigma : float
           Proposal standard deviation.
       seed : int
           Random seed.

       Returns
       -------
       samples : list of dict
           Posterior samples of node times.
       acceptance_rate : float
       """
       rng = np.random.RandomState(seed)

       # Identify leaf and internal nodes
       leaf_ids = []
       internal_ids = []

       def collect_nodes(node):
           if node.is_leaf:
               leaf_ids.append(node.id)
           else:
               internal_ids.append(node.id)
               collect_nodes(node.left)
               collect_nodes(node.right)

       collect_nodes(root)

       # Initialize node times: leaves at 0, internals spaced evenly
       node_times = {}
       for lid in leaf_ids:
           node_times[lid] = 0.0

       # Sort internal nodes by depth (shallowest = youngest first)
       # Use a simple heuristic: assign times based on tree depth
       def assign_initial_times(node, depth=0):
           if node.is_leaf:
               return
           assign_initial_times(node.left, depth + 1)
           assign_initial_times(node.right, depth + 1)
           # Deeper nodes are older
           max_child = max(node_times.get(node.left.id, 0),
                           node_times.get(node.right.id, 0))
           node_times[node.id] = max_child + N_e / 5  # rough spacing

       assign_initial_times(root)

       # Get parent/child relationships for constraint checking
       parent_of = {}
       children_of = {}

       def build_relationships(node):
           children_of[node.id] = []
           if not node.is_leaf:
               children_of[node.id] = [node.left.id, node.right.id]
               parent_of[node.left.id] = node.id
               parent_of[node.right.id] = node.id
               build_relationships(node.left)
               build_relationships(node.right)

       build_relationships(root)

       # Current log posterior
       current_lp = log_posterior(node_times, branch_mutations, mu, span,
                                   N_e, internal_ids, leaf_ids)

       # MCMC loop
       samples = []
       n_accept = 0
       total_steps = burn_in + n_samples

       for step in range(total_steps):
           # Pick a random internal node to update
           target = rng.choice(internal_ids)

           # Propose new time
           old_time = node_times[target]
           new_time = old_time + rng.normal(0, sigma)

           # Check constraints: must be > all children, < parent (if exists)
           min_time = max(node_times[c] for c in children_of[target]) \
                      if children_of[target] else 0.0
           max_time = node_times[parent_of[target]] \
                      if target in parent_of else np.inf

           if new_time <= min_time or new_time >= max_time:
               continue  # reject: violates constraints

           # Compute proposed log posterior
           node_times[target] = new_time
           proposed_lp = log_posterior(node_times, branch_mutations, mu,
                                       span, N_e, internal_ids, leaf_ids)

           # Accept/reject
           log_alpha = proposed_lp - current_lp
           if np.log(rng.uniform()) < log_alpha:
               # Accept
               current_lp = proposed_lp
               n_accept += 1
           else:
               # Reject: revert
               node_times[target] = old_time

           # Collect sample (after burn-in)
           if step >= burn_in:
               samples.append(dict(node_times))

       acceptance_rate = n_accept / total_steps
       return samples, acceptance_rate

   # Example: run MCMC on the example tree
   samples, acc_rate = mcmc_branch_lengths(
       root, branch_muts, mu=1.25e-8, span=1e4, N_e=10000,
       n_samples=500, burn_in=200, sigma=100.0)

   print(f"Acceptance rate: {acc_rate:.1%}")

   # Posterior mean for each internal node
   for nid in [4, 5, 6]:
       times = [s[nid] for s in samples]
       print(f"  Node {nid}: mean={np.mean(times):.0f}, "
             f"std={np.std(times):.0f}, "
             f"95% CI=[{np.percentile(times, 2.5):.0f}, "
             f"{np.percentile(times, 97.5):.0f}]")


Step 6: Posterior Samples for Downstream Analysis
==================================================

A key advantage of Relate's MCMC approach is that it produces **posterior
samples** of branch lengths, not just point estimates. These samples can be
used downstream for:

- **CLUES** (Timepiece XV): estimating allele frequency trajectories and
  selection coefficients via importance sampling over branch length
  uncertainty
- **Population size estimation**: the EM algorithm in Gear 4 uses posterior
  branch length samples as its E-step
- **Uncertainty quantification**: reporting confidence intervals on
  coalescence times, divergence dates, and TMRCA estimates

.. code-block:: python

   def posterior_summary(samples, node_id):
       """Summarize the posterior distribution for a node's time.

       Parameters
       ----------
       samples : list of dict
           MCMC samples.
       node_id : int
           Node to summarize.

       Returns
       -------
       dict
           Mean, median, std, and 95% credible interval.
       """
       times = np.array([s[node_id] for s in samples])
       return {
           'mean': np.mean(times),
           'median': np.median(times),
           'std': np.std(times),
           'ci_lower': np.percentile(times, 2.5),
           'ci_upper': np.percentile(times, 97.5),
       }

   # Example
   for nid in [4, 5, 6]:
       summary = posterior_summary(samples, nid)
       print(f"Node {nid}: mean={summary['mean']:.0f} "
             f"[{summary['ci_lower']:.0f}, {summary['ci_upper']:.0f}]")


Verification
=============

We verify the MCMC on a case where we can compute the posterior analytically:
a single branch with known mutation count.

.. code-block:: python

   def verify_mcmc_single_branch():
       """Verify MCMC on a 2-leaf tree (single branch).

       Tree: root -> leaf (one branch)
       If we observe m mutations on a branch of span L,
       the posterior for the branch length dt is:
         P(dt | m) ~ Poisson(m | mu*L*dt) * Exp(dt | rate=1/N_e)
                    = Gamma(m+1, mu*L + 1/N_e)
       """
       mu = 1.25e-8
       span = 1e6  # 1 Mb
       N_e = 10000
       m = 5  # observed mutations

       # Analytical posterior: Gamma(m+1, mu*L + 1/N_e)
       alpha_post = m + 1
       beta_post = mu * span + 1.0 / N_e
       analytical_mean = alpha_post / beta_post
       analytical_std = np.sqrt(alpha_post) / beta_post

       print("Single-branch verification:")
       print(f"  Observed mutations: {m}")
       print(f"  Analytical posterior: Gamma({alpha_post}, {beta_post:.6f})")
       print(f"  Analytical mean: {analytical_mean:.0f}")
       print(f"  Analytical std:  {analytical_std:.0f}")

       # Build a trivial 2-leaf tree
       leaf = TreeNode(0)
       root_node = TreeNode(1, left=leaf, right=None, is_leaf=False)
       # Hack: make it a single branch by using just one branch
       branch_muts_simple = {(1, 0): m}

       # Run MCMC
       # (In practice you'd use the full MCMC; here we demonstrate
       #  with a simple 1D sampler for clarity)
       rng = np.random.RandomState(42)
       dt_current = analytical_mean  # start at the mean
       sigma = 500
       mcmc_samples = []
       coal_rate = 1.0 / N_e  # 2 lineages, so rate = 1/N_e

       for step in range(5000):
           # Propose
           dt_new = dt_current + rng.normal(0, sigma)
           if dt_new <= 0:
               continue

           # Log posterior ratio
           rate = mu * span
           lp_new = m * np.log(rate * dt_new) - rate * dt_new \
                    - coal_rate * dt_new
           lp_old = m * np.log(rate * dt_current) - rate * dt_current \
                    - coal_rate * dt_current

           if np.log(rng.uniform()) < lp_new - lp_old:
               dt_current = dt_new

           if step >= 1000:
               mcmc_samples.append(dt_current)

       mcmc_mean = np.mean(mcmc_samples)
       mcmc_std = np.std(mcmc_samples)

       print(f"\n  MCMC mean: {mcmc_mean:.0f}")
       print(f"  MCMC std:  {mcmc_std:.0f}")
       print(f"  Relative error (mean): "
             f"{abs(mcmc_mean - analytical_mean) / analytical_mean:.1%}")
       print(f"  [{'ok' if abs(mcmc_mean - analytical_mean) / analytical_mean < 0.1 else 'FAIL'}] "
             f"MCMC mean within 10% of analytical mean")

   verify_mcmc_single_branch()


Exercises
==========

.. admonition:: Exercise 1: Acceptance rate tuning

   Run the MCMC with proposal standard deviations :math:`\sigma = 10, 50, 200,
   1000`. Plot the acceptance rate and the effective sample size (ESS) as a
   function of :math:`\sigma`. What value gives the best mixing? (Target:
   20-40% acceptance rate.)

.. admonition:: Exercise 2: Gibbs vs. Metropolis

   The current implementation updates one node at a time (component-wise
   Metropolis). Implement a Gibbs sampler for the case where the prior is
   conjugate to the Poisson likelihood (it is -- the posterior for each
   node time, conditional on its neighbors, is a truncated gamma). Compare
   the mixing of Gibbs and Metropolis.

.. admonition:: Exercise 3: Multiple trees

   Extend the MCMC to handle a sequence of local trees along the genome.
   Relate assumes trees are independent (no explicit SMC coupling between
   adjacent trees). Run the MCMC independently for each tree and verify
   that adjacent trees produce consistent node times for shared lineages.

Next: :ref:`relate_population_size` -- using the EM algorithm to jointly
estimate population size and branch lengths.
