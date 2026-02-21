.. _msprime_mutations:

===========
Mutations
===========

   *The final touch: painting variation onto the genealogy.*

The genealogy (tree sequence) from Phase 1 tells us *who is related to whom*
and *when* they shared ancestors. But it says nothing about *what their DNA
looks like*. Mutations -- heritable changes in the DNA sequence -- are what
create the observable genetic variation.

In msprime, mutations are simulated as a **separate post-processing step** on
the tree sequence. This is both conceptually clean (the genealogy doesn't
depend on mutations) and computationally efficient (we can reuse the same
genealogy with different mutation models).

In the watch metaphor, if the movement (the coalescent, segments, and Hudson's
algorithm) is the hidden mechanism, and the case and dial (demographics) shape
its form, then mutations are the paint on the dial face -- the markings that
make the watch *readable*. Without mutations, the genealogy is invisible:
a perfect mechanism that tells time but shows no numbers.

.. note::

   **Prerequisites.** This chapter builds on all previous chapters in this
   Timepiece. Specifically:

   - The **tree sequence** output from :ref:`hudson_algorithm` -- the edges
     and nodes that define the genealogy.
   - **Branch lengths** -- determined by coalescence times from
     :ref:`coalescent_process`.
   - **Marginal trees** -- the concept from :ref:`msprime_overview` that
     different genomic positions can have different genealogies.

   You should also be familiar with the :ref:`coalescent_theory` chapter's
   treatment of the expected number of segregating sites and the site
   frequency spectrum, which we will rederive here from the simulation
   perspective.


Step 1: The Infinite-Sites Poisson Model
==========================================

The simplest mutation model: mutations arise as a Poisson process along each
branch of the genealogy.

**The setup.** Given a tree sequence with branches of known length (in
generations), and a per-base-pair, per-generation mutation rate :math:`\mu`:

- Each branch of length :math:`t` generations covering :math:`\ell` base pairs
  accumulates mutations at rate :math:`\mu \cdot \ell \cdot t`.
- The number of mutations on a branch is :math:`\text{Poisson}(\mu \ell t)`.
- Each mutation is placed at a uniformly random position along the branch
  (both in time and in genomic position).

Under the **infinite-sites** assumption, every mutation creates a new variant
at a previously-unmutated position. This means no two mutations can hit the
same site. (For short branches and realistic :math:`\mu`, this is an excellent
approximation.)

.. admonition:: Probability Aside -- The Poisson process on branches

   The Poisson process is a natural model for rare, independent events
   occurring in continuous time. If mutations arise independently at rate
   :math:`\mu` per base pair per generation, then over :math:`t` generations
   and :math:`\ell` base pairs, the expected count is :math:`\mu \ell t`.
   The probability of exactly :math:`k` mutations is:

   .. math::

      P(N = k) = \frac{(\mu \ell t)^k e^{-\mu \ell t}}{k!}

   For human parameters (:math:`\mu \approx 1.5 \times 10^{-8}`,
   :math:`\ell = 1000` bp, :math:`t = 10{,}000` generations), the expected
   count is :math:`\approx 0.15`, so most branches get 0 or 1 mutation.
   This makes the infinite-sites assumption excellent in practice.

.. code-block:: python

   import numpy as np

   def simulate_mutations_infinite_sites(edges, nodes, sequence_length, mu):
       """Add mutations to a tree sequence under the infinite-sites model.

       This is the core of Phase 2: walk through every edge (branch) of
       the tree sequence, draw a Poisson number of mutations, and place
       each one at a random position and time.

       Parameters
       ----------
       edges : list of (left, right, parent, child)
           Edges defining the tree sequence.
       nodes : list of (time, population)
           Node times (samples at time 0, ancestors at time > 0).
       sequence_length : float
           Total genome length in base pairs.
       mu : float
           Per-bp, per-generation mutation rate.

       Returns
       -------
       mutations : list of (position, node, time, ancestral, derived)
           Each mutation has a genomic position, the node it sits above,
           the time it occurred, and the allelic states.
       """
       mutations = []

       for left, right, parent, child in edges:
           # Branch length in generations (time difference parent - child)
           branch_length = nodes[parent][0] - nodes[child][0]
           # Genomic span of this edge
           span = right - left
           # Expected number of mutations: mu * span * branch_length
           expected = mu * span * branch_length

           # Draw number of mutations from Poisson distribution
           n_muts = np.random.poisson(expected)

           for _ in range(n_muts):
               # Random position within [left, right)
               position = np.random.uniform(left, right)
               # Random time on the branch (between child and parent times)
               time = np.random.uniform(nodes[child][0], nodes[parent][0])
               # Under infinite sites: ancestral=0, derived=1
               mutations.append((position, child, time, '0', '1'))

       # Sort by position for output
       mutations.sort(key=lambda m: m[0])
       return mutations

   # Example: simple tree with 3 samples
   #       4 (t=1.5)
   #      / \
   #     3   \  (t=0.8)
   #    / \   \
   #   0   1   2
   #  (t=0)
   nodes = [(0, 0), (0, 0), (0, 0), (0.8, 0), (1.5, 0)]
   edges = [
       (0, 1000, 3, 0),  # edge: node 0 is child of node 3
       (0, 1000, 3, 1),  # edge: node 1 is child of node 3
       (0, 1000, 4, 3),  # edge: node 3 is child of node 4
       (0, 1000, 4, 2),  # edge: node 2 is child of node 4
   ]

   np.random.seed(42)
   muts = simulate_mutations_infinite_sites(edges, nodes, 1000, mu=1e-3)
   print(f"Number of mutations: {len(muts)}")
   for pos, node, time, anc, der in muts[:10]:
       print(f"  pos={pos:.1f}, above node {node}, time={time:.4f}")

With the mechanics of mutation placement established, let us derive the
classical results that connect mutations to the coalescent.


Step 2: The Expected Number of Segregating Sites
===================================================

A fundamental result in population genetics: the expected number of
segregating sites (positions with a mutation) in a sample of :math:`n` from
a population of size :math:`N` is:

.. math::

   E[S] = \theta \cdot \sum_{k=1}^{n-1} \frac{1}{k}

where :math:`\theta = 4N_e \mu L` is the population-scaled mutation rate for
the whole genome, and the sum :math:`\sum_{k=1}^{n-1} 1/k` is the
:math:`(n-1)`-th harmonic number.

**Derivation.** The total branch length of the coalescent tree is:

.. math::

   E[L_{\text{total}}] = \sum_{k=2}^{n} k \cdot E[T_k]
   = \sum_{k=2}^{n} k \cdot \frac{2}{k(k-1)}
   = 2 \sum_{k=2}^{n} \frac{1}{k-1}
   = 2 \sum_{j=1}^{n-1} \frac{1}{j}

(In coalescent units where :math:`N_e = 1`.)

.. admonition:: Calculus Aside -- The harmonic number

   The sum :math:`H_n = \sum_{k=1}^{n} 1/k` is the :math:`n`-th harmonic
   number. It grows logarithmically: :math:`H_n \approx \ln(n) + \gamma`
   where :math:`\gamma \approx 0.5772` is the Euler-Mascheroni constant.
   This means the total branch length grows as :math:`\sim 2\ln(n)`, so
   the expected number of segregating sites grows as
   :math:`\sim \theta \ln(n)`. Doubling the sample size adds only
   :math:`\theta \ln(2) \approx 0.69\theta` additional segregating sites --
   a consequence of the "diminishing returns" of adding more samples to the
   coalescent tree (recall from :ref:`coalescent_process` that most of the
   tree height comes from the last few lineages).

Each unit of branch length produces mutations at rate :math:`\theta/2` per
unit length (in coalescent units, the mutation rate is :math:`\theta/2`
because :math:`\theta = 4N_e\mu` and time is in units of :math:`N_e`
generations for haploids). So:

.. math::

   E[S] = \frac{\theta}{2} \cdot E[L_{\text{total}}]
   = \frac{\theta}{2} \cdot 2\sum_{j=1}^{n-1} \frac{1}{j}
   = \theta \sum_{j=1}^{n-1} \frac{1}{j}

This is **Watterson's estimator** in reverse: given observed :math:`S`, we
can estimate :math:`\hat{\theta}_W = S / \sum_{j=1}^{n-1} 1/j`.

.. code-block:: python

   def expected_segregating_sites(n, theta):
       """Expected number of segregating sites.

       Uses the formula E[S] = theta * H_{n-1}, where H_k is the
       k-th harmonic number.
       """
       harmonic = sum(1.0 / k for k in range(1, n))  # H_{n-1}
       return theta * harmonic

   def watterson_estimator(S, n):
       """Estimate theta from the number of segregating sites.

       This inverts E[S] = theta * H_{n-1} to get theta_hat = S / H_{n-1}.
       """
       harmonic = sum(1.0 / k for k in range(1, n))
       return S / harmonic

   # Example
   n, theta = 50, 100
   E_S = expected_segregating_sites(n, theta)
   print(f"n={n}, theta={theta}")
   print(f"Expected segregating sites: {E_S:.1f}")
   print(f"Watterson's estimate from E[S]: {watterson_estimator(E_S, n):.1f}")

The number of segregating sites is a single summary statistic. A much richer
summary is the site frequency spectrum.


Step 3: The Site Frequency Spectrum
=====================================

The **Site Frequency Spectrum (SFS)** counts how many sites have a derived
allele at each possible frequency. If :math:`\xi_i` is the number of sites
where exactly :math:`i` out of :math:`n` samples carry the derived allele:

.. math::

   E[\xi_i] = \frac{\theta}{i}, \quad i = 1, 2, \ldots, n-1

This beautiful result says that singletons (:math:`i = 1`) are the most
common class of variants, and the frequency spectrum falls off as :math:`1/i`.

**Derivation.** A mutation creates a variant at frequency :math:`i/n` if and
only if it falls on a branch subtending exactly :math:`i` leaves. The expected
length of branches subtending :math:`i` leaves is :math:`2/i` (in coalescent
units). The mutation rate is :math:`\theta/2` per unit length. So:

.. math::

   E[\xi_i] = \frac{\theta}{2} \cdot \frac{2}{i} = \frac{\theta}{i}

.. admonition:: Probability Aside -- Why :math:`2/i` for the branch length subtending :math:`i` leaves?

   This result comes from the exchangeability of the coalescent. In a
   coalescent tree with :math:`n` leaves, the total branch length at level
   :math:`k` (when there are :math:`k` lineages) is :math:`k \cdot T_k`,
   with :math:`E[T_k] = 2/[k(k-1)]`. A branch at level :math:`k` subtends
   some number of leaves. Summing over all levels and using linearity of
   expectation, one can show that the expected total length of branches
   subtending exactly :math:`i` leaves is :math:`2/i`, independent of
   :math:`n` (for :math:`i < n`). This is a remarkable symmetry of
   Kingman's coalescent. For a full proof, see the derivation in
   :ref:`coalescent_theory`.

.. code-block:: python

   def compute_sfs(mutations, genotype_matrix, n):
       """Compute the site frequency spectrum from genotype data.

       Parameters
       ----------
       genotype_matrix : ndarray of shape (n_sites, n_samples)
           0 = ancestral, 1 = derived.

       Returns
       -------
       sfs : ndarray of shape (n - 1,)
           sfs[i-1] = number of sites with derived allele count i.
       """
       sfs = np.zeros(n - 1, dtype=int)
       for site in genotype_matrix:
           count = int(site.sum())  # number of derived alleles at this site
           if 1 <= count <= n - 1:
               sfs[count - 1] += 1  # sfs is 0-indexed: sfs[0] = singletons
       return sfs

   def expected_sfs(n, theta):
       """Expected SFS under the standard neutral model.

       E[xi_i] = theta / i for i = 1, ..., n-1.
       """
       return np.array([theta / i for i in range(1, n)])

   # Example
   n, theta = 20, 50
   exp_sfs = expected_sfs(n, theta)
   print("Expected SFS:")
   for i, e in enumerate(exp_sfs):
       bar = '#' * int(e)
       print(f"  freq {i+1:>2d}/{n}: E[xi] = {e:.2f}  {bar}")

The infinite-sites model is elegant, but it breaks down when mutation rates
are high or branches are long. For those cases, we need finite-sites models.


Step 4: Finite-Sites Mutation Models
=======================================

The infinite-sites model breaks down when the mutation rate is high or branches
are long: the same site can be hit by multiple mutations. msprime supports
several **finite-sites** models.

Matrix Mutation Model
-----------------------

Mutations follow a Markov chain on allelic states (e.g., A, C, G, T). The
transition matrix :math:`P` gives the probability of each state change:

.. math::

   P = \begin{pmatrix}
   0 & p_{AC} & p_{AG} & p_{AT} \\
   p_{CA} & 0 & p_{CG} & p_{CT} \\
   p_{GA} & p_{GC} & 0 & p_{GT} \\
   p_{TA} & p_{TC} & p_{TG} & 0
   \end{pmatrix}

where each row sums to 1 (the diagonal is 0 because a "mutation" must change
the state).

.. admonition:: Probability Aside -- Mutation as a Markov chain

   Each site's allelic state follows a continuous-time Markov chain (CTMC).
   When a mutation event occurs (Poisson process), the state transitions
   according to the matrix :math:`P`. The diagonal is zero because a
   "mutation" that does not change the state is unobservable (and by
   convention, msprime only records observable changes). The stationary
   distribution of this chain determines the long-run base composition.
   For the Jukes-Cantor model (all transitions equally likely), the
   stationary distribution is uniform: :math:`(1/4, 1/4, 1/4, 1/4)`.

.. code-block:: python

   class MatrixMutationModel:
       """Finite-sites mutation model with transition matrix.

       Each mutation event changes the allelic state according to the
       transition matrix. The root state is drawn from root_distribution.
       """

       def __init__(self, alleles, root_distribution, transition_matrix):
           """
           Parameters
           ----------
           alleles : list of str
               Allelic states (e.g., ['A', 'C', 'G', 'T']).
           root_distribution : ndarray
               Probability of each allele at the root.
           transition_matrix : ndarray of shape (k, k)
               P[i, j] = probability of mutating from allele i to allele j.
               Diagonal must be 0 (mutations must change the state).
           """
           self.alleles = alleles
           self.root_distribution = np.array(root_distribution)
           self.transition_matrix = np.array(transition_matrix)
           assert np.allclose(self.transition_matrix.diagonal(), 0)
           assert np.allclose(self.transition_matrix.sum(axis=1), 1)

       def draw_root_state(self):
           """Sample the ancestral allele at the root."""
           return np.random.choice(len(self.alleles), p=self.root_distribution)

       def mutate(self, current_state):
           """Apply one mutation: change the state according to the matrix."""
           return np.random.choice(len(self.alleles),
                                    p=self.transition_matrix[current_state])

   # Jukes-Cantor model: all transitions equally likely
   jc_model = MatrixMutationModel(
       alleles=['A', 'C', 'G', 'T'],
       root_distribution=[0.25, 0.25, 0.25, 0.25],
       transition_matrix=[
           [0, 1/3, 1/3, 1/3],   # from A: equal prob of C, G, or T
           [1/3, 0, 1/3, 1/3],   # from C: equal prob of A, G, or T
           [1/3, 1/3, 0, 1/3],   # from G: equal prob of A, C, or T
           [1/3, 1/3, 1/3, 0],   # from T: equal prob of A, C, or G
       ]
   )

   # Simulate 10 mutations starting from 'A'
   state = 0  # 'A'
   print("Mutation chain starting from A:")
   chain = ['A']
   for _ in range(10):
       state = jc_model.mutate(state)
       chain.append(jc_model.alleles[state])
   print(" -> ".join(chain))


The Binary Mutation Model
---------------------------

The simplest finite-sites model: two alleles (0 and 1), with equal transition
probabilities:

.. math::

   P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

Every mutation flips the allele. This is equivalent to the infinite-sites model
when the mutation rate is low enough that each site is hit at most once, but
allows back-mutations when rates are higher.

.. admonition:: Closing a confusion gap -- Infinite sites vs. finite sites

   Under the infinite-sites model, each mutation creates a *new* variant at
   an unused position. No site is ever hit twice. Under finite sites, a site
   can be hit multiple times, potentially reverting to the ancestral state
   ("back-mutation"). For typical human parameters (:math:`\mu \sim 10^{-8}`
   per bp per generation, branch lengths of :math:`\sim 10^4` generations),
   the probability of two mutations at the same site is about
   :math:`(\mu \cdot t)^2 / 2 \approx 5 \times 10^{-9}` -- negligible.
   But for high-mutation-rate organisms (viruses, microsatellites) or very
   deep genealogies, finite-sites models are essential.

Now let us see how mutations are placed on a tree sequence, integrating over
all marginal trees.


Step 5: Placing Mutations on the Tree Sequence
=================================================

The mutation placement algorithm for a tree sequence:

1. For each site position :math:`x` along the genome:

   a. Find the marginal tree at position :math:`x`
   b. Draw the root allele from the root distribution
   c. Walk down the tree from root to leaves
   d. On each branch, the number of mutations is Poisson with rate
      :math:`\mu \cdot t_{\text{branch}}` (branch length in generations)
   e. Each mutation changes the allelic state according to the model

2. The allele at each leaf is the final state after all mutations along
   the path from root to leaf.

.. code-block:: python

   def place_mutations_on_tree(tree, mu, model, sequence_length):
       """Place mutations on a single marginal tree.

       This walks from root to leaves, drawing Poisson-distributed
       mutations on each branch and tracking allelic state changes.

       Parameters
       ----------
       tree : dict
           Tree as {node: (parent, time, children)}.
       mu : float
           Per-site, per-generation mutation rate.
       model : MatrixMutationModel
       sequence_length : float
           Genomic span of this tree.

       Returns
       -------
       mutations : list of (position, node, parent_node, derived_state, time)
       leaf_states : dict of {leaf: allele_index}
       """
       mutations = []
       root = find_root(tree)

       # Draw root state from the model's stationary distribution
       root_state = model.draw_root_state()
       node_states = {root: root_state}

       # DFS traversal: root to leaves, propagating allelic state
       stack = [root]
       while stack:
           node = stack.pop()
           current_state = node_states[node]
           parent, time, children = tree[node]

           for child in children:
               _, child_time, _ = tree[child]
               branch_length = time - child_time  # time in generations

               # Number of mutations on this branch: Poisson(mu * t)
               n_muts = np.random.poisson(mu * branch_length)

               state = current_state
               for _ in range(n_muts):
                   new_state = model.mutate(state)  # apply transition matrix
                   # Random time on the branch
                   mut_time = np.random.uniform(child_time, time)
                   # Random position within the tree's span
                   position = np.random.uniform(0, sequence_length)
                   mutations.append((position, child, node,
                                      model.alleles[new_state], mut_time))
                   state = new_state

               # Child inherits the final state after all mutations
               node_states[child] = state
               stack.append(child)

       # Collect leaf states (leaves have no children)
       leaf_states = {node: node_states[node]
                      for node in tree
                      if not tree[node][2]}  # no children = leaf

       return mutations, leaf_states

   def find_root(tree):
       """Find the root of a tree (node with no parent)."""
       for node, (parent, time, children) in tree.items():
           if parent is None:
               return node
       raise ValueError("No root found")


Step 6: The Mutation Rate Map
================================

Like recombination, mutation rates can vary along the genome. A **mutation
rate map** specifies the local rate :math:`\mu(x)` at each position:

.. admonition:: Calculus Aside -- Mutation mass and expected counts

   Just as with recombination, the "mass" of an interval :math:`[a, b)` is
   :math:`\int_a^b \mu(x)\,dx`. The expected number of mutations on a branch
   of length :math:`t` spanning :math:`[a, b)` is
   :math:`t \cdot \int_a^b \mu(x)\,dx`. For a piecewise-constant rate, this
   integral reduces to a sum of rate-times-length terms, exactly as in the
   recombination rate map from :ref:`segments_fenwick`.

.. code-block:: python

   class MutationRateMap:
       """Piecewise-constant mutation rate along the genome.

       Analogous to the recombination RateMap, but for mutations.
       """

       def __init__(self, positions, rates):
           self.positions = np.array(positions)
           self.rates = np.array(rates)

       def rate_at(self, position):
           """Get mutation rate at a specific position."""
           idx = np.searchsorted(self.positions, position, side='right') - 1
           idx = max(0, min(idx, len(self.rates) - 1))
           return self.rates[idx]

       def total_mass(self, left, right):
           """Total mutation mass over [left, right).

           This is the integral of mu(x) from left to right.
           """
           total = 0
           for i in range(len(self.rates)):
               seg_left = self.positions[i]
               seg_right = self.positions[i + 1]
               # Compute overlap with [left, right)
               ol = max(seg_left, left)
               or_ = min(seg_right, right)
               if or_ > ol:
                   total += self.rates[i] * (or_ - ol)
           return total

   # Example: mutation rate with a cold spot (centromere)
   mut_map = MutationRateMap(
       positions=[0, 4000, 6000, 10000],
       rates=[1.5e-8, 1e-9, 1.5e-8]  # low rate in [4000, 6000)
   )

   print("Mutation rates:")
   for x in [1000, 5000, 8000]:
       print(f"  position {x}: mu = {mut_map.rate_at(x):.2e}")
   print(f"Total mass [0, 10000): {mut_map.total_mass(0, 10000):.2e}")

With the rate map handling spatial variation, let us see the final step:
converting mutations into observable genotype data.


Step 7: From Mutations to Genotype Matrix
============================================

The final output is a **genotype matrix**: for each variant site, the allele
carried by each sample.

.. code-block:: python

   def build_genotype_matrix(mutations, tree_sequence, n_samples):
       """Convert mutations to a genotype matrix.

       For each mutation, determine which samples carry the derived allele
       by checking if they descend from the mutated node in the marginal
       tree at the mutation's position.

       Parameters
       ----------
       mutations : list of (position, node, ...)
       tree_sequence : object
           The tree sequence (for determining descendant sets).
       n_samples : int

       Returns
       -------
       genotypes : ndarray of shape (n_sites, n_samples)
           0 = ancestral, 1 = derived (for biallelic sites).
       positions : ndarray of shape (n_sites,)
       """
       n_sites = len(mutations)
       genotypes = np.zeros((n_sites, n_samples), dtype=int)
       positions = np.zeros(n_sites)

       for i, (pos, node, *_) in enumerate(mutations):
           positions[i] = pos
           # Find all samples descending from 'node' at position 'pos'
           # This requires looking up the marginal tree at that position
           descendants = get_descendants(tree_sequence, node, pos)
           for sample in descendants:
               if sample < n_samples:
                   genotypes[i, sample] = 1  # mark as carrying derived allele

       return genotypes, positions

   def get_descendants(tree_sequence, node, position):
       """Get all leaf descendants of a node at a genomic position.

       This requires finding the marginal tree at 'position' and
       traversing below 'node'.
       """
       # (Simplified placeholder -- in tskit, this is tree.samples(node))
       return []


Putting It All Together
========================

The complete mutation simulation pipeline:

.. code-block:: python

   def sim_mutations(tree_sequence, rate, model=None):
       """Simulate mutations on a tree sequence.

       This is Phase 2 of msprime's simulation pipeline.
       Phase 1 (ancestry) built the movement; Phase 2 paints the dial.

       Parameters
       ----------
       tree_sequence : object
           Output of ancestry simulation (Phase 1).
       rate : float or MutationRateMap
           Per-bp, per-generation mutation rate.
       model : MutationModel, optional
           Defaults to infinite-sites binary model.

       Returns
       -------
       mutated_ts : object
           Tree sequence with mutations added.
       """
       if model is None:
           # Default: binary (0/1) model -- equivalent to infinite-sites
           # when mutation rate is low
           model = MatrixMutationModel(
               alleles=['0', '1'],
               root_distribution=[1, 0],  # root is always '0' (ancestral)
               transition_matrix=[[0, 1], [1, 0]]  # every mutation flips
           )

       mutations = []
       for tree in tree_sequence.trees():
           span = tree.interval.right - tree.interval.left

           for node in tree.nodes():
               if tree.parent(node) is not None:
                   # Compute branch length (parent time - child time)
                   branch_length = (tree.time(tree.parent(node)) -
                                     tree.time(node))

                   # Get the effective mutation rate for this tree's span
                   if isinstance(rate, MutationRateMap):
                       mu = rate.total_mass(tree.interval.left,
                                             tree.interval.right) / span
                   else:
                       mu = rate

                   # Expected mutations on this branch = mu * span * t
                   expected = mu * span * branch_length
                   n_muts = np.random.poisson(expected)

                   for _ in range(n_muts):
                       # Random position within the tree's genomic interval
                       pos = np.random.uniform(tree.interval.left,
                                                tree.interval.right)
                       # Random time on the branch
                       time = np.random.uniform(tree.time(node),
                                                 tree.time(tree.parent(node)))
                       mutations.append({
                           'position': pos,
                           'node': node,
                           'time': time,
                           'derived_state': '1',
                       })

       return mutations

.. admonition:: Why separate ancestry and mutations?

   1. **Efficiency**: The same genealogy can be used with different mutation
      rates or models without re-simulating ancestry.

   2. **Modularity**: Ancestry simulation is complex (recombination, demographics,
      migration). Mutation simulation is simple (Poisson process on branches).
      Separating them keeps both implementations clean.

   3. **Flexibility**: You can use nucleotide models (A/C/G/T), infinite
      alleles models, or even custom models. The genealogy doesn't change.

   4. **Analysis**: Many population genetic statistics (e.g., :math:`F_{ST}`,
      :math:`\pi`) can be computed directly from the tree sequence without
      mutations, using branch length statistics. Mutations are only needed
      for statistics that depend on allele frequencies.


Exercises
=========

.. admonition:: Exercise 1: Watterson's estimator

   Simulate 1000 genealogies with ``msprime.sim_ancestry(n=50, sequence_length=1e6)``.
   Add mutations with :math:`\mu = 1.5 \times 10^{-8}`. For each, compute
   :math:`\hat{\theta}_W = S / \sum_{j=1}^{n-1} 1/j`. Verify that the mean
   of :math:`\hat{\theta}_W` matches the true :math:`\theta`.

.. admonition:: Exercise 2: Site frequency spectrum

   Using the same simulations, compute the empirical SFS and compare to
   :math:`E[\xi_i] = \theta / i`. Plot both on the same axes.

.. admonition:: Exercise 3: Nucleotide model

   Implement a Hasegawa-Kishino-Yano (HKY) mutation model with transition/
   transversion ratio :math:`\kappa = 2` and base frequencies
   :math:`\pi_A = 0.3, \pi_C = 0.2, \pi_G = 0.2, \pi_T = 0.3`. Simulate
   mutations on a simple 4-tip tree and verify the base composition at the
   tips.

.. admonition:: Exercise 4: Multiple hits

   Compare the infinite-sites model to a finite-sites binary model for
   :math:`\theta = 0.001, 0.01, 0.1, 1.0`. At what :math:`\theta` does the
   infinite-sites assumption break down (measured by the fraction of sites
   with >1 mutation)?

----

Congratulations. You've now disassembled and rebuilt every gear on the master
clockmaker's bench:

- **The Coalescent** -- How lineages find common ancestors (the escapement)
- **Segments & the Fenwick Tree** -- The linked-list track that follows each
  lineage's ancestral material, and the clever indexing mechanism for fast
  event scheduling (the gear train)
- **Hudson's Algorithm** -- The main simulation loop -- the ticking of the
  clock (the mainspring)
- **Demographics** -- Population structure, growth, and migration (the case
  and dial)
- **Mutations** -- Painting variation onto the genealogy (the dial markings)

You built it yourself. No black boxes remain.

*The watch ticks. And you know exactly why.*
