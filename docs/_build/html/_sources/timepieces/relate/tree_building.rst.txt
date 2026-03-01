.. _relate_tree_building:

====================================
Gear 2: Tree Building
====================================

   *Standard clustering sees only distance. Relate's algorithm sees direction --
   and that makes all the difference.*

With the asymmetric distance matrix :math:`d(i,j)` from
:ref:`Gear 1 <relate_asymmetric_painting>`, we now build local trees. This
chapter implements Relate's bespoke **agglomerative tree-building algorithm**
-- a bottom-up clustering method that exploits the asymmetry to correctly
identify which pairs of lineages coalesce first.

Standard hierarchical clustering (UPGMA, neighbor-joining) requires symmetric
distances. Relate's distances are asymmetric by design, and this asymmetry
carries the directional signal needed for correct topology. The algorithm here
is purpose-built for this asymmetric case.

.. admonition:: Prerequisites

   - :ref:`relate_asymmetric_painting` (Gear 1): the asymmetric distance
     matrix :math:`d(i,j)` and why it matters
   - Basic familiarity with agglomerative (bottom-up) clustering: start with
     :math:`N` clusters, iteratively merge the two closest, repeat until one
     cluster remains


The Agglomerative Idea
========================

Agglomerative clustering builds a tree from the **leaves up**:

1. Start with :math:`N` singleton clusters (one per haplotype)
2. Find the pair of clusters that should merge next
3. Create a new internal node as the parent of those two clusters
4. Update the distance matrix
5. Repeat until a single root remains

The result is a rooted binary tree with :math:`N` leaves and :math:`N-1`
internal nodes. For Relate, each such tree is the **local genealogy** at a
specific genomic position.

The question is: **how do we choose which pair to merge?** Standard UPGMA
merges the pair with the smallest average distance. Neighbor-joining uses a
corrected distance that accounts for the "neighborhood" of each node. Neither
works correctly with asymmetric distances.


Step 1: The Merging Criterion
==============================

Relate's merging criterion uses the asymmetric distance matrix to identify
the pair :math:`(i, j)` that **coalesces with each other before coalescing
with any other lineage**. The key idea is:

For each pair :math:`(i, j)`, the **minimum symmetrized distance** is:

.. math::

   d_{\min}(i, j) = \min\bigl(d(i, j),\; d(j, i)\bigr)

This minimum captures the "most recent shared ancestry" between :math:`i`
and :math:`j`. If :math:`d_{\min}(i,j)` is small, at least one direction
of the relationship suggests close ancestry.

The pair to merge is:

.. math::

   (i^*, j^*) = \arg\min_{i \neq j} d_{\min}(i, j)

with ties broken by the symmetrized distance :math:`d(i,j) + d(j,i)`.

.. admonition:: Probability Aside -- Why Minimum, Not Average?

   The minimum :math:`\min(d(i,j), d(j,i))` selects the direction that
   suggests the most recent common ancestor. Consider two haplotypes where
   :math:`d(i,j)` is large (many derived alleles in :math:`i` not in
   :math:`j`) but :math:`d(j,i)` is small (few derived alleles in :math:`j`
   not in :math:`i`). The small :math:`d(j,i)` tells us that :math:`j` has
   very few private derived alleles relative to :math:`i` -- their most
   recent common ancestor is on the lineage that :math:`j` provides the
   cleanest view of. Taking the average would dilute this signal with the
   large :math:`d(i,j)`, which primarily reflects mutations private to
   :math:`i`'s deeper lineage.

.. code-block:: python

   import numpy as np

   def find_pair_to_merge(D, active):
       """Find the pair of active clusters to merge next.

       Parameters
       ----------
       D : ndarray of shape (N, N)
           Asymmetric distance matrix.
       active : set of int
           Indices of currently active (unmerged) clusters.

       Returns
       -------
       i, j : int
           Indices of the pair to merge.
       """
       best_d_min = np.inf
       best_d_sym = np.inf
       best_pair = None

       active_list = sorted(active)
       for idx_a, i in enumerate(active_list):
           for j in active_list[idx_a + 1:]:
               d_min = min(D[i, j], D[j, i])
               d_sym = D[i, j] + D[j, i]

               # Primary criterion: smallest min distance
               # Tiebreaker: smallest symmetrized distance
               if (d_min < best_d_min or
                   (d_min == best_d_min and d_sym < best_d_sym)):
                   best_d_min = d_min
                   best_d_sym = d_sym
                   best_pair = (i, j)

       return best_pair

   # Example
   D_example = np.array([
       [0.0, 1.2, 3.5, 3.8],
       [0.8, 0.0, 3.2, 3.5],
       [3.5, 3.2, 0.0, 0.5],
       [3.8, 3.5, 0.7, 0.0],
   ])
   active = {0, 1, 2, 3}
   i, j = find_pair_to_merge(D_example, active)
   print(f"Pair to merge: ({i}, {j})")
   print(f"  d_min({i},{j}) = {min(D_example[i,j], D_example[j,i]):.2f}")


Step 2: Updating the Distance Matrix
======================================

After merging clusters :math:`i` and :math:`j` into a new cluster :math:`c`,
we need to compute distances between :math:`c` and every remaining cluster
:math:`k`. Relate uses a simple rule:

.. math::

   d(c, k) = \min\bigl(d(i, k),\; d(j, k)\bigr)

.. math::

   d(k, c) = \min\bigl(d(k, i),\; d(k, j)\bigr)

.. admonition:: Probability Aside -- Why Minimum Linkage?

   The minimum linkage rule (also called "single linkage" in classical
   clustering) takes the distance from the closer member of the merged
   cluster. This is appropriate because once :math:`i` and :math:`j` have
   coalesced into a common ancestor :math:`c`, the distance from :math:`k`
   to :math:`c` should reflect the closest path through either :math:`i`
   or :math:`j`. If :math:`k` is close to :math:`i` but far from :math:`j`,
   the relevant genealogical distance is the small one -- because :math:`k`
   can reach the common ancestor of :math:`i` and :math:`j` via :math:`i`.

.. code-block:: python

   def update_distances(D, i, j, c, active):
       """Update the distance matrix after merging i and j into c.

       Parameters
       ----------
       D : ndarray of shape (M, M)
           Distance matrix (will be modified in-place, M >= max index).
       i, j : int
           Indices of the merged pair.
       c : int
           Index of the new cluster.
       active : set of int
           Currently active clusters (should include c, not i or j).
       """
       for k in active:
           if k == c:
               continue
           # Distance from new cluster to k: minimum of the two children
           D[c, k] = min(D[i, k], D[j, k])
           # Distance from k to new cluster
           D[k, c] = min(D[k, i], D[k, j])

       # Self-distance
       D[c, c] = 0.0


Step 3: The Complete Tree-Building Algorithm
=============================================

Putting it all together: start with :math:`N` leaves, iteratively merge pairs,
and record the tree structure.

.. code-block:: python

   class TreeNode:
       """A node in a binary tree."""

       def __init__(self, node_id, left=None, right=None, is_leaf=True):
           self.id = node_id
           self.left = left
           self.right = right
           self.is_leaf = is_leaf
           self.leaf_ids = {node_id} if is_leaf else set()

       def __repr__(self):
           if self.is_leaf:
               return f"Leaf({self.id})"
           return f"Node({self.id}, L={self.left.id}, R={self.right.id})"


   def build_tree(D_orig, N):
       """Build a rooted binary tree from an asymmetric distance matrix.

       Parameters
       ----------
       D_orig : ndarray of shape (N, N)
           Asymmetric distance matrix for N haplotypes.
       N : int
           Number of haplotypes (leaves).

       Returns
       -------
       root : TreeNode
           Root of the binary tree.
       merge_order : list of (int, int, int)
           Sequence of (child1, child2, parent) merges.
       """
       # Allocate space for up to 2N-1 nodes (N leaves + N-1 internal)
       max_nodes = 2 * N - 1
       D = np.full((max_nodes, max_nodes), np.inf)
       D[:N, :N] = D_orig.copy()
       for i in range(N):
           D[i, i] = 0.0

       # Initialize: each haplotype is a leaf node
       nodes = {}
       for i in range(N):
           nodes[i] = TreeNode(i)

       active = set(range(N))
       next_id = N
       merge_order = []

       # Iteratively merge pairs
       for step in range(N - 1):
           # Find the best pair to merge
           i, j = find_pair_to_merge(D, active)

           # Create new internal node
           c = next_id
           parent_node = TreeNode(c, left=nodes[i], right=nodes[j],
                                   is_leaf=False)
           parent_node.leaf_ids = nodes[i].leaf_ids | nodes[j].leaf_ids
           nodes[c] = parent_node

           # Update bookkeeping
           active.remove(i)
           active.remove(j)
           active.add(c)

           # Update distances
           update_distances(D, i, j, c, active)

           merge_order.append((i, j, c))
           next_id += 1

       # The last remaining node is the root
       root_id = active.pop()
       return nodes[root_id], merge_order


   # Example: build a tree from the example distance matrix
   D_example = np.array([
       [0.0, 1.2, 3.5, 3.8],
       [0.8, 0.0, 3.2, 3.5],
       [3.5, 3.2, 0.0, 0.5],
       [3.8, 3.5, 0.7, 0.0],
   ])

   root, merges = build_tree(D_example, N=4)
   print("Merge order:")
   for c1, c2, parent in merges:
       print(f"  Merge {c1} + {c2} -> {parent}")
   print(f"\nRoot: {root}")
   print(f"Root leaves: {root.leaf_ids}")


Step 4: Visualizing the Tree
==============================

A Newick-format string lets us see the tree structure:

.. code-block:: python

   def to_newick(node):
       """Convert a TreeNode to Newick format string.

       Parameters
       ----------
       node : TreeNode

       Returns
       -------
       str
           Newick representation.
       """
       if node.is_leaf:
           return str(node.id)
       left_str = to_newick(node.left)
       right_str = to_newick(node.right)
       return f"({left_str},{right_str})"

   newick = to_newick(root) + ";"
   print(f"Newick: {newick}")


Step 5: Position-Specific Trees
==================================

In practice, Relate builds a tree at each genomic position (or more precisely,
at each **focal SNP** where the tree topology may change). Adjacent SNPs may
share the same tree if no recombination occurred between them.

The full procedure:

1. For each focal SNP :math:`s`, compute the :math:`N \times N` asymmetric
   distance matrix :math:`D(s)` using the painting from Gear 1
2. Build a tree :math:`\mathcal{T}(s)` from :math:`D(s)` using the
   agglomerative algorithm
3. Adjacent trees that are identical can be merged into a single tree
   spanning a genomic interval

.. code-block:: python

   def build_local_trees(haplotypes, positions, recomb_rate, mu):
       """Build a local tree at each focal SNP.

       Parameters
       ----------
       haplotypes : ndarray of shape (N, L)
       positions : ndarray of float, shape (L,)
       recomb_rate : float
       mu : float

       Returns
       -------
       trees : list of (start_pos, end_pos, root)
           Local trees with genomic intervals.
       """
       N, L = haplotypes.shape
       trees = []
       prev_newick = None

       for s in range(L):
           # Compute asymmetric distance matrix at this focal SNP
           D = compute_distance_matrix(
               haplotypes, positions, recomb_rate, mu, focal_snp=s)

           # Build tree
           root, _ = build_tree(D, N)
           newick = to_newick(root)

           if newick != prev_newick:
               # New tree topology -- start a new interval
               start_pos = positions[s]
               trees.append({
                   'start': start_pos,
                   'root': root,
                   'newick': newick,
                   'focal_snp': s,
               })
               prev_newick = newick
           # If same topology, the previous tree's interval extends

       # Set end positions
       for i in range(len(trees) - 1):
           trees[i]['end'] = trees[i + 1]['start']
       if trees:
           trees[-1]['end'] = positions[-1] + 1

       return trees

   # Example
   np.random.seed(42)
   N, L = 4, 15
   haps = np.random.binomial(1, 0.3, size=(N, L))
   positions = np.arange(L, dtype=float) * 1000

   local_trees = build_local_trees(haps, positions, recomb_rate=1e-3,
                                    mu=0.01)
   print(f"Number of distinct local trees: {len(local_trees)}")
   for t in local_trees:
       print(f"  [{t['start']:.0f}, {t['end']:.0f}): {t['newick']}")


Step 6: Correctness Under the Infinite-Sites Model
====================================================

Under the infinite-sites model with no recombination, every pair of
haplotypes is separated by a set of mutations that partition the sample
into carriers and non-carriers. The asymmetric distance :math:`d(i,j)`
counts the mutations derived in :math:`i` and ancestral in :math:`j`,
which directly reflects the tree topology.

Let's verify this on a simulated example:

.. code-block:: python

   def verify_tree_building():
       """Verify tree building on a known genealogy.

       True tree (no recombination):
              6
             / \\
            5    \\
           / \\    \\
          4   \\    \\
         / \\   \\    \\
        0   1   2   3

       Haplotypes (each column = one mutation):
         mutation on branch to (0,1) clade:  site 0
         mutation on branch to (0,1,2) clade: site 1
         mutation on branch to 0:            site 2
         mutation on branch to 3:            site 3
       """
       haps = np.array([
           [1, 1, 1, 0, 0, 0],  # Haplotype 0
           [1, 1, 0, 0, 0, 0],  # Haplotype 1
           [0, 1, 0, 0, 0, 0],  # Haplotype 2
           [0, 0, 0, 1, 0, 0],  # Haplotype 3
       ])
       L = haps.shape[1]
       positions = np.arange(L, dtype=float) * 100

       D = compute_distance_matrix(haps, positions, recomb_rate=1e-3,
                                    mu=0.001, focal_snp=3)

       root, merges = build_tree(D, N=4)
       newick = to_newick(root)

       print("Verification: known tree")
       print(f"  True tree:  ((0,1),2),3)")
       print(f"  Built tree: {newick}")

       # Check: 0 and 1 should be siblings (merged first among {0,1,2})
       first_merge = merges[0]
       print(f"\n  First merge: {first_merge[0]} + {first_merge[1]}")

       # Check: 0 and 1 are in the same subtree
       def get_leaves(node):
           if node.is_leaf:
               return {node.id}
           return get_leaves(node.left) | get_leaves(node.right)

       left_leaves = get_leaves(root.left)
       right_leaves = get_leaves(root.right)
       print(f"  Left subtree:  {left_leaves}")
       print(f"  Right subtree: {right_leaves}")

       # 0 and 1 should be in the same subtree
       if {0, 1}.issubset(left_leaves) or {0, 1}.issubset(right_leaves):
           print("  [ok] 0 and 1 are siblings")
       else:
           print("  [FAIL] 0 and 1 are not siblings")

   verify_tree_building()


Exercises
==========

.. admonition:: Exercise 1: UPGMA comparison

   Implement UPGMA (Unweighted Pair Group Method with Arithmetic Mean) on
   the symmetrized distance matrix :math:`d_s(i,j) = (d(i,j) + d(j,i))/2`.
   Compare the resulting tree with Relate's tree on a simulated dataset
   where the true tree is known. When do they agree? When do they differ?

.. admonition:: Exercise 2: Neighbor-joining comparison

   Implement the neighbor-joining algorithm on the symmetrized distances.
   Compare with Relate's agglomerative algorithm. Neighbor-joining is
   known to be statistically consistent for additive distances -- does
   it outperform Relate when the distance matrix is nearly symmetric?

.. admonition:: Exercise 3: Scaling test

   Time the tree-building algorithm for :math:`N = 10, 50, 100, 500`.
   What is the empirical scaling? (Hint: the find-pair step is
   :math:`O(N^2)` and there are :math:`N-1` merges, so the total should
   be :math:`O(N^3)`.)

Next: :ref:`relate_branch_lengths` -- estimating coalescence times by MCMC.
