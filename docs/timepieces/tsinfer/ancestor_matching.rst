.. _tsinfer_ancestor_matching:

===================================
Gear 3: Ancestor Matching
===================================

   *Build the scaffold from the top down. The oldest beams bear the weight
   of everything below.*

Ancestor matching is where the tree sequence takes shape -- where we
**assemble the movement**. We take the putative ancestors generated in
:ref:`Gear 1 <tsinfer_ancestor_generation>` and thread them together using
the copying model from :ref:`Gear 2 <tsinfer_copying_model>`, building the
genealogical scaffold from the root down to the most recent ancestors.

The principle is simple: each ancestor is a mosaic of *older* ancestors.
By processing ancestors from oldest to youngest, we ensure that when we
match an ancestor, all its potential parents are already in the tree.
In a watchmaker's workshop, this is the stage where the mainplate is laid
down first, then the bridges, then the wheel train -- each layer resting
on the one below it.

.. admonition:: Prerequisites

   This chapter builds directly on:

   - :ref:`tsinfer_ancestor_generation` (Gear 1): you need to understand
     how ancestors are constructed, what their time proxy means, and how
     they are grouped by time
   - :ref:`tsinfer_copying_model` (Gear 2): you need to understand the
     Viterbi algorithm, NONCOPY handling, and the path-to-edges conversion
   - The :ref:`Li & Stephens HMM chapter <lshmm_timepiece>` for the
     theoretical foundations of the copying model


Step 1: The Matching Order
============================

Ancestors are processed **from oldest to youngest**. Recall that "time"
is the derived allele frequency:

.. math::

   t_a = \frac{\text{count of derived alleles at focal site of } a}{n}

The oldest ancestor (the **ultimate ancestor** at :math:`t = 1.0`) is
placed first. It carries the ancestral allele at every site and serves
as the root of the tree.

Next, ancestors are processed in **time groups**: all ancestors at the
same time are matched simultaneously against the already-placed ancestors.

.. admonition:: Probability Aside -- Why Oldest First?

   The oldest-first ordering is not arbitrary. It follows from a fundamental
   property of genealogies: an ancestor at time :math:`t_1` can only be the
   parent of an ancestor at time :math:`t_2 < t_1` (you must exist before
   your descendants). By processing from oldest to youngest, we guarantee
   that when we run the Li & Stephens HMM for a given ancestor, every
   possible parent is already in the reference panel. This is a
   **topological sort** of the genealogical DAG. If we processed ancestors
   in random order, a young ancestor might be placed before its true parent,
   and the Viterbi algorithm would be forced to choose a suboptimal copying
   source.

.. code-block:: python

   import numpy as np

   def matching_order(ancestors):
       """Determine the order for ancestor matching.

       Parameters
       ----------
       ancestors : list of dict
           Ancestors with 'time' field, sorted oldest first.

       Returns
       -------
       groups : list of list of dict
           Groups of ancestors at the same time.
       """
       groups = []
       current_time = None
       current_group = []

       for anc in ancestors:
           if anc['time'] != current_time:
               # New time group -- flush the previous group
               if current_group:
                   groups.append(current_group)
               current_group = [anc]
               current_time = anc['time']
           else:
               # Same time -- add to current group
               current_group.append(anc)

       if current_group:
           groups.append(current_group)

       return groups

   # Example
   ancestors_example = [
       {'time': 1.0, 'focal': -1},   # Ultimate ancestor
       {'time': 0.8, 'focal': 3},
       {'time': 0.8, 'focal': 7},
       {'time': 0.6, 'focal': 1},
       {'time': 0.4, 'focal': 5},
       {'time': 0.4, 'focal': 9},
       {'time': 0.4, 'focal': 12},
       {'time': 0.2, 'focal': 2},
   ]

   groups = matching_order(ancestors_example)
   for i, group in enumerate(groups):
       times = [a['time'] for a in group]
       focals = [a['focal'] for a in group]
       print(f"Group {i}: time={times[0]:.1f}, "
             f"{len(group)} ancestors, focals={focals}")

With the matching order established, the next step is to construct the
reference panel that each ancestor will be matched against.


Step 2: Building the Reference Panel
=======================================

When matching ancestors in time group :math:`g`, the reference panel
consists of **all ancestors from groups** :math:`0, 1, \ldots, g-1`
(i.e., all strictly older ancestors).

The panel is stored as a matrix :math:`H` of shape :math:`(m, k)` where
:math:`m` is the number of inference sites and :math:`k` is the number
of older ancestors. Entries where an ancestor doesn't span a site are
set to ``NONCOPY``.

.. admonition:: Confusion Buster -- Why the Panel Grows Over Time

   Notice that the reference panel gets *larger* with each time group. When
   matching the oldest real ancestors (those just below the ultimate ancestor),
   the panel contains only the ultimate ancestor -- a single all-zeros
   haplotype. When matching the youngest ancestors, the panel contains *every*
   older ancestor. This means the HMM has more states to choose from for
   younger ancestors, which makes their matching both more accurate (more
   potential parents) and more expensive (larger state space). The
   :math:`O(k)` trick from :ref:`Gear 2 <tsinfer_copying_model>` keeps
   the per-site cost linear in :math:`k`.

.. code-block:: python

   NONCOPY = -2

   def build_reference_panel(placed_ancestors, num_inference_sites):
       """Build the reference panel from already-placed ancestors.

       Parameters
       ----------
       placed_ancestors : list of dict
           Ancestors already in the tree sequence.
       num_inference_sites : int
           Total number of inference sites.

       Returns
       -------
       panel : ndarray of shape (num_inference_sites, k)
           Reference panel with NONCOPY entries.
       node_ids : list of int
           Node ID for each column of the panel.
       """
       k = len(placed_ancestors)
       # Initialize everything as NONCOPY (not available for copying)
       panel = np.full((num_inference_sites, k), NONCOPY, dtype=int)

       for col, anc in enumerate(placed_ancestors):
           start = anc['start']
           end = anc['end']
           # Fill in the ancestor's haplotype where it is defined
           panel[start:end, col] = anc['haplotype']

       node_ids = [anc.get('node_id', col) for col, anc in
                   enumerate(placed_ancestors)]
       return panel, node_ids

   # Example
   placed = [
       {'haplotype': np.zeros(10, dtype=int), 'start': 0, 'end': 10,
        'node_id': 0},  # Ultimate ancestor
       {'haplotype': np.array([0, 0, 1, 1, 0, 0, 1, 0]),
        'start': 1, 'end': 9, 'node_id': 1},
   ]
   panel, node_ids = build_reference_panel(placed, num_inference_sites=10)
   print(f"Panel shape: {panel.shape}")
   print(f"Panel (showing NONCOPY as '.'):")
   for site in range(10):
       row = ''.join(str(x) if x >= 0 else '.' for x in panel[site])
       print(f"  Site {site}: {row}")

Now we have the pieces: ancestors to match, a reference panel to match
against, and the Viterbi engine from Gear 2. Let's put them together.


Step 3: Match and Add Edges
==============================

For each ancestor in the current time group, we:

1. Run the Viterbi algorithm against the reference panel
2. Convert the Viterbi path to edges
3. Add the edges and the ancestor node to the growing tree sequence

Each edge represents a segment of the ancestor's genome that was "copied"
from a specific older ancestor -- in genealogical terms, a parent-child
relationship over a genomic interval.

.. code-block:: python

   class TreeSequenceBuilder:
       """Incrementally builds a tree sequence from matching results.

       This is a simplified version of tsinfer's internal builder.
       It accumulates nodes and edges as ancestors and samples are matched.
       """

       def __init__(self, sequence_length, num_inference_sites, positions):
           self.sequence_length = sequence_length
           self.positions = positions
           self.num_inference_sites = num_inference_sites
           self.nodes = []   # (time, is_sample)
           self.edges = []   # (left, right, parent, child)
           self.next_id = 0

       def add_node(self, time, is_sample=False):
           """Add a node and return its ID."""
           node_id = self.next_id
           self.nodes.append({'id': node_id, 'time': time,
                              'is_sample': is_sample})
           self.next_id += 1
           return node_id

       def add_edges_from_path(self, path, child_id, ref_node_ids):
           """Convert a Viterbi path to edges and add them.

           Parameters
           ----------
           path : ndarray of shape (m,)
               Viterbi path (index into reference panel).
           child_id : int
               Node ID of the child.
           ref_node_ids : list of int
               Node IDs for each reference index.
           """
           m = len(path)
           # Walk through the path, emitting one edge per contiguous segment
           seg_start = 0
           current_ref = path[0]

           for ell in range(1, m):
               if path[ell] != current_ref:
                   # Copying source changed -- emit edge for old segment
                   left = self.positions[seg_start]
                   right = self.positions[ell]
                   parent = ref_node_ids[current_ref]
                   self.edges.append((left, right, parent, child_id))
                   seg_start = ell
                   current_ref = path[ell]

           # Final segment extends to end of sequence
           left = self.positions[seg_start]
           right = self.positions[m - 1] + 1  # Or sequence_length
           parent = ref_node_ids[current_ref]
           self.edges.append((left, right, parent, child_id))

       def summary(self):
           """Print a summary of the tree sequence."""
           print(f"Nodes: {len(self.nodes)}")
           print(f"Edges: {len(self.edges)}")
           samples = sum(1 for n in self.nodes if n['is_sample'])
           print(f"Samples: {samples}")

   # Example: build a small tree sequence
   positions = np.arange(0, 10000, 1000, dtype=float)
   builder = TreeSequenceBuilder(
       sequence_length=10000,
       num_inference_sites=10,
       positions=positions
   )

   # Add the ultimate ancestor (root)
   root_id = builder.add_node(time=1.0)
   print(f"Root node: {root_id}")

   # Add an ancestor at time 0.8
   anc_id = builder.add_node(time=0.8)
   # Suppose Viterbi says it copies from the root everywhere
   path = np.zeros(10, dtype=int)  # Always copying from ref 0 (the root)
   builder.add_edges_from_path(path, anc_id, ref_node_ids=[root_id])

   builder.summary()

With the individual matching step defined, we can now write the complete
loop that processes all ancestors from oldest to youngest.


Step 4: The Complete Ancestor Matching Loop
=============================================

Putting it all together:

.. code-block:: python

   def match_ancestors(ancestors, inference_sites, positions,
                       recombination_rate, mismatch_ratio,
                       sequence_length):
       """Run the complete ancestor matching phase.

       Parameters
       ----------
       ancestors : list of dict
           Ancestors sorted by time (oldest first). First is the ultimate
           ancestor.
       inference_sites : ndarray of int
           Inference site positions.
       positions : ndarray of float
           Genomic positions of inference sites.
       recombination_rate : float
           Per-unit recombination rate.
       mismatch_ratio : float
           Mismatch-to-recombination ratio.
       sequence_length : float
           Total sequence length.

       Returns
       -------
       builder : TreeSequenceBuilder
           The constructed tree sequence.
       """
       m = len(inference_sites)
       builder = TreeSequenceBuilder(sequence_length, m, positions)

       # Phase 1: Add the ultimate ancestor as root (the mainplate)
       ultimate = ancestors[0]
       root_id = builder.add_node(time=ultimate['time'])
       ultimate['node_id'] = root_id

       placed = [ultimate]

       # Phase 2: Process remaining ancestors by time groups
       groups = matching_order(ancestors[1:])

       for group_idx, group in enumerate(groups):
           # Build reference panel from all placed (older) ancestors
           panel, ref_node_ids = build_reference_panel(placed, m)
           k = len(ref_node_ids)

           # Compute HMM parameters for this panel size
           rho = np.zeros(m)
           mu = np.zeros(m)
           for ell in range(1, m):
               d = positions[ell] - positions[ell - 1]
               rho[ell] = 1 - np.exp(-d * recombination_rate / max(k, 1))
               mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                                     / max(k, 1))
           mu[0] = mu[1] if m > 1 else 1e-6

           # Match each ancestor in this group against the panel
           for anc in group:
               node_id = builder.add_node(time=anc['time'])
               anc['node_id'] = node_id

               # Build the query (ancestor's haplotype over its interval)
               query = np.full(m, -1, dtype=int)  # -1 = undefined
               query[anc['start']:anc['end']] = anc['haplotype']

               # Run Viterbi (only over the ancestor's interval)
               start, end = anc['start'], anc['end']
               if end - start < 2:
                   # Too short for HMM -- just parent to root
                   left = positions[start]
                   right = positions[end - 1] + 1
                   builder.edges.append((left, right, root_id, node_id))
               else:
                   sub_query = query[start:end]
                   sub_panel = panel[start:end]
                   sub_rho = rho[start:end]
                   sub_mu = mu[start:end]
                   sub_rho[0] = 0.0  # No recombination at first site

                   # Only use columns that have copiable entries
                   copiable_cols = []
                   for col in range(k):
                       if np.any(sub_panel[:, col] != NONCOPY):
                           copiable_cols.append(col)

                   if len(copiable_cols) == 0:
                       # No references available -- parent to root
                       left = positions[start]
                       right = positions[end - 1] + 1
                       builder.edges.append((left, right, root_id, node_id))
                   else:
                       sub_panel_c = sub_panel[:, copiable_cols]
                       sub_ref_ids = [ref_node_ids[c] for c in copiable_cols]
                       # Use viterbi_ls_with_noncopy from Gear 2
                       path = viterbi_ls_with_noncopy(
                           sub_query, sub_panel_c, sub_rho, sub_mu)
                       # Map path back to node IDs
                       mapped_path = np.array([copiable_cols[p]
                                               for p in path])
                       builder.add_edges_from_path(
                           mapped_path, node_id,
                           ref_node_ids=ref_node_ids)

               # This ancestor is now placed and available for future groups
               placed.append(anc)

           print(f"  Group {group_idx}: time={group[0]['time']:.2f}, "
                 f"matched {len(group)} ancestors, "
                 f"panel size={k}")

       return builder

After matching, the raw tree sequence may contain **polytomies** --
multiple children sharing the same parent over the same interval. Path
compression resolves these into a more refined topology.


Step 5: Path Compression
==========================

After matching, the tree sequence often contains many edges that share
the same parent-child-interval pattern. **Path compression** merges these
redundant edges through synthetic **path compression (PC) nodes**.

The problem
-----------

Consider two ancestors :math:`a_1` and :math:`a_2` that both copy from
ancestor :math:`a_0` over the same genomic interval :math:`[l, r)`. In
the raw tree sequence, we have two edges:

.. code-block:: text

   Edge: [l, r) parent=a0, child=a1
   Edge: [l, r) parent=a0, child=a2

If :math:`a_1` and :math:`a_2` are siblings in the true tree (they coalesce
before reaching :math:`a_0`), we're missing the intermediate coalescent
node. The tree has a **polytomy** (multiple children sharing one parent)
that should be resolved.

The solution
--------------

Path compression inserts a synthetic node :math:`c` between :math:`a_0`
and its children:

.. code-block:: text

   Edge: [l, r) parent=a0, child=c
   Edge: [l, r) parent=c,  child=a1
   Edge: [l, r) parent=c,  child=a2

The PC node :math:`c` is assigned a time slightly less than :math:`a_0`:

.. math::

   t_c = t_{a_0} - \frac{1}{2^{32}}

This epsilon offset ensures the node is strictly younger than its parent,
which is required for a valid tree sequence.

.. admonition:: Probability Aside -- Path Compression and Tree Topology

   Path compression doesn't change the *data likelihood* of the tree
   sequence -- the mutation patterns are identical whether we have a
   polytomy or a resolved binary split. But it can change downstream
   inferences. A polytomy says "we don't know the order of coalescence
   among these lineages," while a resolved binary tree implies a specific
   order. Path compression is a heuristic: it assumes that if multiple
   children share the same parent over the same interval, they likely
   coalesced together just below that parent. This is often (but not
   always) correct. The epsilon time offset means the PC node has no
   meaningful biological time -- it is a topological device.

.. code-block:: python

   PC_TIME_EPSILON = 1.0 / (2**32)

   def path_compress(edges, nodes):
       """Apply path compression to a set of edges.

       Parameters
       ----------
       edges : list of (left, right, parent, child)
           Raw edges from matching.
       nodes : list of dict
           Node information.

       Returns
       -------
       new_edges : list of (left, right, parent, child)
           Compressed edges.
       new_nodes : list of dict
           Updated nodes (may include new PC nodes).
       """
       from collections import defaultdict

       # Group edges by (left, right, parent) to find shared patterns
       groups = defaultdict(list)
       for left, right, parent, child in edges:
           groups[(left, right, parent)].append(child)

       new_edges = []
       new_nodes = list(nodes)
       next_id = max(n['id'] for n in nodes) + 1

       for (left, right, parent), children in groups.items():
           if len(children) <= 1:
               # Only one child -- no compression needed
               for child in children:
                   new_edges.append((left, right, parent, child))
           else:
               # Multiple children share the same parent and interval
               # Insert a PC node between parent and children
               parent_time = None
               for n in nodes:
                   if n['id'] == parent:
                       parent_time = n['time']
                       break

               # PC node sits just below the parent in time
               pc_time = parent_time - PC_TIME_EPSILON
               pc_node = {'id': next_id, 'time': pc_time,
                          'is_sample': False}
               new_nodes.append(pc_node)

               # Parent -> PC node (single edge replaces multiple)
               new_edges.append((left, right, parent, next_id))

               # PC node -> each child
               for child in children:
                   new_edges.append((left, right, next_id, child))

               next_id += 1

       return new_edges, new_nodes

   # Example
   edges_raw = [
       (0, 5000, 0, 1),
       (0, 5000, 0, 2),
       (0, 5000, 0, 3),
       (5000, 10000, 0, 1),
       (5000, 10000, 1, 4),
   ]
   nodes_raw = [
       {'id': 0, 'time': 1.0, 'is_sample': False},
       {'id': 1, 'time': 0.8, 'is_sample': False},
       {'id': 2, 'time': 0.6, 'is_sample': False},
       {'id': 3, 'time': 0.4, 'is_sample': False},
       {'id': 4, 'time': 0.2, 'is_sample': True},
   ]

   compressed_edges, compressed_nodes = path_compress(edges_raw, nodes_raw)
   print("Original edges:")
   for e in edges_raw:
       print(f"  [{e[0]}, {e[1]}): {e[2]} -> {e[3]}")
   print(f"\nCompressed edges:")
   for e in compressed_edges:
       print(f"  [{e[0]}, {e[1]}): {e[2]} -> {e[3]}")
   print(f"\nNew PC nodes:")
   for n in compressed_nodes[len(nodes_raw):]:
       print(f"  Node {n['id']}: time={n['time']:.10f}")


Step 6: Verification
=====================

After ancestor matching, we can verify the ancestor tree sequence:

.. code-block:: python

   def verify_ancestor_tree(builder):
       """Verify basic properties of the ancestor tree sequence."""
       print("Ancestor tree verification:")

       # 1. Every non-root node has at least one parent edge
       root_id = 0  # Ultimate ancestor
       children_seen = set()
       for left, right, parent, child in builder.edges:
           children_seen.add(child)
       non_root_nodes = {n['id'] for n in builder.nodes if n['id'] != root_id}
       orphans = non_root_nodes - children_seen
       print(f"  [{'ok' if len(orphans) == 0 else 'FAIL'}] "
             f"All non-root nodes have parent edges "
             f"(orphans: {len(orphans)})")

       # 2. No self-loops (a node cannot be its own parent)
       self_loops = [(l, r, p, c) for l, r, p, c in builder.edges
                     if p == c]
       print(f"  [{'ok' if len(self_loops) == 0 else 'FAIL'}] "
             f"No self-loops")

       # 3. Parent time > child time for all edges
       time_map = {n['id']: n['time'] for n in builder.nodes}
       bad_times = []
       for left, right, parent, child in builder.edges:
           if time_map.get(parent, 0) <= time_map.get(child, 0):
               bad_times.append((parent, child))
       print(f"  [{'ok' if len(bad_times) == 0 else 'FAIL'}] "
             f"Parent time > child time for all edges "
             f"(violations: {len(bad_times)})")

       # 4. Summary statistics
       print(f"\n  Nodes: {len(builder.nodes)}")
       print(f"  Edges: {len(builder.edges)}")
       print(f"  Time range: [{min(time_map.values()):.4f}, "
             f"{max(time_map.values()):.4f}]")

With the ancestor tree built and verified, we have the scaffold of the
genealogy -- the movement's mainplate, bridges, and wheel train are in
place. The final chapter, :ref:`Gear 4 <tsinfer_sample_matching>`, will
thread the actual samples through this scaffold and polish the result
into a finished tree sequence.


Exercises
==========

.. admonition:: Exercise 1: Panel growth analysis

   As ancestors are matched, the reference panel grows. Plot the panel
   size :math:`k` as a function of the time group index. How does this
   affect the computational cost per ancestor?

.. admonition:: Exercise 2: Edge count analysis

   For a simulated dataset (use ``msprime`` with :math:`n = 100`,
   :math:`L = 10^5`), run ancestor generation and matching. How many
   edges are created? How does this compare to the number of edges in
   the true tree sequence from the simulation?

.. admonition:: Exercise 3: Path compression impact

   Compare the tree sequence before and after path compression. How many
   PC nodes are created? What fraction of edges are affected? Visualize
   a local tree before and after compression.

Next: :ref:`tsinfer_sample_matching` -- threading the actual samples through the ancestor tree.
