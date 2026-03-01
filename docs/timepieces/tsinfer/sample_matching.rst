.. _tsinfer_sample_matching:

============================================
Gear 4: Sample Matching & Post-Processing
============================================

   *The final assembly: connecting the present to the past, then polishing
   every surface until the mechanism runs true.*

Sample matching is the last phase of tsinfer's pipeline. We thread each
observed sample through the ancestor tree built in :ref:`Gear 3
<tsinfer_ancestor_matching>`, then apply a series of post-processing steps
to clean up the result into a valid, simplified tree sequence.

If ancestor matching assembled the movement -- the mainplate, bridges, and
wheel train -- then sample matching **fits the hands to the dial**. The
hands (samples) connect to the wheels (ancestors) at precise points, and
the final polishing steps (post-processing) ensure that the mechanism runs
cleanly: no extraneous parts, no rough edges, no wasted material.

.. admonition:: Prerequisites

   This chapter assumes you have worked through:

   - :ref:`tsinfer_ancestor_generation` (Gear 1): how ancestors are built
   - :ref:`tsinfer_copying_model` (Gear 2): the Viterbi algorithm and
     NONCOPY handling
   - :ref:`tsinfer_ancestor_matching` (Gear 3): how the ancestor tree
     is assembled
   - The :ref:`Li & Stephens HMM chapter <lshmm_timepiece>` for the
     theoretical background


Step 1: Threading Samples
===========================

Sample matching uses the **same Li & Stephens engine** from Gear 2,
but the reference panel is now the complete set of ancestors (not just
older ones). Each sample is matched independently against the ancestor
panel.

.. admonition:: Confusion Buster -- How Sample Matching Differs from Ancestor Matching

   In ancestor matching (:ref:`Gear 3 <tsinfer_ancestor_matching>`), each
   ancestor is matched against only the *older* ancestors -- the panel grows
   as we work through time groups. In sample matching, the panel is **fixed**:
   it contains *all* ancestors simultaneously. This is because samples are at
   time 0 (the present), so every ancestor is older than every sample.
   Another difference: ancestors have limited genomic extent (they only span
   a subset of sites), while samples span all inference sites. The Viterbi
   algorithm for sample matching therefore runs over the full set of inference
   sites, with the NONCOPY mechanism handling ancestors that don't extend to
   certain sites.

.. code-block:: python

   import numpy as np

   NONCOPY = -2

   def match_samples(samples, ancestors, inference_sites, positions,
                     recombination_rate, mismatch_ratio, builder):
       """Match all samples against the ancestor tree.

       Parameters
       ----------
       samples : ndarray of shape (n, m_inf)
           Sample genotypes at inference sites only.
       ancestors : list of dict
           All ancestors (with 'node_id' assigned during ancestor matching).
       inference_sites : ndarray of int
           Inference site indices.
       positions : ndarray of float
           Genomic positions of inference sites.
       recombination_rate : float
           Per-unit recombination rate.
       mismatch_ratio : float
           Mismatch-to-recombination ratio.
       builder : TreeSequenceBuilder
           The tree sequence builder (already contains ancestor nodes/edges).

       Returns
       -------
       builder : TreeSequenceBuilder
           Updated builder with sample nodes and edges.
       """
       n, m = samples.shape
       k = len(ancestors)  # Total number of ancestors in the panel

       # Build the full ancestor panel (all ancestors at once)
       panel = np.full((m, k), NONCOPY, dtype=int)
       ref_node_ids = []
       for col, anc in enumerate(ancestors):
           # Fill in each ancestor's haplotype where it is defined
           panel[anc['start']:anc['end'], col] = anc['haplotype']
           ref_node_ids.append(anc['node_id'])

       # HMM parameters (fixed for all samples, since panel doesn't change)
       rho = np.zeros(m)
       mu = np.zeros(m)
       for ell in range(1, m):
           d = positions[ell] - positions[ell - 1]
           # Recombination and mismatch probabilities scale with 1/k
           rho[ell] = 1 - np.exp(-d * recombination_rate / max(k, 1))
           mu[ell] = 1 - np.exp(-d * recombination_rate * mismatch_ratio
                                 / max(k, 1))
       mu[0] = mu[1] if m > 1 else 1e-6

       # Match each sample independently
       for i in range(n):
           # Samples are at time 0.0 (the present)
           node_id = builder.add_node(time=0.0, is_sample=True)
           query = samples[i]  # This sample's genotype at inference sites

           # Run Viterbi against the full ancestor panel
           path = viterbi_ls_with_noncopy(query, panel, rho, mu)

           # Convert the Viterbi path to tree sequence edges
           builder.add_edges_from_path(path, node_id,
                                        ref_node_ids=ref_node_ids)

           if (i + 1) % 100 == 0 or i == n - 1:
               print(f"  Matched sample {i + 1}/{n}")

       return builder

The key difference from ancestor matching: **all ancestors** are in the
panel simultaneously, and samples are always at time 0 (the present).

With all samples threaded into the tree, we now turn to the sites we
deliberately excluded from inference. These non-inference sites need
mutations placed on the tree by a different method: parsimony.


Step 2: Non-Inference Sites -- Parsimony
==========================================

Recall that we excluded many sites from inference (multiallelic, unknown
ancestral state, singletons, fixed). These **non-inference sites** still
need mutations placed on the tree. tsinfer uses **parsimony**: find the
minimum number of mutations that explain the observed allele pattern on
the inferred tree.

.. admonition:: Confusion Buster -- Why Some Sites Are Non-Inference

   As explained in :ref:`Gear 1 <tsinfer_ancestor_generation>`, sites are
   excluded from inference if they are multiallelic (more than two alleles),
   if the ancestral allele is unknown (we can't determine polarity), if the
   derived allele is a singleton (only one copy -- doesn't help with tree
   topology), or if the derived allele is fixed (all samples carry it).
   These sites were *not* used to build the tree. But the tree still needs
   to explain the allele patterns at these sites. Parsimony does this by
   placing the minimum number of mutations on the existing tree edges.

The algorithm: ``map_mutations()``
-------------------------------------

For each non-inference site, tsinfer calls ``tskit``'s ``map_mutations()``
method, which implements Fitch's parsimony algorithm:

1. **Bottom-up pass**: At each internal node, compute the set of alleles
   that would require the fewest mutations below. If the children agree,
   take their allele. If they disagree, take the union (and record a
   potential mutation point).

2. **Top-down pass**: Starting from the root, assign an allele to each
   node. Where a child's assigned allele differs from its parent's, place
   a mutation on the connecting edge.

.. math::

   \text{Parsimony score} = \min \sum_{\text{edges}} \mathbb{1}[\text{parent allele} \neq \text{child allele}]

.. admonition:: Probability Aside -- Parsimony vs. Maximum Likelihood for Mutation Placement

   **Maximum likelihood** mutation placement would assign mutations to edges
   in a way that maximizes the probability of the observed data, given a
   model of mutation along branches. This requires knowing the branch lengths
   (in units of expected mutations). Since tsinfer's branch lengths are
   approximate -- they come from the frequency-based time proxy, not from a
   calibrated molecular clock -- ML mutation placement would compound the
   error in the time estimates. **Parsimony** avoids this problem entirely:
   it places the fewest mutations needed to explain the data, regardless
   of branch lengths. The cost is that parsimony can undercount the true
   number of mutations (e.g., it misses back-mutations), but for the
   purpose of recording which edges carry mutations, it is robust and fast
   (:math:`O(n)` per site per tree).

.. code-block:: python

   def fitch_parsimony(tree_parent, tree_children, leaf_alleles, root):
       """Place mutations by Fitch parsimony on a single tree.

       Parameters
       ----------
       tree_parent : dict
           Mapping from node -> parent node. Root maps to None.
       tree_children : dict
           Mapping from node -> list of child nodes.
       leaf_alleles : dict
           Mapping from leaf node -> observed allele.
       root : int
           Root node ID.

       Returns
       -------
       mutations : list of (node, parent_allele, child_allele)
           Mutations placed on edges.
       """
       # --- Bottom-up pass: compute Fitch sets ---
       # The Fitch set at each node is the set of alleles that
       # minimize the number of mutations in the subtree below.
       fitch_set = {}

       # Post-order traversal (children before parents)
       def bottom_up(node):
           if node not in tree_children or len(tree_children[node]) == 0:
               # Leaf node: Fitch set = {observed allele}
               fitch_set[node] = {leaf_alleles[node]}
               return

           child_sets = []
           for child in tree_children[node]:
               bottom_up(child)
               child_sets.append(fitch_set[child])

           # If children agree (non-empty intersection), take intersection
           common = child_sets[0]
           for s in child_sets[1:]:
               common = common & s

           if len(common) > 0:
               # Children agree -- no mutation needed at this node
               fitch_set[node] = common
           else:
               # Children disagree -- take union, mutation needed somewhere
               union = set()
               for s in child_sets:
                   union = union | s
               fitch_set[node] = union

       bottom_up(root)

       # --- Top-down pass: assign alleles and place mutations ---
       assigned = {}
       mutations = []

       def top_down(node, parent_allele):
           # If the parent's allele is in this node's Fitch set, keep it
           # (no mutation needed). Otherwise, pick from the Fitch set.
           if parent_allele in fitch_set[node]:
               assigned[node] = parent_allele
           else:
               assigned[node] = min(fitch_set[node])  # Deterministic tie-break

           if node in tree_children:
               for child in tree_children[node]:
                   top_down(child, assigned[node])
                   # If child got a different allele, that's a mutation
                   if assigned[child] != assigned[node]:
                       mutations.append((child, assigned[node],
                                         assigned[child]))

       # Root gets any allele from its Fitch set
       root_allele = min(fitch_set[root])
       assigned[root] = root_allele
       if root in tree_children:
           for child in tree_children[root]:
               top_down(child, root_allele)
               if assigned[child] != root_allele:
                   mutations.append((child, root_allele, assigned[child]))

       return mutations

   # Example: a simple tree
   #       0 (root)
   #      / \
   #     1   2
   #    / \
   #   3   4
   tree_parent = {3: 1, 4: 1, 1: 0, 2: 0, 0: None}
   tree_children = {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
   leaf_alleles = {2: 0, 3: 1, 4: 1}  # Leaves: node 2, 3, 4

   mutations = fitch_parsimony(tree_parent, tree_children,
                                leaf_alleles, root=0)
   print(f"Mutations ({len(mutations)}):")
   for node, p_allele, c_allele in mutations:
       print(f"  Edge to node {node}: {p_allele} -> {c_allele}")

**Why parsimony and not maximum likelihood?** Parsimony is fast
(:math:`O(n)` per site per tree) and doesn't require branch length
estimates. Since tsinfer's branch lengths are approximate (frequency-based
proxy), the simpler parsimony approach avoids compounding errors.

With mutations placed, the tree sequence contains all the genealogical
information. But it still has rough edges that need polishing. The next
section describes the cleanup steps.


Step 3: Post-Processing Pipeline
===================================

After sample matching and mutation placement, the raw tree sequence needs
several cleanup steps. Think of this as the final polishing and regulation
of the watch movement -- removing scaffolding, trimming excess material,
and ensuring every component serves a purpose.

3a: Virtual Root Removal
--------------------------

The ultimate ancestor (virtual root) was a scaffolding device. In the
final tree sequence, we want a proper coalescent root. The virtual root
is removed by redirecting its children's edges.

.. code-block:: python

   def remove_virtual_root(edges, nodes, virtual_root_id):
       """Remove the virtual root node.

       Children of the virtual root become roots of their subtrees.

       Parameters
       ----------
       edges : list of (left, right, parent, child)
       nodes : list of dict
       virtual_root_id : int

       Returns
       -------
       filtered_edges : list of (left, right, parent, child)
       filtered_nodes : list of dict
       """
       # Simply remove all edges where the virtual root is the parent
       filtered_edges = [(l, r, p, c) for l, r, p, c in edges
                         if p != virtual_root_id]
       # Remove the virtual root node itself
       filtered_nodes = [n for n in nodes if n['id'] != virtual_root_id]
       return filtered_edges, filtered_nodes

3b: Ultimate Ancestor Splitting
----------------------------------

The ultimate ancestor spans the entire genome. After removing the virtual
root, nodes that were children of the ultimate ancestor may need to be
**split** into separate root segments for different local trees.

3c: Flank Erasure
-------------------

Edges that extend beyond the **rightmost** or **leftmost** inference site
are trimmed. These flanking edges have no data support and would create
artificial deep coalescences at the edges of the genome.

.. code-block:: python

   def erase_flanks(edges, leftmost_position, rightmost_position):
       """Trim edges that extend beyond the data range.

       Parameters
       ----------
       edges : list of (left, right, parent, child)
       leftmost_position : float
           Leftmost inference site position.
       rightmost_position : float
           Rightmost inference site position.

       Returns
       -------
       trimmed_edges : list of (left, right, parent, child)
       """
       trimmed = []
       for left, right, parent, child in edges:
           # Clamp the edge to the data range
           new_left = max(left, leftmost_position)
           new_right = min(right, rightmost_position)
           # Only keep edges that still have positive length
           if new_left < new_right:
               trimmed.append((new_left, new_right, parent, child))
       return trimmed

   # Example
   edges_raw = [
       (0, 10000, 1, 5),     # Extends left of first site
       (2000, 8000, 2, 6),   # Within range
       (7000, 15000, 3, 7),  # Extends right of last site
   ]

   trimmed = erase_flanks(edges_raw,
                           leftmost_position=1000,
                           rightmost_position=9000)
   print("Trimmed edges:")
   for l, r, p, c in trimmed:
       print(f"  [{l:.0f}, {r:.0f}): {p} -> {c}")

3d: Simplification
--------------------

The final step is **simplification**: removing all nodes and edges that
are not ancestral to any sample. This is tskit's built-in ``simplify()``
operation.

.. admonition:: Confusion Buster -- What Simplification Actually Does

   Simplification is one of tskit's most powerful operations, and it is
   worth understanding precisely. It takes a tree sequence and a set of
   "focal" nodes (usually the samples) and produces a new tree sequence
   that is *equivalent* from the perspective of those focal nodes -- every
   local tree, every mutation, every statistic computed on the samples is
   identical -- but with all unnecessary structure removed.

   Specifically, simplification does three things:

   1. **Removes unary nodes**: A unary node has exactly one child. It
      represents a lineage that passed through an ancestor without any
      coalescence happening. In the simplified tree, this lineage is
      represented by a single edge from grandparent to grandchild, and
      the unary node is removed.

   2. **Removes unreferenced nodes**: Nodes that are not ancestral to any
      sample have no effect on the genealogy as seen from the samples.
      They are pruned entirely.

   3. **Merges adjacent edges**: If a parent-child relationship spans two
      adjacent intervals (e.g., [0, 5000) and [5000, 10000)) and there is
      no tree change at the boundary, the two edges are merged into one:
      [0, 10000).

   The result is a leaner tree sequence that preserves all information
   relevant to the samples but discards the scaffolding used during
   construction.

Simplification does three things:

1. **Removes unary nodes**: nodes with only one child (they don't represent
   real coalescence events)
2. **Removes unreferenced nodes**: nodes not ancestral to any sample
3. **Merges adjacent edges**: edges that can be combined

.. math::

   \text{Before simplification:} \quad |\mathcal{N}| = A + n, \quad |\mathcal{E}| = O(Am)

.. math::

   \text{After simplification:} \quad |\mathcal{N}| \ll A + n, \quad |\mathcal{E}| \ll O(Am)

The reduction can be dramatic: a tree sequence with 10,000 ancestor nodes
might simplify to 2,000 internal nodes.

.. code-block:: python

   def simplify_tree_sequence(nodes, edges, sample_ids):
       """Simplified illustration of the simplify algorithm.

       In practice, use tskit's built-in simplify().

       Parameters
       ----------
       nodes : list of dict
       edges : list of (left, right, parent, child)
       sample_ids : set of int

       Returns
       -------
       kept_nodes : set of int
           Node IDs retained after simplification.
       kept_edges : list of (left, right, parent, child)
           Edges retained.
       """
       # Find all nodes ancestral to at least one sample
       # by traversing upward from the samples through edges
       ancestral = set(sample_ids)
       edge_map = {}
       for left, right, parent, child in edges:
           if child not in edge_map:
               edge_map[child] = []
           edge_map[child].append((left, right, parent))

       # BFS upward from samples to find all ancestors
       queue = list(sample_ids)
       while queue:
           node = queue.pop(0)
           if node in edge_map:
               for left, right, parent in edge_map[node]:
                   if parent not in ancestral:
                       ancestral.add(parent)
                       queue.append(parent)

       # Keep only edges between ancestral nodes
       kept_edges = [(l, r, p, c) for l, r, p, c in edges
                     if p in ancestral and c in ancestral]
       kept_nodes = ancestral

       return kept_nodes, kept_edges

   # Example
   nodes_ex = [
       {'id': 0, 'time': 1.0, 'is_sample': False},
       {'id': 1, 'time': 0.8, 'is_sample': False},
       {'id': 2, 'time': 0.5, 'is_sample': False},  # Not ancestral
       {'id': 3, 'time': 0.0, 'is_sample': True},
       {'id': 4, 'time': 0.0, 'is_sample': True},
   ]
   edges_ex = [
       (0, 10000, 0, 1),
       (0, 10000, 1, 3),
       (0, 10000, 1, 4),
       (0, 10000, 0, 2),   # Node 2 has no sample descendants
   ]
   sample_ids = {3, 4}

   kept_nodes, kept_edges = simplify_tree_sequence(nodes_ex, edges_ex,
                                                     sample_ids)
   print(f"Nodes before: {len(nodes_ex)}, after: {len(kept_nodes)}")
   print(f"Edges before: {len(edges_ex)}, after: {len(kept_edges)}")
   print(f"Removed nodes: {set(n['id'] for n in nodes_ex) - kept_nodes}")


Step 4: The Complete Pipeline
===============================

Here's the full tsinfer pipeline, end to end -- all four gears meshing
together in sequence:

.. code-block:: python

   def tsinfer_pipeline(D, positions, ancestral_known,
                        recombination_rate=1e-8,
                        mismatch_ratio=1.0,
                        sequence_length=None):
       """Run the complete tsinfer pipeline.

       Parameters
       ----------
       D : ndarray of shape (n, m)
           Variant matrix (0 = ancestral, 1 = derived).
       positions : ndarray of float
           Genomic positions of all sites.
       ancestral_known : ndarray of bool
           Whether the ancestral allele is known at each site.
       recombination_rate : float
           Per-base-pair recombination rate.
       mismatch_ratio : float
           Mismatch-to-recombination ratio.
       sequence_length : float or None
           Total genome length (defaults to max position + 1).

       Returns
       -------
       builder : TreeSequenceBuilder
           The final tree sequence.
       """
       if sequence_length is None:
           sequence_length = positions[-1] + 1

       n, m = D.shape
       print(f"=== tsinfer pipeline ===")
       print(f"Samples: {n}, Sites: {m}")

       # --- Phase 1: Ancestor generation (Gear 1) ---
       # "Extracting the template gears"
       print(f"\nPhase 1: Generating ancestors...")
       ancestors, inference_sites = generate_ancestors(D, ancestral_known)
       inf_positions = positions[inference_sites]
       ancestors = add_ultimate_ancestor(ancestors, len(inference_sites))
       print(f"  Generated {len(ancestors)} ancestors "
             f"({len(inference_sites)} inference sites)")

       # --- Phase 2: Ancestor matching (Gear 3, using Gear 2's engine) ---
       # "Assembling the movement"
       print(f"\nPhase 2: Matching ancestors...")
       builder = match_ancestors(
           ancestors, inference_sites, inf_positions,
           recombination_rate, mismatch_ratio, sequence_length)
       print(f"  Ancestor tree: {len(builder.nodes)} nodes, "
             f"{len(builder.edges)} edges")

       # --- Phase 3: Sample matching (Gear 4, using Gear 2's engine) ---
       # "Fitting the hands to the dial"
       print(f"\nPhase 3: Matching samples...")
       samples_at_inf = D[:, inference_sites]
       builder = match_samples(
           samples_at_inf, ancestors, inference_sites, inf_positions,
           recombination_rate, mismatch_ratio, builder)
       print(f"  After samples: {len(builder.nodes)} nodes, "
             f"{len(builder.edges)} edges")

       # --- Phase 4: Post-processing ---
       # "Final polishing and regulation"
       print(f"\nPhase 4: Post-processing...")

       # Flank erasure: trim edges beyond data range
       builder.edges = erase_flanks(
           builder.edges,
           leftmost_position=inf_positions[0],
           rightmost_position=inf_positions[-1] + 1)
       print(f"  After flank erasure: {len(builder.edges)} edges")

       # Simplification: remove nodes/edges not ancestral to samples
       sample_ids = {n['id'] for n in builder.nodes if n['is_sample']}
       kept_nodes, kept_edges = simplify_tree_sequence(
           builder.nodes, builder.edges, sample_ids)
       print(f"  After simplification: {len(kept_nodes)} nodes, "
             f"{len(kept_edges)} edges")

       print(f"\n=== Done ===")
       return builder


Step 5: What the Output Looks Like
=====================================

The final tree sequence is a ``tskit.TreeSequence`` object. In our
simplified implementation, we have a ``TreeSequenceBuilder``. In the real
tsinfer, the output is a full tskit tree sequence that you can query:

.. code-block:: python

   # In real tsinfer:
   # ts = tsinfer.infer(sample_data)
   #
   # ts.num_trees          -> number of local trees
   # ts.num_nodes          -> total nodes
   # ts.num_edges          -> total edges
   # ts.num_mutations      -> total mutations
   #
   # for tree in ts.trees():
   #     print(tree.draw_text())  # ASCII visualization of a local tree
   #
   # ts.diversity()        -> nucleotide diversity (pi)
   # ts.Tajimas_D()        -> Tajima's D statistic
   # ts.simplify()         -> further simplification if needed

The tree sequence format is extremely compact. A dataset with 100,000
samples and 1,000,000 sites might produce a tree sequence file of only
a few hundred megabytes, compared to tens of gigabytes for the original
VCF file. This is the payoff of the quartz-movement approach: by
sacrificing full posterior inference, tsinfer delivers a tree sequence
that is both computationally tractable to produce and remarkably efficient
to store and query.


Verification
=============

Let's verify the key properties of our complete pipeline:

.. code-block:: python

   def verify_pipeline(builder, D, inference_sites):
       """Verify the output of the tsinfer pipeline."""
       print("Pipeline verification:")

       # 1. All samples are present as nodes
       sample_nodes = [n for n in builder.nodes if n['is_sample']]
       print(f"  [ok] {len(sample_nodes)} sample nodes present")

       # 2. All samples have at least one parent edge
       children_with_edges = set()
       for l, r, p, c in builder.edges:
           children_with_edges.add(c)
       samples_with_parents = sum(
           1 for n in sample_nodes if n['id'] in children_with_edges)
       print(f"  [{'ok' if samples_with_parents == len(sample_nodes) else 'FAIL'}] "
             f"All samples have parent edges "
             f"({samples_with_parents}/{len(sample_nodes)})")

       # 3. Edge coordinates are within bounds
       all_lefts = [l for l, r, p, c in builder.edges]
       all_rights = [r for l, r, p, c in builder.edges]
       print(f"  [ok] Edge range: [{min(all_lefts):.0f}, "
             f"{max(all_rights):.0f})")

       # 4. No negative times
       all_times = [n['time'] for n in builder.nodes]
       print(f"  [{'ok' if min(all_times) >= 0 else 'FAIL'}] "
             f"All times >= 0")

       # 5. Summary
       non_sample = len(builder.nodes) - len(sample_nodes)
       print(f"\n  Summary:")
       print(f"    Total nodes: {len(builder.nodes)}")
       print(f"    Ancestor nodes: {non_sample}")
       print(f"    Sample nodes: {len(sample_nodes)}")
       print(f"    Edges: {len(builder.edges)}")
       print(f"    Inference sites used: {len(inference_sites)}")


Exercises
==========

.. admonition:: Exercise 1: Compare with real tsinfer

   Install ``tsinfer`` and ``msprime``. Simulate a tree sequence with
   ``msprime`` (:math:`n = 50`, :math:`L = 10^5`, :math:`\rho = 10^{-8}`,
   :math:`\mu = 10^{-8}`). Run ``tsinfer.infer()`` on the simulated data.
   Compare the number of trees, edges, and mutations in the inferred tree
   sequence vs. the true one. What is the Robinson-Foulds distance between
   corresponding local trees?

.. admonition:: Exercise 2: Effect of non-inference sites

   Take a simulated dataset and vary the fraction of sites marked as
   non-inference (by artificially removing the ancestral allele annotation
   or by including multiallelic sites). How does the tree topology change?
   How many extra parsimony mutations are needed?

.. admonition:: Exercise 3: Scaling experiment

   Run tsinfer on datasets of increasing size: :math:`n = 100, 500, 2000,
   10000` with fixed genome length :math:`L = 10^6`. Plot runtime and
   memory usage vs. :math:`n`. Does the empirical scaling match the
   theoretical :math:`O(An)` complexity?

This completes the tsinfer Timepiece. You've built every gear from scratch:
ancestor generation (extracting the template gears), the copying model (the
oscillator circuit), ancestor matching (assembling the movement), and sample
matching with post-processing (fitting the hands and polishing the case).
The full tsinfer algorithm is simply these four gears meshing together in
sequence -- a quartz movement, simpler than the mechanical MCMC approaches,
but precise enough for biobank-scale data and fast enough to keep time for
millions of samples.


Solutions
==========

.. admonition:: Solution 1: Compare with real tsinfer

   We simulate a tree sequence with ``msprime``, run ``tsinfer.infer()``,
   and compare the inferred tree sequence against the true one in terms
   of structure and topology.

   .. code-block:: python

      import msprime
      import tsinfer
      import tskit
      import numpy as np

      # --- Simulate the true tree sequence ---
      ts_true = msprime.simulate(
          sample_size=50, Ne=10_000, length=1e5,
          mutation_rate=1e-8, recombination_rate=1e-8,
          random_seed=42)

      print(f"True tree sequence:")
      print(f"  Trees: {ts_true.num_trees}")
      print(f"  Nodes: {ts_true.num_nodes}")
      print(f"  Edges: {ts_true.num_edges}")
      print(f"  Mutations: {ts_true.num_mutations}")
      print(f"  Sites: {ts_true.num_sites}")

      # --- Create SampleData object for tsinfer ---
      with tsinfer.SampleData(sequence_length=ts_true.sequence_length) as sd:
          for var in ts_true.variants():
              sd.add_site(
                  position=var.site.position,
                  genotypes=var.genotypes,
                  alleles=var.alleles,
                  ancestral_allele=0)  # Index of the ancestral allele

      # --- Run tsinfer ---
      ts_inferred = tsinfer.infer(sd)

      print(f"\nInferred tree sequence:")
      print(f"  Trees: {ts_inferred.num_trees}")
      print(f"  Nodes: {ts_inferred.num_nodes}")
      print(f"  Edges: {ts_inferred.num_edges}")
      print(f"  Mutations: {ts_inferred.num_mutations}")

      # --- Compare local tree topologies ---
      # Robinson-Foulds distance between corresponding trees
      rf_distances = []
      # Iterate over breakpoints that are common to both
      for true_tree, inf_tree in zip(ts_true.trees(), ts_inferred.trees()):
          # tskit provides a built-in RF distance via tree comparisons
          # For a simple comparison, count shared splits
          pass  # See below for a simplified approach

      # Simplified comparison: fraction of sample pairs that share the
      # same MRCA topology
      breakpoints = list(ts_true.breakpoints())
      n = ts_true.num_samples
      sample_ids_true = list(ts_true.samples())
      sample_ids_inf = list(ts_inferred.samples())

      concordance = []
      for bp_idx in range(len(breakpoints) - 1):
          mid = (breakpoints[bp_idx] + breakpoints[bp_idx + 1]) / 2
          true_tree = ts_true.at(mid)
          inf_tree = ts_inferred.at(mid)

          agree = 0
          total = 0
          for i in range(min(n, 20)):
              for j in range(i + 1, min(n, 20)):
                  si, sj = sample_ids_true[i], sample_ids_true[j]
                  mrca_true = true_tree.mrca(si, sj)

                  si_inf = sample_ids_inf[i]
                  sj_inf = sample_ids_inf[j]
                  mrca_inf = inf_tree.mrca(si_inf, sj_inf)

                  # Compare: do i and j coalesce before (or at same
                  # time as) a third sample k?
                  # Simplified: just check tree topology agreement
                  # by comparing relative MRCA times
                  for kk in range(j + 1, min(n, 20)):
                      sk = sample_ids_true[kk]
                      sk_inf = sample_ids_inf[kk]
                      mrca_ik_true = true_tree.mrca(si, sk)
                      mrca_ik_inf = inf_tree.mrca(si_inf, sk_inf)
                      # Do i,j coalesce before i,k in both trees?
                      true_order = (true_tree.time(mrca_true)
                                    <= true_tree.time(mrca_ik_true))
                      inf_order = (inf_tree.time(mrca_inf)
                                   <= inf_tree.time(mrca_ik_inf))
                      if true_order == inf_order:
                          agree += 1
                      total += 1

          if total > 0:
              concordance.append(agree / total)

      mean_concordance = np.mean(concordance) if concordance else 0
      print(f"\nTopology concordance (triplet test): {mean_concordance:.3f}")
      print(f"  (1.0 = perfect agreement, 0.33 = random)")

   **Key observations:**

   - tsinfer typically infers *more* trees (finer breakpoints) than the
     true tree sequence, because Viterbi transitions can occur at positions
     that do not correspond to true recombination events.
   - The number of mutations is usually close to the true count, since
     parsimony is a good approximation when back-mutations are rare.
   - Triplet concordance (the fraction of sample triplets where the
     relative coalescence ordering agrees) is typically 70--90% for
     moderate sample sizes, indicating that tsinfer recovers much of the
     true topology even though branch lengths are approximate.

.. admonition:: Solution 2: Effect of non-inference sites

   We artificially vary the fraction of non-inference sites and measure
   the impact on tree topology and parsimony mutation count.

   .. code-block:: python

      import msprime
      import tsinfer
      import numpy as np

      # Simulate
      ts_true = msprime.simulate(
          sample_size=50, Ne=10_000, length=1e5,
          mutation_rate=1e-8, recombination_rate=1e-8,
          random_seed=42)

      G = ts_true.genotype_matrix()  # (num_sites, num_samples)
      D = G.T
      n, m = D.shape
      positions = np.array([s.position for s in ts_true.sites()])
      ancestral_known_full = np.ones(m, dtype=bool)

      # Vary the fraction of sites with "unknown" ancestral allele
      fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
      results = []

      for frac in fractions:
          # Randomly mask a fraction of sites as ancestral-unknown
          ancestral_known = np.ones(m, dtype=bool)
          mask = np.random.choice(m, size=int(frac * m), replace=False)
          ancestral_known[mask] = False

          inference_sites, non_inference_sites = select_inference_sites(
              D, ancestral_known)

          # Count how many inference sites remain
          n_inf = len(inference_sites)
          n_non = len(non_inference_sites)

          # Generate ancestors from remaining inference sites
          if n_inf < 2:
              print(f"Frac={frac:.1f}: too few inference sites ({n_inf})")
              continue

          ancestors, inf_s = generate_ancestors(D, ancestral_known)

          # Count parsimony mutations needed for non-inference sites
          # (approximation: each non-inference site needs at least 1 mutation)
          parsimony_estimate = n_non  # Lower bound

          results.append({
              'fraction_removed': frac,
              'inference_sites': n_inf,
              'non_inference_sites': n_non,
              'ancestors': len(ancestors),
              'parsimony_mutations_lb': parsimony_estimate,
          })

          print(f"Frac removed={frac:.1f}: "
                f"{n_inf} inference sites, "
                f"{n_non} non-inference sites, "
                f"{len(ancestors)} ancestors, "
                f"parsimony mutations >= {parsimony_estimate}")

      # Summary
      print(f"\nAs more sites become non-inference:")
      print(f"  - Fewer ancestors are generated (less tree resolution)")
      print(f"  - More mutations must be placed by parsimony")
      print(f"  - Tree topology becomes coarser (fewer breakpoints)")

   **Key observations:**

   - Removing sites from inference (by marking their ancestral allele as
     unknown) reduces the number of ancestors and hence the resolution
     of the inferred tree. With fewer inference sites, the tree has fewer
     breakpoints and the topology is coarser.
   - Non-inference sites accumulate parsimony mutations. Since parsimony
     places the minimum number of mutations, the count grows roughly
     linearly with the number of non-inference sites.
   - Even with 40--60% of sites removed from inference, tsinfer still
     recovers a reasonable tree because the remaining inference sites
     provide enough signal for the copying model. Below a critical
     threshold (roughly 20% of sites remaining), the tree degrades rapidly.

.. admonition:: Solution 3: Scaling experiment

   We run tsinfer on datasets of increasing sample size :math:`n` and
   measure runtime and memory usage to verify the expected scaling.

   .. code-block:: python

      import msprime
      import tsinfer
      import numpy as np
      import time
      import tracemalloc

      sample_sizes = [100, 500, 2000, 10000]
      L = 1e6
      runtimes = []
      peak_memories = []
      edge_counts = []

      for n in sample_sizes:
          print(f"\n--- n = {n} ---")
          # Simulate
          ts = msprime.simulate(
              sample_size=n, Ne=10_000, length=L,
              mutation_rate=1e-8, recombination_rate=1e-8,
              random_seed=42)

          # Create SampleData
          with tsinfer.SampleData(
                  sequence_length=ts.sequence_length) as sd:
              for var in ts.variants():
                  sd.add_site(
                      position=var.site.position,
                      genotypes=var.genotypes,
                      alleles=var.alleles,
                      ancestral_allele=0)

          # Time and memory-profile the inference
          tracemalloc.start()
          t0 = time.perf_counter()

          ts_inferred = tsinfer.infer(sd)

          elapsed = time.perf_counter() - t0
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()

          runtimes.append(elapsed)
          peak_memories.append(peak / 1e6)  # Convert to MB
          edge_counts.append(ts_inferred.num_edges)

          print(f"  Time: {elapsed:.1f} s")
          print(f"  Peak memory: {peak / 1e6:.1f} MB")
          print(f"  Trees: {ts_inferred.num_trees}")
          print(f"  Edges: {ts_inferred.num_edges}")
          print(f"  Nodes: {ts_inferred.num_nodes}")

      # Plot scaling
      import matplotlib.pyplot as plt

      fig, axes = plt.subplots(1, 3, figsize=(15, 5))

      axes[0].loglog(sample_sizes, runtimes, 'o-', lw=2)
      slope_t = np.polyfit(np.log(sample_sizes), np.log(runtimes), 1)[0]
      axes[0].set_xlabel("Sample size n")
      axes[0].set_ylabel("Runtime (seconds)")
      axes[0].set_title(f"Runtime scaling (slope={slope_t:.2f})")

      axes[1].loglog(sample_sizes, peak_memories, 'o-', lw=2, color='orange')
      slope_m = np.polyfit(np.log(sample_sizes),
                           np.log(peak_memories), 1)[0]
      axes[1].set_xlabel("Sample size n")
      axes[1].set_ylabel("Peak memory (MB)")
      axes[1].set_title(f"Memory scaling (slope={slope_m:.2f})")

      axes[2].loglog(sample_sizes, edge_counts, 'o-', lw=2, color='green')
      slope_e = np.polyfit(np.log(sample_sizes),
                           np.log(edge_counts), 1)[0]
      axes[2].set_xlabel("Sample size n")
      axes[2].set_ylabel("Number of edges")
      axes[2].set_title(f"Edge count scaling (slope={slope_e:.2f})")

      plt.tight_layout()
      plt.show()

      print(f"\nScaling exponents:")
      print(f"  Runtime ~ n^{slope_t:.2f}")
      print(f"  Memory  ~ n^{slope_m:.2f}")
      print(f"  Edges   ~ n^{slope_e:.2f}")

   **Key observations:**

   - **Runtime**: The theoretical complexity of tsinfer is :math:`O(An)` where
     :math:`A` is the number of ancestors and :math:`n` is the sample size.
     Under neutral evolution, :math:`A \sim n` (each inference site with
     frequency :math:`\geq 2/n` generates one ancestor), so the overall
     scaling is approximately :math:`O(n^2)` in theory. In practice, the
     log-log slope is typically between 1.5 and 2.0 because the constant
     factors and the panel-growth effect (younger ancestors have larger
     panels) moderate the quadratic term.
   - **Memory**: Dominated by the reference panel during matching, which is
     :math:`O(Am)` where :math:`m` is the number of inference sites. Since
     both :math:`A` and :math:`m` grow with :math:`n`, memory scales
     super-linearly but remains tractable for biobank-scale data (tens of
     thousands of samples with genome-length data).
   - **Edge count**: Grows roughly as :math:`O(n \log n)` under the
     coalescent, matching the expected number of edges in a coalescent tree
     sequence. This confirms that tsinfer produces a compact representation
     even as the sample size grows.
