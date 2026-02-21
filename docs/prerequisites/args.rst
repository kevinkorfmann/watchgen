.. _args:

====================================
Ancestral Recombination Graphs
====================================

   *A single tree tells one story. An ARG tells the whole history.*

Why Trees Aren't Enough
=========================

In the :ref:`coalescent_theory` chapter, we built genealogical trees -- the
escapement mechanism that drives every Timepiece in this book. We derived the
exponential waiting times, the coalescence rates :math:`\binom{k}{2}`, and the
mutation model. But we made one critical simplifying assumption: **no
recombination**.

That assumption is like describing a watch movement by tracing a single gear
train from mainspring to hands. It gives a correct picture of *one* pathway of
force transmission, but a real movement has many interacting gear trains --
the going train for timekeeping, the keyless works for winding, perhaps a
chronograph train for elapsed timing. The complete mechanical drawing must show
all of these, along with every point where one train's output becomes another
train's input.

An **Ancestral Recombination Graph (ARG)** is that complete mechanical drawing
for a genome's history. It captures *every* coalescence event and *every*
recombination event across the entire sequence. Where a single coalescent tree
shows one gear train, the ARG shows all of them, plus the couplings between
them.

But before we can understand the ARG, we need to understand what recombination
is and why it matters.

What Is Recombination?
========================

Recombination is a biological process that occurs during **meiosis** -- the
special cell division that produces sperm and egg cells (gametes). In meiosis,
an organism that carries two copies of each chromosome (one from each parent)
produces cells with just one copy. But before that division happens, something
remarkable occurs: the maternal and paternal copies of each chromosome
physically align, and segments are swapped between them.

.. admonition:: Biology aside: why does recombination exist?

   Recombination shuffles genetic material between the two copies of a
   chromosome you inherited from your parents. The result is that the
   chromosome you pass on to *your* child is a mosaic -- some segments came
   from your mother's chromosome, others from your father's.

   Why would evolution favor this shuffling? There are several theories, but
   the dominant one is that recombination breaks up linkage between
   deleterious mutations, allowing natural selection to act more efficiently.
   Without recombination, a bad mutation on a chromosome is permanently
   linked to every other variant on that chromosome. With recombination,
   good and bad variants can be separated, giving selection a clearer target.

   For our purposes, the key consequence is simple: **different positions
   along your genome have different genealogical histories.** The stretch of
   DNA near the beginning of chromosome 1 traces back through a different
   sequence of ancestors than the stretch near the end.

Here's the concrete implication. Consider three individuals -- Alice, Bob, and
Carol -- and look at two positions on the same chromosome, position 1 and
position 1000. At position 1, perhaps Alice and Bob share a recent common
ancestor (their great-great-grandmother), while Carol's lineage diverged much
earlier. At position 1000, the situation might be reversed: Alice and Carol
share a recent ancestor, while Bob's lineage is the outlier. Both genealogies
are correct -- they just reflect different parts of the genome's history, shaped
by recombination events in the intervening generations.

This is the fundamental reason we need ARGs rather than single trees.

What We've Established So Far
-------------------------------

From the coalescent theory chapter, we have these tools:

- Coalescence times are exponential with rate :math:`\binom{k}{2}` when there
  are :math:`k` lineages (measured in coalescent time units -- multiples of
  :math:`2N_e` generations)
- Mutations arise as a Poisson process at rate :math:`\theta/2` per coalescent
  time unit per base pair
- The expected time to the most recent common ancestor is
  :math:`2(1 - 1/n)` coalescent time units

Now we need to extend this framework to handle recombination.

Recombination in the Coalescent
================================

In the coalescent with recombination, we still trace lineages backwards in time,
but now two types of events can happen:

1. **Coalescence**: Two lineages merge (as before). Rate :math:`\binom{k}{2}`
   when there are :math:`k` lineages. This is the same mechanism we derived in
   the :ref:`coalescent_theory` chapter.

2. **Recombination**: A single lineage **splits** into two at a random
   breakpoint. Each of the :math:`k` lineages recombines at rate
   :math:`\rho/2`, where :math:`\rho = 4N_e r` and :math:`r` is the
   per-generation recombination probability across the region.

.. admonition:: Where does :math:`\rho = 4N_e r` come from?

   This follows the same logic as :math:`\theta = 4N_e\mu` for mutation (derived
   in the :ref:`coalescent_theory` chapter). In the Wright-Fisher model, a
   recombination event occurs with probability :math:`r` per generation per
   lineage. A branch of length :math:`\ell` in coalescent time units spans
   :math:`2N_e \cdot \ell` generations. The expected number of recombination
   events on this branch is:

   .. math::

      r \times 2N_e \ell = \frac{4N_e r}{2} \cdot \ell = \frac{\rho}{2} \cdot \ell

   So the recombination rate **per coalescent time unit per lineage** is
   :math:`\rho/2`, exactly paralleling the mutation rate :math:`\theta/2`.

.. admonition:: What is a "breakpoint"?

   A breakpoint is the position along the genome where a recombination event
   cuts a lineage in two. Think of the genome as a long strip of paper
   representing the sequence from position 0 to position :math:`S`. A
   recombination event at breakpoint :math:`x` tears the strip at position
   :math:`x`: the left piece (positions :math:`[0, x)`) traces back through
   one ancestor, and the right piece (positions :math:`[x, S)`) traces back
   through a different ancestor. Before the breakpoint, the genealogy is one
   tree; after it, the genealogy may be a different tree.

   In the watchmaking metaphor, a breakpoint is like a point where one gear
   train hands off to another -- the same shaft, but driven by different
   mechanisms on either side.

Going backwards in time, these two event types have opposite effects:

- **Coalescence** reduces the number of lineages by 1 (two become one)
- **Recombination** increases the number of lineages by 1 (one becomes two)

This creates a competition. Coalescence simplifies the history; recombination
complicates it. The total rate at which *something* happens when there are
:math:`k` lineages is:

.. math::

   \text{Rate of next event} = \underbrace{\binom{k}{2}}_{\text{coalescence}} +
   \underbrace{\frac{k\rho}{2}}_{\text{recombination}}

.. admonition:: Probability aside: competing exponential processes

   As we established in the :ref:`coalescent_theory` chapter, when multiple
   independent exponential processes are racing, the time to the first event
   is exponential with rate equal to the **sum** of the individual rates. Here,
   the coalescence rate is :math:`\binom{k}{2}` and the recombination rate is
   :math:`k\rho/2`, so the next event -- whichever type it is -- occurs after
   an exponential waiting time with rate :math:`\binom{k}{2} + k\rho/2`.

   The probability that the event is a coalescence (rather than a
   recombination) is proportional to its rate:

   .. math::

      P(\text{coalescence}) = \frac{\binom{k}{2}}{\binom{k}{2} + k\rho/2}

   This is a general property of competing Poisson processes: the probability
   that a particular process "wins" is its rate divided by the total rate.

Let's simulate this process. The code below builds an ARG by tracing lineages
backwards, allowing both coalescence and recombination events.

.. code-block:: python

   import numpy as np

   def simulate_arg(n, rho, seq_length=1.0):
       """Simulate an Ancestral Recombination Graph.

       Parameters
       ----------
       n : int
           Number of samples.
       rho : float
           Population-scaled recombination rate (4*Ne*r) for the whole region.
       seq_length : float
           Length of the genomic region.

       Returns
       -------
       coal_events : list of (time, child1, child2, parent)
       recomb_events : list of (time, lineage, breakpoint, left_lineage, right_lineage)
       """
       # Each lineage carries an interval [left, right) of ancestral material.
       # This tracks which segment of the genome each lineage is responsible for.
       next_label = n
       # dict: lineage_label -> list of (left, right) intervals.
       # Initially, each of the n samples covers the full sequence.
       lineages = {}
       for i in range(n):
           lineages[i] = [(0.0, seq_length)]  # full sequence

       coal_events = []
       recomb_events = []
       current_time = 0.0

       # Continue until only one lineage remains (the MRCA for the whole genome)
       while len(lineages) > 1:
           k = len(lineages)
           coal_rate = k * (k - 1) / 2      # binom(k, 2)
           recomb_rate = k * rho / 2          # each lineage recombines at rate rho/2
           total_rate = coal_rate + recomb_rate

           # Sample waiting time from Exp(total_rate).
           # np.random.exponential takes the *scale* (= 1/rate), not the rate.
           wait = np.random.exponential(1.0 / total_rate)
           current_time += wait

           # Decide which event occurs: coalescence or recombination.
           # np.random.random() draws a uniform random number in [0, 1).
           if np.random.random() < coal_rate / total_rate:
               # --- Coalescence event ---
               # Convert dict keys to a list so we can index into them.
               labels = list(lineages.keys())
               # Pick two distinct lineages at random.
               # np.random.choice(n, size=2, replace=False) picks 2 distinct
               # indices from {0, 1, ..., n-1}.
               i, j = np.random.choice(len(labels), size=2, replace=False)
               c1, c2 = labels[i], labels[j]
               parent = next_label
               next_label += 1

               # Merge ancestral material: the parent inherits all intervals
               # from both children. The '+' operator concatenates two lists.
               merged = lineages[c1] + lineages[c2]
               lineages[parent] = merged
               # 'del' removes a key from the dictionary, since these lineages
               # no longer exist as separate entities after merging.
               del lineages[c1]
               del lineages[c2]

               coal_events.append((current_time, c1, c2, parent))
           else:
               # --- Recombination event ---
               labels = list(lineages.keys())
               # np.random.randint(0, k) picks a random integer in {0, ..., k-1}
               idx = np.random.randint(0, k)
               lineage = labels[idx]

               # Choose a breakpoint uniformly on [0, seq_length]
               breakpoint = np.random.uniform(0, seq_length)

               # Split ancestral material at the breakpoint.
               # These are list comprehensions: concise ways to build a new list
               # by filtering and transforming elements of an existing list.
               #
               # left_material: keep intervals (or parts of intervals) that fall
               # to the LEFT of the breakpoint.
               left_material = [(l, min(r, breakpoint))
                                for l, r in lineages[lineage] if l < breakpoint]
               # right_material: keep intervals (or parts of intervals) that fall
               # to the RIGHT of the breakpoint.
               right_material = [(max(l, breakpoint), r)
                                 for l, r in lineages[lineage] if r > breakpoint]

               # Only create new lineages if both sides have material.
               # (If the breakpoint falls outside all intervals, one side is empty
               # and the split has no effect.)
               if left_material and right_material:
                   left_label = next_label
                   right_label = next_label + 1
                   next_label += 2

                   lineages[left_label] = left_material
                   lineages[right_label] = right_material
                   del lineages[lineage]

                   recomb_events.append((current_time, lineage, breakpoint,
                                        left_label, right_label))

       return coal_events, recomb_events

   np.random.seed(42)
   coal_events, recomb_events = simulate_arg(n=5, rho=5.0)
   print(f"Coalescence events: {len(coal_events)}")
   print(f"Recombination events: {len(recomb_events)}")
   for t, c1, c2, p in coal_events:
       print(f"  Coal at t={t:.4f}: {c1} + {c2} -> {p}")
   for t, lin, bp, l, r in recomb_events:
       print(f"  Recomb at t={t:.4f}: {lin} splits at {bp:.4f} -> {l}, {r}")

Notice how recombination events increase the number of active lineages. With
:math:`\rho = 5`, recombination is fairly frequent -- you should see several
recombination events interspersed with the coalescences. If you increase
:math:`\rho`, recombination events become more common relative to coalescences,
and the ARG becomes more complex. If you set :math:`\rho = 0`, no
recombination occurs and you recover the simple coalescent tree from the
previous chapter.

The Structure of an ARG: A Directed Acyclic Graph
====================================================

A coalescent tree is, mathematically, a **tree**: each node has exactly one
parent (except the root, which has none). But an ARG is something more complex.
Recombination events create nodes with **two parents** -- one for the left side
of the breakpoint, one for the right side. This means the ARG is not a tree but
a **directed acyclic graph (DAG)**.

.. admonition:: What is a directed acyclic graph?

   A **directed graph** is a collection of nodes connected by arrows (directed
   edges). "Directed" means each edge has a direction: from child to parent,
   in our case. "Acyclic" means there are no cycles -- you can never follow
   arrows and return to where you started. In the ARG, edges always point
   backwards in time (from descendant to ancestor), and time only flows one
   way, so cycles are impossible.

   A tree is a special case of a DAG where every node has at most one parent.
   In an ARG, recombination nodes have two parents (one for each side of the
   breakpoint), making it a DAG but not a tree. Coalescence nodes still have
   one parent but two children, just as in a tree.

In the watchmaking metaphor, a tree is like a simple gear train -- force flows
in one direction from mainspring to escapement with no branching. An ARG is
like the full movement, where a wheel might receive force from two different
sources (one driving the hours, another driving the minutes), and where the
same barrel might power multiple complication trains simultaneously.

Marginal Trees
===============

An ARG encodes a **marginal tree** at every position along the genome. The
marginal tree at position :math:`x` is the genealogical tree for that specific
position, constructed by taking the ARG and ignoring all recombination events
that don't affect position :math:`x`.

Here's the key intuition: as you slide a pointer from left to right along the
genome, the genealogical tree changes. At each recombination breakpoint, some
branch in the tree detaches and reattaches elsewhere (or a new branch appears,
or one disappears). Between breakpoints, the tree is constant.

This is like examining different cross-sections of a watch movement. At one
cross-section, you see a particular arrangement of gears. Slide to a different
cross-section, and some gears have been swapped -- a different train is visible.
But between the swap points, the arrangement stays the same.

.. code-block:: python

   def extract_marginal_trees(coal_events, recomb_events, seq_length):
       """Extract the breakpoints where marginal trees change.

       Returns the breakpoints that partition the genome into segments
       with constant tree topology.
       """
       # Build a sorted list of unique breakpoints.
       # set() removes duplicates, sorted() puts them in order.
       # The list comprehension [bp for _, _, bp, _, _ in recomb_events]
       # extracts the third element (the breakpoint position) from each
       # recombination event tuple, ignoring the other fields (marked _).
       breakpoints = sorted(set([0.0, seq_length] +
                                [bp for _, _, bp, _, _ in recomb_events]))
       print(f"Number of distinct trees: {len(breakpoints) - 1}")
       print(f"Breakpoints: {[f'{b:.4f}' for b in breakpoints]}")
       return breakpoints

   breakpoints = extract_marginal_trees(coal_events, recomb_events, 1.0)

The number of distinct marginal trees equals the number of recombination events
plus one (the original tree, plus one new tree for each recombination). For our
simulation with :math:`\rho = 5`, you should see several distinct trees.

The key property that connects all of this:

.. math::

   \text{ARG} = \text{sequence of marginal trees} + \text{recombination events connecting them}

This is the data structure that SINGER infers from sequence data. The marginal
trees tell us the genealogy at each position; the recombination events tell us
how the genealogy changes as we move along the genome.

The Tree Sequence Representation
=================================

Storing an explicit tree for every base pair along the genome would be enormously
wasteful -- a human chromosome has hundreds of millions of base pairs but
typically only thousands to tens of thousands of distinct marginal trees. Modern
tools represent ARGs as **tree sequences** (Kelleher et al., 2016): an ordered
list of marginal trees along the genome, stored efficiently by recording only
the **changes** (which branches are removed and added) at each recombination
breakpoint.

This is a form of **differential encoding** -- rather than writing out each
tree in full, we store the first tree completely and then, at each breakpoint,
record only what changed. This is analogous to how a watchmaker's technical
manual might describe the base caliber in full, then describe each complication
variant by listing only the modified components.

The fundamental data structure has two tables:

- **Nodes**: Each node has a time (0 for samples, positive for ancestors) and a
  flag indicating whether it's a sample
- **Edges**: Each edge has a genomic interval :math:`[\text{left}, \text{right})`
  and connects a child to a parent. The edge is "active" only within its
  genomic interval.

.. code-block:: python

   class SimpleTreeSequence:
       """A minimal tree sequence representation for educational purposes.

       Nodes are stored as (time, is_sample) tuples.
       Edges are stored as (left, right, parent, child) tuples, where
       [left, right) is the genomic interval where this parent-child
       relationship holds.
       """

       def __init__(self):
           self.nodes = []   # list of (time, is_sample) tuples
           self.edges = []   # list of (left, right, parent, child) tuples

       def add_node(self, time, is_sample=False):
           """Add a node and return its integer ID (0-indexed)."""
           node_id = len(self.nodes)
           self.nodes.append((time, is_sample))
           return node_id

       def add_edge(self, left, right, parent, child):
           """Add an edge active over the genomic interval [left, right)."""
           self.edges.append((left, right, parent, child))

       def trees(self, seq_length):
           """Iterate over marginal trees.

           Yields (left, right, active_edges) for each genomic interval
           where the tree topology is constant.
           """
           # Collect all breakpoints from edge boundaries.
           # The set comprehension gathers every left and right coordinate,
           # plus the sequence endpoints, then sorts them.
           breakpoints = sorted(set(
               [0.0, seq_length] +
               [l for l, r, p, c in self.edges] +
               [r for l, r, p, c in self.edges]
           ))

           for i in range(len(breakpoints) - 1):
               # Check which edges are active at the midpoint of this interval.
               # Using the midpoint avoids boundary ambiguity.
               pos = (breakpoints[i] + breakpoints[i+1]) / 2
               # List comprehension: keep only edges whose interval contains pos.
               active = [(p, c) for l, r, p, c in self.edges
                         if l <= pos < r]
               yield breakpoints[i], breakpoints[i+1], active

   # Example: two marginal trees for 4 samples
   ts = SimpleTreeSequence()
   # Samples at time 0 (the present)
   # The underscore _ is a Python convention for a variable we don't need --
   # here we don't use the loop variable because add_node tracks IDs internally.
   for _ in range(4):
       ts.add_node(0.0, is_sample=True)
   # Internal (ancestor) nodes at various times in the past
   ts.add_node(0.5)  # node 4
   ts.add_node(0.8)  # node 5
   ts.add_node(1.2)  # node 6

   # First tree (positions 0.0 to 0.6): topology ((0,1), (2,3))
   # Nodes 0 and 1 coalesce into node 4; nodes 2 and 3 into node 5;
   # then nodes 4 and 5 into node 6.
   ts.add_edge(0.0, 0.6, 4, 0)   # 0 -> 4 for positions [0.0, 0.6)
   ts.add_edge(0.0, 1.0, 4, 1)   # 1 -> 4 for positions [0.0, 1.0)
   ts.add_edge(0.0, 1.0, 5, 2)   # 2 -> 5 for positions [0.0, 1.0)
   ts.add_edge(0.0, 1.0, 5, 3)   # 3 -> 5 for positions [0.0, 1.0)
   ts.add_edge(0.0, 0.6, 6, 4)   # 4 -> 6 for positions [0.0, 0.6)
   ts.add_edge(0.0, 1.0, 6, 5)   # 5 -> 6 for positions [0.0, 1.0)

   # After recombination at position 0.6, node 0 moves:
   # Instead of joining node 4 (with node 1), node 0 now joins node 5
   # (with nodes 2 and 3). This represents a change in genealogy.
   ts.add_edge(0.6, 1.0, 5, 0)   # 0 -> 5 for positions [0.6, 1.0)

   for left, right, edges in ts.trees(1.0):
       print(f"\nTree at [{left:.1f}, {right:.1f}):")
       for p, c in edges:
           print(f"  {c} -> {p}")

Examine the output: in the first tree (positions 0.0 to 0.6), nodes 0 and 1 are
siblings (both children of node 4). In the second tree (positions 0.6 to 1.0),
node 0 has moved to become a child of node 5, joining nodes 2 and 3. Only one
edge changed -- this is the efficiency of the tree sequence representation.

Branch Lengths and the ARG
===========================

A crucial concept for SINGER: the **total branch length** of a marginal tree.
This quantity determines how many mutations we expect to see, and it connects
the genealogy (which is hidden) to the sequence data (which we observe).

For a marginal tree :math:`\Psi` at position :math:`x`, the total branch length
is the sum of all branch lengths in the tree:

.. math::

   L(\Psi) = \sum_{\text{branch } b \in \Psi} \ell(b)

where :math:`\ell(b)` is the length (in coalescent time units) of branch
:math:`b`. Recall from the :ref:`coalescent_theory` chapter that branch lengths
are measured in coalescent time units (multiples of :math:`2N_e` generations).

The total branch length of the entire ARG across the genome is obtained by
**integrating** the branch length of each marginal tree over the positions it
covers:

.. math::

   L(\mathcal{G}) = \int_0^{S} L(\Psi_x) \, dx

where :math:`S` is the sequence length and :math:`\Psi_x` is the marginal tree
at position :math:`x`.

.. admonition:: Calculus aside: why integrate?

   Each base pair has its own marginal tree, and mutations at each base pair
   arise on the branches of that specific tree. So the total "opportunity" for
   mutations is the sum of all branch lengths across all positions. Since the
   marginal tree is constant within each genomic interval between breakpoints,
   the integral simplifies to a sum:

   .. math::

      L(\mathcal{G}) = \sum_{i=1}^{T} (\text{right}_i - \text{left}_i) \times L(\Psi_i)

   where the sum runs over the :math:`T` distinct marginal trees and
   :math:`[\text{left}_i, \text{right}_i)` is the genomic interval where tree
   :math:`\Psi_i` applies. Each tree's branch length is weighted by how many
   base pairs it covers (its "span").

This total branch length determines the expected number of mutations across the
entire ARG:

.. math::

   \mathbb{E}[\text{number of mutations}] = \frac{\theta}{2} L(\mathcal{G})

.. admonition:: Probability aside: why :math:`\theta/2 \times L`?

   This follows from the Poisson mutation model we established in the
   :ref:`coalescent_theory` chapter. Each unit of branch length (in coalescent
   time) at each base pair independently produces mutations at rate
   :math:`\theta/2`. Since the Poisson distribution has the property that the
   mean of a sum of independent Poissons equals the sum of the means, the
   expected total number of mutations is simply :math:`\theta/2` times the
   total branch-length-times-span across the entire ARG. This is a
   consequence of the **additivity property** of Poisson processes.

.. code-block:: python

   def total_branch_length(node_times, edges, position):
       """Compute total branch length of the marginal tree at a given position.

       For each edge active at 'position', the branch length is the difference
       between the parent's time and the child's time.

       Parameters
       ----------
       node_times : dict
           Mapping from node ID to time (in coalescent units).
       edges : list of (left, right, parent, child)
           Tree sequence edges.
       position : float
           Genomic position to query.

       Returns
       -------
       float
           Total branch length at the given position.
       """
       total = 0.0
       for left, right, parent, child in edges:
           # Check if this edge is active at the query position
           if left <= position < right:
               total += node_times[parent] - node_times[child]
       return total

Why ARG Inference Is Hard
==========================

We've now seen what an ARG is and how to represent one. The challenge that
motivates the rest of this book is the **inverse problem**: given observed DNA
sequences, can we reconstruct the ARG that produced them?

This turns out to be extraordinarily difficult, for several reinforcing reasons:

1. **Enormous state space.** The number of possible ARGs grows
   super-exponentially with sample size. Even for a handful of samples and a
   short genomic region, the space of possible histories is vast beyond
   enumeration. It is as if a watchmaker had to deduce the complete history of a
   movement's assembly -- every gear placed, every screw turned, in exact order
   -- by examining only the finished watch.

2. **Recombination creates reticulations.** Unlike trees, ARGs are directed
   acyclic graphs with **reticulation nodes** -- nodes that have two parents
   (one from each side of a recombination breakpoint). This makes the
   combinatorial structure far richer than for trees, where each node has
   exactly one parent.

3. **Data is sparse.** We only observe the *leaves* of the ARG (the sampled
   sequences at the present), and we only see positions where mutations happened
   to fall. The vast majority of the tree's branches carry no mutations at all
   (recall from the coalescent chapter that mutation probabilities per site are
   tiny: :math:`\theta\ell/2 \approx 10^{-4}` for typical human parameters).
   Reconstructing a complex structure from such sparse observations is inherently
   underdetermined.

4. **Identifiability.** Many different ARGs can produce the same pattern of
   mutations. Two different genealogical histories might yield identical sequence
   data, making it impossible to distinguish them even in principle.

.. admonition:: Probability aside: the curse of dimensionality

   The space of possible ARGs is a high-dimensional combinatorial object. For
   :math:`n` samples and a genome of length :math:`S` with recombination rate
   :math:`\rho`, the expected number of recombination events is
   :math:`O(n \rho S)`, and each one creates a branching point in the space of
   possible histories. Exhaustive enumeration is out of the question. This is
   why Bayesian methods that *sample* from the posterior distribution of ARGs
   -- rather than searching for a single best ARG -- are essential.

This is why methods like SINGER exist. Rather than attempting to search the
full space of ARGs, SINGER uses two powerful mathematical tools to make
inference tractable:

- The **Sequentially Markov Coalescent (SMC)**, which approximates the ARG as a
  Markov chain of marginal trees along the genome (covered in :ref:`smc`)
- **Hidden Markov Models (HMMs)**, which provide efficient algorithms for
  inference in Markov chains (covered in :ref:`hmms`)

Together, these tools reduce an impossible combinatorial problem to a sequence
of manageable probabilistic calculations.

Summary
=======

We've moved from the single-tree world of the coalescent to the richer landscape
of ARGs -- from tracing one gear train to reading the complete mechanical
drawing. Here is what we've established:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Concept
     - Key Idea
   * - Recombination
     - Different genome positions can have different trees (shuffling during meiosis)
   * - ARG
     - Complete genealogical history: all trees + recombinations (a directed acyclic graph)
   * - Marginal tree
     - The genealogy at a single genome position; constant between breakpoints
   * - Breakpoint
     - A genomic position where recombination changes the marginal tree
   * - Tree sequence
     - Efficient storage: record only the changes between adjacent marginal trees
   * - Total branch length
     - Determines expected mutation count: :math:`\mathbb{E}[\text{muts}] = \frac{\theta}{2}L(\mathcal{G})`
   * - Event rates
     - Coalescence at :math:`\binom{k}{2}`, recombination at :math:`k\rho/2`

The ARG is the object we ultimately want to infer -- it is the master blueprint
of the genome's history. But as we've seen, direct inference is intractable.
The next two chapters introduce the mathematical machinery that makes it
possible: Hidden Markov Models and the Sequentially Markov Coalescent.

Next: :ref:`hmms` -- the computational engine that makes ARG inference tractable.
