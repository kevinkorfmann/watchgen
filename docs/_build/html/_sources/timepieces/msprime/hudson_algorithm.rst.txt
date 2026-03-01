.. _hudson_algorithm:

====================
Hudson's Algorithm
====================

   *The mainspring: the event-driven loop that brings the whole mechanism to life.*

Hudson's algorithm (Hudson 1983, 2002; Kelleher et al. 2016) is the main
simulation loop of msprime -- the ticking of the clock. Everything we have
built so far comes together here: the exponential race from
:ref:`coalescent_process`, the segment chains and Fenwick tree from
:ref:`segments_fenwick`, and the event-driven philosophy described in the
:ref:`msprime_overview`.

If the coalescent process is the escapement (setting the rhythm), and the
segments and Fenwick tree are the gear train (transmitting that rhythm
efficiently), then Hudson's algorithm is the mainspring -- the driving force
that makes the whole mechanism tick. It orchestrates the exponential race
between coalescence, recombination, migration, and demographic events --
always executing whichever happens first, updating the state, and repeating
until every genomic position has found its MRCA.

.. note::

   **Prerequisites.** This chapter assumes familiarity with:

   - The **exponential race** (Step 3 of :ref:`coalescent_process`): how
     competing exponential waiting times determine which event fires next.
   - **Segment chains** and the **Fenwick tree** (:ref:`segments_fenwick`):
     the data structures for tracking ancestral material and selecting
     breakpoints.
   - The concept of **event-driven simulation** (explained in the
     :ref:`msprime_overview`): jumping directly from event to event rather
     than stepping through time uniformly.


Step 1: The Main Loop Structure
=================================

The algorithm is conceptually simple. At each iteration:

1. Compute the rate of every possible event
2. Draw independent exponential waiting times
3. Check if a demographic event (population size change, etc.) occurs first
4. Execute the earliest event
5. Update data structures
6. Repeat

This is the ticking of the clock: each tick is one event, and between ticks,
nothing happens. The clock never idles -- it always jumps directly to the
next meaningful moment.

.. admonition:: Closing a confusion gap -- Event-driven vs. time-stepped simulation

   In a **time-stepped** simulation (like the Wright-Fisher model), you
   advance the clock by one fixed unit (one generation) and process
   everything that happens in that unit. In an **event-driven** simulation,
   you ask "when does the *next* event happen?" and jump directly there.
   Event-driven simulation is vastly more efficient when events are rare
   relative to the time scale -- which they are in the coalescent, where
   :math:`O(k^2)` lineages might go thousands of generations between events.
   The expected waiting time between events is :math:`O(1/k^2)` in
   coalescent units, so jumping directly saves :math:`O(k^2)` wasted
   iterations per event.

.. code-block:: python

   import numpy as np
   import math

   INFINITY = float('inf')

   def hudson_simulate(state, end_time=INFINITY):
       """The main Hudson simulation loop.

       This is the heart of msprime -- the ticking of the clock.
       Each iteration: compute rates, race exponentials, execute the winner.

       Parameters
       ----------
       state : SimulationState
           Contains lineages, Fenwick trees, rate maps, populations, etc.

       Returns
       -------
       state : SimulationState
           Final state with completed tree sequence tables.
       """
       while not state.is_completed():
           if state.t >= end_time:
               break

           # === Step 1: Compute event rates ===
           # Each event type proposes an exponential waiting time.

           # Recombination: rate comes from the Fenwick tree -- O(log n)
           re_rate = state.get_total_recombination_rate()
           t_re = np.random.exponential(1.0 / re_rate) if re_rate > 0 else INFINITY

           # Gene conversion (within segments): also from a Fenwick tree
           gc_rate = state.get_total_gc_rate()
           t_gc = np.random.exponential(1.0 / gc_rate) if gc_rate > 0 else INFINITY

           # Gene conversion (left of first segment): proportional to lineage count
           gc_left_rate = state.get_total_gc_left_rate()
           t_gc_left = (np.random.exponential(1.0 / gc_left_rate)
                        if gc_left_rate > 0 else INFINITY)

           # Coalescence: one exponential per population
           # (each population has its own rate based on lineage count and size)
           t_ca = INFINITY
           ca_pop = -1
           for pop in state.populations:
               if pop.num_ancestors > 1:
                   t = pop.get_coalescence_waiting_time(state.t)
                   if t < t_ca:
                       t_ca = t
                       ca_pop = pop.id

           # Migration: one exponential per (source, dest) pair
           t_mig = INFINITY
           mig_source, mig_dest = -1, -1
           for j, pop in enumerate(state.populations):
               if pop.num_ancestors == 0:
                   continue
               for k in range(len(state.populations)):
                   if j == k:
                       continue
                   rate = pop.num_ancestors * state.migration_matrix[j][k]
                   if rate > 0:
                       t = np.random.exponential(1.0 / rate)
                       if t < t_mig:
                           t_mig = t
                           mig_source, mig_dest = j, k

           # === Step 2: Find the minimum (the exponential race winner) ===
           min_time = min(t_re, t_ca, t_gc, t_gc_left, t_mig)

           # === Step 3: Check demographic events ===
           # Demographic events (size changes, splits) are deterministic
           # in time, not random. If one falls before the next random event,
           # it fires first.
           if state.t + min_time > state.next_demographic_event_time():
               state.execute_demographic_event()
               continue  # re-enter the loop with updated parameters

           # === Step 4: Advance time and execute the winning event ===
           state.t += min_time

           if min_time == t_re:
               state.recombination_event()
           elif min_time == t_gc:
               state.gene_conversion_within_event()
           elif min_time == t_gc_left:
               state.gene_conversion_left_event()
           elif min_time == t_ca:
               state.coalescence_event(ca_pop)
           elif min_time == t_mig:
               state.migration_event(mig_source, mig_dest)

       return state

Now let us examine each event handler in detail, starting with recombination.


Step 2: The Recombination Event in Detail
============================================

When recombination wins the race, we split one lineage into two. The full
procedure:

1. **Choose the breakpoint** using the Fenwick tree (as derived in
   :ref:`segments_fenwick`).

2. **Find the segment** containing the breakpoint.

3. **Split the segment chain** into left and right parts.

4. **Update the Fenwick tree** with new mass values for both parts.

5. **Create a new lineage** for the right part.

There are two sub-cases depending on where the breakpoint falls:

.. code-block:: text

   Case 1: Breakpoint falls WITHIN a segment

     Before:   ... x ======|===== y ...
                          bp

     After:    ... x ====== y          (left lineage)
                           a ===== ... (right lineage)

   Case 2: Breakpoint falls in a GAP between segments

     Before:   ... x =====     y ========= ...
                         |
                        bp

     After:    ... x =====                 (left lineage)
                         y ========= ...   (right lineage)

.. admonition:: Closing a confusion gap -- Why two cases?

   After multiple recombination events, a lineage's segment chain may have
   gaps -- intervals where ancestry has been transferred to another lineage.
   A recombination breakpoint can land either *within* a segment (splitting
   it) or *in a gap* (simply detaching the right portion of the chain).
   Both cases reduce to pointer rewiring, but the bookkeeping differs
   slightly: in Case 1, we must create a new segment for the right half;
   in Case 2, we just disconnect existing segments.

.. code-block:: python

   def recombination_event(state, label=0):
       """Execute a recombination event.

       This is a detailed implementation following msprime's algorithms.py.
       """
       state.num_re_events += 1

       # Step 1: Choose breakpoint using the Fenwick tree -- O(log n)
       y, bp = choose_breakpoint(
           state.recomb_mass_index[label],
           state.recomb_map
       )
       left_lineage = y.lineage
       x = y.prev  # segment to the left of y (may be None if y is head)

       if y.left < bp:
           # Case 1: breakpoint falls WITHIN segment y
           #   x         y
           # =====  ===|====  ...
           #          bp
           #
           # Split y into y=[left, bp) and alpha=[bp, right)
           alpha = state.pool.copy(y)      # create new segment for right half
           alpha.left = bp                 # alpha starts at the breakpoint
           alpha.prev = None               # alpha is the head of the new chain

           if y.next is not None:
               y.next.prev = alpha  # alpha inherits y's successors
           y.next = None            # y is now the tail of the left chain
           y.right = bp             # trim y to end at breakpoint

           # Update Fenwick tree for the trimmed y -- O(log n)
           state.set_segment_mass(y)
           left_lineage.tail = y

       else:
           # Case 2: breakpoint falls in a GAP to the left of y
           #   x            y
           # =====  |   =========  ...
           #       bp
           #
           # Just detach y from x -- pure pointer rewiring, O(1)
           x.next = None
           y.prev = None
           alpha = y
           left_lineage.tail = x

       # Step 2: Create new lineage for the right part
       right_lineage = Lineage(head=alpha, tail=find_tail(alpha),
                                population=left_lineage.population,
                                label=label)
       alpha.lineage = right_lineage
       state.set_segment_mass(alpha)  # register mass in Fenwick tree
       state.add_lineage(right_lineage)

       return left_lineage, right_lineage

   def find_tail(seg):
       """Follow the chain to find the tail."""
       while seg.next is not None:
           seg = seg.next
       return seg

The recombination event *increases* the lineage count by one. The coalescence
event, which we examine next, *decreases* it by one. The interplay between
these two forces drives the entire simulation.


Step 3: The Coalescence Event in Detail
==========================================

The coalescence event is the most complex operation. Two randomly chosen
lineages merge their ancestral material, and at positions where they overlap,
a new common ancestor node is created.

**The core difficulty**: when two segment chains overlap, we need to:

- Create a new ancestor node at overlapping positions
- Record edges in the tree sequence tables
- Update the overlap counter :math:`S`
- Handle non-overlapping regions (they just pass through)
- Handle partial overlaps (segment boundaries don't align)

The merge algorithm walks through both chains simultaneously, like the merge
step of merge sort, but with genealogical bookkeeping.

.. code-block:: text

   Lineage x:  [0, 300)      [500, 800)         [900, 1000)
   Lineage y:       [200, 600)              [850, 950)

   Merge result:
   [0, 200):     only x  ->  passes through from x
   [200, 300):   overlap ->  COALESCENCE! new ancestor u
   [300, 500):   only y  ->  passes through from y
   [500, 600):   overlap ->  COALESCENCE! (same ancestor u)
   [600, 800):   only x  ->  passes through from x
   [850, 900):   only y  ->  passes through from y
   [900, 950):   overlap ->  COALESCENCE! (same ancestor u)
   [950, 1000):  only x  ->  passes through from x

.. admonition:: Closing a confusion gap -- The merge walk

   The merge algorithm is often the hardest part of the simulator to
   understand. Think of it this way: lay both segment chains side by side on
   the genome number line. Walk from left to right. At each position, one
   of three things is true:

   1. **Only one chain has material here** -- that material passes through
      unchanged to the merged lineage.
   2. **Both chains have material here** -- this is a *coalescence*: we
      create a new ancestor node, record edges linking both children to it,
      and decrement the overlap counter.
   3. **Neither chain has material here** -- nothing happens; skip ahead.

   The walk advances by jumping to the next boundary (left or right endpoint
   of any segment in either chain). At each boundary, the situation may
   change from case 1 to case 2 or vice versa.

.. code-block:: python

   def merge_two_ancestors(state, pop_index, label, x, y):
       """Merge two lineages' segment chains.

       This is the heart of the coalescence operation. It walks through
       both chains simultaneously, handling overlaps and pass-throughs.

       Parameters
       ----------
       state : SimulationState
       pop_index : int
           Population where the coalescence occurs.
       x, y : Segment
           Head segments of the two lineages to merge.

       Returns
       -------
       new_lineage : Lineage
           The merged lineage (may be None if fully coalesced).
       """
       state.num_ca_events += 1
       new_lineage = Lineage(head=None, tail=None, population=pop_index,
                              label=label)
       coalescence = False
       u = -1  # tree-sequence node for the common ancestor

       while x is not None or y is not None:
           alpha = None  # segment to add to the merged chain

           if x is None or y is None:
               # === One chain exhausted: absorb the rest of the other ===
               if x is not None:
                   alpha = x
                   x = None  # mark as consumed
               if y is not None:
                   alpha = y
                   y = None

           else:
               # === Both chains still active ===
               # Ensure x starts at or before y (simplifies case analysis)
               if y.left < x.left:
                   x, y = y, x

               if x.right <= y.left:
                   # NO OVERLAP: x ends before y starts
                   #   x          y
                   # =====    =========
                   alpha = x
                   x = x.next
                   alpha.next = None

               elif x.left != y.left:
                   # PARTIAL OVERLAP: x starts before y
                   #   x
                   # ====|=====
                   #     y
                   #     =========
                   # Trim the non-overlapping prefix of x
                   alpha = state.pool.alloc(
                       left=x.left, right=y.left, node=x.node)
                   x.left = y.left  # advance x to where y starts

               else:
                   # === COALESCENCE: x and y start at the same position ===
                   if not coalescence:
                       coalescence = True
                       u = state.store_node(pop_index)  # create ancestor node

                   left = x.left
                   right = min(x.right, y.right)

                   # Update overlap counter: one fewer lineage covers [left, right)
                   state.decrement_overlap(left, right)

                   # Record edges: both x and y are children of u
                   state.store_edge(left, right, parent=u, child=x.node)
                   state.store_edge(left, right, parent=u, child=y.node)

                   # Create the coalesced segment (under ancestor u)
                   alpha = state.pool.alloc(left=left, right=right, node=u)

                   # Trim x and y: consume the overlapping portion
                   if x.right == right:
                       state.pool.free(x)  # x is fully consumed
                       x = x.next
                   else:
                       x.left = right  # x has leftover material

                   if y.right == right:
                       state.pool.free(y)  # y is fully consumed
                       y = y.next
                   else:
                       y.left = right  # y has leftover material

           # === Append alpha to the merged chain ===
           if alpha is not None:
               alpha.lineage = new_lineage
               alpha.prev = new_lineage.tail
               state.set_segment_mass(alpha)  # register in Fenwick tree

               if new_lineage.head is None:
                   new_lineage.head = alpha
               else:
                   new_lineage.tail.next = alpha
               new_lineage.tail = alpha

       # If the merged lineage has remaining segments, add it back
       if new_lineage.head is not None:
           state.defrag_segment_chain(new_lineage)
           state.add_lineage(new_lineage)

       return new_lineage

After the merge, the resulting chain may contain adjacent segments with the
same node ID -- a cosmetic issue that the next step addresses.


Step 4: Segment Chain Defragmentation
========================================

After a coalescence, adjacent segments in the merged chain might have the
same node ID. This happens when two non-overlapping segments with the same
ancestry end up next to each other. We merge them to keep the chains clean:

.. code-block:: python

   def defrag_segment_chain(lineage):
       """Merge adjacent segments with the same node ID.

       Before:  [0, 300: node 5] -> [300, 800: node 5] -> [800, 1000: node 3]

       After:   [0, 800: node 5] -> [800, 1000: node 3]

       This reduces the number of segments (and Fenwick tree entries),
       keeping the data structures lean.
       """
       seg = lineage.head
       while seg is not None and seg.next is not None:
           if seg.node == seg.next.node and seg.right == seg.next.left:
               # Merge: extend seg to cover seg.next
               to_free = seg.next
               seg.right = to_free.right
               seg.next = to_free.next
               if seg.next is not None:
                   seg.next.prev = seg
               # Free the absorbed segment
               # (in practice: return to pool and update Fenwick tree)
           else:
               seg = seg.next

       # Update tail
       lineage.tail = seg


Step 5: The Migration Event
==============================

Migration is the simplest event. A randomly chosen lineage moves from one
population to another. In the backward-time view, this means the lineage's
ancestor lived in a different population one generation earlier.

.. code-block:: python

   def migration_event(state, source, dest):
       """Move a random lineage from source to dest population.

       This is the simplest event type: no segment manipulation needed,
       just update the lineage's population assignment.
       """
       # Choose a random lineage from the source population
       source_pop = state.populations[source]
       idx = np.random.randint(source_pop.num_ancestors)
       lineage = source_pop.remove(idx)

       # Add to destination
       lineage.population = dest
       state.populations[dest].add(lineage)

With all event handlers defined, let us assemble the complete algorithm.


Step 6: The Complete Algorithm
================================

Let's put it all together into a minimal but complete implementation. This
is the master clockmaker's bench in miniature: a working simulation that
uses all the data structures from :ref:`segments_fenwick` and all the
mathematics from :ref:`coalescent_process`.

.. code-block:: python

   class MinimalSimulator:
       """A minimal but complete Hudson's algorithm implementation.

       This implements the core simulation loop with coalescence and
       recombination (no gene conversion, no migration for simplicity).
       It is the ticking of the clock -- the main event loop.
       """

       def __init__(self, n, sequence_length, recombination_rate, pop_size=1.0):
           self.n = n
           self.L = sequence_length
           self.Ne = pop_size
           self.t = 0.0  # current simulation time (backwards from present)

           # Rate map (uniform recombination rate)
           self.recomb_rate = recombination_rate
           self.rho = 4 * pop_size * recombination_rate * sequence_length

           # Segments and Fenwick tree -- the gear train
           self.max_segs = 100 * n
           self.segments = [None] * (self.max_segs + 1)
           self.free_segs = list(range(self.max_segs, 0, -1))
           self.mass_index = FenwickTree(self.max_segs)  # the clever indexing mechanism

           # Overlap counter -- tracks when we're done
           self.S = {0: n, sequence_length: -1}

           # Lineages -- initially n lineages, each covering [0, L)
           self.lineages = []
           for i in range(n):
               seg_idx = self.free_segs.pop()
               seg = Segment(index=seg_idx, left=0, right=sequence_length,
                              node=i)
               self.segments[seg_idx] = seg
               lin = Lineage(head=seg, tail=seg, population=0)
               seg.lineage = lin
               self.lineages.append(lin)

               # Register segment mass in Fenwick tree
               mass = recombination_rate * sequence_length
               self.mass_index.set_value(seg_idx, mass)

           # Output tables: edges and nodes form the tree sequence
           self.edges = []  # (left, right, parent, child)
           self.nodes = []  # (time, population)
           for i in range(n):
               self.nodes.append((0.0, 0))  # sample nodes at time 0

           self.num_re_events = 0
           self.num_ca_events = 0

       def is_completed(self):
           """Check if all positions have found their MRCA."""
           for x, count in self.S.items():
               if count > 1:
                   return False
           return True

       def get_recomb_mass(self, seg):
           """Recombination mass of a segment.

           The left bound accounts for the gap subtlety: if this segment
           has a predecessor, the effective left bound is prev.right (not
           seg.left), because recombination in the gap has no effect.
           """
           left_bound = seg.prev.right if seg.prev is not None else seg.left
           return self.recomb_rate * (seg.right - left_bound)

       def simulate(self):
           """Run the simulation until completion.

           This is the main loop -- each iteration is one tick of the clock.
           """
           while not self.is_completed():
               k = len(self.lineages)

               # Recombination rate: from the Fenwick tree -- O(log n)
               re_total = self.mass_index.get_total()
               t_re = (np.random.exponential(1.0 / re_total)
                       if re_total > 0 else INFINITY)

               # Coalescence rate: k choose 2, scaled by population size
               coal_rate = k * (k - 1) / 2
               t_ca = INFINITY
               if coal_rate > 0:
                   # Draw from Exp(2 * binom(k,2)), then scale by Ne
                   t_ca = self.Ne * np.random.exponential(
                       1.0 / (2 * coal_rate))

               # === The exponential race: smallest time wins ===
               min_t = min(t_re, t_ca)
               self.t += min_t

               if min_t == t_re:
                   self._recombination_event()
               else:
                   self._coalescence_event()

           return self.edges, self.nodes

       def _recombination_event(self):
           """Handle a recombination event: split one lineage into two."""
           self.num_re_events += 1
           # Choose breakpoint using Fenwick tree -- O(log n)
           random_mass = np.random.uniform(0, self.mass_index.get_total())
           seg_idx = self.mass_index.find(random_mass)
           y = self.segments[seg_idx]

           # Convert mass coordinate to genomic position
           cum_mass = self.mass_index.get_cumulative_sum(seg_idx)
           mass_from_right = cum_mass - random_mass
           bp = y.right - mass_from_right / self.recomb_rate

           if bp <= y.left or bp >= y.right:
               return  # degenerate breakpoint (at segment boundary)

           # Split the segment -- O(1) pointer rewiring
           new_idx = self.free_segs.pop()
           alpha = Segment(index=new_idx, left=bp, right=y.right,
                            node=y.node)
           self.segments[new_idx] = alpha
           alpha.next = y.next
           if y.next is not None:
               y.next.prev = alpha
           y.next = None
           y.right = bp

           # Update Fenwick tree -- O(log n)
           self.mass_index.set_value(y.index, self.get_recomb_mass(y))
           self.mass_index.set_value(alpha.index,
                                      self.recomb_rate * (alpha.right - alpha.left))

           # Create new lineage for the right part
           old_lin = y.lineage
           old_lin.tail = y
           new_lin = Lineage(head=alpha, tail=alpha, population=0)
           alpha.lineage = new_lin
           self.lineages.append(new_lin)

       def _coalescence_event(self):
           """Handle a coalescence event: merge two lineages into one."""
           self.num_ca_events += 1
           k = len(self.lineages)

           # Pick two random lineages (uniformly at random)
           i, j = sorted(np.random.choice(k, 2, replace=False))
           y_lin = self.lineages.pop(j)  # remove j first (higher index)
           x_lin = self.lineages.pop(i)  # then i

           # Create ancestor node in the tree sequence
           u = len(self.nodes)
           self.nodes.append((self.t, 0))

           # Walk through both chains -- the merge algorithm
           x, y = x_lin.head, y_lin.head
           new_lin = Lineage(head=None, tail=None, population=0)
           coalescence = False

           while x is not None or y is not None:
               alpha = None

               if x is None:
                   # x exhausted: absorb rest of y
                   alpha = y
                   y = None
               elif y is None:
                   # y exhausted: absorb rest of x
                   alpha = x
                   x = None
               else:
                   # Both active: ensure x starts at or before y
                   if y.left < x.left:
                       x, y = y, x

                   if x.right <= y.left:
                       # No overlap: x passes through
                       alpha = x
                       x = x.next
                       alpha.next = None
                   elif x.left < y.left:
                       # Partial overlap: trim x's prefix
                       new_idx = self.free_segs.pop()
                       alpha = Segment(new_idx, left=x.left, right=y.left,
                                        node=x.node)
                       self.segments[new_idx] = alpha
                       x.left = y.left  # advance x
                   else:
                       # Coalescence! x and y start at the same position
                       coalescence = True
                       left = x.left
                       right = min(x.right, y.right)

                       # Record edges: both are children of ancestor u
                       self.edges.append((left, right, u, x.node))
                       self.edges.append((left, right, u, y.node))

                       # Update overlap counter
                       self._decrement_overlap(left, right)

                       # Create coalesced segment under ancestor u
                       new_idx = self.free_segs.pop()
                       alpha = Segment(new_idx, left=left, right=right,
                                        node=u)
                       self.segments[new_idx] = alpha

                       # Consume overlapping portions of x and y
                       if x.right == right:
                           old_x = x
                           x = x.next
                       else:
                           x.left = right  # x has leftover
                       if y.right == right:
                           old_y = y
                           y = y.next
                       else:
                           y.left = right  # y has leftover

               # Append alpha to the merged chain
               if alpha is not None:
                   alpha.lineage = new_lin
                   alpha.prev = new_lin.tail
                   mass = self.recomb_rate * (alpha.right - alpha.left)
                   self.mass_index.set_value(alpha.index, mass)
                   if new_lin.head is None:
                       new_lin.head = alpha
                   else:
                       new_lin.tail.next = alpha
                   new_lin.tail = alpha

           # Add the merged lineage back (if it has remaining segments)
           if new_lin.head is not None:
               self.lineages.append(new_lin)

       def _decrement_overlap(self, left, right):
           """Decrement overlap counter in [left, right)."""
           # Simplified version (the full version uses an AVL tree)
           keys = sorted(self.S.keys())
           for k in keys:
               if k >= left and k < right:
                   self.S[k] -= 1

   # Run a small simulation
   np.random.seed(42)
   sim = MinimalSimulator(n=5, sequence_length=1000,
                           recombination_rate=0.001, pop_size=1.0)
   edges, nodes = sim.simulate()
   print(f"Simulation complete!")
   print(f"  Coalescence events: {sim.num_ca_events}")
   print(f"  Recombination events: {sim.num_re_events}")
   print(f"  Nodes: {len(nodes)} ({sim.n} samples + "
         f"{len(nodes) - sim.n} ancestors)")
   print(f"  Edges: {len(edges)}")
   print(f"  TMRCA: {sim.t:.4f} coalescent units")

You have now seen the complete algorithm -- every tick of the clock, from
rate computation through event execution. Let us analyze its computational
complexity.


Step 7: Complexity Analysis
==============================

Let's count the operations per simulation step:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Operation
     - Complexity
     - Why
   * - Compute recomb rate
     - :math:`O(\log n)`
     - ``FenwickTree.get_total()``
   * - Compute coal rate
     - :math:`O(1)`
     - Just :math:`k(k-1)/2`
   * - Draw exponentials
     - :math:`O(1)`
     - Random number generation
   * - Choose breakpoint
     - :math:`O(\log n)`
     - ``FenwickTree.find()``
   * - Split segment
     - :math:`O(1)`
     - Pointer rewiring
   * - Update Fenwick tree
     - :math:`O(\log n)`
     - ``FenwickTree.set_value()``
   * - Choose pair for coal.
     - :math:`O(1)`
     - Random selection
   * - Merge two chains
     - :math:`O(s)`
     - :math:`s` = segments in the two chains
   * - Update overlap counter
     - :math:`O(\log n)`
     - AVL tree operations

**Total per event**: :math:`O(s \log n)` where :math:`s` is the number of
segments in the merging chains. In practice, :math:`s` is small for most
events.

**Total simulation**: :math:`O(n + E \log n)` where :math:`E` is the total
number of events. The number of events is :math:`O(n + \rho)` where
:math:`\rho = 4N_e r L` is the population-scaled recombination rate. This
gives msprime its celebrated near-linear scaling in :math:`n`.

.. admonition:: Probability Aside -- Why :math:`O(n + \rho)` events?

   The expected number of coalescence events is :math:`n - 1` (each reduces
   the lineage count by 1, from :math:`n` to 1). The expected number of
   recombination events is :math:`O(\rho)` because the total recombination
   rate summed over the coalescent tree is proportional to :math:`\rho`.
   More precisely, the expected number of recombination events is
   :math:`\frac{\rho}{2} \cdot E[L_{\text{total}}]` where
   :math:`E[L_{\text{total}}] = 2\sum_{k=1}^{n-1} 1/k` is the expected
   total branch length. For large :math:`\rho`, this is the dominant term.


Step 8: Comparison with ms
==============================

The original ``ms`` program (Hudson 2002) uses the same algorithm but with
:math:`O(n)` data structures instead of :math:`O(\log n)` Fenwick trees. This
makes ``ms`` quadratic in the number of segments for large genomes.

.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Property
     - ms (Hudson 2002)
     - msprime (Kelleher 2016)
   * - Breakpoint selection
     - :math:`O(n)` linear scan
     - :math:`O(\log n)` Fenwick tree
   * - Rate maintenance
     - :math:`O(n)` recompute
     - :math:`O(\log n)` incremental update
   * - Output format
     - Text (variant matrix)
     - Tree sequence (tskit)
   * - Memory
     - :math:`O(n \cdot L)` for output
     - :math:`O(n + E)` for tree sequence
   * - Scaling
     - :math:`\sim n^2 \cdot L`
     - :math:`\sim n \log n + n \cdot L`

The tree sequence output is particularly important: instead of storing the
full variant matrix (which is :math:`O(n \cdot S)` where :math:`S` is the
number of segregating sites), msprime stores edges and nodes, which is
:math:`O(n + \rho)` -- often orders of magnitude smaller.

This is the culmination of the master clockmaker's bench: a simulation
algorithm that scales gracefully to millions of samples and whole chromosomes.
In the next chapter, we add population structure and demographic history --
the case and dial that shape the genealogy's form.


Exercises
=========

.. admonition:: Exercise 1: Build a minimal simulator

   Use the ``MinimalSimulator`` class to simulate genealogies for
   :math:`n = 10, 50, 100` with :math:`L = 10^5` bp and
   :math:`r = 10^{-8}`. For each, record the number of trees in the
   tree sequence (= number of distinct marginal trees). Plot
   :math:`E[\text{num trees}]` vs :math:`n` and compare to the
   theoretical expectation.

.. admonition:: Exercise 2: Verify the exponential race

   Instrument the main loop to record which event type wins at each step.
   Verify that the fraction of coalescence vs recombination events matches
   the ratio of their rates.

.. admonition:: Exercise 3: The coalescence merge

   Create two lineages with overlapping segment chains (draw them on paper
   first). Walk through ``merge_two_ancestors`` by hand, tracking which
   segments are created, which edges are recorded, and how the overlap
   counter changes.

.. admonition:: Exercise 4: Fenwick vs naive

   Time the simulation with and without the Fenwick tree (replace
   ``FenwickTree.find()`` with a linear scan). At what genome length does
   the Fenwick tree become faster?

Next: :ref:`msprime_demographics` -- how population size changes, migration,
and growth affect the simulation.


Solutions
=========

.. admonition:: Solution 1: Build a minimal simulator

   The expected number of distinct marginal trees grows roughly linearly with
   the population-scaled recombination rate :math:`\rho = 4 N_e r L`. For
   :math:`N_e = 1` (coalescent units), :math:`\rho = 4 r L`. We count trees
   by counting how many recombination events produced distinct breakpoints.

   .. code-block:: python

      import numpy as np

      r = 1e-8
      L = 1e5
      n_reps = 50

      print(f"{'n':>5s}  {'E[num trees]':>14s}  {'E[coal events]':>16s}  "
            f"{'E[recomb events]':>18s}")

      for n in [10, 50, 100]:
          num_trees_list = []
          coal_list = []
          recomb_list = []

          for _ in range(n_reps):
              sim = MinimalSimulator(n=n, sequence_length=L,
                                     recombination_rate=r, pop_size=1.0)
              edges, nodes = sim.simulate()

              # Number of trees = number of distinct edge left boundaries + 1
              # (each recombination creates a new breakpoint and a new tree)
              breakpoints = set()
              for left, right, parent, child in edges:
                  breakpoints.add(left)
                  breakpoints.add(right)
              num_trees = len(breakpoints) - 1  # subtract the 0 and L endpoints
              num_trees = max(1, num_trees)

              num_trees_list.append(num_trees)
              coal_list.append(sim.num_ca_events)
              recomb_list.append(sim.num_re_events)

          print(f"{n:5d}  {np.mean(num_trees_list):14.1f}  "
                f"{np.mean(coal_list):16.1f}  "
                f"{np.mean(recomb_list):18.1f}")

      # The number of trees grows roughly as rho * log(n) for the
      # standard coalescent, since the expected number of recombination
      # events on the tree is proportional to rho * sum(1/k).

.. admonition:: Solution 2: Verify the exponential race

   We instrument the ``MinimalSimulator`` to track which event type wins at
   each step, then compare to the expected fractions based on the rates.

   .. code-block:: python

      import numpy as np

      n = 20
      L = 1e4
      r = 1e-4
      rho = 4 * 1.0 * r * L  # pop_size=1.0 in coalescent units

      sim = MinimalSimulator(n=n, sequence_length=L,
                              recombination_rate=r, pop_size=1.0)
      edges, nodes = sim.simulate()

      total_events = sim.num_ca_events + sim.num_re_events
      coal_frac = sim.num_ca_events / total_events
      recomb_frac = sim.num_re_events / total_events

      print(f"Total events: {total_events}")
      print(f"Coalescence events: {sim.num_ca_events} ({coal_frac:.3f})")
      print(f"Recombination events: {sim.num_re_events} ({recomb_frac:.3f})")
      print(f"\nNote: The fraction varies over the simulation because rates")
      print(f"change as lineages are added (recombination) and removed")
      print(f"(coalescence). The coalescence rate is k*(k-1)/2 while the")
      print(f"recombination rate is proportional to total segment mass.")
      print(f"At each step, P(coal) = coal_rate / (coal_rate + recomb_rate).")

.. admonition:: Solution 3: The coalescence merge

   Consider two lineages with the following segment chains:

   .. code-block:: text

      Lineage x: [0, 400: node 0] -> [600, 1000: node 0]
      Lineage y: [200, 800: node 1]

   Walking through ``merge_two_ancestors``:

   1. Both active, x.left=0 < y.left=200: x starts before y.
      x.right=400 > y.left=200, and x.left=0 < y.left=200, so partial overlap.
      Trim prefix: create segment [0, 200: node 0] (passes through from x).
      Set x.left = 200.

   2. Now x=[200, 400: node 0], y=[200, 800: node 1]. Both start at 200.
      **Coalescence!** Create ancestor node u. left=200, right=min(400,800)=400.
      Record edges: (200, 400, u, 0) and (200, 400, u, 1).
      Decrement overlap counter for [200, 400).
      Create coalesced segment [200, 400: node u].
      x is fully consumed (x.right==400), advance to x.next=[600, 1000: node 0].
      y has leftover: y.left = 400.

   3. Now x=[600, 1000: node 0], y=[400, 800: node 1]. y.left=400 < x.left=600.
      Swap so x=y: x=[400, 800: node 1], y=[600, 1000: node 0].
      x.right=800 > y.left=600, and x.left=400 < y.left=600: partial overlap.
      Create segment [400, 600: node 1] (passes through from x).
      Set x.left = 600.

   4. Now x=[600, 800: node 1], y=[600, 1000: node 0]. Both start at 600.
      **Coalescence!** left=600, right=min(800,1000)=800.
      Record edges: (600, 800, u, 1) and (600, 800, u, 0).
      Decrement overlap for [600, 800).
      Create coalesced segment [600, 800: node u].
      x is fully consumed, x=None.
      y has leftover: y.left = 800.

   5. x is None: absorb rest of y=[800, 1000: node 0].
      Segment [800, 1000: node 0] passes through.

   Final merged chain:

   .. code-block:: text

      [0, 200: node 0] -> [200, 400: node u] -> [400, 600: node 1]
          -> [600, 800: node u] -> [800, 1000: node 0]

   Edges recorded: (200, 400, u, 0), (200, 400, u, 1),
   (600, 800, u, 1), (600, 800, u, 0).

   The overlap counter was decremented at [200, 400) and [600, 800) --
   the intervals where both lineages had ancestral material.

.. admonition:: Solution 4: Fenwick vs naive

   We replace ``FenwickTree.find()`` with a linear scan and compare wall-clock
   times for increasing genome lengths.

   .. code-block:: python

      import numpy as np
      import time

      def naive_find(values, target):
          """Linear scan to find the index where cumulative sum >= target."""
          cumsum = 0
          for i in range(1, len(values)):
              cumsum += values[i]
              if cumsum >= target:
                  return i
          return len(values) - 1

      n = 50
      r = 1e-8

      print(f"{'L':>10s}  {'Fenwick (s)':>12s}  {'Naive (s)':>12s}  {'Speedup':>8s}")
      for L in [1e4, 1e5, 1e6, 1e7]:
          # Fenwick-based simulation
          start = time.time()
          sim = MinimalSimulator(n=n, sequence_length=L,
                                  recombination_rate=r, pop_size=1.0)
          sim.simulate()
          t_fenwick = time.time() - start

          # For the naive version, we estimate the time per find() operation
          # by running find() on a flat array with the same number of segments.
          n_segs = sim.num_re_events + n  # approximate number of segments
          values = np.random.exponential(1.0, size=n_segs + 1)
          total = values.sum()

          n_finds = 1000
          start = time.time()
          for _ in range(n_finds):
              target = np.random.uniform(0, total)
              naive_find(values, target)
          t_naive_per_find = (time.time() - start) / n_finds
          total_events = sim.num_ca_events + sim.num_re_events
          t_naive_estimate = t_naive_per_find * total_events + t_fenwick * 0.5

          speedup = t_naive_estimate / max(t_fenwick, 1e-6)
          print(f"{L:10.0f}  {t_fenwick:12.4f}  {t_naive_estimate:12.4f}  "
                f"{speedup:8.1f}x")

      print(f"\nThe Fenwick tree becomes faster when the number of segments is")
      print(f"large enough that O(log n) << O(n). For typical genome lengths")
      print(f"(L >= 1e5 bp), the Fenwick tree provides a significant speedup.")
