.. _segments_fenwick:

===============================
Segments & the Fenwick Tree
===============================

   *The gear train: the data structures that turn coalescent math into an
   efficient algorithm.*

In the previous chapter (:ref:`coalescent_process`), we derived all the rates
and distributions that govern the coalescent with recombination. We even wrote
a simple simulator. But that simulator was slow: it recomputed the total
segment length from scratch at every step, iterating over all segments to sum
their lengths every time we needed a rate. That is :math:`O(n)` per event.
With millions of events, this is too slow.

This chapter introduces the two data structures that transform the coalescent
from a mathematical specification into an efficient algorithm. Think of them
as the gear train of the master clockmaker's bench -- the mechanical linkage
that translates the escapement's regular beats (the coalescent math) into
the smooth motion of the hands (the simulation output).

msprime solves the performance problem with two data structures:

1. **Segment chains** -- doubly-linked lists representing the ancestral material
   of each lineage, enabling :math:`O(1)` splits and merges. These are the
   linked-list track that follows each lineage's ancestral material along the
   genome.

2. **Fenwick trees** -- cumulative frequency trees that maintain the total
   recombination mass across all segments, enabling :math:`O(\log n)` rate
   queries and updates. The Fenwick tree is a clever indexing mechanism for
   fast event scheduling.

.. note::

   **Prerequisites.** This chapter builds directly on :ref:`coalescent_process`,
   where we defined segments, lineages, recombination mass, and the event
   rates. If you need a refresher on what "recombination mass" means or why
   the total mass determines the recombination rate, revisit Steps 4-5 of
   that chapter.


Step 1: The Segment
=====================

A **segment** represents a contiguous stretch of ancestral genome carried by a
lineage. It has four essential fields:

- ``left``: the start position on the genome (inclusive)
- ``right``: the end position on the genome (exclusive)
- ``node``: the tree-sequence node ID where this ancestry was born
- ``next`` / ``prev``: pointers to adjacent segments in the chain

.. admonition:: Closing a confusion gap -- What is a segment, concretely?

   A segment is a small data object that says: "This lineage carries
   ancestral material for the genomic interval ``[left, right)``." At the
   start of the simulation, each of the :math:`n` sample lineages has
   exactly one segment covering the full genome ``[0, L)``. As
   recombination events split lineages, segments get shorter and lineages
   accumulate multiple segments separated by gaps. As coalescence events
   merge lineages, overlapping segments are combined, and edges are recorded
   in the tree sequence. The segment is the fundamental unit of bookkeeping
   in msprime.

.. code-block:: python

   import dataclasses

   @dataclasses.dataclass
   class Segment:
       """A contiguous stretch of ancestral genome.

       The segment covers the half-open interval [left, right) on the genome.
       Segments are linked into doubly-linked lists via prev/next pointers.
       """
       index: int          # unique ID (position in the segment pool)
       left: float = 0     # start position (inclusive)
       right: float = 0    # end position (exclusive)
       node: int = -1      # tree-sequence node ID
       prev: object = None # previous segment in the chain (toward left end of genome)
       next: object = None # next segment in the chain (toward right end of genome)

       @property
       def length(self):
           """Genomic span of this segment in base pairs."""
           return self.right - self.left

       def __repr__(self):
           return f"Seg({self.index}: [{self.left}, {self.right}), node={self.node})"

       @staticmethod
       def show_chain(seg):
           """Print the entire chain starting from seg."""
           parts = []
           while seg is not None:
               parts.append(f"[{seg.left}, {seg.right}: node {seg.node}]")
               seg = seg.next
           return " -> ".join(parts)

   # Example: a lineage carrying two non-contiguous segments
   # This happens after a coalescence event removed the middle portion.
   s1 = Segment(index=0, left=0, right=500, node=3)
   s2 = Segment(index=1, left=800, right=1000, node=3)
   s1.next = s2    # wire s1's "next" pointer to s2
   s2.prev = s1    # wire s2's "prev" pointer back to s1

   print("Segment chain:")
   print(f"  {Segment.show_chain(s1)}")
   print(f"  Total ancestry: {s1.length + s2.length} bp out of 1000 bp")


Why Linked Lists?
------------------

Why not arrays? Because the two most frequent operations are:

1. **Split** (recombination): break a segment at position :math:`x`
2. **Merge** (coalescence): combine two segment chains into one

Both are :math:`O(1)` with linked lists (just rewire pointers) but
:math:`O(n)` with arrays (shifting elements).

In the watch metaphor, segments are the linked-list track that follows each
lineage's ancestral material. Like the links of a fine watch bracelet, each
segment connects to the next, and you can open any link to insert or remove a
piece without disturbing the rest of the chain.

.. code-block:: python

   def split_segment(seg, breakpoint):
       """Split segment at breakpoint, returning (left_part, right_part).

       Before:  seg = [left, .... bp .... right)

       After:   seg = [left, bp)   alpha = [bp, right)

       This is O(1): we just create a new segment and rewire pointers.
       No array copying or shifting is needed.
       """
       alpha = Segment(
           index=-1,  # will be assigned by the pool
           left=breakpoint,
           right=seg.right,
           node=seg.node,
       )
       # Wire up the linked list: alpha inherits seg's successor
       alpha.next = seg.next
       if seg.next is not None:
           seg.next.prev = alpha
       alpha.prev = None  # alpha is the head of the right chain

       # Trim seg to end at the breakpoint
       seg.right = breakpoint
       seg.next = None  # seg is now the tail of the left chain

       return seg, alpha

   # Example
   seg = Segment(index=0, left=100, right=900, node=5)
   left, right = split_segment(seg, 400)
   print(f"Before split: [{100}, {900})")
   print(f"Left:  [{left.left}, {left.right})")
   print(f"Right: [{right.left}, {right.right})")

With the Segment defined, let us wrap it in a higher-level abstraction: the
Lineage.


Step 2: The Lineage
=====================

A **lineage** wraps a segment chain and adds metadata:

.. code-block:: python

   @dataclasses.dataclass
   class Lineage:
       """A single haploid genome in the simulation.

       The ancestry is stored as a linked list of Segments,
       accessed via head and tail pointers. The head points to the
       leftmost segment, the tail to the rightmost.
       """
       head: Segment       # first segment in the chain (leftmost on genome)
       tail: Segment       # last segment in the chain (rightmost on genome)
       population: int = 0 # which population this lineage resides in
       label: int = 0      # sub-label (used for selective sweeps)

       @property
       def total_length(self):
           """Total ancestral material carried by this lineage.

           Walk the chain and sum each segment's length. This is O(s)
           where s is the number of segments -- but we rarely call this
           because the Fenwick tree maintains the running total.
           """
           length = 0
           seg = self.head
           while seg is not None:
               length += seg.length
               seg = seg.next
           return length

       def __repr__(self):
           return (f"Lineage(pop={self.population}, "
                   f"chain={Segment.show_chain(self.head)})")

   # Example: lineage with two segments
   s1 = Segment(0, left=0, right=500, node=0)
   s2 = Segment(1, left=800, right=1000, node=0)
   s1.next = s2
   s2.prev = s1
   lin = Lineage(head=s1, tail=s2, population=0)
   print(lin)
   print(f"Total ancestry: {lin.total_length} bp")

Now we arrive at the key innovation that makes msprime fast.


Step 3: The Fenwick Tree
==========================

This is the key data structure that makes msprime fast. The Fenwick tree
(also called a Binary Indexed Tree or BIT) maintains a collection of values
and supports two operations in :math:`O(\log n)` time:

1. **Update**: change the value at an index
2. **Prefix sum**: compute the sum of values from index 1 to :math:`k`

From these, we can also:

3. **Total sum**: prefix sum up to the maximum index
4. **Find**: given a target sum :math:`v`, find the smallest index whose
   prefix sum :math:`\geq v`

.. admonition:: Closing a confusion gap -- Why do we need a Fenwick tree?

   The simulation needs to answer two questions very frequently:

   1. **"What is the total recombination rate?"** -- This is the sum of
      recombination masses over all segments. It determines the rate
      parameter for the exponential waiting time.
   2. **"Which segment should the next recombination hit?"** -- Given a
      random number, we need to find the segment whose cumulative mass
      contains that number (weighted random selection).

   A naive approach answers question 1 in :math:`O(n)` by summing all
   masses, and question 2 in :math:`O(n)` by scanning through segments.
   The Fenwick tree answers both in :math:`O(\log n)`. With millions of
   events, this is the difference between seconds and hours.

Let's build it from scratch.

The Key Insight
----------------

The Fenwick tree uses the **binary representation** of indices to organize
partial sums. Each position :math:`i` stores the sum of a specific range of
values, where the range is determined by the lowest set bit of :math:`i`.

The lowest set bit of :math:`i` is :math:`i \mathbin{\&} (-i)` (using two's complement).

- :math:`i = 1 = \texttt{0001}`: lowest bit = 1, stores value at index 1
- :math:`i = 2 = \texttt{0010}`: lowest bit = 2, stores sum of indices 1-2
- :math:`i = 3 = \texttt{0011}`: lowest bit = 1, stores value at index 3
- :math:`i = 4 = \texttt{0100}`: lowest bit = 4, stores sum of indices 1-4
- :math:`i = 5 = \texttt{0101}`: lowest bit = 1, stores value at index 5
- :math:`i = 6 = \texttt{0110}`: lowest bit = 2, stores sum of indices 5-6
- :math:`i = 7 = \texttt{0111}`: lowest bit = 1, stores value at index 7
- :math:`i = 8 = \texttt{1000}`: lowest bit = 8, stores sum of indices 1-8

.. admonition:: Closing a confusion gap -- The ``i & -i`` trick

   In two's complement representation, ``-i`` flips all bits of ``i`` and
   adds 1. The bitwise AND of ``i`` and ``-i`` isolates the lowest set bit.
   For example: ``6 = 0110``, ``-6 = 1010`` (in 4-bit two's complement),
   ``6 & -6 = 0010 = 2``. This single expression tells us the "responsibility
   range" of each position in the Fenwick tree. It is the fundamental building
   block of all Fenwick tree operations: to move *up* (toward larger ranges),
   we add ``i & -i``; to move *down* (toward smaller ranges), we subtract it.

.. code-block:: text

   Index:    1    2    3    4    5    6    7    8
   Values:  [3]  [1]  [4]  [1]  [5]  [9]  [2]  [6]

   Tree:    [3] [3+1] [4] [3+1+4+1] [5] [5+9] [2] [3+1+4+1+5+9+2+6]
          = [3]  [4]  [4]    [9]    [5]  [14] [2]       [31]

   To get prefix_sum(6): start at 6 = 0110
     tree[6] = 14       (sum of 5-6)
     6 - (6 & -6) = 6 - 2 = 4
     tree[4] = 9        (sum of 1-4)
     4 - (4 & -4) = 4 - 4 = 0  -> stop
     Result: 14 + 9 = 23  (3+1+4+1+5+9 = 23)


The Implementation
-------------------

.. code-block:: python

   class FenwickTree:
       """A Fenwick Tree for cumulative frequency tables.

       Supports O(log n) updates, prefix sums, and searches.
       Indices are 1-based (index 0 is unused).

       In msprime, this tree stores the recombination mass of each segment.
       It is the clever indexing mechanism for fast event scheduling:
       it lets the simulator quickly answer "what is the total recombination
       rate?" and "which segment should be hit next?"
       """

       def __init__(self, max_index):
           assert max_index > 0
           self.max_index = max_index
           self.tree = [0] * (max_index + 1)   # partial sums (the Fenwick structure)
           self.value = [0] * (max_index + 1)   # actual values at each index

           # Precompute the largest power of 2 <= max_index
           # (used by the find() method for efficient top-down search)
           u = max_index
           self.log_max = 0
           while u != 0:
               self.log_max = u
               u -= u & -u  # strip lowest set bit

       def increment(self, index, delta):
           """Add delta to the value at index. O(log n).

           This propagates the change upward through the tree:
           every ancestor node that includes this index in its range
           is also incremented.
           """
           assert 1 <= index <= self.max_index
           self.value[index] += delta
           j = index
           while j <= self.max_index:
               self.tree[j] += delta
               j += j & -j  # move to parent (next larger range)

       def set_value(self, index, new_value):
           """Set the value at index. O(log n).

           Computes the delta from the old value and calls increment.
           """
           old_value = self.value[index]
           self.increment(index, new_value - old_value)

       def get_value(self, index):
           """Return the value at index. O(1).

           The actual value is stored separately from the tree structure.
           """
           return self.value[index]

       def get_cumulative_sum(self, index):
           """Return the sum of values from 1 to index. O(log n).

           Walks downward through the tree, accumulating partial sums.
           At each step, we strip the lowest set bit to move to the
           next non-overlapping range.
           """
           assert 1 <= index <= self.max_index
           s = 0
           j = index
           while j > 0:
               s += self.tree[j]
               j -= j & -j  # strip lowest set bit (move to next range)
           return s

       def get_total(self):
           """Return the sum of all values. O(log n)."""
           return self.get_cumulative_sum(self.max_index)

       def find(self, target):
           """Find smallest index with cumulative sum >= target. O(log n).

           This is the inverse of get_cumulative_sum: given a target sum,
           find which index it falls in. This is used to select a random
           segment weighted by recombination mass.

           The algorithm performs a top-down binary search through the
           Fenwick tree, halving the search range at each step.
           """
           j = 0
           remaining = target
           half = self.log_max

           while half > 0:
               # Skip indices beyond max_index
               while j + half > self.max_index:
                   half >>= 1
               k = j + half
               if remaining > self.tree[k]:
                   # Target is beyond this subtree: skip it
                   j = k
                   remaining -= self.tree[j]
               half >>= 1  # halve the search range

           return j + 1

   # Demonstration
   ft = FenwickTree(8)
   values = [3, 1, 4, 1, 5, 9, 2, 6]
   for i, v in enumerate(values):
       ft.set_value(i + 1, v)  # Fenwick tree is 1-indexed

   print("Values:", [ft.get_value(i+1) for i in range(8)])
   print("Prefix sums:", [ft.get_cumulative_sum(i+1) for i in range(8)])
   print("Total:", ft.get_total())

   # Find: which index does cumulative sum 15 fall in?
   idx = ft.find(15)
   print(f"\nfind(15) = {idx}")
   print(f"  cumsum({idx-1}) = {ft.get_cumulative_sum(idx-1) if idx > 1 else 0}")
   print(f"  cumsum({idx}) = {ft.get_cumulative_sum(idx)}")
   print(f"  (15 falls in index {idx})")

Now let us see how the ``find()`` method powers the simulation.


Why the ``find()`` Method Matters
-----------------------------------

The ``find()`` method is the heart of msprime's breakpoint selection. Here's
how it works in the context of the simulation:

1. Each segment :math:`i` has a recombination "mass" :math:`m_i` stored in the
   Fenwick tree at index :math:`i`.

2. To choose a random breakpoint, we draw
   :math:`U \sim \text{Uniform}(0, M_{\text{total}})` where
   :math:`M_{\text{total}}` is the total mass.

3. We call ``find(U)`` to find which segment :math:`i` the random mass falls
   in. This segment will experience the recombination.

4. Within that segment, we compute the exact breakpoint position using the
   rate map.

This gives us a **weighted random selection** of segments in :math:`O(\log n)`
time. Without the Fenwick tree, we'd need :math:`O(n)` to iterate over all
segments.

.. admonition:: Probability Aside -- Weighted random selection via inverse CDF

   The ``find()`` operation is an instance of the **inverse CDF method**.
   If we have weights :math:`m_1, m_2, \ldots, m_n` with total
   :math:`M = \sum m_i`, then drawing :math:`U \sim \text{Uniform}(0, M)` and
   finding the smallest :math:`k` such that :math:`\sum_{i=1}^k m_i \geq U`
   selects index :math:`k` with probability :math:`m_k / M`. The Fenwick tree
   makes this :math:`O(\log n)` instead of :math:`O(n)` by organizing the
   partial sums hierarchically.

.. code-block:: python

   import numpy as np

   def choose_random_segment(fenwick_tree, segments):
       """Choose a random segment weighted by recombination mass.

       This is the core selection operation used every time a
       recombination event occurs in the simulation.

       Parameters
       ----------
       fenwick_tree : FenwickTree
           Stores recombination mass for each segment.
       segments : dict of {index: Segment}
           All active segments.

       Returns
       -------
       segment : Segment
           The chosen segment.
       mass_within : float
           How far into this segment's mass the random point fell.
       """
       total_mass = fenwick_tree.get_total()
       random_mass = np.random.uniform(0, total_mass)

       # Find which segment contains this mass -- O(log n)
       seg_index = fenwick_tree.find(random_mass)
       segment = segments[seg_index]

       # How far into this segment?
       cumsum = fenwick_tree.get_cumulative_sum(seg_index)
       mass_within_segment = fenwick_tree.get_value(seg_index)
       mass_from_right = cumsum - random_mass

       return segment, mass_from_right

   # Example: 5 segments with different masses
   ft = FenwickTree(5)
   masses = [10.0, 25.0, 5.0, 30.0, 15.0]
   for i, m in enumerate(masses):
       ft.set_value(i + 1, m)

   # Sample 10000 segments -- verify proportional selection
   counts = np.zeros(5)
   for _ in range(10000):
       total = ft.get_total()
       r = np.random.uniform(0, total)
       idx = ft.find(r)
       counts[idx - 1] += 1

   print("Sampling frequencies vs expected:")
   total = sum(masses)
   for i in range(5):
       print(f"  Segment {i}: observed={counts[i]/100:.1f}%, "
             f"expected={masses[i]/total*100:.1f}%")

The Fenwick tree handles the "which segment?" question. But we also need to
convert the random mass into a genomic position, which requires the rate map.


Step 4: The Rate Map
=====================

The recombination rate is not uniform across the genome. Humans, for example,
have recombination hotspots where the rate can be 100x higher than the
background. msprime handles this through **rate maps**.

A rate map is a piecewise-constant function :math:`r(x)` defined by breakpoints
and rates:

.. code-block:: text

   Position:  0      1000     2000     5000     10000
   Rate:        1e-8     5e-8     1e-8      2e-8

The **mass** of a genomic interval :math:`[a, b)` is the integral of the rate:

.. math::

   m(a, b) = \int_a^b r(x) \, dx

For a piecewise-constant rate, this is just the sum of rate times length for
each piece.

.. admonition:: Calculus Aside -- Piecewise integration

   For a piecewise-constant function :math:`r(x) = r_i` on
   :math:`[p_i, p_{i+1})`, the integral over :math:`[a, b)` is:

   .. math::

      \int_a^b r(x)\,dx = \sum_{i} r_i \cdot \max(0, \min(b, p_{i+1}) - \max(a, p_i))

   Each term contributes only for the part of :math:`[p_i, p_{i+1})` that
   overlaps with :math:`[a, b)`. In the implementation below, we precompute
   cumulative masses at each breakpoint so that ``mass_between(a, b)`` can
   be answered in :math:`O(\log m)` time (where :math:`m` is the number of
   rate intervals) using binary search.

.. code-block:: python

   class RateMap:
       """A piecewise-constant rate function over the genome.

       The rate in interval [positions[i], positions[i+1]) is rates[i].
       This class handles both recombination and mutation rate maps.
       """

       def __init__(self, positions, rates):
           """
           Parameters
           ----------
           positions : list of float
               Breakpoints (including 0 and L).
           rates : list of float
               Rate in each interval (len = len(positions) - 1).
           """
           assert len(rates) == len(positions) - 1
           self.positions = np.array(positions, dtype=float)
           self.rates = np.array(rates, dtype=float)

           # Precompute cumulative mass at each breakpoint
           # cumulative[i] = integral of r(x) from position[0] to position[i]
           self.cumulative = np.zeros(len(positions))
           for i in range(len(rates)):
               span = positions[i + 1] - positions[i]
               self.cumulative[i + 1] = self.cumulative[i] + rates[i] * span

       @property
       def total_mass(self):
           return self.cumulative[-1]

       def mass_between(self, left, right):
           """Compute the recombination mass of interval [left, right)."""
           return self.position_to_mass(right) - self.position_to_mass(left)

       def position_to_mass(self, pos):
           """Convert a genomic position to cumulative mass.

           This is the forward mapping: position -> mass.
           """
           # Find which interval pos falls in
           idx = np.searchsorted(self.positions, pos, side='right') - 1
           idx = max(0, min(idx, len(self.rates) - 1))
           # Mass up to the start of this interval + mass within
           return (self.cumulative[idx] +
                   self.rates[idx] * (pos - self.positions[idx]))

       def mass_to_position(self, mass):
           """Convert a cumulative mass back to genomic position (inverse).

           This is the inverse mapping: mass -> position.
           Used to convert a random mass coordinate into a breakpoint.
           """
           idx = np.searchsorted(self.cumulative, mass, side='right') - 1
           idx = max(0, min(idx, len(self.rates) - 1))
           # Position at start of interval + offset
           remaining_mass = mass - self.cumulative[idx]
           if self.rates[idx] == 0:
               return self.positions[idx]
           return self.positions[idx] + remaining_mass / self.rates[idx]

   # Example: genome with a recombination hotspot
   rate_map = RateMap(
       positions=[0, 5000, 6000, 10000],
       rates=[1e-8, 1e-6, 1e-8]  # 100x hotspot in [5000, 6000)
   )

   print(f"Total mass: {rate_map.total_mass:.2e}")
   print(f"Mass [0, 5000): {rate_map.mass_between(0, 5000):.2e}")
   print(f"Mass [5000, 6000): {rate_map.mass_between(5000, 6000):.2e}")
   print(f"Mass [6000, 10000): {rate_map.mass_between(6000, 10000):.2e}")
   print(f"\nThe 1kb hotspot has {rate_map.mass_between(5000, 6000) / rate_map.total_mass * 100:.1f}% "
         f"of total recombination mass")


Why Mass, Not Position?
------------------------

The Fenwick tree stores **mass** (rate-weighted length), not raw genomic
length. This is crucial: when we draw a random breakpoint, we want it
proportional to the local rate. By storing mass in the Fenwick tree, the
``find()`` method automatically gives us rate-weighted selection.

The conversion from mass back to position is handled by
``RateMap.mass_to_position()`` -- the inverse function.

Here is the full breakpoint selection pipeline, showing how the Fenwick tree,
the rate map, and the segment chain work together:

.. code-block:: python

   def choose_breakpoint(fenwick_tree, segments, rate_map):
       """Choose a random recombination breakpoint.

       This is the core of msprime's breakpoint selection:
       1. Draw random mass from [0, total_mass)
       2. Use Fenwick.find() to locate the segment   -- O(log n)
       3. Convert mass coordinate to genomic position -- O(log m)

       Parameters
       ----------
       fenwick_tree : FenwickTree
       segments : dict of {index: Segment}
       rate_map : RateMap

       Returns
       -------
       segment : Segment
           The segment where recombination occurs.
       breakpoint : float
           The genomic position of the breakpoint.
       """
       total_mass = fenwick_tree.get_total()
       random_mass = np.random.uniform(0, total_mass)

       # Step 1: find which segment (using the Fenwick tree's find)
       seg_index = fenwick_tree.find(random_mass)
       seg = segments[seg_index]

       # Step 2: compute breakpoint position
       # The cumulative mass up to this segment's right end
       cum_mass = fenwick_tree.get_cumulative_sum(seg_index)
       # Mass of the breakpoint from the right end of the segment
       mass_from_right = cum_mass - random_mass
       # Convert to genomic position using the rate map inverse
       right_mass = rate_map.position_to_mass(seg.right)
       bp_mass = right_mass - mass_from_right
       bp = rate_map.mass_to_position(bp_mass)

       return seg, bp

.. admonition:: The left-bound subtlety

   In msprime's implementation, the recombination mass of a segment is
   computed from ``get_recomb_left_bound(seg)`` to ``seg.right``. The left
   bound is ``seg.prev.right`` if the segment has a predecessor (i.e., it's
   not the head of the chain), because recombination between two adjacent
   segments of the same lineage has no effect -- both pieces already belong
   to the same lineage. Only recombination that falls in a **gap** or within
   a segment creates a meaningful split. This subtlety is easy to miss but
   essential for correctness.

With the rate map and Fenwick tree working together, we have efficient
breakpoint selection. Next, we need efficient memory management for the
millions of segments created and destroyed during the simulation.


Step 5: The Segment Pool
==========================

Creating and destroying segment objects millions of times would be slow due
to memory allocation overhead. msprime uses a **segment pool**: a pre-allocated
array of segments that are recycled.

.. admonition:: Closing a confusion gap -- Why a pool?

   In languages like Python, creating an object involves memory allocation,
   constructor calls, and eventually garbage collection. For an object
   created and destroyed millions of times per second, this overhead
   dominates. A pool pre-allocates all objects at startup and reuses them:
   "allocation" just pops an index from a free list (:math:`O(1)`), and
   "deallocation" pushes it back (:math:`O(1)`). The C implementation of
   msprime uses the same pattern for maximum performance.

.. code-block:: python

   class SegmentPool:
       """Pre-allocated pool of Segment objects.

       Avoids repeated memory allocation during the simulation.
       'Allocating' a segment just pops an index from the free list.
       'Freeing' a segment pushes it back.
       """

       def __init__(self, max_segments):
           # Pre-create all segment objects at once
           self.segments = [Segment(index=i) for i in range(max_segments + 1)]
           self.free_list = list(range(1, max_segments + 1))  # 1-indexed (0 unused)

       def alloc(self, left=0, right=0, node=-1):
           """Allocate a segment from the pool."""
           if not self.free_list:
               raise RuntimeError("Segment pool exhausted")
           index = self.free_list.pop()  # O(1) -- just pop from the stack
           seg = self.segments[index]
           seg.left = left
           seg.right = right
           seg.node = node
           seg.prev = None
           seg.next = None
           return seg

       def free(self, seg):
           """Return a segment to the pool."""
           self.free_list.append(seg.index)  # O(1) -- push back onto the stack
           seg.prev = None
           seg.next = None

       def copy(self, seg):
           """Allocate a new segment as a copy of an existing one."""
           new_seg = self.alloc(seg.left, seg.right, seg.node)
           new_seg.next = seg.next
           if seg.next is not None:
               seg.next.prev = new_seg
           return new_seg

   # Example
   pool = SegmentPool(100)
   s1 = pool.alloc(left=0, right=500, node=0)
   s2 = pool.alloc(left=500, right=1000, node=0)
   s1.next = s2
   s2.prev = s1

   print(f"Allocated: {Segment.show_chain(s1)}")
   print(f"Free slots remaining: {len(pool.free_list)}")

   pool.free(s2)
   print(f"After freeing s2: free slots = {len(pool.free_list)}")

The segment pool, the Fenwick tree, and the segment chains form the "gear
train" of the simulation. There is one more data structure to introduce: the
overlap counter that tells the simulation when it is done.


Step 6: The Overlap Counter S
================================

The simulation needs to know when it's done. It's done when every genomic
position has exactly one ancestral lineage (the MRCA). msprime tracks this
with an **overlap counter** :math:`S`: an AVL tree mapping genomic positions
to the number of lineages carrying ancestral material at that position.

.. admonition:: Closing a confusion gap -- What is an overlap counter?

   Imagine the genome as a number line from 0 to :math:`L`. Each lineage
   "paints" a color over the intervals where it carries ancestral material.
   The overlap counter :math:`S[x]` counts how many colors are stacked at
   position :math:`x`. At the start, all :math:`n` lineages cover the entire
   genome, so :math:`S[x] = n` everywhere. Each coalescence event at
   interval :math:`[a, b)` reduces :math:`S[x]` by 1 for :math:`x \in [a, b)`,
   because two lineages become one. When :math:`S[x] \leq 1` everywhere,
   every position has found its MRCA and the simulation is complete.

   The AVL tree (implemented here as a ``SortedDict``) stores this count
   as a piecewise-constant function: only the breakpoints where the count
   changes are stored, not every base pair individually.

.. code-block:: python

   from sortedcontainers import SortedDict

   class OverlapCounter:
       """Tracks the number of lineages at each genomic position.

       Uses an AVL tree (here SortedDict) to store a piecewise-constant
       function: S[x] gives the number of lineages at positions [x, next_key).
       """

       def __init__(self, sequence_length):
           self.S = SortedDict()
           self.S[0] = 0                    # count starts at 0
           self.S[sequence_length] = -1     # sentinel marking the end

       def increment(self, left, right, delta=1):
           """Increment the count in [left, right) by delta."""
           # Ensure breakpoints exist at left and right
           if left not in self.S:
               # Find the value just before left and copy it
               idx = self.S.bisect_left(left) - 1
               prev_key = self.S.keys()[idx]
               self.S[left] = self.S[prev_key]
           if right not in self.S:
               idx = self.S.bisect_left(right) - 1
               prev_key = self.S.keys()[idx]
               self.S[right] = self.S[prev_key]

           # Increment all positions in [left, right)
           for key in list(self.S.irange(left, right, (True, False))):
               self.S[key] += delta

       def is_complete(self):
           """Check if all positions have count <= 1 (MRCA found)."""
           for key in self.S:
               if self.S[key] > 1:
                   return False
           return True

       def __repr__(self):
           parts = []
           keys = list(self.S.keys())
           for i in range(len(keys) - 1):
               parts.append(f"  [{keys[i]}, {keys[i+1]}): {self.S[keys[i]]}")
           return "OverlapCounter:\n" + "\n".join(parts)

   # Example: 3 lineages with overlapping segments
   S = OverlapCounter(1000)
   S.increment(0, 1000)    # lineage 0: full genome
   S.increment(0, 1000)    # lineage 1: full genome
   S.increment(0, 1000)    # lineage 2: full genome
   print("Before any coalescence:")
   print(S)

   # After first coalescence at [0, 500)
   S.increment(0, 500, delta=-1)
   print("\nAfter coalescence at [0, 500):")
   print(S)
   print(f"Complete? {S.is_complete()}")

With all the data structures defined, let us see how they work together in
a single simulation step.


Step 7: Putting It All Together
=================================

Here's how the data structures work together in a single simulation step.
This is a preview of what :ref:`hudson_algorithm` -- the main simulation
loop, the ticking of the clock -- will orchestrate at full scale.

.. code-block:: python

   class SimulationState:
       """Minimal simulation state demonstrating the data structures.

       This ties together the segment pool, the Fenwick tree, the rate map,
       and the lineage list. In the full simulator (hudson_algorithm), these
       are augmented with populations, migration, and demographic events.
       """

       def __init__(self, n, L, recomb_rate):
           self.n = n
           self.L = L
           self.pool = SegmentPool(10 * n)
           self.rate_map = RateMap([0, L], [recomb_rate])

           # Fenwick tree for recombination mass -- the clever indexing mechanism
           self.mass_index = FenwickTree(10 * n)

           # Create initial lineages: each carries [0, L)
           self.lineages = []
           for i in range(n):
               seg = self.pool.alloc(left=0, right=L, node=i)
               lin = Lineage(head=seg, tail=seg, population=0)
               seg.lineage = lin
               self.lineages.append(lin)

               # Register this segment's mass in the Fenwick tree
               mass = self.rate_map.mass_between(0, L)
               self.mass_index.set_value(seg.index, mass)

       def get_total_recomb_rate(self):
           """Total recombination rate across all lineages.

           Thanks to the Fenwick tree, this is O(log n), not O(n).
           """
           return self.mass_index.get_total()

       def recombination_event(self):
           """Execute one recombination event."""
           # Step 1: Choose breakpoint using Fenwick tree -- O(log n)
           total_mass = self.mass_index.get_total()
           random_mass = np.random.uniform(0, total_mass)
           seg_index = self.mass_index.find(random_mass)
           seg = self.pool.segments[seg_index]

           # Step 2: Convert mass to position using rate map
           cum_mass = self.mass_index.get_cumulative_sum(seg_index)
           right_mass = self.rate_map.position_to_mass(seg.right)
           bp_mass = right_mass - (cum_mass - random_mass)
           bp = self.rate_map.mass_to_position(bp_mass)

           # Step 3: Split the segment -- O(1) pointer rewiring
           alpha = self.pool.copy(seg)
           alpha.left = bp
           alpha.prev = None
           if seg.next is not None:
               seg.next.prev = alpha
           seg.next = None
           seg.right = bp

           # Step 4: Update Fenwick tree -- O(log n)
           left_mass = self.rate_map.mass_between(seg.left, seg.right)
           self.mass_index.set_value(seg.index, left_mass)
           right_mass = self.rate_map.mass_between(alpha.left, alpha.right)
           self.mass_index.set_value(alpha.index, right_mass)

           # Step 5: Create new lineage for the right part
           old_lineage = seg.lineage
           new_lineage = Lineage(head=alpha, tail=alpha, population=0)
           alpha.lineage = new_lineage
           old_lineage.tail = seg
           self.lineages.append(new_lineage)

           return bp

   # Demo
   state = SimulationState(n=3, L=1000, recomb_rate=1e-3)
   print(f"Initial: {len(state.lineages)} lineages")
   print(f"Total recomb mass: {state.get_total_recomb_rate():.4f}")

   bp = state.recombination_event()
   print(f"\nAfter recombination at {bp:.1f}:")
   print(f"Now {len(state.lineages)} lineages")
   print(f"Total recomb mass: {state.get_total_recomb_rate():.4f}")

You have now seen every data structure that powers the master clockmaker's
bench. The segment chains are the linked-list track that follows each
lineage's ancestral material. The Fenwick tree is the clever indexing
mechanism for fast event scheduling. The segment pool eliminates memory
allocation overhead. And the overlap counter tracks progress toward completion.

In the next chapter, we assemble these parts into the complete simulation loop.


Exercises
=========

.. admonition:: Exercise 1: Fenwick tree operations

   Build a Fenwick tree with 16 elements. Set random values, then verify that
   ``get_cumulative_sum(i)`` matches a naive prefix sum for all :math:`i`.
   Also verify that ``find(v)`` returns the correct index for 100 random
   target values.

.. admonition:: Exercise 2: Weighted segment selection

   Create 100 segments with random masses. Use the Fenwick tree to sample
   10,000 segments. Verify that the empirical selection frequency of each
   segment matches its mass fraction to within 1%.

.. admonition:: Exercise 3: Breakpoint distribution with hotspots

   Create a rate map with a 100x hotspot covering 1% of the genome. Sample
   10,000 breakpoints using the Fenwick tree + rate map. Plot the breakpoint
   density and verify that ~50% of breakpoints fall in the hotspot.

.. admonition:: Exercise 4: Segment chain operations

   Implement a full recombination-coalescence cycle: start with 3 lineages
   each carrying [0, 1000), perform a recombination on lineage 1, then
   coalesce two lineages. Verify the segment chains are correct at each step.

Next: :ref:`hudson_algorithm` -- the main simulation loop that orchestrates
these data structures, the ticking of the clock.
