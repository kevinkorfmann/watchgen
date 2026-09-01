"""
Mini-msprime: Coalescent Simulation with Recombination.

This module implements the core algorithms from msprime (Kelleher et al. 2016),
a coalescent simulator that generates random ancestral histories (genealogies)
consistent with the evolutionary process. It works backwards in time, starting
from sampled genomes in the present and tracing their ancestry until all
lineages have found common ancestors.

The approach:
1. The Coalescent Process -- how lineages find common ancestors backwards in
   time, governed by exponential waiting times with rate binom(k,2) for k
   lineages.
2. Segments & the Fenwick Tree -- linked-list segments track which parts of
   the genome each lineage carries; Fenwick trees enable O(log n) event
   scheduling and breakpoint selection.
3. Hudson's Algorithm -- the main event-driven simulation loop that races
   coalescence, recombination, and migration against each other, always
   executing whichever happens first.
4. Demographics & Mutations -- population size changes, migration, growth,
   and the final step of painting mutations onto the genealogy via a Poisson
   process on branches.

Key concepts:
- Exponential race: min of independent Exp(lambda_i) is Exp(sum lambda_i),
  and event i wins with probability lambda_i / sum(lambda_j).
- Fenwick tree (Binary Indexed Tree): O(log n) prefix sums, updates, and
  weighted random selection for breakpoint choice.
- Segment chains: doubly-linked lists for O(1) splits and merges of ancestral
  material during recombination and coalescence events.
- Tree sequence output: edges and nodes encoding the genealogical history
  compactly as a sequence of marginal trees.

Reference:
    Kelleher J, Etheridge AM, McVean G (2016). Efficient coalescent
    simulation and genealogical analysis for large sample sizes. PLoS
    Computational Biology, 12(5): e1004842.
"""

import numpy as np
import math
import dataclasses


# ============================================================================
# Coalescent Process (coalescent.rst)
# ============================================================================

def simulate_coalescence_time_discrete(N, n_replicates=10000):
    """Simulate coalescence time for 2 lineages in discrete generations.

    Each generation, the two lineages independently choose a parent
    uniformly from N individuals. If they pick the same one, they coalesce.
    """
    times = np.zeros(n_replicates)
    for rep in range(n_replicates):
        g = 0
        while True:
            g += 1
            if np.random.random() < 1.0 / N:
                break
        times[rep] = g
    return times


def simulate_coalescence_time_continuous(n_replicates=10000):
    """Simulate coalescence time for 2 lineages (exponential).

    In the continuous-time limit, the waiting time is Exp(1)
    in coalescent units (units of N generations).
    """
    return np.random.exponential(1.0, size=n_replicates)


def simulate_coalescent(n, n_replicates=1):
    """Simulate the standard coalescent for n samples.

    Returns
    -------
    all_results : list of (times, pairs)
        times : list of float
            Coalescence times (in coalescent units of N generations).
        pairs : list of (int, int)
            Which pair coalesced at each event.
    """
    all_results = []

    for _ in range(n_replicates):
        lineages = list(range(n))
        times = []
        pairs = []
        t = 0.0

        while len(lineages) > 1:
            k = len(lineages)
            rate = k * (k - 1) / 2
            t += np.random.exponential(1.0 / rate)
            i, j = sorted(np.random.choice(len(lineages), 2, replace=False))
            pairs.append((lineages[i], lineages[j]))
            times.append(t)
            new_node = max(lineages) + 1
            lineages[i] = new_node
            lineages.pop(j)

        all_results.append((times, pairs))

    return all_results


def expected_tmrca(n):
    """Expected MRCA time for n samples (in coalescent units)."""
    return 2 * (1 - 1.0 / n)


def expected_total_branch_length(n):
    """Expected total branch length of the coalescent tree.

    This equals 2 * H_{n-1}, where H_k is the k-th harmonic number.
    """
    return 2 * sum(1.0 / k for k in range(1, n))


def exponential_race(*rates):
    """Simulate an exponential race.

    Each "competitor" proposes a random waiting time drawn from
    Exp(rate_i). The competitor with the shortest time wins.

    Parameters
    ----------
    rates : floats
        Rate of each competing process.

    Returns
    -------
    winner : int
        Index of the process that fired first.
    time : float
        Waiting time until the first event.
    """
    times = []
    for rate in rates:
        if rate > 0:
            times.append(np.random.exponential(1.0 / rate))
        else:
            times.append(np.inf)

    winner = np.argmin(times)
    return winner, times[winner]


def merge_segments(segs_a, segs_b):
    """Merge two segment lists (simplified: just concatenate and sort)."""
    all_segs = sorted(segs_a + segs_b)
    if not all_segs:
        return []
    merged = [all_segs[0]]
    for l, r in all_segs[1:]:
        if l <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r))
        else:
            merged.append((l, r))
    return merged


def pick_random_breakpoint(segs, L):
    """Pick a random breakpoint within the segment list."""
    total = sum(r - l for l, r in segs)
    target = np.random.uniform(0, total)
    cumulative = 0
    for l, r in segs:
        cumulative += r - l
        if cumulative >= target:
            bp = r - (cumulative - target)
            return bp
    return segs[-1][1]


def split_at_breakpoint(segs, bp):
    """Split segments at breakpoint bp into left and right."""
    left, right = [], []
    for l, r in segs:
        if r <= bp:
            left.append((l, r))
        elif l >= bp:
            right.append((l, r))
        else:
            left.append((l, bp))
            right.append((bp, r))
    return left, right


def coalescent_with_recombination_simple(n, L, rho, max_events=10000):
    """A count-level event sketch for the coalescent with recombination.

    This teaching helper does not retain parent-child ancestry or return a
    tree sequence; use :class:`MinimalSimulator` for a genealogical result.

    Parameters
    ----------
    n : int
        Sample size.
    L : float
        Genome length.
    rho : float
        Diploid population-scaled recombination rate (4*Ne*r*L).

    Returns
    -------
    events : list of (time, event_type, details)
    """
    lineages = [[(0, L)] for _ in range(n)]
    events = []
    t = 0.0

    for _ in range(max_events):
        k = len(lineages)
        if k <= 1:
            break

        coal_rate = k * (k - 1) / 2

        total_length = sum(
            sum(r - l for l, r in segs) for segs in lineages
        )
        recomb_rate = rho * total_length / (2 * L)

        winner, dt = exponential_race(coal_rate, recomb_rate)
        t += dt

        if winner == 0:
            i, j = sorted(np.random.choice(k, 2, replace=False))
            merged = merge_segments(lineages[i], lineages[j])
            lineages.pop(j)
            lineages[i] = merged
            events.append((t, 'coal', len(lineages)))

        else:
            lengths = [sum(r - l for l, r in segs) for segs in lineages]
            probs = np.array(lengths) / sum(lengths)
            idx = np.random.choice(k, p=probs)

            bp = pick_random_breakpoint(lineages[idx], L)
            left_segs, right_segs = split_at_breakpoint(lineages[idx], bp)

            if left_segs and right_segs:
                lineages[idx] = left_segs
                lineages.append(right_segs)
                events.append((t, 'recomb', len(lineages)))

    return events


def coalescent_waiting_time_constant(k, N):
    """Waiting time for k lineages, constant population size N."""
    rate = k * (k - 1) / 2
    u = np.random.exponential(1.0 / rate)
    return N * u


def coalescent_waiting_time_growth(k, N0, alpha, t0):
    """Waiting time for k lineages, exponential growth.

    Parameters
    ----------
    k : int
        Number of lineages.
    N0 : float
        Population size at time zero.
    alpha : float
        Growth rate (positive = population was smaller in the past).
    t0 : float
        Current time.

    Returns
    -------
    w : float
        Waiting time (can be inf if coalescence doesn't occur).
    """
    rate = k * (k - 1) / 2
    u = np.random.exponential(1.0 / rate)

    if alpha == 0:
        return N0 * u

    z = 1 + alpha * N0 * math.exp(-alpha * t0) * u
    if z <= 0:
        return np.inf

    return math.log(z) / alpha


def gene_conversion_event(segments, gc_position, tract_length, L):
    """Simulate a gene conversion event.

    Gene conversion removes a short tract from the main lineage and
    places it on a new lineage. This is like recombination, but
    instead of splitting left/right, it punches a "hole" in the middle.

    Parameters
    ----------
    segments : list of (left, right)
        Segments of the lineage.
    gc_position : float
        Starting position of the gene conversion.
    tract_length : float
        Length of the converted tract.
    L : float
        Sequence length.

    Returns
    -------
    main_segs : list of (left, right)
        Remaining segments of the main lineage.
    tract_segs : list of (left, right)
        Segments of the gene conversion tract lineage.
    """
    gc_left = gc_position
    gc_right = min(gc_position + tract_length, L)

    main_segs = []
    tract_segs = []

    for l, r in segments:
        if r <= gc_left or l >= gc_right:
            main_segs.append((l, r))
        elif l >= gc_left and r <= gc_right:
            tract_segs.append((l, r))
        elif l < gc_left and r > gc_right:
            main_segs.append((l, gc_left))
            tract_segs.append((gc_left, gc_right))
            main_segs.append((gc_right, r))
        elif l < gc_left:
            main_segs.append((l, gc_left))
            tract_segs.append((gc_left, r))
        else:
            tract_segs.append((l, gc_right))
            main_segs.append((gc_right, r))

    return main_segs, tract_segs


# ============================================================================
# Segments & the Fenwick Tree (segments_and_fenwick.rst)
# ============================================================================

@dataclasses.dataclass
class Segment:
    """A contiguous stretch of ancestral genome.

    The segment covers the half-open interval [left, right) on the genome.
    Segments are linked into doubly-linked lists via prev/next pointers.
    """
    index: int
    left: float = 0
    right: float = 0
    node: int = -1
    prev: object = None
    next: object = None

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


def split_segment(seg, breakpoint):
    """Split segment at breakpoint, returning (left_part, right_part).

    Before:  seg = [left, .... bp .... right)
    After:   seg = [left, bp)   alpha = [bp, right)

    This is O(1): we just create a new segment and rewire pointers.
    """
    alpha = Segment(
        index=-1,
        left=breakpoint,
        right=seg.right,
        node=seg.node,
    )
    alpha.next = seg.next
    if seg.next is not None:
        seg.next.prev = alpha
    alpha.prev = None

    seg.right = breakpoint
    seg.next = None

    return seg, alpha


@dataclasses.dataclass
class Lineage:
    """A single haploid genome in the simulation.

    The ancestry is stored as a linked list of Segments,
    accessed via head and tail pointers.
    """
    head: Segment
    tail: Segment
    population: int = 0
    label: int = 0

    @property
    def total_length(self):
        """Total ancestral material carried by this lineage."""
        length = 0
        seg = self.head
        while seg is not None:
            length += seg.length
            seg = seg.next
        return length


class FenwickTree:
    """A Fenwick Tree for cumulative frequency tables.

    Supports O(log n) updates, prefix sums, and searches.
    Indices are 1-based (index 0 is unused).

    In msprime, this tree stores the recombination mass of each segment.
    """

    def __init__(self, max_index):
        assert max_index > 0
        self.max_index = max_index
        self.tree = [0] * (max_index + 1)
        self.value = [0] * (max_index + 1)

        u = max_index
        self.log_max = 0
        while u != 0:
            self.log_max = u
            u -= u & -u

    def increment(self, index, delta):
        """Add delta to the value at index. O(log n)."""
        assert 1 <= index <= self.max_index
        self.value[index] += delta
        j = index
        while j <= self.max_index:
            self.tree[j] += delta
            j += j & -j

    def set_value(self, index, new_value):
        """Set the value at index. O(log n)."""
        old_value = self.value[index]
        self.increment(index, new_value - old_value)

    def get_value(self, index):
        """Return the value at index. O(1)."""
        return self.value[index]

    def get_cumulative_sum(self, index):
        """Return the sum of values from 1 to index. O(log n)."""
        assert 1 <= index <= self.max_index
        s = 0
        j = index
        while j > 0:
            s += self.tree[j]
            j -= j & -j
        return s

    def get_total(self):
        """Return the sum of all values. O(log n)."""
        return self.get_cumulative_sum(self.max_index)

    def find(self, target):
        """Find smallest index with cumulative sum >= target. O(log n)."""
        j = 0
        remaining = target
        half = self.log_max

        while half > 0:
            while j + half > self.max_index:
                half >>= 1
            if half == 0:
                break
            k = j + half
            if remaining > self.tree[k]:
                j = k
                remaining -= self.tree[j]
            half >>= 1

        return j + 1


class RateMap:
    """A piecewise-constant rate function over the genome.

    The rate in interval [positions[i], positions[i+1]) is rates[i].
    """

    def __init__(self, positions, rates):
        assert len(rates) == len(positions) - 1
        self.positions = np.array(positions, dtype=float)
        self.rates = np.array(rates, dtype=float)

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
        """Convert a genomic position to cumulative mass."""
        idx = np.searchsorted(self.positions, pos, side='right') - 1
        idx = max(0, min(idx, len(self.rates) - 1))
        return (self.cumulative[idx] +
                self.rates[idx] * (pos - self.positions[idx]))

    def mass_to_position(self, mass):
        """Convert a cumulative mass back to genomic position (inverse)."""
        idx = np.searchsorted(self.cumulative, mass, side='right') - 1
        idx = max(0, min(idx, len(self.rates) - 1))
        remaining_mass = mass - self.cumulative[idx]
        if self.rates[idx] == 0:
            return self.positions[idx]
        return self.positions[idx] + remaining_mass / self.rates[idx]


class SegmentPool:
    """Pre-allocated pool of Segment objects.

    Avoids repeated memory allocation during the simulation.
    """

    def __init__(self, max_segments):
        self.segments = [Segment(index=i) for i in range(max_segments + 1)]
        self.free_list = list(range(1, max_segments + 1))

    def alloc(self, left=0, right=0, node=-1):
        """Allocate a segment from the pool."""
        if not self.free_list:
            raise RuntimeError("Segment pool exhausted")
        index = self.free_list.pop()
        seg = self.segments[index]
        seg.left = left
        seg.right = right
        seg.node = node
        seg.prev = None
        seg.next = None
        return seg

    def free(self, seg):
        """Return a segment to the pool."""
        self.free_list.append(seg.index)
        seg.prev = None
        seg.next = None

    def copy(self, seg):
        """Allocate a new segment as a copy of an existing one."""
        new_seg = self.alloc(seg.left, seg.right, seg.node)
        new_seg.next = seg.next
        if seg.next is not None:
            seg.next.prev = new_seg
        return new_seg


# ============================================================================
# Hudson's Algorithm (hudson_algorithm.rst)
# ============================================================================

INFINITY = float('inf')


class MinimalSimulator:
    """A minimal but complete Hudson's algorithm implementation.

    This implements the core simulation loop with coalescence and
    recombination (no gene conversion, no migration for simplicity).
    """

    def __init__(self, n, sequence_length, recombination_rate, pop_size=1.0):
        self.n = n
        self.L = sequence_length
        self.Ne = pop_size
        self.t = 0.0

        self.recomb_rate = recombination_rate
        # This teaching simulator uses a haploid population-size convention.
        self.rho = 2 * pop_size * recombination_rate * sequence_length

        self.max_segs = 100 * n
        self.segments = [None] * (self.max_segs + 1)
        self.free_segs = list(range(self.max_segs, 0, -1))
        self.mass_index = FenwickTree(self.max_segs)

        self.lineages = []
        for i in range(n):
            seg_idx = self.free_segs.pop()
            seg = Segment(index=seg_idx, left=0, right=sequence_length,
                          node=i)
            self.segments[seg_idx] = seg
            lin = Lineage(head=seg, tail=seg, population=0)
            seg.lineage = lin
            self.lineages.append(lin)

            mass = recombination_rate * sequence_length
            self.mass_index.set_value(seg_idx, mass)

        self.edges = []
        self.nodes = []
        for i in range(n):
            self.nodes.append((0.0, 0))

        self.num_re_events = 0
        self.num_ca_events = 0

    def is_completed(self):
        """Check if all positions have found their MRCA."""
        return len(self.lineages) == 0

    def get_recomb_mass(self, seg):
        """Recombination mass of a segment."""
        left_bound = seg.prev.right if seg.prev is not None else seg.left
        return self.recomb_rate * (seg.right - left_bound)

    def simulate(self):
        """Run the simulation until completion."""
        while not self.is_completed():
            k = len(self.lineages)

            re_total = self.mass_index.get_total()
            t_re = (np.random.exponential(1.0 / re_total)
                    if re_total > 0 else INFINITY)

            coal_rate = k * (k - 1) / 2
            t_ca = INFINITY
            if coal_rate > 0:
                t_ca = self.Ne * np.random.exponential(
                    1.0 / coal_rate)

            min_t = min(t_re, t_ca)
            self.t += min_t

            if min_t == t_re:
                self._recombination_event()
            else:
                self._coalescence_event()

        return self.edges, self.nodes

    @staticmethod
    def _segments_in(lineage):
        """Return the segments in a lineage as a list."""
        segments = []
        seg = lineage.head
        while seg is not None:
            segments.append(seg)
            seg = seg.next
        return segments

    def _register_lineage(self, lineage):
        """Set lineage pointers and recombination masses for one chain."""
        segments = self._segments_in(lineage)
        if not segments:
            return
        lineage.head = segments[0]
        lineage.tail = segments[-1]
        for seg in segments:
            seg.lineage = lineage
            self.mass_index.set_value(seg.index, self.get_recomb_mass(seg))

    def _release_segments(self, segments):
        """Return segments to the simulator's small educational pool."""
        for seg in segments:
            self.mass_index.set_value(seg.index, 0.0)
            self.segments[seg.index] = None
            self.free_segs.append(seg.index)

    def _alloc_segment(self, left, right, node):
        if not self.free_segs:
            raise RuntimeError(
                "Segment pool exhausted; reduce the recombination rate or "
                "increase MinimalSimulator.max_segs"
            )
        index = self.free_segs.pop()
        seg = Segment(index=index, left=left, right=right, node=node)
        self.segments[index] = seg
        return seg

    @staticmethod
    def _node_at(segments, position):
        for seg in segments:
            if seg.left <= position < seg.right:
                return seg.node
        return None

    def _coverage(self, lineages, position):
        """Count extant ancestors at a genomic position."""
        return sum(
            self._node_at(self._segments_in(lineage), position) is not None
            for lineage in lineages
        )

    def _recombination_event(self):
        """Handle a recombination event: split one lineage into two."""
        self.num_re_events += 1
        random_mass = np.random.uniform(0, self.mass_index.get_total())
        seg_idx = self.mass_index.find(random_mass)
        y = self.segments[seg_idx]

        cum_mass = self.mass_index.get_cumulative_sum(seg_idx)
        mass_from_right = cum_mass - random_mass
        bp = y.right - mass_from_right / self.recomb_rate

        if bp <= y.left or bp >= y.right:
            return

        old_lin = y.lineage
        old_segments = self._segments_in(old_lin)
        for seg in old_segments:
            self.mass_index.set_value(seg.index, 0.0)

        if bp <= y.left:
            # The breakpoint lies in a gap between ancestral segments.
            left_tail = y.prev
            if left_tail is None:
                self._register_lineage(old_lin)
                return
            left_tail.next = None
            y.prev = None
            left_head = old_lin.head
            right_head = y
        else:
            old_next = y.next
            alpha = self._alloc_segment(bp, y.right, y.node)
            alpha.next = old_next
            if old_next is not None:
                old_next.prev = alpha
            y.right = bp
            y.next = None
            left_head = old_lin.head
            left_tail = y
            right_head = alpha

        right_tail = right_head
        while right_tail.next is not None:
            right_tail = right_tail.next

        self.lineages.remove(old_lin)
        left_lin = Lineage(left_head, left_tail, old_lin.population, old_lin.label)
        right_lin = Lineage(right_head, right_tail, old_lin.population, old_lin.label)
        self._register_lineage(left_lin)
        self._register_lineage(right_lin)
        self.lineages.extend([left_lin, right_lin])

    def _coalescence_event(self):
        """Handle a coalescence event: merge two lineages into one."""
        self.num_ca_events += 1
        k = len(self.lineages)

        all_lineages = list(self.lineages)
        i, j = sorted(np.random.choice(k, 2, replace=False))
        y_lin = self.lineages.pop(j)
        x_lin = self.lineages.pop(i)
        x_segments = self._segments_in(x_lin)
        y_segments = self._segments_in(y_lin)

        breakpoints = sorted({
            coordinate
            for seg in x_segments + y_segments
            for coordinate in (seg.left, seg.right)
        })
        output = []
        parent = None

        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            midpoint = (left + right) / 2
            x_node = self._node_at(x_segments, midpoint)
            y_node = self._node_at(y_segments, midpoint)
            if x_node is None and y_node is None:
                continue
            if x_node is None or y_node is None:
                output.append((left, right, x_node if y_node is None else y_node))
                continue

            if parent is None:
                parent = len(self.nodes)
                self.nodes.append((self.t, x_lin.population))
            self.edges.append((left, right, parent, x_node))
            self.edges.append((left, right, parent, y_node))

            # Once only the chosen pair covers an interval, their common
            # ancestor is its local MRCA and no material remains to simulate.
            if self._coverage(all_lineages, midpoint) > 2:
                output.append((left, right, parent))

        self._release_segments(x_segments + y_segments)

        if output:
            # Coalesce adjacent pieces represented by the same node.
            merged = []
            for left, right, node in output:
                if merged and merged[-1][1] == left and merged[-1][2] == node:
                    merged[-1] = (merged[-1][0], right, node)
                else:
                    merged.append((left, right, node))

            new_lin = Lineage(head=None, tail=None, population=x_lin.population)
            for left, right, node in merged:
                seg = self._alloc_segment(left, right, node)
                seg.prev = new_lin.tail
                if new_lin.head is None:
                    new_lin.head = seg
                else:
                    new_lin.tail.next = seg
                new_lin.tail = seg
            self._register_lineage(new_lin)
            self.lineages.append(new_lin)


# ============================================================================
# Demographics (demographics.rst)
# ============================================================================

class Population:
    """A population with time-varying size.

    Population size follows the formula:
      N(t) = start_size * exp(-growth_rate * (t - start_time))
    """

    def __init__(self, start_size=1.0, growth_rate=0.0, start_time=0.0):
        self.start_size = start_size
        self.growth_rate = growth_rate
        self.start_time = start_time
        self.num_ancestors = 0

    def get_size(self, t):
        """Population size at time t (backwards)."""
        dt = t - self.start_time
        return self.start_size * np.exp(-self.growth_rate * dt)

    def set_size(self, new_size, time):
        """Change population size at the given time."""
        self.start_size = new_size
        self.growth_rate = 0
        self.start_time = time

    def set_growth_rate(self, rate, time):
        """Change growth rate at the given time."""
        self.start_size = self.get_size(time)
        self.start_time = time
        self.growth_rate = rate


def bottleneck_event(lineages, intensity):
    """Collapse a Kingman interval into an instantaneous bottleneck.

    Parameters
    ----------
    lineages : list
        Current lineages.
    intensity : float
        Bottleneck intensity (higher = more coalescence).

    Returns
    -------
    lineages : list
        Lineages after the bottleneck.
    coalesced_pairs : list of (int, int)
        Representative input indices for each binary merger.
    """
    if intensity < 0:
        raise ValueError("intensity must be nonnegative")

    coalesced_pairs = []
    # Each entry stores a surviving lineage object and an input index used
    # solely to make the event log interpretable.
    remaining = [(lineage, index) for index, lineage in enumerate(lineages)]
    elapsed = 0.0
    while len(remaining) > 1:
        k = len(remaining)
        rate = k * (k - 1) / 2
        wait = np.random.exponential(1.0 / rate)
        if elapsed + wait > intensity:
            break
        elapsed += wait

        first, second = np.random.choice(k, size=2, replace=False)
        if first < second:
            first, second = second, first
        lineage_i, label_i = remaining.pop(first)
        _lineage_j, label_j = remaining.pop(second)
        coalesced_pairs.append((label_i, label_j))
        # A full ARG implementation would create a new ancestor here.  This
        # count-level helper retains one representative lineage instead.
        remaining.append((lineage_i, min(label_i, label_j)))

    return [lineage for lineage, _ in remaining], coalesced_pairs


def migration_event(populations, migration_matrix, source, dest):
    """Move a random lineage from source to dest.

    Parameters
    ----------
    populations : list of Population
    migration_matrix : 2D array
        M[j][k] = migration rate from j to k (backward).
    source : int
        Source population index.
    dest : int
        Destination population index.
    """
    pop = populations[source]
    idx = np.random.randint(pop.num_ancestors)

    pop.num_ancestors -= 1
    populations[dest].num_ancestors += 1


def mass_migration_event(populations, source, dest, fraction=1.0):
    """Move a fraction of lineages from source to dest.

    fraction=1.0 means all lineages move (population join).
    fraction<1.0 means a subset moves (admixture).

    Parameters
    ----------
    populations : list of Population
    source : int
    dest : int
    fraction : float
        Fraction of lineages to move (0 to 1).

    Returns
    -------
    n_to_move : int
        Number of lineages moved.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie between 0 and 1")
    n_source = populations[source].num_ancestors
    # Each lineage moves independently, matching msprime's MassMigration.
    n_to_move = np.random.binomial(n_source, fraction)

    populations[source].num_ancestors -= n_to_move
    populations[dest].num_ancestors += n_to_move

    return n_to_move


class DemographicEventQueue:
    """Priority queue of demographic events sorted by time."""

    def __init__(self):
        self.events = []

    def add_size_change(self, time, pop_id, new_size):
        """Schedule a population size change."""
        self.events.append((time, 'size_change', (pop_id, new_size)))
        self.events.sort()

    def add_growth_rate_change(self, time, pop_id, rate):
        """Schedule a growth rate change."""
        self.events.append((time, 'growth_rate', (pop_id, rate)))
        self.events.sort()

    def add_mass_migration(self, time, source, dest, fraction):
        """Schedule a mass migration (population split/join/admixture)."""
        self.events.append((time, 'mass_migration',
                            (source, dest, fraction)))
        self.events.sort()

    def add_migration_rate_change(self, time, source, dest, rate):
        """Schedule a migration rate change."""
        self.events.append((time, 'migration_rate',
                            (source, dest, rate)))
        self.events.sort()

    def next_event_time(self):
        """Time of the next scheduled event (or infinity if none)."""
        if self.events:
            return self.events[0][0]
        return INFINITY

    def pop_event(self):
        """Remove and return the next event."""
        return self.events.pop(0)


def simulate_with_demographics(n, L, recomb_rate, populations,
                               migration_matrix, event_queue):
    """Run a counts-only Hudson simulation with demographics.

    This teaching implementation follows lineage counts rather than genomic
    segments, but it executes every sampled coalescence, recombination, and
    migration event. It therefore illustrates the demographic event race
    without pretending to return a tree sequence.

    Parameters
    ----------
    n : int
        Total sample size across all populations.
    L : float
        Sequence length.
    recomb_rate : float
    populations : list of Population
    migration_matrix : 2D array
    event_queue : DemographicEventQueue
    """
    if n != sum(p.num_ancestors for p in populations):
        raise ValueError("n must equal the initial number of ancestors")
    if recomb_rate < 0 or L <= 0:
        raise ValueError("recomb_rate must be nonnegative and L must be positive")

    t = 0.0
    total_events = 0

    while True:
        total_lineages = sum(p.num_ancestors for p in populations)
        if total_lineages <= 1:
            break

        candidates = []
        for pop_id, pop in enumerate(populations):
            k = pop.num_ancestors
            if k > 1:
                wait = coalescent_waiting_time_growth(
                    k,
                    pop.start_size,
                    pop.growth_rate,
                    t - pop.start_time,
                )
                candidates.append((wait, "coalescence", (pop_id,)))

        recomb_total = total_lineages * recomb_rate * L
        if recomb_total > 0:
            candidates.append((
                np.random.exponential(1.0 / recomb_total),
                "recombination",
                (),
            ))

        for source in range(len(populations)):
            for dest in range(len(populations)):
                if source != dest and migration_matrix[source][dest] > 0:
                    rate = (populations[source].num_ancestors
                            * migration_matrix[source][dest])
                    if rate > 0:
                        candidates.append((
                            np.random.exponential(1.0 / rate),
                            "migration",
                            (source, dest),
                        ))

        stochastic = min(candidates, default=(INFINITY, None, ()))
        min_event_time, event_type, args = stochastic

        if t + min_event_time > event_queue.next_event_time():
            event_time, etype, args = event_queue.pop_event()
            t = event_time

            if etype == 'size_change':
                pop_id, new_size = args
                populations[pop_id].set_size(new_size, t)
            elif etype == 'growth_rate':
                pop_id, rate = args
                populations[pop_id].set_growth_rate(rate, t)
            elif etype == 'mass_migration':
                source, dest, fraction = args
                mass_migration_event(populations, source, dest, fraction)
            elif etype == 'migration_rate':
                source, dest, rate = args
                migration_matrix[source][dest] = rate
        else:
            if not np.isfinite(min_event_time):
                raise RuntimeError(
                    "Simulation cannot complete: lineages are isolated and "
                    "no future demographic event can connect them"
                )
            t += min_event_time
            total_events += 1

            if event_type == "coalescence":
                (pop_id,) = args
                populations[pop_id].num_ancestors -= 1
            elif event_type == "recombination":
                weights = np.array(
                    [pop.num_ancestors for pop in populations], dtype=float
                )
                pop_id = np.random.choice(len(populations), p=weights / weights.sum())
                populations[pop_id].num_ancestors += 1
            else:
                source, dest = args
                populations[source].num_ancestors -= 1
                populations[dest].num_ancestors += 1

    return t, total_events


def dtwf_generation(lineages, N, recomb_rate, L):
    """Simulate one generation of the DTWF model.

    Parameters
    ----------
    lineages : list
        Current lineages.
    N : int
        Population size.
    recomb_rate : float
        Per-bp recombination rate per generation.
    L : float
        Sequence length.

    Returns
    -------
    lineages : list
        Lineages after one generation.
    events : list
        Events that occurred.
    """
    events = []

    parents = {}
    for i, lin in enumerate(lineages):
        parent_id = np.random.randint(N)
        if parent_id not in parents:
            parents[parent_id] = []
        parents[parent_id].append(i)

    new_lineages = []
    for parent_id, children in parents.items():
        if len(children) == 1:
            new_lineages.append(lineages[children[0]])
        else:
            merged = lineages[children[0]]
            for c in children[1:]:
                events.append(('coal', children[0], c))
            new_lineages.append(merged)

    for lin in new_lineages:
        n_recombs = np.random.poisson(recomb_rate * L)
        for _ in range(n_recombs):
            bp = np.random.randint(1, int(L))
            events.append(('recomb', bp))

    return new_lineages, events


# ============================================================================
# Mutations (mutations.rst)
# ============================================================================

def simulate_mutations_infinite_sites(edges, nodes, sequence_length, mu):
    """Add mutations to a tree sequence under the infinite-sites model.

    Parameters
    ----------
    edges : list of (left, right, parent, child)
    nodes : list of (time, population)
    sequence_length : float
    mu : float
        Per-bp, per-generation mutation rate.

    Returns
    -------
    mutations : list of (position, node, time, ancestral, derived)
    """
    mutations = []

    for left, right, parent, child in edges:
        branch_length = nodes[parent][0] - nodes[child][0]
        span = right - left
        expected = mu * span * branch_length

        n_muts = np.random.poisson(expected)

        for _ in range(n_muts):
            position = np.random.uniform(left, right)
            time = np.random.uniform(nodes[child][0], nodes[parent][0])
            mutations.append((position, child, time, '0', '1'))

    mutations.sort(key=lambda m: m[0])
    return mutations


def expected_segregating_sites(n, theta):
    """Expected number of segregating sites.

    Uses the formula E[S] = theta * H_{n-1}, where H_k is the
    k-th harmonic number.
    """
    harmonic = sum(1.0 / k for k in range(1, n))
    return theta * harmonic


def watterson_estimator(S, n):
    """Estimate theta from the number of segregating sites.

    This inverts E[S] = theta * H_{n-1} to get theta_hat = S / H_{n-1}.
    """
    harmonic = sum(1.0 / k for k in range(1, n))
    return S / harmonic


def compute_sfs(mutations, genotype_matrix, n):
    """Compute the site frequency spectrum from genotype data.

    Parameters
    ----------
    genotype_matrix : ndarray of shape (n_sites, n_samples)
        0 = ancestral, 1 = derived.
    n : int
        Sample size.

    Returns
    -------
    sfs : ndarray of shape (n - 1,)
        sfs[i-1] = number of sites with derived allele count i.
    """
    sfs = np.zeros(n - 1, dtype=int)
    for site in genotype_matrix:
        count = int(site.sum())
        if 1 <= count <= n - 1:
            sfs[count - 1] += 1
    return sfs


def expected_sfs(n, theta):
    """Expected SFS under the standard neutral model.

    E[xi_i] = theta / i for i = 1, ..., n-1.
    """
    return np.array([theta / i for i in range(1, n)])


class MatrixMutationModel:
    """Finite-sites mutation model with transition matrix.

    Each mutation event changes the allelic state according to the
    transition matrix. The root state is drawn from root_distribution.
    """

    def __init__(self, alleles, root_distribution, transition_matrix):
        self.alleles = alleles
        self.root_distribution = np.asarray(root_distribution, dtype=float)
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        n = len(alleles)
        if self.root_distribution.shape != (n,):
            raise ValueError("root_distribution must have one value per allele")
        if self.transition_matrix.shape != (n, n):
            raise ValueError("transition_matrix must be square with one row per allele")
        if np.any(self.root_distribution < 0) or np.any(self.transition_matrix < 0):
            raise ValueError("mutation probabilities must be nonnegative")
        if not np.isclose(self.root_distribution.sum(), 1):
            raise ValueError("root_distribution must sum to one")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1):
            raise ValueError("each transition_matrix row must sum to one")

    def draw_root_state(self):
        """Sample the ancestral allele at the root."""
        return np.random.choice(len(self.alleles), p=self.root_distribution)

    def mutate(self, current_state):
        """Apply one mutation: change the state according to the matrix."""
        return np.random.choice(len(self.alleles),
                                p=self.transition_matrix[current_state])


class MutationRateMap:
    """Piecewise-constant mutation rate along the genome."""

    def __init__(self, positions, rates):
        self.positions = np.array(positions)
        self.rates = np.array(rates)

    def rate_at(self, position):
        """Get mutation rate at a specific position."""
        idx = np.searchsorted(self.positions, position, side='right') - 1
        idx = max(0, min(idx, len(self.rates) - 1))
        return self.rates[idx]

    def total_mass(self, left, right):
        """Total mutation mass over [left, right)."""
        total = 0
        for i in range(len(self.rates)):
            seg_left = self.positions[i]
            seg_right = self.positions[i + 1]
            ol = max(seg_left, left)
            or_ = min(seg_right, right)
            if or_ > ol:
                total += self.rates[i] * (or_ - ol)
        return total


def find_root(tree):
    """Find the root of a tree (node with no parent)."""
    for node, (parent, time, children) in tree.items():
        if parent is None:
            return node
    raise ValueError("No root found")


def place_mutations_on_tree(tree, mu, model, sequence_length):
    """Place mutations on a single marginal tree.

    Parameters
    ----------
    tree : dict
        Tree as {node: (parent, time, children)}.
    mu : float
        Per-site, per-generation mutation rate.
    model : MatrixMutationModel
    sequence_length : float

    Returns
    -------
    mutations : list of (position, node, parent_node, derived_state, time)
    leaf_states : dict of {leaf: allele_index}
    """
    mutations = []
    root = find_root(tree)

    root_state = model.draw_root_state()
    node_states = {root: root_state}

    stack = [root]
    while stack:
        node = stack.pop()
        current_state = node_states[node]
        parent, time, children = tree[node]

        for child in children:
            _, child_time, _ = tree[child]
            branch_length = time - child_time

            n_muts = np.random.poisson(
                mu * sequence_length * branch_length
            )

            state = current_state
            # Traverse from the older parent toward the younger child so the
            # recorded sequence of states follows ancestry forward in time.
            mutation_times = np.sort(
                np.random.uniform(child_time, time, size=n_muts)
            )[::-1]
            for mut_time in mutation_times:
                new_state = model.mutate(state)
                position = np.random.uniform(0, sequence_length)
                mutations.append((position, child, node,
                                  model.alleles[new_state], mut_time))
                state = new_state

            node_states[child] = state
            stack.append(child)

    leaf_states = {node: node_states[node]
                   for node in tree
                   if not tree[node][2]}

    return mutations, leaf_states


def build_genotype_matrix(mutations, tree_sequence, n_samples):
    """Convert mutations to a genotype matrix.

    Parameters
    ----------
    mutations : list of (position, node, ...)
    tree_sequence : object
    n_samples : int

    Returns
    -------
    genotypes : ndarray of shape (n_sites, n_samples)
    positions : ndarray of shape (n_sites,)
    """
    n_sites = len(mutations)
    genotypes = np.zeros((n_sites, n_samples), dtype=int)
    positions = np.zeros(n_sites)

    for i, (pos, node, *_) in enumerate(mutations):
        positions[i] = pos
        descendants = get_descendants(tree_sequence, node, pos)
        for sample in descendants:
            if sample < n_samples:
                genotypes[i, sample] = 1

    return genotypes, positions


def get_descendants(tree_sequence, node, position):
    """Get all leaf descendants of a node at a genomic position."""
    tree = tree_sequence.at(position)
    return list(tree.samples(node))


# ============================================================================
# Solution functions from exercises
# ============================================================================

def simulate_island_coalescence(same_deme=True, N=1000, m=0.001, d=3):
    """Simulate coalescence time for 2 lineages in an island model."""
    if same_deme:
        deme = [0, 0]
    else:
        deme = [0, 1]
    t = 0.0
    while True:
        coal_rate = (1.0 / N) if deme[0] == deme[1] else 0.0
        mig_rate = 2 * m * (d - 1)
        total_rate = coal_rate + mig_rate
        dt = np.random.exponential(1.0 / total_rate)
        t += dt
        if np.random.random() < coal_rate / total_rate:
            return t
        else:
            lin = np.random.randint(2)
            new_deme = np.random.choice(
                [x for x in range(d) if x != deme[lin]])
            deme[lin] = new_deme


def simulate_dtwf_tmrca(n, N, n_reps=5000):
    """Simulate T_MRCA using the exact DTWF model."""
    tmrca_values = []
    for _ in range(n_reps):
        k = n
        t = 0
        while k > 1:
            t += 1
            parents = np.random.randint(0, N, size=k)
            k = len(set(parents))
        tmrca_values.append(t)
    return np.array(tmrca_values)


def simulate_coalescent_tmrca(n, N, n_reps=5000):
    """Simulate T_MRCA using the continuous-time coalescent."""
    tmrca_values = []
    for _ in range(n_reps):
        t = 0.0
        k = n
        while k > 1:
            rate = k * (k - 1) / 2
            t += 2 * N * np.random.exponential(1.0 / rate)
            k -= 1
        tmrca_values.append(t)
    return np.array(tmrca_values)


# ============================================================================
# Demo function
# ============================================================================

def demo():
    """Demonstrate the core msprime algorithms."""
    print("=" * 60)
    print("Mini-msprime: Coalescent Simulation with Recombination")
    print("=" * 60)

    # --- Coalescent Process ---
    print("\n--- The Coalescent Process ---")
    N = 10000
    discrete_times = simulate_coalescence_time_discrete(N, n_replicates=5000) / N
    continuous_times = simulate_coalescence_time_continuous(n_replicates=5000)
    print(f"Discrete:   mean = {discrete_times.mean():.4f}, "
          f"var = {discrete_times.var():.4f}")
    print(f"Continuous: mean = {continuous_times.mean():.4f}, "
          f"var = {continuous_times.var():.4f}")
    print(f"Theory:     mean = 1.0000, var = 1.0000")

    print("\nCoalescent for n=5:")
    np.random.seed(42)
    results = simulate_coalescent(5, n_replicates=1)
    times, pairs = results[0]
    for i, (t, (a, b)) in enumerate(zip(times, pairs)):
        print(f"  k={5 - i}: t={t:.4f}, lineages {a} and {b} coalesce")

    print("\nExpected TMRCA and total branch length:")
    for n in [2, 5, 10, 50, 100, 1000]:
        print(f"  n={n:>5d}: E[T_MRCA] = {expected_tmrca(n):.4f}, "
              f"E[total length] = {expected_total_branch_length(n):.4f}")

    # --- Exponential Race ---
    print("\n--- Exponential Race ---")
    np.random.seed(42)
    wins = np.zeros(2)
    for _ in range(10000):
        w, t = exponential_race(10.0, 5.0)
        wins[w] += 1
    print(f"Coalescence wins: {wins[0] / 100:.1f}% "
          f"(expected: {10 / 15 * 100:.1f}%)")
    print(f"Recombination wins: {wins[1] / 100:.1f}% "
          f"(expected: {5 / 15 * 100:.1f}%)")

    # --- Segments and Fenwick Tree ---
    print("\n--- Segment Chain ---")
    s1 = Segment(index=0, left=0, right=500, node=3)
    s2 = Segment(index=1, left=800, right=1000, node=3)
    s1.next = s2
    s2.prev = s1
    print(f"  {Segment.show_chain(s1)}")
    print(f"  Total ancestry: {s1.length + s2.length} bp out of 1000 bp")

    print("\n--- Fenwick Tree ---")
    ft = FenwickTree(8)
    values = [3, 1, 4, 1, 5, 9, 2, 6]
    for i, v in enumerate(values):
        ft.set_value(i + 1, v)
    print(f"Values: {[ft.get_value(i + 1) for i in range(8)]}")
    print(f"Prefix sums: {[ft.get_cumulative_sum(i + 1) for i in range(8)]}")
    print(f"Total: {ft.get_total()}")
    idx = ft.find(15)
    print(f"find(15) = {idx}")

    # --- Rate Map ---
    print("\n--- Rate Map ---")
    rate_map = RateMap(
        positions=[0, 5000, 6000, 10000],
        rates=[1e-8, 1e-6, 1e-8]
    )
    print(f"Total mass: {rate_map.total_mass:.2e}")
    print(f"Mass [5000, 6000) (hotspot): {rate_map.mass_between(5000, 6000):.2e}")

    # --- MinimalSimulator ---
    print("\n--- Hudson's Algorithm (MinimalSimulator) ---")
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

    # --- Demographics ---
    print("\n--- Demographics ---")
    pop = Population(start_size=10000)
    print(f"Present size: {pop.get_size(0):.0f}")
    pop.set_size(1000, time=1000)
    print(f"Size after bottleneck (t=1000): {pop.get_size(1000):.0f}")

    pop2 = Population(start_size=10000, growth_rate=0.005, start_time=0)
    for t in [0, 100, 500, 1000, 2000]:
        print(f"  t={t:>5d}: N = {pop2.get_size(t):.0f}")

    # --- Mutations ---
    print("\n--- Mutations ---")
    n, theta = 50, 100
    E_S = expected_segregating_sites(n, theta)
    print(f"n={n}, theta={theta}")
    print(f"Expected segregating sites: {E_S:.1f}")
    print(f"Watterson's estimate from E[S]: {watterson_estimator(E_S, n):.1f}")

    print("\nExpected SFS (first 5 entries):")
    exp_sfs = expected_sfs(20, 50)
    for i in range(5):
        bar = '#' * int(exp_sfs[i])
        print(f"  freq {i + 1:>2d}/20: E[xi] = {exp_sfs[i]:.2f}  {bar}")

    # --- Mutation Model ---
    print("\n--- Jukes-Cantor Mutation Model ---")
    np.random.seed(42)
    jc_model = MatrixMutationModel(
        alleles=['A', 'C', 'G', 'T'],
        root_distribution=[0.25, 0.25, 0.25, 0.25],
        transition_matrix=[
            [0, 1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 0, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 0, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3, 0],
        ]
    )
    state = 0
    chain = ['A']
    for _ in range(10):
        state = jc_model.mutate(state)
        chain.append(jc_model.alleles[state])
    print("Mutation chain starting from A:")
    print(" -> ".join(chain))

    # --- Mutation Rate Map ---
    print("\n--- Mutation Rate Map ---")
    mut_map = MutationRateMap(
        positions=[0, 4000, 6000, 10000],
        rates=[1.5e-8, 1e-9, 1.5e-8]
    )
    for x in [1000, 5000, 8000]:
        print(f"  position {x}: mu = {mut_map.rate_at(x):.2e}")
    print(f"Total mass [0, 10000): {mut_map.total_mass(0, 10000):.2e}")

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    demo()
