"""Tests for Python code examples in docs/prerequisites/args.rst.

Each test re-defines the function/class from the RST documentation, then verifies
correctness via structural properties, mathematical invariants, or known behaviors.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Code block 1: simulate_arg
# ---------------------------------------------------------------------------

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
    next_label = n
    lineages = {}
    for i in range(n):
        lineages[i] = [(0.0, seq_length)]

    coal_events = []
    recomb_events = []
    current_time = 0.0

    while len(lineages) > 1:
        k = len(lineages)
        coal_rate = k * (k - 1) / 2
        recomb_rate = k * rho / 2
        total_rate = coal_rate + recomb_rate

        wait = np.random.exponential(1.0 / total_rate)
        current_time += wait

        if np.random.random() < coal_rate / total_rate:
            labels = list(lineages.keys())
            i, j = np.random.choice(len(labels), size=2, replace=False)
            c1, c2 = labels[i], labels[j]
            parent = next_label
            next_label += 1

            merged = lineages[c1] + lineages[c2]
            lineages[parent] = merged
            del lineages[c1]
            del lineages[c2]

            coal_events.append((current_time, c1, c2, parent))
        else:
            labels = list(lineages.keys())
            idx = np.random.randint(0, k)
            lineage = labels[idx]

            breakpoint = np.random.uniform(0, seq_length)

            left_material = [(l, min(r, breakpoint))
                             for l, r in lineages[lineage] if l < breakpoint]
            right_material = [(max(l, breakpoint), r)
                              for l, r in lineages[lineage] if r > breakpoint]

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


class TestSimulateArg:
    """Tests for the simulate_arg function."""

    def test_returns_two_lists(self):
        """The function should return two lists."""
        np.random.seed(0)
        coal, recomb = simulate_arg(5, rho=5.0)
        assert isinstance(coal, list)
        assert isinstance(recomb, list)

    def test_no_recombination_when_rho_zero(self):
        """With rho=0, there should be no recombination events, and exactly n-1 coalescences."""
        np.random.seed(1)
        n = 6
        coal, recomb = simulate_arg(n, rho=0.0)
        assert len(recomb) == 0
        assert len(coal) == n - 1

    def test_coalescence_events_have_correct_structure(self):
        """Each coalescence event should be (time, child1, child2, parent)."""
        np.random.seed(2)
        coal, _ = simulate_arg(5, rho=2.0)
        for event in coal:
            assert len(event) == 4
            t, c1, c2, p = event
            assert t > 0.0
            assert c1 != c2

    def test_recombination_events_have_correct_structure(self):
        """Each recombination event should be (time, lineage, breakpoint, left, right)."""
        np.random.seed(3)
        _, recomb = simulate_arg(5, rho=10.0)
        for event in recomb:
            assert len(event) == 5
            t, lineage, bp, left, right = event
            assert t > 0.0
            assert 0.0 <= bp <= 1.0
            assert left != right

    def test_times_are_positive(self):
        """All event times should be positive."""
        np.random.seed(4)
        coal, recomb = simulate_arg(5, rho=5.0)
        for t, _, _, _ in coal:
            assert t > 0
        for t, _, _, _, _ in recomb:
            assert t > 0

    def test_breakpoints_within_sequence(self):
        """Recombination breakpoints should be within [0, seq_length]."""
        np.random.seed(5)
        seq_length = 2.0
        _, recomb = simulate_arg(5, rho=10.0, seq_length=seq_length)
        for _, _, bp, _, _ in recomb:
            assert 0.0 <= bp <= seq_length

    def test_more_recombination_with_higher_rho(self):
        """Higher rho should generally produce more recombination events.

        This is a statistical test over many replicates.
        """
        np.random.seed(6)
        n_reps = 100
        low_rho_recomb = []
        high_rho_recomb = []
        for _ in range(n_reps):
            _, recomb_low = simulate_arg(5, rho=1.0)
            low_rho_recomb.append(len(recomb_low))
            _, recomb_high = simulate_arg(5, rho=20.0)
            high_rho_recomb.append(len(recomb_high))
        assert np.mean(high_rho_recomb) > np.mean(low_rho_recomb)

    def test_terminates_with_two_samples(self):
        """The simplest case (n=2) should terminate and produce at least one coalescence."""
        np.random.seed(7)
        coal, recomb = simulate_arg(2, rho=5.0)
        # Must have at least 1 coalescence event to reduce 2 lineages to 1
        assert len(coal) >= 1

    def test_custom_seq_length(self):
        """Should work with different sequence lengths."""
        np.random.seed(8)
        seq_length = 10.0
        coal, recomb = simulate_arg(4, rho=3.0, seq_length=seq_length)
        for _, _, bp, _, _ in recomb:
            assert 0.0 <= bp <= seq_length
        assert len(coal) >= 1

    def test_doc_example_runs(self):
        """Run the documented example: seed=42, n=5, rho=5.0."""
        np.random.seed(42)
        coal_events, recomb_events = simulate_arg(n=5, rho=5.0)
        assert len(coal_events) >= 1
        # The doc prints both event lists; just verify they're non-empty lists
        assert isinstance(coal_events, list)
        assert isinstance(recomb_events, list)


# ---------------------------------------------------------------------------
# Code block 2: extract_marginal_trees
# ---------------------------------------------------------------------------

def extract_marginal_trees(coal_events, recomb_events, seq_length):
    """Extract the breakpoints where marginal trees change.

    Returns the breakpoints that partition the genome into segments
    with constant tree topology.
    """
    breakpoints = sorted(set([0.0, seq_length] +
                             [bp for _, _, bp, _, _ in recomb_events]))
    return breakpoints


class TestExtractMarginalTrees:
    """Tests for the extract_marginal_trees function."""

    def test_always_includes_endpoints(self):
        """Breakpoints should always include 0.0 and seq_length."""
        np.random.seed(10)
        coal, recomb = simulate_arg(5, rho=5.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        assert bp[0] == 0.0
        assert bp[-1] == 1.0

    def test_sorted_breakpoints(self):
        """Breakpoints should be sorted in increasing order."""
        np.random.seed(11)
        coal, recomb = simulate_arg(5, rho=10.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        for i in range(1, len(bp)):
            assert bp[i] > bp[i - 1]

    def test_no_recombination_gives_one_tree(self):
        """With no recombination events, there should be exactly 1 marginal tree
        (breakpoints = [0.0, seq_length], so number of trees = 1).
        """
        np.random.seed(12)
        coal, recomb = simulate_arg(5, rho=0.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        assert len(bp) == 2  # [0.0, 1.0]
        # Number of distinct trees = len(bp) - 1 = 1
        assert len(bp) - 1 == 1

    def test_num_trees_equals_recomb_plus_one(self):
        """Number of distinct marginal trees should equal number of recombination events + 1."""
        np.random.seed(13)
        coal, recomb = simulate_arg(5, rho=5.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        # All recombination breakpoints are unique (from set), plus the two endpoints
        # The number of unique breakpoints from recomb events might be less than len(recomb)
        # if two events happen at the same position (very unlikely with continuous dist).
        num_unique_recomb_breakpoints = len(set(b for _, _, b, _, _ in recomb))
        assert len(bp) - 1 == num_unique_recomb_breakpoints + 1

    def test_works_with_custom_seq_length(self):
        """Should work correctly with a non-unit sequence length."""
        np.random.seed(14)
        seq_length = 5.0
        coal, recomb = simulate_arg(4, rho=3.0, seq_length=seq_length)
        bp = extract_marginal_trees(coal, recomb, seq_length)
        assert bp[0] == 0.0
        assert bp[-1] == seq_length

    def test_doc_example_runs(self):
        """Run the documented example after simulating with seed=42."""
        np.random.seed(42)
        coal_events, recomb_events = simulate_arg(n=5, rho=5.0)
        breakpoints = extract_marginal_trees(coal_events, recomb_events, 1.0)
        assert len(breakpoints) >= 2


# ---------------------------------------------------------------------------
# Code block 3: SimpleTreeSequence
# ---------------------------------------------------------------------------

class SimpleTreeSequence:
    """A minimal tree sequence representation for educational purposes.

    Nodes are stored as (time, is_sample) tuples.
    Edges are stored as (left, right, parent, child) tuples, where
    [left, right) is the genomic interval where this parent-child
    relationship holds.
    """

    def __init__(self):
        self.nodes = []
        self.edges = []

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
        breakpoints = sorted(set(
            [0.0, seq_length] +
            [l for l, r, p, c in self.edges] +
            [r for l, r, p, c in self.edges]
        ))

        for i in range(len(breakpoints) - 1):
            pos = (breakpoints[i] + breakpoints[i + 1]) / 2
            active = [(p, c) for l, r, p, c in self.edges
                      if l <= pos < r]
            yield breakpoints[i], breakpoints[i + 1], active


def _build_doc_example_ts():
    """Helper to build the documented tree sequence example with 4 samples."""
    ts = SimpleTreeSequence()
    # Samples at time 0
    for _ in range(4):
        ts.add_node(0.0, is_sample=True)
    # Internal nodes
    ts.add_node(0.5)   # node 4
    ts.add_node(0.8)   # node 5
    ts.add_node(1.2)   # node 6

    # First tree (positions 0.0 to 0.6): topology ((0,1), (2,3))
    ts.add_edge(0.0, 0.6, 4, 0)
    ts.add_edge(0.0, 1.0, 4, 1)
    ts.add_edge(0.0, 1.0, 5, 2)
    ts.add_edge(0.0, 1.0, 5, 3)
    ts.add_edge(0.0, 0.6, 6, 4)
    ts.add_edge(0.0, 1.0, 6, 5)

    # After recombination at position 0.6, node 0 moves to node 5
    ts.add_edge(0.6, 1.0, 5, 0)

    return ts


class TestSimpleTreeSequence:
    """Tests for the SimpleTreeSequence class."""

    def test_add_node_returns_sequential_ids(self):
        """Node IDs should be 0, 1, 2, ... in order of creation."""
        ts = SimpleTreeSequence()
        assert ts.add_node(0.0, is_sample=True) == 0
        assert ts.add_node(0.0, is_sample=True) == 1
        assert ts.add_node(0.5) == 2
        assert ts.add_node(1.0) == 3

    def test_add_node_stores_time_and_flag(self):
        """Nodes should store their time and is_sample flag."""
        ts = SimpleTreeSequence()
        ts.add_node(0.0, is_sample=True)
        ts.add_node(0.5, is_sample=False)
        assert ts.nodes[0] == (0.0, True)
        assert ts.nodes[1] == (0.5, False)

    def test_add_edge_stores_correctly(self):
        """Edges should be stored as (left, right, parent, child) tuples."""
        ts = SimpleTreeSequence()
        ts.add_node(0.0, is_sample=True)
        ts.add_node(1.0)
        ts.add_edge(0.0, 1.0, 1, 0)
        assert ts.edges[0] == (0.0, 1.0, 1, 0)

    def test_trees_covers_full_sequence(self):
        """The marginal trees should cover the full sequence without gaps."""
        ts = _build_doc_example_ts()
        intervals = list(ts.trees(1.0))
        # Check that intervals tile [0, 1)
        assert intervals[0][0] == pytest.approx(0.0)
        assert intervals[-1][1] == pytest.approx(1.0)
        # No gaps
        for i in range(1, len(intervals)):
            assert intervals[i][0] == pytest.approx(intervals[i - 1][1])

    def test_doc_example_two_trees(self):
        """The documented example should produce exactly 2 distinct tree intervals."""
        ts = _build_doc_example_ts()
        intervals = list(ts.trees(1.0))
        # There should be 2 intervals: [0.0, 0.6) and [0.6, 1.0)
        assert len(intervals) == 2

    def test_doc_example_first_tree_topology(self):
        """In the first tree [0.0, 0.6), nodes 0 and 1 should be children of node 4."""
        ts = _build_doc_example_ts()
        intervals = list(ts.trees(1.0))
        left, right, edges = intervals[0]
        assert left == pytest.approx(0.0)
        assert right == pytest.approx(0.6)
        # edges are (parent, child) tuples
        edge_set = set(edges)
        assert (4, 0) in edge_set  # node 0 -> node 4
        assert (4, 1) in edge_set  # node 1 -> node 4
        assert (5, 2) in edge_set  # node 2 -> node 5
        assert (5, 3) in edge_set  # node 3 -> node 5
        assert (6, 4) in edge_set  # node 4 -> node 6
        assert (6, 5) in edge_set  # node 5 -> node 6

    def test_doc_example_second_tree_topology(self):
        """In the second tree [0.6, 1.0), node 0 should be a child of node 5 (not 4)."""
        ts = _build_doc_example_ts()
        intervals = list(ts.trees(1.0))
        left, right, edges = intervals[1]
        assert left == pytest.approx(0.6)
        assert right == pytest.approx(1.0)
        edge_set = set(edges)
        # Node 0 is now a child of node 5
        assert (5, 0) in edge_set
        # Node 0 should NOT be a child of node 4
        assert (4, 0) not in edge_set
        # Node 1 is still a child of node 4
        assert (4, 1) in edge_set
        # Nodes 2, 3 still children of 5
        assert (5, 2) in edge_set
        assert (5, 3) in edge_set

    def test_empty_tree_sequence(self):
        """A tree sequence with no edges should yield intervals with no active edges."""
        ts = SimpleTreeSequence()
        ts.add_node(0.0, is_sample=True)
        intervals = list(ts.trees(1.0))
        assert len(intervals) == 1
        left, right, edges = intervals[0]
        assert left == pytest.approx(0.0)
        assert right == pytest.approx(1.0)
        assert len(edges) == 0

    def test_single_edge_whole_sequence(self):
        """A single edge spanning the whole sequence should appear in all trees."""
        ts = SimpleTreeSequence()
        ts.add_node(0.0, is_sample=True)
        ts.add_node(1.0)
        ts.add_edge(0.0, 1.0, 1, 0)
        intervals = list(ts.trees(1.0))
        assert len(intervals) == 1
        _, _, edges = intervals[0]
        assert (1, 0) in edges


# ---------------------------------------------------------------------------
# Code block 4: total_branch_length
# ---------------------------------------------------------------------------

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
        if left <= position < right:
            total += node_times[parent] - node_times[child]
    return total


class TestTotalBranchLength:
    """Tests for the total_branch_length function."""

    def test_simple_two_sample_tree(self):
        """For a tree with 2 samples coalescing at time 1.0, total branch length = 2.0."""
        node_times = {0: 0.0, 1: 0.0, 2: 1.0}
        edges = [
            (0.0, 1.0, 2, 0),
            (0.0, 1.0, 2, 1),
        ]
        tbl = total_branch_length(node_times, edges, 0.5)
        assert tbl == pytest.approx(2.0)

    def test_position_outside_edge_range(self):
        """Querying a position outside all edge ranges should give 0."""
        node_times = {0: 0.0, 1: 0.0, 2: 1.0}
        edges = [
            (0.0, 0.5, 2, 0),
            (0.0, 0.5, 2, 1),
        ]
        tbl = total_branch_length(node_times, edges, 0.75)
        assert tbl == pytest.approx(0.0)

    def test_no_edges(self):
        """With no edges, total branch length should be 0."""
        tbl = total_branch_length({}, [], 0.5)
        assert tbl == pytest.approx(0.0)

    def test_doc_example_tree_sequence(self):
        """Test with the documented SimpleTreeSequence example."""
        # Node times from the doc example
        node_times = {
            0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,  # samples
            4: 0.5, 5: 0.8, 6: 1.2              # internal nodes
        }
        edges = [
            (0.0, 0.6, 4, 0),
            (0.0, 1.0, 4, 1),
            (0.0, 1.0, 5, 2),
            (0.0, 1.0, 5, 3),
            (0.0, 0.6, 6, 4),
            (0.0, 1.0, 6, 5),
            (0.6, 1.0, 5, 0),
        ]

        # First tree at position 0.3:
        # Branches: 0->4 (0.5), 1->4 (0.5), 2->5 (0.8), 3->5 (0.8), 4->6 (0.7), 5->6 (0.4)
        # Total = 0.5 + 0.5 + 0.8 + 0.8 + 0.7 + 0.4 = 3.7
        tbl_first = total_branch_length(node_times, edges, 0.3)
        assert tbl_first == pytest.approx(3.7, abs=1e-10)

    def test_doc_example_second_tree(self):
        """Test the second marginal tree at position 0.8 in the doc example."""
        node_times = {
            0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
            4: 0.5, 5: 0.8, 6: 1.2
        }
        edges = [
            (0.0, 0.6, 4, 0),
            (0.0, 1.0, 4, 1),
            (0.0, 1.0, 5, 2),
            (0.0, 1.0, 5, 3),
            (0.0, 0.6, 6, 4),
            (0.0, 1.0, 6, 5),
            (0.6, 1.0, 5, 0),
        ]

        # Second tree at position 0.8:
        # Active edges: 1->4 (0.5), 2->5 (0.8), 3->5 (0.8), 5->6 (0.4), 0->5 (0.8)
        # Note: 0->4 is NOT active (only [0.0, 0.6)), 4->6 is NOT active (only [0.0, 0.6))
        # Total = 0.5 + 0.8 + 0.8 + 0.4 + 0.8 = 3.3
        tbl_second = total_branch_length(node_times, edges, 0.8)
        assert tbl_second == pytest.approx(3.3, abs=1e-10)

    def test_branch_length_nonnegative(self):
        """Total branch length should never be negative (parent times >= child times)."""
        node_times = {0: 0.0, 1: 0.0, 2: 0.3, 3: 0.7, 4: 1.5}
        edges = [
            (0.0, 1.0, 2, 0),
            (0.0, 1.0, 2, 1),
            (0.0, 1.0, 3, 2),
            (0.0, 1.0, 4, 3),
        ]
        tbl = total_branch_length(node_times, edges, 0.5)
        assert tbl >= 0.0

    def test_cherry_tree_known_value(self):
        """A symmetric tree ((0,1),(2,3)) coalescing at t=1 then t=2.

        Branches: 0->A (1.0), 1->A (1.0), 2->B (1.0), 3->B (1.0), A->R (1.0), B->R (1.0)
        Total = 6.0
        """
        node_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0, 6: 2.0}
        edges = [
            (0.0, 1.0, 4, 0),
            (0.0, 1.0, 4, 1),
            (0.0, 1.0, 5, 2),
            (0.0, 1.0, 5, 3),
            (0.0, 1.0, 6, 4),
            (0.0, 1.0, 6, 5),
        ]
        tbl = total_branch_length(node_times, edges, 0.5)
        assert tbl == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Integration tests: combining multiple code blocks
# ---------------------------------------------------------------------------

class TestArgIntegration:
    """Integration tests combining simulate_arg with extract_marginal_trees."""

    def test_simulate_then_extract(self):
        """Simulating an ARG and extracting breakpoints should work end-to-end."""
        np.random.seed(50)
        coal, recomb = simulate_arg(5, rho=5.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        assert len(bp) >= 2
        assert bp[0] == 0.0
        assert bp[-1] == 1.0

    def test_zero_rho_produces_single_tree(self):
        """With rho=0, we should get exactly one marginal tree."""
        np.random.seed(51)
        coal, recomb = simulate_arg(10, rho=0.0)
        bp = extract_marginal_trees(coal, recomb, 1.0)
        assert len(bp) == 2  # [0.0, 1.0]

    def test_all_coalescence_times_before_recombination_check(self):
        """All events should have positive times, and the simulation should terminate."""
        np.random.seed(52)
        coal, recomb = simulate_arg(8, rho=8.0)
        all_times = [t for t, _, _, _ in coal] + [t for t, _, _, _, _ in recomb]
        assert all(t > 0 for t in all_times)

    def test_large_rho_many_recombinations(self):
        """With very large rho, expect many recombination events."""
        np.random.seed(53)
        _, recomb = simulate_arg(5, rho=50.0)
        # With rho=50 and n=5, we expect many recombinations
        assert len(recomb) > 0

    def test_arg_event_labels_are_unique(self):
        """All parent and new lineage labels across all events should be unique."""
        np.random.seed(54)
        coal, recomb = simulate_arg(5, rho=5.0)
        all_new_labels = []
        for _, _, _, parent in coal:
            all_new_labels.append(parent)
        for _, _, _, left_label, right_label in recomb:
            all_new_labels.append(left_label)
            all_new_labels.append(right_label)
        assert len(all_new_labels) == len(set(all_new_labels))
