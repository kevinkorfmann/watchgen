"""Tests for Python code examples in docs/prerequisites/args.rst.

Each test re-defines the function/class from the RST documentation, then verifies
correctness via structural properties, mathematical invariants, or known behaviors.
"""

import numpy as np
import pytest
import msprime


# ---------------------------------------------------------------------------
# Code block 1: simulate_arg
# ---------------------------------------------------------------------------

def simulate_arg(n, rho, seq_length=1.0, Ne=10_000, random_seed=None):
    """Simulate a haploid sample under the coalescent with recombination.

    Parameters
    ----------
    n : int
        Number of samples.
    rho : float
        Population-scaled recombination rate (4*Ne*r) for the whole region.
    seq_length : float
        Length of the continuous genomic region.
    Ne : float
        Diploid effective population size.
    random_seed : int or None
        Seed passed to msprime.

    Returns
    -------
    tskit.TreeSequence
        Correlated marginal genealogies. Node times are in generations.
    """
    if n < 2 or rho < 0 or seq_length <= 0 or Ne <= 0:
        raise ValueError("require n >= 2, rho >= 0, seq_length > 0, and Ne > 0")
    recombination_rate = rho / (4 * Ne * seq_length)
    return msprime.sim_ancestry(
        samples=n,
        ploidy=1,
        population_size=Ne,
        sequence_length=seq_length,
        recombination_rate=recombination_rate,
        discrete_genome=False,
        record_full_arg=True,
        random_seed=random_seed,
    )


class TestSimulateArg:
    """Tests for the simulate_arg function."""

    def test_returns_tree_sequence_with_requested_samples(self):
        ts = simulate_arg(5, rho=5.0, random_seed=1)
        assert ts.num_samples == 5
        assert ts.sequence_length == pytest.approx(1.0)

    def test_no_recombination_when_rho_zero(self):
        """With rho=0, the tree sequence contains one marginal tree."""
        ts = simulate_arg(6, rho=0.0, random_seed=2)
        assert ts.num_trees == 1

    def test_parent_times_are_older(self):
        ts = simulate_arg(5, rho=5.0, random_seed=3)
        for edge in ts.edges():
            assert ts.node(edge.parent).time > ts.node(edge.child).time

    def test_breakpoints_within_sequence(self):
        """Tree intervals should stay within [0, seq_length]."""
        ts = simulate_arg(5, rho=10.0, seq_length=2.0, random_seed=4)
        for tree in ts.trees():
            assert 0 <= tree.interval.left < tree.interval.right <= 2.0

    def test_more_recombination_with_higher_rho(self):
        """Higher rho should generally produce more recombination events.

        This is a statistical test over many replicates.
        """
        low = [simulate_arg(5, 0.5, random_seed=j).num_trees for j in range(1, 31)]
        high = [simulate_arg(5, 10, random_seed=j).num_trees for j in range(31, 61)]
        assert np.mean(high) > np.mean(low)

    def test_terminates_with_two_samples(self):
        """The simplest valid sample should produce rooted marginal trees."""
        ts = simulate_arg(2, rho=5.0, random_seed=5)
        assert all(tree.num_roots == 1 for tree in ts.trees())

    def test_custom_seq_length(self):
        ts = simulate_arg(4, rho=3.0, seq_length=10.0, random_seed=6)
        assert ts.sequence_length == pytest.approx(10.0)

    def test_doc_example_runs(self):
        ts = simulate_arg(n=5, rho=5.0, random_seed=42)
        assert ts.num_samples == 5
        assert list(ts.breakpoints())[0] == 0
        assert list(ts.breakpoints())[-1] == 1

    @pytest.mark.parametrize("kwargs", [
        {"n": 1, "rho": 1}, {"n": 2, "rho": -1},
        {"n": 2, "rho": 1, "seq_length": 0}, {"n": 2, "rho": 1, "Ne": 0},
    ])
    def test_rejects_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            simulate_arg(**kwargs)


# ---------------------------------------------------------------------------
# Code block 2: extract_tree_intervals
# ---------------------------------------------------------------------------

def extract_tree_intervals(ts):
    """Return intervals on which the represented marginal tree is constant.

    These are tskit's actual tree intervals, not inferred directly from a
    raw list of recombination events.
    """
    return [(tree.interval.left, tree.interval.right) for tree in ts.trees()]


class TestExtractTreeIntervals:

    def test_always_includes_endpoints(self):
        """Breakpoints should always include 0.0 and seq_length."""
        ts = simulate_arg(5, rho=5.0, random_seed=10)
        intervals = extract_tree_intervals(ts)
        assert intervals[0][0] == 0.0
        assert intervals[-1][1] == 1.0

    def test_sorted_breakpoints(self):
        ts = simulate_arg(5, rho=10.0, random_seed=11)
        intervals = extract_tree_intervals(ts)
        for previous, current in zip(intervals, intervals[1:]):
            assert previous[1] == pytest.approx(current[0])

    def test_no_recombination_gives_one_tree(self):
        ts = simulate_arg(5, rho=0.0, random_seed=12)
        assert extract_tree_intervals(ts) == [(0.0, 1.0)]

    def test_works_with_custom_seq_length(self):
        ts = simulate_arg(4, rho=3.0, seq_length=5.0, random_seed=14)
        intervals = extract_tree_intervals(ts)
        assert intervals[0][0] == 0.0
        assert intervals[-1][1] == 5.0

    def test_doc_example_runs(self):
        ts = simulate_arg(n=5, rho=5.0, random_seed=42)
        assert len(extract_tree_intervals(ts)) == ts.num_trees


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

    # Left tree: ((0,1),(2,3)); right tree: ((0,2),(1,3)).
    ts.add_edge(0.0, 1.0, 4, 0)
    ts.add_edge(0.0, 0.6, 4, 1)
    ts.add_edge(0.6, 1.0, 4, 2)
    ts.add_edge(0.0, 0.6, 5, 2)
    ts.add_edge(0.6, 1.0, 5, 1)
    ts.add_edge(0.0, 1.0, 5, 3)
    ts.add_edge(0.0, 1.0, 6, 4)
    ts.add_edge(0.0, 1.0, 6, 5)

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
        """The documented example should produce exactly 2 tree intervals."""
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
        """In the second tree, samples 0 and 2 should be children of node 4."""
        ts = _build_doc_example_ts()
        intervals = list(ts.trees(1.0))
        left, right, edges = intervals[1]
        assert left == pytest.approx(0.6)
        assert right == pytest.approx(1.0)
        edge_set = set(edges)
        assert (4, 0) in edge_set
        assert (4, 2) in edge_set
        assert (5, 1) in edge_set
        assert (5, 3) in edge_set
        assert len(edge_set) == 6

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
            (0.0, 1.0, 4, 0),
            (0.0, 0.6, 4, 1),
            (0.6, 1.0, 4, 2),
            (0.0, 0.6, 5, 2),
            (0.6, 1.0, 5, 1),
            (0.0, 1.0, 5, 3),
            (0.0, 1.0, 6, 4),
            (0.0, 1.0, 6, 5),
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
            (0.0, 1.0, 4, 0),
            (0.0, 0.6, 4, 1),
            (0.6, 1.0, 4, 2),
            (0.0, 0.6, 5, 2),
            (0.6, 1.0, 5, 1),
            (0.0, 1.0, 5, 3),
            (0.0, 1.0, 6, 4),
            (0.0, 1.0, 6, 5),
        ]

        # Second tree at position 0.8:
        # Topology ((0,2),(1,3)); total = 0.5 + 0.5 + 0.8 + 0.8 + 0.7 + 0.4 = 3.7
        tbl_second = total_branch_length(node_times, edges, 0.8)
        assert tbl_second == pytest.approx(3.7, abs=1e-10)

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
    """Integration tests combining simulation and tree iteration."""

    def test_simulate_then_extract(self):
        ts = simulate_arg(5, rho=5.0, random_seed=50)
        intervals = extract_tree_intervals(ts)
        assert intervals[0][0] == 0.0
        assert intervals[-1][1] == 1.0

    def test_zero_rho_produces_single_tree(self):
        """With rho=0, we should get exactly one marginal tree."""
        ts = simulate_arg(10, rho=0.0, random_seed=51)
        assert extract_tree_intervals(ts) == [(0.0, 1.0)]

    def test_all_marginal_trees_are_rooted(self):
        ts = simulate_arg(8, rho=8.0, random_seed=52)
        assert all(tree.num_roots == 1 for tree in ts.trees())

    def test_large_rho_many_recombinations(self):
        """This seeded high-rho simulation has multiple tree intervals."""
        ts = simulate_arg(5, rho=10.0, random_seed=53)
        assert ts.num_trees > 1

    def test_node_ids_are_unique(self):
        ts = simulate_arg(5, rho=5.0, random_seed=54)
        ids = [node.id for node in ts.nodes()]
        assert ids == list(range(ts.num_nodes))
