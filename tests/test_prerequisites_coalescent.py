"""Tests for Python code examples in docs/prerequisites/coalescent_theory.rst.

Each test re-defines the function from the RST documentation, then verifies
correctness via mathematical properties, known analytical results, or
simulation-based consistency checks.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Code block 1: wright_fisher_forward
# ---------------------------------------------------------------------------

def wright_fisher_forward(two_N, n_generations):
    """Simulate the Wright-Fisher model forward in time.

    Parameters
    ----------
    two_N : int
        Total number of gene copies (2 * diploid population size).
    n_generations : int
        Number of generations to simulate.

    Returns
    -------
    parent_table : ndarray of shape (n_generations, two_N)
        parent_table[g, i] = parent index of individual i in generation g.
    """
    parent_table = np.zeros((n_generations, two_N), dtype=int)
    for g in range(n_generations):
        parent_table[g] = np.random.randint(0, two_N, size=two_N)
    return parent_table


class TestWrightFisherForward:
    """Tests for the wright_fisher_forward function."""

    def test_output_shape(self):
        """Verify the parent table has the correct shape."""
        np.random.seed(0)
        two_N = 20
        n_gen = 10
        parents = wright_fisher_forward(two_N, n_gen)
        assert parents.shape == (n_gen, two_N)

    def test_parent_indices_in_range(self):
        """Verify all parent indices are valid (between 0 and 2N-1)."""
        np.random.seed(1)
        two_N = 50
        parents = wright_fisher_forward(two_N, 30)
        assert np.all(parents >= 0)
        assert np.all(parents < two_N)

    def test_integer_dtype(self):
        """Verify the parent table contains integers."""
        np.random.seed(2)
        parents = wright_fisher_forward(10, 5)
        assert parents.dtype == int

    def test_reproducibility_with_seed(self):
        """Verify that setting the seed produces identical results."""
        np.random.seed(42)
        p1 = wright_fisher_forward(10, 20)
        np.random.seed(42)
        p2 = wright_fisher_forward(10, 20)
        assert np.array_equal(p1, p2)

    def test_matches_doc_example(self):
        """Verify the documented example (seed=42, two_N=10, 20 gens) runs."""
        np.random.seed(42)
        parents = wright_fisher_forward(10, 20)
        # The doc prints parents[0]; just check it is a valid array of length 10.
        assert len(parents[0]) == 10
        assert np.all(parents[0] >= 0)
        assert np.all(parents[0] < 10)


# ---------------------------------------------------------------------------
# Code block 2: coalescence_time_two_lineages
# ---------------------------------------------------------------------------

def coalescence_time_two_lineages(two_N, n_replicates=10000):
    """Simulate coalescence times for pairs of lineages.

    Each replicate: two lineages independently pick random parents
    each generation until they pick the same one (coalescence).

    Parameters
    ----------
    two_N : int
        Total number of gene copies (2N).
    n_replicates : int
        Number of independent simulations to run.

    Returns
    -------
    times : ndarray
        Coalescence times in generations.
    """
    times = np.zeros(n_replicates)
    for rep in range(n_replicates):
        t = 0
        while True:
            t += 1
            parent1 = np.random.randint(0, two_N)
            parent2 = np.random.randint(0, two_N)
            if parent1 == parent2:
                break
        times[rep] = t
    return times


class TestCoalescenceTimeTwoLineages:
    """Tests for the coalescence_time_two_lineages function."""

    def test_all_positive_times(self):
        """Coalescence times must be at least 1 generation."""
        np.random.seed(10)
        times = coalescence_time_two_lineages(100, n_replicates=500)
        assert np.all(times >= 1)

    def test_mean_close_to_two_N(self):
        """Mean coalescence time should be approximately 2N generations.

        Under a geometric distribution with success probability 1/(2N),
        the expected value is 2N.
        """
        np.random.seed(11)
        two_N = 100
        times = coalescence_time_two_lineages(two_N, n_replicates=20000)
        assert abs(times.mean() - two_N) / two_N < 0.05  # within 5%

    def test_scaled_mean_close_to_one(self):
        """Mean of times/(2N) should be close to 1 (Exp(1) mean)."""
        np.random.seed(12)
        two_N = 100
        times = coalescence_time_two_lineages(two_N, n_replicates=50000)
        scaled = times / two_N
        assert abs(scaled.mean() - 1.0) < 0.05

    def test_scaled_variance_close_to_one(self):
        """Variance of times/(2N) should be close to 1 (Exp(1) variance)."""
        np.random.seed(13)
        two_N = 100
        times = coalescence_time_two_lineages(two_N, n_replicates=50000)
        scaled = times / two_N
        assert abs(scaled.var() - 1.0) < 0.15  # generous tolerance

    def test_matches_doc_example(self):
        """Run the documented example (two_N=100, 50000 replicates)."""
        np.random.seed(42)
        two_N = 100
        times = coalescence_time_two_lineages(two_N, n_replicates=50000)
        scaled_times = times / two_N
        # The doc says mean and variance should be close to 1.000
        assert abs(scaled_times.mean() - 1.0) < 0.05
        assert abs(scaled_times.var() - 1.0) < 0.15


# ---------------------------------------------------------------------------
# Code block 3: simulate_coalescent
# ---------------------------------------------------------------------------

def simulate_coalescent(n):
    """Simulate a coalescent tree for n samples.

    This simulates Kingman's coalescent: starting with n lineages,
    we repeatedly wait an exponential time and merge a random pair,
    until only one lineage (the MRCA) remains.

    Parameters
    ----------
    n : int
        Number of samples (leaf nodes).

    Returns
    -------
    events : list of (time, child1, child2, parent)
        Coalescence events in chronological order (going back in time).
        'time' is in coalescent units (multiples of 2N generations).
    """
    lineages = list(range(n))
    next_label = n
    events = []
    current_time = 0.0

    while len(lineages) > 1:
        k = len(lineages)
        rate = k * (k - 1) / 2

        wait = np.random.exponential(1.0 / rate)
        current_time += wait

        i, j = np.random.choice(len(lineages), size=2, replace=False)
        child1 = lineages[i]
        child2 = lineages[j]

        parent = next_label
        next_label += 1
        events.append((current_time, child1, child2, parent))

        lineages = [l for idx, l in enumerate(lineages)
                    if idx != i and idx != j]
        lineages.append(parent)

    return events


class TestSimulateCoalescent:
    """Tests for the simulate_coalescent function."""

    def test_correct_number_of_events(self):
        """A coalescent tree with n samples has exactly n-1 coalescence events."""
        np.random.seed(20)
        for n in [2, 5, 10, 20]:
            events = simulate_coalescent(n)
            assert len(events) == n - 1

    def test_times_are_increasing(self):
        """Coalescence times must be strictly increasing."""
        np.random.seed(21)
        events = simulate_coalescent(10)
        times = [t for t, _, _, _ in events]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1]

    def test_all_times_positive(self):
        """All coalescence times should be positive."""
        np.random.seed(22)
        events = simulate_coalescent(8)
        for t, _, _, _ in events:
            assert t > 0.0

    def test_parent_labels_unique(self):
        """Parent labels in each event should be unique."""
        np.random.seed(23)
        events = simulate_coalescent(10)
        parents = [p for _, _, _, p in events]
        assert len(parents) == len(set(parents))

    def test_parent_labels_start_at_n(self):
        """Internal node labels should start at n and increment."""
        np.random.seed(24)
        n = 7
        events = simulate_coalescent(n)
        parents = [p for _, _, _, p in events]
        assert parents == list(range(n, n + n - 1))

    def test_children_are_distinct(self):
        """In each event, the two children should be different."""
        np.random.seed(25)
        events = simulate_coalescent(10)
        for _, c1, c2, _ in events:
            assert c1 != c2

    def test_average_tmrca_close_to_formula(self):
        """Average TMRCA over many replicates should be close to 2(1 - 1/n).

        For n=5, expected TMRCA = 2*(1 - 1/5) = 1.6.
        """
        np.random.seed(26)
        n = 5
        n_reps = 5000
        tmrcas = []
        for _ in range(n_reps):
            events = simulate_coalescent(n)
            tmrcas.append(events[-1][0])
        mean_tmrca = np.mean(tmrcas)
        expected = 2 * (1 - 1 / n)
        assert abs(mean_tmrca - expected) / expected < 0.05

    def test_two_samples_tmrca_exponential(self):
        """For n=2, TMRCA ~ Exp(1). Mean should be close to 1."""
        np.random.seed(27)
        n_reps = 10000
        tmrcas = []
        for _ in range(n_reps):
            events = simulate_coalescent(2)
            tmrcas.append(events[-1][0])
        mean_tmrca = np.mean(tmrcas)
        assert abs(mean_tmrca - 1.0) < 0.05

    def test_doc_example_runs(self):
        """Run the documented example with seed=42, n=5."""
        np.random.seed(42)
        events = simulate_coalescent(5)
        assert len(events) == 4
        tmrca = events[-1][0]
        assert tmrca > 0


# ---------------------------------------------------------------------------
# Code block 4: expected_lineages and simulate_lineage_count
# ---------------------------------------------------------------------------

def expected_lineages(t, n):
    """Expected number of lineages at time t for n initial samples.

    Uses the large-n deterministic approximation (Frost & Volz, 2010).

    Parameters
    ----------
    t : float
        Time in coalescent units.
    n : int
        Number of initial samples.

    Returns
    -------
    float
        Expected number of lineages at time t.
    """
    return n / (n + (1 - n) * np.exp(-t / 2))


def simulate_lineage_count(n, t, n_replicates=10000):
    """Estimate E[lineages at time t] by simulation.

    Simulates the coalescent n_replicates times, counts how many
    lineages remain at time t, and returns the average.
    """
    counts = np.zeros(n_replicates)
    for rep in range(n_replicates):
        k = n
        current_time = 0.0
        while k > 1 and current_time < t:
            rate = k * (k - 1) / 2
            wait = np.random.exponential(1.0 / rate)
            if current_time + wait > t:
                break
            current_time += wait
            k -= 1
        counts[rep] = k
    return counts.mean()


class TestExpectedLineages:
    """Tests for the expected_lineages analytical formula."""

    def test_at_time_zero_returns_n(self):
        """At t=0, the number of lineages should be n."""
        for n in [2, 5, 10, 50, 100]:
            assert expected_lineages(0.0, n) == pytest.approx(n, rel=1e-10)

    def test_monotonically_decreasing(self):
        """Expected lineages should decrease over time."""
        n = 50
        times = np.linspace(0, 5, 100)
        values = [expected_lineages(t, n) for t in times]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1]

    def test_large_time_approaches_one(self):
        """As t -> infinity, expected lineages should approach 1 (the MRCA)."""
        n = 50
        val = expected_lineages(100.0, n)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_always_at_least_one(self):
        """Expected lineages should never go below 1."""
        n = 10
        for t in [0, 0.5, 1, 2, 5, 10, 50]:
            assert expected_lineages(t, n) >= 1.0


class TestSimulateLineageCount:
    """Tests for simulate_lineage_count and its agreement with expected_lineages."""

    def test_at_time_zero(self):
        """At t=0, all n lineages should still be present."""
        np.random.seed(30)
        result = simulate_lineage_count(20, 0.0, n_replicates=100)
        assert result == pytest.approx(20.0, abs=0.01)

    def test_agreement_with_formula(self):
        """Simulated lineage counts should agree with the analytical formula.

        Tests several time points for n=50.
        """
        np.random.seed(31)
        n = 50
        for t in [0.05, 0.1, 0.5, 1.0, 2.0]:
            approx = expected_lineages(t, n)
            simulated = simulate_lineage_count(n, t, n_replicates=5000)
            # Allow generous tolerance for stochastic simulation
            assert abs(simulated - approx) / max(approx, 1) < 0.15

    def test_doc_example_runs(self):
        """Run the documented example: n=50, multiple time points."""
        np.random.seed(42)
        n = 50
        for t in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
            approx = expected_lineages(t, n)
            simulated = simulate_lineage_count(n, t, n_replicates=2000)
            # Just confirm both produce finite positive numbers
            assert approx > 0
            assert simulated > 0


# ---------------------------------------------------------------------------
# Code block 5: add_mutations
# ---------------------------------------------------------------------------

def add_mutations(events, n, theta, seq_length=1):
    """Add mutations to a coalescent tree under the infinite sites model.

    For each branch in the tree, we draw a Poisson number of mutations
    (proportional to the branch length and the mutation rate), and place
    each mutation at a random position along the genome.

    Parameters
    ----------
    events : list of (time, child1, child2, parent)
        Coalescence events from simulate_coalescent().
    n : int
        Number of samples (leaf nodes, labeled 0 to n-1).
    theta : float
        Population-scaled mutation rate (4*Ne*mu).
    seq_length : int
        Sequence length in base pairs.

    Returns
    -------
    mutations : list of (position, branch_node, time)
        Each mutation has a genomic position, the node whose branch it
        sits on, and the time at which it occurred.
    """
    node_times = {i: 0.0 for i in range(n)}
    children = {}
    for t, c1, c2, p in events:
        node_times[p] = t
        children[p] = (c1, c2)

    mutations = []
    root = events[-1][3]

    for node in node_times:
        if node == root:
            continue
        for t, c1, c2, p in events:
            if c1 == node or c2 == node:
                branch_length = node_times[p] - node_times[node]
                n_muts = np.random.poisson(theta / 2 * branch_length * seq_length)
                for _ in range(n_muts):
                    pos = np.random.uniform(0, seq_length)
                    mut_time = np.random.uniform(node_times[node], node_times[p])
                    mutations.append((pos, node, mut_time))
                break

    return sorted(mutations)


class TestAddMutations:
    """Tests for the add_mutations function."""

    def test_returns_list(self):
        """The function should return a list."""
        np.random.seed(40)
        events = simulate_coalescent(5)
        muts = add_mutations(events, 5, theta=10, seq_length=100)
        assert isinstance(muts, list)

    def test_no_mutations_with_zero_theta(self):
        """With theta=0, no mutations should occur."""
        np.random.seed(41)
        events = simulate_coalescent(5)
        muts = add_mutations(events, 5, theta=0.0, seq_length=1000)
        assert len(muts) == 0

    def test_mutations_sorted_by_position(self):
        """Mutations should be sorted by position (first element of tuple)."""
        np.random.seed(42)
        events = simulate_coalescent(10)
        muts = add_mutations(events, 10, theta=100, seq_length=1000)
        positions = [m[0] for m in muts]
        assert positions == sorted(positions)

    def test_mutation_positions_in_range(self):
        """All mutation positions should be within [0, seq_length)."""
        np.random.seed(43)
        seq_length = 500
        events = simulate_coalescent(8)
        muts = add_mutations(events, 8, theta=50, seq_length=seq_length)
        for pos, _, _ in muts:
            assert 0 <= pos <= seq_length

    def test_mutation_times_positive(self):
        """All mutation times should be positive (they occur on branches above time 0)."""
        np.random.seed(44)
        events = simulate_coalescent(6)
        muts = add_mutations(events, 6, theta=80, seq_length=200)
        for _, _, mt in muts:
            assert mt >= 0.0

    def test_mutation_times_within_tree_height(self):
        """Mutation times should not exceed the TMRCA."""
        np.random.seed(45)
        events = simulate_coalescent(10)
        tmrca = events[-1][0]
        muts = add_mutations(events, 10, theta=100, seq_length=1000)
        for _, _, mt in muts:
            assert mt <= tmrca + 1e-10

    def test_expected_mutation_count(self):
        """Average number of mutations should be close to (theta/2) * total_branch_length * seq_length.

        For n=10, total branch length has E[TBL] = 2 * sum(1/k for k=1..n-1).
        We check the average mutation count over many replicates.
        """
        np.random.seed(46)
        n = 10
        theta = 10.0
        seq_length = 100
        n_reps = 2000
        mut_counts = []
        for _ in range(n_reps):
            events = simulate_coalescent(n)
            muts = add_mutations(events, n, theta=theta, seq_length=seq_length)
            mut_counts.append(len(muts))

        # Expected total branch length = 2 * H_{n-1} where H_k = sum(1/i, i=1..k)
        # For n=10: H_9 = 1 + 1/2 + ... + 1/9 ~ 2.8289
        # E[TBL] ~ 2 * 2.8289 ~ 5.6579
        # E[mutations] = theta/2 * E[TBL] * seq_length = 5 * 5.6579 * 100 = 2828.9
        harmonic = sum(1.0 / k for k in range(1, n))
        expected_tbl = 2 * harmonic
        expected_muts = theta / 2 * expected_tbl * seq_length
        mean_muts = np.mean(mut_counts)
        assert abs(mean_muts - expected_muts) / expected_muts < 0.1

    def test_doc_example_runs(self):
        """Run the documented example: seed=42, n=10, theta=100, seq_length=1000."""
        np.random.seed(42)
        events = simulate_coalescent(10)
        mutations = add_mutations(events, 10, theta=100, seq_length=1000)
        # The doc says it prints "Number of segregating sites: {len(mutations)}"
        assert len(mutations) >= 0
        assert isinstance(mutations, list)
