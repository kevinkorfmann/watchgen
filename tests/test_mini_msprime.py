"""
Tests for watchgen.mini_msprime module.

All functions and classes are imported from the module -- none are redefined here.
Tests verify mathematical correctness of coalescent simulation, data structures,
demographic models, and mutation placement algorithms.
"""

import numpy as np
import math
import dataclasses
import pytest

from watchgen.mini_msprime import (
    # Coalescent process
    simulate_coalescence_time_discrete,
    simulate_coalescence_time_continuous,
    simulate_coalescent,
    expected_tmrca,
    expected_total_branch_length,
    exponential_race,
    merge_segments,
    pick_random_breakpoint,
    split_at_breakpoint,
    coalescent_with_recombination_simple,
    coalescent_waiting_time_constant,
    coalescent_waiting_time_growth,
    gene_conversion_event,
    # Segments & Fenwick Tree
    Segment,
    split_segment,
    Lineage,
    FenwickTree,
    RateMap,
    SegmentPool,
    # Hudson's Algorithm
    INFINITY,
    MinimalSimulator,
    # Demographics
    Population,
    bottleneck_event,
    migration_event,
    mass_migration_event,
    DemographicEventQueue,
    simulate_with_demographics,
    dtwf_generation,
    # Mutations
    simulate_mutations_infinite_sites,
    expected_segregating_sites,
    watterson_estimator,
    compute_sfs,
    expected_sfs,
    MatrixMutationModel,
    MutationRateMap,
    find_root,
    place_mutations_on_tree,
    build_genotype_matrix,
    get_descendants,
    # Solution functions
    simulate_island_coalescence,
    simulate_dtwf_tmrca,
    simulate_coalescent_tmrca,
)


# ============================================================================
# Tests for coalescent.rst functions
# ============================================================================

class TestExpectedTMRCA:
    def test_two_samples(self):
        """E[T_MRCA] for 2 samples = 2*(1-1/2) = 1."""
        assert np.isclose(expected_tmrca(2), 1.0)

    def test_large_n(self):
        """E[T_MRCA] approaches 2 for large n."""
        assert np.isclose(expected_tmrca(1000), 2.0, atol=0.01)

    def test_monotone_increasing(self):
        """E[T_MRCA] should increase with n."""
        vals = [expected_tmrca(n) for n in [2, 5, 10, 50, 100]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_formula(self):
        for n in [3, 7, 20]:
            assert np.isclose(expected_tmrca(n), 2 * (1 - 1.0 / n))


class TestExpectedTotalBranchLength:
    def test_two_samples(self):
        """For n=2, total length = 2 * H_1 = 2."""
        assert np.isclose(expected_total_branch_length(2), 2.0)

    def test_increases_with_n(self):
        vals = [expected_total_branch_length(n) for n in [2, 5, 10, 50]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]

    def test_formula(self):
        for n in [3, 5, 10]:
            harmonic = sum(1.0 / k for k in range(1, n))
            assert np.isclose(expected_total_branch_length(n), 2 * harmonic)


class TestSimulateCoalescent:
    def test_correct_number_of_events(self):
        """n samples should produce n-1 coalescence events."""
        for n in [2, 5, 10]:
            results = simulate_coalescent(n, n_replicates=1)
            times, pairs = results[0]
            assert len(times) == n - 1
            assert len(pairs) == n - 1

    def test_times_increasing(self):
        """Coalescence times should be non-decreasing."""
        np.random.seed(42)
        results = simulate_coalescent(10, n_replicates=1)
        times, _ = results[0]
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1]

    def test_mean_tmrca(self):
        """Mean TMRCA should be close to theoretical 2*(1-1/n)."""
        np.random.seed(42)
        n = 5
        n_reps = 5000
        results = simulate_coalescent(n, n_replicates=n_reps)
        tmrcas = [res[0][-1] for res in results]
        mean_tmrca = np.mean(tmrcas)
        expected = expected_tmrca(n)
        assert np.isclose(mean_tmrca, expected, atol=0.1)


class TestExponentialRace:
    def test_always_returns_winner(self):
        np.random.seed(42)
        for _ in range(100):
            w, t = exponential_race(1.0, 2.0, 3.0)
            assert 0 <= w <= 2
            assert t > 0

    def test_winner_probabilities(self):
        """Event i should win with probability rate_i / sum(rates)."""
        np.random.seed(42)
        rates = [1.0, 2.0, 3.0]
        n_reps = 20000
        wins = np.zeros(3)
        for _ in range(n_reps):
            w, _ = exponential_race(*rates)
            wins[w] += 1

        total_rate = sum(rates)
        for i in range(3):
            expected = rates[i] / total_rate
            observed = wins[i] / n_reps
            assert np.isclose(observed, expected, atol=0.02), \
                f"Rate {i}: expected {expected:.3f}, got {observed:.3f}"

    def test_minimum_rate(self):
        """Mean of minimum should be close to 1/sum(rates)."""
        np.random.seed(42)
        rates = [2.0, 3.0, 5.0]
        n_reps = 10000
        min_times = []
        for _ in range(n_reps):
            _, t = exponential_race(*rates)
            min_times.append(t)

        expected_mean = 1.0 / sum(rates)
        observed_mean = np.mean(min_times)
        assert np.isclose(observed_mean, expected_mean, rtol=0.05)

    def test_zero_rate(self):
        """A rate of 0 should never win."""
        np.random.seed(42)
        for _ in range(100):
            w, _ = exponential_race(0.0, 1.0)
            assert w == 1


class TestMergeSegments:
    def test_non_overlapping(self):
        result = merge_segments([(0, 100)], [(200, 300)])
        assert result == [(0, 100), (200, 300)]

    def test_overlapping(self):
        result = merge_segments([(0, 200)], [(100, 300)])
        assert result == [(0, 300)]

    def test_adjacent(self):
        result = merge_segments([(0, 100)], [(100, 200)])
        assert result == [(0, 200)]

    def test_contained(self):
        result = merge_segments([(0, 500)], [(100, 200)])
        assert result == [(0, 500)]

    def test_empty(self):
        result = merge_segments([], [(100, 200)])
        assert result == [(100, 200)]

    def test_multiple_segments(self):
        result = merge_segments([(0, 100), (200, 300)], [(50, 250)])
        assert result == [(0, 300)]


class TestSplitAtBreakpoint:
    def test_split_in_middle(self):
        left, right = split_at_breakpoint([(0, 1000)], 500)
        assert left == [(0, 500)]
        assert right == [(500, 1000)]

    def test_split_with_gap(self):
        segs = [(0, 300), (500, 1000)]
        left, right = split_at_breakpoint(segs, 400)
        assert left == [(0, 300)]
        assert right == [(500, 1000)]

    def test_split_preserves_total_length(self):
        segs = [(100, 500), (700, 1000)]
        bp = 600
        left, right = split_at_breakpoint(segs, bp)
        orig_len = sum(r - l for l, r in segs)
        left_len = sum(r - l for l, r in left)
        right_len = sum(r - l for l, r in right)
        assert np.isclose(orig_len, left_len + right_len)

    def test_split_at_segment_boundary(self):
        segs = [(0, 500), (500, 1000)]
        left, right = split_at_breakpoint(segs, 500)
        assert left == [(0, 500)]
        assert right == [(500, 1000)]


class TestPickRandomBreakpoint:
    def test_within_bounds(self):
        np.random.seed(42)
        segs = [(100, 500), (700, 1000)]
        for _ in range(100):
            bp = pick_random_breakpoint(segs, 1000)
            in_segment = any(l <= bp <= r for l, r in segs)
            assert in_segment, f"Breakpoint {bp} not in any segment"


class TestGeneConversionEvent:
    def test_gc_within_segment(self):
        segs = [(100, 500), (700, 1000)]
        main, tract = gene_conversion_event(segs, 400, 350, 1000)
        assert (100, 400) in main
        assert (750, 1000) in main
        assert (400, 500) in tract
        assert (700, 750) in tract

    def test_gc_outside_segments(self):
        segs = [(100, 200)]
        main, tract = gene_conversion_event(segs, 300, 100, 1000)
        assert main == [(100, 200)]
        assert tract == []

    def test_gc_total_length_preserved(self):
        segs = [(100, 500), (700, 1000)]
        orig_len = sum(r - l for l, r in segs)
        main, tract = gene_conversion_event(segs, 400, 350, 1000)
        main_len = sum(r - l for l, r in main)
        tract_len = sum(r - l for l, r in tract)
        assert np.isclose(orig_len, main_len + tract_len)

    def test_gc_at_edge(self):
        segs = [(0, 1000)]
        main, tract = gene_conversion_event(segs, 900, 200, 1000)
        assert (0, 900) in main
        assert (900, 1000) in tract

    def test_gc_covers_entire_segment(self):
        segs = [(200, 300)]
        main, tract = gene_conversion_event(segs, 100, 500, 1000)
        assert tract == [(200, 300)]
        assert main == []


class TestCoalescentWaitingTimeGrowth:
    def test_constant_is_finite(self):
        np.random.seed(42)
        t = coalescent_waiting_time_growth(5, 1000, 0, 0)
        assert np.isfinite(t)
        assert t > 0

    def test_growth_is_finite(self):
        np.random.seed(42)
        t = coalescent_waiting_time_growth(5, 1000, 0.01, 0)
        assert np.isfinite(t)
        assert t > 0

    def test_growth_gives_shorter_times(self):
        """Positive growth (pop was smaller) -> faster coalescence on average."""
        np.random.seed(42)
        n_reps = 5000
        k = 10
        N0 = 1000

        const_times = [coalescent_waiting_time_constant(k, N0) for _ in range(n_reps)]
        growth_times = [coalescent_waiting_time_growth(k, N0, 0.01, 0) for _ in range(n_reps)]

        mean_const = np.mean(const_times)
        mean_growth = np.mean(growth_times)
        assert mean_growth < mean_const


# ============================================================================
# Tests for demographics.rst functions
# ============================================================================

class TestPopulation:
    def test_constant_size(self):
        pop = Population(start_size=10000)
        assert pop.get_size(0) == 10000
        assert pop.get_size(100) == 10000
        assert pop.get_size(1000) == 10000

    def test_growth(self):
        """Positive growth rate: population was smaller in the past."""
        pop = Population(start_size=10000, growth_rate=0.01)
        assert pop.get_size(0) == 10000
        assert pop.get_size(100) < 10000

    def test_decline(self):
        """Negative growth rate: population was larger in the past."""
        pop = Population(start_size=10000, growth_rate=-0.01)
        assert pop.get_size(0) == 10000
        assert pop.get_size(100) > 10000

    def test_set_size(self):
        pop = Population(start_size=10000)
        pop.set_size(1000, time=500)
        assert pop.get_size(500) == 1000
        assert pop.growth_rate == 0

    def test_set_growth_rate_continuity(self):
        """Setting growth rate should preserve continuity in size."""
        pop = Population(start_size=10000, growth_rate=0.005, start_time=0)
        size_at_100 = pop.get_size(100)
        pop.set_growth_rate(0.01, time=100)
        assert np.isclose(pop.get_size(100), size_at_100)

    def test_exponential_formula(self):
        """N(t) = start_size * exp(-growth_rate * (t - start_time))."""
        pop = Population(start_size=5000, growth_rate=0.002, start_time=50)
        t = 200
        expected = 5000 * np.exp(-0.002 * (200 - 50))
        assert np.isclose(pop.get_size(t), expected)


class TestMassMigrationEvent:
    def test_full_migration(self):
        pops = [Population(start_size=5000), Population(start_size=5000)]
        pops[0].num_ancestors = 20
        pops[1].num_ancestors = 15
        n_moved = mass_migration_event(pops, source=1, dest=0, fraction=1.0)
        assert pops[1].num_ancestors == 0
        assert pops[0].num_ancestors == 35
        assert n_moved == 15

    def test_partial_migration(self):
        pops = [Population(start_size=5000), Population(start_size=5000)]
        pops[0].num_ancestors = 10
        pops[1].num_ancestors = 20
        n_moved = mass_migration_event(pops, source=1, dest=0, fraction=0.5)
        assert pops[1].num_ancestors == 10
        assert pops[0].num_ancestors == 20
        assert n_moved == 10

    def test_conserves_total_lineages(self):
        pops = [Population(), Population(), Population()]
        pops[0].num_ancestors = 10
        pops[1].num_ancestors = 20
        pops[2].num_ancestors = 5
        total_before = sum(p.num_ancestors for p in pops)
        mass_migration_event(pops, source=1, dest=0, fraction=0.5)
        total_after = sum(p.num_ancestors for p in pops)
        assert total_before == total_after


# ============================================================================
# Tests for mutations.rst functions
# ============================================================================

class TestExpectedSegregatingSites:
    def test_formula(self):
        n, theta = 10, 50
        result = expected_segregating_sites(n, theta)
        harmonic = sum(1.0 / k for k in range(1, n))
        assert np.isclose(result, theta * harmonic)

    def test_increases_with_theta(self):
        n = 20
        assert expected_segregating_sites(n, 100) > expected_segregating_sites(n, 50)

    def test_increases_with_n(self):
        theta = 50
        assert expected_segregating_sites(10, theta) < expected_segregating_sites(50, theta)

    def test_two_samples(self):
        """For n=2, H_1 = 1, so E[S] = theta."""
        theta = 42
        assert np.isclose(expected_segregating_sites(2, theta), theta)


class TestWattersonEstimator:
    def test_inverse_of_expected(self):
        """watterson_estimator(E[S], n) should return theta."""
        n, theta = 50, 100
        E_S = expected_segregating_sites(n, theta)
        theta_hat = watterson_estimator(E_S, n)
        assert np.isclose(theta_hat, theta)

    def test_positive(self):
        assert watterson_estimator(100, 20) > 0


class TestExpectedSFS:
    def test_shape(self):
        n, theta = 20, 50
        sfs = expected_sfs(n, theta)
        assert len(sfs) == n - 1

    def test_formula(self):
        n, theta = 10, 100
        sfs = expected_sfs(n, theta)
        for i in range(n - 1):
            assert np.isclose(sfs[i], theta / (i + 1))

    def test_singletons_most_common(self):
        """Singletons (freq 1) should be the most common class."""
        n, theta = 20, 50
        sfs = expected_sfs(n, theta)
        assert sfs[0] == np.max(sfs)

    def test_decreasing(self):
        """SFS should be decreasing."""
        n, theta = 20, 50
        sfs = expected_sfs(n, theta)
        for i in range(len(sfs) - 1):
            assert sfs[i] > sfs[i + 1]

    def test_sum_equals_expected_seg_sites(self):
        """Sum of SFS should equal expected number of segregating sites."""
        n, theta = 20, 50
        sfs = expected_sfs(n, theta)
        E_S = expected_segregating_sites(n, theta)
        assert np.isclose(np.sum(sfs), E_S)


class TestMatrixMutationModel:
    def test_jc_construction(self):
        jc = MatrixMutationModel(
            alleles=['A', 'C', 'G', 'T'],
            root_distribution=[0.25, 0.25, 0.25, 0.25],
            transition_matrix=[
                [0, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 0, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 0, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3, 0],
            ]
        )
        assert len(jc.alleles) == 4

    def test_mutate_changes_state(self):
        """Under JC, mutate should always change the state."""
        np.random.seed(42)
        jc = MatrixMutationModel(
            alleles=['A', 'C', 'G', 'T'],
            root_distribution=[0.25, 0.25, 0.25, 0.25],
            transition_matrix=[
                [0, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 0, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 0, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3, 0],
            ]
        )
        for _ in range(100):
            old_state = 0
            new_state = jc.mutate(old_state)
            assert new_state != old_state

    def test_root_distribution(self):
        np.random.seed(42)
        jc = MatrixMutationModel(
            alleles=['A', 'C', 'G', 'T'],
            root_distribution=[0.25, 0.25, 0.25, 0.25],
            transition_matrix=[
                [0, 1 / 3, 1 / 3, 1 / 3],
                [1 / 3, 0, 1 / 3, 1 / 3],
                [1 / 3, 1 / 3, 0, 1 / 3],
                [1 / 3, 1 / 3, 1 / 3, 0],
            ]
        )
        counts = np.zeros(4)
        n_reps = 10000
        for _ in range(n_reps):
            counts[jc.draw_root_state()] += 1
        for i in range(4):
            assert np.isclose(counts[i] / n_reps, 0.25, atol=0.03)

    def test_invalid_transition_matrix(self):
        """Diagonal not zero should raise."""
        with pytest.raises(AssertionError):
            MatrixMutationModel(
                alleles=['0', '1'],
                root_distribution=[1, 0],
                transition_matrix=[[0.5, 0.5], [0.5, 0.5]]
            )


class TestMutationRateMap:
    def test_uniform_rate(self):
        rate_map = MutationRateMap(
            positions=[0, 10000],
            rates=[1.5e-8]
        )
        assert np.isclose(rate_map.rate_at(5000), 1.5e-8)
        assert np.isclose(rate_map.total_mass(0, 10000), 1.5e-8 * 10000)

    def test_variable_rate(self):
        rate_map = MutationRateMap(
            positions=[0, 4000, 6000, 10000],
            rates=[1.5e-8, 1e-9, 1.5e-8]
        )
        assert np.isclose(rate_map.rate_at(1000), 1.5e-8)
        assert np.isclose(rate_map.rate_at(5000), 1e-9)
        assert np.isclose(rate_map.rate_at(8000), 1.5e-8)

    def test_total_mass_piecewise(self):
        rate_map = MutationRateMap(
            positions=[0, 4000, 6000, 10000],
            rates=[1.5e-8, 1e-9, 1.5e-8]
        )
        expected = 1.5e-8 * 4000 + 1e-9 * 2000 + 1.5e-8 * 4000
        assert np.isclose(rate_map.total_mass(0, 10000), expected)

    def test_partial_interval_mass(self):
        rate_map = MutationRateMap(
            positions=[0, 1000],
            rates=[2e-8]
        )
        assert np.isclose(rate_map.total_mass(0, 500), 2e-8 * 500)


class TestSimulateMutationsInfiniteSites:
    def test_no_mutations_on_zero_length_branch(self):
        """Branches of length 0 should produce no mutations."""
        nodes = [(0, 0), (0, 0), (0, 0)]
        edges = [(0, 1000, 2, 0), (0, 1000, 2, 1)]
        np.random.seed(42)
        muts = simulate_mutations_infinite_sites(edges, nodes, 1000, mu=1e-3)
        assert len(muts) == 0

    def test_mutations_sorted_by_position(self):
        np.random.seed(42)
        nodes = [(0, 0), (0, 0), (0, 0), (0.8, 0), (1.5, 0)]
        edges = [
            (0, 1000, 3, 0),
            (0, 1000, 3, 1),
            (0, 1000, 4, 3),
            (0, 1000, 4, 2),
        ]
        muts = simulate_mutations_infinite_sites(edges, nodes, 1000, mu=1e-3)
        positions = [m[0] for m in muts]
        assert positions == sorted(positions)

    def test_mutation_positions_in_range(self):
        np.random.seed(42)
        nodes = [(0, 0), (0, 0), (1.0, 0)]
        edges = [(100, 500, 2, 0), (100, 500, 2, 1)]
        muts = simulate_mutations_infinite_sites(edges, nodes, 1000, mu=0.01)
        for pos, node, time, anc, der in muts:
            assert 100 <= pos <= 500
            assert 0 <= time <= 1.0

    def test_expected_number_of_mutations(self):
        """Mean mutation count should be close to mu * span * branch_length."""
        np.random.seed(42)
        nodes = [(0, 0), (0, 0), (10.0, 0)]
        edges = [(0, 1000, 2, 0), (0, 1000, 2, 1)]
        mu = 0.001
        total_muts = 0
        n_reps = 1000
        for _ in range(n_reps):
            muts = simulate_mutations_infinite_sites(edges, nodes, 1000, mu=mu)
            total_muts += len(muts)
        mean_muts = total_muts / n_reps
        expected = mu * 1000 * 10 * 2
        assert np.isclose(mean_muts, expected, rtol=0.1)


# ============================================================================
# Tests for segments_and_fenwick.rst functions
# ============================================================================

class TestSegment:
    def test_length(self):
        s = Segment(index=0, left=100, right=500)
        assert s.length == 400

    def test_show_chain(self):
        s1 = Segment(index=0, left=0, right=500, node=3)
        s2 = Segment(index=1, left=800, right=1000, node=3)
        s1.next = s2
        s2.prev = s1
        chain_str = Segment.show_chain(s1)
        assert "[0, 500: node 3]" in chain_str
        assert "[800, 1000: node 3]" in chain_str


class TestSplitSegment:
    def test_split_produces_two_parts(self):
        seg = Segment(index=0, left=100, right=900, node=5)
        left, right = split_segment(seg, 400)
        assert left.left == 100
        assert left.right == 400
        assert right.left == 400
        assert right.right == 900

    def test_split_preserves_total_length(self):
        seg = Segment(index=0, left=0, right=1000, node=0)
        left, right = split_segment(seg, 300)
        assert left.length + right.length == 1000

    def test_split_disconnects_chains(self):
        seg = Segment(index=0, left=0, right=1000, node=0)
        left, right = split_segment(seg, 500)
        assert left.next is None
        assert right.prev is None

    def test_split_preserves_node(self):
        seg = Segment(index=0, left=0, right=1000, node=7)
        left, right = split_segment(seg, 500)
        assert left.node == 7
        assert right.node == 7


class TestLineage:
    def test_total_length_single(self):
        seg = Segment(index=0, left=0, right=1000, node=0)
        lin = Lineage(head=seg, tail=seg)
        assert lin.total_length == 1000

    def test_total_length_multiple(self):
        s1 = Segment(index=0, left=0, right=500, node=0)
        s2 = Segment(index=1, left=800, right=1000, node=0)
        s1.next = s2
        s2.prev = s1
        lin = Lineage(head=s1, tail=s2)
        assert lin.total_length == 700


class TestFenwickTree:
    def test_set_and_get(self):
        ft = FenwickTree(8)
        ft.set_value(3, 42.0)
        assert ft.get_value(3) == 42.0

    def test_cumulative_sum(self):
        ft = FenwickTree(8)
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        for i, v in enumerate(values):
            ft.set_value(i + 1, v)

        expected_cum = np.cumsum(values)
        for i in range(8):
            assert np.isclose(ft.get_cumulative_sum(i + 1), expected_cum[i])

    def test_total(self):
        ft = FenwickTree(8)
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        for i, v in enumerate(values):
            ft.set_value(i + 1, v)
        assert np.isclose(ft.get_total(), sum(values))

    def test_find(self):
        ft = FenwickTree(8)
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        for i, v in enumerate(values):
            ft.set_value(i + 1, v)

        idx = ft.find(15)
        assert idx == 6

    def test_find_boundary(self):
        ft = FenwickTree(4)
        ft.set_value(1, 10)
        ft.set_value(2, 10)
        ft.set_value(3, 10)
        ft.set_value(4, 10)

        assert ft.find(5) == 1
        assert ft.find(15) == 2
        assert ft.find(25) == 3
        assert ft.find(35) == 4

    def test_increment(self):
        ft = FenwickTree(4)
        ft.set_value(1, 5)
        ft.increment(1, 3)
        assert ft.get_value(1) == 8
        assert ft.get_cumulative_sum(1) == 8

    def test_update_value(self):
        ft = FenwickTree(4)
        ft.set_value(2, 10)
        ft.set_value(2, 20)
        assert ft.get_value(2) == 20
        assert ft.get_cumulative_sum(2) == 20

    def test_weighted_selection(self):
        """Verify that find() produces proportional selection."""
        np.random.seed(42)
        ft = FenwickTree(5)
        masses = [10.0, 25.0, 5.0, 30.0, 15.0]
        for i, m in enumerate(masses):
            ft.set_value(i + 1, m)

        counts = np.zeros(5)
        n_reps = 10000
        for _ in range(n_reps):
            total = ft.get_total()
            r = np.random.uniform(0, total)
            idx = ft.find(r)
            counts[idx - 1] += 1

        total_mass = sum(masses)
        for i in range(5):
            expected = masses[i] / total_mass
            observed = counts[i] / n_reps
            assert np.isclose(observed, expected, atol=0.03), \
                f"Segment {i}: expected {expected:.3f}, got {observed:.3f}"


class TestRateMap:
    def test_uniform_rate(self):
        rm = RateMap(positions=[0, 10000], rates=[1e-8])
        assert np.isclose(rm.total_mass, 1e-8 * 10000)
        assert np.isclose(rm.mass_between(0, 5000), 1e-8 * 5000)

    def test_hotspot(self):
        rm = RateMap(
            positions=[0, 5000, 6000, 10000],
            rates=[1e-8, 1e-6, 1e-8]
        )
        hotspot_mass = rm.mass_between(5000, 6000)
        cold_mass_left = rm.mass_between(0, 5000)
        assert hotspot_mass > cold_mass_left

    def test_position_to_mass_roundtrip(self):
        rm = RateMap(positions=[0, 10000], rates=[2e-8])
        pos = 3000
        mass = rm.position_to_mass(pos)
        pos_back = rm.mass_to_position(mass)
        assert np.isclose(pos, pos_back)

    def test_mass_to_position_roundtrip_hotspot(self):
        rm = RateMap(
            positions=[0, 5000, 6000, 10000],
            rates=[1e-8, 1e-6, 1e-8]
        )
        for pos in [1000, 5500, 8000]:
            mass = rm.position_to_mass(pos)
            pos_back = rm.mass_to_position(mass)
            assert np.isclose(pos, pos_back, atol=0.01)

    def test_mass_between_entire_genome(self):
        rm = RateMap(positions=[0, 10000], rates=[1e-8])
        assert np.isclose(rm.mass_between(0, 10000), rm.total_mass)

    def test_mass_between_piecewise(self):
        rm = RateMap(
            positions=[0, 5000, 10000],
            rates=[1e-8, 2e-8]
        )
        expected = 1e-8 * 5000 + 2e-8 * 5000
        assert np.isclose(rm.mass_between(0, 10000), expected)


class TestSegmentPool:
    def test_alloc_and_free(self):
        pool = SegmentPool(10)
        initial_free = len(pool.free_list)
        seg = pool.alloc(left=0, right=500, node=0)
        assert len(pool.free_list) == initial_free - 1
        pool.free(seg)
        assert len(pool.free_list) == initial_free

    def test_alloc_values(self):
        pool = SegmentPool(10)
        seg = pool.alloc(left=100, right=500, node=3)
        assert seg.left == 100
        assert seg.right == 500
        assert seg.node == 3
        assert seg.prev is None
        assert seg.next is None

    def test_copy(self):
        pool = SegmentPool(10)
        s1 = pool.alloc(left=0, right=500, node=0)
        s2 = pool.alloc(left=500, right=1000, node=0)
        s1.next = s2
        s2.prev = s1

        s3 = pool.copy(s1)
        assert s3.left == s1.left
        assert s3.right == s1.right
        assert s3.node == s1.node
        assert s3.next == s2

    def test_exhaustion(self):
        pool = SegmentPool(2)
        pool.alloc()
        pool.alloc()
        with pytest.raises(RuntimeError):
            pool.alloc()

    def test_reuse_after_free(self):
        pool = SegmentPool(1)
        seg = pool.alloc(left=0, right=100, node=0)
        pool.free(seg)
        seg2 = pool.alloc(left=200, right=300, node=1)
        assert seg2.left == 200
        assert seg2.right == 300


# ============================================================================
# Integration tests
# ============================================================================

class TestCoalescentIntegration:
    def test_tmrca_statistics(self):
        """TMRCA statistics should match theoretical values for moderate n."""
        np.random.seed(42)
        n = 10
        n_reps = 3000
        results = simulate_coalescent(n, n_replicates=n_reps)
        tmrcas = [res[0][-1] for res in results]

        mean_tmrca = np.mean(tmrcas)
        expected = expected_tmrca(n)
        assert np.isclose(mean_tmrca, expected, atol=0.1), \
            f"Mean TMRCA {mean_tmrca:.4f} not close to {expected:.4f}"

    def test_total_branch_length_statistics(self):
        """Total branch length should match theoretical value."""
        np.random.seed(42)
        n = 5
        n_reps = 3000
        results = simulate_coalescent(n, n_replicates=n_reps)

        total_lengths = []
        for times, pairs in results:
            total = 0
            for i in range(len(times)):
                k = n - i
                if i == 0:
                    dt = times[0]
                else:
                    dt = times[i] - times[i - 1]
                total += k * dt
            total_lengths.append(total)

        mean_total = np.mean(total_lengths)
        expected = expected_total_branch_length(n)
        assert np.isclose(mean_total, expected, atol=0.2), \
            f"Mean total branch length {mean_total:.4f} not close to {expected:.4f}"

    def test_fenwick_tree_with_segment_chain(self):
        """Combine Fenwick tree with segment operations."""
        ft = FenwickTree(10)
        pool = SegmentPool(10)
        rate = 1e-3

        seg = pool.alloc(left=0, right=1000, node=0)
        ft.set_value(seg.index, rate * seg.length)
        initial_mass = ft.get_total()
        assert np.isclose(initial_mass, rate * 1000)

        # Split segment
        left, right = split_segment(seg, 400)
        right_seg = pool.alloc(left=right.left, right=right.right, node=right.node)
        ft.set_value(left.index, rate * left.length)
        ft.set_value(right_seg.index, rate * right_seg.length)

        # Total mass should be preserved
        assert np.isclose(ft.get_total(), rate * 1000)
