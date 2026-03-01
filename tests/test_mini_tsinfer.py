"""
Tests for watchgen.mini_tsinfer module.

Imports all functions from watchgen.mini_tsinfer and adapts test logic from
tests/test_timepieces_tsinfer.py.

Covers:
- ancestor_generation: select_inference_sites, compute_ancestor_times,
    get_focal_samples, build_ancestor, generate_ancestors,
    group_ancestors_by_time, add_ultimate_ancestor, verify_ancestors
- ancestor_matching: matching_order, build_reference_panel, path_compress,
    TreeSequenceBuilder
- copying_model: compute_recombination_probs, compute_mismatch_probs,
    viterbi_ls, viterbi_ls_with_noncopy, path_to_edges, find_breakpoints
- sample_matching: fitch_parsimony, remove_virtual_root, erase_flanks,
    simplify_tree_sequence
"""

import numpy as np

from watchgen.mini_tsinfer import (
    # Constants
    NONCOPY,
    PC_TIME_EPSILON,
    # Gear 1: Ancestor Generation
    select_inference_sites,
    compute_ancestor_times,
    get_focal_samples,
    build_ancestor,
    generate_ancestors,
    group_ancestors_by_time,
    add_ultimate_ancestor,
    verify_ancestors,
    # Gear 2: The Copying Model
    compute_recombination_probs,
    compute_mismatch_probs,
    viterbi_ls,
    viterbi_ls_with_noncopy,
    path_to_edges,
    find_breakpoints,
    # Gear 3: Ancestor Matching
    matching_order,
    build_reference_panel,
    path_compress,
    TreeSequenceBuilder,
    match_ancestors,
    verify_ancestor_tree,
    # Gear 4: Sample Matching & Post-Processing
    match_samples,
    fitch_parsimony,
    remove_virtual_root,
    erase_flanks,
    simplify_tree_sequence,
    tsinfer_pipeline,
    verify_pipeline,
)


# ============================================================================
# Tests for Gear 1: Ancestor Generation
# ============================================================================

class TestSelectInferenceSites:
    """Tests for site selection."""

    def test_exclude_fixed_derived(self):
        """Sites where all samples are derived should be excluded."""
        D = np.ones((10, 3), dtype=int)
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert len(inf) == 0

    def test_exclude_fixed_ancestral(self):
        """Sites where all samples are ancestral should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert len(inf) == 0

    def test_exclude_singleton(self):
        """Sites with only one derived copy should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        D[0, 0] = 1  # singleton
        D[0, 1] = 1; D[1, 1] = 1  # doubleton (valid)
        D[0, 2] = 1  # singleton
        ancestral_known = np.ones(3, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        assert 0 not in inf
        assert 2 not in inf
        assert 1 in inf

    def test_exclude_unknown_ancestral(self):
        """Sites with unknown ancestral allele should be excluded."""
        D = np.zeros((10, 3), dtype=int)
        D[:5, :] = 1  # Half derived
        ancestral_known = np.array([True, False, True])
        inf, _ = select_inference_sites(D, ancestral_known)
        assert 1 not in inf

    def test_partition(self):
        """Inference and non-inference sites should partition all sites."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, non_inf = select_inference_sites(D, ancestral_known)
        assert len(inf) + len(non_inf) == 10
        assert len(set(inf) & set(non_inf)) == 0

    def test_valid_inference_sites(self):
        """All selected inference sites should meet the criteria."""
        np.random.seed(42)
        n, m = 20, 15
        D = np.random.binomial(1, 0.3, size=(n, m))
        ancestral_known = np.ones(m, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        for j in inf:
            derived_count = D[:, j].sum()
            ancestral_count = n - derived_count
            assert derived_count >= 2
            assert ancestral_count >= 1
            assert len(np.unique(D[:, j])) == 2


class TestComputeAncestorTimes:
    """Tests for ancestor time computation."""

    def test_range(self):
        """Times should be in (0, 1) for valid inference sites."""
        np.random.seed(42)
        n, m = 20, 10
        D = np.random.binomial(1, 0.3, size=(n, m))
        ancestral_known = np.ones(m, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            # Inference sites have >= 2 derived and >= 1 ancestral
            for t in times:
                assert 0 < t < 1

    def test_equals_derived_frequency(self):
        """Times should equal the derived allele frequency."""
        D = np.zeros((10, 3), dtype=int)
        D[:5, 0] = 1  # freq 0.5
        D[:3, 1] = 1  # freq 0.3
        D[:8, 2] = 1  # freq 0.8
        inf = np.array([0, 1, 2])
        times = compute_ancestor_times(D, inf)
        assert np.isclose(times[0], 0.5)
        assert np.isclose(times[1], 0.3)
        assert np.isclose(times[2], 0.8)


class TestGetFocalSamples:
    """Tests for focal sample identification."""

    def test_correct_samples(self):
        """Should return indices where D[:, j] == 1."""
        D = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 0, 1]])
        focal = get_focal_samples(D, 0)
        assert set(focal) == {1, 2}

    def test_all_derived(self):
        """If all samples are derived, all should be focal."""
        D = np.ones((5, 3), dtype=int)
        focal = get_focal_samples(D, 0)
        assert len(focal) == 5

    def test_none_derived(self):
        """If no samples are derived, result should be empty."""
        D = np.zeros((5, 3), dtype=int)
        focal = get_focal_samples(D, 0)
        assert len(focal) == 0


class TestBuildAncestor:
    """Tests for ancestor construction."""

    def test_focal_site_is_derived(self):
        """The ancestor should carry derived allele at its focal site."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            anc = build_ancestor(D, inf, times, 0)
            focal_in_hap = anc['focal'] - anc['start']
            assert anc['haplotype'][focal_in_hap] == 1

    def test_haplotype_is_binary(self):
        """Ancestor haplotype should contain only 0s and 1s."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            anc = build_ancestor(D, inf, times, 0)
            assert set(anc['haplotype']).issubset({0, 1})

    def test_span_within_bounds(self):
        """Start and end should be within valid range."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        inf, _ = select_inference_sites(D, ancestral_known)
        if len(inf) > 0:
            times = compute_ancestor_times(D, inf)
            for idx in range(len(inf)):
                anc = build_ancestor(D, inf, times, idx)
                assert 0 <= anc['start'] <= anc['focal']
                assert anc['focal'] < anc['end'] <= len(inf)


class TestGenerateAncestors:
    """Tests for the full ancestor generation algorithm."""

    def test_sorted_by_time_descending(self):
        """Ancestors should be sorted by time (oldest first)."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        ancestors, _ = generate_ancestors(D, ancestral_known)
        for i in range(len(ancestors) - 1):
            assert ancestors[i]['time'] >= ancestors[i + 1]['time']

    def test_one_ancestor_per_inference_site(self):
        """Should generate one ancestor per inference site."""
        np.random.seed(42)
        D = np.random.binomial(1, 0.3, size=(20, 10))
        ancestral_known = np.ones(10, dtype=bool)
        ancestors, inf_sites = generate_ancestors(D, ancestral_known)
        assert len(ancestors) == len(inf_sites)


class TestGroupAncestorsByTime:
    """Tests for ancestor grouping."""

    def test_grouping(self):
        """Ancestors at the same time should be in the same group."""
        ancestors = [
            {'time': 0.8, 'focal': 1},
            {'time': 0.8, 'focal': 2},
            {'time': 0.5, 'focal': 3},
            {'time': 0.3, 'focal': 4},
            {'time': 0.3, 'focal': 5},
            {'time': 0.3, 'focal': 6},
        ]
        groups = group_ancestors_by_time(ancestors)
        assert len(groups) == 3
        # Oldest first
        assert groups[0][0] == 0.8
        assert len(groups[0][1]) == 2
        assert groups[1][0] == 0.5
        assert len(groups[1][1]) == 1
        assert groups[2][0] == 0.3
        assert len(groups[2][1]) == 3

    def test_all_ancestors_accounted_for(self):
        """Total ancestors across groups should equal input count."""
        ancestors = [
            {'time': 0.8, 'focal': 1},
            {'time': 0.5, 'focal': 2},
            {'time': 0.5, 'focal': 3},
        ]
        groups = group_ancestors_by_time(ancestors)
        total = sum(len(g) for _, g in groups)
        assert total == len(ancestors)


class TestAddUltimateAncestor:
    """Tests for the ultimate ancestor."""

    def test_time_is_one(self):
        """Ultimate ancestor should be at time 1.0."""
        ancestors = [{'time': 0.5, 'focal': 0}]
        result = add_ultimate_ancestor(ancestors, 10)
        assert result[0]['time'] == 1.0

    def test_all_zeros(self):
        """Ultimate ancestor should carry ancestral allele everywhere."""
        result = add_ultimate_ancestor([], 10)
        assert np.all(result[0]['haplotype'] == 0)

    def test_full_span(self):
        """Ultimate ancestor should span all inference sites."""
        result = add_ultimate_ancestor([], 15)
        assert result[0]['start'] == 0
        assert result[0]['end'] == 15

    def test_prepended(self):
        """Ultimate ancestor should be first in the list."""
        ancestors = [{'time': 0.5, 'focal': 0}]
        result = add_ultimate_ancestor(ancestors, 10)
        assert len(result) == 2
        assert result[0]['focal'] == -1


# ============================================================================
# Tests for Gear 2: The Copying Model
# ============================================================================

class TestComputeRecombinationProbs:
    """Tests for recombination probability computation."""

    def test_first_site_zero(self):
        """Recombination at site 0 should be 0."""
        positions = np.array([0.0, 1000.0, 2000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[0] == 0.0

    def test_positive_for_positive_distance(self):
        """Recombination prob should be positive for positive distance."""
        positions = np.array([0.0, 1000.0, 2000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[1] > 0
        assert rho[2] > 0

    def test_bounded_below_one(self):
        """Recombination probability should be < 1."""
        positions = np.array([0.0, 1000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[1] < 1.0

    def test_uniform_spacing_equal_probs(self):
        """Uniformly spaced sites should have equal recombination probs."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert np.allclose(rho[1:], rho[1])

    def test_larger_distance_higher_prob(self):
        """Larger distance should give higher recombination probability."""
        positions = np.array([0.0, 1000.0, 5000.0])
        rho = compute_recombination_probs(positions, 1e-4, 50)
        assert rho[2] > rho[1]


class TestComputeMismatchProbs:
    """Tests for mismatch probability computation."""

    def test_first_site_equals_second(self):
        """Mismatch at site 0 should be set to mu[1]."""
        positions = np.array([0.0, 1000.0, 2000.0])
        mu = compute_mismatch_probs(positions, 1e-4, 1.0, 50)
        assert mu[0] == mu[1]

    def test_positive(self):
        """Mismatch probabilities should be positive."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        mu = compute_mismatch_probs(positions, 1e-4, 1.0, 50)
        assert np.all(mu > 0)

    def test_ratio_effect(self):
        """Higher mismatch ratio should give higher mismatch probs."""
        positions = np.array([0.0, 1000.0])
        mu_low = compute_mismatch_probs(positions, 1e-4, 0.1, 50)
        mu_high = compute_mismatch_probs(positions, 1e-4, 10.0, 50)
        assert mu_high[1] > mu_low[1]


class TestViterbiLS:
    """Tests for the Li & Stephens Viterbi algorithm."""

    def test_perfect_copy(self):
        """Query that is an exact copy of one reference should match it."""
        np.random.seed(42)
        k, m = 5, 20
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = panel[:, 2].copy()  # Exact copy of ref 2
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        # Path should be all 2 (or very close)
        assert np.mean(path == 2) > 0.8

    def test_mosaic_recovery(self):
        """A mosaic query should recover the correct segments."""
        np.random.seed(42)
        k, m = 5, 30
        panel = np.random.binomial(1, 0.3, size=(m, k))
        # Mosaic: ref 1 for first half, ref 3 for second half
        true_path = np.array([1] * 15 + [3] * 15)
        query = np.array([panel[ell, true_path[ell]] for ell in range(m)])
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        accuracy = np.mean(path == true_path)
        assert accuracy > 0.6

    def test_output_length(self):
        """Path length should equal number of sites."""
        panel = np.random.binomial(1, 0.3, size=(10, 3))
        query = panel[:, 0]
        rho = np.full(10, 0.05)
        rho[0] = 0.0
        mu = np.full(10, 0.01)
        path, _ = viterbi_ls(query, panel, rho, mu)
        assert len(path) == 10

    def test_valid_state_indices(self):
        """Path values should be valid indices into the panel."""
        np.random.seed(42)
        k, m = 4, 15
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = np.random.binomial(1, 0.3, size=m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path, _ = viterbi_ls(query, panel, rho, mu)
        assert np.all(path >= 0) and np.all(path < k)

    def test_deterministic_example(self):
        """Verify on a hand-crafted deterministic example."""
        panel = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ])
        query = np.array([0, 0, 1, 0, 1])
        rho = np.array([0.0, 0.05, 0.05, 0.05, 0.05])
        mu = np.full(5, 0.001)
        path, _ = viterbi_ls(query, panel, rho, mu)
        # Path should match without any mismatches
        for ell in range(5):
            assert query[ell] == panel[ell, path[ell]]


class TestViterbiLSWithNoncopy:
    """Tests for the NONCOPY-aware Viterbi."""

    def test_no_noncopy_selected(self):
        """No NONCOPY reference should appear in the path."""
        np.random.seed(42)
        m, k = 20, 5
        panel = np.full((m, k), NONCOPY, dtype=int)
        panel[:15, 0] = np.random.binomial(1, 0.3, 15)
        panel[:, 1] = np.random.binomial(1, 0.3, m)
        panel[5:, 2] = np.random.binomial(1, 0.3, 15)
        panel[:, 3] = np.random.binomial(1, 0.3, m)
        panel[:, 4] = np.random.binomial(1, 0.3, m)

        query = np.random.binomial(1, 0.3, m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)

        path = viterbi_ls_with_noncopy(query, panel, rho, mu)
        for ell in range(m):
            assert panel[ell, path[ell]] != NONCOPY

    def test_output_length(self):
        """Path length should equal number of sites."""
        m, k = 10, 3
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = np.random.binomial(1, 0.3, m)
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path = viterbi_ls_with_noncopy(query, panel, rho, mu)
        assert len(path) == m

    def test_all_copiable_same_as_standard(self):
        """When no NONCOPY entries, should give same result as standard Viterbi."""
        np.random.seed(42)
        m, k = 15, 3
        panel = np.random.binomial(1, 0.3, size=(m, k))
        query = panel[:, 0].copy()
        rho = np.full(m, 0.05)
        rho[0] = 0.0
        mu = np.full(m, 0.01)
        path_nc = viterbi_ls_with_noncopy(query, panel, rho, mu)
        path_std, _ = viterbi_ls(query, panel, rho, mu)
        assert np.array_equal(path_nc, path_std)


class TestPathToEdges:
    """Tests for converting Viterbi paths to edges."""

    def test_constant_path_one_edge(self):
        """A constant path should produce one edge."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        path = np.zeros(5, dtype=int)
        ref_ids = np.array([100])
        edges = path_to_edges(path, positions, child_id=200,
                               ref_node_ids=ref_ids)
        assert len(edges) == 1
        assert edges[0] == (0.0, 4001.0, 100, 200)

    def test_two_segments(self):
        """A path with one switch should produce two edges."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        path = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ref_ids = np.array([100, 101])
        edges = path_to_edges(path, positions, child_id=200,
                               ref_node_ids=ref_ids)
        assert len(edges) == 2

    def test_edges_cover_full_range(self):
        """Edges should cover the full genomic range."""
        positions = np.arange(0, 20000, 1000, dtype=float)
        path = np.array([1] * 7 + [3] * 8 + [1] * 5)
        ref_ids = np.array([100, 101, 102, 103, 104])
        edges = path_to_edges(path, positions, child_id=200,
                               ref_node_ids=ref_ids)
        # First edge starts at positions[0], last edge ends at positions[-1]+1
        assert edges[0][0] == positions[0]
        assert edges[-1][1] == positions[-1] + 1


class TestFindBreakpoints:
    """Tests for breakpoint detection."""

    def test_constant_path_no_breakpoints(self):
        """A constant path should have no breakpoints."""
        path = np.zeros(10, dtype=int)
        positions = np.arange(0, 10000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 0

    def test_one_switch(self):
        """A path with one switch should have one breakpoint."""
        path = np.array([0, 0, 0, 1, 1, 1])
        positions = np.arange(0, 6000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 1
        assert bps[0] == (3000.0, 0, 1)

    def test_multiple_switches(self):
        """Multiple switches should produce corresponding breakpoints."""
        path = np.array([0, 0, 1, 1, 2, 2])
        positions = np.arange(0, 6000, 1000, dtype=float)
        bps = find_breakpoints(path, positions)
        assert len(bps) == 2


# ============================================================================
# Tests for Gear 3: Ancestor Matching
# ============================================================================

class TestMatchingOrder:
    """Tests for ancestor matching order."""

    def test_groups_by_time(self):
        """Ancestors at same time should be grouped together."""
        ancestors = [
            {'time': 1.0, 'focal': -1},
            {'time': 0.8, 'focal': 3},
            {'time': 0.8, 'focal': 7},
            {'time': 0.6, 'focal': 1},
            {'time': 0.4, 'focal': 5},
        ]
        groups = matching_order(ancestors)
        assert len(groups) == 4

    def test_preserves_order(self):
        """Groups should appear in the same order as input."""
        ancestors = [
            {'time': 0.9, 'focal': 1},
            {'time': 0.7, 'focal': 2},
            {'time': 0.5, 'focal': 3},
        ]
        groups = matching_order(ancestors)
        times = [g[0]['time'] for g in groups]
        assert times == [0.9, 0.7, 0.5]

    def test_single_group(self):
        """If all ancestors have the same time, should be one group."""
        ancestors = [
            {'time': 0.5, 'focal': i} for i in range(5)
        ]
        groups = matching_order(ancestors)
        assert len(groups) == 1
        assert len(groups[0]) == 5


class TestBuildReferencePanel:
    """Tests for reference panel construction."""

    def test_shape(self):
        """Panel shape should be (num_sites, num_ancestors)."""
        placed = [
            {'haplotype': np.zeros(10, dtype=int), 'start': 0, 'end': 10,
             'node_id': 0},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel.shape == (10, 1)

    def test_noncopy_default(self):
        """Positions outside an ancestor's interval should be NONCOPY."""
        placed = [
            {'haplotype': np.array([1, 0, 1]), 'start': 3, 'end': 6,
             'node_id': 0},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel[0, 0] == NONCOPY
        assert panel[3, 0] == 1
        assert panel[5, 0] == 1
        assert panel[6, 0] == NONCOPY

    def test_multiple_ancestors(self):
        """Multiple ancestors should fill different columns."""
        placed = [
            {'haplotype': np.zeros(10, dtype=int), 'start': 0, 'end': 10,
             'node_id': 0},
            {'haplotype': np.ones(5, dtype=int), 'start': 2, 'end': 7,
             'node_id': 1},
        ]
        panel, _ = build_reference_panel(placed, num_inference_sites=10)
        assert panel.shape == (10, 2)
        assert panel[0, 0] == 0
        assert panel[0, 1] == NONCOPY
        assert panel[3, 1] == 1


class TestPathCompress:
    """Tests for path compression."""

    def test_no_shared_edges(self):
        """With no shared (left, right, parent), no compression occurs."""
        edges = [
            (0, 5000, 0, 1),
            (5000, 10000, 0, 2),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        assert len(new_edges) == 2
        assert len(new_nodes) == 3

    def test_shared_edges_compressed(self):
        """Multiple children sharing same parent and interval get a PC node."""
        edges = [
            (0, 5000, 0, 1),
            (0, 5000, 0, 2),
            (0, 5000, 0, 3),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
            {'id': 3, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        # Should have 1 edge from parent to PC node + 3 edges from PC to children
        assert len(new_edges) == 4
        # One new PC node
        assert len(new_nodes) == 5

    def test_pc_node_time(self):
        """PC node time should be parent_time - epsilon."""
        edges = [
            (0, 5000, 0, 1),
            (0, 5000, 0, 2),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': True},
            {'id': 2, 'time': 0.5, 'is_sample': True},
        ]
        new_edges, new_nodes = path_compress(edges, nodes)
        pc_nodes = [n for n in new_nodes if n['id'] >= 3]
        assert len(pc_nodes) == 1
        assert np.isclose(pc_nodes[0]['time'], 1.0 - PC_TIME_EPSILON)


class TestTreeSequenceBuilder:
    """Tests for the TreeSequenceBuilder."""

    def test_add_node(self):
        """Adding a node should increment the ID."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        builder = TreeSequenceBuilder(10000, 10, positions)
        id0 = builder.add_node(time=1.0)
        id1 = builder.add_node(time=0.5)
        assert id0 == 0
        assert id1 == 1
        assert len(builder.nodes) == 2

    def test_add_edges_constant_path(self):
        """A constant path should produce one edge."""
        positions = np.arange(0, 5000, 1000, dtype=float)
        builder = TreeSequenceBuilder(5000, 5, positions)
        root_id = builder.add_node(time=1.0)
        child_id = builder.add_node(time=0.0, is_sample=True)
        path = np.zeros(5, dtype=int)
        builder.add_edges_from_path(path, child_id, ref_node_ids=[root_id])
        assert len(builder.edges) == 1

    def test_add_edges_with_switch(self):
        """A path with one switch should produce two edges."""
        positions = np.arange(0, 10000, 1000, dtype=float)
        builder = TreeSequenceBuilder(10000, 10, positions)
        id0 = builder.add_node(time=1.0)
        id1 = builder.add_node(time=0.8)
        child = builder.add_node(time=0.0, is_sample=True)
        path = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        builder.add_edges_from_path(path, child, ref_node_ids=[id0, id1])
        assert len(builder.edges) == 2


# ============================================================================
# Tests for Gear 4: Sample Matching & Post-Processing
# ============================================================================

class TestFitchParsimony:
    """Tests for Fitch parsimony mutation placement."""

    def test_no_mutations_all_same(self):
        """If all leaves have the same allele, no mutations needed."""
        tree_parent = {1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [], 2: []}
        leaf_alleles = {1: 0, 2: 0}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 0

    def test_one_mutation(self):
        """A single derived leaf should produce one mutation."""
        tree_parent = {1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [], 2: []}
        leaf_alleles = {1: 0, 2: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 1

    def test_parsimony_optimal(self):
        """Fitch should find the minimum number of mutations."""
        tree_parent = {3: 1, 4: 1, 1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []}
        leaf_alleles = {2: 0, 3: 1, 4: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 1

    def test_two_mutations(self):
        """Alternating alleles require at least two mutations."""
        tree_parent = {3: 1, 4: 1, 5: 2, 6: 2, 1: 0, 2: 0, 0: None}
        tree_children = {0: [1, 2], 1: [3, 4], 2: [5, 6],
                         3: [], 4: [], 5: [], 6: []}
        leaf_alleles = {3: 0, 4: 1, 5: 0, 6: 1}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 2

    def test_star_tree(self):
        """Star tree: root with multiple leaves, all same allele."""
        tree_parent = {1: 0, 2: 0, 3: 0, 0: None}
        tree_children = {0: [1, 2, 3], 1: [], 2: [], 3: []}
        leaf_alleles = {1: 0, 2: 0, 3: 0}
        mutations = fitch_parsimony(tree_parent, tree_children,
                                     leaf_alleles, root=0)
        assert len(mutations) == 0


class TestRemoveVirtualRoot:
    """Tests for virtual root removal."""

    def test_removes_root_edges(self):
        """Edges with virtual root as parent should be removed."""
        edges = [
            (0, 10000, 0, 1),
            (0, 10000, 1, 2),
        ]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': False},
            {'id': 2, 'time': 0.0, 'is_sample': True},
        ]
        filtered_edges, filtered_nodes = remove_virtual_root(edges, nodes, 0)
        assert len(filtered_edges) == 1
        assert filtered_edges[0] == (0, 10000, 1, 2)

    def test_removes_root_node(self):
        """Virtual root node should be removed."""
        edges = [(0, 10000, 0, 1)]
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.0, 'is_sample': True},
        ]
        _, filtered_nodes = remove_virtual_root(edges, nodes, 0)
        ids = [n['id'] for n in filtered_nodes]
        assert 0 not in ids


class TestEraseFlanks:
    """Tests for flank erasure."""

    def test_trim_left(self):
        """Edge extending left of data range should be trimmed."""
        edges = [(0, 10000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=1000,
                                rightmost_position=10000)
        assert len(trimmed) == 1
        assert trimmed[0][0] == 1000

    def test_trim_right(self):
        """Edge extending right of data range should be trimmed."""
        edges = [(0, 15000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=0,
                                rightmost_position=10000)
        assert len(trimmed) == 1
        assert trimmed[0][1] == 10000

    def test_remove_out_of_range(self):
        """Edge entirely outside range should be removed."""
        edges = [(20000, 30000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=0,
                                rightmost_position=10000)
        assert len(trimmed) == 0

    def test_within_range_unchanged(self):
        """Edge within range should be unchanged."""
        edges = [(2000, 8000, 1, 2)]
        trimmed = erase_flanks(edges, leftmost_position=1000,
                                rightmost_position=9000)
        assert len(trimmed) == 1
        assert trimmed[0] == (2000, 8000, 1, 2)

    def test_example_from_docs(self):
        """Test the example from the RST documentation."""
        edges_raw = [
            (0, 10000, 1, 5),
            (2000, 8000, 2, 6),
            (7000, 15000, 3, 7),
        ]
        trimmed = erase_flanks(edges_raw, leftmost_position=1000,
                                rightmost_position=9000)
        assert len(trimmed) == 3
        assert trimmed[0] == (1000, 9000, 1, 5)
        assert trimmed[1] == (2000, 8000, 2, 6)
        assert trimmed[2] == (7000, 9000, 3, 7)


class TestSimplifyTreeSequence:
    """Tests for tree sequence simplification."""

    def test_removes_non_ancestral_nodes(self):
        """Nodes not ancestral to any sample should be removed."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.8, 'is_sample': False},
            {'id': 2, 'time': 0.5, 'is_sample': False},  # orphan
            {'id': 3, 'time': 0.0, 'is_sample': True},
            {'id': 4, 'time': 0.0, 'is_sample': True},
        ]
        edges = [
            (0, 10000, 0, 1),
            (0, 10000, 1, 3),
            (0, 10000, 1, 4),
            (0, 10000, 0, 2),   # Node 2 has no sample descendants
        ]
        sample_ids = {3, 4}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        assert 2 not in kept_nodes
        assert 3 in kept_nodes
        assert 4 in kept_nodes
        assert 0 in kept_nodes
        assert 1 in kept_nodes

    def test_keeps_edges_between_ancestral(self):
        """Edges between kept nodes should be preserved."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.0, 'is_sample': True},
        ]
        edges = [(0, 10000, 0, 1)]
        sample_ids = {1}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        assert len(kept_edges) == 1

    def test_removes_orphan_edges(self):
        """Edges involving removed nodes should not be kept."""
        nodes = [
            {'id': 0, 'time': 1.0, 'is_sample': False},
            {'id': 1, 'time': 0.5, 'is_sample': False},
            {'id': 2, 'time': 0.0, 'is_sample': True},
        ]
        edges = [
            (0, 10000, 0, 2),
            (0, 10000, 0, 1),   # Node 1 has no sample descendants
        ]
        sample_ids = {2}
        kept_nodes, kept_edges = simplify_tree_sequence(nodes, edges,
                                                         sample_ids)
        # Node 1 is not ancestral to any sample
        assert 1 not in kept_nodes
        # Only the edge from 0 to 2 should remain
        assert len(kept_edges) == 1
        assert kept_edges[0] == (0, 10000, 0, 2)
