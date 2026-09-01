"""Source-derived checks for the bounded tsinfer teaching mechanisms."""

import itertools

import numpy as np
import pytest

from watchgen.mini_tsinfer import (
    UNKNOWN,
    ancestor_descriptors,
    build_ancestor,
    compute_ancestor_times,
    compute_mismatch_probs,
    compute_recombination_probs,
    erase_flanks,
    fitch_parsimony,
    generate_ancestors,
    path_to_edges,
    select_inference_sites,
    shared_path_segments,
    simplify_ancestral_subgraph,
    viterbi_ls,
)


def test_inference_site_partition_ignores_missing_calls():
    D = np.array(
        [
            [1, 1, 1, 0, -1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    inference, other = select_inference_sites(D, [True, True, True, True, True])
    np.testing.assert_array_equal(inference, [0, 4])
    np.testing.assert_array_equal(other, [1, 2, 3])


def test_unknown_ancestral_state_excludes_site():
    D = np.array([[1], [1], [0]], dtype=np.int8)
    inference, other = select_inference_sites(D, [False])
    assert len(inference) == 0
    np.testing.assert_array_equal(other, [0])


def test_paper_release_times_rank_distinct_counts():
    D = np.array(
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(compute_ancestor_times(D, [0, 1, 2]), [1, 2, 3])


def test_equal_count_sites_share_descriptor_when_carrier_pattern_matches():
    # Rows are samples. Sites 0 and 2 have the same two carriers. The
    # intervening older site is constant among those carriers, so no split.
    D = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    descriptors = ancestor_descriptors(D, [0, 1, 2])
    assert (2, (0, 2)) in descriptors


def test_intervening_older_polymorphism_splits_descriptor():
    D = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=np.int8)
    descriptors = ancestor_descriptors(D, [0, 1, 2])
    assert (2, (0,)) in descriptors
    assert (2, (2,)) in descriptors
    assert (2, (0, 2)) not in descriptors


def test_build_ancestor_keeps_multiple_focal_sites():
    D = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    ancestor = build_ancestor(D, [0, 1, 2], [0, 2], time=1)
    assert ancestor.focal_sites == (0, 2)
    np.testing.assert_array_equal(ancestor.haplotype, [1, 1, 1])
    assert (ancestor.start, ancestor.end) == (0, 3)


def test_generation_adds_virtual_and_ultimate_zero_ancestors():
    D = np.array([[1, 1], [1, 1], [0, 0]], dtype=np.int8)
    ancestors, sites = generate_ancestors(D, [True, True])
    np.testing.assert_array_equal(sites, [0, 1])
    assert [a.kind for a in ancestors[:2]] == ["virtual_root", "ultimate_ancestor"]
    assert ancestors[0].time > ancestors[1].time > ancestors[2].time
    np.testing.assert_array_equal(ancestors[0].haplotype, [0, 0])
    np.testing.assert_array_equal(ancestors[1].haplotype, [0, 0])


def test_haldane_probability_matches_tsinfer_041_transform():
    positions = np.array([0, 10, 30], dtype=float)
    rho = compute_recombination_probs(positions, 0.01, num_ref=10_000)
    expected = np.array([0, (1 - np.exp(-0.2)) / 2, (1 - np.exp(-0.4)) / 2])
    np.testing.assert_allclose(rho, expected)
    assert rho[-1] < 0.5


def test_mismatch_probability_uses_median_distance_not_panel_size():
    positions = np.array([0, 10, 30], dtype=float)
    mu = compute_mismatch_probs(
        positions, 0.01, mismatch_ratio=2, num_ref=10_000, num_alleles=2
    )
    median_distance = 0.15
    expected = (1 - np.exp(-2 * 2 * median_distance)) / 2
    np.testing.assert_allclose(mu, expected)


def _direct_path_probability(path, query, panel, rho, mu):
    k = panel.shape[1]
    p = 1 / k
    for site, state in enumerate(path):
        if site:
            p *= (
                1 - rho[site] + rho[site] / k
                if state == path[site - 1]
                else rho[site] / k
            )
        p *= 1 - mu[site] if query[site] == panel[site, state] else mu[site]
    return p


def test_viterbi_equals_exhaustive_path_enumeration():
    panel = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.int8)
    query = np.array([0, 0, 0, 0], dtype=np.int8)
    rho = np.array([0, 0.1, 0.1, 0.1])
    mu = np.full(4, 0.01)
    path, log_probability = viterbi_ls(query, panel, rho, mu)
    paths = list(itertools.product(range(2), repeat=4))
    probabilities = [_direct_path_probability(p, query, panel, rho, mu) for p in paths]
    assert tuple(path) == paths[int(np.argmax(probabilities))]
    np.testing.assert_allclose(np.exp(log_probability), max(probabilities))


def test_viterbi_missing_query_state_has_neutral_emission():
    panel = np.array([[0, 1], [0, 1]], dtype=np.int8)
    path, _ = viterbi_ls([UNKNOWN, 1], panel, rho=[0, 0], mu=[0.01, 0.01])
    np.testing.assert_array_equal(path, [1, 1])


def test_partial_reference_panel_is_explicitly_out_of_scope():
    with pytest.raises(ValueError, match="production tsinfer"):
        viterbi_ls([0], [[UNKNOWN]], rho=[0], mu=[0.01])


def test_path_coordinates_use_zero_and_sequence_length_boundaries():
    edges = path_to_edges(
        path=[0, 0, 1],
        positions=[10, 20, 40],
        child_id=9,
        ref_node_ids=[3, 4],
        sequence_length=100,
    )
    assert edges == [(0.0, 40.0, 3, 9), (40.0, 100.0, 4, 9)]


def test_path_compression_requires_repeated_multi_edge_run():
    edges = [
        (0, 5, 1, 10),
        (5, 9, 2, 10),
        (0, 5, 1, 11),
        (5, 9, 2, 11),
        (0, 5, 1, 12),  # a single shared edge is insufficient
    ]
    shared = shared_path_segments(edges)
    assert shared[((0.0, 5.0, 1), (5.0, 9.0, 2))] == (10, 11)
    assert all(len(signature) >= 2 for signature in shared)


def test_parsimony_respects_known_root_state():
    children = {0: [1, 2], 1: [3, 4]}
    mutations = fitch_parsimony(children, {2: 0, 3: 1, 4: 1}, root=0, root_state=0)
    assert mutations == [(1, 0, 1)]


def test_flank_erasure_includes_last_site_and_caps_at_sequence_length():
    edges = [(0, 100, 1, 2), (12, 18, 2, 3)]
    assert erase_flanks(edges, 10, 19.5, 20) == [
        (10.0, 20.0, 1, 2),
        (12.0, 18.0, 2, 3),
    ]


def test_teaching_simplifier_removes_nonancestral_branch():
    nodes = [{"id": j} for j in range(5)]
    edges = [(0, 10, 0, 1), (0, 10, 1, 3), (0, 10, 0, 2)]
    kept_nodes, kept_edges = simplify_ancestral_subgraph(nodes, edges, {3})
    assert [node["id"] for node in kept_nodes] == [0, 1, 3]
    assert kept_edges == [(0, 10, 0, 1), (0, 10, 1, 3)]
