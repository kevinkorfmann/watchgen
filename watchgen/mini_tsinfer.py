"""Small, source-guided mechanisms from the tsinfer inference pipeline.

This is deliberately not a production reimplementation. It isolates the
paper-release ancestor builder, a dense Li--Stephens-like Viterbi recurrence,
and conversion of copying paths to genomic intervals. Use tsinfer and tskit
for real inference and post-processing.

References: Kelleher et al. (2019), doi:10.1038/s41588-019-0483-y;
tsinfer 0.1.4 ``algorithm.py``/``inference.py``; and tsinfer 0.4.1
``inference.py``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

UNKNOWN = -1


@dataclass(frozen=True)
class Ancestor:
    """A partially defined ancestral haplotype over inference-site indexes."""

    start: int
    end: int
    time: float
    focal_sites: tuple[int, ...]
    haplotype: np.ndarray
    frequency: int | None = None
    kind: str = "inferred"


def _as_genotypes(D):
    D = np.asarray(D, dtype=np.int8)
    if D.ndim != 2:
        raise ValueError("D must have shape (samples, sites)")
    if not np.all(np.isin(D, [UNKNOWN, 0, 1])):
        raise ValueError("only -1 (missing), 0, and 1 are supported")
    return D


def select_inference_sites(D, ancestral_known):
    """Partition sites using the paper-era default site criteria.

    Retained sites have known ancestry, both alleles among called samples,
    at least two derived copies, and at least one ancestral copy.
    """

    D = _as_genotypes(D)
    ancestral_known = np.asarray(ancestral_known, dtype=bool)
    if ancestral_known.shape != (D.shape[1],):
        raise ValueError("ancestral_known must have one entry per site")
    keep = np.zeros(D.shape[1], dtype=bool)
    for j in range(D.shape[1]):
        called = D[:, j] >= 0
        alleles = np.unique(D[called, j])
        keep[j] = (
            ancestral_known[j]
            and np.array_equal(alleles, np.array([0, 1], dtype=np.int8))
            and np.sum(D[:, j] == 1) >= 2
            and np.sum(D[:, j] == 0) >= 1
        )
    return np.flatnonzero(keep), np.flatnonzero(~keep)


def compute_ancestor_times(D, inference_sites):
    """Return 0.1.4's ordinal proxy, not ages in generations.

    Distinct derived-allele counts are ranked: the smallest count has time 1,
    the next distinct count time 2, and so on.
    """

    D = _as_genotypes(D)
    sites = np.asarray(inference_sites, dtype=int)
    counts = np.sum(D[:, sites] == 1, axis=0)
    rank = {count: j + 1 for j, count in enumerate(sorted(set(counts.tolist())))}
    return np.array([rank[int(count)] for count in counts], dtype=float)


def get_focal_samples(D, site_index):
    """Return samples carrying the derived allele at one input site."""

    return np.flatnonzero(_as_genotypes(D)[:, site_index] == 1)


def _break_ancestor(G, a, b, carriers, focal_frequency):
    index = np.flatnonzero(carriers == 1)
    frequencies = np.sum(G == 1, axis=1)
    for j in range(a + 1, b):
        if frequencies[j] > focal_frequency:
            states = G[j, index]
            if not (np.all(states == 0) or np.all(states == 1)):
                return True
    return False


def ancestor_descriptors(D, inference_sites):
    """Group and split focal sites exactly as the 0.1.4 Python reference.

    Equal-count sites with the same carrier pattern can share one ancestor.
    An intervening older site splits them if it is polymorphic among those
    carriers.
    """

    D = _as_genotypes(D)
    sites = np.asarray(inference_sites, dtype=int)
    G = D[:, sites].T
    if np.any(G < 0):
        raise ValueError("ancestor construction requires called inference sites")
    grouped = defaultdict(lambda: defaultdict(list))
    for site, genotypes in enumerate(G):
        grouped[int(np.sum(genotypes))][genotypes.tobytes()].append(site)
    descriptors = []
    for frequency in sorted(grouped, reverse=True):
        for pattern in sorted(grouped[frequency]):
            carriers = np.frombuffer(pattern, dtype=np.int8)
            focal = grouped[frequency][pattern]
            start = 0
            for j in range(len(focal) - 1):
                if _break_ancestor(G, focal[j], focal[j + 1], carriers, frequency):
                    descriptors.append((frequency, tuple(focal[start : j + 1])))
                    start = j + 1
            descriptors.append((frequency, tuple(focal[start:])))
    return descriptors


def _extend_ancestor(G, haplotype, focal_site, site_order):
    """The carrier-filtering extension rule from tsinfer 0.1.4."""

    focal_frequency = int(np.sum(G[focal_site]))
    minimum = focal_frequency // 2
    carriers = set(np.flatnonzero(G[focal_site] == 1).tolist())
    remove_buffer = []
    last_site = focal_site
    frequencies = np.sum(G == 1, axis=1)
    for site in site_order:
        haplotype[site] = 0
        last_site = site
        if frequencies[site] > focal_frequency:
            ones = sum(int(G[site, sample]) for sample in carriers)
            consensus = int(ones >= len(carriers) - ones)  # ties choose 1
            for sample in remove_buffer:
                if G[site, sample] != consensus:
                    carriers.remove(sample)
            if len(carriers) <= minimum:
                break
            remove_buffer = [
                sample for sample in carriers if G[site, sample] != consensus
            ]
            haplotype[site] = consensus
    return last_site


def build_ancestor(D, inference_sites, focal_sites, time=None):
    """Build one ancestor; focal indexes refer to ``inference_sites``."""

    D = _as_genotypes(D)
    sites = np.asarray(inference_sites, dtype=int)
    G = D[:, sites].T
    if np.any(G < 0):
        raise ValueError("ancestor construction requires called inference sites")
    if np.isscalar(focal_sites):
        focal_sites = (int(focal_sites),)
    else:
        focal_sites = tuple(int(site) for site in focal_sites)
    if not focal_sites or tuple(sorted(focal_sites)) != focal_sites:
        raise ValueError("focal_sites must be a nonempty increasing sequence")
    frequency = int(np.sum(G[focal_sites[0]]))
    if any(int(np.sum(G[site])) != frequency for site in focal_sites):
        raise ValueError("focal sites must have the same derived count")

    haplotype = np.full(len(sites), UNKNOWN, dtype=np.int8)
    haplotype[list(focal_sites)] = 1
    carriers = set(np.flatnonzero(G[focal_sites[0]] == 1).tolist())
    frequencies = np.sum(G == 1, axis=1)
    for left, right in pairwise(focal_sites):
        for site in range(left + 1, right):
            haplotype[site] = 0
            if frequencies[site] > frequency:
                ones = sum(int(G[site, sample]) for sample in carriers)
                haplotype[site] = int(ones >= len(carriers) - ones)

    last = focal_sites[-1]
    end_site = _extend_ancestor(G, haplotype, last, range(last + 1, len(sites)))
    first = focal_sites[0]
    start = _extend_ancestor(G, haplotype, first, range(first - 1, -1, -1))
    return Ancestor(
        start=start,
        end=end_site + 1,
        time=float(frequency if time is None else time),
        focal_sites=focal_sites,
        haplotype=haplotype[start : end_site + 1].copy(),
        frequency=frequency,
    )


def generate_ancestors(D, ancestral_known):
    """Generate 0.1.4-style ancestors, including its two all-zero roots."""

    D = _as_genotypes(D)
    sites, _ = select_inference_sites(D, ancestral_known)
    descriptors = ancestor_descriptors(D, sites)
    if len(sites) == 0:
        return [], sites
    counts = sorted({frequency for frequency, _ in descriptors})
    time_map = {frequency: j + 1 for j, frequency in enumerate(counts)}
    root_time = len(time_map) + 1
    zero = np.zeros(len(sites), dtype=np.int8)
    ancestors = [
        Ancestor(0, len(sites), root_time + 1, (), zero.copy(), kind="virtual_root"),
        Ancestor(0, len(sites), root_time, (), zero.copy(), kind="ultimate_ancestor"),
    ]
    ancestors.extend(
        build_ancestor(D, sites, focal, time_map[frequency])
        for frequency, focal in descriptors
    )
    return ancestors, sites


def compute_recombination_probs(positions, recombination_rate, num_ref=None):
    """Use 0.4.1's Haldane transform ``(1-exp(-2d))/2``.

    ``num_ref`` is retained for compatibility with the previous chapter API;
    the reference-panel size does not divide the genetic distance.
    """

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 1 or np.any(np.diff(positions) < 0):
        raise ValueError("positions must be a sorted one-dimensional array")
    if recombination_rate < 0:
        raise ValueError("recombination_rate must be nonnegative")
    probability = np.zeros(len(positions))
    distance = np.diff(positions) * recombination_rate
    probability[1:] = (1 - np.exp(-2 * distance)) / 2
    return probability


def compute_mismatch_probs(
    positions, recombination_rate, mismatch_ratio, num_ref=None, num_alleles=2
):
    """Use 0.4.1's constant mismatch probability across sites.

    For ``A`` alleles this is ``(1-exp(-A*X*median(d)))/A``.
    """

    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return np.array([], dtype=float)
    if recombination_rate < 0 or mismatch_ratio < 0:
        raise ValueError("rates and ratios must be nonnegative")
    if num_alleles < 2:
        raise ValueError("num_alleles must be at least two")
    distances = np.diff(positions) * recombination_rate
    median = 0.0 if len(distances) == 0 else float(np.median(distances))
    p = (1 - np.exp(-median * mismatch_ratio * num_alleles)) / num_alleles
    return np.full(len(positions), p)


def viterbi_ls(query, panel, rho, mu, num_alleles=2):
    """Dense log-space Viterbi check for the Li--Stephens-like model.

    Production tsinfer performs the likelihood updates on marginal trees and
    supports partially defined ancestors; this teaching function does neither.
    """

    query = np.asarray(query, dtype=np.int8)
    panel = np.asarray(panel, dtype=np.int8)
    rho = np.asarray(rho, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if panel.ndim != 2 or panel.shape[0] != len(query):
        raise ValueError("panel must have shape (sites, references)")
    m, k = panel.shape
    if m == 0 or k == 0:
        raise ValueError("query and panel must be nonempty")
    if rho.shape != (m,) or mu.shape != (m,):
        raise ValueError("rho and mu must have one entry per site")
    if np.any(panel < 0):
        raise ValueError("use production tsinfer for partial ancestors")
    if np.any((rho < 0) | (rho > 1)) or np.any((mu < 0) | (mu > 1)):
        raise ValueError("probabilities must lie in [0, 1]")

    score = np.full((m, k), -np.inf)
    traceback = np.zeros((m, k), dtype=int)

    def log_emission(site):
        if query[site] == UNKNOWN:
            return np.zeros(k)
        matches = panel[site] == query[site]
        p = np.where(matches, 1 - mu[site], mu[site] / (num_alleles - 1))
        with np.errstate(divide="ignore"):
            return np.log(p)

    score[0] = -np.log(k) + log_emission(0)
    for site in range(1, m):
        switch = rho[site] / k
        transition = np.full((k, k), switch)
        np.fill_diagonal(transition, 1 - rho[site] + switch)
        with np.errstate(divide="ignore"):
            candidates = score[site - 1, :, None] + np.log(transition)
        traceback[site] = np.argmax(candidates, axis=0)
        score[site] = np.max(candidates, axis=0) + log_emission(site)

    path = np.empty(m, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for site in range(m - 1, 0, -1):
        path[site - 1] = traceback[site, path[site]]
    return path, float(score[-1, path[-1]])


def path_to_edges(path, positions, child_id, ref_node_ids, sequence_length):
    """Map site-index copying segments to genomic half-open intervals."""

    path = np.asarray(path, dtype=int)
    positions = np.asarray(positions, dtype=float)
    refs = np.asarray(ref_node_ids, dtype=int)
    if len(path) == 0 or len(path) != len(positions):
        raise ValueError("path and positions must have the same nonzero length")
    boundaries = np.append(positions, float(sequence_length))
    boundaries[0] = 0.0  # tsinfer's paper/stable-release coordinate map
    edges = []
    start = 0
    for site in range(1, len(path) + 1):
        if site == len(path) or path[site] != path[start]:
            edges.append(
                (
                    float(boundaries[start]),
                    float(boundaries[site]),
                    int(refs[path[start]]),
                    int(child_id),
                )
            )
            start = site
    return edges


def find_breakpoints(path, positions):
    """Return source changes at inference-site positions."""

    path = np.asarray(path, dtype=int)
    positions = np.asarray(positions, dtype=float)
    return [
        (float(positions[j]), int(path[j - 1]), int(path[j]))
        for j in range(1, len(path))
        if path[j] != path[j - 1]
    ]


def shared_path_segments(edges, minimum_edges=2):
    """Detect repeated contiguous multi-edge paths eligible for compression.

    Sharing one edge is not sufficient. This detector does not mutate topology
    or invent a synthetic-node time; the production builder owns those steps.
    """

    by_child = defaultdict(list)
    for left, right, parent, child in edges:
        by_child[int(child)].append((float(left), float(right), int(parent)))
    occurrences = defaultdict(set)
    for child, child_edges in by_child.items():
        child_edges.sort()
        for size in range(minimum_edges, len(child_edges) + 1):
            for start in range(len(child_edges) - size + 1):
                run = child_edges[start : start + size]
                if all(run[j - 1][1] == run[j][0] for j in range(1, size)):
                    occurrences[tuple(run)].add(child)
    return {
        signature: tuple(sorted(children))
        for signature, children in occurrences.items()
        if len(children) >= 2
    }


def fitch_parsimony(tree_children, leaf_alleles, root, root_state=0):
    """Place a deterministic minimum-change character on one rooted tree.

    ``root_state`` constrains the known ancestral state. General mutation
    mapping in real workflows is delegated to tskit.
    """

    sets = {}

    def upward(node):
        children = tree_children.get(node, ())
        if not children:
            if node not in leaf_alleles:
                raise ValueError(f"missing allele for leaf {node}")
            sets[node] = {leaf_alleles[node]}
            return
        for child in children:
            upward(child)
        intersection = set.intersection(*(sets[child] for child in children))
        sets[node] = intersection or set.union(*(sets[child] for child in children))

    upward(root)
    mutations = []

    def downward(node, state):
        for child in tree_children.get(node, ()):
            child_state = state if state in sets[child] else min(sets[child])
            if child_state != state:
                mutations.append((child, state, child_state))
            downward(child, child_state)

    downward(root, root_state)
    return mutations


def erase_flanks(edges, first_site, last_site, sequence_length):
    """Keep topology from the first site through ``last_site + 1``."""

    left_bound = float(first_site)
    right_bound = min(float(last_site) + 1, float(sequence_length))
    trimmed = []
    for left, right, parent, child in edges:
        left = max(float(left), left_bound)
        right = min(float(right), right_bound)
        if left < right:
            trimmed.append((left, right, int(parent), int(child)))
    return trimmed


def simplify_ancestral_subgraph(nodes, edges, sample_ids):
    """Find nodes and edges ancestral to samples (teaching projection only)."""

    parents = defaultdict(list)
    for edge in edges:
        parents[int(edge[3])].append(edge)
    keep = {int(sample) for sample in sample_ids}
    stack = list(keep)
    while stack:
        child = stack.pop()
        for _, _, parent, _ in parents.get(child, ()):
            if int(parent) not in keep:
                keep.add(int(parent))
                stack.append(int(parent))
    kept_edges = [edge for edge in edges if edge[2] in keep and edge[3] in keep]
    kept_nodes = [node for node in nodes if node["id"] in keep]
    return kept_nodes, kept_edges


def demo():
    """Run the auditable mechanisms on a deterministic matrix."""

    D = np.array(
        [
            [1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    ancestors, sites = generate_ancestors(D, np.ones(D.shape[1], dtype=bool))
    print(f"{len(sites)} inference sites; {len(ancestors)} ancestors including roots")


if __name__ == "__main__":
    demo()
