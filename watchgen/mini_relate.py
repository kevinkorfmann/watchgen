"""Source-guided teaching kernels for Relate.

Relate combines a modified Li--Stephens painting, a directional hierarchical
tree builder, robust mutation mapping, branch association across neighbouring
trees, and branch-length MCMC.  This module isolates a few mathematical kernels;
it is not a reimplementation of the Relate executable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def modified_emission(target: int, reference: int, mismatch: float) -> float:
    """Return Relate's directional emission probability (Supplement eq. 12).

    A derived target copied from an ancestral reference receives ``mismatch``.
    The other three allele pairs all receive ``1 - mismatch``.
    """

    if target not in (0, 1) or reference not in (0, 1):
        raise ValueError("alleles must be coded 0 (ancestral) or 1 (derived)")
    if not 0.0 < mismatch < 0.5:
        raise ValueError("mismatch must lie between 0 and 0.5")
    return mismatch if target == 1 and reference == 0 else 1.0 - mismatch


def copying_posterior(
    target: Sequence[int],
    panel: np.ndarray,
    recombination: Sequence[float],
    mismatch: float = 0.025,
) -> np.ndarray:
    """Run a dense modified Li--Stephens forward--backward recursion.

    ``panel`` has shape ``(sites, references)`` and ``recombination[l]`` is the
    switch probability between sites ``l - 1`` and ``l``.  Production Relate
    skips many ancestral target sites and uses a stepping-stone implementation;
    this dense form exposes the same small-example HMM probabilities.
    """

    target = np.asarray(target, dtype=int)
    panel = np.asarray(panel, dtype=int)
    recombination = np.asarray(recombination, dtype=float)
    if panel.ndim != 2 or target.ndim != 1 or panel.shape[0] != target.size:
        raise ValueError("panel must have shape (len(target), references)")
    if panel.shape[1] == 0:
        raise ValueError("at least one reference haplotype is required")
    if recombination.shape != target.shape:
        raise ValueError("recombination must contain one value per site")
    if np.any((target < 0) | (target > 1)) or np.any((panel < 0) | (panel > 1)):
        raise ValueError("haplotypes must be binary")
    if np.any((recombination < 0.0) | (recombination > 1.0)):
        raise ValueError("switch probabilities must lie in [0, 1]")
    if not 0.0 < mismatch < 0.5:
        raise ValueError("mismatch must lie between 0 and 0.5")

    sites, references = panel.shape
    emission = np.where(
        (target[:, None] == 1) & (panel == 0), mismatch, 1.0 - mismatch
    )
    forward = np.empty((sites, references), dtype=float)
    forward[0] = emission[0] / references
    forward[0] /= forward[0].sum()
    for site in range(1, sites):
        switch = recombination[site]
        prediction = (1.0 - switch) * forward[site - 1] + switch / references
        forward[site] = prediction * emission[site]
        forward[site] /= forward[site].sum()

    backward = np.ones((sites, references), dtype=float)
    for site in range(sites - 2, -1, -1):
        switch = recombination[site + 1]
        weighted = emission[site + 1] * backward[site + 1]
        backward[site] = (
            (1.0 - switch) * weighted + switch * weighted.sum() / references
        )
        backward[site] /= backward[site].sum()
    posterior = forward * backward
    posterior /= posterior.sum(axis=1, keepdims=True)
    return posterior


def relative_distance_row(
    posterior: Sequence[float], mismatch: float = 0.025
) -> np.ndarray:
    """Convert posterior copying weights to Relate's row-centred score.

    The full rescaling has an additive row constant involving sequence length.
    Relate subtracts each row minimum, so that constant cancels.  Normalized
    posterior probabilities preserve the required ordering, not an absolute
    mutation count.
    """

    posterior = np.asarray(posterior, dtype=float)
    if posterior.ndim != 1 or posterior.size == 0:
        raise ValueError("posterior must be a nonempty vector")
    if np.any(posterior <= 0.0) or not np.isclose(posterior.sum(), 1.0):
        raise ValueError("posterior entries must be positive and sum to one")
    if not 0.0 < mismatch < 0.5:
        raise ValueError("mismatch must lie between 0 and 0.5")
    distance = np.log(posterior) / math.log(mismatch / (1.0 - mismatch))
    return distance - distance.min()


def painting_distance_matrix(
    haplotypes: np.ndarray,
    recombination: Sequence[float],
    focal_site: int,
    mismatch: float = 0.025,
) -> np.ndarray:
    """Paint every haplotype against all others at one focal site."""

    haplotypes = np.asarray(haplotypes, dtype=int)
    if haplotypes.ndim != 2:
        raise ValueError("haplotypes must have shape (samples, sites)")
    samples, sites = haplotypes.shape
    if samples < 2 or not 0 <= focal_site < sites:
        raise ValueError("invalid sample count or focal site")
    matrix = np.zeros((samples, samples), dtype=float)
    for target_index in range(samples):
        references = [j for j in range(samples) if j != target_index]
        posterior = copying_posterior(
            haplotypes[target_index],
            haplotypes[references].T,
            recombination,
            mismatch,
        )
        matrix[target_index, references] = relative_distance_row(
            posterior[focal_site], mismatch
        )
    return matrix


def directional_mutation_distance(haplotypes: np.ndarray) -> np.ndarray:
    """Count sites derived in row ``i`` and ancestral in column ``j``."""

    haplotypes = np.asarray(haplotypes, dtype=int)
    if haplotypes.ndim != 2 or np.any((haplotypes < 0) | (haplotypes > 1)):
        raise ValueError("haplotypes must be a binary samples-by-sites matrix")
    return np.sum(
        (haplotypes[:, None, :] == 1) & (haplotypes[None, :, :] == 0), axis=2
    ).astype(float)


@dataclass
class TreeNode:
    """A rooted binary-tree node used by the teaching tree builder."""

    id: int
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def leaves(self) -> frozenset[int]:
        if self.is_leaf:
            return frozenset({self.id})
        if self.left is None or self.right is None:
            raise ValueError("tree must be binary")
        return self.left.leaves | self.right.leaves


def cluster_distance(
    distance: np.ndarray, source: Iterable[int], destination: Iterable[int]
) -> float:
    """Return the cardinality-weighted mean directional cluster distance."""

    source, destination = tuple(source), tuple(destination)
    if not source or not destination:
        raise ValueError("clusters must be nonempty")
    return float(distance[np.ix_(source, destination)].mean())


def find_mutual_minimum_pair(
    distance: np.ndarray,
    clusters: Mapping[int, frozenset[int]],
    tolerance: float = 0.2,
) -> tuple[int, int]:
    """Choose a mutual-row-minimum pair, with Relate's fallback."""

    if tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative")
    active = sorted(clusters)
    if len(active) < 2:
        raise ValueError("at least two clusters are required")
    directed = {
        (a, b): cluster_distance(distance, clusters[a], clusters[b])
        for a in active
        for b in active
        if a != b
    }
    row_minimum = {
        a: min(directed[a, b] for b in active if b != a) for a in active
    }
    pairs = [(a, b) for x, a in enumerate(active) for b in active[x + 1 :]]
    feasible = [
        (a, b)
        for a, b in pairs
        if directed[a, b] <= row_minimum[a] + tolerance
        and directed[b, a] <= row_minimum[b] + tolerance
    ]
    candidates = feasible or pairs
    return min(candidates, key=lambda pair: directed[pair] + directed[pair[::-1]])


def build_tree(distance: np.ndarray, tolerance: float = 0.2) -> TreeNode:
    """Build one rooted binary tree from a directional distance matrix."""

    distance = np.asarray(distance, dtype=float)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    if not np.all(np.isfinite(distance)) or distance.shape[0] < 2:
        raise ValueError("a finite matrix for at least two samples is required")
    samples = distance.shape[0]
    nodes = {i: TreeNode(i) for i in range(samples)}
    clusters = {i: frozenset({i}) for i in range(samples)}
    next_id = samples
    while len(clusters) > 1:
        first, second = find_mutual_minimum_pair(distance, clusters, tolerance)
        nodes[next_id] = TreeNode(next_id, nodes[first], nodes[second])
        merged = clusters[first] | clusters[second]
        del clusters[first], clusters[second]
        clusters[next_id] = merged
        next_id += 1
    return nodes[next(iter(clusters))]


def iter_nodes(root: TreeNode) -> list[TreeNode]:
    """Return nodes in preorder."""

    out = [root]
    if root.left is not None:
        out.extend(iter_nodes(root.left))
    if root.right is not None:
        out.extend(iter_nodes(root.right))
    return out


def to_newick(root: TreeNode) -> str:
    """Return topology-only Newick text."""

    if root.is_leaf:
        return str(root.id)
    return f"({to_newick(root.left)},{to_newick(root.right)})"


def map_mutation_exact(
    root: TreeNode, carriers: Iterable[int], allow_flip: bool = True
) -> tuple[int | None, bool]:
    """Map a mutation to one exact branch, optionally flipping allele labels.

    Production Relate also has approximate thresholds and a fractional
    multi-branch fallback.  The mini deliberately exposes only its exact stage.
    """

    carriers = frozenset(carriers)
    samples = root.leaves
    if not carriers or not carriers < samples:
        return None, False
    branches = [node for node in iter_nodes(root) if node is not root]
    exact = [node.id for node in branches if node.leaves == carriers]
    if len(exact) == 1:
        return exact[0], False
    if allow_flip:
        exact = [node.id for node in branches if node.leaves == samples - carriers]
        if len(exact) == 1:
            return exact[0], True
    return None, False


def _parent_map(root: TreeNode) -> dict[int, int]:
    parents: dict[int, int] = {}
    for node in iter_nodes(root):
        for child in (node.left, node.right):
            if child is not None:
                parents[child.id] = node.id
    return parents


def compatible_event_orders(root: TreeNode) -> list[int]:
    """Return one deterministic young-to-old order of internal nodes."""

    order: list[int] = []

    def visit(node: TreeNode) -> None:
        if node.is_leaf:
            return
        visit(node.left)
        visit(node.right)
        order.append(node.id)

    visit(root)
    return order


def node_times_from_intervals(
    root: TreeNode, event_order: Sequence[int], intervals: Sequence[float]
) -> dict[int, float]:
    """Convert a compatible event order and ``tau_N,...,tau_2`` to node times."""

    internal = {node.id for node in iter_nodes(root) if not node.is_leaf}
    if set(event_order) != internal or len(event_order) != len(internal):
        raise ValueError("event_order must contain every internal node exactly once")
    intervals = np.asarray(intervals, dtype=float)
    if intervals.shape != (len(internal),) or np.any(intervals <= 0.0):
        raise ValueError("one positive interval is required per event")
    position = {node_id: index for index, node_id in enumerate(event_order)}
    for node in iter_nodes(root):
        if node.is_leaf:
            continue
        for child in (node.left, node.right):
            if child is not None and not child.is_leaf:
                if position[child.id] >= position[node.id]:
                    raise ValueError("event order violates the tree topology")
    times = {node.id: 0.0 for node in iter_nodes(root) if node.is_leaf}
    cumulative = np.cumsum(intervals)
    times.update({node_id: float(cumulative[i]) for i, node_id in enumerate(event_order)})
    return times


def branch_lengths(root: TreeNode, node_times: Mapping[int, float]) -> dict[int, float]:
    """Return each child-to-parent branch length, keyed by child ID."""

    lengths = {
        child: float(node_times[parent] - node_times[child])
        for child, parent in _parent_map(root).items()
    }
    if any(length <= 0.0 for length in lengths.values()):
        raise ValueError("node times must increase strictly toward the root")
    return lengths


def log_branch_mutation_likelihood(
    root: TreeNode,
    node_times: Mapping[int, float],
    mutations: Mapping[int, float],
    exposure: Mapping[int, float],
    theta: float,
) -> float:
    """Poisson log likelihood for mutation counts on associated branches."""

    if theta <= 0.0:
        raise ValueError("theta must be positive")
    total = 0.0
    for branch, length in branch_lengths(root, node_times).items():
        count = float(mutations.get(branch, 0.0))
        span = float(exposure.get(branch, 0.0))
        if count < 0.0 or span < 0.0:
            raise ValueError("counts and exposures must be nonnegative")
        mean = theta * span * length / 2.0
        if mean == 0.0:
            if count > 0.0:
                return -math.inf
            continue
        total += count * math.log(mean) - mean - math.lgamma(count + 1.0)
    return total


def log_coalescent_interval_prior(intervals: Sequence[float], samples: int) -> float:
    """Log density of ``tau_N,...,tau_2`` under the standard coalescent."""

    intervals = np.asarray(intervals, dtype=float)
    if intervals.shape != (samples - 1,) or np.any(intervals <= 0.0):
        return -math.inf
    lineages = np.arange(samples, 1, -1, dtype=float)
    rates = lineages * (lineages - 1.0) / 2.0
    return float(np.sum(np.log(rates) - rates * intervals))


def log_ranked_tree_posterior(
    root: TreeNode,
    event_order: Sequence[int],
    intervals: Sequence[float],
    mutations: Mapping[int, float],
    exposure: Mapping[int, float],
    theta: float,
) -> float:
    """Log target for the mini's fixed-event-order interval sampler."""

    prior = log_coalescent_interval_prior(intervals, len(root.leaves))
    if not math.isfinite(prior):
        return -math.inf
    times = node_times_from_intervals(root, event_order, intervals)
    return prior + log_branch_mutation_likelihood(
        root, times, mutations, exposure, theta
    )


def sample_ranked_branch_lengths(
    root: TreeNode,
    event_order: Sequence[int],
    mutations: Mapping[int, float],
    exposure: Mapping[int, float],
    theta: float,
    iterations: int = 5_000,
    burn_in: int = 1_000,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Sample intervals for a fixed event order using exponential proposals.

    Relate additionally swaps incomparable event orders and pools information
    across equivalent branches.  This smaller sampler includes the exact
    Hastings correction for its exponential random-walk proposal.
    """

    samples = len(root.leaves)
    if iterations <= burn_in or burn_in < 0:
        raise ValueError("iterations must exceed a nonnegative burn-in")
    rng = np.random.default_rng(seed)
    lineages = np.arange(samples, 1, -1, dtype=float)
    intervals = 1.0 / (lineages * (lineages - 1.0) / 2.0)
    current = log_ranked_tree_posterior(
        root, event_order, intervals, mutations, exposure, theta
    )
    draws, accepted = [], 0
    for iteration in range(iterations):
        index = int(rng.integers(intervals.size))
        old = intervals[index]
        proposed_value = float(rng.exponential(old))
        proposal = intervals.copy()
        proposal[index] = proposed_value
        candidate = log_ranked_tree_posterior(
            root, event_order, proposal, mutations, exposure, theta
        )
        log_hastings = (
            math.log(old / proposed_value) - old / proposed_value + proposed_value / old
        )
        if math.log(rng.random()) < candidate - current + log_hastings:
            intervals, current = proposal, candidate
            accepted += 1
        if iteration >= burn_in:
            draws.append(intervals.copy())
    return np.asarray(draws), accepted / iterations


def piecewise_coalescence_rate_mle(
    tmrcas: Sequence[float], boundaries: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pairwise coalescence rates as events divided by exposure.

    This is Supplementary equation 50.  Relate averages pair-specific estimates
    and alternates rate estimation with branch-length re-estimation; it is not a
    generic hidden-variable EM M-step.
    """

    tmrcas = np.asarray(tmrcas, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    if tmrcas.ndim != 1 or tmrcas.size == 0 or np.any(tmrcas < 0.0):
        raise ValueError("tmrcas must be a nonempty nonnegative vector")
    if boundaries.ndim != 1 or boundaries.size < 2:
        raise ValueError("at least two epoch boundaries are required")
    if boundaries[0] != 0.0 or np.any(np.diff(boundaries) <= 0.0):
        raise ValueError("boundaries must increase strictly from zero")
    if tmrcas.max() >= boundaries[-1]:
        raise ValueError("the final boundary must exceed every TMRCA")
    events = np.zeros(boundaries.size - 1, dtype=float)
    exposure = np.zeros_like(events)
    for time in tmrcas:
        for epoch, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            exposure[epoch] += max(0.0, min(time, end) - start)
            if start <= time < end:
                events[epoch] += 1.0
    rates = np.divide(events, exposure, out=np.zeros_like(events), where=exposure > 0.0)
    return rates, events, exposure


def demo(seed: int = 13) -> dict[str, object]:
    """Run a small deterministic example spanning the documented kernels."""

    haplotypes = np.array(
        [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=int
    )
    root = build_tree(directional_mutation_distance(haplotypes))
    order = compatible_event_orders(root)
    exposure = {branch: 1.0 for branch in _parent_map(root)}
    draws, acceptance = sample_ranked_branch_lengths(
        root,
        order,
        mutations={0: 1, 1: 1, 2: 1, 3: 0},
        exposure=exposure,
        theta=2.0,
        iterations=1_500,
        burn_in=500,
        seed=seed,
    )
    rates, _, _ = piecewise_coalescence_rate_mle(
        [0.2, 0.4, 1.2, 1.8], [0.0, 0.5, 1.0, 2.0]
    )
    return {
        "newick": to_newick(root) + ";",
        "mean_intervals": draws.mean(axis=0),
        "acceptance": acceptance,
        "coalescence_rates": rates,
    }


if __name__ == "__main__":
    result = demo()
    print("tree:", result["newick"])
    print("mean intervals:", np.round(result["mean_intervals"], 4))
    print("acceptance:", round(float(result["acceptance"]), 3))
    print("pairwise coalescence rates:", np.round(result["coalescence_rates"], 4))
