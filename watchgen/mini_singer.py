"""Small, source-guided mechanisms from SINGER.

This module implements scalar equations and single-tree analogues used in the
SINGER chapter. It is deliberately *not* an ARG sampler: production SINGER
maintains partial-branch states across a tree sequence, performs stochastic
traceback in two HMMs, maps mutations, and executes SGPR proposals on an ARG.

Ground truth is Deng, Nielsen & Song (2025), Methods and Supplement B.1-B.4,
and popgenmethods/SINGER commit eb8e39b1a15be4a9a4df4fdaab61847bf73515d7.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log

import numpy as np
from scipy.integrate import quad


def _check_interval(lower, upper):
    if not (isfinite(lower) and isfinite(upper) and 0 <= lower < upper):
        raise ValueError("expected a finite interval with 0 <= lower < upper")


# Branch sampling (Supplement B.1)

def joining_probability_exact(x, y, tree_intervals):
    """Integrate the exact survival function over branch interval ``[x,y)``."""
    _check_interval(x, y)
    intervals = [(float(lo), float(hi)) for lo, hi in tree_intervals]
    if any(lo < 0 or hi <= lo for lo, hi in intervals):
        raise ValueError("tree intervals must have 0 <= lower < upper")
    points = sorted({0.0, x, y, *(z for p in intervals for z in p if z <= y)})
    survival = 1.0
    answer = 0.0
    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2
        lineages = sum(lo <= mid < hi for lo, hi in intervals)
        width = right - left
        if right > x and left < y:
            a, b = max(left, x), min(right, y)
            local = survival * exp(-lineages * (a - left))
            answer += (local * (1 - exp(-lineages * (b - a))) / lineages
                       if lineages else local * (b - a))
        survival *= exp(-lineages * width)
    return answer


def lambda_approx(t, n):
    """Deterministic lineage-count approximation in Supplement B.1.2."""
    if n <= 1:
        raise ValueError("n must exceed one")
    t = np.asarray(t)
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    return n / (n + (1 - n) * np.exp(-t / 2))


def F_bar_approx(t, n):
    """Approximate exceedance probability of a new lineage's joining time."""
    t = np.asarray(t)
    denominator = n + (1 - n) * np.exp(-t / 2)
    return np.exp(-t) / denominator**2


def f_approx(t, n):
    """Approximate joining-time density ``lambda(t) * F_bar(t)``."""
    return lambda_approx(t, n) * F_bar_approx(t, n)


def joining_prob_approx(x, y, n):
    """Approximate probability of joining a branch spanning ``[x,y)``."""
    _check_interval(x, y)
    value, _ = quad(lambda t: float(F_bar_approx(t, n)), x, y)
    return value


def lambda_inverse(ell, n):
    """Inverse of :func:`lambda_approx` for ``1 < ell <= n``."""
    if n <= 1 or not 1 < ell <= n:
        raise ValueError("expected n > 1 and 1 < ell <= n")
    return -2 * log(n * (ell - 1) / (ell * (n - 1)))


def representative_time(x, y, n):
    """SINGER's heuristic representative time for a branch ``[x,y)``."""
    _check_interval(x, y)
    target = np.sqrt(lambda_approx(x, n) * lambda_approx(y, n))
    return lambda_inverse(float(target), n)


def poisson_edge_probability(changes, length, theta):
    """Poisson probability for zero or one binary change on an edge."""
    if changes not in (0, 1):
        raise ValueError("the binary model permits zero or one change")
    if length < 0 or theta < 0:
        raise ValueError("length and theta must be non-negative")
    mean = theta * length / 2
    return exp(-mean) * (mean if changes else 1.0)


def emission_probability(allele_new, allele_lower, allele_upper, tau,
                         branch_lower, branch_upper, theta):
    """Three-edge emission from Supplementary Figure 22.

    The joining-point state is imputed by binary parsimony (majority of the
    three incident states). The product covers the new lineage and both pieces
    of the bisected joining branch. A one-change edge uses ``m exp(-m)``, not
    the probability of at least one mutation.
    """
    alleles = (allele_new, allele_lower, allele_upper)
    if any(a not in (0, 1) for a in alleles):
        raise ValueError("alleles must be binary")
    if not 0 <= branch_lower <= tau <= branch_upper:
        raise ValueError("tau must lie on the joining branch")
    join = int(sum(alleles) >= 2)
    pieces = (
        (abs(allele_new - join), tau),
        (abs(allele_lower - join), tau - branch_lower),
        (abs(allele_upper - join), branch_upper - tau),
    )
    return float(np.prod([poisson_edge_probability(k, length, theta)
                          for k, length in pieces]))


@dataclass(frozen=True)
class BranchState:
    """Full or carried partial-branch state in the branch HMM."""
    child: int
    parent: int
    lower_time: float
    upper_time: float
    is_partial: bool = False

    @property
    def length(self):
        return self.upper_time - self.lower_time


def build_state_space(full_branches, partial_branches, forward_probs=None,
                      epsilon=0.01):
    """Keep all full branches and partial states above the forward threshold."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    if forward_probs is not None:
        partial_branches = zip(partial_branches, forward_probs)
    return list(full_branches) + [state for state, probability in partial_branches
                                  if probability >= epsilon]


def branch_recombination_probability(tau, rho):
    """Return ``r_i = 1 - exp(-rho*tau/2)`` (Supplement B.1.5)."""
    if tau < 0 or rho < 0:
        raise ValueError("tau and rho must be non-negative")
    return 1 - exp(-rho * tau / 2)


def branch_transition_prob(tau_i, tau_j, p_j, rho, is_partial_j,
                           q_sum, same_branch):
    """Li-Stephens-like branch transition in Supplement B.1.5."""
    if p_j < 0 or q_sum <= 0:
        raise ValueError("p_j must be non-negative and q_sum positive")
    r_i = branch_recombination_probability(tau_i, rho)
    q_j = (0.0 if is_partial_j else
           branch_recombination_probability(tau_j, rho) * p_j)
    return (1 - r_i if same_branch else 0.0) + r_i * q_j / q_sum


def split_branch_transition(full_branch, segments, n):
    """Split forward mass among carried segments by joining probability."""
    del full_branch
    if not segments:
        raise ValueError("at least one segment is required")
    mass = np.array([joining_prob_approx(s.lower_time, s.upper_time, n)
                     for s in segments])
    if mass.sum() <= 0:
        raise ValueError("segments have no joining mass")
    return (mass / mass.sum()).tolist()


# Time sampling (Supplement B.2)

def partition_branch(x, y, d=20):
    """Partition ``[x,y)`` into equal Exp(1) probability intervals."""
    _check_interval(x, y)
    if d < 1:
        raise ValueError("d must be positive")
    fractions = np.linspace(0, 1, d + 1)
    survival = np.exp(-x) - fractions * (np.exp(-x) - np.exp(-y))
    return -np.log(survival)


def representative_times_ts(boundaries):
    """Representative times defined by midpoints in exponential space."""
    boundaries = np.asarray(boundaries, dtype=float)
    if len(boundaries) < 2 or np.any(np.diff(boundaries) <= 0):
        raise ValueError("boundaries must be strictly increasing")
    return -np.log((np.exp(-boundaries[:-1]) +
                    np.exp(-boundaries[1:])) / 2)


def psmc_transition_density(t, s, rho):
    """Continuous density, or no-recombination point mass at ``t=s``.

    This follows Supplement Eq. (3). The equality value is a discrete mass;
    callers must not integrate it as if it were an ordinary density.
    """
    if t < 0 or s <= 0 or rho < 0:
        raise ValueError("expected t >= 0, s > 0, and rho >= 0")
    recomb = 1 - exp(-rho * s)
    if np.isclose(t, s, rtol=0, atol=1e-12):
        return exp(-rho * s)
    if t < s:
        return recomb / s * (1 - exp(-t))
    return recomb / s * (exp(-(t - s)) - exp(-t))


def psmc_transition_cdf(t, s, rho):
    """CDF from Supplement Eq. (3), including its atom at ``s``."""
    if t < 0 or s <= 0 or rho < 0:
        raise ValueError("expected t >= 0, s > 0, and rho >= 0")
    recomb = 1 - exp(-rho * s)
    if t < s:
        return recomb / s * (t + exp(-t) - 1)
    return recomb / s * (s - exp(-(t - s)) + exp(-t)) + exp(-rho * s)


def time_transition_matrix(boundaries_prev, taus_prev, boundaries_next, rho):
    """Conditional interval probabilities from Supplement Eq. (4)."""
    del boundaries_prev
    taus_prev = np.asarray(taus_prev, dtype=float)
    boundaries_next = np.asarray(boundaries_next, dtype=float)
    if len(boundaries_next) < 2 or np.any(np.diff(boundaries_next) <= 0):
        raise ValueError("next-state boundaries must be strictly increasing")
    matrix = np.empty((len(taus_prev), len(boundaries_next) - 1))
    lower, upper = boundaries_next[0], boundaries_next[-1]
    for i, source in enumerate(taus_prev):
        denominator = (psmc_transition_cdf(upper, source, rho) -
                       psmc_transition_cdf(lower, source, rho))
        if denominator <= 0:
            raise ValueError("target branch has zero transition mass")
        for j, (left, right) in enumerate(zip(boundaries_next[:-1],
                                              boundaries_next[1:])):
            matrix[i, j] = (psmc_transition_cdf(right, source, rho) -
                            psmc_transition_cdf(left, source, rho)) / denominator
    return matrix


def forward_linearized(alpha_prev, Q, emissions):
    """Exact linear forward recursion for a type-A transition.

    The supplement writes the upper-triangular contribution using a simple
    suffix sum.  Here rows have already been conditioned on the allowed branch
    interval, so source-specific normalizers remain.  Propagating the weighted
    suffix by its adjacent-column ratio preserves the same factorization and
    agrees with dense multiplication.
    """
    alpha = np.asarray(alpha_prev, dtype=float)
    Q = np.asarray(Q, dtype=float)
    emissions = np.asarray(emissions, dtype=float)
    d = len(alpha)
    if Q.shape != (d, d) or emissions.shape != (d,):
        raise ValueError("incompatible forward arrays")
    below = np.zeros(d)
    above = np.zeros(d)
    for j in range(1, d):
        carried = 0.0
        if j > 1:
            if Q[0, j - 1] <= 0:
                raise ValueError("matrix lacks positive type-A structure")
            carried = Q[0, j] / Q[0, j - 1] * below[j - 1]
        below[j] = alpha[j - 1] * Q[j - 1, j] + carried
    for j in range(d - 2, -1, -1):
        carried = 0.0
        if j < d - 2:
            if Q[-1, j + 1] <= 0:
                raise ValueError("matrix lacks positive type-A structure")
            carried = Q[-1, j] / Q[-1, j + 1] * above[j + 1]
        above[j] = alpha[j + 1] * Q[j + 1, j] + carried
    out = np.empty(d)
    for j in range(d):
        out[j] = emissions[j] * (
            below[j] + alpha[j] * Q[j, j] + above[j]
        )
    return out


def type_b_transition(alpha_prev, boundaries_prev, boundaries_next,
                      mapped_intervals, rho=None):
    """Transfer mass along supplied hitchhiking interval mappings.

    This is only the deterministic bookkeeping part of type B. Production
    SINGER also creates uncovered states and applies their HMM emissions.
    """
    del boundaries_prev, rho
    out = np.zeros(len(boundaries_next) - 1)
    for source, target in enumerate(mapped_intervals):
        if target is not None:
            out[target] += alpha_prev[source]
    return out


def type_c_transition(alpha_prev, taus_prev, boundaries_next):
    """Type-C transition conditioned on a new recombination (rho -> infinity)."""
    return np.asarray(alpha_prev) @ time_transition_matrix(
        None, taus_prev, boundaries_next, rho=1e12)


def recombination_time_median(lower, upper, recoalescence_time):
    """Median recombination time under truncated density proportional to exp(x)."""
    _check_interval(lower, upper)
    if recoalescence_time < upper:
        raise ValueError("recoalescence must not precede the upper bound")
    return log((exp(lower) + exp(upper)) / 2)


# ARG rescaling (Supplement B.3)

def compute_arg_length_in_window(branches, window_lower, window_upper):
    """Span-weighted branch length overlapping one time window."""
    return sum(span * max(0.0, min(hi, window_upper) - max(lo, window_lower))
               for span, lo, hi in branches)


def partition_time_axis(branches, J=100):
    """Choose J windows with equal span-weighted ARG length."""
    if not branches or J < 1:
        raise ValueError("non-empty branches and positive J are required")
    endpoints = sorted({z for _, lo, hi in branches for z in (lo, hi)})
    if endpoints[0] < 0:
        raise ValueError("branch times must be non-negative")
    total = compute_arg_length_in_window(branches, endpoints[0], endpoints[-1])
    if total <= 0:
        raise ValueError("ARG length must be positive")
    targets = np.linspace(0, total, J + 1)
    result = [endpoints[0]]
    cumulative = 0.0
    target_index = 1
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        rate = sum(span for span, lo, hi in branches if lo < right and hi > left)
        segment = rate * (right - left)
        while target_index < J and cumulative + segment >= targets[target_index]:
            result.append(left + (targets[target_index] - cumulative) / rate)
            target_index += 1
        cumulative += segment
    result.append(endpoints[-1])
    return np.asarray(result)


def count_mutations_per_window(mutations, boundaries):
    """Fractionally assign each mapped mutation across its carrier branch."""
    boundaries = np.asarray(boundaries)
    counts = np.zeros(len(boundaries) - 1)
    for lower, upper in mutations:
        _check_interval(lower, upper)
        for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            overlap = max(0.0, min(upper, right) - max(lower, left))
            counts[i] += overlap / (upper - lower)
    return counts


def compute_scaling_factors(counts, total_arg_length, theta, J):
    """Return ``c_i = 2 J m_i / (theta L(G))`` (Supplement B.3.1)."""
    if total_arg_length <= 0 or theta <= 0 or J < 1:
        raise ValueError("ARG length, theta, and J must be positive")
    counts = np.asarray(counts, dtype=float)
    if len(counts) != J or np.any(counts < 0):
        raise ValueError("counts must be J non-negative values")
    return 2 * J * counts / (theta * total_arg_length)


def rescale_times(node_times, boundaries, scaling_factors):
    """Apply SINGER's continuous piecewise-linear monotone time map."""
    boundaries = np.asarray(boundaries, dtype=float)
    factors = np.asarray(scaling_factors, dtype=float)
    if len(boundaries) != len(factors) + 1 or np.any(factors < 0):
        raise ValueError("one non-negative factor is required per window")
    new_boundaries = np.r_[0.0, np.cumsum(factors * np.diff(boundaries))]
    answer = {}
    for node, time in node_times.items():
        if time < boundaries[0] or time > boundaries[-1]:
            raise ValueError("node time lies outside the grid")
        index = min(np.searchsorted(boundaries, time, side="right") - 1,
                    len(factors) - 1)
        answer[node] = (new_boundaries[index] +
                        factors[index] * (time - boundaries[index]))
    return answer


def count_mutations_with_rate_variation(branches, mutations, boundaries,
                                         mutation_rate_map):
    """Expected and observed counts under an integrated genomic rate map."""
    boundaries = np.asarray(boundaries)
    expected = np.zeros(len(boundaries) - 1)
    observed = np.zeros_like(expected)
    for start, end, lower, upper in branches:
        if end <= start:
            raise ValueError("genomic branch spans must be positive")
        integrated_rate, _ = quad(mutation_rate_map, start, end)
        for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            overlap = max(0.0, min(upper, right) - max(lower, left))
            expected[i] += integrated_rate * overlap
    for _position, lower, upper in mutations:
        _check_interval(lower, upper)
        for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            overlap = max(0.0, min(upper, right) - max(lower, left))
            observed[i] += overlap / (upper - lower)
    return expected, observed


# Single-tree SPR analogue and SGPR ratio (Supplement B.4)

class SimpleTree:
    """Minimal rooted binary tree used only to illustrate one marginal SPR."""
    def __init__(self, parent, time):
        self.parent = dict(parent)
        self.time = dict(time)
        self.children = {}
        for child, parent_node in self.parent.items():
            if parent_node is not None:
                self.children.setdefault(parent_node, []).append(child)

    def branches(self):
        return [(child, parent, self.time[parent] - self.time[child])
                for child, parent in self.parent.items() if parent is not None]

    def height(self):
        return max(self.time.values())


def spr_move(tree, cut_node, new_parent, new_time):
    """Apply a valid single-tree subtree-prune-and-regraft move.

    ``new_parent`` identifies the child endpoint of the target branch. This is
    a pedagogical operation, not SINGER's chromosome-spanning SGPR proposal.
    """
    if cut_node not in tree.parent or tree.parent[cut_node] is None:
        raise ValueError("cut_node must have a parent")
    if new_parent not in tree.time or new_parent == cut_node:
        raise ValueError("invalid target branch")
    descendants = {cut_node}
    frontier = [cut_node]
    while frontier:
        node = frontier.pop()
        for child in tree.children.get(node, []):
            descendants.add(child)
            frontier.append(child)
    if new_parent in descendants:
        raise ValueError("cannot regraft into the pruned subtree")
    target_upper = tree.time.get(tree.parent.get(new_parent), np.inf)
    if not tree.time[new_parent] < new_time < target_upper:
        raise ValueError("new_time must lie strictly inside the target branch")
    if new_time <= tree.time[cut_node]:
        raise ValueError("new parent must be older than the pruned subtree root")

    parent = dict(tree.parent)
    times = dict(tree.time)
    old_parent = parent[cut_node]
    old_grandparent = parent.get(old_parent)
    siblings = [c for c in tree.children.get(old_parent, []) if c != cut_node]
    if len(siblings) != 1:
        raise ValueError("the teaching move requires a binary old parent")
    sibling = siblings[0]
    parent[sibling] = old_grandparent
    parent.pop(old_parent)
    times.pop(old_parent)

    target_parent = parent.get(new_parent)
    internal = max(times) + 1
    times[internal] = new_time
    parent[new_parent] = internal
    parent[cut_node] = internal
    parent[internal] = target_parent
    return SimpleTree(parent, times)


def select_cut(tree, rng=None):
    """Use SINGER's time-slice cut rule from Supplement B.4.1."""
    rng = np.random.default_rng() if rng is None else rng
    cut_time = rng.uniform(0, tree.height())
    crossing = [child for child, parent, _ in tree.branches()
                if tree.time[child] <= cut_time < tree.time[parent]]
    if not crossing:
        raise ValueError("no branch crosses the sampled time")
    return int(rng.choice(crossing)), float(cut_time)


def sgpr_acceptance_ratio(old_tree_height, new_tree_height):
    """Approximate SGPR acceptance ``min(1,h(Psi_x)/h(Psi'_x))``."""
    if old_tree_height <= 0 or new_tree_height <= 0:
        raise ValueError("tree heights must be positive")
    return min(1.0, old_tree_height / new_tree_height)


def simulate_tree_height_variability(n, n_replicates=10_000, rng=None):
    """Draw standard-coalescent heights for an acceptance illustration."""
    if n < 2 or n_replicates < 1:
        raise ValueError("n >= 2 and n_replicates >= 1 are required")
    rng = np.random.default_rng() if rng is None else rng
    heights = np.zeros(n_replicates)
    for k in range(n, 1, -1):
        heights += rng.exponential(1 / (k * (k - 1) / 2), n_replicates)
    return heights


def demo():
    branch = (0.2, 0.8)
    tau = representative_time(*branch, 50)
    print(f"representative joining time: {tau:.6f}")
    print(f"joining probability: {joining_prob_approx(*branch, 50):.6g}")
    print("SGPR height-ratio acceptance:", sgpr_acceptance_ratio(1.9, 2.0))


if __name__ == "__main__":
    demo()
