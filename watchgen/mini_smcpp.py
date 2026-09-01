"""Small, faithful building blocks for SMC++.

SMC++ tracks the time to the most recent common ancestor (TMRCA) of a
*distinguished pair* of haplotypes. Extra, undistinguished haplotypes enter
through the conditioned sample-frequency spectrum (CSFS), not by changing the
pair's coalescence hazard. The production program evaluates the CSFS with
closed-form matrix identities and uses the continuous-time two-locus kernel of
Hobolth and Jensen for transitions.

This pedagogical module implements the same statistical objects in a form that
is practical for small samples:

* an exact partition-state calculation of the one-population CSFS;
* the mutation transform used by ``src/conditioned_sfs.cpp``;
* interval averaging over the distinguished-pair coalescence density;
* the exact three-state two-locus kernel in ``src/transition.cpp``;
* its constant-demography discretization and a scaled forward algorithm.

The partition state space grows as a Bell number, so this is deliberately not a
replacement for the highly optimized SMC++ implementation.

Reference
---------
Terhorst, J., Kamm, J. A., & Song, Y. S. (2017). Robust and scalable
inference of population history from hundreds of unphased whole genomes.
Nature Genetics, 49(2), 303-309. https://doi.org/10.1038/ng.3748
"""

from collections import deque
from functools import cache
from itertools import combinations, pairwise

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.linalg import expm


def expected_first_coalescence(n, N):
    """Return the mean first-coalescence time for ``n`` haploid lineages.

    ``N`` is the diploid effective size, so each pair coalesces at rate
    ``1 / (2N)`` per generation. The total rate is therefore
    ``binom(n, 2) / (2N)``.
    """
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer of at least two")
    if not np.isfinite(N) or N <= 0:
        raise ValueError("N must be finite and positive")
    return 2.0 * N / (n * (n - 1) / 2.0)


def _canonical_partition(blocks):
    """Represent a set partition as a stable tuple of sorted tuples."""
    return tuple(
        sorted((tuple(sorted(block)) for block in blocks), key=lambda x: (x[0], x))
    )


def _merge_blocks(partition, i, j):
    blocks = list(partition)
    merged = tuple(sorted(blocks[i] + blocks[j]))
    return _canonical_partition(
        [block for k, block in enumerate(blocks) if k not in (i, j)] + [merged]
    )


def _distinguished_block_indices(partition):
    i0 = next(i for i, block in enumerate(partition) if 0 in block)
    i1 = next(i for i, block in enumerate(partition) if 1 in block)
    return i0, i1


def _successors(partition, forbid_distinguished_merge):
    i0, i1 = _distinguished_block_indices(partition)
    for i, j in combinations(range(len(partition)), 2):
        if forbid_distinguished_merge and {i, j} == {i0, i1}:
            continue
        yield _merge_blocks(partition, i, j)


def _reachable_partitions(initial, forbid_distinguished_merge):
    seen = set(initial)
    queue = deque(initial)
    while queue:
        state = queue.popleft()
        for nxt in _successors(state, forbid_distinguished_merge):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return tuple(sorted(seen, key=lambda p: (-len(p), p)))


def _partition_generator(states, forbid_distinguished_merge):
    """Kingman generator with unit rate for each allowed pair of blocks."""
    index = {state: i for i, state in enumerate(states)}
    Q = np.zeros((len(states), len(states)))
    for i, state in enumerate(states):
        for nxt in _successors(state, forbid_distinguished_merge):
            Q[i, index[nxt]] += 1.0
            Q[i, i] -= 1.0
    return Q


def _branch_rewards(states, n_undistinguished):
    """Count extant branches by descendant category ``(a, b)``."""
    total = n_undistinguished + 2
    rewards = np.zeros((len(states), 3 * (n_undistinguished + 1)))
    for i, partition in enumerate(states):
        if len(partition) == 1:
            continue
        for block in partition:
            a = int(0 in block) + int(1 in block)
            b = len(block) - a
            if 0 < a + b < total:
                rewards[i, a * (n_undistinguished + 1) + b] += 1.0
    return rewards


def _finite_occupation(p0, Q, rewards, duration, coal_rate):
    """Evolve a row distribution and integrate its branch rewards."""
    if duration == 0:
        return p0.copy(), np.zeros(rewards.shape[1])
    n, r = rewards.shape
    augmented = np.zeros((n + r, n + r))
    augmented[:n, :n] = coal_rate * Q
    augmented[:n, n:] = rewards
    E = expm(duration * augmented)
    return p0 @ E[:n, :n], p0 @ E[:n, n:]


def _force_distinguished_coalescence(partition):
    i0, i1 = _distinguished_block_indices(partition)
    if i0 == i1:
        return partition
    return _merge_blocks(partition, i0, i1)


@cache
def _conditioned_state_system(n_undistinguished):
    total = n_undistinguished + 2
    initial = _canonical_partition([(i,) for i in range(total)])
    below_states = _reachable_partitions((initial,), True)
    below_Q = _partition_generator(below_states, True)
    below_rewards = _branch_rewards(below_states, n_undistinguished)

    forced = tuple({_force_distinguished_coalescence(s) for s in below_states})
    above_states = _reachable_partitions(forced, False)
    above_Q = _partition_generator(above_states, False)
    above_rewards = _branch_rewards(above_states, n_undistinguished)
    return (
        initial,
        below_states,
        below_Q,
        below_rewards,
        above_states,
        above_Q,
        above_rewards,
    )


def conditioned_sfs(tau, n_undistinguished, relative_size=1.0):
    """Expected branch lengths conditional on distinguished-pair TMRCA ``tau``.

    The returned ``(3, n_undistinguished + 1)`` array is the CSFS. Entry
    ``[a, b]`` is the expected total branch length subtending ``a`` of the two
    distinguished haplotypes and ``b`` undistinguished haplotypes.

    The calculation is exact for a constant population but exponential in
    sample size. It is intended for small pedagogical examples (normally no
    more than six undistinguished haplotypes).
    """
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    if int(n_undistinguished) != n_undistinguished or n_undistinguished < 0:
        raise ValueError("n_undistinguished must be a non-negative integer")
    if not np.isfinite(relative_size) or relative_size <= 0:
        raise ValueError("relative_size must be finite and positive")

    (
        initial,
        below_states,
        below_Q,
        below_rewards,
        above_states,
        above_Q,
        above_rewards,
    ) = _conditioned_state_system(int(n_undistinguished))
    coal_rate = 1.0 / relative_size
    p0 = np.zeros(len(below_states))
    p0[below_states.index(initial)] = 1.0
    p_tau, below_lengths = _finite_occupation(
        p0, below_Q, below_rewards, tau, coal_rate
    )

    above_index = {state: i for i, state in enumerate(above_states)}
    p_above = np.zeros(len(above_states))
    for probability, state in zip(p_tau, below_states):
        p_above[above_index[_force_distinguished_coalescence(state)]] += probability

    transient = np.array([len(state) > 1 for state in above_states])
    if np.any(transient):
        Qtt = coal_rate * above_Q[np.ix_(transient, transient)]
        expected_occupation = np.linalg.solve(-Qtt.T, p_above[transient]).T
        above_lengths = expected_occupation @ above_rewards[transient]
    else:
        above_lengths = np.zeros(above_rewards.shape[1])

    csfs = (below_lengths + above_lengths).reshape(3, n_undistinguished + 1)
    csfs[np.abs(csfs) < 1e-12] = 0.0
    if np.min(csfs) < -1e-9:
        raise RuntimeError("negative branch length from conditioned coalescent")
    return np.maximum(csfs, 0.0)


def incorporate_theta(csfs, theta):
    """Convert CSFS branch lengths to the SMC++ emission distribution.

    This mirrors ``incorporate_theta`` in the original C++ source. The
    probability of at least one mutation is ``1 - exp(-theta * L)`` where
    ``L`` is total expected tree length; polymorphic outcomes are allocated in
    proportion to their CSFS branch lengths. The remaining mass is assigned
    to the monomorphic ancestral observation ``(0, 0)``.
    """
    csfs = np.asarray(csfs, dtype=float)
    if csfs.ndim != 2 or csfs.shape[0] != 3:
        raise ValueError("csfs must have shape (3, n_undistinguished + 1)")
    if np.any(~np.isfinite(csfs)) or np.any(csfs < -1e-12):
        raise ValueError("csfs must contain finite non-negative branch lengths")
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    lengths = np.maximum(csfs, 0.0).copy()
    lengths[0, 0] = 0.0
    lengths[2, -1] = 0.0
    total_length = lengths.sum()
    probabilities = np.zeros_like(lengths)
    if total_length > 0:
        probabilities = lengths * (-np.expm1(-theta * total_length) / total_length)
    probabilities[0, 0] = 1.0 - probabilities.sum()
    return probabilities


def pair_interval_probabilities(time_breaks, relative_size=1.0):
    """Stationary probabilities for discretized distinguished-pair TMRCA."""
    breaks = np.asarray(time_breaks, dtype=float)
    if breaks.ndim != 1 or len(breaks) < 2 or breaks[0] != 0:
        raise ValueError("time_breaks must be a one-dimensional array starting at zero")
    if np.any(np.diff(breaks) <= 0) or not np.isinf(breaks[-1]):
        raise ValueError("time_breaks must increase strictly and end at infinity")
    if not np.isfinite(relative_size) or relative_size <= 0:
        raise ValueError("relative_size must be finite and positive")
    survival = np.exp(-breaks / relative_size)
    survival[-1] = 0.0
    return survival[:-1] - survival[1:]


def interval_conditioned_sfs(
    time_breaks,
    n_undistinguished,
    relative_size=1.0,
    quadrature_order=12,
):
    """Average the CSFS within each hidden TMRCA interval."""
    breaks = np.asarray(time_breaks, dtype=float)
    pair_interval_probabilities(breaks, relative_size)
    if int(n_undistinguished) != n_undistinguished or n_undistinguished < 0:
        raise ValueError("n_undistinguished must be a non-negative integer")
    n_undistinguished = int(n_undistinguished)
    if int(quadrature_order) != quadrature_order or quadrature_order < 2:
        raise ValueError("quadrature_order must be an integer of at least two")
    leg_x, leg_w = leggauss(int(quadrature_order))
    lag_x, lag_w = laggauss(int(quadrature_order))
    result = []
    rate = 1.0 / relative_size
    for lo, hi in pairwise(breaks):
        average = np.zeros((3, n_undistinguished + 1))
        if np.isinf(hi):
            # Conditional on T >= lo, memorylessness gives T = lo + Exp(rate).
            for x, weight in zip(lag_x, lag_w):
                average += weight * conditioned_sfs(
                    lo + x / rate, n_undistinguished, relative_size
                )
        else:
            midpoint = 0.5 * (lo + hi)
            halfwidth = 0.5 * (hi - lo)
            denom = np.exp(-rate * lo) - np.exp(-rate * hi)
            for x, weight in zip(leg_x, leg_w):
                tau = midpoint + halfwidth * x
                density = rate * np.exp(-rate * tau) / denom
                average += (
                    halfwidth
                    * weight
                    * density
                    * conditioned_sfs(tau, n_undistinguished, relative_size)
                )
        result.append(average)
    return np.asarray(result)


def emission_probabilities(
    time_breaks,
    n_undistinguished,
    theta,
    relative_size=1.0,
    quadrature_order=12,
):
    """Return one CSFS emission table for every hidden TMRCA interval."""
    averaged = interval_conditioned_sfs(
        time_breaks,
        n_undistinguished,
        relative_size,
        quadrature_order,
    )
    return np.asarray([incorporate_theta(csfs, theta) for csfs in averaged])


def two_locus_kernel(duration, coal_rate, rho):
    """Exact Hobolth-Jensen three-state continuous-time transition kernel.

    The generator is the one used in ``src/transition.cpp`` of SMC++::

        linked --rho--> floating --coal_rate--> linked
                                --coal_rate--> absorbed

    State ``absorbed`` means that both marginal genealogies have coalesced.
    """
    for name, value in (
        ("duration", duration),
        ("coal_rate", coal_rate),
        ("rho", rho),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    generator = np.array(
        [
            [-rho, rho, 0.0],
            [coal_rate, -2.0 * coal_rate, coal_rate],
            [0.0, 0.0, 0.0],
        ]
    )
    kernel = expm(duration * generator)
    kernel[np.abs(kernel) < 1e-15] = 0.0
    return kernel


def _conditional_exponential_mean(lo, hi, rate):
    if np.isinf(hi):
        return lo + 1.0 / rate
    width = hi - lo
    return lo + 1.0 / rate - width / np.expm1(rate * width)


def constant_csc_transition_matrix(time_breaks, relative_size, rho):
    """Discretize SMC++'s continuous-time two-locus kernel.

    This is a direct constant-demography specialization of ``HJTransition`` in
    the original source. Like the production code, it represents each source
    interval by its conditional mean coalescence time. Unlike production
    SMC++, it does not add the tiny uniform numerical regularizer.
    """
    breaks = np.asarray(time_breaks, dtype=float)
    pair_interval_probabilities(breaks, relative_size)
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    rate = 1.0 / relative_size
    K = len(breaks) - 1
    P = np.zeros((K, K))
    boundary_absorption = np.zeros(K + 1)
    for i, boundary in enumerate(breaks):
        boundary_absorption[i] = (
            1.0 if np.isinf(boundary) else two_locus_kernel(boundary, rate, rho)[0, 2]
        )

    for j in range(K):
        if j > 0:
            P[j, :j] = np.diff(boundary_absorption[: j + 1])

        mean_t = _conditional_exponential_mean(breaks[j], breaks[j + 1], rate)
        if not np.isinf(breaks[j + 1]):
            floating = two_locus_kernel(mean_t, rate, rho)[0, 1]
            # HJTransition's Rj is the cumulative coalescence hazard across
            # the *entire source interval*, not merely from mean_t to its
            # upper boundary (src/transition.cpp, p_float construction).
            floating *= np.exp(-rate * (breaks[j + 1] - breaks[j]))
            for k in range(j + 1, K):
                survival = np.exp(-rate * (breaks[k] - breaks[j + 1]))
                coal_in_interval = (
                    1.0
                    if np.isinf(breaks[k + 1])
                    else -np.expm1(-rate * (breaks[k + 1] - breaks[k]))
                )
                P[j, k] = floating * survival * coal_in_interval

        P[j, j] = 1.0 - P[j].sum()

    if np.min(P) < -1e-10:
        raise RuntimeError("negative discretized transition probability")
    P = np.maximum(P, 0.0)
    P /= P.sum(axis=1, keepdims=True)
    return P


def forward_log_likelihood(observations, transitions, emissions, initial=None):
    """Scaled HMM likelihood for CSFS observations ``(a, b)``.

    Missing observations may be encoded by a negative value in either column;
    their emission probability is one in every hidden state.
    """
    observations = np.asarray(observations, dtype=int)
    transitions = np.asarray(transitions, dtype=float)
    emissions = np.asarray(emissions, dtype=float)
    if observations.ndim != 2 or observations.shape[1] != 2:
        raise ValueError("observations must have shape (sites, 2)")
    K = transitions.shape[0]
    if transitions.shape != (K, K) or emissions.shape[0] != K:
        raise ValueError("transition and emission state dimensions do not agree")
    if initial is None:
        initial = np.full(K, 1.0 / K)
    initial = np.asarray(initial, dtype=float)
    if (
        initial.shape != (K,)
        or np.any(initial < 0)
        or not np.isclose(initial.sum(), 1.0)
    ):
        raise ValueError("initial must be a probability vector over hidden states")
    if len(observations) == 0:
        return 0.0

    def emit(obs):
        a, b = obs
        if a < 0 or b < 0:
            return np.ones(K)
        if a >= emissions.shape[1] or b >= emissions.shape[2]:
            raise ValueError("observation is outside the CSFS emission table")
        return emissions[:, a, b]

    alpha = initial * emit(observations[0])
    scale = alpha.sum()
    if scale <= 0:
        return -np.inf
    alpha /= scale
    log_likelihood = np.log(scale)
    for obs in observations[1:]:
        alpha = (alpha @ transitions) * emit(obs)
        scale = alpha.sum()
        if scale <= 0:
            return -np.inf
        alpha /= scale
        log_likelihood += np.log(scale)
    return float(log_likelihood)


def composite_log_likelihood(datasets, transitions, emissions, initial=None):
    """Sum HMM log likelihoods for multiple distinguished-pair data sets.

    If the data sets reuse the same chromosome with different distinguished
    pairs, this sum is a composite likelihood because those terms are not
    independent.
    """
    return sum(
        forward_log_likelihood(data, transitions, emissions, initial)
        for data in datasets
    )


def cross_population_survival(t, split_time, ancestral_rate):
    """Survival of a TMRCA for a distinguished pair sampled apart.

    Under the clean-split model the two lineages cannot coalesce more recently
    than ``split_time``. After the split (backwards in time), their survival is
    governed by the ancestral pairwise coalescence rate. Production SMC++ also
    computes a joint conditioned SFS; this helper represents only the TMRCA
    support constraint.
    """
    if not np.isfinite(t) or t < 0 or not np.isfinite(split_time) or split_time < 0:
        raise ValueError("times must be finite and non-negative")
    if t <= split_time:
        return 1.0
    integral, _ = quad(ancestral_rate, split_time, t)
    if integral < -1e-10:
        raise ValueError("ancestral_rate must be non-negative")
    return float(np.exp(-max(integral, 0.0)))


def demo():
    """Run a small, deterministic SMC++ building-block demonstration."""
    breaks = np.array([0.0, 0.25, 0.75, 2.0, np.inf])
    n_undistinguished = 2
    theta = 0.02
    rho = 0.1
    emissions = emission_probabilities(
        breaks, n_undistinguished, theta, quadrature_order=8
    )
    transitions = constant_csc_transition_matrix(breaks, 1.0, rho)
    initial = pair_interval_probabilities(breaks)
    observations = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]])
    ll = forward_log_likelihood(observations, transitions, emissions, initial)
    print("Hidden-state probabilities:", initial)
    print("Transition row sums:", transitions.sum(axis=1))
    print("Emission row sums:", emissions.sum(axis=(1, 2)))
    print("Log likelihood:", ll)


if __name__ == "__main__":
    demo()
