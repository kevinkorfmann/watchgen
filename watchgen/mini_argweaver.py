"""
Mini-implementation of the ARGweaver algorithm for Bayesian sampling of
Ancestral Recombination Graphs via the Discrete SMC.

ARGweaver is a Bayesian method for sampling Ancestral Recombination Graphs
(ARGs) from their posterior distribution given observed sequence data. It
uses an iterative threading algorithm with a crucial design choice:
discretizing time into a finite grid, making the HMM state space finite
and enabling exact forward-backward computation.

The algorithm uses a single HMM whose states are (node, time-index) pairs.
Each state specifies both where and when the new lineage joins the partial
tree. The five key components are:

1. Time Discretization -- A log-spaced time grid that makes the state space
   finite while concentrating resolution near the present.
2. Transition Probabilities -- The HMM transition matrix encoding
   recombination and re-coalescence under the discrete SMC.
3. Emission Probabilities -- A parsimony-based likelihood that scores how
   well each state explains the observed data.
4. MCMC Sampling -- Subtree re-threading with Gibbs updates: remove one
   chromosome, re-sample its path through the ARG using forward-backward.
5. Switch Transitions -- Special transition matrices at recombination
   breakpoints in the partial ARG.

References
----------
Rasmussen, Hubisz, Gronau, Siepel (2014). Genome-wide inference of
ancestral recombination graphs. PLoS Genetics 10(5): e1004342.
"""

import numpy as np
import random
from math import exp, log


# ============================================================================
# Chapter 1: Time Discretization
# ============================================================================

def get_time_point(i, ntimes, maxtime, delta=0.01):
    """
    Compute the i-th discretized time point.

    Parameters
    ----------
    i : int
        Index of the time point (0 <= i <= ntimes-1).
    ntimes : int
        Total number of time intervals (ntimes-1 is the last index).
    maxtime : float
        Maximum time in generations.
    delta : float
        Controls log-spacing. Smaller delta -> more uniform spacing.
        Larger delta -> more concentration near present.

    Returns
    -------
    float
        The i-th time point in generations.

    Examples
    --------
    >>> get_time_point(0, 19, 160000, 0.01)
    0.0
    >>> round(get_time_point(1, 19, 160000, 0.01), 1)
    52.6
    >>> round(get_time_point(19, 19, 160000, 0.01), 1)
    160000.0
    """
    return (exp(i / float(ntimes) * log(1 + delta * maxtime)) - 1) / delta


def get_time_points(ntimes=20, maxtime=160000, delta=0.01):
    """
    Compute all discretized time points.

    Parameters
    ----------
    ntimes : int
        Number of time points (including t_0 = 0).
    maxtime : float
        Maximum time in generations.
    delta : float
        Controls log-spacing.

    Returns
    -------
    list of float
        The ntimes time points.

    Examples
    --------
    >>> times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
    >>> len(times)
    20
    >>> times[0]
    0.0
    >>> round(times[-1], 1)
    160000.0
    """
    return [get_time_point(i, ntimes - 1, maxtime, delta)
            for i in range(ntimes)]


def get_time_steps(times):
    """
    Compute time step sizes from time points.

    Parameters
    ----------
    times : list of float
        Discretized time points.

    Returns
    -------
    list of float
        Time steps: times[i+1] - times[i] for each interval.
    """
    ntimes = len(times) - 1
    return [times[i+1] - times[i] for i in range(ntimes)]


def get_coal_times(times):
    """
    Compute coal times (geometric mean midpoints) for each interval.

    The coal_times list interleaves boundary times and midpoints:
      [t_0, mid_0, t_1, mid_1, ..., t_{n-1}, mid_{n-1}, t_n]

    This interleaved structure is used internally for computing
    transition probabilities.

    Parameters
    ----------
    times : list of float
        Discretized time points (length ntimes).

    Returns
    -------
    list of float
        Interleaved boundary times and midpoints (length 2*ntimes - 1).

    Examples
    --------
    >>> times = [0.0, 100.0, 1000.0, 10000.0]
    >>> ct = get_coal_times(times)
    >>> len(ct)
    7
    >>> round(ct[0], 1)
    0.0
    >>> round(ct[1], 1)
    9.0
    >>> round(ct[2], 1)
    100.0
    """
    ntimes = len(times) - 1
    times2 = []
    for i in range(ntimes):
        times2.append(times[i])
        times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
    times2.append(times[ntimes])
    return times2


def get_coal_time_steps(times):
    """
    Compute the effective time step for coalescence at each time point.

    For time point i, the coal time step spans from the midpoint below
    to the midpoint above:
      coal_step[i] = mid[i] - mid[i-1]

    Parameters
    ----------
    times : list of float
        Discretized time points.

    Returns
    -------
    list of float
        Coal time steps for each time index.
    """
    ntimes = len(times) - 1
    times2 = []
    for i in range(ntimes):
        times2.append(times[i])
        times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
    times2.append(times[ntimes])

    coal_time_steps = []
    for i in range(0, len(times2), 2):
        coal_time_steps.append(
            times2[min(i + 1, len(times2) - 1)] -
            times2[max(i - 1, 0)]
        )
    return coal_time_steps


def max_step_ratio(ntimes, maxtime=160000, delta=0.01):
    """
    Compute the maximum ratio of consecutive time steps.

    For each grid, compute the time steps and then find the maximum ratio
    of consecutive steps. Because the grid is (approximately) geometric,
    consecutive steps grow by a nearly constant factor.

    Parameters
    ----------
    ntimes : int
        Number of time points.
    maxtime : float
        Maximum time in generations.
    delta : float
        Controls log-spacing.

    Returns
    -------
    float
        Maximum ratio of consecutive time steps.
    """
    times = get_time_points(ntimes=ntimes, maxtime=maxtime, delta=delta)
    steps = get_time_steps(times)
    ratios = [steps[i+1] / steps[i] for i in range(len(steps) - 1)
              if steps[i] > 0]
    return max(ratios)


def count_states(tree_dict, node_ages, root, times):
    """
    Count HMM states from a tree described as a parent-child dict.

    Parameters
    ----------
    tree_dict : dict
        Maps child -> parent (None for root).
    node_ages : dict
        Maps node name -> age (float).
    root : str
        Name of the root node.
    times : list of float
        Discretized time points.

    Returns
    -------
    int
        Total number of valid (branch, time_index) states.
    """
    ntimes = len(times) - 1
    total = 0
    for child, parent in tree_dict.items():
        child_age = node_ages[child]
        i = next(k for k, t in enumerate(times) if t >= child_age)
        if parent is not None:
            parent_age = node_ages[parent]
            while i < ntimes and times[i] <= parent_age:
                total += 1
                i += 1
        else:
            while i < ntimes:
                total += 1
                i += 1
    return total


# ============================================================================
# Chapter 2: Transition Probabilities
# ============================================================================

def logsumexp(x):
    """Numerically stable log-sum-exp.

    Computes log(sum(exp(x))) by factoring out the maximum value:
    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    This prevents overflow (if max(x) is large) and underflow (if
    all x values are very negative).

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    float
        log(sum(exp(x))).
    """
    m = np.max(x)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(x - m)))


def verify_transition_matrix(trans, tol=1e-6):
    """
    Verify that transition matrix rows sum to 1.

    Parameters
    ----------
    trans : numpy.ndarray
        Transition probability matrix.
    tol : float
        Tolerance for row sums.

    Returns
    -------
    bool
        True if all rows sum to 1 within tolerance.
    """
    row_sums = trans.sum(axis=1)
    ok = np.allclose(row_sums, 1.0, atol=tol)
    return ok


def recoal_distribution(nbranches_const, Ne, times):
    """
    Compute the re-coalescence PMF across time indices,
    starting from time 0, for a constant population size
    and constant lineage count.

    Parameters
    ----------
    nbranches_const : int
        Constant number of branches.
    Ne : float
        Effective population size.
    times : list of float
        Discretized time points.

    Returns
    -------
    list of float
        Probability mass function over time indices.
    """
    coal_times_list = get_coal_times(times)
    ntimes = len(times) - 1
    pmf = []
    cum_log_surv = 0.0

    for j in range(ntimes):
        nbr = nbranches_const
        # Non-overlapping interval: from boundary point to next boundary point
        t_lo = times[j]
        t_hi = times[j + 1]
        A = (t_hi - t_lo) * nbr
        coal_prob = 1.0 - exp(-A / Ne)
        pmf.append(exp(cum_log_surv) * coal_prob)
        cum_log_surv += log(1.0 - coal_prob) if coal_prob < 1.0 else -1e20

    return pmf


def build_simple_transition_matrix(ntimes, nbranches, ncoals,
                                   popsizes, rho, treelen, times):
    """
    Build the normal transition matrix for a simplified model
    where all states share the same lineage counts.

    Returns a matrix indexed by time index (ignoring branch identity
    for simplicity, since all branches at the same time contribute
    equally in the rank-1 structure).

    Parameters
    ----------
    ntimes : int
        Number of time intervals.
    nbranches : list of int
        Number of branches at each time interval.
    ncoals : list of int
        Number of coalescence points at each time index.
    popsizes : list of float
        Population sizes at each time index.
    rho : float
        Per-base recombination rate.
    treelen : float
        Total tree length.
    times : list of float
        Discretized time points.

    Returns
    -------
    numpy.ndarray
        Transition probability matrix.
    """
    time_steps = get_time_steps(times)
    coal_times_list = get_coal_times(times)
    root_idx = ntimes - 1
    no_recomb = exp(-rho * treelen)

    nstates = sum(ncoals[j] for j in range(ntimes))

    q = np.zeros(nstates)
    state_map = []
    idx = 0
    for j in range(ntimes):
        for b in range(ncoals[j]):
            state_map.append(j)
            idx += 1

    for s_idx in range(nstates):
        j = state_map[s_idx]
        total = 0.0
        for k in range(root_idx + 1):
            if j < k:
                continue
            recomb_p = (nbranches[k] * time_steps[k] / treelen
                        * (1 - no_recomb))
            surv = 1.0
            for m in range(k, j):
                A = (coal_times_list[2*m+1] - coal_times_list[2*m]) * nbranches[m]
                if m > k:
                    A += (coal_times_list[2*m] - coal_times_list[2*m-1]) * nbranches[max(m-1, 0)]
                surv *= exp(-A / popsizes[m])
            A = (coal_times_list[2*j+1] - coal_times_list[2*j]) * nbranches[j]
            if j > k:
                A += (coal_times_list[2*j] - coal_times_list[2*j-1]) * nbranches[max(j-1, 0)]
            cp = 1.0 - exp(-A / popsizes[j])
            total += recomb_p * surv * cp / max(ncoals[j], 1)
        q[s_idx] = total

    T = np.outer(np.ones(nstates), q) + no_recomb * np.eye(nstates)
    return T


# ============================================================================
# Chapter 3: Emission Probabilities
# ============================================================================

def felsenstein_pruning(tree_children, tree_parent, leaf_bases,
                        mu, branch_lengths):
    """
    Felsenstein's pruning algorithm for a 4-base alphabet.

    Returns log-likelihood at the root for each possible root base.

    Parameters
    ----------
    tree_children : dict
        Maps internal node -> list of children.
    tree_parent : dict
        Maps child -> parent.
    leaf_bases : dict
        Maps leaf -> observed base.
    mu : float
        Mutation rate.
    branch_lengths : dict
        Maps child -> branch length to parent.

    Returns
    -------
    dict
        Maps base -> log-likelihood at root.
    """
    bases = ['A', 'C', 'G', 'T']
    L = {}

    def compute(node):
        if node in leaf_bases:
            L[node] = {}
            for b in bases:
                L[node][b] = 0.0 if b == leaf_bases[node] else -float('inf')
            return

        for child in tree_children[node]:
            compute(child)

        L[node] = {}
        for b in bases:
            ll = 0.0
            for child in tree_children[node]:
                blen = branch_lengths[child]
                p_no_mut = exp(-mu * blen)
                p_mut = (1.0 - exp(-mu * blen)) / 3.0
                child_vals = []
                for cb in bases:
                    p_trans = p_no_mut if cb == b else p_mut
                    if L[child][cb] > -float('inf'):
                        child_vals.append(log(p_trans) + L[child][cb])
                if child_vals:
                    mx = max(child_vals)
                    ll += mx + log(sum(exp(v - mx) for v in child_vals))
                else:
                    ll += -float('inf')
            L[node][b] = ll

    compute('root')
    return L['root']


# ============================================================================
# Chapter 4: MCMC Sampling
# ============================================================================

def sample_tree(k, popsizes, times):
    """
    Sample a coalescent tree using a discrete-time coalescent.

    Starting with k lineages at time 0, simulate coalescence events
    through the time grid. At each time interval, the coalescence rate
    depends on the number of lineages and the local population size.

    Parameters
    ----------
    k : int
        Number of lineages (chromosomes).
    popsizes : list of float
        Effective population size at each time interval.
    times : list of float
        Discretized time points.

    Returns
    -------
    list of float
        Coalescence times (one per coalescence event).
    """
    ntimes = len(times)
    coal_times = []

    timei = 0
    n = popsizes[timei]
    t = 0.0
    k2 = k

    while k2 > 1:
        coal_rate = (k2 * (k2 - 1) / 2) / float(n)
        t2 = random.expovariate(coal_rate)

        if timei < ntimes - 2 and t + t2 > times[timei + 1]:
            timei += 1
            t = times[timei]
            n = popsizes[timei]
            continue

        t += t2
        coal_times.append(t)
        k2 -= 1

    return coal_times


def sample_next_recomb(treelen, rho):
    """
    Sample the distance to the next recombination event.

    The waiting time (in base pairs) is exponential with rate
    rho * treelen.

    Parameters
    ----------
    treelen : float
        Total tree length.
    rho : float
        Per-base recombination rate.

    Returns
    -------
    float
        Distance in base pairs to the next recombination.
    """
    rate = max(treelen * rho, rho)
    return random.expovariate(rate)


def sample_recomb_time(nbranches, time_steps, root_age_index):
    """
    Sample which time interval the recombination falls in.

    Probability is proportional to the amount of branch material
    at each time interval: nbranches[k] * time_steps[k].

    Parameters
    ----------
    nbranches : list of int
        Number of branches at each time interval.
    time_steps : list of float
        Time step sizes.
    root_age_index : int
        Time index of the root (recombination can only happen below).

    Returns
    -------
    int
        Time index of the recombination.
    """
    weights = [nbranches[i] * time_steps[i]
               for i in range(root_age_index + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]

    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return i
    return len(probs) - 1


def simplified_mcmc(n_haps=4, n_sites=100, n_iters=1000,
                    Ne=10000, mu=1.4e-8, rho=1e-8,
                    ntimes=20, maxtime=160000, delta=0.01):
    """
    Simplified ARGweaver MCMC loop for demonstration.

    This implementation tracks only the total tree length at a
    fixed site across iterations, omitting the full ARG data
    structure for clarity.

    Parameters
    ----------
    n_haps : int
        Number of haplotypes.
    n_sites : int
        Number of genomic sites.
    n_iters : int
        Number of MCMC iterations.
    Ne : float
        Effective population size.
    mu : float
        Mutation rate.
    rho : float
        Recombination rate.
    ntimes : int
        Number of time points.
    maxtime : float
        Maximum time in generations.
    delta : float
        Controls log-spacing.

    Returns
    -------
    tree_lengths : list of float
        Total tree length at site 0, one per iteration.
    """
    times = get_time_points(ntimes=ntimes, maxtime=maxtime, delta=delta)
    popsizes = [Ne] * (ntimes - 1)

    coal_events = sorted(sample_tree(n_haps, popsizes, times))

    tree_lengths = []

    for iteration in range(n_iters):
        remove_idx = random.randint(0, n_haps - 1)

        if coal_events:
            event_idx = min(remove_idx, len(coal_events) - 1)
            coal_events.pop(event_idx)

            k_remaining = n_haps - len(coal_events)
            if k_remaining >= 2:
                new_coal = sample_tree(k_remaining, popsizes, times)
                if new_coal:
                    coal_events.append(new_coal[0])
                    coal_events.sort()

        total_length = 0.0
        k = n_haps
        prev_t = 0.0
        for ct in coal_events:
            total_length += k * (ct - prev_t)
            prev_t = ct
            k -= 1
        if coal_events:
            total_length += 1 * (times[-1] - coal_events[-1])

        tree_lengths.append(total_length)

    return tree_lengths


def harmonic(n):
    """Compute the n-th harmonic number H_n = sum(1/i for i=1..n).

    Parameters
    ----------
    n : int
        Number of terms.

    Returns
    -------
    float
        The n-th harmonic number.
    """
    return sum(1.0 / i for i in range(1, n + 1))


# ============================================================================
# Demo function
# ============================================================================

def demo():
    """Demonstrate the core ARGweaver components."""
    print("=" * 60)
    print("ARGweaver Mini-Implementation Demo")
    print("=" * 60)

    # --- Time Discretization ---
    print("\n--- Time Discretization ---")
    times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
    for i, t in enumerate(times):
        step = times[i] - times[i-1] if i > 0 else 0
        print(f"t[{i:2d}] = {t:10.1f}   step = {step:10.1f}")

    # Delta parameter effect
    print("\nEffect of delta on first 5 time points:")
    for delta in [0.001, 0.01, 0.1]:
        t = get_time_points(ntimes=20, maxtime=160000, delta=delta)
        print(f"delta={delta}: first 5 times = "
              f"{[round(x, 1) for x in t[:5]]}")

    # Time steps
    time_steps = get_time_steps(times)
    print(f"\nNumber of time steps: {len(time_steps)}")
    print(f"Sum of time steps: {sum(time_steps):.1f}")

    # Coal times
    coal_times = get_coal_times(times)
    print(f"\nCoal times (interleaved): {len(coal_times)} entries")
    for i in range(min(5, len(times) - 1)):
        t_lo = times[i]
        t_hi = times[i + 1]
        t_mid = coal_times[2 * i + 1]
        arith = (t_lo + t_hi) / 2
        print(f"Interval [{t_lo:.1f}, {t_hi:.1f}): "
              f"geometric mid = {t_mid:.1f}, arithmetic mid = {arith:.1f}")

    # Coal time steps
    coal_time_steps_list = get_coal_time_steps(times)
    print(f"\nCoal time steps: {len(coal_time_steps_list)} entries")

    # Max step ratios
    print("\nMax step ratios for different grid sizes:")
    for nt in [10, 20, 40]:
        r = max_step_ratio(nt)
        print(f"n_t = {nt:3d}:  max step ratio = {r:.4f}")

    # State counting example
    print("\nState counting for tree ((A,B),(C,D)):")
    tree = {
        'A': 'AB', 'B': 'AB',
        'C': 'CD', 'D': 'CD',
        'AB': 'root', 'CD': 'root',
        'root': None
    }
    ages = {
        'A': times[0], 'B': times[0],
        'C': times[0], 'D': times[0],
        'AB': times[3], 'CD': times[5],
        'root': times[8]
    }
    n_states = count_states(tree, ages, 'root', times)
    print(f"Total states: {n_states}")

    # --- Transition Probabilities ---
    print("\n--- Transition Probabilities ---")
    print(f"logsumexp([1.0, 2.0]) = {logsumexp(np.array([1.0, 2.0])):.6f}")
    print(f"Expected: {2.0 + np.log(1 + np.exp(-1.0)):.6f}")

    # Verify a simple transition matrix
    T = np.array([[0.7, 0.2, 0.1],
                   [0.3, 0.5, 0.2],
                   [0.1, 0.1, 0.8]])
    print(f"Transition matrix valid: {verify_transition_matrix(T)}")

    # Re-coalescence distribution
    pmf = recoal_distribution(10, 10000, times)
    mode_idx = int(np.argmax(pmf))
    print(f"\nRe-coalescence distribution mode at index {mode_idx} "
          f"(t = {times[mode_idx]:.1f})")

    # --- Emission Cases ---
    print("\n--- Emission Cases ---")
    mu = 1.4e-8
    t = 1000
    t1 = 2000
    t2 = 1000

    case1 = -mu * t
    case2 = log(1.0/3 - 1.0/3 * exp(-mu * t))
    case3 = log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                * exp(-mu * (t + t1 - t2)))
    t3 = t
    case5 = log((1 - exp(-mu * t2)) * (1 - exp(-mu * t3))
                / (1 - exp(-mu * t1))
                * exp(-mu * (t + t1 - t2 - t3)))

    print(f"Case 1 (no mutation):    log P = {case1:.6e}")
    print(f"Case 2 (new branch mut): log P = {case2:.6e}")
    print(f"Case 3 (lower segment):  log P = {case3:.6e}")
    print(f"Case 5 (two mutations):  log P = {case5:.6e}")
    print(f"Case 1 / Case 2: {exp(case1 - case2):.2e}")
    print(f"Case 1 / Case 5: {exp(case1 - case5):.2e}")

    # --- MCMC Sampling ---
    print("\n--- MCMC Sampling ---")
    random.seed(42)
    popsizes = [10000.0] * 20
    coal_events = sample_tree(5, popsizes, times)
    print(f"Sampled tree with 5 lineages: {len(coal_events)} coalescence events")
    for i, ct in enumerate(coal_events):
        print(f"  coalescence {i+1}: t = {ct:.1f}")

    # Recombination sampling
    random.seed(42)
    d = sample_next_recomb(1000.0, 1e-8)
    print(f"\nDistance to next recombination: {d:.1f} bp")

    # Harmonic numbers
    n = 10
    Hn = harmonic(n)
    print(f"\nCoupon collector: n={n}, H_n={Hn:.4f}, "
          f"expected iterations = {n * Hn:.1f}")

    # Simplified MCMC
    random.seed(42)
    tree_lengths = simplified_mcmc(n_iters=100)
    mean_tl = sum(tree_lengths[-50:]) / 50
    print(f"\nSimplified MCMC (100 iters): "
          f"mean tree length (last 50) = {mean_tl:.0f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
