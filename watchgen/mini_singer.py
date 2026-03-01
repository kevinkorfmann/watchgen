"""
Mini-implementation of the SINGER algorithm for ARG inference.

SINGER (Sampling and Inference of Genealogies with Recombination) is a Bayesian
method for sampling Ancestral Recombination Graphs (ARGs) from their posterior
distribution, given observed genetic variation data. It uses an iterative
threading algorithm: one haplotype at a time is "threaded" onto a growing
partial ARG using Hidden Markov Models.

The four gears of SINGER:

1. Branch Sampling -- An HMM that determines *which branch* each new lineage
   joins at each genomic position (the topological question).

2. Time Sampling -- A second HMM that determines *when* (at what time) the
   lineage joins, conditioned on the branch choice. This uses the PSMC
   transition density as a component.

3. ARG Rescaling -- A post-processing step that adjusts coalescence times to
   match the mutation clock, using a piecewise linear transformation based on
   observed vs expected mutation counts in time windows.

4. SGPR (Sub-Graph Pruning and Re-grafting) -- The MCMC update mechanism that
   explores the space of ARGs by removing and re-threading subsets of the
   genealogy. The acceptance ratio reduces to a simple tree height ratio.

References
----------
Deng, Nielsen, Song (2024). SINGER: Sampling and Inference of Genealogies
with Recombination.
"""

import numpy as np
from scipy.integrate import quad


# ============================================================================
# Chapter 1: Branch Sampling
# ============================================================================

def joining_probability_exact(x, y, tree_intervals):
    """Compute the exact probability of joining a branch spanning [x, y].

    Parameters
    ----------
    x, y : float
        Lower and upper time of the branch.
    tree_intervals : list of (lower, upper)
        All branch intervals in the marginal tree, defining lambda(t).

    Returns
    -------
    p : float
        Joining probability for this branch.
    """
    def lambda_psi(t):
        """Number of lineages at time t.

        Count how many branches in the tree span time t.
        This is a step function that decreases as we go deeper
        in time (further from the present), because lineages
        merge at coalescence events.
        """
        return sum(1 for lo, hi in tree_intervals if lo <= t < hi)

    def integrand_for_F_bar(t):
        """exp(-integral_0^t lambda(x) dx) = F_bar(t).

        This is the survival probability: the chance that the
        new lineage has NOT coalesced by time t.  We compute
        it by numerically integrating lambda from 0 to t.
        """
        integral, _ = quad(lambda_psi, 0, t)
        return np.exp(-integral)

    # Integrate the survival function over the branch interval [x, y].
    # This gives the probability of joining this specific branch.
    p, _ = quad(integrand_for_F_bar, x, y)
    return p


def lambda_approx(t, n):
    """Deterministic approximation for number of lineages at time t.

    This replaces the stochastic step-function lambda(t) with a
    smooth curve that tracks the expected lineage count.  The
    approximation improves as n increases (law of large numbers).
    """
    return n / (n + (1 - n) * np.exp(-t / 2))


def F_bar_approx(t, n):
    """Exceedance probability P(T > t) under the approximation.

    This is the probability that the new lineage has NOT coalesced
    by time t.  The formula was derived by integrating the smooth
    lambda approximation.
    """
    return np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**2


def f_approx(t, n):
    """Density of joining time under the approximation.

    f(t) = lambda(t) * F_bar(t): the rate of coalescence at time t
    times the probability of having survived to time t.
    """
    return n * np.exp(-t) / (n + (1 - n) * np.exp(-t / 2))**3


def joining_prob_approx(x, y, n):
    """Joining probability for branch [x, y] using the approximation.

    Numerically integrates the survival function over the branch
    interval.  This is much faster than the exact calculation
    because F_bar_approx is a smooth closed-form function.
    """
    result, _ = quad(lambda t: F_bar_approx(t, n), x, y)
    return result


def lambda_inverse(ell, n):
    """Inverse of the lambda function: find t such that lambda(t) = ell.

    This inverts the deterministic approximation formula.
    We need this to convert from a target lineage count (the
    geometric mean) back to a physical time.
    """
    # lambda(t) = n / (n + (1-n)*exp(-t/2))
    # Solving for t:
    # ell * (n + (1-n)*exp(-t/2)) = n
    # (1-n)*exp(-t/2) = n/ell - n = n(1-ell)/ell
    # exp(-t/2) = n(1-ell) / (ell*(1-n))
    # Since n > 1 and 1-n < 0: n(1-ell)/(ell*(1-n)) should be positive
    # when ell < n (which it always is for branches below TMRCA)
    ratio = (n - n * ell) / (ell - n * ell)
    if ratio <= 0:
        return np.inf  # edge case: lineage count at or beyond n
    return -2 * np.log(ratio)


def representative_time(x, y, n):
    """Compute representative joining time for branch [x, y].

    Uses the geometric mean of lambda at the endpoints to find
    a single representative time for the branch.  This time is
    used in emission and transition probability calculations.
    """
    lam_x = lambda_approx(x, n)
    lam_y = lambda_approx(y, n)
    # Geometric mean: the "natural midpoint" on a log scale
    lam_tau = np.sqrt(lam_x * lam_y)
    tau = lambda_inverse(lam_tau, n)
    return tau


def emission_probability(allele_new, allele_at_join, tau, branch_lower,
                         branch_upper, theta):
    """Compute emission probability for one site.

    Parameters
    ----------
    allele_new : int
        Allele (0 or 1) at the new node (sample).
    allele_at_join : int
        Imputed allele at the joining point (by parsimony).
    tau : float
        Representative joining time.
    branch_lower : float
        Lower time of the joining branch.
    branch_upper : float
        Upper time of the joining branch.
    theta : float
        Population-scaled mutation rate (4*Ne*mu).

    Returns
    -------
    prob : float
        Per-site emission probability.
    """
    # Branch lengths: how much "time" each branch spans
    l1 = tau                    # new lineage: from present to joining point
    l2 = tau - branch_lower     # lower part of joining branch
    l3 = branch_upper - tau     # upper part of joining branch

    def p_mutation(length):
        """Probability of mutation on a branch of given length.

        Under the Poisson mutation model, mutations occur at rate
        theta/2 per unit time.  1 - exp(-rate * time) is the
        probability of at least one event.
        """
        return 1 - np.exp(-theta / 2 * length)

    def p_no_mutation(length):
        """Probability of no mutation on a branch of given length.

        exp(-rate * time) is the probability of zero events in
        a Poisson process -- the "survival" probability for mutations.
        """
        return np.exp(-theta / 2 * length)

    # New lineage: needs mutation if allele_new != allele_at_join
    if allele_new != allele_at_join:
        e1 = p_mutation(l1)
    else:
        e1 = p_no_mutation(l1)

    # For the two parts of the joining branch, we need to consider
    # what alleles are expected above and below the joining point.
    # The detailed calculation depends on the tree topology above,
    # but the core structure is:
    e2 = p_no_mutation(l2)  # Simplified: no mutation on lower part
    e3 = p_no_mutation(l3)  # Simplified: no mutation on upper part

    # Independence: multiply the three branch probabilities
    return e1 * e2 * e3


class BranchState:
    """A branch state in the SINGER HMM (full or partial).

    Each state represents either a complete branch of the marginal
    tree (full) or a segment of a branch that was split by a
    recombination event in the partial ARG (partial).
    """

    def __init__(self, child, parent, lower_time, upper_time, is_partial=False):
        self.child = child            # child node ID in the tree
        self.parent = parent          # parent node ID in the tree
        self.lower_time = lower_time  # start of the time interval
        self.upper_time = upper_time  # end of the time interval
        self.is_partial = is_partial  # True if this is a sub-segment

    @property
    def length(self):
        """Time span of this branch (or branch segment)."""
        return self.upper_time - self.lower_time

    def __repr__(self):
        kind = "partial" if self.is_partial else "full"
        return (f"BranchState({self.child},{self.parent}): "
                f"[{self.lower_time:.4f},{self.upper_time:.4f}] ({kind})")


def build_state_space(full_branches, partial_branches, forward_probs,
                       epsilon=0.01):
    """Build the state space for a bin, pruning unlikely partial branches.

    Parameters
    ----------
    full_branches : list of BranchState
        Full branches of the marginal tree at this bin.
    partial_branches : list of (BranchState, float)
        Candidate partial branches with their forward probabilities.
    epsilon : float
        Pruning threshold: partial branches with forward probability
        below epsilon are discarded to keep the state space manageable.

    Returns
    -------
    states : list of BranchState
        Active states for this bin.
    """
    # All full branches are always included -- they are always valid
    # joining targets for the new lineage
    states = list(full_branches)

    # Partial branches are included only if their forward probability
    # exceeds epsilon.  This keeps the state space from growing
    # unboundedly while preserving the most important constraints.
    for branch, fwd_prob in partial_branches:
        if fwd_prob >= epsilon:
            states.append(branch)

    return states


def branch_transition_prob(tau_i, tau_j, p_j, rho, is_partial_j,
                           q_sum, same_branch):
    """Compute transition probability from branch i to branch j.

    Parameters
    ----------
    tau_i : float
        Representative time of the current branch.
    tau_j : float
        Representative time of the target branch.
    p_j : float
        Coalescence probability for target branch.
    rho : float
        Population-scaled recombination rate per bin.
    is_partial_j : bool
        Whether target branch is a partial branch state.
    q_sum : float
        Sum of q_k over all states in S_ell.
    same_branch : bool
        Whether i == j (same branch, no recombination).

    Returns
    -------
    prob : float
    """
    # r_i: probability of recombination on the new lineage up to tau_i
    r_i = 1 - np.exp(-rho / 2 * tau_i)

    if is_partial_j:
        q_j = 0.0  # partial branches get zero weight
    else:
        r_j = 1 - np.exp(-rho / 2 * tau_j)
        q_j = r_j * p_j  # product ensures correct stationary distribution

    if same_branch:
        # No-recombination term + recombination-to-self term
        return (1 - r_i) + r_i * q_j / q_sum
    else:
        # Pure recombination term: probability of jumping to branch j
        return r_i * q_j / q_sum


def split_branch_transition(full_branch, segments, n):
    """Distribute transition probability when a branch splits.

    When the partial ARG has a recombination that splits a branch
    into segments, the probability mass from the full branch must
    be distributed among the segments.  We weight each segment
    by its coalescence probability -- how likely the new lineage
    is to join that particular segment.

    Parameters
    ----------
    full_branch : BranchState
        The full branch being split.
    segments : list of BranchState
        The resulting segments.
    n : int
        Number of samples (for joining probability calculation).

    Returns
    -------
    weights : list of float
        Transition weight for each segment (sums to 1).
    """
    probs = []
    for seg in segments:
        # Each segment's weight is its joining probability
        p = joining_prob_approx(seg.lower_time, seg.upper_time, n)
        probs.append(p)

    total = sum(probs)
    if total == 0:
        # Fallback: if all probabilities are zero (degenerate case),
        # distribute uniformly
        return [1.0 / len(segments)] * len(segments)
    return [p / total for p in probs]


# ============================================================================
# Chapter 2: Time Sampling
# ============================================================================

def partition_branch(x, y, d=20):
    """Partition a branch [x, y) into d sub-intervals.

    Uses uniform spacing in the exponential CDF, so sub-intervals
    have equal probability mass under Exp(1).  This gives denser
    spacing near the present (x) and sparser spacing toward the
    past (y), matching where coalescence events are most likely.

    Parameters
    ----------
    x, y : float
        Lower and upper time of the branch.
    d : int
        Number of sub-intervals.

    Returns
    -------
    boundaries : ndarray of shape (d + 1,)
        Time boundaries [t_0, t_1, ..., t_d].
    """
    exp_x = np.exp(-x)  # e^{-x}: survival probability at lower endpoint
    exp_y = np.exp(-y)  # e^{-y}: survival probability at upper endpoint
    # fractions = [0, 1/d, 2/d, ..., 1]: the quantile levels
    fractions = np.linspace(0, 1, d + 1)
    # Linear interpolation between exp_x and exp_y, then invert
    # via -log to get the time boundaries
    boundaries = -np.log(exp_x - fractions * (exp_x - exp_y))
    return boundaries


def representative_times_ts(boundaries):
    """Compute representative time for each sub-interval.

    The representative time sits at the center of mass of the
    sub-interval under the exponential distribution, not at the
    arithmetic midpoint.

    Parameters
    ----------
    boundaries : ndarray of shape (d + 1,)

    Returns
    -------
    taus : ndarray of shape (d,)
    """
    d = len(boundaries) - 1
    taus = np.zeros(d)
    for i in range(d):
        # Average in survival-probability space, then invert
        avg_exp = (np.exp(-boundaries[i]) + np.exp(-boundaries[i+1])) / 2
        taus[i] = -np.log(avg_exp)
    return taus


def psmc_transition_density(t, s, rho):
    """PSMC transition density q_rho(t | s).

    This is the probability density of the new coalescence time t,
    given the old coalescence time s and recombination rate rho.

    Parameters
    ----------
    t : float
        Target time.
    s : float
        Source time.
    rho : float
        Recombination rate per bin.

    Returns
    -------
    density : float
    """
    # Probability that recombination occurred on the branch [0, s]
    p_recomb = 1 - np.exp(-rho * s)

    if abs(t - s) < 1e-12:
        # Point mass (no recombination): probability e^{-rho*s}
        return np.exp(-rho * s)

    if t < s:
        # New coalescence is shallower (more recent) than old
        return (p_recomb / s) * (1 - np.exp(-t))
    else:
        # New coalescence is deeper (more ancient) than old
        return (p_recomb / s) * (np.exp(-(t - s)) - np.exp(-t))


def psmc_transition_cdf(t, s, rho):
    """PSMC transition CDF Q_rho(t | s).

    The cumulative distribution function: P(T_new <= t | T_old = s).
    Includes both the continuous density (recombination cases) and
    the point mass at t = s (no recombination).
    """
    p_recomb = 1 - np.exp(-rho * s)
    p_no_recomb = np.exp(-rho * s)

    if t < s:
        # Only the t < s continuous density contributes
        return (p_recomb / s) * (t + np.exp(-t) - 1)
    else:
        # The t < s density (integrated to s), plus the point mass,
        # plus the t > s density (integrated from s to t)
        return (p_recomb / s) * (s - np.exp(-(t - s)) + np.exp(-t)) + p_no_recomb


def time_transition_matrix(boundaries_prev, taus_prev, boundaries_next, rho):
    """Compute transition matrix between time sub-intervals.

    Each entry Q[i, j] gives the probability of transitioning from
    sub-interval i at the previous bin to sub-interval j at the
    current bin, conditioned on the coalescence falling within the
    current branch interval.

    Parameters
    ----------
    boundaries_prev : ndarray of shape (d_prev + 1,)
        Sub-interval boundaries at bin ell-1.
    taus_prev : ndarray of shape (d_prev,)
        Representative times at bin ell-1.
    boundaries_next : ndarray of shape (d_next + 1,)
        Sub-interval boundaries at bin ell.
    rho : float

    Returns
    -------
    Q : ndarray of shape (d_prev, d_next)
        Transition matrix.
    """
    d_prev = len(taus_prev)
    d_next = len(boundaries_next) - 1
    Q = np.zeros((d_prev, d_next))

    # Branch interval boundaries for normalization
    x_ell = boundaries_next[0]   # lower bound of joining branch
    y_ell = boundaries_next[-1]  # upper bound of joining branch

    for i in range(d_prev):
        # Denominator: total mass in the branch interval [x, y)
        denom = (psmc_transition_cdf(y_ell, taus_prev[i], rho) -
                 psmc_transition_cdf(x_ell, taus_prev[i], rho))
        if denom < 1e-15:
            Q[i, :] = 1.0 / d_next  # uniform fallback for degenerate cases
            continue

        for j in range(d_next):
            # Numerator: mass in sub-interval j
            numer = (psmc_transition_cdf(boundaries_next[j+1],
                                         taus_prev[i], rho) -
                     psmc_transition_cdf(boundaries_next[j],
                                         taus_prev[i], rho))
            Q[i, j] = numer / denom  # conditional probability

    return Q


def forward_linearized(alpha_prev, Q, emissions):
    """Linear-time forward step for Type A transitions.

    Exploits Properties 1 and 2 to compute the forward step in
    O(d) time instead of O(d^2).  The result is identical to the
    standard matrix-vector product alpha_prev @ Q * emissions.

    Parameters
    ----------
    alpha_prev : ndarray of shape (d,)
        Forward probabilities at previous bin.
    Q : ndarray of shape (d, d)
        Transition matrix (Type A: same state space).
    emissions : ndarray of shape (d,)
        Emission probabilities at current bin.

    Returns
    -------
    alpha_curr : ndarray of shape (d,)
    """
    d = len(alpha_prev)

    # Compute kappa values: the geometric ratio from Property 2
    # kappa[j] = q[i,j] / q[i,j-1] for any i < j
    kappa = np.zeros(d)
    for j in range(1, d):
        kappa[j] = Q[0, j] / Q[0, j-1] if Q[0, j-1] > 0 else 0

    # Compute S_j (from below): accumulates contributions from i < j
    # S_j = alpha_{j-1} * q_{j-1,j} + kappa_j * S_{j-1}
    S = np.zeros(d)
    for j in range(1, d):
        S[j] = alpha_prev[j-1] * Q[j-1, j] + kappa[j] * S[j-1]

    # Compute A_j (from above): accumulates contributions from i > j
    # A_j = alpha_{j+1} + A_{j+1}
    # By Property 1, all i > j contribute the same q_{j+1,j}, so
    # we just need the sum of alpha values above j
    A = np.zeros(d)
    for j in range(d - 2, -1, -1):
        A[j] = alpha_prev[j+1] + A[j+1]

    # Forward probabilities: combine below, diagonal, and above
    alpha_curr = np.zeros(d)
    for j in range(d):
        alpha_curr[j] = emissions[j] * (
            S[j] + alpha_prev[j] * Q[j, j] + A[j] * Q[j+1, j] if j < d-1
            else S[j] + alpha_prev[j] * Q[j, j]
        )

    return alpha_curr


def type_b_transition(alpha_prev, boundaries_prev, boundaries_next,
                       mapped_intervals, rho):
    """Handle Type B transition (recombination hitchhiking).

    When the partial ARG has a recombination, some sub-intervals
    from the previous bin map directly to sub-intervals in the
    current bin (the new lineage hitchhikes on the existing
    recombination), while others do not (wrong branch).

    Parameters
    ----------
    alpha_prev : ndarray
        Forward probabilities at previous bin.
    boundaries_prev : ndarray
        Time boundaries at previous bin.
    boundaries_next : ndarray
        Time boundaries at current bin.
    mapped_intervals : list of (prev_idx, next_idx) or None
        Maps previous sub-intervals to current ones.
        None means the interval doesn't contribute (wrong branch).
    rho : float

    Returns
    -------
    alpha_curr : ndarray
    """
    d_next = len(boundaries_next) - 1
    alpha_curr = np.zeros(d_next)

    for prev_idx, mapping in enumerate(mapped_intervals):
        if mapping is not None:
            next_idx = mapping
            # Forward probability transfers directly: the new lineage
            # stays at the same time, just in the new tree's coordinates
            alpha_curr[next_idx] += alpha_prev[prev_idx]

    # Sub-intervals not covered by hitchhiking get zero forward probability
    # from the hitchhiked states but may receive probability from
    # the transition matrix for newly created states

    return alpha_curr


def type_c_transition(alpha_prev, taus_prev, boundaries_next):
    """Handle Type C transition (new recombination in the new lineage).

    When rho -> infinity, the transition is just the unconditional
    coalescence density restricted to the new branch.

    Parameters
    ----------
    alpha_prev : ndarray of shape (d_prev,)
    taus_prev : ndarray of shape (d_prev,)
    boundaries_next : ndarray of shape (d_next + 1,)

    Returns
    -------
    alpha_curr : ndarray of shape (d_next,)
    """
    # With rho -> infinity, the no-recombination term vanishes
    # and we use the conditional transition q_0(t|s)
    Q = time_transition_matrix(
        None,  # boundaries don't matter for rho=infinity
        taus_prev,
        boundaries_next,
        rho=1e10  # approximate infinity: e^{-1e10 * s} is essentially 0
    )

    # Standard matrix-vector multiply: sum over source sub-intervals
    alpha_curr = alpha_prev @ Q
    return alpha_curr


# ============================================================================
# Chapter 3: ARG Rescaling
# ============================================================================

def compute_arg_length_in_window(branches, window_lower, window_upper):
    """Compute total ARG length overlapping a time window.

    For each branch in the ARG, compute the time overlap between
    the branch's time interval and the window, then weight by the
    branch's genomic span (number of base pairs it covers).

    Parameters
    ----------
    branches : list of (span, lower_time, upper_time)
        Each branch has a genomic span and a time interval.
    window_lower, window_upper : float
        Time window boundaries.

    Returns
    -------
    length : float
        Total branch length in this window, weighted by span.
    """
    total = 0.0
    for span, lo, hi in branches:
        # Overlap between [lo, hi) and [window_lower, window_upper)
        overlap_lo = max(lo, window_lower)
        overlap_hi = min(hi, window_upper)
        if overlap_hi > overlap_lo:
            # span * time_overlap = total branch-length contribution
            total += span * (overlap_hi - overlap_lo)
    return total


def partition_time_axis(branches, J=100):
    """Partition time axis into J equal-ARG-length windows.

    Finds time boundaries such that each window contains
    1/J of the total ARG branch length.  Uses a sweep through
    all distinct time points in the ARG.

    Parameters
    ----------
    branches : list of (span, lower_time, upper_time)
    J : int
        Number of windows.

    Returns
    -------
    boundaries : ndarray of shape (J + 1,)
    """
    # Total ARG length (sum of span * branch_length for all branches)
    t_max = max(hi for _, _, hi in branches)
    total_length = compute_arg_length_in_window(branches, 0, t_max)
    target_per_window = total_length / J

    # Find boundaries by scanning through time.
    # Collect all distinct time points (branch endpoints) to avoid
    # missing discontinuities in the branch-length function.
    time_points = sorted(set(
        [0.0, t_max] +
        [lo for _, lo, _ in branches] +
        [hi for _, _, hi in branches]
    ))

    boundaries = [0.0]
    cumulative = 0.0

    for k in range(len(time_points) - 1):
        segment_length = compute_arg_length_in_window(
            branches, time_points[k], time_points[k + 1])
        cumulative += segment_length

        # When we've accumulated enough length, place a boundary
        while cumulative >= target_per_window and len(boundaries) < J:
            # Interpolate to find exact boundary within this segment
            overshoot = cumulative - target_per_window
            segment_total = segment_length
            if segment_total > 0:
                fraction = 1 - overshoot / segment_total
                boundary = (time_points[k] +
                            fraction * (time_points[k + 1] - time_points[k]))
            else:
                boundary = time_points[k + 1]
            boundaries.append(boundary)
            cumulative -= target_per_window

    boundaries.append(t_max)
    return np.array(boundaries[:J + 1])


def count_mutations_per_window(mutations, boundaries):
    """Count mutations in each time window, fractionally.

    Each mutation is distributed across windows proportionally
    to the overlap between its branch and each window.

    Parameters
    ----------
    mutations : list of (branch_lower, branch_upper)
        Time interval of the branch carrying each mutation.
    boundaries : ndarray of shape (J + 1,)

    Returns
    -------
    counts : ndarray of shape (J,)
    """
    J = len(boundaries) - 1
    counts = np.zeros(J)

    for branch_lo, branch_hi in mutations:
        branch_length = branch_hi - branch_lo
        if branch_length == 0:
            continue  # degenerate branch: skip

        for i in range(J):
            # How much of this branch falls in window i?
            overlap_lo = max(branch_lo, boundaries[i])
            overlap_hi = min(branch_hi, boundaries[i + 1])
            if overlap_hi > overlap_lo:
                fraction = (overlap_hi - overlap_lo) / branch_length
                counts[i] += fraction  # fractional mutation count

    return counts


def compute_scaling_factors(counts, total_arg_length, theta, J):
    """Compute rescaling factors for each time window.

    Each factor c_i = observed / expected mutations in window i.
    A factor > 1 means time was compressed (too few mutations for
    the branch length); < 1 means time was stretched.

    Parameters
    ----------
    counts : ndarray of shape (J,)
        Mutation counts per window.
    total_arg_length : float
    theta : float
        Population-scaled mutation rate.
    J : int
        Number of windows.

    Returns
    -------
    c : ndarray of shape (J,)
        Scaling factors.
    """
    # Expected mutations per window: theta/2 * (total_length / J)
    expected_per_window = theta * total_arg_length / (2 * J)
    if expected_per_window == 0:
        return np.ones(J)  # nothing to rescale

    c = counts / expected_per_window
    return c


def rescale_times(node_times, boundaries, scaling_factors):
    """Rescale all coalescence times using window-specific scaling.

    Each node's time is transformed according to the scaling factor
    of the window it falls in.  The transformation is piecewise
    linear and monotonically increasing.

    Parameters
    ----------
    node_times : dict of {node_id: time}
    boundaries : ndarray of shape (J + 1,)
    scaling_factors : ndarray of shape (J,)

    Returns
    -------
    new_times : dict of {node_id: rescaled_time}
    """
    J = len(scaling_factors)

    # Compute new window boundaries by applying the rescaling
    new_boundaries = np.zeros(J + 1)
    for i in range(J):
        # Each new boundary = previous boundary + scaled window width
        new_boundaries[i + 1] = (scaling_factors[i] *
                                  (boundaries[i + 1] - boundaries[i]) +
                                  new_boundaries[i])

    # Rescale each node time
    new_times = {}
    for node_id, t in node_times.items():
        if t <= 0:
            new_times[node_id] = 0.0  # leaf nodes stay at time 0
            continue

        # Find which window this time falls in
        for i in range(J):
            if boundaries[i] <= t < boundaries[i + 1]:
                # Apply piecewise linear rescaling:
                # offset within window * scaling factor + new window start
                new_t = (scaling_factors[i] * (t - boundaries[i]) +
                         new_boundaries[i])
                new_times[node_id] = new_t
                break
        else:
            # Time is at or beyond t_max: map to the end
            new_times[node_id] = new_boundaries[-1]

    return new_times


def count_mutations_with_rate_variation(branches, mutations, boundaries,
                                         mutation_rate_map):
    """Count expected mutations per window accounting for rate variation.

    Instead of assuming a constant mutation rate, this function uses
    a position-dependent rate map to compute expected mutations.

    Parameters
    ----------
    branches : list of (start_pos, end_pos, lower_time, upper_time)
    mutations : list of (position, branch_lower, branch_upper)
    boundaries : ndarray of shape (J + 1,)
    mutation_rate_map : callable
        mutation_rate_map(x) returns the local mutation rate at position x.

    Returns
    -------
    expected : ndarray of shape (J,)
        Expected mutations per window.
    observed : ndarray of shape (J,)
        Observed mutations per window.
    """
    J = len(boundaries) - 1
    expected = np.zeros(J)
    observed = np.zeros(J)

    # Expected: integrate over branches, weighting by local mutation rate
    for start, end, lo, hi in branches:
        # Average mutation rate over this branch's genomic span
        # (simplified: evaluate at midpoint instead of integrating)
        mu_avg = mutation_rate_map((start + end) / 2)
        span = end - start

        for i in range(J):
            overlap_lo = max(lo, boundaries[i])
            overlap_hi = min(hi, boundaries[i + 1])
            if overlap_hi > overlap_lo:
                # Expected mutations = rate * span * time_overlap
                expected[i] += mu_avg * span * (overlap_hi - overlap_lo)

    # Observed: count actual mutations (same as before)
    for pos, branch_lo, branch_hi in mutations:
        branch_length = branch_hi - branch_lo
        if branch_length == 0:
            continue
        for i in range(J):
            overlap_lo = max(branch_lo, boundaries[i])
            overlap_hi = min(branch_hi, boundaries[i + 1])
            if overlap_hi > overlap_lo:
                observed[i] += (overlap_hi - overlap_lo) / branch_length

    return expected, observed


# ============================================================================
# Chapter 4: SGPR (Sub-Graph Pruning and Re-grafting)
# ============================================================================

class SimpleTree:
    """A minimal tree for demonstrating SPR.

    Stores the tree as a parent map (each node points to its parent)
    and a time map (each node has a time/height).  This is the same
    representation used internally by tree sequence libraries like
    tskit (see the ARG prerequisite chapter).
    """

    def __init__(self, parent, time):
        """
        Parameters
        ----------
        parent : dict of {node: parent_node}
        time : dict of {node: time}
        """
        self.parent = parent
        self.time = time
        # Build children map by inverting the parent map
        self.children = {}
        for child, par in parent.items():
            if par is not None:
                self.children.setdefault(par, []).append(child)

    def branches(self):
        """Return all branches as (child, parent, length)."""
        result = []
        for child, par in self.parent.items():
            if par is not None:
                result.append((child, par,
                               self.time[par] - self.time[child]))
        return result

    def height(self):
        """Return the tree height (TMRCA).

        The TMRCA (Time to Most Recent Common Ancestor) is the time
        of the root node -- the oldest node in the tree.
        """
        return max(self.time.values())


def spr_move(tree, cut_node, new_parent, new_time):
    """Perform an SPR move on a tree.

    This implements the three-step SPR procedure:
    1. Detach the subtree rooted at cut_node
    2. Remove the now-unary node (the old parent of cut_node)
    3. Re-attach cut_node to new_parent at new_time

    Parameters
    ----------
    tree : SimpleTree
    cut_node : int
        The node whose branch we cut above.
    new_parent : int
        The branch (identified by its child node) to re-attach to.
    new_time : float
        The time of the re-attachment point.

    Returns
    -------
    new_tree : SimpleTree
    """
    new_parent_dict = dict(tree.parent)
    new_time_dict = dict(tree.time)

    # Find the old parent and grandparent of cut_node
    old_parent = new_parent_dict[cut_node]
    old_grandparent = new_parent_dict.get(old_parent)

    # Find sibling of cut_node (the other child of old_parent)
    siblings = [c for c in tree.children.get(old_parent, [])
                 if c != cut_node]

    if siblings and old_grandparent is not None:
        sibling = siblings[0]
        # Remove old_parent node: connect sibling directly to grandparent
        # This eliminates the now-unary internal node
        new_parent_dict[sibling] = old_grandparent
        del new_parent_dict[old_parent]

    # Re-attach cut_node to new_parent at new_time
    # Create a new internal node at the re-attachment point
    new_internal = max(new_time_dict.keys()) + 1
    new_time_dict[new_internal] = new_time

    # Re-wire: cut_node and new_parent both become children
    # of the new internal node
    target_parent = new_parent_dict.get(new_parent)
    new_parent_dict[new_parent] = new_internal
    new_parent_dict[cut_node] = new_internal
    if target_parent is not None:
        new_parent_dict[new_internal] = target_parent

    return SimpleTree(new_parent_dict, new_time_dict)


def select_cut(tree):
    """Select a random cut on a marginal tree.

    The cut is chosen by first sampling a time uniformly in
    [0, tree height], then choosing uniformly among branches
    that cross that time.  This two-step procedure gives each
    branch a probability proportional to its length.

    Parameters
    ----------
    tree : SimpleTree

    Returns
    -------
    cut_node : int
        The branch being cut (identified by child node).
    cut_time : float
        The time of the cut.
    """
    # Sample time uniformly in [0, tree height]
    h = tree.height()
    cut_time = np.random.uniform(0, h)

    # Find branches that cross this time
    crossing_branches = []
    for child, parent, length in tree.branches():
        if tree.time[child] <= cut_time < tree.time[parent]:
            crossing_branches.append(child)

    # Choose one uniformly at random
    cut_node = np.random.choice(crossing_branches)
    return cut_node, cut_time


def sgpr_acceptance_ratio(old_tree_height, new_tree_height):
    """Compute the SGPR Metropolis-Hastings acceptance ratio.

    The acceptance ratio is simply the ratio of the old to new
    tree heights.  This remarkably simple formula follows from
    the cancellation of posterior ratios (see derivation above).

    Parameters
    ----------
    old_tree_height : float
        Height of the marginal tree before the move.
    new_tree_height : float
        Height of the marginal tree after the move.

    Returns
    -------
    ratio : float
        Acceptance probability.
    """
    return min(1.0, old_tree_height / new_tree_height)


def simulate_tree_height_variability(n, n_replicates=10000):
    """Simulate TMRCA for n samples to see height variability.

    Under the standard coalescent (see the coalescent_theory chapter),
    the TMRCA converges to 2 in coalescent units as n -> infinity.
    The coefficient of variation (CV) decreases with n, which is
    why SGPR's acceptance rate approaches 1.
    """
    heights = np.zeros(n_replicates)
    for rep in range(n_replicates):
        k = n
        t = 0.0
        while k > 1:
            # Coalescence rate with k lineages: k*(k-1)/2
            rate = k * (k - 1) / 2
            # Wait an exponential time before the next coalescence
            t += np.random.exponential(1.0 / rate)
            k -= 1
        heights[rep] = t
    return heights


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate key components of the SINGER algorithm."""

    print("=" * 60)
    print("SINGER Mini-Implementation Demo")
    print("=" * 60)

    # --- Branch Sampling: Joining Probabilities ---
    print("\n--- Branch Sampling: Joining Probabilities ---")

    # Exact joining probability
    t1, t2 = 0.3, 0.8
    tree_intervals = [
        (0, t1), (0, t1), (0, t1), (0, t1),  # 4 leaf branches
        (t1, t2), (t1, t2),                    # 2 internal branches
        (t2, 5.0),                              # root branch (truncated)
    ]

    for i, (lo, hi) in enumerate(tree_intervals):
        p = joining_probability_exact(lo, hi, tree_intervals)
        print(f"Branch {i} [{lo:.1f}, {hi:.1f}]: p = {p:.6f}")

    # Approximate joining probability
    print("\n--- Deterministic Approximation ---")
    n = 50
    t1, t2, t3 = 0.01, 0.05, 0.2
    branches_approx = [(0, t1), (t1, t2), (t2, t3), (t3, 2.0)]

    print(f"{'Branch':<20} {'p_approx':>10}")
    print("-" * 32)
    total = 0
    for lo, hi in branches_approx:
        p = joining_prob_approx(lo, hi, n)
        total += p
        print(f"[{lo:.2f}, {hi:.2f}]{'':<10} {p:>10.6f}")
    print(f"{'Sum':<20} {total:>10.6f}")

    # Representative time
    print("\n--- Representative Joining Times ---")
    for x, y in [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]:
        tau = representative_time(max(x, 1e-10), y, n)
        print(f"Branch [{x:.2f}, {y:.2f}]: tau = {tau:.6f}, "
              f"lambda(x)={lambda_approx(max(x,1e-10),n):.2f}, "
              f"lambda(y)={lambda_approx(y,n):.2f}, "
              f"lambda(tau)={lambda_approx(tau,n):.2f}")

    # Emission probability
    print("\n--- Emission Probabilities ---")
    theta = 0.001
    tau = 0.5
    branch = (0.1, 1.2)
    for allele_new, allele_join in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        e = emission_probability(allele_new, allele_join, tau,
                                 branch[0], branch[1], theta)
        print(f"allele_new={allele_new}, allele_join={allele_join}: "
              f"emission={e:.8f}")

    # Transition probability
    print("\n--- Branch Transition Matrix ---")
    rho = 0.5
    branches_tr = [(0.0, 0.02), (0.02, 0.06), (0.06, 0.15),
                   (0.15, 0.4), (0.4, 2.0)]
    taus = [representative_time(max(x, 1e-10), y, n) for x, y in branches_tr]
    probs = [joining_prob_approx(x, y, n) for x, y in branches_tr]

    r_vals = [1 - np.exp(-rho / 2 * t) for t in taus]
    q_vals = [r * p for r, p in zip(r_vals, probs)]
    q_sum = sum(q_vals)

    print("Transition matrix:")
    T = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            T[i, j] = branch_transition_prob(
                taus[i], taus[j], probs[j], rho,
                is_partial_j=False, q_sum=q_sum, same_branch=(i == j)
            )
        print(f"  From branch {i}: {np.round(T[i], 4)}")
    print(f"Row sums: {T.sum(axis=1)}")

    # Partial branch states
    print("\n--- Partial Branch States ---")
    full = [
        BranchState(0, 4, 0.0, 0.3),
        BranchState(1, 4, 0.0, 0.3),
        BranchState(2, 5, 0.0, 0.7),
        BranchState(4, 5, 0.3, 0.7),
        BranchState(5, -1, 0.7, float('inf')),
    ]
    partial_candidates = [
        (BranchState(0, 4, 0.0, 0.15, is_partial=True), 0.05),
        (BranchState(0, 4, 0.15, 0.3, is_partial=True), 0.002),
    ]
    states = build_state_space(full, partial_candidates, None, epsilon=0.01)
    print(f"State space size: {len(states)}")
    for s in states:
        print(f"  {s}")

    # Split branch transition
    print("\n--- Split Branch Transition ---")
    full_b = BranchState(1, 5, 0.0, 1.0)
    seg_lower = BranchState(1, 5, 0.0, 0.3, is_partial=True)
    seg_upper = BranchState(1, 5, 0.3, 1.0, is_partial=True)
    weights = split_branch_transition(full_b, [seg_lower, seg_upper], n=50)
    print(f"Lower segment weight: {weights[0]:.4f}")
    print(f"Upper segment weight: {weights[1]:.4f}")

    # --- Time Sampling ---
    print("\n--- Time Sampling: Branch Partition ---")
    boundaries = partition_branch(0.1, 2.0, d=10)
    print("Sub-interval boundaries:")
    for i in range(len(boundaries) - 1):
        width = boundaries[i+1] - boundaries[i]
        print(f"  [{boundaries[i]:.4f}, {boundaries[i+1]:.4f}) width={width:.4f}")

    taus_ts = representative_times_ts(boundaries)
    print("\nRepresentative times:")
    for i, tau_val in enumerate(taus_ts):
        print(f"  Sub-interval {i}: tau = {tau_val:.4f}")

    # PSMC transition CDF verification
    print(f"\nCDF(100 | s=1, rho=0.5) = {psmc_transition_cdf(100, 1.0, 0.5):.6f}")

    # Time transition matrix
    Q = time_transition_matrix(boundaries, taus_ts, boundaries, rho=0.5)
    print(f"\nTransition matrix shape: {Q.shape}")
    print(f"Row sums: {np.round(Q.sum(axis=1), 6)}")
    print(f"\nFirst row: {np.round(Q[0], 4)}")

    # Linearized forward step verification
    print("\n--- Linearized Forward Step ---")
    d = 10
    np.random.seed(42)
    alpha_prev = np.random.dirichlet(np.ones(d))
    emissions = np.random.uniform(0.1, 0.9, size=d)

    alpha_quad = emissions * (alpha_prev @ Q)
    alpha_lin = forward_linearized(alpha_prev, Q, emissions)
    print(f"Max difference (linear vs quadratic): {np.max(np.abs(alpha_quad - alpha_lin)):.2e}")

    # --- ARG Rescaling ---
    print("\n--- ARG Rescaling ---")
    branches_rescale = [
        (1000, 0.0, 0.3),
        (1000, 0.0, 0.3),
        (1000, 0.0, 0.7),
        (1000, 0.0, 0.7),
        (1000, 0.3, 0.7),
        (1000, 0.3, 0.7),
        (1000, 0.7, 1.5),
    ]
    boundaries_rescale = partition_time_axis(branches_rescale, J=5)
    print("Time window boundaries:")
    for i in range(len(boundaries_rescale) - 1):
        length = compute_arg_length_in_window(branches_rescale,
                                               boundaries_rescale[i],
                                               boundaries_rescale[i + 1])
        print(f"  [{boundaries_rescale[i]:.4f}, {boundaries_rescale[i+1]:.4f}): "
              f"ARG length = {length:.1f}")

    # Mutation counts
    np.random.seed(42)
    mutations = [(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.5))
                 for _ in range(20)]
    counts = count_mutations_per_window(mutations, boundaries_rescale)
    print("\nMutation counts per window:")
    for i in range(len(counts)):
        print(f"  Window {i}: {counts[i]:.2f} mutations")

    # Scaling factors
    total_length = sum(span * (hi - lo) for span, lo, hi in branches_rescale)
    theta_rescale = 0.001
    c = compute_scaling_factors(counts, total_length, theta_rescale, len(counts))
    print("\nScaling factors:")
    for i, ci in enumerate(c):
        print(f"  Window {i}: c = {ci:.4f}")

    # Rescale times
    node_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
                  4: 0.3, 5: 0.7, 6: 1.5}
    new_times = rescale_times(node_times, boundaries_rescale, c)
    print("\nRescaled coalescence times:")
    for node_id in sorted(new_times.keys()):
        print(f"  Node {node_id}: {node_times[node_id]:.4f} -> "
              f"{new_times[node_id]:.4f}")

    # --- SGPR ---
    print("\n--- SGPR: SPR Moves ---")
    tree = SimpleTree(
        parent={0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6},
        time={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.3, 5: 0.7, 6: 1.5}
    )
    print("Original branches:")
    for c_node, p_node, l in tree.branches():
        print(f"  {c_node} -> {p_node}: length={l:.2f}")

    np.random.seed(42)
    cut_node, cut_time = select_cut(tree)
    print(f"\nCut: node {cut_node} at time {cut_time:.4f}")

    # SGPR acceptance ratio
    print("\n--- SGPR Acceptance Ratio ---")
    print(f"sgpr_acceptance_ratio(2.0, 2.0) = {sgpr_acceptance_ratio(2.0, 2.0)}")
    print(f"sgpr_acceptance_ratio(1.5, 2.0) = {sgpr_acceptance_ratio(1.5, 2.0)}")
    print(f"sgpr_acceptance_ratio(2.0, 1.5) = {sgpr_acceptance_ratio(2.0, 1.5)}")

    # Tree height variability
    print("\n--- Tree Height Variability ---")
    for n_sample in [10, 50, 100, 500]:
        heights = simulate_tree_height_variability(n_sample, n_replicates=5000)
        cv = heights.std() / heights.mean()
        print(f"n={n_sample:>4d}: mean height={heights.mean():.4f}, "
              f"CV={cv:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
