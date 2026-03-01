"""
Mini-discoal: a pedagogical implementation of the discoal algorithm.

discoal (Kern & Schrider, 2016) is a coalescent simulator with selection.
Given a sample size, genome length, recombination rate, and a selection
coefficient, it produces random genealogies shaped by a selective sweep.

The algorithm proceeds in two steps:

1. **Generate an allele frequency trajectory** for the beneficial allele
   (forward in time), either deterministically (logistic growth) or
   stochastically (conditioned Wright-Fisher diffusion).

2. **Run the structured coalescent** conditioned on that trajectory
   (backward in time). During the sweep, lineages are assigned to one of
   two backgrounds -- beneficial (B) or wild-type (b) -- and coalescence
   rates within each background depend on the background's size, which
   changes with the trajectory. Recombination between the neutral locus and
   the selected site moves lineages between backgrounds.

This module extracts all code blocks from the discoal Timepiece (XVIII)
documentation and provides a self-contained, numpy/scipy-only reference
implementation.

Functions from allele_trajectory.rst:
    deterministic_trajectory, fixation_probability, stochastic_trajectory,
    compare_trajectories, sweep_duration_table

Functions from structured_coalescent.rst:
    coalescence_rate, migration_rates, structured_coalescent_sweep,
    escape_probability, demonstrate_bottleneck, simulate_sweep_genealogy

Functions from sweep_types.rst:
    hard_sweep_genealogy, soft_sweep_standing_variation,
    expected_independent_origins, partial_sweep_genealogy,
    compare_sweep_types

Functions from msprime_comparison.rst:
    discoal_to_msprime, msprime_to_discoal, SweepState, discoal_core,
    minimal_discoal
"""

import numpy as np
from dataclasses import dataclass, field


# ============================================================================
# Functions from allele_trajectory.rst
# ============================================================================

def deterministic_trajectory(s, N, x0=None, dt=1.0):
    """Generate a deterministic (logistic) allele frequency trajectory.

    Parameters
    ----------
    s : float
        Selection coefficient (genic/additive).
    N : int
        Diploid effective population size.
    x0 : float, optional
        Initial frequency. Defaults to 1/(2N) for a hard sweep.
    dt : float
        Time step in generations.

    Returns
    -------
    trajectory : ndarray, shape (T,)
        Allele frequency at each generation, from origin to fixation.
    """
    if x0 is None:
        x0 = 1.0 / (2 * N)

    trajectory = [x0]
    x = x0
    t = 0
    while x < 1.0 - 1.0 / (2 * N):
        t += dt
        x = 1.0 / (1.0 + ((1.0 - x0) / x0) * np.exp(-s * t))
        trajectory.append(x)
    trajectory.append(1.0)  # fixation
    return np.array(trajectory)


def fixation_probability(x, two_N_s):
    """Fixation probability from frequency x under genic selection.

    Parameters
    ----------
    x : float
        Current allele frequency.
    two_N_s : float
        Scaled selection coefficient 2Ns.

    Returns
    -------
    h : float
        Probability of eventual fixation.
    """
    if abs(two_N_s) < 1e-10:
        return x  # neutral case: fixation prob = current frequency
    return (1.0 - np.exp(-two_N_s * x)) / (1.0 - np.exp(-two_N_s))


def stochastic_trajectory(s, N, x0=None, rng=None):
    """Generate a stochastic allele frequency trajectory conditioned on fixation.

    Uses the conditioned jump process approximation (Coop & Griffiths 2004).
    Simulates backward from fixation, then reverses.

    Parameters
    ----------
    s : float
        Selection coefficient (genic).
    N : int
        Diploid effective population size.
    x0 : float, optional
        Initial frequency. Defaults to 1/(2N) for a hard sweep.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    trajectory : ndarray
        Allele frequencies from x0 to 1.0, one entry per generation.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if x0 is None:
        x0 = 1.0 / (2 * N)

    two_N = 2 * N
    two_N_s = two_N * s
    k0 = max(1, int(x0 * two_N))

    # Precompute fixation probabilities for all possible allele counts
    h = np.array([fixation_probability(k / two_N, two_N_s)
                  for k in range(two_N + 1)])

    # Simulate backward from fixation (k = 2N)
    k = two_N
    counts = [k]

    # First step back from fixation: must decrease by 1
    k -= 1
    counts.append(k)

    while k > k0:
        if k <= 0:
            break

        # Going backward from count k, the probability that the
        # previous count was k-1 (meaning it went up to k going forward):
        p_was_lower = k * (two_N - k) * h[min(k + 1, two_N)]
        p_was_higher = (
            (k + 1) * (two_N - k - 1) * h[k] if k < two_N - 1 else 0
        )

        denom = p_was_lower + p_was_higher
        if denom == 0:
            break
        p_down = p_was_lower / denom

        if rng.random() < p_down:
            k -= 1
        else:
            k += 1
            # Prevent going above 2N
            k = min(k, two_N - 1)
        counts.append(k)

    # Reverse to get forward-time trajectory
    counts.reverse()
    trajectory = np.array(counts) / two_N
    return trajectory


def compare_trajectories(s, N, n_stochastic=5, seed=42):
    """Compare deterministic vs stochastic trajectories.

    Parameters
    ----------
    s : float
        Selection coefficient.
    N : int
        Diploid effective population size.
    n_stochastic : int
        Number of stochastic trajectories to overlay.
    seed : int
        Random seed.

    Returns
    -------
    det : ndarray
        Deterministic trajectory.
    stochastics : list of ndarray
        Stochastic trajectories.
    """
    rng = np.random.default_rng(seed)

    det = deterministic_trajectory(s, N)

    stochastics = []
    for _ in range(n_stochastic):
        stoch = stochastic_trajectory(s, N, rng=rng)
        stochastics.append(stoch)

    return det, stochastics


def sweep_duration_table(N, alphas):
    """Compute sweep durations for different selection strengths.

    Parameters
    ----------
    N : int
        Diploid effective population size.
    alphas : list of float
        Values of 2Ns to tabulate.

    Returns
    -------
    results : list of dict
        Each dict contains 'alpha', 's', 'T_gen', 'T_coal', 'T_4N'.
    """
    two_N = 2 * N
    results = []
    for alpha in alphas:
        s = alpha / two_N
        T_gen = 2 * np.log(two_N) / s
        T_coal = T_gen / two_N
        T_4N = T_gen / (4 * N)
        results.append({
            'alpha': alpha, 's': s, 'T_gen': T_gen,
            'T_coal': T_coal, 'T_4N': T_4N
        })
    return results


# ============================================================================
# Functions from structured_coalescent.rst
# ============================================================================

def coalescence_rate(n_lineages, background_size):
    """Coalescence rate for n_lineages in a background of given size.

    Parameters
    ----------
    n_lineages : int
        Number of lineages in this background.
    background_size : float
        Effective number of haploid chromosomes in this background.

    Returns
    -------
    rate : float
        Total pairwise coalescence rate (per generation).
    """
    if n_lineages < 2 or background_size <= 0:
        return 0.0
    return n_lineages * (n_lineages - 1) / (2.0 * background_size)


def migration_rates(n_B, n_b, r, x):
    """Compute migration rates between backgrounds due to recombination.

    Parameters
    ----------
    n_B : int
        Number of lineages in the beneficial background.
    n_b : int
        Number of lineages in the wild-type background.
    r : float
        Recombination rate between neutral locus and selected site.
    x : float
        Current frequency of the beneficial allele.

    Returns
    -------
    m_B_to_b : float
        Total rate of migration from B to b (per generation).
    m_b_to_B : float
        Total rate of migration from b to B (per generation).
    """
    m_B_to_b = n_B * r * (1.0 - x)
    m_b_to_B = n_b * r * x
    return m_B_to_b, m_b_to_B


def structured_coalescent_sweep(trajectory, n_sample, r_site, N, rng=None):
    """Simulate the structured coalescent through a selective sweep.

    Runs backward through the trajectory, simulating coalescence and
    migration events. Returns the coalescence times.

    Parameters
    ----------
    trajectory : ndarray
        Allele frequency x(t) at each generation, from origin to fixation.
        We process this backward (from fixation to origin).
    n_sample : int
        Number of sampled lineages.
    r_site : float
        Recombination rate between the neutral locus and the selected site
        (per generation).
    N : int
        Diploid effective population size.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    coal_times : list of float
        Times (in generations backward from fixation) of coalescence events.
    n_remaining_B : int
        Number of lineages remaining in B at the end of the sweep.
    n_remaining_b : int
        Number of lineages remaining in b at the end of the sweep.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    two_N = 2 * N

    # All lineages start in B (the sweep just fixed)
    n_B = n_sample
    n_b = 0

    coal_times = []
    residual = rng.exponential(1.0)  # time until next event (in rate-units)

    # Process trajectory backward: from fixation (end) to origin (start)
    T_sweep = len(trajectory)
    for step in range(T_sweep - 1, -1, -1):
        x = trajectory[step]

        if n_B + n_b <= 1:
            break  # MRCA found

        # Compute rates for this time step
        bg_B = two_N * x
        bg_b = two_N * (1.0 - x)

        rate_coal_B = (
            (n_B * (n_B - 1) / (2.0 * bg_B))
            if (n_B >= 2 and bg_B > 0) else 0.0
        )
        rate_coal_b = (
            (n_b * (n_b - 1) / (2.0 * bg_b))
            if (n_b >= 2 and bg_b > 0) else 0.0
        )
        rate_mig_Bb = n_B * r_site * (1.0 - x) if n_B > 0 else 0.0
        rate_mig_bB = n_b * r_site * x if n_b > 0 else 0.0

        total_rate = rate_coal_B + rate_coal_b + rate_mig_Bb + rate_mig_bB

        if total_rate == 0:
            continue

        # Can an event happen within this generation?
        # The residual is in units of "rate * time". One generation = total_rate.
        while residual < total_rate:
            # An event happens within this step
            backward_gen = T_sweep - step
            coal_times_entry = None

            # Determine which event
            u = rng.random() * total_rate
            if u < rate_coal_B:
                # Coalescence in B
                n_B -= 1
                coal_times_entry = backward_gen
            elif u < rate_coal_B + rate_coal_b:
                # Coalescence in b
                n_b -= 1
                coal_times_entry = backward_gen
            elif u < rate_coal_B + rate_coal_b + rate_mig_Bb:
                # Migration B -> b
                n_B -= 1
                n_b += 1
            else:
                # Migration b -> B
                n_b -= 1
                n_B += 1

            if coal_times_entry is not None:
                coal_times.append(coal_times_entry)

            if n_B + n_b <= 1:
                break

            # Draw next event residual
            residual = rng.exponential(1.0)

        # Subtract this step's contribution from the residual
        residual -= total_rate

    # At the origin of the sweep, all B lineages coalesce
    # (they all trace to the single original mutant)
    if n_B > 1:
        for _ in range(n_B - 1):
            coal_times.append(T_sweep)
        n_B = 1

    return coal_times, n_B, n_b


def escape_probability(r_site, s, N):
    """Approximate probability that a lineage escapes a hard sweep.

    Parameters
    ----------
    r_site : float
        Recombination rate between neutral locus and selected site.
    s : float
        Selection coefficient.
    N : int
        Diploid population size.

    Returns
    -------
    p_escape : float
        Probability of escaping the sweep via recombination.
    """
    return 1.0 - np.exp(-r_site / s * 2 * np.log(2 * N))


def demonstrate_bottleneck(N, alpha, n_sample=20, n_reps=500, seed=42):
    """Compare coalescence times under a sweep vs. neutrality.

    Parameters
    ----------
    N : int
        Diploid population size.
    alpha : float
        Scaled selection coefficient 2Ns.
    n_sample : int
        Number of sampled lineages.
    n_reps : int
        Number of simulation replicates.
    seed : int
        Random seed.

    Returns
    -------
    sweep_mean_times : list of float
        Mean coalescence times from each sweep replicate.
    neutral_expected : float
        Expected total coalescence time under neutrality.
    """
    rng = np.random.default_rng(seed)
    s = alpha / (2 * N)

    # Simulate under sweep: coalescence times within the sweep
    sweep_mean_times = []
    for _ in range(n_reps):
        traj = deterministic_trajectory(s, N)
        # Use r_site = 0 (perfectly linked) to see pure bottleneck effect
        coal_times, _, _ = structured_coalescent_sweep(
            traj, n_sample, r_site=0.0, N=N, rng=rng
        )
        if coal_times:
            sweep_mean_times.append(np.mean(coal_times))

    # Neutral comparison: expected total coalescence time for k lineages
    neutral_expected = sum(
        2 * N / (k * (k - 1) / 2) for k in range(n_sample, 1, -1)
    )

    return sweep_mean_times, neutral_expected


def simulate_sweep_genealogy(n_sample, N, s, r_site, tau_gen=0,
                              trajectory_mode='deterministic', rng=None):
    """Simulate a genealogy at a neutral locus linked to a selective sweep.

    The full algorithm:
    1. Generate the allele frequency trajectory.
    2. Run neutral coalescent from present to tau (time since fixation).
    3. Run structured coalescent through the sweep.
    4. Run neutral coalescent for remaining lineages before the sweep.

    Parameters
    ----------
    n_sample : int
        Number of sampled lineages.
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    r_site : float
        Recombination rate between neutral locus and selected site.
    tau_gen : int
        Generations since the sweep completed (0 = just fixed).
    trajectory_mode : str
        'deterministic' or 'stochastic'.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    total_coal_times : list of float
        All coalescence times (in generations backward from present).
    """
    if rng is None:
        rng = np.random.default_rng()

    two_N = 2 * N
    n = n_sample
    total_coal_times = []

    # Phase 1: Neutral coalescent from present to tau
    t_current = 0
    while n > 1 and t_current < tau_gen:
        # Rate of coalescence: choose(n,2) / (2N)
        rate = n * (n - 1) / (2.0 * two_N)
        wait = rng.exponential(1.0 / rate)
        if t_current + wait > tau_gen:
            break  # No coalescence before the sweep phase starts
        t_current += wait
        n -= 1
        total_coal_times.append(t_current)

    if n <= 1:
        return total_coal_times

    # Phase 2: Generate trajectory and run structured coalescent
    if trajectory_mode == 'deterministic':
        traj = deterministic_trajectory(s, N)
    else:
        traj = stochastic_trajectory(s, N, rng=rng)

    coal_times_sweep, n_B, n_b = structured_coalescent_sweep(
        traj, n, r_site, N, rng=rng
    )

    # Offset sweep coalescence times by tau
    for ct in coal_times_sweep:
        total_coal_times.append(tau_gen + ct)

    # Phase 3: Neutral coalescent for remaining lineages
    n_remaining = n_B + n_b
    t_current = tau_gen + len(traj)
    while n_remaining > 1:
        rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
        wait = rng.exponential(1.0 / rate)
        t_current += wait
        n_remaining -= 1
        total_coal_times.append(t_current)

    return total_coal_times


# ============================================================================
# Functions from sweep_types.rst
# ============================================================================

def hard_sweep_genealogy(n_sample, N, s, r_site, rng=None):
    """Simulate a hard sweep genealogy at a linked neutral locus.

    A hard sweep starts from frequency 1/(2N) and reaches fixation.

    Parameters
    ----------
    n_sample : int
        Number of sampled lineages.
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    r_site : float
        Recombination rate to the selected site.
    rng : np.random.Generator, optional

    Returns
    -------
    coal_times : list of float
        Coalescence times (generations backward from present).
    n_escaped : int
        Number of lineages that escaped to background b.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    traj = deterministic_trajectory(s, N, x0=1.0 / (2 * N))
    coal_times, n_B, n_b = structured_coalescent_sweep(
        traj, n_sample, r_site, N, rng=rng
    )
    return coal_times, n_b


def soft_sweep_standing_variation(n_sample, N, s, x0, r_site, rng=None):
    """Simulate a soft sweep from standing variation.

    The trajectory runs from x0 to fixation under selection, then
    the remaining lineages coalesce under neutral drift.

    Parameters
    ----------
    n_sample : int
        Number of sampled lineages.
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    x0 : float
        Frequency of the allele at the onset of selection.
    r_site : float
        Recombination rate to the selected site.
    rng : np.random.Generator, optional

    Returns
    -------
    coal_times : list of float
        All coalescence times.
    n_surviving_B : int
        Number of distinct B lineages that survived the sweep
        (these still need to coalesce under neutral drift from x0).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Step 1: trajectory from x0 to fixation (under selection)
    traj = deterministic_trajectory(s, N, x0=x0)

    # Step 2: structured coalescent through the sweep
    coal_times_sweep, n_B, n_b = structured_coalescent_sweep(
        traj, n_sample, r_site, N, rng=rng
    )

    # Step 3: the n_B surviving lineages must coalesce under neutral drift
    coal_times_all = list(coal_times_sweep)
    sweep_duration = len(traj)

    # Remaining lineages (from both B and b) coalesce neutrally
    n_remaining = n_B + n_b
    two_N = 2 * N
    t_offset = sweep_duration
    while n_remaining > 1:
        rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
        wait = rng.exponential(1.0 / rate)
        t_offset += wait
        n_remaining -= 1
        coal_times_all.append(t_offset)

    return coal_times_all, n_B


def expected_independent_origins(s, N, mu_a):
    """Expected number of independent origins of a beneficial allele
    that contribute to a soft sweep from recurrent mutation.

    From Pennings & Hermisson (2006): the expected number of
    independent origins that survive drift and contribute to
    fixation is approximately:

        E[K] = 2 * N * mu_a * log(2Ns) / s

    Parameters
    ----------
    s : float
        Selection coefficient.
    N : int
        Diploid effective population size.
    mu_a : float
        Mutation rate to the beneficial allele (per generation).

    Returns
    -------
    E_K : float
        Expected number of independent surviving origins.
    """
    two_N_s = 2 * N * s
    return 2 * N * mu_a * np.log(two_N_s) / s


def partial_sweep_genealogy(n_sample, N, s, c, r_site, rng=None):
    """Simulate a genealogy under a partial sweep.

    The beneficial allele is currently at frequency c < 1.

    Parameters
    ----------
    n_sample : int
        Total number of sampled lineages.
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    c : float
        Current frequency of the beneficial allele (0 < c < 1).
    r_site : float
        Recombination rate to the selected site.
    rng : np.random.Generator, optional

    Returns
    -------
    coal_times : list of float
        All coalescence times.
    initial_B : int
        Number of lineages initially in background B.
    initial_b : int
        Number of lineages initially in background b.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Assign each lineage to B or b based on current frequency c
    assignments = rng.random(n_sample) < c
    n_B = int(np.sum(assignments))
    n_b = n_sample - n_B

    # Generate trajectory from 1/(2N) to c
    traj = deterministic_trajectory(s, N, x0=1.0 / (2 * N))
    # Truncate trajectory at frequency c
    idx = np.searchsorted(traj, c)
    traj = traj[:idx + 1]
    if len(traj) > 0 and traj[-1] < c:
        traj = np.append(traj, c)

    # Run structured coalescent backward through this partial trajectory
    coal_times_sweep, n_B_final, n_b_final = structured_coalescent_sweep(
        traj, n_B + n_b, r_site, N, rng=rng
    )

    # Remaining lineages coalesce neutrally
    coal_times_all = list(coal_times_sweep)
    n_remaining = n_B_final + n_b_final
    two_N = 2 * N
    t_offset = len(traj)
    while n_remaining > 1:
        rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
        wait = rng.exponential(1.0 / rate)
        t_offset += wait
        n_remaining -= 1
        coal_times_all.append(t_offset)

    return coal_times_all, n_B, n_b


def compare_sweep_types(N, s, n_sample=50, n_reps=200, seed=42):
    """Compare diversity under hard, soft, and partial sweeps.

    Measures the total tree length (proportional to expected number
    of segregating sites) for each sweep type.

    Parameters
    ----------
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    n_sample : int
        Number of sampled lineages.
    n_reps : int
        Number of simulation replicates.
    seed : int
        Random seed.

    Returns
    -------
    hard_lengths : list of float
        Total tree lengths under hard sweep.
    soft_lengths : list of float
        Total tree lengths under soft sweep.
    neutral_lengths : list of float
        Total tree lengths under neutrality.
    """
    rng = np.random.default_rng(seed)

    hard_lengths = []
    soft_lengths = []
    neutral_lengths = []

    for _ in range(n_reps):
        # Hard sweep (perfectly linked)
        hard_ct, _ = hard_sweep_genealogy(n_sample, N, s, 0.0, rng=rng)
        if hard_ct:
            hard_lengths.append(sum(hard_ct))

        # Soft sweep (x0 = 0.05)
        soft_ct, _ = soft_sweep_standing_variation(
            n_sample, N, s, x0=0.05, r_site=0.0, rng=rng)
        if soft_ct:
            soft_lengths.append(sum(soft_ct))

        # Neutral coalescent
        n_temp = n_sample
        t = 0
        total = 0
        while n_temp > 1:
            rate = n_temp * (n_temp - 1) / (2.0 * 2 * N)
            wait = rng.exponential(1.0 / rate)
            t += wait
            total += t
            n_temp -= 1
        neutral_lengths.append(total)

    return hard_lengths, soft_lengths, neutral_lengths


# ============================================================================
# Functions from msprime_comparison.rst
# ============================================================================

def discoal_to_msprime(theta, rho, alpha, n, L, N):
    """Convert discoal's scaled parameters to msprime's raw parameters.

    Parameters
    ----------
    theta : float
        Population-scaled mutation rate (4*N*mu*L).
    rho : float
        Population-scaled recombination rate (4*N*r*L).
    alpha : float
        Population-scaled selection coefficient (2*N*s).
    n : int
        Sample size.
    L : int
        Sequence length in bp.
    N : int
        Diploid effective population size.

    Returns
    -------
    params : dict
        Dictionary of msprime parameters.
    """
    mu = theta / (4 * N * L)
    r = rho / (4 * N * L)
    s = alpha / (2 * N)

    return {
        'samples': n,
        'sequence_length': L,
        'mutation_rate': mu,
        'recombination_rate': r,
        'population_size': N,
        'selection_coefficient': s,
    }


def msprime_to_discoal(n, L, mu, r, s, N):
    """Convert msprime's raw parameters to discoal's scaled parameters.

    Parameters
    ----------
    n : int
        Sample size.
    L : int
        Sequence length in bp.
    mu : float
        Mutation rate per bp per generation.
    r : float
        Recombination rate per bp per generation.
    s : float
        Selection coefficient.
    N : int
        Diploid effective population size.

    Returns
    -------
    params : dict
        Dictionary of discoal command-line parameters.
    """
    theta = 4 * N * mu * L
    rho = 4 * N * r * L
    alpha = 2 * N * s

    return {
        'n': n,
        'L': L,
        'theta': theta,
        'rho': rho,
        'alpha': alpha,
    }


@dataclass
class SweepState:
    """State of the structured coalescent during a sweep."""
    n_B: int = 0          # lineages in beneficial background
    n_b: int = 0          # lineages in wild-type background
    coal_times_B: list = field(default_factory=list)
    coal_times_b: list = field(default_factory=list)
    migrations_Bb: int = 0
    migrations_bB: int = 0


def discoal_core(n, N, s, r_per_site, L, tau_4N=0.0,
                  trajectory_mode='deterministic', seed=42):
    """A Python translation of discoal's core simulation algorithm.

    Simulates a selective sweep at position L/2, with neutral sites
    along a chromosome of length L.

    Parameters
    ----------
    n : int
        Sample size (haploid).
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient (genic).
    r_per_site : float
        Recombination rate per site per generation.
    L : int
        Sequence length (number of sites).
    tau_4N : float
        Time since fixation in units of 4N generations.
    trajectory_mode : str
        'deterministic' or 'stochastic'.
    seed : int
        Random seed.

    Returns
    -------
    result : dict
        Contains 'selected_position', 'sweep_duration_gen', and 'sites'
        with diagnostic information about the sweep.
    """
    rng = np.random.default_rng(seed)
    two_N = 2 * N
    tau_gen = int(tau_4N * 4 * N)

    # --- Step 1: Generate trajectory ---
    x0 = 1.0 / two_N
    if trajectory_mode == 'deterministic':
        traj = []
        x = x0
        while x < 1.0 - 1.0 / two_N:
            traj.append(x)
            # One generation of logistic growth
            x = x * (1 + s) / (1 + s * x)
            # Clip to [0, 1]
            x = min(x, 1.0)
        traj.append(1.0)
        traj = np.array(traj)
    else:
        traj = stochastic_trajectory(s, N, rng=rng)

    T_sweep = len(traj)

    # --- Step 2: Simulate at multiple neutral sites ---
    selected_pos = L // 2
    site_tmrcas = []

    # Sample a few representative sites
    test_positions = [selected_pos,            # at the selected site
                      selected_pos + L // 10,  # nearby
                      selected_pos + L // 4,   # intermediate
                      0]                        # far away

    for pos in test_positions:
        r_site = r_per_site * abs(pos - selected_pos)

        state = SweepState(n_B=n, n_b=0)
        coal_times = []

        # Phase A: Neutral coalescent (present to tau)
        n_current = n
        t = 0
        while n_current > 1 and t < tau_gen:
            rate = n_current * (n_current - 1) / (2.0 * two_N)
            wait = rng.exponential(1.0 / rate)
            if t + wait > tau_gen:
                break
            t += wait
            n_current -= 1
            coal_times.append(t)

        # Phase B: Structured coalescent through sweep
        state.n_B = n_current
        state.n_b = 0

        for step in range(T_sweep - 1, -1, -1):
            x = traj[step]
            if state.n_B + state.n_b <= 1:
                break

            bg_B = max(two_N * x, 1.0)
            bg_b = max(two_N * (1.0 - x), 1.0)

            # Compute rates
            rate_cB = (
                state.n_B * (state.n_B - 1) / (2.0 * bg_B)
                if state.n_B >= 2 else 0
            )
            rate_cb = (
                state.n_b * (state.n_b - 1) / (2.0 * bg_b)
                if state.n_b >= 2 else 0
            )
            rate_mBb = (
                state.n_B * r_site * (1.0 - x)
                if state.n_B > 0 else 0
            )
            rate_mbB = (
                state.n_b * r_site * x
                if state.n_b > 0 else 0
            )
            total = rate_cB + rate_cb + rate_mBb + rate_mbB

            if total == 0:
                continue

            # Check if event happens in this generation
            if rng.exponential(1.0 / total) < 1.0:
                u = rng.random() * total
                t_event = tau_gen + (T_sweep - step)
                if u < rate_cB:
                    state.n_B -= 1
                    coal_times.append(t_event)
                elif u < rate_cB + rate_cb:
                    state.n_b -= 1
                    coal_times.append(t_event)
                elif u < rate_cB + rate_cb + rate_mBb:
                    state.n_B -= 1
                    state.n_b += 1
                    state.migrations_Bb += 1
                else:
                    state.n_b -= 1
                    state.n_B += 1
                    state.migrations_bB += 1

        # Forced coalescence at sweep origin for remaining B lineages
        if state.n_B > 1:
            for _ in range(state.n_B - 1):
                coal_times.append(tau_gen + T_sweep)
            state.n_B = 1

        # Phase C: Neutral coalescent before sweep
        n_remaining = state.n_B + state.n_b
        t = tau_gen + T_sweep
        while n_remaining > 1:
            rate = n_remaining * (n_remaining - 1) / (2.0 * two_N)
            wait = rng.exponential(1.0 / rate)
            t += wait
            n_remaining -= 1
            coal_times.append(t)

        tmrca = max(coal_times) if coal_times else 0
        site_tmrcas.append((pos, abs(pos - selected_pos), tmrca, state))

    # Report
    result = {
        'selected_position': selected_pos,
        'sweep_duration_gen': T_sweep,
        'sites': []
    }
    for pos, dist, tmrca, state in site_tmrcas:
        result['sites'].append({
            'position': pos,
            'distance_to_selected': dist,
            'r_to_selected': r_per_site * dist,
            'tmrca_gen': tmrca,
            'tmrca_coalescent_units': tmrca / two_N,
            'migrations_Bb': state.migrations_Bb,
            'migrations_bB': state.migrations_bB,
        })
    return result


def minimal_discoal(n, N, s, r_per_site, L, n_sites=50, seed=42):
    """A minimal discoal-like simulator that computes the diversity
    profile around a selective sweep.

    Parameters
    ----------
    n : int
        Sample size.
    N : int
        Diploid effective population size.
    s : float
        Selection coefficient.
    r_per_site : float
        Recombination rate per site per generation.
    L : int
        Sequence length.
    n_sites : int
        Number of sites to evaluate along the sequence.
    seed : int
        Random seed.

    Returns
    -------
    positions : ndarray
        Positions along the sequence.
    relative_diversity : ndarray
        Diversity at each position relative to neutral expectation.
    """
    rng = np.random.default_rng(seed)
    two_N = 2 * N
    selected_pos = L / 2

    # Generate trajectory (deterministic for speed)
    x0 = 1.0 / two_N
    traj = []
    x = x0
    while x < 1.0 - 1.0 / two_N:
        traj.append(x)
        x = x * (1 + s) / (1 + s * x)
        x = min(x, 1.0)
    traj.append(1.0)
    traj = np.array(traj)
    T_sweep = len(traj)

    # Neutral expected TMRCA for comparison
    neutral_expected_tmrca = sum(two_N / (k * (k - 1) / 2)
                                  for k in range(n, 1, -1))

    positions = np.linspace(0, L, n_sites)
    tmrcas = np.zeros(n_sites)

    for i, pos in enumerate(positions):
        r_site = r_per_site * abs(pos - selected_pos)

        # Run structured coalescent (simplified: track only TMRCA)
        n_B = n
        n_b = 0
        n_total = n

        for step in range(T_sweep - 1, -1, -1):
            if n_total <= 1:
                break
            x = traj[step]
            bg_B = max(two_N * x, 1.0)
            bg_b = max(two_N * (1.0 - x), 1.0)

            # Check for events this generation
            rate_cB = (
                n_B * (n_B - 1) / (2.0 * bg_B) if n_B >= 2 else 0
            )
            rate_cb = (
                n_b * (n_b - 1) / (2.0 * bg_b) if n_b >= 2 else 0
            )
            rate_mBb = n_B * r_site * (1.0 - x) if n_B > 0 else 0
            rate_mbB = n_b * r_site * x if n_b > 0 else 0
            total = rate_cB + rate_cb + rate_mBb + rate_mbB

            if total > 0 and rng.exponential(1.0 / total) < 1.0:
                u = rng.random() * total
                if u < rate_cB:
                    n_B -= 1
                    n_total -= 1
                elif u < rate_cB + rate_cb:
                    n_b -= 1
                    n_total -= 1
                elif u < rate_cB + rate_cb + rate_mBb:
                    n_B -= 1
                    n_b += 1
                else:
                    n_b -= 1
                    n_B += 1

        # Forced coalescence of remaining B lineages
        if n_B > 1:
            n_total -= (n_B - 1)
            n_B = 1

        # Neutral coalescent for remaining lineages
        t = T_sweep
        while n_total > 1:
            rate = n_total * (n_total - 1) / (2.0 * two_N)
            t += rng.exponential(1.0 / rate)
            n_total -= 1

        tmrcas[i] = t

    relative_diversity = tmrcas / neutral_expected_tmrca
    return positions, relative_diversity


# ============================================================================
# Demo function
# ============================================================================

def demo():
    """Demonstrate the discoal algorithm with examples from the documentation."""
    print("=" * 70)
    print("Mini-discoal: Coalescent Simulation with Selection")
    print("=" * 70)

    # --- Deterministic trajectory demo ---
    print("\n--- Deterministic Trajectory ---")
    N = 10_000
    s = 0.05
    traj = deterministic_trajectory(s, N)
    print(f"Sweep duration: {len(traj)} generations")
    print(f"In coalescent units (2N gen): {len(traj) / (2*N):.4f}")
    print(f"Theory: 2*ln(2N)/s = {2*np.log(2*N)/s:.0f} generations")

    # --- Stochastic trajectory demo ---
    print("\n--- Comparing Trajectories ---")
    N = 10_000
    s = 0.01  # 2Ns = 200
    det_traj = deterministic_trajectory(s, N)
    stoch_traj = stochastic_trajectory(s, N)
    print(f"Deterministic sweep: {len(det_traj)} generations")
    print(f"Stochastic sweep:    {len(stoch_traj)} generations")

    # --- Sweep duration table ---
    print("\n--- Sweep Duration Table ---")
    N = 10_000
    alphas = [10, 50, 100, 500, 1000, 5000]
    results = sweep_duration_table(N, alphas)
    print(f"{'alpha (2Ns)':>12} {'s':>10} {'T_sweep (gen)':>15} "
          f"{'T_sweep (coal)':>16} {'T_sweep (4N gen)':>18}")
    print("-" * 75)
    for r in results:
        print(f"{r['alpha']:12.0f} {r['s']:10.5f} {r['T_gen']:15.0f} "
              f"{r['T_coal']:16.4f} {r['T_4N']:18.6f}")

    # --- Coalescence rate demo ---
    print("\n--- Coalescence Rate Bottleneck ---")
    rate_B = coalescence_rate(5, 200)
    rate_neutral = coalescence_rate(5, 20000)
    print(f"Coalescence rate in B (x=0.01): {rate_B:.4f} per generation")
    print(f"Neutral coalescence rate:       {rate_neutral:.6f} per generation")
    print(f"Ratio (speed-up):               {rate_B / rate_neutral:.0f}x")

    # --- Escape probability demo ---
    print("\n--- Escape Probability ---")
    N = 10_000
    s = 0.01
    r_half = s / (2 * np.log(2 * N))
    print("Recombination distance at 50% escape probability:")
    print(f"  r_50 = s / (2*ln(2N)) = {r_half:.2e}")
    print(f"  Physical distance (at r=1e-8/bp): {r_half / 1e-8:.0f} bp")

    # --- Hard vs soft sweep demo ---
    print("\n--- Hard vs Soft Sweep ---")
    N = 10_000
    s = 0.01
    rng = np.random.default_rng(42)
    hard_times, _ = hard_sweep_genealogy(20, N, s, 0.0, rng=rng)
    soft_times, n_surv = soft_sweep_standing_variation(
        20, N, s, x0=0.1, r_site=0.0, rng=rng
    )
    print(f"Hard sweep (x0=1/2N): TMRCA = {max(hard_times):,.0f} gen")
    print(f"Soft sweep (x0=0.1):  TMRCA = {max(soft_times):,.0f} gen, "
          f"{n_surv} ancestral B lineages survived sweep")

    # --- Expected independent origins ---
    print("\n--- Expected Independent Origins (Soft vs Hard) ---")
    N = 10_000
    s = 0.01
    for mu_a in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        E_K = expected_independent_origins(s, N, mu_a)
        sweep_type = "hard" if E_K < 1 else "soft"
        print(f"mu_a = {mu_a:.0e}: E[K] = {E_K:.2f} origins --> "
              f"{sweep_type} sweep")

    # --- Parameter conversion demo ---
    print("\n--- Parameter Conversion ---")
    N = 10_000
    L = 100_000
    mu = 1.25e-8
    r = 1e-8
    s = 0.01
    discoal_params = msprime_to_discoal(100, L, mu, r, s, N)
    print("discoal command:")
    print(f"  discoal {discoal_params['n']} 1 {discoal_params['L']} "
          f"-t {discoal_params['theta']:.1f} "
          f"-r {discoal_params['rho']:.1f} "
          f"-ws 0 -a {discoal_params['alpha']:.0f} -x 0.5")

    msprime_params = discoal_to_msprime(
        discoal_params['theta'], discoal_params['rho'],
        discoal_params['alpha'], 100, L, N
    )
    print(f"\nmsprime equivalent:")
    print(f"  samples={msprime_params['samples']}")
    print(f"  sequence_length={msprime_params['sequence_length']}")
    print(f"  mutation_rate={msprime_params['mutation_rate']:.2e}")
    print(f"  recombination_rate={msprime_params['recombination_rate']:.2e}")
    print(f"  population_size={msprime_params['population_size']}")
    print(f"  s={msprime_params['selection_coefficient']}")

    # --- discoal core demo ---
    print("\n--- discoal Core Algorithm ---")
    result = discoal_core(
        n=20, N=10_000, s=0.01, r_per_site=1e-8, L=100_000, seed=42
    )
    print(f"Sweep duration: {result['sweep_duration_gen']} generations\n")
    print(f"{'Position':>10} {'Distance':>10} {'r_site':>12} "
          f"{'TMRCA (gen)':>12} {'TMRCA (coal)':>14} {'Mig B->b':>10}")
    print("-" * 75)
    for site in result['sites']:
        print(f"{site['position']:>10} {site['distance_to_selected']:>10} "
              f"{site['r_to_selected']:>12.2e} "
              f"{site['tmrca_gen']:>12,.0f} "
              f"{site['tmrca_coalescent_units']:>14.4f} "
              f"{site['migrations_Bb']:>10}")

    # --- Diversity profile demo ---
    print("\n--- Diversity Profile Around a Hard Sweep ---")
    positions, rel_div = minimal_discoal(
        n=20, N=10_000, s=0.01, r_per_site=1e-8, L=200_000, n_sites=100
    )
    print("Diversity profile around a hard sweep (s=0.01, N=10000):")
    print(f"{'Position':>10} {'Relative diversity':>20}")
    for j in range(0, len(positions), 10):
        print(f"{positions[j]:>10.0f} {rel_div[j]:>20.4f}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
