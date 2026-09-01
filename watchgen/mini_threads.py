"""
Mini-implementation of the Threads algorithm for ARG inference.

Threads is a deterministic method for inferring Ancestral Recombination Graphs
(ARGs) from phased genotype data. It produces threading instructions -- for each
sample at each genomic position, a threading target (the closest genealogical
relative) and a coalescence time.

The Threads pipeline has three stages:

1. PBWT Haplotype Matching: Uses the positional Burrows-Wheeler transform to
   identify a sparse, locally selected candidate panel for each sample, reducing
   the practical Li--Stephens state space far below the full panel.

2. Memory-Efficient Viterbi: A branch-and-bound implementation of the Viterbi
   algorithm under the Li-Stephens model that finds the optimal threading path
   in O(NM) time and O(N) average memory.

3. Segment Dating: Assigns coalescence times to each Viterbi segment using
   Bayesian estimators that model segments as pairwise IBD regions.  The MLEs
   below are useful derivations from the paper, but production Threads uses the
   Bayesian piecewise-demographic estimator (and a high-mutation shortcut).

This module implements the mathematical estimators from the Segment Dating step,
covering both maximum likelihood and Bayesian approaches under constant and
piecewise-constant demographic models.

References
----------
Gunnarsson, Zhu, Zhang et al. (2024). A scalable approach for genome-wide
inference of ancestral recombination graphs.
"""

import numpy as np
from scipy.special import gammaincc  # regularized upper incomplete gamma


def _validate_count(m):
    if (isinstance(m, bool) or not np.isscalar(m) or not np.isfinite(m) or
            int(m) != m or m < 0):
        raise ValueError("m must be a non-negative integer")
    return int(m)


def _validate_piecewise(time_boundaries, coal_rates):
    """Return validated one-boundary-per-epoch arrays."""
    times = np.asarray(time_boundaries, dtype=float)
    rates = np.asarray(coal_rates, dtype=float)
    if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates):
        raise ValueError("time_boundaries and coal_rates must be equal-length 1D arrays")
    if len(times) == 0 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_boundaries must start at 0 and be strictly increasing")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("boundaries must be finite and coalescence rates positive")
    return times, rates


# ============================================================================
# Maximum Likelihood Estimators
# ============================================================================

def mle_recombination_only(rho):
    """Maximum likelihood estimator of coalescence time from recombination only.

    Conditional on age ``t``, genetic length in Morgans has exponential rate
    ``2t``.  With ``rho`` equal to twice the observed genetic length, the
    likelihood is proportional to ``t * exp(-t*rho)``. The MLE follows by differentiating the
    log-likelihood log(2t) - t*rho and setting to zero:

        t_hat = 1 / rho

    Parameters
    ----------
    rho : float
        Recombination measure for the segment: 2 * 0.01 * l_cM.

    Returns
    -------
    float
        Maximum likelihood age estimate (in generations).
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive")
    return 1.0 / rho


def mle_recombination_and_mutations(m, rho, mu):
    """MLE of coalescence time from recombination and mutations.

    Adding the Poisson mutation model with m heterozygous sites to the
    recombination likelihood, the MLE becomes:

        t_hat = (m + 1) / (rho + mu)

    The numerator m+1 counts the m observed mutations plus one count from
    the recombination boundary. The denominator rho+mu is the total rate at
    which events accumulate with time.

    Parameters
    ----------
    m : int
        Number of heterozygous sites in the segment.
    rho : float
        Recombination measure for the segment.
    mu : float
        Mutation measure: 2 * c * l_bp.

    Returns
    -------
    float
        Maximum likelihood age estimate (in generations).
    """
    m = _validate_count(m)
    if not np.isfinite(rho) or not np.isfinite(mu) or rho < 0 or mu < 0 or rho + mu <= 0:
        raise ValueError("rho and mu must be finite, non-negative, and not both zero")
    return (m + 1) / (rho + mu)


# ============================================================================
# Bayesian Estimators (constant population size)
# ============================================================================

def bayesian_recombination_only(rho, gamma):
    """Bayesian posterior mean of coalescence time (recombination only).

    Places an exponential prior Exp(gamma) on the segment age, where gamma
    is the coalescence rate (1/N_e). The posterior is Erlang-2 with rate
    rho + gamma, giving:

        E[t | rho] = 2 / (rho + gamma)

    Parameters
    ----------
    rho : float
        Recombination measure for the segment.
    gamma : float
        Coalescence rate (1 / N_e).

    Returns
    -------
    float
        Posterior mean age estimate (in generations).
    """
    if (not np.isfinite(rho) or not np.isfinite(gamma) or
            rho < 0 or gamma <= 0):
        raise ValueError("rho must be non-negative and gamma must be positive")
    return 2.0 / (rho + gamma)


def bayesian_full(m, rho, mu, gamma):
    """Bayesian posterior mean of coalescence time (recombination + mutations).

    Including the mutation likelihood with the exponential prior, the posterior
    is Erlang-(m+2) with rate rho + mu + gamma, giving:

        E[t | rho, m] = (m + 2) / (rho + mu + gamma)

    The numerator gains an extra count from the prior compared to the MLE,
    and the denominator includes the coalescence rate gamma.

    Parameters
    ----------
    m : int
        Number of heterozygous sites in the segment.
    rho : float
        Recombination measure for the segment.
    mu : float
        Mutation measure.
    gamma : float
        Coalescence rate (1 / N_e).

    Returns
    -------
    float
        Posterior mean age estimate (in generations).
    """
    m = _validate_count(m)
    if (not np.isfinite(rho) or not np.isfinite(mu) or not np.isfinite(gamma) or
            rho < 0 or mu < 0 or gamma <= 0):
        raise ValueError("rho and mu must be non-negative and gamma must be positive")
    return (m + 2) / (rho + mu + gamma)


# ============================================================================
# Piecewise-Constant Demographic Model Estimators
# ============================================================================

def piecewise_constant_bayesian_recomb_only(rho, time_boundaries, coal_rates):
    """Bayesian posterior mean under a piecewise-constant demographic model.

    Uses recombination only. The prior becomes piecewise exponential with
    coalescence rate gamma_k = 1/N_e^(k) in each time interval [T_k, T_{k+1}).

    Parameters
    ----------
    rho : float
        Recombination rate for the segment.
    time_boundaries : array-like
        Boundaries of the time intervals [T_0=0, T_1, T_2, ...].
        The last interval extends to infinity.
    coal_rates : array-like
        Coalescence rate gamma_k = 1/N_e^(k) for each interval.

    Returns
    -------
    float
        E[t | rho] under the piecewise-constant model.
    """
    T, coal_rates = _validate_piecewise(time_boundaries, coal_rates)
    if not np.isfinite(rho) or rho < 0:
        raise ValueError("rho must be finite and non-negative")
    K = len(coal_rates)

    numerator = 0.0
    denominator = 0.0

    for k in range(K):
        gamma_k = coal_rates[k]
        lambda_k = rho + gamma_k

        # Cumulative integral of rate up to T_k
        cum_rate = 0.0
        for j in range(k):
            delta_j = T[j + 1] - T[j]
            cum_rate += delta_j * coal_rates[j]

        prefactor = gamma_k * np.exp(-cum_rate + T[k] * gamma_k)

        # Upper boundary for this interval
        if k < K - 1:
            T_upper = T[k + 1]
        else:
            T_upper = np.inf

        # Threads C++ uses differences of the regularized upper incomplete
        # gamma.  This is also stable when both lower-CDF values round to one.
        z_upper = lambda_k * T_upper if not np.isinf(T_upper) else np.inf
        z_lower = lambda_k * T[k]

        q3 = gammaincc(3, z_lower) - (0.0 if np.isinf(z_upper) else gammaincc(3, z_upper))
        q2 = gammaincc(2, z_lower) - (0.0 if np.isinf(z_upper) else gammaincc(2, z_upper))

        numerator += prefactor * (2.0 / lambda_k**3) * q3
        denominator += prefactor * (1.0 / lambda_k**2) * q2

    if denominator == 0:
        return np.inf
    return numerator / denominator


def piecewise_constant_bayesian_full(rho, mu, m, time_boundaries, coal_rates):
    """Bayesian posterior mean under a piecewise-constant model with mutations.

    Extends the piecewise-constant demographic prior to include mutation
    information. With m heterozygous sites and lambda_k = rho + mu + gamma_k,
    computes E[t | rho, m] using regularized incomplete gamma functions.

    Parameters
    ----------
    rho : float
        Recombination rate.
    mu : float
        Mutation rate.
    m : int
        Number of heterozygous sites.
    time_boundaries : array-like
        Boundaries of time intervals.
    coal_rates : array-like
        Coalescence rates per interval.

    Returns
    -------
    float
        E[t | rho, m] under the piecewise-constant model.
    """
    m = _validate_count(m)
    T, coal_rates = _validate_piecewise(time_boundaries, coal_rates)
    if (not np.isfinite(rho) or not np.isfinite(mu) or rho < 0 or mu < 0):
        raise ValueError("rho and mu must be finite and non-negative")
    K = len(coal_rates)

    numerator = 0.0
    denominator = 0.0

    for k in range(K):
        gamma_k = coal_rates[k]
        lambda_k = rho + mu + gamma_k

        cum_rate = 0.0
        for j in range(k):
            delta_j = T[j + 1] - T[j]
            cum_rate += delta_j * coal_rates[j]

        prefactor = gamma_k * np.exp(-cum_rate + T[k] * gamma_k)

        if k < K - 1:
            T_upper = T[k + 1]
        else:
            T_upper = np.inf

        z_upper = lambda_k * T_upper if not np.isinf(T_upper) else np.inf
        z_lower = lambda_k * T[k]

        a_num = m + 3
        a_den = m + 2

        q_num = gammaincc(a_num, z_lower) - (
            0.0 if np.isinf(z_upper) else gammaincc(a_num, z_upper))
        q_den = gammaincc(a_den, z_lower) - (
            0.0 if np.isinf(z_upper) else gammaincc(a_den, z_upper))

        # mu**m is common to every epoch and cancels from this ratio.  Omitting
        # it avoids 0/0 and underflow without changing positive-mu results.
        numerator += prefactor * (m + 2) / lambda_k**(m + 3) * q_num
        denominator += prefactor / lambda_k**(m + 2) * q_den

    if denominator == 0:
        return np.inf
    return numerator / denominator


def threads_date_segment(m, cm_size, bp_size, mutation_rate,
                         time_boundaries, population_sizes, sparse=False):
    """Mirror the production ``ThreadsFastLS::date_segment`` decision logic.

    The official implementation uses the mutation-free piecewise estimator for
    sparse data.  For sequence data with at most 15 differences it uses the full
    piecewise estimator.  Above 15 it switches to a constant-rate approximation
    based on ``Demography::std_to_gen(1)``.
    """
    m = _validate_count(m)
    times = np.asarray(time_boundaries, dtype=float)
    sizes = np.asarray(population_sizes, dtype=float)
    if times.ndim != 1 or sizes.ndim != 1 or len(times) != len(sizes):
        raise ValueError("time_boundaries and population_sizes must be equal-length")
    if len(times) == 0 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_boundaries must start at 0 and be strictly increasing")
    if np.any(~np.isfinite(times)) or np.any(~np.isfinite(sizes)) or np.any(sizes <= 0):
        raise ValueError("times must be finite and population sizes positive")
    if (not np.isfinite(cm_size) or not np.isfinite(bp_size) or
            not np.isfinite(mutation_rate) or cm_size < 0 or bp_size < 0 or
            mutation_rate < 0):
        raise ValueError("segment sizes and mutation_rate must be finite and non-negative")

    rho = 2.0 * 0.01 * cm_size
    rates = 1.0 / sizes
    if sparse:
        return piecewise_constant_bayesian_recomb_only(rho, times, rates)

    mu = 2.0 * mutation_rate * bp_size
    if m <= 15:
        return piecewise_constant_bayesian_full(rho, mu, m, times, rates)

    std_times = np.zeros(len(times))
    if len(times) > 1:
        std_times[1:] = np.cumsum(np.diff(times) / sizes[:-1])
    epoch = np.searchsorted(std_times, 1.0, side="right") - 1
    expected_time = times[epoch] + (1.0 - std_times[epoch]) * sizes[epoch]
    gamma = 1.0 / expected_time
    return bayesian_full(m, rho, mu, gamma)


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the Threads dating estimators with example values."""
    print("=" * 65)
    print("Threads Segment Dating Estimators")
    print("=" * 65)

    # Set up segment parameters
    l_cM = 1.0      # 1 centimorgan segment
    l_bp = 1e6      # ~1 Mb
    c = 1.25e-8     # per-base mutation rate
    rho = 2 * 0.01 * l_cM
    mu = 2 * c * l_bp

    print(f"\nSegment: {l_cM} cM, {l_bp/1e6:.0f} Mb")
    print(f"  rho = {rho:.4f}, mu = {mu:.5f}")

    # --- MLE estimators ---
    print("\n--- Maximum Likelihood Estimators ---")
    t_recomb = mle_recombination_only(rho)
    print(f"  MLE (recomb only): {t_recomb:.1f} generations")
    for m in [0, 1, 3, 10]:
        t_full = mle_recombination_and_mutations(m, rho, mu)
        print(f"  MLE (recomb + {m:2d} hets): {t_full:.1f} generations")

    # --- Bayesian estimators (constant N_e) ---
    N_e = 10000
    gamma = 1.0 / N_e
    print(f"\n--- Bayesian Estimators (N_e = {N_e}) ---")
    print(f"  gamma = {gamma:.6f}")
    t_bayes_r = bayesian_recombination_only(rho, gamma)
    print(f"  Bayes (recomb only): {t_bayes_r:.1f} generations")
    for m in [0, 1, 3, 10]:
        t_mle = mle_recombination_and_mutations(m, rho, mu)
        t_bayes = bayesian_full(m, rho, mu, gamma)
        print(f"  m={m:2d}: MLE = {t_mle:8.1f}, Bayes = {t_bayes:8.1f} generations")

    # --- Piecewise-constant demography ---
    print("\n--- Piecewise-Constant Demographic Model ---")
    print("  Single-epoch (should match constant-size Bayesian):")
    time_boundaries_single = [0.0]
    coal_rates_single = [gamma]
    for m in [0, 3, 10]:
        t_const = bayesian_full(m, rho, mu, gamma)
        t_pw = piecewise_constant_bayesian_full(
            rho, mu, m, time_boundaries_single, coal_rates_single)
        print(f"    m={m:2d}: constant = {t_const:.2f}, "
              f"piecewise = {t_pw:.2f} generations")

    print("\n  Two-epoch model (bottleneck):")
    time_boundaries = [0.0, 200.0]
    coal_rates = [0.001, 0.01]  # Recent large pop, then bottleneck
    print(f"    [0, 200): gamma = {coal_rates[0]}, "
          f"[200, inf): gamma = {coal_rates[1]}")
    t_recomb_pw = piecewise_constant_bayesian_recomb_only(
        rho, time_boundaries, coal_rates)
    print(f"    Recomb-only estimate: {t_recomb_pw:.2f} generations")
    for m in [0, 1, 5]:
        t_pw = piecewise_constant_bayesian_full(
            rho, mu, m, time_boundaries, coal_rates)
        print(f"    m={m}: full estimate = {t_pw:.2f} generations")

    # --- Key mathematical properties ---
    print("\n--- Key Properties ---")
    rho_test = 0.1
    gamma_test = 0.001
    mle_val = mle_recombination_only(rho_test)
    bayes_val = bayesian_recombination_only(rho_test, gamma_test)
    print(f"  MLE (1/rho) = {mle_val:.4f}")
    print(f"  Bayes (2/(rho+gamma)) = {bayes_val:.4f}")
    print(f"  Bayesian > MLE when gamma < rho: {bayes_val > mle_val} "
          f"(gamma={gamma_test}, rho={rho_test})")

    m_test = 5
    erlang_mean = (m_test + 2) / (rho_test + mu + gamma_test)
    bayes_full_val = bayesian_full(m_test, rho_test, mu, gamma_test)
    print(f"  Erlang-(m+2) mean matches bayesian_full: "
          f"{np.isclose(erlang_mean, bayes_full_val)}")

    print("\nDone.")


if __name__ == "__main__":
    demo()
