"""
Mini-implementation of the Threads algorithm for ARG inference.

Threads is a deterministic method for inferring Ancestral Recombination Graphs
(ARGs) from phased genotype data. It produces threading instructions -- for each
sample at each genomic position, a threading target (the closest genealogical
relative) and a coalescence time.

The Threads pipeline has three stages:

1. PBWT Haplotype Matching: Uses the positional Burrows-Wheeler transform to
   identify a small set of candidate haplotype matches for each sample, reducing
   the search space from O(N) to O(L) candidates per sample (L << N).

2. Memory-Efficient Viterbi: A branch-and-bound implementation of the Viterbi
   algorithm under the Li-Stephens model that finds the optimal threading path
   in O(NM) time and O(N) average memory.

3. Segment Dating: Assigns coalescence times to each Viterbi segment using
   likelihood-based and Bayesian estimators that model segments as pairwise IBD
   regions.

This module implements the mathematical estimators from the Segment Dating step,
covering both maximum likelihood and Bayesian approaches under constant and
piecewise-constant demographic models.

References
----------
Brandt, Chiang, Guo et al. (2024). Threads: Threading Instructions for
Ancestral Recombination Graphs.
"""

import numpy as np
from scipy.special import gammainc  # regularized lower incomplete gamma


# ============================================================================
# Maximum Likelihood Estimators
# ============================================================================

def mle_recombination_only(rho):
    """Maximum likelihood estimator of coalescence time from recombination only.

    Under the SMC model, the length of an IBD segment follows an exponential
    distribution with rate 1/(2t). The MLE is obtained by differentiating the
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
    K = len(coal_rates)
    T = list(time_boundaries)

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

        # Regularized lower incomplete gamma
        z_upper = lambda_k * T_upper if not np.isinf(T_upper) else np.inf
        z_lower = lambda_k * T[k]

        P3_upper = gammainc(3, z_upper) if not np.isinf(z_upper) else 1.0
        P3_lower = gammainc(3, z_lower)
        P2_upper = gammainc(2, z_upper) if not np.isinf(z_upper) else 1.0
        P2_lower = gammainc(2, z_lower)

        numerator += prefactor * (2.0 / lambda_k**3) * (P3_upper - P3_lower)
        denominator += prefactor * (1.0 / lambda_k**2) * (P2_upper - P2_lower)

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
    K = len(coal_rates)
    T = list(time_boundaries)

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

        P_num_upper = gammainc(a_num, z_upper) if not np.isinf(z_upper) else 1.0
        P_num_lower = gammainc(a_num, z_lower)
        P_den_upper = gammainc(a_den, z_upper) if not np.isinf(z_upper) else 1.0
        P_den_lower = gammainc(a_den, z_lower)

        mu_m = mu**m
        numerator += prefactor * mu_m * (m + 2) / lambda_k**(m + 3) * (P_num_upper - P_num_lower)
        denominator += prefactor * mu_m / lambda_k**(m + 2) * (P_den_upper - P_den_lower)

    if denominator == 0:
        return np.inf
    return numerator / denominator


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
