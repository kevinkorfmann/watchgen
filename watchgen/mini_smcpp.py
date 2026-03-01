"""
Mini SMC++ -- a pedagogical implementation of the SMC++ algorithm.

SMC++ (Terhorst, Kamm & Song, 2017) extends PSMC from a single diploid genome
to multiple unphased diploid genomes by introducing a **distinguished lineage**
whose coalescence time is the hidden state in an HMM.  The remaining n-1
undistinguished lineages form a demographic background that modifies the
coalescence rate.

The algorithm has four main components:

1. **The Distinguished Lineage** -- one lineage is singled out and its
   coalescence time T is tracked as a hidden variable.  The remaining n-1
   lineages are interchangeable and tracked only by their count.

2. **The ODE System** -- a system of ODEs tracks the probability p_j(t) that
   j undistinguished lineages remain at time t.  The matrix exponential of the
   rate matrix gives exact transition probabilities for piecewise-constant
   population size.

3. **The Continuous HMM** -- a modified transition matrix built from the ODE
   rates, combined via composite likelihood across pairs of sites.  Gradient-
   based optimization (L-BFGS-B) estimates the piecewise-constant population
   size function lambda(t).

4. **Population Splits** -- cross-population analysis via modified ODEs that
   track lineage counts before and after a population split, enabling joint
   estimation of population-specific size histories and split times.

Reference
---------
Terhorst, J., Kamm, J. A., & Song, Y. S. (2017). Robust and scalable
inference of population history from hundreds of unphased whole genomes.
*Nature Genetics*, 49(2), 303-309.
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Overview (overview.rst)
# ---------------------------------------------------------------------------

def expected_first_coalescence(n, N):
    """Expected time to first coalescence among n lineages in population N.

    With n lineages, the rate of coalescence (any pair finding a common ancestor)
    is C(n,2) / N = n(n-1) / (2N). The expected waiting time is the inverse of
    this rate.

    Parameters
    ----------
    n : int
        Number of haploid lineages (= 2 * number of diploid individuals).
    N : int
        Effective population size (diploid).

    Returns
    -------
    float
        Expected time to first coalescence, in generations.
    """
    rate = n * (n - 1) / (2 * N)
    return 1 / rate


# ---------------------------------------------------------------------------
# Distinguished Lineage (distinguished_lineage.rst)
# ---------------------------------------------------------------------------

def undistinguished_coalescence_rate(j, lam):
    """Rate at which j undistinguished lineages coalesce among themselves.

    Parameters
    ----------
    j : int
        Number of undistinguished lineages currently present.
    lam : float
        Relative population size lambda(t) at current time.

    Returns
    -------
    float
        Coalescence rate C(j,2) / lambda.
    """
    return j * (j - 1) / (2 * lam)


def distinguished_coalescence_rate(j, lam):
    """Rate at which the distinguished lineage coalesces with an undistinguished one.

    Parameters
    ----------
    j : int
        Number of undistinguished lineages currently present.
    lam : float
        Relative population size lambda(t) at current time.

    Returns
    -------
    float
        Rate j / lambda.
    """
    return j / lam


def emission_unphased(genotype, t, theta):
    """Emission probability for an unphased diploid genotype.

    Parameters
    ----------
    genotype : int
        Number of derived alleles: 0, 1, or 2.
    t : float
        Coalescence time of the distinguished lineage.
    theta : float
        Scaled mutation rate per bin.

    Returns
    -------
    float
        P(genotype | T = t).
    """
    # Probability that the distinguished lineage carries the derived allele
    # is approximately proportional to coalescence time (under infinite-sites
    # model, each lineage mutates with rate theta/2 per unit time).
    p_derived = 1 - np.exp(-theta * t)

    if genotype == 0:
        return (1 - p_derived) ** 2
    elif genotype == 1:
        return 2 * p_derived * (1 - p_derived)
    else:  # genotype == 2
        return p_derived ** 2


def compute_h(t, p_j, lam):
    """Compute the effective coalescence rate h(t) of the distinguished lineage.

    Parameters
    ----------
    t : float
        Time (used only for lambda evaluation).
    p_j : ndarray of shape (n,)
        p_j[j] = P(J(t) = j), for j = 0, 1, ..., n-1.
    lam : float
        Relative population size lambda(t).

    Returns
    -------
    float
        The effective coalescence rate h(t).
    """
    n_minus_1 = len(p_j) - 1
    h = 0.0
    for j in range(1, n_minus_1 + 1):
        h += j / lam * p_j[j]
    return h


# ---------------------------------------------------------------------------
# ODE System (ode_system.rst)
# ---------------------------------------------------------------------------

def build_rate_matrix(n_undist):
    """Build the rate matrix Q for the undistinguished lineage count process.

    Parameters
    ----------
    n_undist : int
        Number of undistinguished lineages at time 0 (= n - 1).

    Returns
    -------
    Q : ndarray of shape (n_undist, n_undist)
        Rate matrix. States are indexed 1, 2, ..., n_undist, stored as
        0-indexed array positions.
    """
    Q = np.zeros((n_undist, n_undist))
    for j in range(1, n_undist + 1):
        # j is the number of undistinguished lineages (1-indexed)
        # Array index is j - 1
        idx = j - 1
        # Outflow: C(j,2) = j*(j-1)/2
        Q[idx, idx] = -j * (j - 1) / 2
        # Inflow from state j+1 (if it exists)
        if j < n_undist:
            Q[idx, idx + 1] = (j + 1) * j / 2
    return Q


def solve_ode_piecewise(n_undist, time_breaks, lambdas):
    """Solve the ODE system for piecewise-constant population size.

    Parameters
    ----------
    n_undist : int
        Number of undistinguished lineages at time 0.
    time_breaks : array-like
        Time points [t_0, t_1, ..., t_K] defining intervals.
    lambdas : array-like
        Relative population sizes [lambda_0, ..., lambda_{K-1}] in each interval.

    Returns
    -------
    p_at_breaks : ndarray of shape (K+1, n_undist)
        p_at_breaks[k, j-1] = P(J(t_k) = j) for j = 1, ..., n_undist.
    """
    Q = build_rate_matrix(n_undist)

    # Initial condition: all n_undist lineages present
    p = np.zeros(n_undist)
    p[-1] = 1.0  # p_{n_undist}(0) = 1

    p_at_breaks = np.zeros((len(time_breaks), n_undist))
    p_at_breaks[0] = p.copy()

    for k in range(len(time_breaks) - 1):
        dt = time_breaks[k + 1] - time_breaks[k]
        lam = lambdas[k]
        # Matrix exponential: p(t + dt) = expm(dt/lam * Q) @ p(t)
        M = expm(dt / lam * Q)
        p = M @ p
        p_at_breaks[k + 1] = p.copy()

    return p_at_breaks


def compute_h_values(time_breaks, p_history, lambdas):
    """Compute h(t) at each time break.

    Parameters
    ----------
    time_breaks : array-like
        Time points.
    p_history : ndarray of shape (K+1, n_undist)
        Lineage count probabilities at each time.
    lambdas : array-like
        Population sizes in each interval.

    Returns
    -------
    h : ndarray
        h[k] = effective coalescence rate at time_breaks[k].
    """
    n_undist = p_history.shape[1]
    h = np.zeros(len(time_breaks))
    j_values = np.arange(1, n_undist + 1)

    for k in range(len(time_breaks)):
        lam = lambdas[min(k, len(lambdas) - 1)]
        expected_j = np.dot(j_values, p_history[k])
        h[k] = expected_j / lam

    return h


def eigendecompose_rate_matrix(n_undist):
    """Compute eigendecomposition of the rate matrix Q.

    Returns
    -------
    eigenvalues : ndarray of shape (n_undist,)
    V : ndarray of shape (n_undist, n_undist)
        Right eigenvectors as columns.
    V_inv : ndarray of shape (n_undist, n_undist)
        Inverse of V.
    """
    Q = build_rate_matrix(n_undist)

    # Eigenvalues are the diagonal entries
    eigenvalues = np.diag(Q)

    # Compute eigenvectors by solving (Q - mu_j I) v_j = 0
    # Since Q is upper triangular, this is a back-substitution
    V = np.zeros((n_undist, n_undist))
    for j in range(n_undist):
        # Start with v[j] = 1, solve upward
        v = np.zeros(n_undist)
        v[j] = 1.0
        for i in range(j - 1, -1, -1):
            # Q[i, i] * v[i] + Q[i, j] * v[j] + ... = eigenvalues[j] * v[i]
            # (eigenvalues[j] - Q[i,i]) * v[i] = sum of Q[i, k] * v[k] for k > i
            rhs = sum(Q[i, k] * v[k] for k in range(i + 1, j + 1))
            denom = eigenvalues[j] - Q[i, i]
            if abs(denom) > 1e-15:
                v[i] = rhs / denom
        V[:, j] = v

    V_inv = np.linalg.inv(V)
    return eigenvalues, V, V_inv


def fast_matrix_exp(eigenvalues, V, V_inv, t, lam):
    """Compute exp(t/lam * Q) using precomputed eigendecomposition.

    Parameters
    ----------
    eigenvalues, V, V_inv : from eigendecompose_rate_matrix
    t : float
        Time interval.
    lam : float
        Relative population size.

    Returns
    -------
    M : ndarray
        The matrix exponential.
    """
    D = np.diag(np.exp(eigenvalues * t / lam))
    return V @ D @ V_inv


# ---------------------------------------------------------------------------
# Continuous HMM (continuous_hmm.rst)
# ---------------------------------------------------------------------------

def emission_probability(genotype, t, theta, allele_count, n_undist):
    """Emission probability for unphased diploid data at the distinguished individual.

    Parameters
    ----------
    genotype : int
        0, 1, or 2 (count of derived alleles at the focal individual).
    t : float
        Coalescence time of the distinguished lineage.
    theta : float
        Scaled mutation rate per bin.
    allele_count : int
        Number of derived alleles observed in the undistinguished panel.
    n_undist : int
        Total number of undistinguished haplotypes.

    Returns
    -------
    float
        P(genotype, allele_count | T = t).
    """
    p_mut = 1 - np.exp(-theta * t)

    # For the distinguished haplotype:
    # If genotype = 0: both alleles ancestral -> (1 - p_mut)
    # If genotype = 1: one derived, one ancestral -> p_mut (approximately)
    # If genotype = 2: both derived -> p_mut (rare case)

    # Simplified emission (full model integrates over allele assignments)
    if genotype == 0:
        return np.exp(-theta * t)
    elif genotype == 1:
        return 1 - np.exp(-theta * t)
    else:
        return (1 - np.exp(-theta * t)) ** 2


def compute_transition_matrix(time_breaks, lambdas, rho, n_undist):
    """Build the SMC++ transition matrix.

    Parameters
    ----------
    time_breaks : array-like
        K+1 time boundaries [t_0, ..., t_K].
    lambdas : array-like
        Relative population sizes in each time interval.
    rho : float
        Scaled recombination rate per bin.
    n_undist : int
        Number of undistinguished lineages.

    Returns
    -------
    P : ndarray of shape (K, K)
        Transition matrix.
    """
    K = len(time_breaks) - 1

    # First solve the ODE to get h(t) at each time break
    Q_rate = build_rate_matrix(n_undist)
    p0 = np.zeros(n_undist)
    p0[-1] = 1.0

    # Compute h(t) at midpoints of each interval
    h = np.zeros(K)
    p_current = p0.copy()
    for k in range(K):
        dt = time_breaks[k + 1] - time_breaks[k]
        lam = lambdas[k]
        # Expected undistinguished lineages at midpoint
        M = expm(dt / (2 * lam) * Q_rate)
        p_mid = M @ p_current
        j_values = np.arange(1, n_undist + 1)
        h[k] = np.dot(j_values, p_mid) / lam
        # Advance to end of interval
        M_full = expm(dt / lam * Q_rate)
        p_current = M_full @ p_current

    # Build transition matrix
    P = np.zeros((K, K))

    for k in range(K):
        t_mid = (time_breaks[k] + time_breaks[k + 1]) / 2

        # Recombination probability: 1 - exp(-rho * t_mid)
        r_k = 1 - np.exp(-rho * t_mid)

        # No recombination: stay in same state
        P[k, k] += 1 - r_k

        # With recombination: transition to new state via h(t)
        for l in range(K):
            dt_l = time_breaks[l + 1] - time_breaks[l]
            # Approximate: probability of landing in interval l
            # proportional to h * exp(-integral of h) * interval width
            q_kl = h[l] * np.exp(-sum(
                h[m] * (time_breaks[m + 1] - time_breaks[m])
                for m in range(l)
            )) * dt_l
            P[k, l] += r_k * q_kl

    # Normalize rows
    P = P / P.sum(axis=1, keepdims=True)
    return P


def composite_log_likelihood(data, time_breaks, lambdas, theta, rho):
    """Compute the composite log-likelihood for SMC++.

    Parameters
    ----------
    data : list of ndarray
        data[i] is the observation sequence for the i-th distinguished sample.
    time_breaks : array-like
        Time interval boundaries.
    lambdas : array-like
        Piecewise-constant population sizes.
    theta : float
        Scaled mutation rate.
    rho : float
        Scaled recombination rate.

    Returns
    -------
    float
        Composite log-likelihood.
    """
    n_samples = len(data)
    n_undist = 2 * n_samples - 1  # Haploid lineages minus the distinguished one

    K = len(time_breaks) - 1
    total_ll = 0.0

    for i in range(n_samples):
        # Build HMM for this distinguished sample
        P = compute_transition_matrix(time_breaks, lambdas, rho, n_undist)

        # Initial distribution (stationary)
        pi = np.ones(K) / K  # Simplified; true stationary from h(t)

        # Forward algorithm
        obs = data[i]
        L = len(obs)
        alpha = np.zeros((L, K))

        # Initialize
        for k in range(K):
            t_mid = (time_breaks[k] + time_breaks[k + 1]) / 2
            alpha[0, k] = pi[k] * emission_probability(obs[0], t_mid, theta, 0, n_undist)

        # Scale for numerical stability
        scale = np.zeros(L)
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]

        # Forward recursion
        for a in range(1, L):
            for l in range(K):
                alpha[a, l] = sum(alpha[a - 1, k] * P[k, l] for k in range(K))
                t_mid = (time_breaks[l] + time_breaks[l + 1]) / 2
                alpha[a, l] *= emission_probability(obs[a], t_mid, theta, 0, n_undist)
            scale[a] = alpha[a].sum()
            if scale[a] > 0:
                alpha[a] /= scale[a]

        total_ll += np.sum(np.log(scale[scale > 0]))

    return total_ll


def fit_smcpp(data, time_breaks, theta, rho, max_iter=100):
    """Fit SMC++ model using L-BFGS-B optimization.

    Parameters
    ----------
    data : list of ndarray
        Observation sequences for each distinguished sample.
    time_breaks : array-like
        Time interval boundaries.
    theta : float
        Scaled mutation rate (fixed).
    rho : float
        Scaled recombination rate (fixed).
    max_iter : int
        Maximum optimization iterations.

    Returns
    -------
    lambdas : ndarray
        Estimated piecewise-constant population sizes.
    """
    K = len(time_breaks) - 1

    # Optimize in log-space for positivity
    def objective(log_lambdas):
        lambdas = np.exp(log_lambdas)
        # Negative log-likelihood (minimize)
        return -composite_log_likelihood(data, time_breaks, lambdas, theta, rho)

    # Initial guess: constant population
    x0 = np.zeros(K)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'disp': False},
    )

    return np.exp(result.x)


# ---------------------------------------------------------------------------
# Population Splits (population_splits.rst)
# ---------------------------------------------------------------------------

def solve_split_ode(n_A, n_B, time_breaks, lambdas_A, lambdas_B,
                    lambdas_anc, t_split):
    """Solve the ODE system for a two-population split model.

    Parameters
    ----------
    n_A : int
        Number of undistinguished haploid lineages from population A.
    n_B : int
        Number of undistinguished haploid lineages from population B.
    time_breaks : array-like
        Time boundaries for piecewise-constant intervals.
    lambdas_A, lambdas_B : array-like
        Population sizes for A and B (pre-split intervals only).
    lambdas_anc : array-like
        Ancestral population sizes (post-split intervals only).
    t_split : float
        Split time in coalescent units.

    Returns
    -------
    h_A : ndarray
        Coalescence rate for a distinguished lineage from pop A.
    h_B : ndarray
        Coalescence rate for a distinguished lineage from pop B.
    """
    Q_A = build_rate_matrix(n_A)
    Q_B = build_rate_matrix(n_B)

    # Initialize: all lineages present
    p_A = np.zeros(n_A)
    p_A[-1] = 1.0
    p_B = np.zeros(n_B)
    p_B[-1] = 1.0

    K = len(time_breaks) - 1
    h_A_values = np.zeros(K)
    h_B_values = np.zeros(K)

    j_A_vals = np.arange(1, n_A + 1)
    j_B_vals = np.arange(1, n_B + 1)

    for k in range(K):
        t_lo = time_breaks[k]
        t_hi = time_breaks[k + 1]
        dt = t_hi - t_lo

        if t_hi <= t_split:
            # Pre-split: populations evolve independently
            lam_A = lambdas_A[k]
            lam_B = lambdas_B[k]

            M_A = expm(dt / lam_A * Q_A)
            p_A = M_A @ p_A

            M_B = expm(dt / lam_B * Q_B)
            p_B = M_B @ p_B

            # h for distinguished from A: only A lineages are partners
            h_A_values[k] = np.dot(j_A_vals, p_A) / lam_A
            # h for distinguished from B: only B lineages are partners
            h_B_values[k] = np.dot(j_B_vals, p_B) / lam_B

        else:
            # Post-split: combined ancestral population
            # Merge lineage counts at the split
            if t_lo < t_split:
                # This interval spans the split -- handle the boundary
                pass  # Simplified: assume breaks align with t_split

            # After merging, use combined rate matrix
            n_anc = n_A + n_B
            Q_anc = build_rate_matrix(n_anc)

            # Combine the lineage count distributions
            # (convolution of A and B distributions)
            # For simplicity, use the expected counts
            lam_anc = lambdas_anc[k - len(lambdas_A)]

            p_anc = np.zeros(n_anc)
            # Initial condition for ancestral: convolution of p_A and p_B
            for ja in range(n_A):
                for jb in range(n_B):
                    j_total = (ja + 1) + (jb + 1)  # 1-indexed counts
                    if j_total <= n_anc:
                        p_anc[j_total - 1] += p_A[ja] * p_B[jb]

            M_anc = expm(dt / lam_anc * Q_anc)
            p_anc = M_anc @ p_anc

            j_anc_vals = np.arange(1, n_anc + 1)
            h_val = np.dot(j_anc_vals, p_anc) / lam_anc
            h_A_values[k] = h_val
            h_B_values[k] = h_val

    return h_A_values, h_B_values


def cross_population_survival(t, t_split, h_anc_func):
    """Survival function for cross-population TMRCA.

    Parameters
    ----------
    t : float
        Time point.
    t_split : float
        Population split time.
    h_anc_func : callable
        Ancestral coalescence rate function.

    Returns
    -------
    float
        P(T_cross > t).
    """
    if t < t_split:
        return 1.0
    else:
        # Numerical integration of h_anc from t_split to t
        integral, _ = quad(h_anc_func, t_split, t)
        return np.exp(-integral)


def fit_split_model(data_A, data_B, time_breaks, theta, rho):
    """Fit SMC++ split model to two-population data.

    Parameters
    ----------
    data_A : list of ndarray
        Observation sequences from population A samples.
    data_B : list of ndarray
        Observation sequences from population B samples.
    time_breaks : array-like
        Time interval boundaries.
    theta, rho : float
        Scaled mutation and recombination rates.

    Returns
    -------
    dict
        Estimated parameters: lambdas_A, lambdas_B, lambdas_anc, t_split.
    """
    K = len(time_breaks) - 1

    def objective(params):
        # Unpack parameters
        log_lambdas_A = params[:K]
        log_lambdas_B = params[K:2*K]
        log_lambdas_anc = params[2*K:3*K]
        log_t_split = params[3*K]

        lambdas_A = np.exp(log_lambdas_A)
        lambdas_B = np.exp(log_lambdas_B)
        lambdas_anc = np.exp(log_lambdas_anc)
        t_split = np.exp(log_t_split)

        # Compute composite log-likelihood for both populations
        ll = 0.0

        # Distinguished from A, undistinguished from A + B
        # (simplified: separate within-population terms)
        for data_i in data_A:
            # HMM forward with population-A-specific transitions
            ll += forward_log_likelihood(data_i, time_breaks, lambdas_A, theta, rho)

        for data_i in data_B:
            ll += forward_log_likelihood(data_i, time_breaks, lambdas_B, theta, rho)

        return -ll  # Minimize negative log-likelihood

    # Initial guess
    x0 = np.zeros(3 * K + 1)
    x0[3*K] = np.log(0.5)  # Initial split time guess

    result = minimize(objective, x0, method='L-BFGS-B')

    return {
        'lambdas_A': np.exp(result.x[:K]),
        'lambdas_B': np.exp(result.x[K:2*K]),
        'lambdas_anc': np.exp(result.x[2*K:3*K]),
        't_split': np.exp(result.x[3*K]),
    }


def forward_log_likelihood(obs, time_breaks, lambdas, theta, rho):
    """HMM forward algorithm log-likelihood (stub for split model)."""
    # Same as in composite_log_likelihood but for a single sequence
    return 0.0  # Placeholder


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the core SMC++ components."""

    print("=" * 60)
    print("SMC++ Mini Implementation Demo")
    print("=" * 60)

    # --- Overview: expected first coalescence ---
    print("\n--- Expected First Coalescence ---")
    # With just 2 lineages (PSMC):
    print(f"n=2:  {expected_first_coalescence(2, 10000):.0f} generations")
    # With 20 lineages (10 diploid samples, as in SMC++):
    print(f"n=20: {expected_first_coalescence(20, 10000):.0f} generations")
    # With 200 lineages (100 diploid samples):
    print(f"n=200: {expected_first_coalescence(200, 10000):.1f} generations")

    # --- Distinguished Lineage: coalescence rates ---
    print("\n--- Coalescence Rates (j=9 undistinguished, lam=1) ---")
    lam = 1.0
    j = 9
    print(f"Undistinguished coalescence rate (j={j}): "
          f"{undistinguished_coalescence_rate(j, lam):.1f}")
    print(f"Distinguished coalescence rate (j={j}):   "
          f"{distinguished_coalescence_rate(j, lam):.1f}")
    print(f"Total rate out of state j={j}:            "
          f"{undistinguished_coalescence_rate(j, lam) + distinguished_coalescence_rate(j, lam):.1f}")

    # --- ODE System: rate matrix ---
    print("\n--- Rate Matrix Q (n_undist=4) ---")
    Q = build_rate_matrix(4)
    print("Rate matrix Q:")
    print(Q)

    # --- ODE System: solve piecewise ---
    print("\n--- ODE Solution (n_undist=9, constant pop) ---")
    n_undist = 9
    time_breaks = np.linspace(0, 5, 51)
    lambdas = np.ones(50)

    p_history = solve_ode_piecewise(n_undist, time_breaks, lambdas)

    print("Time  p_9    p_5    p_1")
    for i in [0, 5, 10, 20, 50]:
        t = time_breaks[i]
        print(f"{t:.1f}   {p_history[i, 8]:.4f}  {p_history[i, 4]:.4f}  "
              f"{p_history[i, 0]:.4f}")

    # --- ODE System: h(t) values ---
    h_values = compute_h_values(time_breaks, p_history, lambdas)

    print("\nEffective coalescence rate h(t):")
    print("Time  h(t)   E[J(t)]")
    for i in [0, 5, 10, 20, 50]:
        t = time_breaks[i]
        ej = np.dot(np.arange(1, 10), p_history[i])
        print(f"{t:.1f}   {h_values[i]:.3f}  {ej:.3f}")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
