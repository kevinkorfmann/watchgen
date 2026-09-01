"""
Minimal implementation of the PSMC (Pairwise Sequentially Markovian Coalescent) algorithm.

PSMC infers **population size history** N(t) from a **single diploid genome**.
It examines the pattern of heterozygous and homozygous sites along the genome
and asks: what demographic history best explains this pattern?

    Input:  One diploid genome -> a sequence of 0s and 1s (hom/het)
    Output: N_hat(t)  (effective population size as a function of time)

The algorithm consists of four main components:

1. **The Continuous-Time Model** -- The transition density q(t|s) that describes
   how coalescence time changes between adjacent positions under variable N(t).

2. **Discretization** -- Converting continuous coalescence time into discrete
   time intervals with a transition matrix p_{kl} suitable for an HMM.

3. **The HMM and EM** -- The complete Hidden Markov Model with forward-backward
   algorithm and Expectation-Maximization for parameter estimation.

4. **Decoding the Clock** -- Posterior decoding, scaling to real units,
   bootstrapping, and interpreting population size history.

Reference: Li, H. & Durbin, R. (2011). Inference of human population history
from individual whole-genome sequences. Nature, 475, 493-496.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize


# ============================================================
# Chapter 1: The Continuous-Time PSMC Model
# ============================================================

def cumulative_hazard(t, lambda_func):
    """Compute Lambda(t) = integral_0^t 1/lambda(u) du.

    This is the cumulative hazard of the coalescent process.
    """
    result, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
    return result


def cumulative_hazard_piecewise(t, t_boundaries, lambdas):
    """Fast Lambda(t) for piecewise-constant lambda.

    No quadrature needed -- just a running sum over intervals.
    """
    Lambda = 0.0
    for k in range(len(lambdas)):
        t_lo = t_boundaries[k]
        t_hi = t_boundaries[k + 1]
        if t <= t_lo:
            break
        dt = min(t, t_hi) - t_lo
        Lambda += dt / lambdas[k]
        if t <= t_hi:
            break
    return Lambda


def coalescent_density(t, lambda_func):
    """Coalescence time density under variable population size.

    f(t) = (1/lambda(t)) * exp(-Lambda(t))
    """
    Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
    return (1.0 / lambda_func(t)) * np.exp(-Lambda_t)


def coalescent_survival(t, lambda_func):
    """Probability that coalescence has NOT occurred by time t.

    S(t) = exp(-Lambda(t))
    """
    Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
    return np.exp(-Lambda_t)


def psmc_transition_density_general(t, s, lambda_func):
    """PSMC transition density q(t|s) under variable population size.

    This is the probability density of the new coalescence time being t,
    given that the old coalescence time was s and a recombination occurred.

    q(t|s) = (1/lambda(t)) * integral_0^min(s,t) (1/s) * exp(-integral_u^t 1/lambda(v) dv) du
    """
    upper = min(s, t)

    def integrand(u):
        integral, _ = quad(lambda v: 1.0 / lambda_func(v), u, t)
        return (1.0 / s) * np.exp(-integral)

    result, _ = quad(integrand, 0, upper)
    return result / lambda_func(t)


def stationary_distribution(t, lambda_func, C_pi=None):
    """Stationary distribution pi(t) of coalescence time.

    pi(t) = t / (C_pi * lambda(t)) * exp(-Lambda(t))
    """
    if C_pi is None:
        C_pi = compute_C_pi(lambda_func)
    Lambda_t, _ = quad(lambda u: 1.0 / lambda_func(u), 0, t)
    return t / (C_pi * lambda_func(t)) * np.exp(-Lambda_t)


def compute_C_pi(lambda_func, t_max=20):
    """Compute the normalization constant C_pi.

    C_pi = integral_0^inf exp(-Lambda(u)) du
    """
    def integrand(u):
        Lambda_u, _ = quad(lambda v: 1.0 / lambda_func(v), 0, u)
        return np.exp(-Lambda_u)
    C_pi, _ = quad(integrand, 0, t_max)
    return C_pi


def full_transition_density(t, s, rho, lambda_func, tol=1e-8):
    """Full transition density p(t|s) including no-recombination.

    p(t|s) = (1 - exp(-rho*s)) * q(t|s) + exp(-rho*s) * delta(t - s)

    Returns
    -------
    continuous : float
        The continuous part of the density at t.
    point_mass : float
        The weight of the point mass at t = s (nonzero only when |t - s| < tol).
    """
    if s < 0 or rho < 0:
        raise ValueError("s and rho must be non-negative")
    if s == 0:
        return 0.0, 1.0 if abs(t) < tol else 0.0
    recomb_prob = -np.expm1(-rho * s)
    if abs(t - s) < tol:
        q_ts = psmc_transition_density_general(t, s, lambda_func)
        return recomb_prob * q_ts, np.exp(-rho * s)
    else:
        return recomb_prob * psmc_transition_density_general(t, s, lambda_func), 0.0


def full_stationary(t, lambda_func, rho, C_pi=None, C_sigma=None):
    """Full stationary distribution sigma(t).

    sigma(t) = pi(t) / (C_sigma * (1 - exp(-rho*t)))
    """
    if t < 0 or rho <= 0:
        raise ValueError("t must be non-negative and rho must be positive")
    if C_pi is None:
        C_pi = compute_C_pi(lambda_func)
    if C_sigma is None:
        C_sigma = compute_C_sigma(lambda_func, rho, C_pi)
    if t == 0:
        return 1.0 / (C_pi * C_sigma * rho * lambda_func(0.0))
    pi_t = stationary_distribution(t, lambda_func, C_pi)
    return pi_t / (C_sigma * -np.expm1(-rho * t))


def compute_C_sigma(lambda_func, rho, C_pi=None, t_max=20):
    """Compute C_sigma = integral pi(t) / (1 - exp(-rho*t)) dt."""
    if rho <= 0 or t_max <= 0:
        raise ValueError("rho and t_max must be positive")
    if C_pi is None:
        C_pi = compute_C_pi(lambda_func)

    def integrand(t):
        if t == 0:
            return 1.0 / (C_pi * rho * lambda_func(0.0))
        return stationary_distribution(t, lambda_func, C_pi) / -np.expm1(-rho * t)

    C_sigma, _ = quad(integrand, 0, t_max)
    return C_sigma


def estimate_theta_initial(seq):
    """Return the original PSMC initialization for ``theta_0``.

    PSMC uses ``-log(1 - fraction_heterozygous)`` as an initial value.
    Observations greater than one denote missing bins and are excluded.
    This is an initialization heuristic, not the exact marginal estimator
    under a random coalescence time.
    """
    seq = np.asarray(seq)
    observed = seq[seq < 2]
    if observed.size == 0:
        raise ValueError("sequence contains no callable bins")
    if np.any((observed != 0) & (observed != 1)):
        raise ValueError("observed bins must be encoded as 0 or 1")
    frac_het = np.mean(observed)
    if frac_het >= 1:
        return np.inf
    return -np.log1p(-frac_het)


# ============================================================
# Chapter 2: Discretizing Time
# ============================================================

def compute_time_intervals(n, t_max, alpha=0.1):
    """Compute PSMC time interval boundaries.

    Uses log-spacing to give fine resolution in the recent past and
    coarse resolution in the distant past.

    Parameters
    ----------
    n : int
        Number of intervals minus 1 (so there are n+1 intervals).
    t_max : float
        Maximum time (in coalescent units).
    alpha : float
        Controls spacing near t=0.

    Returns
    -------
    t : ndarray of shape (n + 2,)
        Boundaries [t_0, t_1, ..., t_n, t_{n+1}].
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if t_max <= 0 or t_max >= 1000 or alpha <= 0:
        raise ValueError("require 0 < t_max < 1000 and alpha > 0")
    beta = np.log1p(t_max / alpha) / n
    t = np.zeros(n + 2)
    for k in range(n):
        t[k] = alpha * (np.exp(beta * k) - 1)
    t[n] = t_max
    t[n + 1] = 1000.0
    return t


def compute_helpers(n, t, lambdas):
    """Compute the helper quantities for the discrete PSMC.

    Returns
    -------
    tau : ndarray -- interval widths
    alpha : ndarray -- survival factors
    beta : ndarray -- re-coalescence sums
    q_aux : ndarray -- auxiliary quantity for transition matrix
    C_pi : float -- normalization constant
    """
    t = np.asarray(t, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if t.shape != (n + 2,) or lambdas.shape != (n + 1,):
        raise ValueError("t and lambdas have incompatible lengths")
    if np.any(np.diff(t) <= 0) or np.any(lambdas <= 0):
        raise ValueError("time boundaries and lambdas must be strictly positive")

    tau = np.zeros(n + 1)
    alpha = np.zeros(n + 2)
    beta = np.zeros(n + 1)
    q_aux = np.zeros(n)

    for k in range(n + 1):
        tau[k] = t[k + 1] - t[k]

    alpha[0] = 1.0
    for k in range(1, n + 1):
        alpha[k] = alpha[k - 1] * np.exp(-tau[k - 1] / lambdas[k - 1])
    alpha[n + 1] = 0.0

    beta[0] = 0.0
    for k in range(1, n + 1):
        beta[k] = beta[k - 1] + lambdas[k - 1] * (1.0 / alpha[k] - 1.0 / alpha[k - 1])

    for k in range(n):
        ak1 = alpha[k] - alpha[k + 1]
        q_aux[k] = ak1 * (beta[k] - lambdas[k] / alpha[k]) + tau[k]

    C_pi = 0.0
    for k in range(n + 1):
        C_pi += lambdas[k] * (alpha[k] - alpha[k + 1])

    return tau, alpha, beta, q_aux, C_pi


def compute_stationary(n, tau, alpha, lambdas, C_pi, rho):
    """Compute discrete stationary distributions pi_k and sigma_k."""
    pi_k = np.zeros(n + 1)
    sum_tau = 0.0

    for k in range(n + 1):
        ak1 = alpha[k] - alpha[k + 1]
        pi_k[k] = (ak1 * (sum_tau + lambdas[k]) - alpha[k + 1] * tau[k]) / C_pi
        sum_tau += tau[k]

    C_sigma = 1.0 / (C_pi * rho) + 0.5

    sigma_k = np.zeros(n + 1)
    for k in range(n + 1):
        ak1 = alpha[k] - alpha[k + 1]
        sigma_k[k] = (ak1 / (C_pi * rho) + pi_k[k] / 2.0) / C_sigma

    return pi_k, sigma_k, C_sigma


def compute_transition_matrix(n, tau, alpha, beta, q_aux, lambdas,
                               C_pi, C_sigma, pi_k, sigma_k):
    """Compute the full discrete PSMC transition matrix."""
    N = n + 1
    q = np.zeros((N, N))

    for k in range(N):
        ak1 = alpha[k] - alpha[k + 1]
        cpik = ak1 * (sum(tau[:k]) + lambdas[k]) - alpha[k + 1] * tau[k]

        if cpik <= 0 or not np.isfinite(cpik):
            raise FloatingPointError("invalid event-chain interval mass")

        for l in range(k):
            q[k, l] = ak1 / cpik * q_aux[l]

        q[k, k] = (ak1 * ak1 * (beta[k] - lambdas[k] / alpha[k])
                    + 2 * lambdas[k] * ak1
                    - 2 * alpha[k + 1] * tau[k]) / cpik

        if k < n:
            for l in range(k + 1, N):
                q[k, l] = (alpha[l] - alpha[l + 1]) / cpik * q_aux[k]

    p = np.zeros((N, N))
    for k in range(N):
        recomb_prob = pi_k[k] / (C_sigma * sigma_k[k]) if sigma_k[k] > 0 else 0
        for l in range(N):
            p[k, l] = recomb_prob * q[k, l]
            if k == l:
                p[k, l] += (1.0 - recomb_prob)

    return p, q


def compute_avg_times(n, tau, alpha, lambdas, pi_k, sigma_k, C_sigma, rho):
    """Compute the effective coalescence time for each interval."""
    avg_t = np.zeros(n + 1)
    sum_tau = 0.0

    for k in range(n + 1):
        ak1 = alpha[k] - alpha[k + 1]
        recomb_prob = pi_k[k] / (C_sigma * sigma_k[k]) if sigma_k[k] > 0 else 0

        if recomb_prob < 1.0:
            avg_t[k] = -np.log(1.0 - recomb_prob) / rho
        else:
            lak = lambdas[k]
            avg_t[k] = sum_tau + (lak - tau[k] * alpha[k + 1] / ak1
                                   if ak1 > 0 else tau[k] / 2)

        if np.isnan(avg_t[k]) or avg_t[k] < sum_tau or avg_t[k] > sum_tau + tau[k]:
            lak = lambdas[k]
            ak1 = alpha[k] - alpha[k + 1]
            avg_t[k] = sum_tau + (lak - tau[k] * alpha[k + 1] / ak1
                                   if ak1 > 0 else tau[k] / 2)

        sum_tau += tau[k]

    return avg_t


def parse_pattern(pattern):
    """Parse a PSMC pattern string into a parameter map.

    Parameters
    ----------
    pattern : str
        Pattern like "4+25*2+4+6".

    Returns
    -------
    par_map : list of int
        par_map[k] = index of the free parameter for interval k.
    n_free : int
        Number of free parameters.
    n_intervals : int
        Total number of atomic intervals (= n + 1).
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    par_map = []
    free_idx = 0

    for part in pattern.split('+'):
        if '*' in part:
            count, width = part.split('*')
            count, width = int(count), int(width)
            if count <= 0 or width <= 0:
                raise ValueError("pattern counts and widths must be positive")
            for _ in range(count):
                for _ in range(width):
                    par_map.append(free_idx)
                free_idx += 1
        else:
            width = int(part)
            if width <= 0:
                raise ValueError("pattern widths must be positive")
            for _ in range(width):
                par_map.append(free_idx)
            free_idx += 1

    return par_map, free_idx, len(par_map)


def build_psmc_hmm(n, t_max, theta, rho, lambdas, par_map=None, alpha_param=0.1):
    """Build the complete discrete PSMC HMM parameters.

    Parameters
    ----------
    n : int
        Number of time intervals minus 1.
    t_max : float
        Maximum coalescent time.
    theta : float
        Mutation rate per bin.
    rho : float
        Recombination rate per bin.
    lambdas : ndarray
        Relative population sizes.
    par_map : list, optional
        Parameter grouping map.
    alpha_param : float
        Spacing parameter for time intervals.

    Returns
    -------
    transitions : ndarray of shape (n+1, n+1)
    emissions : ndarray of shape (2, n+1)
    initial : ndarray of shape (n+1,)
    """
    if theta < 0 or rho <= 0:
        raise ValueError("theta must be non-negative and rho must be positive")
    lambdas = np.asarray(lambdas, dtype=float)
    if np.any(~np.isfinite(lambdas)) or np.any(lambdas <= 0):
        raise ValueError("lambdas must be finite and positive")
    if par_map is not None:
        if len(par_map) != n + 1 or min(par_map) < 0 or max(par_map) >= len(lambdas):
            raise ValueError("par_map is incompatible with n and lambdas")
        full_lambdas = np.array([lambdas[par_map[k]] for k in range(n + 1)])
    else:
        if len(lambdas) != n + 1:
            raise ValueError("lambdas must contain n + 1 values")
        full_lambdas = lambdas

    t = compute_time_intervals(n, t_max, alpha_param)
    tau, alpha_arr, beta_arr, q_aux, C_pi = compute_helpers(n, t, full_lambdas)
    pi_k, sigma_k, C_sigma = compute_stationary(
        n, tau, alpha_arr, full_lambdas, C_pi, rho)
    transitions, _ = compute_transition_matrix(
        n, tau, alpha_arr, beta_arr, q_aux, full_lambdas,
        C_pi, C_sigma, pi_k, sigma_k)
    avg_t = compute_avg_times(
        n, tau, alpha_arr, full_lambdas, pi_k, sigma_k, C_sigma, rho)

    emissions = np.zeros((2, n + 1))
    for k in range(n + 1):
        emissions[0, k] = np.exp(-theta * avg_t[k])
        emissions[1, k] = 1 - emissions[0, k]

    initial = sigma_k.copy()

    return transitions, emissions, initial


# ============================================================
# Chapter 3: The PSMC HMM and EM Algorithm
# ============================================================

class PSMC_HMM:
    """Complete PSMC Hidden Markov Model.

    Bundles together all the HMM parameters (transitions, emissions,
    initial distribution) and provides methods for computing likelihoods
    and running the forward-backward algorithm.
    """

    def __init__(self, n, theta, rho, lambdas, t_max=15.0, alpha_param=0.1):
        self.n = n
        self.N = n + 1
        self.theta = theta
        self.rho = rho
        self.lambdas = lambdas.copy()
        self.t_max = t_max
        self.alpha_param = alpha_param

        self.transitions, self.emissions, self.initial = build_psmc_hmm(
            n, t_max, theta, rho, lambdas, alpha_param=alpha_param)

    def log_likelihood(self, seq):
        """Compute log-likelihood of an observation sequence."""
        _, ll = self.forward_scaled(seq)
        return ll

    def forward_scaled(self, seq):
        """Scaled forward algorithm.

        Returns
        -------
        alpha_hat : ndarray of shape (L, N)
            Scaled forward probabilities.
        log_likelihood : float
        """
        L = len(seq)
        N = self.N
        alpha_hat = np.zeros((L, N))
        log_likelihood = 0.0

        for k in range(N):
            obs = seq[0]
            if obs >= 2:
                e = 1.0
            else:
                e = self.emissions[obs, k]
            alpha_hat[0, k] = self.initial[k] * e

        c0 = alpha_hat[0].sum()
        if c0 > 0:
            alpha_hat[0] /= c0
            log_likelihood += np.log(c0)

        for a in range(1, L):
            obs = seq[a]
            for l in range(N):
                if obs >= 2:
                    e = 1.0
                else:
                    e = self.emissions[obs, l]
                s = 0.0
                for k in range(N):
                    s += alpha_hat[a - 1, k] * self.transitions[k, l]
                alpha_hat[a, l] = s * e

            c = alpha_hat[a].sum()
            if c > 0:
                alpha_hat[a] /= c
                log_likelihood += np.log(c)

        return alpha_hat, log_likelihood

    def backward_scaled(self, seq, alpha_hat):
        """Scaled backward algorithm.

        Returns
        -------
        beta_hat : ndarray of shape (L, N)
        """
        L = len(seq)
        N = self.N
        beta_hat = np.zeros((L, N))
        beta_hat[L - 1, :] = 1.0

        for a in range(L - 2, -1, -1):
            obs_next = seq[a + 1]
            for k in range(N):
                s = 0.0
                for l in range(N):
                    if obs_next >= 2:
                        e = 1.0
                    else:
                        e = self.emissions[obs_next, l]
                    s += self.transitions[k, l] * e * beta_hat[a + 1, l]
                beta_hat[a, k] = s

            c = beta_hat[a].sum()
            if c > 0:
                beta_hat[a] /= c

        return beta_hat


def compute_expected_counts(hmm, seq):
    """Compute expected counts for the E-step of EM.

    Returns
    -------
    gamma_sum : ndarray of shape (N,)
    xi_sum : ndarray of shape (N, N)
    emission_counts : ndarray of shape (2, N)
    log_likelihood : float
    """
    L = len(seq)
    N = hmm.N

    alpha_hat, ll = hmm.forward_scaled(seq)
    beta_hat = hmm.backward_scaled(seq, alpha_hat)

    gamma_sum = np.zeros(N)
    emission_counts = np.zeros((2, N))

    for pos in range(L):
        gamma = alpha_hat[pos] * beta_hat[pos]
        total = gamma.sum()
        if total > 0:
            gamma /= total

        gamma_sum += gamma

        obs = seq[pos]
        if obs < 2:
            emission_counts[obs] += gamma

    xi_sum = np.zeros((N, N))
    for pos in range(L - 1):
        obs_next = seq[pos + 1]
        # Compute raw xi values and per-position normalization factor.
        # With independently-scaled forward/backward, each position has
        # a different implicit scale, so we must normalize per-position
        # before accumulating (global normalization would mix scales).
        pos_total = 0.0
        for k in range(N):
            for l in range(N):
                if obs_next >= 2:
                    e = 1.0
                else:
                    e = hmm.emissions[obs_next, l]
                pos_total += (alpha_hat[pos, k]
                              * hmm.transitions[k, l]
                              * e * beta_hat[pos + 1, l])
        if pos_total > 0:
            for k in range(N):
                for l in range(N):
                    if obs_next >= 2:
                        e = 1.0
                    else:
                        e = hmm.emissions[obs_next, l]
                    xi_sum[k, l] += (alpha_hat[pos, k]
                                     * hmm.transitions[k, l]
                                     * e * beta_hat[pos + 1, l]) / pos_total

    return gamma_sum, xi_sum, emission_counts, ll


def psmc_em_step(hmm, seq, par_map=None):
    """One EM iteration for PSMC.

    Returns
    -------
    new_hmm : PSMC_HMM
    log_likelihood : float
    """
    N = hmm.N
    n = hmm.n

    gamma_sum, xi_sum, emission_counts, ll = compute_expected_counts(hmm, seq)

    if par_map is None:
        par_map = list(range(N))
    n_free = max(par_map) + 1
    n_params = n_free + 3

    params0 = np.zeros(n_params)
    params0[0] = hmm.theta
    params0[1] = hmm.rho
    params0[2] = hmm.t_max
    for k in range(n_free):
        idx = par_map.index(k)
        params0[3 + k] = hmm.lambdas[idx]

    def neg_Q(params):
        theta = abs(params[0])
        rho = abs(params[1])
        t_max = abs(params[2])
        free_lambdas = np.abs(params[3:])

        full_lambdas = np.array([free_lambdas[par_map[k]] for k in range(N)])

        try:
            transitions, emissions, initial = build_psmc_hmm(
                n, t_max, theta, rho, full_lambdas, alpha_param=hmm.alpha_param)
        except (ValueError, RuntimeWarning):
            return 1e30

        Q = 0.0
        # Match the original PSMC objective (khmm.c:hmm_Q), which optimizes
        # emission and transition terms only. Average state occupancy is not
        # a valid substitute for the posterior initial-state count.

        for k in range(N):
            for l in range(N):
                if transitions[k, l] > 0 and xi_sum[k, l] > 0:
                    Q += xi_sum[k, l] * np.log(transitions[k, l] + 1e-300)

        for b in range(2):
            for k in range(N):
                if emissions[b, k] > 0 and emission_counts[b, k] > 0:
                    Q += emission_counts[b, k] * np.log(emissions[b, k] + 1e-300)

        return -Q

    result = minimize(neg_Q, params0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-6})

    new_params = np.abs(result.x)
    new_theta = new_params[0]
    new_rho = new_params[1]
    new_t_max = new_params[2]
    new_free_lambdas = new_params[3:]
    new_full_lambdas = np.array([new_free_lambdas[par_map[k]] for k in range(N)])

    new_hmm = PSMC_HMM(n, new_theta, new_rho, new_full_lambdas,
                         new_t_max, hmm.alpha_param)

    return new_hmm, ll


def psmc_inference(seq, n=63, t_max=15.0, theta_rho_ratio=5.0,
                    pattern="4+25*2+4+6", n_iters=25, alpha_param=0.1):
    """Run the full PSMC inference.

    Parameters
    ----------
    seq : ndarray of shape (L,), dtype=int
        Observation sequence (0/1/2+).
    n : int
        Number of atomic time intervals - 1.
    t_max : float
    theta_rho_ratio : float
    pattern : str
    n_iters : int
    alpha_param : float

    Returns
    -------
    results : list of dict
    """
    L = len(seq)
    N = n + 1

    par_map, n_free, n_intervals = parse_pattern(pattern)
    assert n_intervals == N, f"Pattern gives {n_intervals} intervals, need {N}"

    theta = estimate_theta_initial(seq)
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError(
            "PSMC inference requires at least one homozygous and one "
            "heterozygous callable bin")
    rho = theta / theta_rho_ratio

    free_lambdas = np.ones(n_free)
    full_lambdas = np.array([free_lambdas[par_map[k]] for k in range(N)])

    hmm = PSMC_HMM(n, theta, rho, full_lambdas, t_max, alpha_param)

    results = []
    print(f"PSMC inference: {L} bins, {N} intervals, {n_free} free params")
    print(f"Initial theta={theta:.6f}, rho={rho:.6f}")

    for iteration in range(n_iters):
        gamma_sum, xi_sum, emission_counts, ll = compute_expected_counts(hmm, seq)
        results.append({
            'iteration': iteration,
            'log_likelihood': ll,
            'theta': hmm.theta,
            'rho': hmm.rho,
            'lambdas': hmm.lambdas.copy(),
        })
        print(f"  Iteration {iteration}: LL = {ll:.2f}, "
              f"theta = {hmm.theta:.6f}, rho = {hmm.rho:.6f}")
        hmm, _ = psmc_em_step(hmm, seq, par_map)

    return results


def posterior_decoding(hmm, seq):
    """Compute the posterior state probabilities at each position.

    Returns
    -------
    posterior : ndarray of shape (L, N)
    map_states : ndarray of shape (L,)
    """
    L = len(seq)
    N = hmm.N

    alpha_hat, _ = hmm.forward_scaled(seq)
    beta_hat = hmm.backward_scaled(seq, alpha_hat)

    posterior = np.zeros((L, N))
    for pos in range(L):
        gamma = alpha_hat[pos] * beta_hat[pos]
        total = gamma.sum()
        if total > 0:
            posterior[pos] = gamma / total
        else:
            posterior[pos] = 1.0 / N

    map_states = np.argmax(posterior, axis=1)
    return posterior, map_states


# ============================================================
# Chapter 4: Decoding the Clock
# ============================================================

def scale_psmc_output(theta_0, lambdas, t_boundaries,
                       mu=1.25e-8, s=100, generation_time=25):
    """Scale PSMC output to real units.

    Parameters
    ----------
    theta_0 : float
        Estimated theta_0 from EM.
    lambdas : ndarray
        Estimated relative population sizes.
    t_boundaries : ndarray
        Time interval boundaries in coalescent units.
    mu : float
        Per-generation, per-base-pair mutation rate.
    s : int
        Bin size in base pairs.
    generation_time : float
        Years per generation.

    Returns
    -------
    N_0 : float
    times_gen : ndarray
    times_years : ndarray
    pop_sizes : ndarray
    """
    N_0 = theta_0 / (4 * mu * s)
    times_gen = 2 * N_0 * t_boundaries
    times_years = times_gen * generation_time
    pop_sizes = N_0 * lambdas
    return N_0, times_gen, times_years, pop_sizes


def scale_mutation_free(theta_0, lambdas, t_boundaries, s=100):
    """Scale without assuming a mutation rate.

    Returns divergence (x-axis) and scaled mutation rate (y-axis).
    """
    divergence = t_boundaries * theta_0 / s
    scaled_theta = lambdas * theta_0 / s
    return divergence, scaled_theta


def correct_for_coverage(theta_0, fnr):
    """Correct the bin-level theta initialization for heterozygote FNR.

    Converts ``theta`` back to a heterozygous-bin probability, corrects that
    probability, and then applies the inverse transformation. Dividing theta
    directly by ``1 - fnr`` is only the small-theta approximation.
    """
    if theta_0 < 0 or not 0 <= fnr < 1:
        raise ValueError("theta_0 must be non-negative and 0 <= fnr < 1")
    p_observed = -np.expm1(-theta_0)
    p_true = p_observed / (1 - fnr)
    if p_true >= 1:
        raise ValueError("false-negative correction implies probability >= 1")
    return -np.log1p(-p_true)


def split_sequence(seq, segment_length=50000):
    """Split a sequence into segments for bootstrapping."""
    segments = []
    for start in range(0, len(seq) - segment_length + 1, segment_length):
        segments.append(seq[start:start + segment_length])
    return segments


def bootstrap_resample(segments, total_length):
    """Create a bootstrap replicate by resampling segments."""
    n_segments = len(segments)
    n_needed = total_length // len(segments[0]) + 1
    indices = np.random.choice(n_segments, size=n_needed, replace=True)
    replicate = np.concatenate([segments[i] for i in indices])
    return replicate[:total_length]


def check_overfitting(pi_k, C_sigma, threshold=20):
    """Apply the interval-support diagnostic from the PSMC derivation.

    Li's PSMC notes use ``C_sigma * pi_k``. Here ``pi_k`` is the stationary
    distribution of the embedded recombination-event chain, not the full-chain
    occupancy ``sigma_k``.
    """
    pi_k = np.asarray(pi_k, dtype=float)
    if pi_k.ndim != 1 or np.any(pi_k < 0):
        raise ValueError("pi_k must be a one-dimensional probability vector")
    if C_sigma <= 0 or threshold < 0:
        raise ValueError("C_sigma must be positive and threshold non-negative")
    expected_segments = C_sigma * pi_k
    warnings = []
    for k, exp_seg in enumerate(expected_segments):
        if exp_seg < threshold:
            warnings.append(k)
    return warnings, expected_segments


def plot_psmc_history(theta_0, lambdas, t_boundaries,
                       mu=1.25e-8, s=100, generation_time=25):
    """Generate data for a PSMC plot (step function).

    Returns
    -------
    x : list of float
        Time points (years ago).
    y : list of float
        Population sizes.
    """
    N_0, t_gen, t_years, N_t = scale_psmc_output(
        theta_0, lambdas, t_boundaries, mu, s, generation_time)
    x = []
    y = []
    for k in range(len(lambdas)):
        x.append(t_years[k])
        y.append(N_t[k])
        x.append(t_years[k + 1])
        y.append(N_t[k])
    return x, y


def goodness_of_fit_sigma(hmm, seq):
    """Compute goodness-of-fit by comparing sigma_k to posterior.

    Returns
    -------
    G_sigma : float
        KL divergence (smaller is better, 0 is perfect).
    sigma_model : ndarray
    sigma_data : ndarray
    """
    N = hmm.N
    sigma_model = hmm.initial.copy()

    alpha_hat, _ = hmm.forward_scaled(seq)
    beta_hat = hmm.backward_scaled(seq, alpha_hat)

    sigma_data = np.zeros(N)
    for pos in range(len(seq)):
        gamma = alpha_hat[pos] * beta_hat[pos]
        total = gamma.sum()
        if total <= 0:
            raise FloatingPointError("posterior state probabilities vanished")
        sigma_data += gamma / total
    sigma_data /= len(seq)

    G_sigma = 0.0
    for k in range(N):
        if sigma_model[k] > 0 and sigma_data[k] > 0:
            G_sigma += sigma_model[k] * np.log(sigma_model[k] / sigma_data[k])

    return G_sigma, sigma_model, sigma_data


# ============================================================
# Overview: Simulation
# ============================================================

def simulate_psmc_input(L, theta, rho, lambda_func, n_bins=None):
    """Simulate a PSMC input sequence.

    Parameters
    ----------
    L : int
        Number of bins.
    theta : float
        Mutation rate per bin.
    rho : float
        Recombination rate per bin.
    lambda_func : callable
        lambda_func(t) returns relative population size at time t.
    n_bins : ignored
        Kept for API compatibility.

    Returns
    -------
    seq : ndarray of shape (L,), dtype=int
        Binary sequence: 0 = homozygous, 1 = heterozygous.
    coal_times : ndarray of shape (L,)
        True coalescence times.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if theta < 0 or rho < 0:
        raise ValueError("theta and rho must be non-negative")

    seq = np.zeros(L, dtype=int)
    coal_times = np.zeros(L)

    def _sample_coalescence_time(lambda_func, t_start=0.0):
        """Sample coalescence time under variable population size."""
        target = np.random.exponential(1.0)

        def integrated_hazard(t):
            value, _ = quad(lambda u: 1.0 / lambda_func(u), t_start, t)
            return value

        lam_start = float(lambda_func(t_start))
        if not np.isfinite(lam_start) or lam_start <= 0:
            raise ValueError("lambda_func must return finite positive values")

        upper = t_start + max(lam_start * target, 1e-12)
        for _ in range(80):
            hazard = integrated_hazard(upper)
            if not np.isfinite(hazard):
                raise ValueError("lambda_func produced a non-finite hazard")
            if hazard >= target:
                return brentq(
                    lambda t: integrated_hazard(t) - target,
                    t_start,
                    upper,
                )
            upper = t_start + 2.0 * (upper - t_start)
        raise ValueError("cumulative coalescence hazard did not reach sample target")

    t = _sample_coalescence_time(lambda_func)
    coal_times[0] = t

    for a in range(L):
        p_het = 1 - np.exp(-theta * t)
        seq[a] = np.random.binomial(1, p_het)
        coal_times[a] = t

        if a < L - 1:
            if np.random.random() < 1 - np.exp(-rho * t):
                u = np.random.uniform(0, t)
                t = _sample_coalescence_time(lambda_func, t_start=u)

    return seq, coal_times


# ============================================================
# Demo
# ============================================================

def demo():
    """Demonstrate the PSMC algorithm components."""

    print("=" * 60)
    print("PSMC (Pairwise Sequentially Markovian Coalescent) Demo")
    print("=" * 60)

    # --- Continuous model ---
    print("\n--- Continuous Model ---")
    t_test = 2.5
    print(f"Lambda({t_test}) with lambda=1: "
          f"{cumulative_hazard(t_test, lambda u: 1.0):.6f} (expected: {t_test})")
    print(f"Lambda({t_test}) with lambda=2: "
          f"{cumulative_hazard(t_test, lambda u: 2.0):.6f} (expected: {t_test / 2})")

    # Coalescent density
    t_vals = np.linspace(0.01, 5, 50)
    for lam_val, label in [(1.0, "constant N"), (2.0, "2x larger N")]:
        densities = [coalescent_density(t, lambda u, lv=lam_val: lv) for t in t_vals]
        mean_t = sum(t * d for t, d in zip(t_vals, densities)) * (t_vals[1] - t_vals[0])
        print(f"{label}: mean coalescence time ~ {mean_t:.2f} "
              f"(expected: {lam_val:.2f})")

    # Stationary distribution
    C_pi_const = compute_C_pi(lambda t: 1.0)
    print(f"\nC_pi (constant pop): {C_pi_const:.6f} (expected: 1.0)")

    mean_T, _ = quad(
        lambda t: t * stationary_distribution(t, lambda u: 1.0, C_pi_const),
        0, 20
    )
    print(f"Mean coalescence time (constant pop): {mean_T:.4f} (expected: 2.0)")

    # --- Simulation ---
    print("\n--- Simulation ---")
    np.random.seed(42)
    seq, times = simulate_psmc_input(10000, theta=0.001, rho=0.0005,
                                      lambda_func=lambda t: 1.0)
    print(f"Sequence length: {len(seq)}")
    print(f"Fraction heterozygous: {seq.mean():.4f}")
    print(f"Mean coalescence time: {times.mean():.4f}")

    # --- Theta estimation ---
    theta_hat = estimate_theta_initial(seq)
    print(f"\nTrue theta: 0.001, Estimated: {theta_hat:.6f}")

    # --- Discretization ---
    print("\n--- Discretization ---")
    n = 10
    t = compute_time_intervals(n, t_max=15.0, alpha=0.1)
    print(f"Time interval boundaries (n={n}):")
    for k in range(min(5, n + 2)):
        print(f"  t[{k}] = {t[k]:.6f}")
    print(f"  ...")
    print(f"  t[{n + 1}] = {t[n + 1]:.1f}")

    # --- Build HMM ---
    print("\n--- Build HMM ---")
    theta = 0.001
    rho = theta / 5
    lambdas = np.ones(n + 1)

    transitions, emissions, initial = build_psmc_hmm(
        n, t_max=15.0, theta=theta, rho=rho, lambdas=lambdas)

    print(f"Transition matrix shape: {transitions.shape}")
    print(f"Row sums: {transitions.sum(axis=1)[:3]}...")
    print(f"Emission matrix shape: {emissions.shape}")
    print(f"Initial distribution sums to: {initial.sum():.6f}")

    # --- Pattern parsing ---
    print("\n--- Pattern Parsing ---")
    par_map, n_free, n_intervals = parse_pattern("4+25*2+4+6")
    print(f"Pattern: 4+25*2+4+6")
    print(f"Total atomic intervals: {n_intervals}")
    print(f"Free parameters: {n_free}")

    # --- Scaling ---
    print("\n--- Scaling ---")
    theta_0 = 0.00069
    t_boundaries = np.array([0, 0.05, 0.12, 0.22, 0.37, 0.6,
                              0.95, 1.5, 2.5, 4.0, 8.0, 1000])
    lambdas_ex = np.array([2.0, 1.8, 1.5, 1.0, 0.5, 0.3,
                            0.5, 1.0, 1.5, 2.0, 2.5])

    N_0, t_gen, t_years, N_t = scale_psmc_output(
        theta_0, lambdas_ex, t_boundaries)
    print(f"N_0 = {N_0:.0f}")
    print(f"Time (years ago)    Pop. size")
    print("-" * 40)
    for k in range(min(5, len(lambdas_ex))):
        print(f"  {t_years[k]:>12,.0f}      {N_t[k]:>10,.0f}")
    print("  ...")

    print("\nDemo complete.")


if __name__ == "__main__":
    demo()
