"""
Mini-implementation of the CLUES algorithm for selection inference.

CLUES (Coalescent Likelihood Under Effects of Selection) is a full-likelihood
method for estimating the selection coefficient s acting on a biallelic SNP,
using modern and ancient DNA. It answers the most direct question in molecular
evolution: is this allele being favored or disfavored by natural selection,
and by how much?

The algorithm models the allele frequency trajectory as a Hidden Markov Model
whose hidden states are discretized allele frequencies and whose emissions come
from the coalescent structure of the gene tree. The four main components are:

1. The Wright-Fisher HMM: A discretized Wright-Fisher diffusion with selection,
   modeling how the allele frequency changes from one generation to the next.

2. Emission Probabilities: Two types of evidence constrain the allele frequency
   at each time point -- coalescent events in the gene tree and ancient genotype
   likelihoods.

3. Importance Sampling: CLUES averages over multiple sampled genealogies using
   importance weights, properly accounting for genealogical uncertainty.

4. Inference and Testing: Maximum likelihood estimation of s via Brent's method,
   a likelihood ratio test calibrated against chi-squared, multi-epoch selection
   estimation, and posterior trajectory reconstruction.

References
----------
Stern, Wilton, Nielsen (2019). An approximate full-likelihood method for
inferring selection and allele frequency trajectories from DNA sequence data.
PLoS Genetics.

Stern, Speidel, Zaitlen, Nielsen (2021). Disentangling selection on genetically
correlated polygenic traits via whole-genome genealogies. American Journal of
Human Genetics.
"""

import numpy as np
from scipy.stats import beta as beta_dist, norm, chi2
from scipy.optimize import minimize_scalar, minimize


# ============================================================================
# Wright-Fisher HMM (from wright_fisher_hmm.rst)
# ============================================================================

def backward_mean(x, s, h=0.5):
    """Expected allele frequency one generation further into the past.

    Given current derived allele frequency x, selection coefficient s,
    and dominance coefficient h, compute the mean of the backward
    Wright-Fisher transition.

    For additive selection (h=0.5), this simplifies to:
        mu = x + s * x * (1-x) / (2 * (1 + s*x))

    Parameters
    ----------
    x : float
        Current derived allele frequency (0 < x < 1).
    s : float
        Selection coefficient. Positive = derived allele favored.
    h : float
        Dominance coefficient. Default 0.5 (additive).

    Returns
    -------
    mu : float
        Expected frequency one generation into the past.
    """
    numerator = s * (-1 + x) * x * (-x + h * (-1 + 2 * x))
    denominator = -1 + s * (2 * h * (-1 + x) - x) * x
    return x + numerator / denominator


def backward_std(x, N):
    """Standard deviation of allele frequency change per generation.

    Parameters
    ----------
    x : float
        Current derived allele frequency.
    N : float
        Haploid effective population size (= 2 * diploid N_e).

    Returns
    -------
    sigma : float
        Standard deviation of the frequency change.
    """
    return np.sqrt(x * (1.0 - x) / N)


def build_frequency_bins(K=450):
    """Construct the K allele frequency bins using Beta(1/2, 1/2) quantiles.

    The bins are denser near 0 and 1, where the Wright-Fisher dynamics
    are slowest (frequency changes are small when the allele is very
    rare or very common).

    Parameters
    ----------
    K : int
        Number of frequency bins. Default 450 (as in CLUES2).

    Returns
    -------
    freqs : ndarray of shape (K,)
        Frequency bin centers. freqs[0] = 0 (loss), freqs[-1] = 1 (fixation).
    logfreqs : ndarray of shape (K,)
        log(freqs), with a small epsilon to avoid -inf at boundaries.
    log1minusfreqs : ndarray of shape (K,)
        log(1 - freqs), with a small epsilon at boundaries.
    """
    # Step 1: equally-spaced points in [0, 1]
    u = np.linspace(0.0, 1.0, K)

    # Step 2: map through Beta(1/2, 1/2) quantile function
    freqs = beta_dist.ppf(u, 0.5, 0.5)

    # Step 3: set boundary values for log computations
    eps = 1e-12
    freqs[0] = eps       # temporarily, for log computation
    freqs[-1] = 1 - eps  # temporarily, for log computation

    logfreqs = np.log(freqs)
    log1minusfreqs = np.log(1.0 - freqs)

    # Now set the actual boundary values
    freqs[0] = 0.0
    freqs[-1] = 1.0

    return freqs, logfreqs, log1minusfreqs


def build_transition_matrix(freqs, N, s, h=0.5):
    """Build the K x K log-transition matrix for the Wright-Fisher HMM.

    Each entry P[i, j] is the log-probability of transitioning from
    frequency bin i to frequency bin j in one generation (backward in time).

    Parameters
    ----------
    freqs : ndarray of shape (K,)
        Frequency bin centers (from build_frequency_bins).
    N : float
        Haploid effective population size.
    s : float
        Selection coefficient.
    h : float
        Dominance coefficient (default 0.5 = additive).

    Returns
    -------
    logP : ndarray of shape (K, K)
        Log-transition matrix.
    """
    K = len(freqs)
    logP = np.full((K, K), -np.inf)

    # Midpoints between consecutive bins (used for CDF integration)
    midpoints = (freqs[1:] + freqs[:-1]) / 2.0

    # Absorbing states: loss and fixation
    logP[0, 0] = 0.0       # log(1) = 0
    logP[K - 1, K - 1] = 0.0

    for i in range(1, K - 1):
        x = freqs[i]
        mu = backward_mean(x, s, h)
        sigma = backward_std(x, N)

        if sigma < 1e-15:
            # Degenerate case: no drift, all mass at mu
            closest = np.argmin(np.abs(freqs - mu))
            logP[i, closest] = 0.0
            continue

        # Only compute within +/- 3.3 sigma (99.9% of probability mass)
        lower_freq = mu - 3.3 * sigma
        upper_freq = mu + 3.3 * sigma
        j_lower = max(0, np.searchsorted(freqs, lower_freq) - 1)
        j_upper = min(K, np.searchsorted(freqs, upper_freq) + 1)

        # Compute probability mass in each bin using normal CDF
        row = np.zeros(K)
        for j in range(j_lower, j_upper):
            if j == 0:
                # All mass below midpoint[0] goes to bin 0 (loss)
                row[j] = norm.cdf(midpoints[0], loc=mu, scale=sigma)
            elif j == K - 1:
                # All mass above midpoint[-1] goes to bin K-1 (fixation)
                row[j] = 1.0 - norm.cdf(midpoints[-1], loc=mu, scale=sigma)
            else:
                # Mass between midpoints[j-1] and midpoints[j]
                row[j] = (norm.cdf(midpoints[j], loc=mu, scale=sigma)
                          - norm.cdf(midpoints[j - 1], loc=mu, scale=sigma))

        # Renormalize (corrects for truncation beyond 3.3 sigma)
        row_sum = row.sum()
        if row_sum > 0:
            row /= row_sum
        else:
            row[np.argmin(np.abs(freqs - mu))] = 1.0

        # Convert to log probabilities
        logP[i, :] = np.where(row > 0, np.log(row), -np.inf)

    return logP


def build_normal_cdf_lookup(n_points=2000):
    """Precompute a lookup table for the standard normal CDF.

    The standard normal CDF Phi(z) is evaluated on a grid of z-values
    and stored for fast interpolation. Any normal CDF can be computed
    from the standard normal by the transformation:
        Phi((x - mu) / sigma) = P(N(mu, sigma^2) <= x)

    Parameters
    ----------
    n_points : int
        Number of grid points for the lookup table.

    Returns
    -------
    z_bins : ndarray
        Grid of z-values (from ~-37 to ~37 for n_points=2000).
    z_cdf : ndarray
        Standard normal CDF values at each z-bin.
    """
    # Create points in (0, 1), avoiding exactly 0 and 1
    u = np.linspace(0.0, 1.0, n_points)
    u[0] = 1e-10
    u[-1] = 1 - 1e-10

    # Map through the inverse normal CDF (quantile function)
    z_bins = norm.ppf(u)
    z_cdf = norm.cdf(z_bins)  # = u (by construction), but useful for interpolation

    return z_bins, z_cdf


def fast_normal_cdf(x, mu, sigma, z_bins, z_cdf):
    """Evaluate the normal CDF at x using the precomputed lookup table.

    Transforms (x - mu) / sigma to a standard normal z-score,
    then interpolates from the lookup table.

    Parameters
    ----------
    x : float
        Point at which to evaluate the CDF.
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation.
    z_bins : ndarray
        Precomputed z-values.
    z_cdf : ndarray
        Precomputed CDF values.

    Returns
    -------
    cdf_value : float
        Phi((x - mu) / sigma).
    """
    z = (x - mu) / sigma
    return np.interp(z, z_bins, z_cdf)


def logsumexp(a):
    """Compute log(sum(exp(a))) in a numerically stable way.

    This is the fundamental building block for all HMM computations
    in log space. The trick: subtract the maximum before exponentiating,
    then add it back after taking the log.

    Parameters
    ----------
    a : ndarray
        Array of log-probabilities.

    Returns
    -------
    result : float
        log(sum(exp(a))).
    """
    a_max = np.max(a)
    if a_max == -np.inf:
        return -np.inf
    return a_max + np.log(np.sum(np.exp(a - a_max)))


def build_transition_matrix_fast(freqs, N, s, z_bins, z_cdf, h=0.5):
    """Build the log-transition matrix using sparse computation.

    This is the optimized version that only computes entries within
    3.3 sigma of the mean, matching the CLUES2 implementation.

    Parameters
    ----------
    freqs : ndarray of shape (K,)
        Frequency bins.
    N : float
        Haploid effective population size.
    s : float
        Selection coefficient.
    z_bins : ndarray
        Precomputed standard normal z-values.
    z_cdf : ndarray
        Precomputed standard normal CDF values.
    h : float
        Dominance coefficient.

    Returns
    -------
    logP : ndarray of shape (K, K)
        Log-transition matrix.
    lower_indices : ndarray of shape (K,)
        First nonzero column index for each row.
    upper_indices : ndarray of shape (K,)
        Last nonzero column index (+1) for each row.
    """
    K = len(freqs)
    logP = np.full((K, K), -np.inf)
    lower_indices = np.zeros(K, dtype=int)
    upper_indices = np.zeros(K, dtype=int)

    # Midpoints between bins
    midpoints = (freqs[1:] + freqs[:-1]) / 2.0

    # Absorbing states
    logP[0, 0] = 0.0
    logP[K - 1, K - 1] = 0.0
    lower_indices[0] = 0
    upper_indices[0] = 1
    lower_indices[K - 1] = K - 1
    upper_indices[K - 1] = K

    for i in range(1, K - 1):
        x = freqs[i]
        mu = backward_mean(x, s, h)
        sigma = backward_std(x, N)

        if sigma < 1e-15:
            closest = np.argmin(np.abs(freqs - mu))
            logP[i, closest] = 0.0
            lower_indices[i] = closest
            upper_indices[i] = closest + 1
            continue

        # Bounds for sparse computation (3.3 sigma captures 99.9%)
        lower_freq = mu - 3.3 * sigma
        upper_freq = mu + 3.3 * sigma
        j_lo = max(0, np.searchsorted(freqs, lower_freq) - 1)
        j_hi = min(K, np.searchsorted(freqs, upper_freq) + 1)

        # Compute row probabilities
        row = np.zeros(K)
        for j in range(j_lo, j_hi):
            if j == 0:
                row[j] = fast_normal_cdf(midpoints[0], mu, sigma,
                                          z_bins, z_cdf)
            elif j == K - 1:
                row[j] = 1.0 - fast_normal_cdf(midpoints[-1], mu, sigma,
                                                 z_bins, z_cdf)
            else:
                row[j] = (fast_normal_cdf(midpoints[j], mu, sigma,
                                           z_bins, z_cdf)
                          - fast_normal_cdf(midpoints[j - 1], mu, sigma,
                                             z_bins, z_cdf))

        # Renormalize and convert to log
        row_sum = row.sum()
        if row_sum > 0:
            row /= row_sum
        logP[i, :] = np.where(row > 0, np.log(row), -np.inf)

        # Record nonzero range for fast summation (Approximation A2)
        nonzero = np.where(row > 0)[0]
        if len(nonzero) > 0:
            lower_indices[i] = nonzero[0]
            upper_indices[i] = nonzero[-1] + 1
        else:
            lower_indices[i] = 0
            upper_indices[i] = K

    return logP, lower_indices, upper_indices


# ============================================================================
# Emission Probabilities (from emission_probabilities.rst)
# ============================================================================

def log_coalescent_density(coal_times, n_lineages, epoch_start, epoch_end,
                            freq, N_diploid, ancestral=False):
    """Compute the log-probability of coalescence events in one epoch.

    This is the core emission computation for the CLUES HMM. It
    computes the probability of observing a specific set of coalescence
    times given the allele frequency and population size.

    Parameters
    ----------
    coal_times : ndarray
        Sorted coalescence times within this epoch. May be empty.
    n_lineages : int
        Number of lineages at the start of the epoch.
    epoch_start : float
        Start time of the epoch (generations).
    epoch_end : float
        End time of the epoch (generations).
    freq : float
        Derived allele frequency (0 < freq < 1).
    N_diploid : float
        Diploid effective population size.
    ancestral : bool
        If True, use ancestral frequency (1 - freq) instead.

    Returns
    -------
    log_prob : float
        Log-probability of the coalescence events.
    """
    if n_lineages <= 1:
        # No coalescence possible with 0 or 1 lineages
        return 0.0

    xi = (1.0 - freq) if ancestral else freq

    if xi * N_diploid == 0.0:
        # Impossible: lineages exist but frequency is 0
        return -1e20

    logp = 0.0
    prev_t = epoch_start
    k = n_lineages

    for t in coal_times:
        # k choose 2, divided by 4 because N is diploid
        # (equivalent to k(k-1)/2 divided by 2*N_diploid = N_haploid)
        kchoose2_over_4 = k * (k - 1) / 4.0
        rate = kchoose2_over_4 / (xi * N_diploid)

        # Exponential density: rate * exp(-rate * dt)
        # In log: log(rate) - rate * dt
        # But we split: -log(xi) accounts for the 1/xi factor in the rate
        dt = t - prev_t
        logp += -np.log(xi) - kchoose2_over_4 / (xi * N_diploid) * dt

        prev_t = t
        k -= 1

    # Survival probability: no further coalescences until epoch end
    if k >= 2:
        kchoose2_over_4 = k * (k - 1) / 4.0
        logp += -kchoose2_over_4 / (xi * N_diploid) * (epoch_end - prev_t)

    return logp


def compute_coalescent_emissions(coal_times_der, coal_times_anc,
                                  n_der, n_anc, epoch_start, epoch_end,
                                  freqs, N_diploid):
    """Compute coalescent emission probabilities for all frequency bins.

    Handles the special cases for mixed lineages:
    - If n_der > 1: standard derived + ancestral coalescences
    - If n_der == 1 and n_anc == 1: treat as 2 ancestral lineages (freq != 0)
    - If n_der == 1 and n_anc > 1: ancestral coalescences, but with n_anc+1
      lineages when freq = 0 (all lineages become ancestral)
    - If n_der == 0: all remaining lineages are ancestral

    Parameters
    ----------
    coal_times_der : ndarray
        Derived coalescence times in this epoch.
    coal_times_anc : ndarray
        Ancestral coalescence times in this epoch.
    n_der : int
        Number of remaining derived lineages.
    n_anc : int
        Number of remaining ancestral lineages.
    epoch_start : float
        Start of epoch.
    epoch_end : float
        End of epoch.
    freqs : ndarray of shape (K,)
        Frequency bins.
    N_diploid : float
        Diploid effective population size.

    Returns
    -------
    emissions : ndarray of shape (K,)
        Log-emission probabilities for each frequency bin.
    """
    K = len(freqs)
    emissions = np.zeros(K)

    for j in range(K):
        x = freqs[j]

        if n_der > 1:
            # Standard case: both derived and ancestral coalescences
            emissions[j] = log_coalescent_density(
                coal_times_der, n_der, epoch_start, epoch_end,
                x, N_diploid, ancestral=False)
            emissions[j] += log_coalescent_density(
                coal_times_anc, n_anc, epoch_start, epoch_end,
                x, N_diploid, ancestral=True)

        elif n_der == 0 and n_anc <= 1:
            # No lineages or single lineage: no coalescence possible
            if j != 0:
                emissions[j] = -1e20  # freq must be 0 (allele lost)

        elif n_der == 0 and n_anc > 1:
            # All remaining lineages are ancestral
            if j != 0:
                emissions[j] = -1e20  # freq must be 0
            else:
                emissions[j] = log_coalescent_density(
                    coal_times_anc, n_anc, epoch_start, epoch_end,
                    x, N_diploid, ancestral=True)

        elif n_der == 1 and n_anc == 1:
            # Mixed lineage: the single derived + single ancestral
            # coalesce as 2 ancestral lineages
            if j != 0:
                emissions[j] = 0.0  # no constraint from freq
            else:
                emissions[j] = log_coalescent_density(
                    coal_times_anc, 2, epoch_start, epoch_end,
                    x, N_diploid, ancestral=True)

        elif n_der == 1 and n_anc > 1:
            # One derived lineage remains, multiple ancestral
            if j != 0:
                emissions[j] = log_coalescent_density(
                    coal_times_anc, n_anc, epoch_start, epoch_end,
                    x, N_diploid, ancestral=True)
            else:
                # At freq = 0, the derived lineage joins the ancestral pool
                emissions[j] = log_coalescent_density(
                    coal_times_anc, n_anc + 1, epoch_start, epoch_end,
                    x, N_diploid, ancestral=True)

    return emissions


def genotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
    """Compute the log-emission probability for a diploid ancient sample.

    Parameters
    ----------
    anc_gl : ndarray of shape (3,)
        Log genotype likelihoods: [log P(R|AA), log P(R|AD), log P(R|DD)].
    log_freq : float
        log(x), where x is the derived allele frequency.
    log_1minus_freq : float
        log(1 - x).

    Returns
    -------
    log_emission : float
        log P(R | freq = x), marginalizing over genotypes.
    """
    # Hardy-Weinberg genotype frequencies (in log space)
    log_geno_freqs = np.array([
        log_1minus_freq + log_1minus_freq,          # log((1-x)^2) = AA
        np.log(2) + log_freq + log_1minus_freq,     # log(2x(1-x)) = AD
        log_freq + log_freq                          # log(x^2) = DD
    ])

    # Combine: P(R|x) = sum_g P(g|x) * P(R|g)
    log_emission = logsumexp(log_geno_freqs + anc_gl)

    if np.isnan(log_emission):
        return -np.inf
    return log_emission


def haplotype_likelihood_emission(anc_gl, log_freq, log_1minus_freq):
    """Compute the log-emission probability for a haploid ancient sample.

    Parameters
    ----------
    anc_gl : ndarray of shape (2,)
        Log haplotype likelihoods: [log P(R|ancestral), log P(R|derived)].
    log_freq : float
        log(x), derived allele frequency.
    log_1minus_freq : float
        log(1 - x).

    Returns
    -------
    log_emission : float
    """
    # Haplotype frequencies: P(ancestral) = 1-x, P(derived) = x
    log_hap_freqs = np.array([log_1minus_freq, log_freq])

    log_emission = logsumexp(log_hap_freqs + anc_gl)
    if np.isnan(log_emission):
        return -np.inf
    return log_emission


def compute_total_emissions(freq_bins, logfreqs, log1minusfreqs,
                             coal_times_der, coal_times_anc,
                             n_der, n_anc, epoch_start, epoch_end,
                             N_diploid,
                             diploid_gls=None, haploid_gls=None,
                             n_der_sampled=0, n_anc_sampled=0):
    """Compute total emission probabilities for all frequency bins.

    Combines coalescent emissions, ancient genotype likelihoods,
    and known haplotype emissions.

    Parameters
    ----------
    freq_bins : ndarray of shape (K,)
        Frequency bins.
    logfreqs : ndarray of shape (K,)
        log(freq_bins).
    log1minusfreqs : ndarray of shape (K,)
        log(1 - freq_bins).
    coal_times_der : ndarray
        Derived coalescence times in this epoch.
    coal_times_anc : ndarray
        Ancestral coalescence times in this epoch.
    n_der : int
        Remaining derived lineages.
    n_anc : int
        Remaining ancestral lineages.
    epoch_start : float
        Start of epoch (generations).
    epoch_end : float
        End of epoch (generations).
    N_diploid : float
        Diploid effective population size.
    diploid_gls : list of ndarray, optional
        Each element is [log P(R|AA), log P(R|AD), log P(R|DD)] for one
        ancient diploid sample in this epoch.
    haploid_gls : list of ndarray, optional
        Each element is [log P(R|A), log P(R|D)] for one ancient haploid
        sample in this epoch.
    n_der_sampled : int
        Number of known derived haplotypes sampled in this epoch.
    n_anc_sampled : int
        Number of known ancestral haplotypes sampled in this epoch.

    Returns
    -------
    total_emissions : ndarray of shape (K,)
        Log-emission probability for each frequency bin.
    """
    K = len(freq_bins)

    # 1. Coalescent emissions
    coal_emissions = compute_coalescent_emissions(
        coal_times_der, coal_times_anc, n_der, n_anc,
        epoch_start, epoch_end, freq_bins, N_diploid)

    # 2. Ancient genotype likelihoods
    gl_emissions = np.zeros(K)
    if diploid_gls is not None:
        for gl in diploid_gls:
            for j in range(K):
                gl_emissions[j] += genotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])

    if haploid_gls is not None:
        for gl in haploid_gls:
            for j in range(K):
                gl_emissions[j] += haplotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])

    # 3. Known haplotype emissions from ARG samples
    hap_emissions = np.zeros(K)
    for j in range(K):
        if n_der_sampled > 0:
            hap_emissions[j] += n_der_sampled * logfreqs[j]
        if n_anc_sampled > 0:
            hap_emissions[j] += n_anc_sampled * log1minusfreqs[j]

    return coal_emissions + gl_emissions + hap_emissions


# ============================================================================
# Inference (from inference.rst)
# ============================================================================

def backward_algorithm(sel, freqs, logfreqs, log1minusfreqs,
                        z_bins, z_cdf, epochs, N_vec, h,
                        coal_times_der_all, coal_times_anc_all,
                        n_der_initial, n_anc_initial,
                        curr_freq,
                        diploid_gls_by_epoch=None,
                        haploid_gls_by_epoch=None,
                        der_sampled_by_epoch=None,
                        anc_sampled_by_epoch=None):
    """Run the CLUES backward algorithm (present to past).

    Parameters
    ----------
    sel : ndarray
        Selection coefficient for each epoch.
    freqs : ndarray of shape (K,)
        Frequency bins.
    logfreqs, log1minusfreqs : ndarray of shape (K,)
        Log-frequencies for emission computation.
    z_bins, z_cdf : ndarray
        Precomputed normal CDF lookup table.
    epochs : ndarray
        Array of generation indices [0, 1, 2, ..., T].
    N_vec : ndarray
        Diploid effective population size at each epoch.
    h : float
        Dominance coefficient.
    coal_times_der_all : ndarray
        All derived coalescence times (sorted).
    coal_times_anc_all : ndarray
        All ancestral coalescence times (sorted).
    n_der_initial : int
        Number of derived lineages at the present.
    n_anc_initial : int
        Number of ancestral lineages at the present.
    curr_freq : float
        Observed modern derived allele frequency.
    diploid_gls_by_epoch : dict, optional
        Maps epoch index to list of diploid GL arrays.
    haploid_gls_by_epoch : dict, optional
        Maps epoch index to list of haploid GL arrays.
    der_sampled_by_epoch : dict, optional
        Maps epoch index to number of derived haplotypes sampled.
    anc_sampled_by_epoch : dict, optional
        Maps epoch index to number of ancestral haplotypes sampled.

    Returns
    -------
    alpha_mat : ndarray of shape (T+1, K)
        Log-probability matrix. alpha_mat[t, k] is the log-probability
        of the data from time 0 to t, with frequency bin k at time t.
    """
    K = len(freqs)
    T = len(epochs)

    # Initialize: delta function at modern frequency
    alpha = np.full(K, -1e20)
    best_bin = np.argmin(np.abs(freqs - curr_freq))
    alpha[best_bin] = 0.0

    alpha_mat = np.full((T, K), -1e20)
    alpha_mat[0, :] = alpha

    # Track remaining lineages
    n_der = n_der_initial
    n_anc = n_anc_initial

    prev_N = -1
    prev_s = -1
    logP = None

    for tb in range(T - 1):
        epoch_start = float(tb)
        epoch_end = float(tb + 1)
        N_t = N_vec[tb]
        s_t = sel[tb] if tb < len(sel) else 0.0

        prev_alpha = alpha.copy()

        # Recompute transition matrix only if N or s changed
        if N_t != prev_N or s_t != prev_s:
            logP, lo_idx, hi_idx = build_transition_matrix_fast(
                freqs, 2 * N_t, s_t, z_bins, z_cdf, h)
            prev_N = N_t
            prev_s = s_t

        # Gather coalescence times in this epoch
        mask_der = (coal_times_der_all > epoch_start) & \
                   (coal_times_der_all <= epoch_end)
        coal_der = coal_times_der_all[mask_der]

        mask_anc = (coal_times_anc_all > epoch_start) & \
                   (coal_times_anc_all <= epoch_end)
        coal_anc = coal_times_anc_all[mask_anc]

        # Gather ancient samples in this epoch
        dip_gls = (diploid_gls_by_epoch or {}).get(tb, [])
        hap_gls = (haploid_gls_by_epoch or {}).get(tb, [])
        n_der_samp = (der_sampled_by_epoch or {}).get(tb, 0)
        n_anc_samp = (anc_sampled_by_epoch or {}).get(tb, 0)

        # Compute emissions
        coal_emissions = compute_coalescent_emissions(
            coal_der, coal_anc, n_der, n_anc,
            epoch_start, epoch_end, freqs, N_t)

        gl_emissions = np.zeros(K)
        for gl in dip_gls:
            for j in range(K):
                gl_emissions[j] += genotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])
        for gl in hap_gls:
            for j in range(K):
                gl_emissions[j] += haplotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])

        hap_emissions = np.zeros(K)
        for j in range(K):
            if n_der_samp > 0:
                hap_emissions[j] += n_der_samp * logfreqs[j]
            if n_anc_samp > 0:
                hap_emissions[j] += n_anc_samp * log1minusfreqs[j]

        total_emissions = gl_emissions + hap_emissions + coal_emissions

        # HMM update: alpha[k] = emission[k] + logsumexp(prev_alpha + P^T[:,k])
        for k in range(K):
            # Use sparse column range for efficiency
            col_lo = lo_idx[k] if lo_idx is not None else 0
            col_hi = hi_idx[k] if hi_idx is not None else K
            # P^T[j, k] = P[j, k] for column k = logP[j, k]
            alpha[k] = total_emissions[k] + logsumexp(
                prev_alpha[col_lo:col_hi] + logP[col_lo:col_hi, k])
            if np.isnan(alpha[k]):
                alpha[k] = -np.inf

        # Update lineage counts
        n_der -= len(coal_der)
        n_anc -= len(coal_anc)
        n_der += n_der_samp
        n_anc += n_anc_samp

        alpha_mat[tb + 1, :] = alpha

    return alpha_mat


def forward_algorithm(sel, freqs, logfreqs, log1minusfreqs,
                       z_bins, z_cdf, epochs, N_vec, h,
                       coal_times_der_all, coal_times_anc_all,
                       n_der_initial, n_anc_initial,
                       diploid_gls_by_epoch=None,
                       haploid_gls_by_epoch=None,
                       der_sampled_by_epoch=None,
                       anc_sampled_by_epoch=None):
    """Run the CLUES forward algorithm (past to present).

    Parameters match backward_algorithm (except no curr_freq needed).

    Returns
    -------
    alpha_mat : ndarray of shape (T+1, K)
        Forward log-probability matrix.
    """
    K = len(freqs)
    T = len(epochs)

    # Initialize: uniform at the oldest time point
    alpha = np.ones(K)
    alpha = np.log(alpha / np.sum(alpha))  # log(1/K)

    alpha_mat = np.full((T, K), -1e20)
    alpha_mat[-1, :] = alpha

    prev_N = -1
    prev_s = -1

    # Track lineages from the past
    # At the deepest time, we start with all lineages minus those that
    # coalesced deeper than our cutoff
    n_der = n_der_initial
    n_anc = n_anc_initial

    # Count lineages remaining at the deepest epoch
    deep_der_coals = coal_times_der_all[coal_times_der_all <= float(T)]
    deep_anc_coals = coal_times_anc_all[coal_times_anc_all <= float(T)]
    n_der_remaining = n_der - len(deep_der_coals)
    n_anc_remaining = n_anc - len(deep_anc_coals)

    for tb in range(T - 2, -1, -1):
        epoch_start = float(tb)
        epoch_end = float(tb + 1)

        N_t = N_vec[tb]
        s_t = sel[tb] if tb < len(sel) else 0.0
        prev_alpha = alpha.copy()

        if N_t != prev_N or s_t != prev_s:
            logP, lo_idx, hi_idx = build_transition_matrix_fast(
                freqs, 2 * N_t, s_t, z_bins, z_cdf, h)
            prev_N = N_t
            prev_s = s_t

        # Gather data for this epoch (reversed direction)
        mask_der = (coal_times_der_all > epoch_start) & \
                   (coal_times_der_all <= epoch_end)
        coal_der = coal_times_der_all[mask_der]

        mask_anc = (coal_times_anc_all > epoch_start) & \
                   (coal_times_anc_all <= epoch_end)
        coal_anc = coal_times_anc_all[mask_anc]

        # Compute emissions for this epoch
        dip_gls = (diploid_gls_by_epoch or {}).get(tb, [])
        hap_gls = (haploid_gls_by_epoch or {}).get(tb, [])

        gl_emissions = np.zeros(K)
        for gl in dip_gls:
            for j in range(K):
                gl_emissions[j] += genotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])
        for gl in hap_gls:
            for j in range(K):
                gl_emissions[j] += haplotype_likelihood_emission(
                    gl, logfreqs[j], log1minusfreqs[j])

        coal_emissions = compute_coalescent_emissions(
            coal_der, coal_anc, n_der_remaining, n_anc_remaining,
            epoch_start, epoch_end, freqs, N_t)

        total_emissions = gl_emissions + coal_emissions

        # Forward update: use P[i,j] (not transposed)
        for k in range(K):
            alpha[k] = logsumexp(
                prev_alpha[lo_idx[k]:hi_idx[k]]
                + logP[k, lo_idx[k]:hi_idx[k]]
                + total_emissions[lo_idx[k]:hi_idx[k]])
            if np.isnan(alpha[k]):
                alpha[k] = -np.inf

        n_der_remaining += len(coal_der)
        n_anc_remaining += len(coal_anc)

        alpha_mat[tb, :] = alpha

    return alpha_mat


def compute_neutral_weights(times_all, freqs, logfreqs, log1minusfreqs,
                             z_bins, z_cdf, epochs, N_vec, h, curr_freq,
                             n_der_initial, n_anc_initial,
                             diploid_gls_by_epoch=None,
                             haploid_gls_by_epoch=None,
                             der_sampled_by_epoch=None,
                             anc_sampled_by_epoch=None):
    """Compute neutral importance weights for each gene tree sample.

    Parameters
    ----------
    times_all : ndarray of shape (2, max_lineages, M)
        Coalescence times. times_all[0] = derived, times_all[1] = ancestral.
        Third axis indexes importance samples.

    Returns
    -------
    weights : ndarray of shape (M,)
        Log-likelihood of each tree under neutrality.
    """
    M = times_all.shape[2]
    weights = np.zeros(M)
    sel_neutral = np.zeros(len(N_vec))

    for m in range(M):
        # Extract coalescence times for this sample
        der_times = times_all[0, :, m]
        der_times = der_times[der_times >= 0]  # -1 marks unused entries
        anc_times = times_all[1, :, m]
        anc_times = anc_times[anc_times >= 0]

        alpha_mat = backward_algorithm(
            sel_neutral, freqs, logfreqs, log1minusfreqs,
            z_bins, z_cdf, epochs, N_vec, h,
            der_times, anc_times,
            n_der_initial, n_anc_initial, curr_freq,
            diploid_gls_by_epoch, haploid_gls_by_epoch,
            der_sampled_by_epoch, anc_sampled_by_epoch)

        weights[m] = logsumexp(alpha_mat[-2, :])

    return weights


def importance_sampled_likelihood(sel_vec, times_all, weights,
                                   freqs, logfreqs, log1minusfreqs,
                                   z_bins, z_cdf, epochs, N_vec, h,
                                   curr_freq,
                                   n_der_initial, n_anc_initial,
                                   diploid_gls_by_epoch=None,
                                   haploid_gls_by_epoch=None,
                                   der_sampled_by_epoch=None,
                                   anc_sampled_by_epoch=None):
    """Compute importance-sampled log-likelihood for a given selection vector.

    Returns the negative log-likelihood (for minimization).
    """
    M = times_all.shape[2]
    log_ratios = np.zeros(M)

    for m in range(M):
        der_times = times_all[0, :, m]
        der_times = der_times[der_times >= 0]
        anc_times = times_all[1, :, m]
        anc_times = anc_times[anc_times >= 0]

        alpha_mat = backward_algorithm(
            sel_vec, freqs, logfreqs, log1minusfreqs,
            z_bins, z_cdf, epochs, N_vec, h,
            der_times, anc_times,
            n_der_initial, n_anc_initial, curr_freq,
            diploid_gls_by_epoch, haploid_gls_by_epoch,
            der_sampled_by_epoch, anc_sampled_by_epoch)

        log_lik = logsumexp(alpha_mat[-2, :])
        log_ratios[m] = log_lik - weights[m]

    # Importance-sampled log-likelihood ratio
    log_lr = -np.log(M) + logsumexp(log_ratios)
    return -log_lr  # negative for minimization


def estimate_selection_single(neg_log_lik_func, s_max=0.1):
    """Estimate the selection coefficient using Brent's method.

    Parameters
    ----------
    neg_log_lik_func : callable
        Function that takes s (float) and returns negative log-likelihood.
    s_max : float
        Maximum absolute selection coefficient to search.

    Returns
    -------
    s_hat : float
        Maximum likelihood estimate of s.
    neg_log_lik : float
        Negative log-likelihood at s_hat.
    """
    # Brent's method with bracket [1-sMax, 1, 1+sMax]
    # (CLUES adds 1 to s for better numerical behavior near 0)
    def shifted_func(theta):
        return neg_log_lik_func(theta - 1.0)

    try:
        result = minimize_scalar(
            shifted_func,
            bracket=[1.0 - s_max, 1.0, 1.0 + s_max],
            method='Brent',
            options={'xtol': 1e-4})
        s_hat = result.x - 1.0
        neg_ll = result.fun
    except ValueError:
        # If bracket fails, try a wider search
        result = minimize_scalar(
            shifted_func,
            bracket=[0.0, 1.0, 2.0],
            method='Brent',
            options={'xtol': 1e-4})
        s_hat = result.x - 1.0
        neg_ll = result.fun

    return s_hat, neg_ll


def estimate_selection_multi_epoch(neg_log_lik_func, n_epochs, s_max=0.1):
    """Estimate epoch-specific selection coefficients using Nelder-Mead.

    Parameters
    ----------
    neg_log_lik_func : callable
        Takes an array of selection coefficients and returns neg log-lik.
    n_epochs : int
        Number of selection epochs.
    s_max : float
        Maximum absolute selection coefficient.

    Returns
    -------
    s_hat : ndarray of shape (n_epochs,)
        MLE selection coefficients for each epoch.
    neg_log_lik : float
        Negative log-likelihood at the optimum.
    """
    # Initial simplex: one vertex at all-zeros, others with 0.01 in each epoch
    initial_simplex = np.zeros((n_epochs + 1, n_epochs))
    for i in range(n_epochs):
        initial_simplex[i, i] = 0.01

    result = minimize(
        neg_log_lik_func,
        x0=np.zeros(n_epochs),
        method='Nelder-Mead',
        options={
            'initial_simplex': initial_simplex,
            'maxfev': n_epochs * 20,
            'xatol': 1e-4,
            'fatol': 1e-4,
        })

    return result.x, result.fun


def likelihood_ratio_test(log_lik_selected, log_lik_neutral, df=1):
    """Perform a likelihood ratio test for selection.

    Parameters
    ----------
    log_lik_selected : float
        Log-likelihood under the selected model.
    log_lik_neutral : float
        Log-likelihood under the neutral model (s=0).
    df : int
        Degrees of freedom (number of selection parameters).

    Returns
    -------
    log_lr : float
        Log-likelihood ratio (2 * (log L_selected - log L_neutral)).
    p_value : float
        p-value from chi-squared distribution.
    neg_log10_p : float
        -log10(p-value) for convenient reporting.
    """
    log_lr = 2 * (log_lik_selected - log_lik_neutral)

    # Ensure log_lr >= 0 (numerical issues can make it slightly negative)
    log_lr = max(log_lr, 0.0)

    # p-value from chi-squared survival function
    p_value = chi2.sf(log_lr, df)
    neg_log10_p = -np.log10(p_value) if p_value > 0 else np.inf

    return log_lr, p_value, neg_log10_p


def reconstruct_trajectory(sel_samples, freqs, logfreqs, log1minusfreqs,
                            z_bins, z_cdf, epochs, N_vec, h,
                            coal_times_der, coal_times_anc,
                            n_der, n_anc, curr_freq,
                            weights=None, times_all=None):
    """Reconstruct the posterior allele frequency trajectory.

    Averages over multiple values of s drawn from the posterior,
    and (if importance sampling) over multiple gene tree samples.

    Parameters
    ----------
    sel_samples : list of ndarray
        Each element is a selection vector [s1, s2, ...] drawn from
        the posterior of s. For single-epoch, each is a 1-element list.
    (other parameters as in backward_algorithm)
    weights : ndarray of shape (M,), optional
        Neutral importance weights (if using importance sampling).
    times_all : ndarray of shape (2, n, M), optional
        All gene tree samples (if using importance sampling).

    Returns
    -------
    posterior : ndarray of shape (K, T)
        Posterior probability matrix. posterior[k, t] is the
        probability that the allele frequency at time t is x_k.
    """
    K = len(freqs)
    T = len(epochs)
    accumulated_post = np.zeros((K, T - 1))

    for sel_vec in sel_samples:
        if times_all is not None and times_all.shape[2] > 1:
            # Importance sampling: average over gene tree samples
            M = times_all.shape[2]
            log_ratios = np.zeros(M)
            posts_by_sample = np.zeros((K, T - 1, M))

            for m in range(M):
                der_t = times_all[0, :, m]
                der_t = der_t[der_t >= 0]
                anc_t = times_all[1, :, m]
                anc_t = anc_t[anc_t >= 0]

                bwd = backward_algorithm(
                    sel_vec, freqs, logfreqs, log1minusfreqs,
                    z_bins, z_cdf, epochs, N_vec, h,
                    der_t, anc_t, n_der, n_anc, curr_freq)
                fwd = forward_algorithm(
                    sel_vec, freqs, logfreqs, log1minusfreqs,
                    z_bins, z_cdf, epochs, N_vec, h,
                    der_t, anc_t, n_der, n_anc)

                log_lik = logsumexp(bwd[-2, :])
                log_ratios[m] = log_lik - weights[m]

                # Posterior at each time: forward * backward
                post = (fwd[1:, :] + bwd[:-1, :]).T
                posts_by_sample[:, :, m] = post

            # Weight-average across samples
            for t in range(T - 1):
                for k in range(K):
                    vals = log_ratios + posts_by_sample[k, t, :]
                    accumulated_post[k, t] += np.exp(logsumexp(vals))

        else:
            # Single tree: no importance sampling
            bwd = backward_algorithm(
                sel_vec, freqs, logfreqs, log1minusfreqs,
                z_bins, z_cdf, epochs, N_vec, h,
                coal_times_der, coal_times_anc,
                n_der, n_anc, curr_freq)
            fwd = forward_algorithm(
                sel_vec, freqs, logfreqs, log1minusfreqs,
                z_bins, z_cdf, epochs, N_vec, h,
                coal_times_der, coal_times_anc,
                n_der, n_anc)

            post = (fwd[1:, :] + bwd[:-1, :]).T
            accumulated_post += np.exp(post - logsumexp(post.flatten()))

    # Normalize columns to sum to 1
    col_sums = accumulated_post.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    posterior = accumulated_post / col_sums

    return posterior


def compute_trajectory_summary(posterior, freqs):
    """Compute summary statistics of the posterior trajectory.

    Parameters
    ----------
    posterior : ndarray of shape (K, T)
        Posterior probability matrix.
    freqs : ndarray of shape (K,)
        Frequency bins.

    Returns
    -------
    mean_freq : ndarray of shape (T,)
        Posterior mean frequency at each time.
    lower_95 : ndarray of shape (T,)
        2.5th percentile of the posterior at each time.
    upper_95 : ndarray of shape (T,)
        97.5th percentile.
    """
    T = posterior.shape[1]
    mean_freq = np.zeros(T)
    lower_95 = np.zeros(T)
    upper_95 = np.zeros(T)

    for t in range(T):
        col = posterior[:, t]
        if col.sum() == 0:
            continue
        col = col / col.sum()
        mean_freq[t] = np.sum(freqs * col)

        # Compute percentiles from the CDF
        cdf = np.cumsum(col)
        lower_95[t] = freqs[np.searchsorted(cdf, 0.025)]
        upper_95[t] = freqs[np.searchsorted(cdf, 0.975)]

    return mean_freq, lower_95, upper_95


def run_clues(curr_freq, N_diploid, t_cutoff, K=450, s_max=0.1, h=0.5,
               coal_times_der=None, coal_times_anc=None,
               times_all=None, ancient_gls=None):
    """Run the complete CLUES inference pipeline.

    This is a simplified version showing the algorithm structure.
    The full implementation handles additional edge cases and
    optimizations.

    Parameters
    ----------
    curr_freq : float
        Modern derived allele frequency.
    N_diploid : float
        Diploid effective population size (constant).
    t_cutoff : int
        Maximum analysis time (generations).
    K : int
        Number of frequency bins.
    s_max : float
        Maximum selection coefficient to search.
    h : float
        Dominance coefficient.
    coal_times_der : ndarray, optional
        Derived coalescence times (single tree).
    coal_times_anc : ndarray, optional
        Ancestral coalescence times (single tree).
    times_all : ndarray of shape (2, n, M), optional
        Multiple tree samples for importance sampling.
    ancient_gls : ndarray of shape (n_samples, 4), optional
        Ancient genotype likelihoods [time, P(AA), P(AD), P(DD)].

    Returns
    -------
    results : dict
        Dictionary with keys: s_hat, log_lr, p_value, neg_log10_p,
        posterior, mean_freq, lower_95, upper_95, freqs.
    """
    # Set up frequency bins and lookup tables
    freqs, logfreqs, log1minusfreqs = build_frequency_bins(K)
    z_bins, z_cdf = build_normal_cdf_lookup()

    # Set up epochs and population sizes
    epochs = np.arange(0.0, t_cutoff)
    N_vec = N_diploid * np.ones(int(t_cutoff))

    # Determine number of initial lineages
    if times_all is not None:
        n_der = int(np.sum(times_all[0, :, 0] >= 0))
        n_anc = int(np.sum(times_all[1, :, 0] >= 0)) + 1
        use_importance_sampling = times_all.shape[2] > 1
    elif coal_times_der is not None:
        n_der = len(coal_times_der) + 1  # n coalescences => n+1 lineages
        n_anc = len(coal_times_anc) + 1
        use_importance_sampling = False
    else:
        n_der = 0
        n_anc = 0
        use_importance_sampling = False

    # Step 1: Compute neutral weights (if importance sampling)
    if use_importance_sampling:
        weights = compute_neutral_weights(
            times_all, freqs, logfreqs, log1minusfreqs,
            z_bins, z_cdf, epochs, N_vec, h, curr_freq,
            n_der, n_anc)
    else:
        weights = None

    # Step 2: Define the negative log-likelihood function
    def neg_log_lik(s_val):
        sel = np.array([s_val])
        if abs(s_val) > s_max:
            return 1e10

        if use_importance_sampling:
            return importance_sampled_likelihood(
                sel, times_all, weights,
                freqs, logfreqs, log1minusfreqs,
                z_bins, z_cdf, epochs, N_vec, h, curr_freq,
                n_der, n_anc)
        else:
            alpha_mat = backward_algorithm(
                sel, freqs, logfreqs, log1minusfreqs,
                z_bins, z_cdf, epochs, N_vec, h,
                coal_times_der, coal_times_anc,
                n_der, n_anc, curr_freq)
            return -logsumexp(alpha_mat[-2, :])

    # Step 3: Find MLE of s
    s_hat, neg_ll_selected = estimate_selection_single(neg_log_lik, s_max)
    neg_ll_neutral = neg_log_lik(0.0)

    # Step 4: Likelihood ratio test
    log_lr, p_value, neg_log10_p = likelihood_ratio_test(
        -neg_ll_selected, -neg_ll_neutral, df=1)

    print(f"Selection MLE:  s_hat = {s_hat:.6f}")
    print(f"Log-LR:         {log_lr:.4f}")
    print(f"p-value:        {p_value:.6f}")
    print(f"-log10(p):      {neg_log10_p:.2f}")

    # Step 5: Reconstruct trajectory
    # Draw samples from approximate posterior of s
    sel_samples = [[s_hat]]  # simplified: just use MLE

    posterior = reconstruct_trajectory(
        sel_samples, freqs, logfreqs, log1minusfreqs,
        z_bins, z_cdf, epochs, N_vec, h,
        coal_times_der or np.array([]),
        coal_times_anc or np.array([]),
        n_der, n_anc, curr_freq,
        weights, times_all)

    mean_freq, lower_95, upper_95 = compute_trajectory_summary(
        posterior, freqs)

    return {
        's_hat': s_hat,
        'log_lr': log_lr,
        'p_value': p_value,
        'neg_log10_p': neg_log10_p,
        'posterior': posterior,
        'mean_freq': mean_freq,
        'lower_95': lower_95,
        'upper_95': upper_95,
        'freqs': freqs,
    }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the CLUES algorithm components."""
    print("=" * 70)
    print("CLUES: Coalescent Likelihood Under Effects of Selection")
    print("=" * 70)

    # --- Wright-Fisher HMM ---
    print("\n--- Wright-Fisher HMM ---\n")

    # Verify: under neutrality (s=0), backward mean equals current frequency
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mu = backward_mean(x, s=0.0)
        print(f"x = {x:.1f}, s = 0:   mu = {mu:.6f} (expected: {x:.6f})")

    # Under positive selection, backward mean < current frequency
    print()
    for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mu = backward_mean(x, s=0.05)
        print(f"x = {x:.1f}, s = 0.05: mu = {mu:.6f} (shift: {mu - x:+.6f})")

    # Build frequency bins
    print("\n--- Frequency Bins ---\n")
    freqs, logfreqs, log1minusfreqs = build_frequency_bins(K=450)
    print(f"Number of bins: {len(freqs)}")
    print(f"First 10 bins: {np.round(freqs[:10], 6)}")
    print(f"Last 10 bins:  {np.round(freqs[-10:], 6)}")
    print(f"Bin spacing near 0: {np.diff(freqs[:5])}")
    print(f"Bin spacing near 0.5: {np.diff(freqs[220:226])}")
    print(f"Bin spacing near 1: {np.diff(freqs[-5:])}")

    # Build and verify transition matrix
    print("\n--- Transition Matrix ---\n")
    freqs_small, _, _ = build_frequency_bins(K=50)
    N_haploid = 20000.0

    logP_neutral = build_transition_matrix(freqs_small, N_haploid, s=0.0)
    P_neutral = np.exp(logP_neutral)
    print("Row sums (should be ~1.0):",
          np.round(P_neutral.sum(axis=1)[:5], 6))

    logP_selected = build_transition_matrix(freqs_small, N_haploid, s=0.05)
    P_selected = np.exp(logP_selected)
    print("Row sums (should be ~1.0):",
          np.round(P_selected.sum(axis=1)[:5], 6))

    # Visualize effect of selection
    target_freq = 0.3
    i = np.argmin(np.abs(freqs_small - target_freq))
    print(f"\nStarting frequency bin: x[{i}] = {freqs_small[i]:.4f}")
    trans_neutral = np.exp(logP_neutral[i, :])
    trans_selected = np.exp(logP_selected[i, :])
    mean_neutral = np.sum(freqs_small * trans_neutral)
    mean_selected = np.sum(freqs_small * trans_selected)
    print(f"Mean next frequency (neutral):  {mean_neutral:.6f}")
    print(f"Mean next frequency (s=0.05):   {mean_selected:.6f}")
    print(f"Shift due to selection:         {mean_selected - mean_neutral:+.6f}")
    print(f"(Going backward, s>0 shifts the allele to lower past frequency)")

    # Compare dominance models
    print("\n--- Dominance Models ---\n")
    x_vals = np.linspace(0.01, 0.99, 50)
    s = 0.05
    for h_val, label in [(0.0, "Recessive (h=0)"),
                           (0.5, "Additive (h=0.5)"),
                           (1.0, "Dominant (h=1)")]:
        shifts = [backward_mean(x, s, h=h_val) - x for x in x_vals]
        max_shift_freq = x_vals[np.argmin(shifts)]
        print(f"{label}: max backward shift at x = {max_shift_freq:.2f}, "
              f"shift = {min(shifts):.6f}")

    # Fast normal CDF lookup
    print("\n--- Fast Normal CDF Lookup ---\n")
    z_bins, z_cdf = build_normal_cdf_lookup()
    test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
    print("Verification of fast normal CDF lookup:")
    for z in test_points:
        exact = norm.cdf(z)
        approx = fast_normal_cdf(z, 0.0, 1.0, z_bins, z_cdf)
        print(f"  z = {z:+.1f}: exact = {exact:.8f}, approx = {approx:.8f}, "
              f"error = {abs(exact - approx):.2e}")

    # Log-sum-exp
    print("\n--- Log-Sum-Exp ---\n")
    result = logsumexp(np.array([-1000.0, -1001.0]))
    print(f"logsumexp([-1000, -1001]) = {result:.4f}")
    print(f"Expected: {-1000 + np.log(1 + np.exp(-1)):.4f}")

    # Fast transition matrix
    print("\n--- Fast Transition Matrix ---\n")
    freqs100, _, _ = build_frequency_bins(K=100)
    logP, lo, hi = build_transition_matrix_fast(
        freqs100, N_haploid, s=0.02, z_bins=z_bins, z_cdf=z_cdf)
    nnz_per_row = [hi[i] - lo[i] for i in range(len(freqs100))]
    print(f"Average nonzero entries per row: {np.mean(nnz_per_row):.1f}")
    print(f"Max nonzero entries: {max(nnz_per_row)}")
    print(f"Matrix size: {len(freqs100)}x{len(freqs100)} = "
          f"{len(freqs100)**2}")
    print(f"Sparsity: "
          f"{100 * (1 - np.mean(nnz_per_row)/len(freqs100)):.1f}% zeros")

    # --- Emission Probabilities ---
    print("\n--- Coalescent Emissions ---\n")
    coal_times = np.array([0.5])
    log_prob = log_coalescent_density(
        coal_times, n_lineages=3, epoch_start=0.0, epoch_end=1.0,
        freq=0.3, N_diploid=10000.0, ancestral=False)
    print(f"Log-probability of coalescence: {log_prob:.4f}")

    print("\nCoalescence probability vs. derived allele frequency:")
    for freq in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        lp = log_coalescent_density(
            coal_times, n_lineages=3, epoch_start=0.0, epoch_end=1.0,
            freq=freq, N_diploid=10000.0)
        print(f"  freq = {freq:.1f}: log P = {lp:.4f}")

    # Ancient genotype likelihoods
    print("\n--- Ancient Genotype Emissions ---\n")
    gl = np.log(np.array([0.01, 0.24, 0.75]))
    print("Ancient genotype emission vs. frequency:")
    for freq in [0.1, 0.3, 0.5, 0.7, 0.9]:
        log_em = genotype_likelihood_emission(
            gl, np.log(freq), np.log(1 - freq))
        print(f"  freq = {freq:.1f}: log P(R|x) = {log_em:.4f}, "
              f"P(R|x) = {np.exp(log_em):.6f}")

    # --- Inference ---
    print("\n--- Likelihood Ratio Test ---\n")
    log_lr, p_val, neg_log10_p = likelihood_ratio_test(
        log_lik_selected=-1000.0, log_lik_neutral=-1005.0, df=1)
    print(f"Log-LR = {log_lr:.2f}")
    print(f"p-value = {p_val:.6f}")
    print(f"-log10(p) = {neg_log10_p:.2f}")
    print(f"Significant at alpha=0.05? {'Yes' if p_val < 0.05 else 'No'}")

    # Selection estimation (toy example)
    print("\n--- Selection Estimation (Toy) ---\n")
    true_s = 0.03

    def toy_neg_log_lik(s_val):
        return (s_val - true_s)**2 / 0.001

    s_hat, nll = estimate_selection_single(toy_neg_log_lik)
    print(f"True s = {true_s}, Estimated s = {s_hat:.6f}")

    # Trajectory summary
    print("\n--- Trajectory Summary ---\n")
    K_demo, T_demo = 50, 10
    freqs_demo = np.linspace(0, 1, K_demo)
    posterior_demo = np.random.dirichlet(np.ones(K_demo), size=T_demo).T
    mean_f, lower, upper = compute_trajectory_summary(posterior_demo,
                                                       freqs_demo)
    print(f"Mean frequencies: {np.round(mean_f, 3)}")
    print(f"95% CI width: {np.round(upper - lower, 3)}")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
