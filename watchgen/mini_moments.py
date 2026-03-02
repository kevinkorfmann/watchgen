"""
Mini-moments: Inferring Demographic History from the Frequency Spectrum.

This module implements the core algorithms from moments (Jouganous et al. 2017),
which infers demographic history by computing how the site frequency spectrum
(SFS) evolves through time under different demographic scenarios, then finding
the scenario that best matches observed data.

The approach:
1. Summarize genetic variation as the site frequency spectrum (SFS) -- a
   histogram of allele frequencies across the sample.
2. Derive ordinary differential equations (ODEs) that govern how each SFS
   entry changes under drift, mutation, selection, and migration.
3. Integrate these ODEs forward through a demographic model to predict the
   expected SFS.
4. Find the demographic parameters that maximize the likelihood of the
   observed data using the Poisson Random Field likelihood.

Key concepts:
- The SFS entry phi[j] is the expected number of sites with derived allele
  count j in a sample of n chromosomes.
- Under the neutral model, E[SFS[j]] = theta / j (the 1/j law).
- The moment equations are a system of coupled ODEs -- one per SFS entry --
  driven by drift (second-order), mutation (singleton injection), selection
  (advection), and migration (coupling between populations).
- Population size nu scales the drift operator: larger nu means weaker drift.
- Population splits use hypergeometric sampling to distribute alleles across
  daughter populations.

Reference:
    Jouganous J, Long W, Ragsdale AP, Gravel S (2017).
    Inferring the joint demographic history of multiple populations from
    multidimensional SNP frequency data using moments. Genetics, 206(2):713-727.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import comb
from scipy.stats import chi2


# ===========================================================================
# Chapter: The Site Frequency Spectrum
# ===========================================================================

def compute_sfs(genotype_matrix, n):
    """Compute the SFS from a genotype matrix.

    Parameters
    ----------
    genotype_matrix : ndarray of shape (L, n)
        Each row is a site, each column a haploid chromosome.
        Entries are 0 (ancestral) or 1 (derived).
    n : int
        Number of chromosomes.

    Returns
    -------
    sfs : ndarray of shape (n+1,)
        sfs[j] = number of sites with derived allele count j.
    """
    sfs = np.zeros(n + 1, dtype=int)
    for site in genotype_matrix:
        j = int(site.sum())  # derived allele count at this site
        sfs[j] += 1          # increment the histogram bin for count j
    return sfs


def compute_joint_sfs(genotypes_pop1, genotypes_pop2, n1, n2):
    """Compute the joint SFS for two populations.

    Parameters
    ----------
    genotypes_pop1 : ndarray of shape (L, n1)
        Binary genotype matrix for population 1.
    genotypes_pop2 : ndarray of shape (L, n2)
        Binary genotype matrix for population 2.
    n1, n2 : int
        Sample sizes.

    Returns
    -------
    sfs : ndarray of shape (n1+1, n2+1)
        Joint SFS: sfs[j1, j2] = number of sites with
        j1 derived in pop1 and j2 derived in pop2.
    """
    L = genotypes_pop1.shape[0]
    sfs = np.zeros((n1 + 1, n2 + 1), dtype=int)
    for i in range(L):
        j1 = int(genotypes_pop1[i].sum())  # derived count in pop 1
        j2 = int(genotypes_pop2[i].sum())  # derived count in pop 2
        sfs[j1, j2] += 1                   # increment the 2D histogram
    return sfs


def expected_sfs_neutral(n, theta=1.0):
    """Expected SFS under the standard neutral model.

    Parameters
    ----------
    n : int
        Haploid sample size.
    theta : float
        Population-scaled mutation rate (4*Ne*mu*L).

    Returns
    -------
    sfs : ndarray of shape (n+1,)
        Expected counts. sfs[0] and sfs[n] are 0 (no fixed sites
        under the infinite-sites model with theta as total rate).
    """
    sfs = np.zeros(n + 1)
    for j in range(1, n):
        sfs[j] = theta / j  # the 1/j law: each bin j gets theta/j expected sites
    return sfs


def fold_sfs(sfs):
    """Fold an unfolded SFS into a minor allele frequency spectrum.

    Parameters
    ----------
    sfs : ndarray of shape (n+1,)
        Unfolded SFS.

    Returns
    -------
    folded : ndarray of shape (n//2 + 1,)
        Folded SFS. folded[0] is unused (monomorphic).
    """
    n = len(sfs) - 1
    folded = np.zeros(n // 2 + 1)
    for j in range(1, n // 2 + 1):
        if j == n - j:
            folded[j] = sfs[j]         # at the midpoint, don't double-count
        else:
            folded[j] = sfs[j] + sfs[n - j]  # sum the mirror bins
    return folded


def project_sfs(sfs, n_new):
    """Project an SFS to a smaller sample size.

    Parameters
    ----------
    sfs : ndarray of shape (n+1,)
        Original SFS with sample size n.
    n_new : int
        Target sample size (n_new < n).

    Returns
    -------
    projected : ndarray of shape (n_new+1,)
        Projected SFS.
    """
    n = len(sfs) - 1
    projected = np.zeros(n_new + 1)

    for j in range(0, n + 1):
        if sfs[j] == 0:
            continue
        for j_new in range(max(0, j - (n - n_new)), min(j, n_new) + 1):
            # Hypergeometric probability of drawing j_new derived
            # from j derived and n-j ancestral, sample size n_new
            prob = (comb(j, j_new, exact=True) *
                    comb(n - j, n_new - j_new, exact=True) /
                    comb(n, n_new, exact=True))
            projected[j_new] += sfs[j] * prob  # accumulate weighted counts

    return projected


def watterson_theta(sfs):
    """Watterson's estimator of theta from an SFS."""
    n = len(sfs) - 1
    S = sfs[1:n].sum()                    # total number of segregating sites
    a_n = sum(1 / j for j in range(1, n))  # (n-1)-th harmonic number
    return S / a_n                          # unbiased estimator of theta


def nucleotide_diversity(sfs):
    """Compute nucleotide diversity (pi) from an SFS."""
    n = len(sfs) - 1
    pi = 0.0
    for j in range(1, n):
        pi += j * (n - j) * sfs[j]  # j*(n-j) = number of differing pairs at this site
    pi /= n * (n - 1) / 2  # divide by total number of pairs = binom(n,2)
    return pi


def tajimas_d(sfs):
    """Compute Tajima's D from an SFS.

    Uses the standard normalization from Tajima (1989).
    """
    n = len(sfs) - 1
    S = sfs[1:n].sum()       # total segregating sites
    if S == 0:
        return 0.0

    pi = nucleotide_diversity(sfs)  # pairwise-difference estimator of theta
    theta_w = watterson_theta(sfs)  # segregating-sites estimator of theta

    # Tajima's normalization constants (derived from coalescent variance)
    a1 = sum(1 / i for i in range(1, n))
    a2 = sum(1 / i**2 for i in range(1, n))

    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))

    c1 = b1 - 1 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / a1**2

    e1 = c1 / a1
    e2 = c2 / (a1**2 + a2)

    var = e1 * S + e2 * S * (S - 1)  # variance of pi - theta_W under neutrality
    if var <= 0:
        return 0.0

    return (pi - theta_w) / np.sqrt(var)  # standardized test statistic


# ===========================================================================
# Chapter: The Moment Equations
# ===========================================================================

def drift_operator(phi, n):
    """Compute the drift contribution to d(phi)/dt.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS (phi[j] = expected count at frequency j/n).
    n : int
        Sample size.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
        Change in SFS due to drift.
    """
    dphi = np.zeros(n + 1)
    for j in range(1, n):
        term_down = (j - 1) * (n - j + 1) * phi[j - 1] if j >= 1 else 0.0
        term_stay = -2 * j * (n - j) * phi[j]
        term_up = (j + 1) * (n - j - 1) * phi[j + 1] if j < n else 0.0
        dphi[j] = (term_down + term_stay + term_up) / (2.0 * n)
    return dphi


def mutation_operator(phi, n, theta):
    """Compute the mutation contribution to d(phi)/dt.

    Under the infinite-sites model, mutations only add singletons.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS.
    n : int
        Sample size.
    theta : float
        Population-scaled mutation rate (4*Ne*mu, per site or total).

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    dphi = np.zeros(n + 1)
    dphi[1] = theta / 2.0
    return dphi


def selection_operator(phi, n, gamma, h=0.5):
    """Compute the selection contribution to d(phi)/dt.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
        Current SFS.
    n : int
        Sample size.
    gamma : float
        Scaled selection coefficient (2*Ne*s).
    h : float
        Dominance coefficient.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    dphi = np.zeros(n + 1)

    # First-order Jackknife: approximate Phi_{n+1}(i) from Phi_n by
    # frequency-matched linear interpolation (Jouganous et al. 2017 App. D).
    def _jk13(i):
        a = phi[i] if 0 < i < n else 0.0
        b = phi[i - 1] if 0 < (i - 1) < n else 0.0
        return ((n + 1 - i) * a + i * b) / (n + 1)

    for j in range(1, n):
        # Additive selection (Jouganous et al. 2017 Eq. A3)
        term1 = gamma * h / (n + 1) * (
            j * (n + 1 - j) * _jk13(j)
            - (j + 1) * (n - j) * _jk13(j + 1)
        )
        # Dominance deviation (Phi_{n+2} via identity Jackknife;
        # vanishes for additive selection h=0.5)
        if (1 - 2 * h) != 0 and n > 1:
            phi_jp1 = phi[j + 1] if j + 1 < n else 0.0
            phi_jp2 = phi[j + 2] if j + 2 < n else 0.0
            term2 = gamma * (1 - 2 * h) * (j + 1) / ((n + 1) * (n + 2)) * (
                j * (n + 1 - j) * phi_jp1
                - (j + 2) * (n - j) * phi_jp2
            )
        else:
            term2 = 0.0
        dphi[j] = term1 + term2
    return dphi


def migration_operator_2pop(phi_2d, n1, n2, M12, M21):
    """Compute migration contribution for a 2D SFS.

    Parameters
    ----------
    phi_2d : ndarray of shape (n1+1, n2+1)
        Joint SFS.
    n1, n2 : int
        Sample sizes.
    M12 : float
        Scaled migration rate from pop 2 into pop 1.
    M21 : float
        Scaled migration rate from pop 1 into pop 2.

    Returns
    -------
    dphi : ndarray of shape (n1+1, n2+1)
    """
    dphi = np.zeros((n1 + 1, n2 + 1))
    for j1 in range(1, n1):
        for j2 in range(1, n2):
            # Migration from pop 2 into pop 1 (rate M12):
            # A derived allele moves from pop 2 to pop 1: gain from (j1-1, j2+1)
            if j1 > 0 and j2 < n2:
                dphi[j1, j2] += M12 * (j2 + 1) / n2 * phi_2d[j1 - 1, j2 + 1]
            # Loss: a derived allele in pop 1 could migrate out
            dphi[j1, j2] -= M12 * j1 / n1 * phi_2d[j1, j2]

            # Migration from pop 1 into pop 2 (rate M21):
            # A derived allele moves from pop 1 to pop 2: gain from (j1+1, j2-1)
            if j2 > 0 and j1 < n1:
                dphi[j1, j2] += M21 * (j1 + 1) / n1 * phi_2d[j1 + 1, j2 - 1]
            # Loss: a derived allele in pop 2 could migrate out
            dphi[j1, j2] -= M21 * j2 / n2 * phi_2d[j1, j2]
    return dphi


def drift_operator_with_size(phi, n, nu):
    """Drift operator scaled by relative population size.

    Parameters
    ----------
    phi : ndarray of shape (n+1,)
    n : int
    nu : float
        Relative population size (N/N_ref). nu > 1 = expansion.

    Returns
    -------
    dphi : ndarray of shape (n+1,)
    """
    return drift_operator(phi, n) / nu


def integrate_sfs(phi_init, n, T, nu_func, theta, gamma=0, h=0.5):
    """Integrate the SFS forward through time.

    Parameters
    ----------
    phi_init : ndarray of shape (n+1,)
        Initial SFS.
    n : int
        Sample size.
    T : float
        Integration time (in 2*Ne generations).
    nu_func : callable
        nu_func(t) returns the relative population size at time t.
    theta : float
        Scaled mutation rate.
    gamma : float
        Scaled selection coefficient.
    h : float
        Dominance coefficient.

    Returns
    -------
    phi_final : ndarray of shape (n+1,)
        SFS after integration.
    """
    y0 = phi_init[1:n].copy()

    def rhs(t, y):
        phi = np.zeros(n + 1)
        phi[1:n] = y
        nu = nu_func(t)
        d_drift = drift_operator(phi, n) / nu
        d_mutation = mutation_operator(phi, n, theta)
        d_selection = selection_operator(phi, n, gamma, h) if gamma != 0 else np.zeros(n + 1)
        dphi = d_drift + d_mutation + d_selection
        return dphi[1:n]

    method = 'Radau' if gamma != 0 else 'RK45'
    sol = solve_ivp(rhs, [0, T], y0, method=method,
                     rtol=1e-10, atol=1e-12,
                     dense_output=True)
    phi_final = np.zeros(n + 1)
    phi_final[1:n] = sol.y[:, -1]
    return phi_final


def split_1d_to_2d(phi_1d, n1, n2):
    """Split a 1D SFS into a 2D joint SFS.

    Parameters
    ----------
    phi_1d : ndarray of shape (n+1,)
        1D SFS with n = n1 + n2.
    n1, n2 : int
        Sample sizes for the two daughter populations.

    Returns
    -------
    phi_2d : ndarray of shape (n1+1, n2+1)
    """
    n = n1 + n2
    phi_2d = np.zeros((n1 + 1, n2 + 1))
    for j in range(n + 1):
        if phi_1d[j] == 0:
            continue
        for j1 in range(max(0, j - n2), min(j, n1) + 1):
            j2 = j - j1
            if j2 < 0 or j2 > n2:
                continue
            prob = (comb(j, j1, exact=True) *
                    comb(n - j, n1 - j1, exact=True) /
                    comb(n, n1, exact=True))
            phi_2d[j1, j2] = phi_1d[j] * prob
    return phi_2d


# ===========================================================================
# Chapter: Demographic Inference
# ===========================================================================

def poisson_log_likelihood(data_sfs, model_sfs):
    """Compute the Poisson log-likelihood of data given model.

    Parameters
    ----------
    data_sfs : ndarray
        Observed SFS (counts).
    model_sfs : ndarray
        Expected SFS under the model.

    Returns
    -------
    ll : float
        Log-likelihood (up to a constant).
    """
    n = len(data_sfs) - 1
    ll = 0.0
    for j in range(1, n):
        if model_sfs[j] <= 0:
            if data_sfs[j] > 0:
                return -np.inf  # impossible observation => -infinity
            continue
        if data_sfs[j] > 0:
            ll += data_sfs[j] * np.log(model_sfs[j])  # D_j * ln(M_j)
        ll -= model_sfs[j]                              # - M_j
    return ll


def optimal_theta_scaling(data_sfs, model_sfs_unit):
    """Find the optimal theta to scale the model SFS.

    Parameters
    ----------
    data_sfs : ndarray
        Observed SFS.
    model_sfs_unit : ndarray
        Model SFS computed at theta = 1.

    Returns
    -------
    theta_opt : float
    """
    n = len(data_sfs) - 1
    S_data = data_sfs[1:n].sum()       # total observed segregating sites
    S_model = model_sfs_unit[1:n].sum()  # total expected at theta=1
    return S_data / S_model if S_model > 0 else 1.0


def fisher_information_numerical(params, data_sfs, model_func, ns, eps=0.01):
    """Compute the Fisher Information Matrix by numerical differentiation.

    Parameters
    ----------
    params : array-like
        Optimized parameters.
    data_sfs : ndarray
        Observed SFS.
    model_func : callable
        Function(params, ns) -> model SFS.
    ns : list
        Sample sizes.
    eps : float
        Relative step size for finite differences.

    Returns
    -------
    FIM : ndarray of shape (k, k)
        Fisher Information Matrix.
    """
    params = np.array(params, dtype=float)
    k = len(params)
    FIM = np.zeros((k, k))

    def neg_ll(p):
        model = model_func(p, ns)
        theta_opt = optimal_theta_scaling(data_sfs, model)
        model_scaled = model * theta_opt
        return -poisson_log_likelihood(data_sfs, model_scaled)

    # Central differences for second derivatives
    for i in range(k):
        for j in range(i, k):
            p_pp = params.copy(); p_pp[i] *= (1 + eps); p_pp[j] *= (1 + eps)
            p_pm = params.copy(); p_pm[i] *= (1 + eps); p_pm[j] *= (1 - eps)
            p_mp = params.copy(); p_mp[i] *= (1 - eps); p_mp[j] *= (1 + eps)
            p_mm = params.copy(); p_mm[i] *= (1 - eps); p_mm[j] *= (1 - eps)

            d2 = (neg_ll(p_pp) - neg_ll(p_pm) - neg_ll(p_mp) + neg_ll(p_mm))
            d2 /= (params[i] * eps * 2) * (params[j] * eps * 2)

            FIM[i, j] = d2
            FIM[j, i] = d2  # symmetric

    return FIM


def godambe_uncertainty(params_opt, data_sfs, model_func, ns,
                         bootstrap_sfss, eps=0.01):
    """Compute parameter uncertainties using the Godambe Information Matrix.

    Parameters
    ----------
    params_opt : array-like
        Optimized parameters.
    data_sfs : ndarray
        Full observed SFS.
    model_func : callable
    ns : list
    bootstrap_sfss : list of ndarray
        SFS from bootstrap resampling of genomic blocks.
    eps : float
        Step size for numerical derivatives.

    Returns
    -------
    se_godambe : ndarray
        Standard errors from GIM.
    """
    params_opt = np.array(params_opt, dtype=float)
    k = len(params_opt)

    # H: Hessian (= FIM) from the full data
    H = fisher_information_numerical(params_opt, data_sfs, model_func, ns, eps)

    # Score function: gradient of log-likelihood at the MLE
    def score(p, data):
        grad = np.zeros(k)
        for i in range(k):
            p_plus = p.copy(); p_plus[i] *= (1 + eps)
            p_minus = p.copy(); p_minus[i] *= (1 - eps)

            model_p = model_func(p_plus, ns)
            model_m = model_func(p_minus, ns)
            theta_p = optimal_theta_scaling(data, model_p)
            theta_m = optimal_theta_scaling(data, model_m)

            ll_p = poisson_log_likelihood(data, model_p * theta_p)
            ll_m = poisson_log_likelihood(data, model_m * theta_m)

            grad[i] = (ll_p - ll_m) / (p[i] * 2 * eps)
        return grad

    # J: empirical variance of the score across bootstraps
    scores = np.array([score(params_opt, bs) for bs in bootstrap_sfss])
    J = np.cov(scores, rowvar=False) * len(bootstrap_sfss)

    # GIM = H^{-1} J H^{-1}  (sandwich estimator)
    H_inv = np.linalg.inv(H)
    GIM = H_inv @ J @ H_inv

    return np.sqrt(np.diag(GIM))


def likelihood_ratio_test(ll_simple, ll_complex, df):
    """Likelihood ratio test for nested models.

    Parameters
    ----------
    ll_simple : float
        Log-likelihood of the simpler model.
    ll_complex : float
        Log-likelihood of the more complex model.
    df : int
        Difference in number of free parameters.

    Returns
    -------
    p_value : float
    """
    lr = 2 * (ll_complex - ll_simple)  # likelihood ratio statistic
    p_value = 1 - chi2.cdf(lr, df)      # p-value from chi-squared distribution
    return p_value


def apply_misidentification(sfs, p_misid):
    """Apply ancestral misidentification to an SFS.

    Parameters
    ----------
    sfs : ndarray of shape (n+1,)
    p_misid : float
        Fraction of sites with ancestral/derived labels swapped.

    Returns
    -------
    sfs_obs : ndarray of shape (n+1,)
    """
    n = len(sfs) - 1
    sfs_obs = np.zeros(n + 1)
    for j in range(n + 1):
        sfs_obs[j] = (1 - p_misid) * sfs[j] + p_misid * sfs[n - j]
    return sfs_obs


# ===========================================================================
# Chapter: Linkage Disequilibrium
# ===========================================================================

def compute_D(haplotypes):
    """Compute D for two biallelic loci from haplotype data.

    Parameters
    ----------
    haplotypes : ndarray of shape (n, 2)
        Each row is a haplotype. Columns are the two loci (0/1).

    Returns
    -------
    D : float
    """
    n = len(haplotypes)
    p = haplotypes[:, 0].mean()  # frequency of allele 1 at locus A
    q = haplotypes[:, 1].mean()  # frequency of allele 1 at locus B
    x11 = ((haplotypes[:, 0] == 1) & (haplotypes[:, 1] == 1)).mean()
    D = x11 - p * q  # departure from independence
    return D


def ld_decay_deterministic(D0, r, t_generations):
    """Deterministic LD decay over t generations.

    Parameters
    ----------
    D0 : float
        Initial D.
    r : float
        Recombination rate between loci.
    t_generations : int
        Number of generations.

    Returns
    -------
    D_t : float
    """
    return D0 * (1 - r) ** t_generations  # exponential decay


def compute_ld_statistics(haplotype_matrix):
    """Compute E[D^2], E[Dz], and pi_2 from a haplotype matrix.

    Parameters
    ----------
    haplotype_matrix : ndarray of shape (n_haplotypes, n_loci)
        Binary matrix (0/1) for each locus.

    Returns
    -------
    D2_mean, Dz_mean, pi2_mean : float
        Average D^2, Dz, and pi_2 over all pairs of loci.
    """
    n_haps, n_loci = haplotype_matrix.shape
    D2_sum, Dz_sum, pi2_sum = 0.0, 0.0, 0.0
    n_pairs = 0

    for i in range(n_loci):
        for j in range(i + 1, n_loci):
            p = haplotype_matrix[:, i].mean()  # allele freq at locus i
            q = haplotype_matrix[:, j].mean()  # allele freq at locus j

            # Skip monomorphic loci (no LD can be computed)
            if p == 0 or p == 1 or q == 0 or q == 1:
                continue

            # Compute D = freq(1,1) - p*q
            x11 = ((haplotype_matrix[:, i] == 1) &
                    (haplotype_matrix[:, j] == 1)).mean()
            D = x11 - p * q

            D2_sum += D ** 2                         # squared LD
            Dz_sum += D * (1 - 2 * p) * (1 - 2 * q)  # admixture-sensitive statistic
            pi2_sum += p * (1 - p) * q * (1 - q)      # product of heterozygosities
            n_pairs += 1

    if n_pairs == 0:
        return 0, 0, 0

    return D2_sum / n_pairs, Dz_sum / n_pairs, pi2_sum / n_pairs


def ld_equilibrium(theta, rho, n_pops=1):
    """Compute equilibrium LD statistics for one population.

    At equilibrium, the rate of LD creation by drift equals the rate
    of LD decay by recombination.

    Parameters
    ----------
    theta : float
        Scaled mutation rate (4*Ne*mu per site).
    rho : float
        Scaled recombination rate (4*Ne*r between the two loci).
    n_pops : int
        Number of populations (unused, kept for API compatibility).

    Returns
    -------
    sigma_d2 : float
        Equilibrium sigma_d^2 = E[D^2] / pi_2.
    """
    return 1.0 / (1.0 + rho)


def gaussian_composite_ll(data_ld, model_ld, varcov_matrices):
    """Compute Gaussian composite log-likelihood for LD statistics.

    Parameters
    ----------
    data_ld : list of ndarray
        Observed LD statistics, one array per recombination bin.
    model_ld : list of ndarray
        Model-predicted LD statistics.
    varcov_matrices : list of ndarray
        Variance-covariance matrix for each bin (from bootstrap).

    Returns
    -------
    ll : float
    """
    ll = 0.0
    for d, mu, sigma in zip(data_ld, model_ld, varcov_matrices):
        residual = d - mu                    # data minus model prediction
        sigma_inv = np.linalg.inv(sigma)      # precision matrix
        ll -= 0.5 * residual @ sigma_inv @ residual  # quadratic form
    return ll


def map_r_bins_to_rho(r_bins, Ne):
    """Convert physical recombination bins to scaled rho values.

    Parameters
    ----------
    r_bins : ndarray
        Physical recombination rates (per generation).
    Ne : float
        Effective population size.

    Returns
    -------
    rho_bins : ndarray
        Scaled recombination rates (4*Ne*r).
    """
    return 4 * Ne * r_bins  # the conversion that links LD to absolute N_e


# ===========================================================================
# Helper: neutral SFS with theta/j shape
# ===========================================================================

def _neutral_sfs(n, theta=1.0):
    """Build the standard neutral SFS: phi[j] = theta / j for j in 1..n-1."""
    phi = np.zeros(n + 1)
    for j in range(1, n):
        phi[j] = theta / j
    return phi


# ===========================================================================
# Demo function
# ===========================================================================

def demo():
    """Demonstrate the key algorithms from the moments framework."""

    print("=" * 70)
    print("Mini-moments: Demographic Inference from the Frequency Spectrum")
    print("=" * 70)

    # --- 1. Expected neutral SFS ---
    print("\n--- 1. Expected Neutral SFS ---")
    n = 50
    theta = 1000
    sfs_neutral = expected_sfs_neutral(n, theta)

    print("Expected neutral SFS (first 10 entries):")
    for j in range(1, 11):
        print(f"  SFS[{j:2d}] = {sfs_neutral[j]:.1f}")

    total_seg = sfs_neutral[1:n].sum()
    harmonic = sum(1 / j for j in range(1, n))
    print(f"\nTotal segregating sites: {total_seg:.1f}")
    print(f"theta * H_(n-1) = {theta * harmonic:.1f}")
    print(f"Match: {np.isclose(total_seg, theta * harmonic)}")

    # --- 2. Summary statistics ---
    print("\n--- 2. Summary Statistics ---")
    n = 50
    theta = 1000
    sfs = expected_sfs_neutral(n, theta)
    print(f"theta (true):     {theta:.2f}")
    print(f"theta_W:          {watterson_theta(sfs):.2f}")
    print(f"pi:               {nucleotide_diversity(sfs):.2f}")

    # --- 3. Projection ---
    print("\n--- 3. Projection ---")
    n = 50
    theta = 500
    sfs_50 = expected_sfs_neutral(n, theta)
    sfs_20 = project_sfs(sfs_50, 20)
    sfs_20_direct = expected_sfs_neutral(20, theta)

    print("Projected vs. direct computation (first 10 entries):")
    print(f"{'j':>3} {'Projected':>12} {'Direct':>12} {'Match':>8}")
    for j in range(1, 11):
        match = np.isclose(sfs_20[j], sfs_20_direct[j], rtol=1e-10)
        print(f"{j:3d} {sfs_20[j]:12.4f} {sfs_20_direct[j]:12.4f} "
              f"{'Y' if match else 'N':>8}")

    # --- 4. Folding ---
    print("\n--- 4. Folding ---")
    n = 10
    theta = 100
    sfs = expected_sfs_neutral(n, theta)
    folded = fold_sfs(sfs)

    print("Unfolded SFS:")
    for j in range(1, n):
        print(f"  SFS[{j}] = {sfs[j]:.2f}")
    print("\nFolded SFS:")
    for j in range(1, n // 2 + 1):
        print(f"  SFS_folded[{j}] = {folded[j]:.2f}")

    # --- 5. Drift operator ---
    print("\n--- 5. Drift Operator ---")
    n = 20
    theta = 1.0
    phi_neutral = expected_sfs_neutral(n, theta)
    dphi_drift = drift_operator(phi_neutral, n)

    print("Drift operator on neutral SFS (should be ~ 0 after adding mutation):")
    for j in range(1, min(n, 8)):
        print(f"  d(phi[{j}])/dt|_drift = {dphi_drift[j]:+.6f}")

    # --- 6. ODE integration ---
    print("\n--- 6. ODE Integration: Neutral Equilibrium Stability ---")
    n = 20
    theta = 1.0
    phi_eq = expected_sfs_neutral(n, theta)
    phi_after = integrate_sfs(phi_eq, n, T=1.0,
                              nu_func=lambda t: 1.0, theta=theta)

    print(f"{'j':>3} {'Before':>10} {'After':>10} {'Diff':>12}")
    for j in range(1, 6):
        print(f"{j:3d} {phi_eq[j]:10.6f} {phi_after[j]:10.6f} "
              f"{phi_after[j] - phi_eq[j]:12.2e}")

    # --- 7. Population expansion ---
    print("\n--- 7. Population Expansion (10x for T=0.5) ---")
    phi_expanded = integrate_sfs(phi_eq, n, T=0.5,
                                  nu_func=lambda t: 10.0, theta=theta)

    print(f"{'j':>3} {'Neutral':>10} {'Expanded':>10} {'Ratio':>8}")
    for j in range(1, 8):
        ratio = phi_expanded[j] / phi_eq[j] if phi_eq[j] > 0 else float('inf')
        print(f"{j:3d} {phi_eq[j]:10.6f} {phi_expanded[j]:10.6f} "
              f"{ratio:8.3f}")
    print("(Ratio > 1 for small j = excess rare variants)")

    # --- 8. Population split ---
    print("\n--- 8. Population Split ---")
    n1, n2 = 8, 12
    n_total = n1 + n2
    phi_1d = expected_sfs_neutral(n_total, theta=1.0)
    phi_2d = split_1d_to_2d(phi_1d, n1, n2)

    phi_marginal_1 = phi_2d.sum(axis=1)
    phi_expected_1 = project_sfs(phi_1d, n1)

    print("Verify: split then marginalize = project")
    for j in range(1, min(n1, 6)):
        print(f"  j={j}: marginal={phi_marginal_1[j]:.6f}, "
              f"projected={phi_expected_1[j]:.6f}, "
              f"match={np.isclose(phi_marginal_1[j], phi_expected_1[j])}")

    # --- 9. Poisson likelihood ---
    print("\n--- 9. Poisson Likelihood ---")
    n = 20
    theta_true = 1000
    data = expected_sfs_neutral(n, theta_true)

    print("Log-likelihood at different theta values:")
    for theta_test in [500, 800, 1000, 1200, 1500]:
        model = expected_sfs_neutral(n, theta_test)
        ll = poisson_log_likelihood(data, model)
        marker = " <-- true value" if theta_test == theta_true else ""
        print(f"  theta = {theta_test:5d}: ll = {ll:10.2f}{marker}")

    # --- 10. Optimal theta scaling ---
    print("\n--- 10. Optimal Theta Scaling ---")
    model_unit = expected_sfs_neutral(n, theta=1.0)
    theta_opt = optimal_theta_scaling(data, model_unit)
    print(f"Optimal theta: {theta_opt:.2f} (true: {theta_true})")

    # --- 11. LD equilibrium ---
    print("\n--- 11. LD Equilibrium ---")
    rho_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"{'rho':>6} {'sigma_d2 (approx)':>18}")
    for rho in rho_values:
        sigma_d2 = ld_equilibrium(0.001, rho)
        print(f"{rho:6.1f} {sigma_d2:18.6f}")

    # --- 12. LD decay ---
    print("\n--- 12. LD Decay ---")
    r = 0.01
    half_life = np.log(2) / r
    print(f"Half-life of LD at r={r}: {half_life:.0f} generations")
    D0 = 0.1
    D_69 = ld_decay_deterministic(D0, r, 69)
    D_500 = ld_decay_deterministic(D0, r, 500)
    print(f"D at t=0: {D0:.4f}")
    print(f"D at t=69: {D_69:.4f} (approximately half-life)")
    print(f"D at t=500: {D_500:.6f}")

    # --- 13. Misidentification ---
    print("\n--- 13. Ancestral Misidentification ---")
    n = 20
    sfs_true = expected_sfs_neutral(n, theta=1000)
    sfs_misid = apply_misidentification(sfs_true, 0.02)

    print("Effect of 2% ancestral misidentification:")
    print(f"{'j':>3} {'True':>10} {'Observed':>10} {'Diff%':>8}")
    for j in range(1, 6):
        diff_pct = (sfs_misid[j] - sfs_true[j]) / sfs_true[j] * 100
        print(f"{j:3d} {sfs_true[j]:10.2f} {sfs_misid[j]:10.2f} "
              f"{diff_pct:+7.2f}%")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
