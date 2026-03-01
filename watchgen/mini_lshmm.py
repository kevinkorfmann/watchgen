"""
Mini Li & Stephens HMM -- a self-contained implementation of the Li & Stephens
Hidden Markov Model for haplotype copying.

The Li & Stephens HMM (Li and Stephens, 2003) models a query haplotype as an
imperfect mosaic of reference haplotypes.  At each genomic site the query
"copies" its allele from one of *n* reference haplotypes.  Between sites a
recombination event may switch the copying source, and at each site a mutation
may change the copied allele.

This module provides:

  Copying-model primitives
  ~~~~~~~~~~~~~~~~~~~~~~~~
  - ``initial_distribution``       -- uniform prior over reference haplotypes
  - ``transition_matrix``          -- Li-Stephens transition matrix
  - ``compute_recombination_probs``-- per-site recombination probabilities
  - ``emission_probability``       -- single-site emission (match / mismatch)
  - ``emission_matrix_haploid``    -- emission matrix for the haploid case
  - ``emission_prob_with_specials``-- emission handling NONCOPY / MISSING
  - ``estimate_mutation_probability`` -- Li-Stephens mutation estimator
  - ``forward_step_naive``         -- naive O(n^2) forward step
  - ``forward_step_fast``          -- O(n) forward step (Li-Stephens trick)
  - ``forward_ls_haploid``         -- complete forward algorithm (simple API)

  Haploid algorithms
  ~~~~~~~~~~~~~~~~~~
  - ``forwards_ls_hap``            -- forward algorithm (lshmm-style API)
  - ``backwards_ls_hap``           -- backward algorithm
  - ``posterior_decoding``         -- posterior decoding from forward-backward
  - ``forwards_viterbi_hap``       -- Viterbi algorithm
  - ``backwards_viterbi_hap``      -- Viterbi traceback
  - ``path_loglik_hap``            -- log-likelihood of a specific path

  Diploid extension
  ~~~~~~~~~~~~~~~~~
  - ``diploid_transition_prob``    -- diploid transition probability
  - ``emission_matrix_diploid``    -- diploid emission matrix
  - ``genotype_comparison_index``  -- map genotype pair to emission index
  - ``build_genotype_matrix``      -- reference genotype matrix from panel
  - ``forward_diploid``            -- diploid forward algorithm
  - ``viterbi_diploid``            -- diploid Viterbi algorithm
  - ``backwards_viterbi_diploid``  -- diploid Viterbi traceback
  - ``get_phased_path``            -- convert flat diploid path to two paths

References
----------
Li, N. and Stephens, M. (2003).  Modeling linkage disequilibrium and
identifying recombination hotspots using single-nucleotide polymorphism data.
*Genetics*, 165(4), 2213-2233.
"""

import numpy as np


# ============================================================================
# Copying-model primitives  (from copying_model.rst)
# ============================================================================

def initial_distribution(n):
    """Uniform initial distribution over n reference haplotypes.

    Parameters
    ----------
    n : int
        Number of reference haplotypes.

    Returns
    -------
    pi : ndarray of shape (n,)
        Initial state probabilities.
    """
    return np.ones(n) / n


def transition_matrix(n, r):
    """Build the Li-Stephens transition matrix.

    Parameters
    ----------
    n : int
        Number of reference haplotypes.
    r : float
        Recombination probability between adjacent sites.

    Returns
    -------
    A : ndarray of shape (n, n)
        Transition matrix.
    """
    A = np.full((n, n), r / n)
    np.fill_diagonal(A, (1 - r) + r / n)
    return A


def compute_recombination_probs(rho, n):
    """Compute per-site recombination probabilities.

    Parameters
    ----------
    rho : ndarray of shape (m,)
        Population-scaled recombination rate at each site.
    n : int
        Number of reference haplotypes.

    Returns
    -------
    r : ndarray of shape (m,)
        Per-site recombination probability (r[0] = 0 by convention).
    """
    r = 1 - np.exp(-rho / n)
    r[0] = 0.0
    return r


def emission_probability(query_allele, ref_allele, mu):
    """Compute emission probability for one site.

    Parameters
    ----------
    query_allele : int
        Allele in the query haplotype (0 or 1).
    ref_allele : int
        Allele in the reference haplotype (0 or 1).
    mu : float
        Mutation probability.

    Returns
    -------
    prob : float
    """
    if query_allele == ref_allele:
        return 1 - mu
    else:
        return mu


def emission_matrix_haploid(mu, num_sites, num_alleles):
    """Compute the emission probability matrix for the haploid case.

    Parameters
    ----------
    mu : float or ndarray of shape (m,)
        Per-site mutation probability.
    num_sites : int
        Number of sites.
    num_alleles : ndarray of shape (m,)
        Number of distinct alleles at each site.

    Returns
    -------
    e : ndarray of shape (m, 2)
        Column 0 = mismatch probability, column 1 = match probability.
    """
    if isinstance(mu, float):
        mu = np.full(num_sites, mu)

    e = np.zeros((num_sites, 2))
    for i in range(num_sites):
        if num_alleles[i] == 1:
            e[i, 0] = 0.0
            e[i, 1] = 1.0
        else:
            e[i, 0] = mu[i] / (num_alleles[i] - 1)
            e[i, 1] = 1 - mu[i]

    return e


def emission_prob_with_specials(ref_allele, query_allele, site, emission_matrix):
    """Compute emission probability handling NONCOPY and MISSING.

    Parameters
    ----------
    ref_allele : int
        Allele in the reference (-2 = NONCOPY).
    query_allele : int
        Allele in the query (-1 = MISSING).
    site : int
        Site index.
    emission_matrix : ndarray of shape (m, 2)
        Emission probabilities.

    Returns
    -------
    prob : float
    """
    NONCOPY = -2
    MISSING = -1

    if ref_allele == NONCOPY:
        return 0.0
    elif query_allele == MISSING:
        return 1.0
    else:
        if ref_allele == query_allele:
            return emission_matrix[site, 1]
        else:
            return emission_matrix[site, 0]


def estimate_mutation_probability(n):
    """Estimate mutation probability from the number of haplotypes.

    Based on Li & Stephens (2003), equations A2 and A3.

    Parameters
    ----------
    n : int
        Number of reference haplotypes (must be >= 3).

    Returns
    -------
    mu : float
        Estimated per-site mutation probability.
    """
    if n < 3:
        raise ValueError("Need at least 3 haplotypes.")
    theta_tilde = 1.0 / sum(1.0 / k for k in range(1, n - 1))
    mu = 0.5 * theta_tilde / (n + theta_tilde)
    return mu


def forward_step_naive(alpha_prev, A, emission, n):
    """Naive O(n^2) forward step -- for comparison only.

    For each state j, we sum over all n previous states i,
    multiplying by the transition probability A[i, j].
    """
    alpha = np.zeros(n)
    for j in range(n):
        alpha[j] = emission[j] * np.sum(alpha_prev * A[:, j])
    return alpha


def forward_step_fast(alpha_prev, r, r_n, emission, n):
    """O(n) forward step using Li-Stephens structure.

    Parameters
    ----------
    alpha_prev : ndarray of shape (n,)
        Forward probabilities at previous site.
    r : float
        Recombination probability at this site.
    r_n : float
        r / n (or r / n_copiable for NONCOPY support).
    emission : ndarray of shape (n,)
        Emission probabilities at this site.

    Returns
    -------
    alpha : ndarray of shape (n,)
    """
    alpha = np.zeros(n)
    for j in range(n):
        alpha[j] = alpha_prev[j] * (1 - r) + r_n
        alpha[j] *= emission[j]
    return alpha


def forward_ls_haploid(H, s, mu, r, normalize=True):
    """Complete forward algorithm for the haploid Li-Stephens model.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Reference panel (m sites, n haplotypes).
    s : ndarray of shape (m,)
        Query haplotype.
    mu : float
        Mutation probability.
    r : ndarray of shape (m,)
        Per-site recombination probability (r[0] should be 0).

    Returns
    -------
    F : ndarray of shape (m, n)
        Forward probabilities.
    c : ndarray of shape (m,)
        Scaling factors (c[l] = sum of unscaled F[l, :]).
    ll : float
        Log-likelihood (base 10).
    """
    m, n = H.shape
    F = np.zeros((m, n))
    c = np.zeros(m) if normalize else np.ones(m)
    r_n = r / n

    for j in range(n):
        if s[0] == H[0, j]:
            F[0, j] = (1 / n) * (1 - mu)
        else:
            F[0, j] = (1 / n) * mu

    if normalize:
        c[0] = F[0, :].sum()
        F[0, :] /= c[0]

    for l in range(1, m):
        if normalize:
            for j in range(n):
                F[l, j] = F[l - 1, j] * (1 - r[l]) + r_n[l]
                if s[l] == H[l, j]:
                    F[l, j] *= (1 - mu)
                else:
                    F[l, j] *= mu
            c[l] = F[l, :].sum()
            F[l, :] /= c[l]
        else:
            S = F[l - 1, :].sum()
            for j in range(n):
                F[l, j] = F[l - 1, j] * (1 - r[l]) + S * r_n[l]
                if s[l] == H[l, j]:
                    F[l, j] *= (1 - mu)
                else:
                    F[l, j] *= mu

    if normalize:
        ll = np.sum(np.log10(c))
    else:
        ll = np.log10(F[m - 1, :].sum())

    return F, c, ll


# ============================================================================
# Haploid algorithms  (from haploid_algorithms.rst)
# ============================================================================

def forwards_ls_hap(n, m, H, s, emission_matrix, r, norm=True):
    """Forward algorithm for the haploid Li-Stephens model.

    Parameters
    ----------
    n : int
        Number of reference haplotypes.
    m : int
        Number of sites.
    H : ndarray of shape (m, n)
        Reference panel.
    s : ndarray of shape (1, m)
        Query haplotype (wrapped in 2D array for API compatibility).
    emission_matrix : ndarray of shape (m, 2)
        Column 0 = mismatch prob, column 1 = match prob.
    r : ndarray of shape (m,)
        Per-site recombination probability.
    norm : bool
        Whether to normalize (scale) the forward probabilities.

    Returns
    -------
    F : ndarray of shape (m, n)
        Forward probabilities.
    c : ndarray of shape (m,)
        Scaling factors.
    ll : float
        Log-likelihood (base 10).
    """
    F = np.zeros((m, n))
    r_n = r / n

    if norm:
        c = np.zeros(m)
        for i in range(n):
            if H[0, i] == s[0, 0]:
                F[0, i] = (1 / n) * emission_matrix[0, 1]
            else:
                F[0, i] = (1 / n) * emission_matrix[0, 0]
            c[0] += F[0, i]
        for i in range(n):
            F[0, i] /= c[0]

        for l in range(1, m):
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + r_n[l]
                if H[l, i] == s[0, l]:
                    F[l, i] *= emission_matrix[l, 1]
                else:
                    F[l, i] *= emission_matrix[l, 0]
                c[l] += F[l, i]
            for i in range(n):
                F[l, i] /= c[l]

        ll = np.sum(np.log10(c))
    else:
        c = np.ones(m)
        for i in range(n):
            if H[0, i] == s[0, 0]:
                F[0, i] = (1 / n) * emission_matrix[0, 1]
            else:
                F[0, i] = (1 / n) * emission_matrix[0, 0]

        for l in range(1, m):
            S = np.sum(F[l - 1, :])
            for i in range(n):
                F[l, i] = F[l - 1, i] * (1 - r[l]) + S * r_n[l]
                if H[l, i] == s[0, l]:
                    F[l, i] *= emission_matrix[l, 1]
                else:
                    F[l, i] *= emission_matrix[l, 0]

        ll = np.log10(np.sum(F[m - 1, :]))

    return F, c, ll


def backwards_ls_hap(n, m, H, s, emission_matrix, c, r):
    """Backward algorithm for the haploid Li-Stephens model.

    Parameters
    ----------
    n, m, H, s, emission_matrix, r : same as forwards_ls_hap.
    c : ndarray of shape (m,)
        Scaling factors from the forward pass.

    Returns
    -------
    B : ndarray of shape (m, n)
        Scaled backward probabilities.
    """
    B = np.zeros((m, n))
    for i in range(n):
        B[m - 1, i] = 1.0

    r_n = r / n

    for l in range(m - 2, -1, -1):
        tmp_B = np.zeros(n)
        tmp_B_sum = 0.0
        for i in range(n):
            if H[l + 1, i] == s[0, l + 1]:
                emission_prob = emission_matrix[l + 1, 1]
            else:
                emission_prob = emission_matrix[l + 1, 0]
            tmp_B[i] = emission_prob * B[l + 1, i]
            tmp_B_sum += tmp_B[i]

        for i in range(n):
            B[l, i] = r_n[l + 1] * tmp_B_sum
            B[l, i] += (1 - r[l + 1]) * tmp_B[i]
            B[l, i] /= c[l + 1]

    return B


def posterior_decoding(F, B):
    """Compute posterior decoding from forward-backward probabilities.

    Parameters
    ----------
    F : ndarray of shape (m, n)
        Scaled forward probabilities.
    B : ndarray of shape (m, n)
        Scaled backward probabilities.

    Returns
    -------
    gamma : ndarray of shape (m, n)
        Posterior state probabilities.
    path : ndarray of shape (m,)
        Most likely state at each site (posterior decoding).
    """
    gamma = F * B
    gamma /= gamma.sum(axis=1, keepdims=True)
    path = np.argmax(gamma, axis=1)
    return gamma, path


def forwards_viterbi_hap(n, m, H, s, emission_matrix, r):
    """Viterbi algorithm for the haploid Li-Stephens model.

    Uses the Li-Stephens structure for O(n) per site.
    Includes rescaling for numerical stability.

    Parameters
    ----------
    n, m, H, s, emission_matrix, r : same as forwards_ls_hap.

    Returns
    -------
    V : ndarray of shape (n,)
        Viterbi probabilities at the last site.
    P : ndarray of shape (m, n), dtype int
        Pointer (traceback) array.
    ll : float
        Log-likelihood of the best path (base 10).
    """
    V = np.zeros(n)
    P = np.zeros((m, n), dtype=np.int64)
    r_n = r / n
    c = np.ones(m)

    for i in range(n):
        if H[0, i] == s[0, 0]:
            V[i] = (1 / n) * emission_matrix[0, 1]
        else:
            V[i] = (1 / n) * emission_matrix[0, 0]

    for j in range(1, m):
        argmax = np.argmax(V)
        c[j] = V[argmax]
        V /= c[j]

        for i in range(n):
            stay = V[i] * (1 - r[j] + r_n[j])
            switch = r_n[j]

            V[i] = stay
            P[j, i] = i
            if V[i] < switch:
                V[i] = switch
                P[j, i] = argmax

            if H[j, i] == s[0, j]:
                V[i] *= emission_matrix[j, 1]
            else:
                V[i] *= emission_matrix[j, 0]

    ll = np.sum(np.log10(c)) + np.log10(np.max(V))
    return V, P, ll


def backwards_viterbi_hap(m, V_last, P):
    """Traceback to find the most likely path.

    Parameters
    ----------
    m : int
        Number of sites.
    V_last : ndarray of shape (n,)
        Viterbi probabilities at the last site.
    P : ndarray of shape (m, n)
        Pointer array from the forward pass.

    Returns
    -------
    path : ndarray of shape (m,)
        Most likely state sequence.
    """
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, path[j + 1]]
    return path


def path_loglik_hap(n, m, H, path, s, emission_matrix, r):
    """Evaluate the log-likelihood of a specific copying path.

    Parameters
    ----------
    n, m, H, s, emission_matrix, r : same as forwards_ls_hap.
    path : ndarray of shape (m,)
        The copying path to evaluate.

    Returns
    -------
    ll : float
        Log-likelihood (base 10).
    """
    r_n = r / n

    if H[0, path[0]] == s[0, 0]:
        ll = np.log10((1 / n) * emission_matrix[0, 1])
    else:
        ll = np.log10((1 / n) * emission_matrix[0, 0])

    old = path[0]

    for l in range(1, m):
        current = path[l]
        if old == current:
            ll += np.log10((1 - r[l]) + r_n[l])
        else:
            ll += np.log10(r_n[l])

        if H[l, current] == s[0, l]:
            ll += np.log10(emission_matrix[l, 1])
        else:
            ll += np.log10(emission_matrix[l, 0])

        old = current

    return ll


# ============================================================================
# Diploid extension  (from diploid.rst)
# ============================================================================

def diploid_transition_prob(j1, j2, k1, k2, r, n):
    """Compute diploid transition probability.

    Parameters
    ----------
    j1, j2 : int
        Previous state (copying sources for chr 1 and chr 2).
    k1, k2 : int
        Next state.
    r : float
        Per-site recombination probability.
    n : int
        Number of reference haplotypes.

    Returns
    -------
    prob : float
    """
    r_n = r / n
    t1 = (1 - r) * (j1 == k1) + r_n
    t2 = (1 - r) * (j2 == k2) + r_n
    return t1 * t2


def emission_matrix_diploid(mu, num_sites, num_alleles):
    """Compute emission probability matrix for diploid genotypes.

    Returns matrix of shape (m, 8) indexed by genotype comparison code.

    Indexing scheme (bit-packed):
        4 = EQUAL_BOTH_HOM   (ref hom, query hom, same genotype)
        0 = UNEQUAL_BOTH_HOM (ref hom, query hom, different genotype)
        7 = BOTH_HET         (ref het, query het)
        1 = REF_HOM_OBS_HET  (ref hom, query het)
        2 = REF_HET_OBS_HOM  (ref het, query hom)
        3 = MISSING_INDEX    (query is MISSING)
    """
    EQUAL_BOTH_HOM = 4
    UNEQUAL_BOTH_HOM = 0
    BOTH_HET = 7
    REF_HOM_OBS_HET = 1
    REF_HET_OBS_HOM = 2
    MISSING_INDEX = 3

    if isinstance(mu, float):
        mu = np.full(num_sites, mu)

    e = np.full((num_sites, 8), -np.inf)

    for i in range(num_sites):
        if num_alleles[i] == 1:
            p_mut = 0.0
            p_no_mut = 1.0
        else:
            p_mut = mu[i] / (num_alleles[i] - 1)
            p_no_mut = 1 - mu[i]

        e[i, EQUAL_BOTH_HOM] = p_no_mut ** 2
        e[i, UNEQUAL_BOTH_HOM] = p_mut ** 2
        e[i, BOTH_HET] = p_no_mut**2 + p_mut**2
        e[i, REF_HOM_OBS_HET] = 2 * p_mut * p_no_mut
        e[i, REF_HET_OBS_HOM] = p_mut * p_no_mut
        e[i, MISSING_INDEX] = 1.0

    return e


def genotype_comparison_index(ref_gt, query_gt):
    """Map (ref_genotype, query_genotype) to emission matrix index.

    Genotypes are allele dosages: 0, 1, or 2.
    Uses bit-packing: index = 4*is_match + 2*is_ref_het + is_query_het
    """
    MISSING = -1
    if query_gt == MISSING:
        return 3
    is_match = int(ref_gt == query_gt)
    is_ref_het = int(ref_gt == 1)
    is_query_het = int(query_gt == 1)
    return 4 * is_match + 2 * is_ref_het + is_query_het


def build_genotype_matrix(H):
    """Build reference genotype matrix from haplotype panel.

    Parameters
    ----------
    H : ndarray of shape (m, n)
        Reference haplotype panel.

    Returns
    -------
    G : ndarray of shape (m, n, n)
        Reference genotype matrix. G[l, j1, j2] = H[l, j1] + H[l, j2].
    """
    m, n = H.shape
    G = np.zeros((m, n, n), dtype=np.int8)
    for l in range(m):
        G[l, :, :] = np.add.outer(H[l, :], H[l, :])
    return G


def forward_diploid(n, m, G, s, emission_matrix, r, norm=True):
    """Forward algorithm for the diploid Li-Stephens model.

    Parameters
    ----------
    n : int
        Number of reference haplotypes.
    m : int
        Number of sites.
    G : ndarray of shape (m, n, n)
        Reference genotype matrix. G[l, j1, j2] = allele dosage
        when copying from (j1, j2).
    s : ndarray of shape (1, m)
        Query genotype (allele dosages: 0, 1, or 2).
    emission_matrix : ndarray of shape (m, 8)
        Diploid emission probabilities.
    r : ndarray of shape (m,)
        Per-site recombination probability.
    norm : bool
        Whether to normalize.

    Returns
    -------
    F : ndarray of shape (m, n, n)
        Forward probabilities (n x n matrix at each site).
    c : ndarray of shape (m,)
        Scaling factors.
    ll : float
        Log-likelihood (base 10).
    """
    F = np.zeros((m, n, n))
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            F[0, j1, j2] = 1 / (n**2)
            ref_gt = G[0, j1, j2]
            idx = genotype_comparison_index(ref_gt, s[0, 0])
            F[0, j1, j2] *= emission_matrix[0, idx]

    if norm:
        c[0] = np.sum(F[0, :, :])
        F[0, :, :] /= c[0]

        for l in range(1, m):
            F_no_change = np.zeros((n, n))
            F_j_change = np.zeros(n)

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1 - r[l])**2 * F[l-1, j1, j2]
                    F_j_change[j1] += (1 - r[l]) * r_n[l] * F[l-1, j2, j1]

            F[l, :, :] = r_n[l]**2

            for j1 in range(n):
                F[l, j1, :] += F_j_change
                F[l, :, j1] += F_j_change
                for j2 in range(n):
                    F[l, j1, j2] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    ref_gt = G[l, j1, j2]
                    idx = genotype_comparison_index(ref_gt, s[0, l])
                    F[l, j1, j2] *= emission_matrix[l, idx]

            c[l] = np.sum(F[l, :, :])
            F[l, :, :] /= c[l]

        ll = np.sum(np.log10(c))

    else:
        for l in range(1, m):
            F_no_change = np.zeros((n, n))
            F_j1_change = np.zeros(n)
            F_j2_change = np.zeros(n)
            F_both_change = 0.0

            for j1 in range(n):
                for j2 in range(n):
                    F_no_change[j1, j2] = (1-r[l])**2 * F[l-1, j1, j2]
                    F_j1_change[j1] += (1-r[l]) * r_n[l] * F[l-1, j2, j1]
                    F_j2_change[j1] += (1-r[l]) * r_n[l] * F[l-1, j1, j2]
                    F_both_change += r_n[l]**2 * F[l-1, j1, j2]

            F[l, :, :] = F_both_change
            for j1 in range(n):
                F[l, j1, :] += F_j2_change
                F[l, :, j1] += F_j1_change
                for j2 in range(n):
                    F[l, j1, j2] += F_no_change[j1, j2]

            for j1 in range(n):
                for j2 in range(n):
                    ref_gt = G[l, j1, j2]
                    idx = genotype_comparison_index(ref_gt, s[0, l])
                    F[l, j1, j2] *= emission_matrix[l, idx]

        ll = np.log10(np.sum(F[m-1, :, :]))

    return F, c, ll


def viterbi_diploid(n, m, G, s, emission_matrix, r):
    """Viterbi algorithm for the diploid Li-Stephens model.

    Returns
    -------
    V : ndarray of shape (n, n)
        Viterbi probabilities at the last site.
    P : ndarray of shape (m, n, n), dtype int
        Pointer array (flattened index into n*n state space).
    ll : float
    """
    V = np.zeros((n, n))
    V_prev = np.zeros((n, n))
    P = np.zeros((m, n, n), dtype=np.int64)
    c = np.ones(m)
    r_n = r / n

    for j1 in range(n):
        for j2 in range(n):
            V_prev[j1, j2] = 1 / (n**2)
            ref_gt = G[0, j1, j2]
            idx = genotype_comparison_index(ref_gt, s[0, 0])
            V_prev[j1, j2] *= emission_matrix[0, idx]

    for l in range(1, m):
        c[l] = np.amax(V_prev)
        argmax = np.argmax(V_prev)
        V_prev /= c[l]

        V_rowcol_max = np.amax(V_prev, axis=1)
        arg_rowcol_max = np.argmax(V_prev, axis=1)

        no_switch = (1 - r[l])**2 + 2*(r_n[l]*(1-r[l])) + r_n[l]**2
        single_switch = r_n[l] * (1 - r[l]) + r_n[l]**2
        double_switch = r_n[l]**2

        j1_j2 = 0
        for j1 in range(n):
            for j2 in range(n):
                V_single = max(V_rowcol_max[j1], V_rowcol_max[j2])
                P_single = np.argmax(
                    np.array([V_rowcol_max[j1], V_rowcol_max[j2]])
                )
                if P_single == 0:
                    template_single = j1 * n + arg_rowcol_max[j1]
                else:
                    template_single = arg_rowcol_max[j2] * n + j2

                V[j1, j2] = V_prev[j1, j2] * no_switch
                P[l, j1, j2] = j1_j2

                single_val = single_switch * V_single
                if single_val > double_switch:
                    if V[j1, j2] < single_val:
                        V[j1, j2] = single_val
                        P[l, j1, j2] = template_single
                else:
                    if V[j1, j2] < double_switch:
                        V[j1, j2] = double_switch
                        P[l, j1, j2] = argmax

                ref_gt = G[l, j1, j2]
                idx = genotype_comparison_index(ref_gt, s[0, l])
                V[j1, j2] *= emission_matrix[l, idx]

                j1_j2 += 1

        V_prev = np.copy(V)

    ll = np.sum(np.log10(c)) + np.log10(np.amax(V))
    return V, P, ll


def backwards_viterbi_diploid(m, V_last, P):
    """Traceback for diploid Viterbi.

    Follows pointers backward from the best final state,
    recovering the flattened index at each site.
    """
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1].ravel()[path[j + 1]]
    return path


def get_phased_path(n, flat_path):
    """Convert flattened diploid path to two haploid paths.

    Parameters
    ----------
    n : int
        Number of reference haplotypes.
    flat_path : ndarray of shape (m,)
        Flattened indices into n*n state space.

    Returns
    -------
    path1, path2 : tuple of ndarray of shape (m,)
        Copying paths for each chromosome.
    """
    return np.unravel_index(flat_path, (n, n))


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate the Li & Stephens HMM on simulated data."""

    print("=" * 60)
    print("Li & Stephens HMM -- Mini Implementation Demo")
    print("=" * 60)

    # --- Copying model primitives ---
    print("\n--- Copying Model Primitives ---\n")

    n = 5
    pi = initial_distribution(n)
    print(f"Initial distribution (n={n}): {pi}")
    print(f"Sum: {pi.sum():.1f}")

    n, r_val = 4, 0.1
    A = transition_matrix(n, r_val)
    print(f"\nTransition matrix (n={n}, r={r_val}):")
    print(np.round(A, 4))
    print(f"Row sums: {A.sum(axis=1)}")
    print(f"Diagonal: {np.diag(A)}")
    print(f"Off-diagonal: {A[0, 1]}")

    m = 10
    n = 100
    rho = np.full(m, 0.04)
    r_arr = compute_recombination_probs(rho, n)
    print(f"\nRecombination probabilities (first 5 sites): "
          f"{np.round(r_arr[:5], 6)}")

    mu = 0.01
    for q, h in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        p = emission_probability(q, h, mu)
        print(f"query={q}, ref={h}: P = {p:.4f}  "
              f"({'match' if q == h else 'mismatch'})")

    num_alleles = np.array([2, 2, 1, 2, 3, 2])
    mu = 0.01
    e = emission_matrix_haploid(mu, 6, num_alleles)
    print("\nEmission matrix (mismatch | match):")
    for i in range(6):
        print(f"  Site {i} ({num_alleles[i]} alleles): "
              f"mismatch={e[i,0]:.6f}, match={e[i,1]:.6f}")

    e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
    print(f"\nNormal match:    {emission_prob_with_specials(0, 0, 0, e):.4f}")
    print(f"Normal mismatch: {emission_prob_with_specials(0, 1, 0, e):.4f}")
    print(f"NONCOPY ref:     {emission_prob_with_specials(-2, 0, 0, e):.4f}")
    print(f"MISSING query:   {emission_prob_with_specials(0, -1, 0, e):.4f}")

    print(f"\n{'n':>5} {'mu':>12} {'1/mu':>12}")
    print("-" * 32)
    for n_val in [5, 10, 50, 100, 500, 1000, 5000]:
        mu_val = estimate_mutation_probability(n_val)
        print(f"{n_val:>5} {mu_val:>12.6f} {1/mu_val:>12.1f}")

    # Verify naive vs fast forward step
    n = 5
    r_val = 0.1
    A = transition_matrix(n, r_val)
    np.random.seed(0)
    alpha_prev = np.random.dirichlet(np.ones(n))
    emission = np.random.uniform(0.5, 1.0, n)

    alpha_naive = forward_step_naive(alpha_prev, A, emission, n)
    alpha_fast = forward_step_fast(alpha_prev, r_val, r_val / n, emission, n)

    print(f"\nNaive:  {np.round(alpha_naive, 8)}")
    print(f"Fast:   {np.round(alpha_fast, 8)}")
    print(f"Match:  {np.allclose(alpha_naive, alpha_fast)}")

    # --- Complete forward algorithm ---
    print("\n--- Forward Algorithm (simple API) ---\n")

    H = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
    ])
    s = np.array([0, 0, 1, 0, 1])
    mu = 0.01
    r_arr = np.array([0.0, 0.05, 0.05, 0.05, 0.05])

    F, c, ll = forward_ls_haploid(H, s, mu, r_arr, normalize=True)
    print(f"Log-likelihood: {ll:.4f}")
    print(f"Forward probs at last site:")
    for j in range(4):
        print(f"  h_{j}: {F[-1, j]:.4f}")

    # --- Mosaic simulation ---
    print("\n--- Mosaic Simulation ---\n")

    np.random.seed(42)
    n = 10
    m = 100
    H = np.random.binomial(1, 0.3, size=(m, n))

    true_path = np.zeros(m, dtype=int)
    true_path[0:30] = 2
    true_path[30:70] = 5
    true_path[70:100] = 8

    s = np.array([H[l, true_path[l]] for l in range(m)])
    mutation_sites = np.random.choice(m, 3, replace=False)
    s[mutation_sites] = 1 - s[mutation_sites]

    print(f"True path: copies from h_{true_path[0]} (sites 0-29), "
          f"h_{true_path[30]} (30-69), h_{true_path[70]} (70-99)")
    print(f"Mutations at sites: {sorted(mutation_sites)}")

    mu = estimate_mutation_probability(n)
    r_arr = np.full(m, 0.05)
    r_arr[0] = 0.0

    F, c, ll = forward_ls_haploid(H, s, mu, r_arr, normalize=True)
    decoded_path = np.argmax(F, axis=1)
    accuracy = np.mean(decoded_path == true_path)
    print(f"Decoded accuracy (argmax of forward): {accuracy:.1%}")
    print(f"Log-likelihood: {ll:.2f}")

    # --- Haploid algorithms (lshmm-style API) ---
    print("\n--- Haploid Algorithms (lshmm-style API) ---\n")

    H = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
    ])
    s_2d = np.array([[0, 0, 1, 1]])
    mu = 0.1
    n = 3
    m = 4

    e_mat = np.zeros((m, 2))
    e_mat[:, 0] = mu
    e_mat[:, 1] = 1 - mu

    r_arr = np.array([0.0, 0.1, 0.1, 0.1])

    F_norm, c, ll_norm = forwards_ls_hap(n, m, H, s_2d, e_mat, r_arr, norm=True)
    F_raw, _, ll_raw = forwards_ls_hap(n, m, H, s_2d, e_mat, r_arr, norm=False)
    print(f"Log-likelihood (normalized): {ll_norm:.4f}")
    print(f"Log-likelihood (raw):        {ll_raw:.4f}")
    print(f"Match: {np.isclose(ll_norm, ll_raw)}")

    # Forward-backward
    B = backwards_ls_hap(n, m, H, s_2d, e_mat, c, r_arr)
    print("\nPosterior P(Z_l = j | all data):")
    for l in range(m):
        posterior = F_norm[l] * B[l]
        posterior /= posterior.sum()
        print(f"  Site {l}: {posterior.round(4)}")

    gamma, decoded = posterior_decoding(F_norm, B)
    print("\nPosterior decoding:")
    for l in range(m):
        print(f"  Site {l}: state={decoded[l]}, "
              f"confidence={gamma[l, decoded[l]]:.3f}")

    # Viterbi
    V, P, ll_vit = forwards_viterbi_hap(n, m, H, s_2d, e_mat, r_arr)
    viterbi_path = backwards_viterbi_hap(m, V, P)
    print(f"\nViterbi path: {viterbi_path}")
    print(f"Viterbi log-likelihood: {ll_vit:.4f}")

    # --- Full forward-backward-Viterbi pipeline ---
    print("\n--- Full Pipeline (n=20, m=200) ---\n")

    np.random.seed(123)
    n = 20
    m = 200
    H = np.random.binomial(1, 0.3, size=(m, n))

    true_path = np.zeros(m, dtype=int)
    true_path[0:50] = 3
    true_path[50:100] = 7
    true_path[100:150] = 12
    true_path[150:200] = 1

    s_flat = np.array([H[l, true_path[l]] for l in range(m)])
    mutation_mask = np.random.random(m) < 0.02
    s_flat[mutation_mask] = 1 - s_flat[mutation_mask]
    s_2d = s_flat.reshape(1, -1)

    mu_est = 1.0 / sum(1.0 / k for k in range(1, n - 1))
    mu = 0.5 * mu_est / (n + mu_est)
    print(f"Estimated mu: {mu:.6f}")

    e_mat = np.zeros((m, 2))
    e_mat[:, 0] = mu
    e_mat[:, 1] = 1 - mu

    r_arr = np.full(m, 0.04)
    r_arr[0] = 0.0

    F, c, ll = forwards_ls_hap(n, m, H, s_2d, e_mat, r_arr, norm=True)
    B = backwards_ls_hap(n, m, H, s_2d, e_mat, c, r_arr)
    gamma, posterior_path = posterior_decoding(F, B)

    V, P, ll_vit = forwards_viterbi_hap(n, m, H, s_2d, e_mat, r_arr)
    viterbi_path = backwards_viterbi_hap(m, V, P)

    posterior_accuracy = np.mean(posterior_path == true_path)
    viterbi_accuracy = np.mean(viterbi_path == true_path)

    print(f"Forward log-likelihood: {ll:.2f}")
    print(f"Viterbi log-likelihood: {ll_vit:.2f}")
    print(f"Posterior decoding accuracy: {posterior_accuracy:.1%}")
    print(f"Viterbi decoding accuracy:   {viterbi_accuracy:.1%}")

    viterbi_breaks = np.where(np.diff(viterbi_path) != 0)[0] + 1
    print(f"True breakpoints:     [50, 100, 150]")
    print(f"Detected breakpoints: {list(viterbi_breaks)}")

    # --- Diploid extension ---
    print("\n--- Diploid Extension ---\n")

    np.random.seed(42)
    n = 6
    m = 50
    H = np.random.binomial(1, 0.3, size=(m, n))
    G = build_genotype_matrix(H)

    true_path1 = np.zeros(m, dtype=int)
    true_path2 = np.zeros(m, dtype=int)
    true_path1[:25] = 1
    true_path1[25:] = 4
    true_path2[:] = 2

    h1 = np.array([H[l, true_path1[l]] for l in range(m)])
    h2 = np.array([H[l, true_path2[l]] for l in range(m)])
    query_gt = (h1 + h2).reshape(1, -1)

    print(f"True copying path, chr 1: h_{true_path1[0]} -> h_{true_path1[25]}")
    print(f"True copying path, chr 2: h_{true_path2[0]} (constant)")

    mu = 0.01
    e_dip = emission_matrix_diploid(mu, m, np.full(m, 2))
    r_arr = np.full(m, 0.05)
    r_arr[0] = 0.0

    V, P, ll = viterbi_diploid(n, m, G, query_gt, e_dip, r_arr)
    flat_path = backwards_viterbi_diploid(m, V, P)
    path1, path2 = get_phased_path(n, flat_path)

    print(f"Viterbi log-likelihood: {ll:.2f}")
    print(f"Decoded chr 1 path (first 10): {path1[:10]}")
    print(f"Decoded chr 2 path (first 10): {path2[:10]}")

    acc_direct = (np.mean(path1 == true_path1) +
                  np.mean(path2 == true_path2)) / 2
    acc_swapped = (np.mean(path1 == true_path2) +
                   np.mean(path2 == true_path1)) / 2
    print(f"Accuracy (direct):  {acc_direct:.1%}")
    print(f"Accuracy (swapped): {acc_swapped:.1%}")
    print(f"Best accuracy:      {max(acc_direct, acc_swapped):.1%}")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
