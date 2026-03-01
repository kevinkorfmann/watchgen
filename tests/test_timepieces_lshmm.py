"""
Tests for the Li & Stephens HMM code from the lshmm RST documentation.

Each function is re-defined from the RST code blocks and then tested
for mathematical correctness.
"""

import numpy as np
import pytest


# ============================================================================
# Functions from copying_model.rst
# ============================================================================

def initial_distribution(n):
    """Uniform initial distribution over n reference haplotypes."""
    return np.ones(n) / n


def transition_matrix(n, r):
    """Build the Li-Stephens transition matrix."""
    A = np.full((n, n), r / n)
    np.fill_diagonal(A, (1 - r) + r / n)
    return A


def compute_recombination_probs(rho, n):
    """Compute per-site recombination probabilities."""
    r = 1 - np.exp(-rho / n)
    r[0] = 0.0
    return r


def emission_probability(query_allele, ref_allele, mu):
    """Compute emission probability for one site."""
    if query_allele == ref_allele:
        return 1 - mu
    else:
        return mu


def emission_matrix_haploid(mu, num_sites, num_alleles):
    """Compute the emission probability matrix for the haploid case."""
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
    """Compute emission probability handling NONCOPY and MISSING."""
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
    """Estimate mutation probability from the number of haplotypes."""
    if n < 3:
        raise ValueError("Need at least 3 haplotypes.")
    theta_tilde = 1.0 / sum(1.0 / k for k in range(1, n - 1))
    mu = 0.5 * theta_tilde / (n + theta_tilde)
    return mu


def forward_step_naive(alpha_prev, A, emission, n):
    """Naive O(n^2) forward step."""
    alpha = np.zeros(n)
    for j in range(n):
        alpha[j] = emission[j] * np.sum(alpha_prev * A[:, j])
    return alpha


def forward_ls_haploid(H, s, mu, r, normalize=True):
    """Complete forward algorithm for the haploid Li-Stephens model."""
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
# Functions from haploid_algorithms.rst
# ============================================================================

def forwards_ls_hap(n, m, H, s, emission_matrix, r, norm=True):
    """Forward algorithm for the haploid Li-Stephens model."""
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
    """Backward algorithm for the haploid Li-Stephens model."""
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
    """Compute posterior decoding from forward-backward probabilities."""
    gamma = F * B
    gamma /= gamma.sum(axis=1, keepdims=True)
    path = np.argmax(gamma, axis=1)
    return gamma, path


def forwards_viterbi_hap(n, m, H, s, emission_matrix, r):
    """Viterbi algorithm for the haploid Li-Stephens model."""
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
    """Traceback to find the most likely path."""
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1, path[j + 1]]
    return path


def path_loglik_hap(n, m, H, path, s, emission_matrix, r):
    """Evaluate the log-likelihood of a specific copying path."""
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
# Functions from diploid.rst
# ============================================================================

def diploid_transition_prob(j1, j2, k1, k2, r, n):
    """Compute diploid transition probability."""
    r_n = r / n
    t1 = (1 - r) * (j1 == k1) + r_n
    t2 = (1 - r) * (j2 == k2) + r_n
    return t1 * t2


def emission_matrix_diploid(mu, num_sites, num_alleles):
    """Compute emission probability matrix for diploid genotypes."""
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
    """Map (ref_genotype, query_genotype) to emission matrix index."""
    MISSING = -1
    if query_gt == MISSING:
        return 3
    is_match = int(ref_gt == query_gt)
    is_ref_het = int(ref_gt == 1)
    is_query_het = int(query_gt == 1)
    return 4 * is_match + 2 * is_ref_het + is_query_het


def build_genotype_matrix(H):
    """Build reference genotype matrix from haplotype panel."""
    m, n = H.shape
    G = np.zeros((m, n, n), dtype=np.int8)
    for l in range(m):
        G[l, :, :] = np.add.outer(H[l, :], H[l, :])
    return G


def backwards_viterbi_diploid(m, V_last, P):
    """Traceback for diploid Viterbi."""
    path = np.zeros(m, dtype=np.int64)
    path[m - 1] = np.argmax(V_last)
    for j in range(m - 2, -1, -1):
        path[j] = P[j + 1].ravel()[path[j + 1]]
    return path


def get_phased_path(n, flat_path):
    """Convert flattened diploid path to two haploid paths."""
    return np.unravel_index(flat_path, (n, n))


# ============================================================================
# Tests for copying_model.rst functions
# ============================================================================

class TestInitialDistribution:
    def test_sums_to_one(self):
        for n in [1, 5, 10, 100]:
            pi = initial_distribution(n)
            assert np.isclose(pi.sum(), 1.0)

    def test_uniform(self):
        n = 7
        pi = initial_distribution(n)
        assert np.allclose(pi, np.ones(n) / n)

    def test_correct_length(self):
        for n in [1, 3, 50]:
            pi = initial_distribution(n)
            assert len(pi) == n


class TestTransitionMatrix:
    def test_rows_sum_to_one(self):
        for n in [2, 5, 10, 50]:
            for r in [0.0, 0.01, 0.1, 0.5, 1.0]:
                A = transition_matrix(n, r)
                row_sums = A.sum(axis=1)
                assert np.allclose(row_sums, np.ones(n)), \
                    f"Row sums failed for n={n}, r={r}"

    def test_shape(self):
        n = 7
        A = transition_matrix(n, 0.1)
        assert A.shape == (n, n)

    def test_diagonal_values(self):
        n, r = 4, 0.1
        A = transition_matrix(n, r)
        expected_diag = (1 - r) + r / n
        assert np.allclose(np.diag(A), expected_diag)

    def test_off_diagonal_values(self):
        n, r = 4, 0.1
        A = transition_matrix(n, r)
        expected_off = r / n
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert np.isclose(A[i, j], expected_off)

    def test_no_recombination(self):
        """With r=0, the transition matrix is the identity."""
        n = 5
        A = transition_matrix(n, 0.0)
        assert np.allclose(A, np.eye(n))

    def test_full_recombination(self):
        """With r=1, all entries should be 1/n."""
        n = 5
        A = transition_matrix(n, 1.0)
        assert np.allclose(A, np.full((n, n), 1.0 / n))

    def test_symmetry(self):
        """The transition matrix should be symmetric."""
        n, r = 6, 0.2
        A = transition_matrix(n, r)
        assert np.allclose(A, A.T)


class TestComputeRecombinationProbs:
    def test_first_site_zero(self):
        rho = np.full(10, 0.04)
        r = compute_recombination_probs(rho, 100)
        assert r[0] == 0.0

    def test_values_between_zero_and_one(self):
        rho = np.full(10, 0.04)
        r = compute_recombination_probs(rho, 100)
        assert np.all(r >= 0.0)
        assert np.all(r <= 1.0)

    def test_increases_with_rho(self):
        """Higher rho should give higher recombination probability."""
        n = 50
        rho_low = np.array([0.0, 0.01])
        rho_high = np.array([0.0, 0.1])
        r_low = compute_recombination_probs(rho_low, n)
        r_high = compute_recombination_probs(rho_high, n)
        assert r_high[1] > r_low[1]

    def test_small_rho_approximation(self):
        """For small rho/n, 1 - exp(-rho/n) ~ rho/n."""
        n = 1000
        rho = np.array([0.0, 0.001])
        r = compute_recombination_probs(rho, n)
        approx = rho[1] / n
        assert np.isclose(r[1], approx, rtol=1e-3)


class TestEmissionProbability:
    def test_match(self):
        assert emission_probability(0, 0, 0.01) == 0.99
        assert emission_probability(1, 1, 0.01) == 0.99

    def test_mismatch(self):
        assert emission_probability(0, 1, 0.01) == 0.01
        assert emission_probability(1, 0, 0.01) == 0.01

    def test_sum_for_biallelic(self):
        """For biallelic site, P(match) + P(mismatch) = 1."""
        mu = 0.05
        p_match = emission_probability(0, 0, mu)
        p_mismatch = emission_probability(0, 1, mu)
        assert np.isclose(p_match + p_mismatch, 1.0)


class TestEmissionMatrixHaploid:
    def test_shape(self):
        e = emission_matrix_haploid(0.01, 5, np.array([2, 2, 2, 2, 2]))
        assert e.shape == (5, 2)

    def test_biallelic_values(self):
        mu = 0.01
        e = emission_matrix_haploid(mu, 3, np.array([2, 2, 2]))
        assert np.allclose(e[:, 0], mu)  # mismatch = mu / (2-1) = mu
        assert np.allclose(e[:, 1], 1 - mu)

    def test_invariant_site(self):
        """Invariant sites should have mismatch=0, match=1."""
        e = emission_matrix_haploid(0.01, 3, np.array([1, 2, 1]))
        assert e[0, 0] == 0.0
        assert e[0, 1] == 1.0
        assert e[2, 0] == 0.0
        assert e[2, 1] == 1.0

    def test_multiallelic_emission_sums(self):
        """For a sites with a alleles, match + (a-1)*mismatch = 1."""
        mu = 0.02
        for a in [2, 3, 4]:
            e = emission_matrix_haploid(mu, 1, np.array([a]))
            total = e[0, 1] + (a - 1) * e[0, 0]
            assert np.isclose(total, 1.0), f"Sum failed for a={a}"

    def test_array_mu(self):
        """Test that array mu works."""
        mu = np.array([0.01, 0.02, 0.03])
        e = emission_matrix_haploid(mu, 3, np.array([2, 2, 2]))
        assert np.isclose(e[0, 1], 0.99)
        assert np.isclose(e[1, 1], 0.98)
        assert np.isclose(e[2, 1], 0.97)


class TestEmissionProbWithSpecials:
    def test_noncopy(self):
        e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
        assert emission_prob_with_specials(-2, 0, 0, e) == 0.0

    def test_missing(self):
        e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
        assert emission_prob_with_specials(0, -1, 0, e) == 1.0

    def test_normal_match(self):
        e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
        assert np.isclose(emission_prob_with_specials(0, 0, 0, e), 0.99)

    def test_normal_mismatch(self):
        e = emission_matrix_haploid(0.01, 3, np.array([2, 2, 2]))
        assert np.isclose(emission_prob_with_specials(0, 1, 0, e), 0.01)


class TestEstimateMutationProbability:
    def test_minimum_n(self):
        with pytest.raises(ValueError):
            estimate_mutation_probability(2)

    def test_positive(self):
        for n in [3, 5, 10, 100, 1000]:
            mu = estimate_mutation_probability(n)
            assert mu > 0

    def test_decreases_with_n(self):
        """mu should decrease as n increases (more haplotypes -> closer match)."""
        mus = [estimate_mutation_probability(n) for n in [5, 10, 50, 100, 500]]
        for i in range(len(mus) - 1):
            assert mus[i] > mus[i + 1]

    def test_less_than_half(self):
        """mu should always be less than 0.5."""
        for n in [3, 10, 100]:
            mu = estimate_mutation_probability(n)
            assert mu < 0.5


class TestForwardLSHaploid:
    def test_normalized_sums_to_one(self):
        """Forward probs should sum to 1 at each site when normalized."""
        np.random.seed(42)
        H = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        s = np.array([0, 0, 1])
        mu = 0.01
        r = np.array([0.0, 0.05, 0.05])
        F, c, ll = forward_ls_haploid(H, s, mu, r, normalize=True)
        for l in range(3):
            assert np.isclose(F[l].sum(), 1.0, atol=1e-10)

    def test_normalized_unnormalized_same_ll(self):
        """Normalized and unnormalized should give same log-likelihood."""
        H = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        s = np.array([0, 0, 1, 0])
        mu = 0.01
        r = np.array([0.0, 0.05, 0.05, 0.05])
        _, _, ll_norm = forward_ls_haploid(H, s, mu, r, normalize=True)
        _, _, ll_raw = forward_ls_haploid(H, s, mu, r, normalize=False)
        assert np.isclose(ll_norm, ll_raw, rtol=1e-6)

    def test_all_match_higher_ll(self):
        """Query matching one haplotype perfectly should have higher LL
        than a query with mismatches."""
        H = np.array([[0, 1], [0, 1], [0, 1]])
        s_match = np.array([0, 0, 0])  # matches h_0 perfectly
        s_mismatch = np.array([1, 1, 1])  # matches h_1 perfectly
        s_bad = np.array([0, 1, 0])  # mixes
        mu = 0.01
        r = np.array([0.0, 0.01, 0.01])
        _, _, ll_match = forward_ls_haploid(H, s_match, mu, r)
        _, _, ll_bad = forward_ls_haploid(H, s_bad, mu, r)
        # Perfect match should have higher (less negative) LL
        assert ll_match > ll_bad


# ============================================================================
# Tests for haploid_algorithms.rst functions
# ============================================================================

class TestForwardsLsHap:
    @pytest.fixture
    def small_example(self):
        H = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ])
        s = np.array([[0, 0, 1, 1]])
        mu = 0.1
        e_mat = np.zeros((4, 2))
        e_mat[:, 0] = mu
        e_mat[:, 1] = 1 - mu
        r = np.array([0.0, 0.1, 0.1, 0.1])
        n = 3
        return n, H, s, e_mat, r

    def test_normalized_sums_to_one(self, small_example):
        n, H, s, e_mat, r = small_example
        F, c, ll = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
        for l in range(4):
            assert np.isclose(F[l].sum(), 1.0, atol=1e-10)

    def test_norm_and_raw_same_ll(self, small_example):
        n, H, s, e_mat, r = small_example
        _, _, ll_norm = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
        _, _, ll_raw = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=False)
        assert np.isclose(ll_norm, ll_raw, rtol=1e-6)

    def test_output_shape(self, small_example):
        n, H, s, e_mat, r = small_example
        F, c, ll = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
        assert F.shape == (4, 3)
        assert c.shape == (4,)

    def test_positive_probabilities(self, small_example):
        n, H, s, e_mat, r = small_example
        F, c, ll = forwards_ls_hap(n, 4, H, s, e_mat, r, norm=True)
        assert np.all(F >= 0)


class TestBackwardsLsHap:
    @pytest.fixture
    def fb_example(self):
        H = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ])
        s = np.array([[0, 0, 1, 1]])
        mu = 0.1
        e_mat = np.zeros((4, 2))
        e_mat[:, 0] = mu
        e_mat[:, 1] = 1 - mu
        r = np.array([0.0, 0.1, 0.1, 0.1])
        n = 3
        m = 4
        F, c, ll = forwards_ls_hap(n, m, H, s, e_mat, r, norm=True)
        B = backwards_ls_hap(n, m, H, s, e_mat, c, r)
        return F, B, n, m

    def test_backward_last_site_ones(self, fb_example):
        F, B, n, m = fb_example
        assert np.allclose(B[m - 1, :], 1.0)

    def test_posterior_sums_to_one(self, fb_example):
        F, B, n, m = fb_example
        for l in range(m):
            posterior = F[l] * B[l]
            posterior /= posterior.sum()
            assert np.isclose(posterior.sum(), 1.0, atol=1e-10)

    def test_positive_backward(self, fb_example):
        F, B, n, m = fb_example
        assert np.all(B >= 0)


class TestPosteriorDecoding:
    def test_output_shape(self):
        F = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
        B = np.ones_like(F)
        gamma, path = posterior_decoding(F, B)
        assert gamma.shape == F.shape
        assert path.shape == (2,)

    def test_gamma_sums_to_one(self):
        F = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
        B = np.ones_like(F)
        gamma, path = posterior_decoding(F, B)
        for l in range(2):
            assert np.isclose(gamma[l].sum(), 1.0, atol=1e-10)

    def test_path_is_argmax(self):
        F = np.array([[0.5, 0.3, 0.2], [0.1, 0.7, 0.2]])
        B = np.ones_like(F)
        gamma, path = posterior_decoding(F, B)
        assert path[0] == 0
        assert path[1] == 1


class TestViterbiHaploid:
    @pytest.fixture
    def viterbi_example(self):
        np.random.seed(42)
        n, m = 5, 20
        H = np.random.binomial(1, 0.3, size=(m, n))
        true_path = np.zeros(m, dtype=int)
        true_path[10:] = 3
        s_flat = np.array([H[l, true_path[l]] for l in range(m)])
        s = s_flat.reshape(1, -1)
        mu = 0.05
        e_mat = np.zeros((m, 2))
        e_mat[:, 0] = mu
        e_mat[:, 1] = 1 - mu
        r = np.full(m, 0.1)
        r[0] = 0.0
        return n, m, H, s, e_mat, r, true_path

    def test_viterbi_path_length(self, viterbi_example):
        n, m, H, s, e_mat, r, true_path = viterbi_example
        V, P, ll = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        path = backwards_viterbi_hap(m, V, P)
        assert len(path) == m

    def test_viterbi_path_valid_states(self, viterbi_example):
        n, m, H, s, e_mat, r, true_path = viterbi_example
        V, P, ll = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        path = backwards_viterbi_hap(m, V, P)
        assert np.all(path >= 0)
        assert np.all(path < n)

    def test_viterbi_ll_not_nan(self, viterbi_example):
        n, m, H, s, e_mat, r, true_path = viterbi_example
        V, P, ll = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        assert not np.isnan(ll)
        assert not np.isinf(ll)

    def test_viterbi_path_beats_random(self, viterbi_example):
        """Viterbi path should have higher LL than a random path."""
        n, m, H, s, e_mat, r, true_path = viterbi_example
        V, P, ll_vit = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        vit_path = backwards_viterbi_hap(m, V, P)

        ll_viterbi = path_loglik_hap(n, m, H, vit_path, s, e_mat, r)

        np.random.seed(123)
        random_path = np.random.randint(0, n, m)
        ll_random = path_loglik_hap(n, m, H, random_path, s, e_mat, r)

        assert ll_viterbi >= ll_random - 1e-10


class TestPathLogLikHap:
    def test_constant_path_positive_finite(self):
        n = 3
        m = 5
        H = np.zeros((m, n), dtype=int)
        H[:, 1] = 1
        s = np.array([[0, 0, 0, 0, 0]])
        e_mat = np.zeros((m, 2))
        e_mat[:, 0] = 0.01
        e_mat[:, 1] = 0.99
        r = np.array([0.0, 0.05, 0.05, 0.05, 0.05])
        path = np.zeros(m, dtype=int)

        ll = path_loglik_hap(n, m, H, path, s, e_mat, r)
        assert np.isfinite(ll)

    def test_switch_penalty(self):
        """A path with a switch should have lower LL than a path without."""
        n = 3
        m = 4
        H = np.zeros((m, n), dtype=int)  # All zeros
        s = np.array([[0, 0, 0, 0]])
        e_mat = np.zeros((m, 2))
        e_mat[:, 0] = 0.01
        e_mat[:, 1] = 0.99
        r = np.array([0.0, 0.05, 0.05, 0.05])

        # Path staying in state 0
        path_stay = np.array([0, 0, 0, 0])
        ll_stay = path_loglik_hap(n, m, H, path_stay, s, e_mat, r)

        # Path switching
        path_switch = np.array([0, 1, 0, 0])
        ll_switch = path_loglik_hap(n, m, H, path_switch, s, e_mat, r)

        assert ll_stay > ll_switch


# ============================================================================
# Tests for diploid.rst functions
# ============================================================================

class TestDiploidTransitionProb:
    def test_row_sums_to_one(self):
        n, r = 4, 0.1
        for j1 in range(n):
            for j2 in range(n):
                total = sum(
                    diploid_transition_prob(j1, j2, k1, k2, r, n)
                    for k1 in range(n)
                    for k2 in range(n)
                )
                assert np.isclose(total, 1.0), \
                    f"Row sum failed for j1={j1}, j2={j2}"

    def test_no_switch(self):
        """Both stay -> product of haploid stays."""
        n, r = 4, 0.1
        r_n = r / n
        expected = ((1 - r) + r_n) ** 2
        result = diploid_transition_prob(0, 1, 0, 1, r, n)
        assert np.isclose(result, expected)

    def test_double_switch(self):
        """Both switch -> (r/n)^2."""
        n, r = 4, 0.1
        r_n = r / n
        result = diploid_transition_prob(0, 1, 2, 3, r, n)
        assert np.isclose(result, r_n ** 2)

    def test_single_switch(self):
        """One stays, one switches."""
        n, r = 4, 0.1
        r_n = r / n
        # j1 stays, j2 switches
        result = diploid_transition_prob(0, 1, 0, 3, r, n)
        expected = ((1 - r) + r_n) * r_n
        assert np.isclose(result, expected)

    def test_factors_as_product(self):
        """Diploid transition should factor as product of two haploid transitions."""
        n, r = 5, 0.15
        for j1, j2, k1, k2 in [(0, 1, 2, 3), (1, 1, 1, 1), (2, 3, 0, 4)]:
            dip = diploid_transition_prob(j1, j2, k1, k2, r, n)
            r_n = r / n
            hap1 = (1 - r) * (j1 == k1) + r_n
            hap2 = (1 - r) * (j2 == k2) + r_n
            assert np.isclose(dip, hap1 * hap2)


class TestEmissionMatrixDiploid:
    def test_shape(self):
        e = emission_matrix_diploid(0.01, 5, np.full(5, 2))
        assert e.shape == (5, 8)

    def test_both_hom_match(self):
        mu = 0.01
        e = emission_matrix_diploid(mu, 1, np.array([2]))
        expected = (1 - mu) ** 2
        assert np.isclose(e[0, 4], expected)  # EQUAL_BOTH_HOM = 4

    def test_both_hom_mismatch(self):
        mu = 0.01
        e = emission_matrix_diploid(mu, 1, np.array([2]))
        expected = mu ** 2
        assert np.isclose(e[0, 0], expected)  # UNEQUAL_BOTH_HOM = 0

    def test_both_het(self):
        mu = 0.01
        e = emission_matrix_diploid(mu, 1, np.array([2]))
        expected = (1 - mu)**2 + mu**2
        assert np.isclose(e[0, 7], expected)  # BOTH_HET = 7

    def test_ref_hom_obs_het(self):
        mu = 0.01
        e = emission_matrix_diploid(mu, 1, np.array([2]))
        expected = 2 * mu * (1 - mu)
        assert np.isclose(e[0, 1], expected)  # REF_HOM_OBS_HET = 1

    def test_ref_het_obs_hom(self):
        mu = 0.01
        e = emission_matrix_diploid(mu, 1, np.array([2]))
        expected = mu * (1 - mu)
        assert np.isclose(e[0, 2], expected)  # REF_HET_OBS_HOM = 2

    def test_missing(self):
        e = emission_matrix_diploid(0.01, 1, np.array([2]))
        assert e[0, 3] == 1.0  # MISSING_INDEX = 3

    def test_invariant_site(self):
        """Invariant site: p_mut=0, p_no_mut=1."""
        e = emission_matrix_diploid(0.01, 1, np.array([1]))
        assert np.isclose(e[0, 4], 1.0)  # EQUAL_BOTH_HOM: 1^2
        assert np.isclose(e[0, 0], 0.0)  # UNEQUAL_BOTH_HOM: 0^2


class TestGenotypeComparisonIndex:
    def test_matching_homozygous(self):
        assert genotype_comparison_index(0, 0) == 4  # EQUAL_BOTH_HOM
        assert genotype_comparison_index(2, 2) == 4

    def test_mismatching_homozygous(self):
        assert genotype_comparison_index(0, 2) == 0  # UNEQUAL_BOTH_HOM
        assert genotype_comparison_index(2, 0) == 0

    def test_both_het(self):
        assert genotype_comparison_index(1, 1) == 7  # BOTH_HET

    def test_ref_hom_query_het(self):
        assert genotype_comparison_index(0, 1) == 1  # REF_HOM_OBS_HET
        assert genotype_comparison_index(2, 1) == 1

    def test_ref_het_query_hom(self):
        assert genotype_comparison_index(1, 0) == 2  # REF_HET_OBS_HOM
        assert genotype_comparison_index(1, 2) == 2

    def test_missing(self):
        assert genotype_comparison_index(0, -1) == 3  # MISSING_INDEX
        assert genotype_comparison_index(1, -1) == 3
        assert genotype_comparison_index(2, -1) == 3


class TestBuildGenotypeMatrix:
    def test_shape(self):
        H = np.array([[0, 1, 0], [1, 0, 1]])
        G = build_genotype_matrix(H)
        assert G.shape == (2, 3, 3)

    def test_diagonal_is_twice_allele(self):
        """G[l, j, j] = H[l, j] + H[l, j] = 2 * H[l, j]."""
        H = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        G = build_genotype_matrix(H)
        for l in range(2):
            for j in range(4):
                assert G[l, j, j] == 2 * H[l, j]

    def test_symmetry(self):
        """G[l, j1, j2] = G[l, j2, j1]."""
        H = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        G = build_genotype_matrix(H)
        for l in range(2):
            assert np.allclose(G[l], G[l].T)

    def test_values_range(self):
        """Genotype values should be in {0, 1, 2}."""
        np.random.seed(42)
        H = np.random.binomial(1, 0.5, size=(10, 6))
        G = build_genotype_matrix(H)
        assert np.all(G >= 0)
        assert np.all(G <= 2)


class TestGetPhasedPath:
    def test_roundtrip(self):
        """Flattening and unflattening should be consistent."""
        n = 5
        # Create a flat path
        flat_path = np.array([0, 6, 12, 18, 24])  # (0,0), (1,1), (2,2), (3,3), (4,4)
        path1, path2 = get_phased_path(n, flat_path)
        for i in range(5):
            assert path1[i] == i
            assert path2[i] == i

    def test_unravel(self):
        n = 4
        flat_path = np.array([5])  # row=1, col=1 for n=4
        path1, path2 = get_phased_path(n, flat_path)
        assert path1[0] == 1
        assert path2[0] == 1


# ============================================================================
# Integration tests
# ============================================================================

class TestForwardBackwardIntegration:
    """Integration tests that combine forward, backward, and Viterbi."""

    def test_forward_backward_complete_pipeline(self):
        """Run the full forward-backward pipeline and verify consistency."""
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
        s = s_flat.reshape(1, -1)

        mu_est = 1.0 / sum(1.0 / k for k in range(1, n - 1))
        mu = 0.5 * mu_est / (n + mu_est)

        e_mat = np.zeros((m, 2))
        e_mat[:, 0] = mu
        e_mat[:, 1] = 1 - mu

        r = np.full(m, 0.04)
        r[0] = 0.0

        F, c, ll = forwards_ls_hap(n, m, H, s, e_mat, r, norm=True)
        B = backwards_ls_hap(n, m, H, s, e_mat, c, r)
        gamma, posterior_path = posterior_decoding(F, B)

        V, P, ll_vit = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        viterbi_path = backwards_viterbi_hap(m, V, P)

        # Forward LL should be >= Viterbi LL (sum over all paths >= max path)
        assert ll >= ll_vit - 1e-6

        # Posterior should sum to 1 at each site
        for l in range(m):
            assert np.isclose(gamma[l].sum(), 1.0, atol=1e-8)

        # Accuracy should be reasonably high for this problem
        posterior_accuracy = np.mean(posterior_path == true_path)
        viterbi_accuracy = np.mean(viterbi_path == true_path)
        assert posterior_accuracy > 0.5
        assert viterbi_accuracy > 0.5

    def test_viterbi_is_optimal(self):
        """Viterbi path should have LL >= any other path's LL."""
        np.random.seed(42)
        n, m = 5, 20
        H = np.random.binomial(1, 0.3, size=(m, n))
        true_path = np.zeros(m, dtype=int)
        true_path[10:] = 3
        s_flat = np.array([H[l, true_path[l]] for l in range(m)])
        s = s_flat.reshape(1, -1)

        mu = 0.05
        e_mat = np.zeros((m, 2))
        e_mat[:, 0] = mu
        e_mat[:, 1] = 1 - mu
        r = np.full(m, 0.1)
        r[0] = 0.0

        V, P, _ = forwards_viterbi_hap(n, m, H, s, e_mat, r)
        vit_path = backwards_viterbi_hap(m, V, P)

        ll_viterbi = path_loglik_hap(n, m, H, vit_path, s, e_mat, r)
        ll_true = path_loglik_hap(n, m, H, true_path, s, e_mat, r)

        assert ll_viterbi >= ll_true - 1e-10
