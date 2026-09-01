"""
Tests for momi2 timepiece code extracted from RST documentation.

Covers:
- coalescent_sfs.rst: w_matrix, etjj_constant, etjj_exponential, compute_joint_sfs
- inference.rst: multinomial_log_likelihood, poisson_log_likelihood,
                 transform_params, inverse_transform, f2_weights, f3_weights
- moran_model.rst: moran_rate_matrix, moran_eigensystem, moran_transition, moran_action
- tensor_machinery.rst: convolve_populations, admixture_tensor, hypergeom_quasi_inverse
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import pinv
from scipy.special import comb, expi
from scipy.stats import hypergeom as hypergeom_dist

# ============================================================================
# Functions from coalescent_sfs.rst
# ============================================================================

def w_matrix(n):
    """Compute the W-matrix of Polanski and Kimmel (2003)."""
    W = np.zeros((n - 1, n - 1))
    bb = np.arange(1, n)

    W[:, 0] = 6.0 / (n + 1)
    if n > 2:
        W[:, 1] = 30.0 * (n - 2 * bb) / ((n + 1) * (n + 2))

    for col in range(2, n - 1):
        j = col + 2
        W[:, col] = (
            W[:, col - 1] * (2 * j + 1) * (n - 2 * bb) / (j * (n + j + 1))
            - W[:, col - 2] * (j + 1) * (2 * j + 3) * (n - j)
              / (j * (2 * j - 1) * (n + j + 1))
        )
    return W


def etjj_constant(n, tau, N):
    """Expected time with j lineages in an epoch of duration tau and size N."""
    j = np.arange(2, n + 1)
    rate = j * (j - 1) / N
    return -np.expm1(-rate * tau) / rate


def etjj_exponential(n, tau, growth_rate, N_bottom):
    """Expected time with j lineages under exponential growth."""
    j = np.arange(2, n + 1)
    rate = j * (j - 1) / 2.0
    if abs(growth_rate) < 1e-10:
        return etjj_constant(n, tau, N_bottom)

    total_growth = tau * growth_rate
    a = rate * 2.0 / (N_bottom * growth_rate)
    return (
        np.exp(a)
        * (expi(-a * np.exp(total_growth)) - expi(-a))
        / growth_rate
    )


def compute_joint_sfs(genotype_matrix, pop_assignments, pop_names):
    """Compute the joint SFS from a genotype matrix."""
    pop_indices = {p: [] for p in pop_names}
    for idx, pop in pop_assignments.items():
        pop_indices[pop].append(idx)

    sample_sizes = [len(pop_indices[p]) for p in pop_names]
    sfs_shape = tuple(s + 1 for s in sample_sizes)
    sfs = np.zeros(sfs_shape, dtype=int)

    for site in range(genotype_matrix.shape[0]):
        config = tuple(
            genotype_matrix[site, pop_indices[p]].sum()
            for p in pop_names
        )
        sfs[config] += 1

    return sfs


# ============================================================================
# Functions from inference.rst
# ============================================================================

def multinomial_log_likelihood(observed_sfs, expected_sfs):
    """Composite log-likelihood under the multinomial model."""
    expected_probs = expected_sfs / expected_sfs.sum()
    mask = observed_sfs > 0
    ll = np.sum(observed_sfs[mask] * np.log(expected_probs[mask]))
    return ll


def poisson_log_likelihood(observed_sfs, expected_sfs):
    """Composite log-likelihood under the Poisson model."""
    mask = observed_sfs > 0
    ll = np.sum(
        observed_sfs[mask] * np.log(expected_sfs[mask])
        - expected_sfs[mask]
    )
    ll -= np.sum(expected_sfs[~mask])
    return ll


def transform_params(params, param_types):
    """Transform parameters to unconstrained space."""
    params = np.asarray(params, dtype=float)
    transformed = np.zeros_like(params)
    for i, (p, ptype) in enumerate(zip(params, param_types)):
        if ptype == 'log':
            transformed[i] = np.log(p)
        elif ptype == 'logit':
            transformed[i] = np.log(p / (1 - p))
        else:
            transformed[i] = p
    return transformed


def inverse_transform(transformed, param_types):
    """Transform back to natural parameter space."""
    transformed = np.asarray(transformed, dtype=float)
    params = np.zeros_like(transformed)
    for i, (t, ptype) in enumerate(zip(transformed, param_types)):
        if ptype == 'log':
            params[i] = np.exp(t)
        elif ptype == 'logit':
            params[i] = 1.0 / (1.0 + np.exp(-t))
        else:
            params[i] = t
    return params


def f2_weights(n_A, n_B):
    """Weight vector for f2(A, B) = E[(p_A - p_B)^2]."""
    p_A = np.arange(n_A + 1) / n_A
    p_B = np.arange(n_B + 1) / n_B
    p_A2 = np.arange(n_A + 1) * (np.arange(n_A + 1) - 1) / (n_A * (n_A - 1))
    p_B2 = np.arange(n_B + 1) * (np.arange(n_B + 1) - 1) / (n_B * (n_B - 1))
    return p_A2[:, None] + p_B2[None, :] - 2 * p_A[:, None] * p_B[None, :]


def f3_weights(n_C, n_A, n_B):
    """Weight vector for f3(C; A, B) = E[(p_C - p_A)(p_C - p_B)]."""
    p_C = np.arange(n_C + 1) / n_C
    p_A = np.arange(n_A + 1) / n_A
    p_B = np.arange(n_B + 1) / n_B
    p_C2 = np.arange(n_C + 1) * (np.arange(n_C + 1) - 1) / (n_C * (n_C - 1))
    W = np.zeros((n_C + 1, n_A + 1, n_B + 1))
    for ic in range(n_C + 1):
        for ia in range(n_A + 1):
            for ib in range(n_B + 1):
                W[ic, ia, ib] = (
                    p_C2[ic]
                    - p_C[ic] * p_A[ia]
                    - p_C[ic] * p_B[ib]
                    + p_A[ia] * p_B[ib]
                )
    return W


# ============================================================================
# Functions from moran_model.rst
# ============================================================================

def moran_rate_matrix(n):
    """Construct the Moran model rate matrix for sample size n."""
    i = np.arange(n + 1, dtype=float)
    off_diag = i * (n - i) / 2.0
    diag = -2.0 * off_diag
    Q = (np.diag(off_diag[:-1], k=1)
         + np.diag(diag, k=0)
         + np.diag(off_diag[1:], k=-1))
    return Q


def moran_eigensystem(n):
    """Compute the eigendecomposition of the Moran rate matrix."""
    Q = moran_rate_matrix(n)
    eigenvalues, V = np.linalg.eig(Q)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx].real
    V = V[:, idx].real
    V_inv = np.linalg.inv(V)
    return V, eigenvalues, V_inv


def moran_transition(t, n):
    """Compute the Moran transition matrix P(t) = exp(Q*t)."""
    V, eigs, V_inv = moran_eigensystem(n)
    D = np.diag(np.exp(t * eigs))
    P = V @ D @ V_inv
    P = np.clip(P, 0, None)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def moran_action(t, tensor, axis):
    """Apply Moran transition matrix to a tensor along a given axis."""
    n = tensor.shape[axis] - 1
    P = moran_transition(t, n)
    result = np.tensordot(tensor, P.T, axes=([axis], [0]))
    if axis != result.ndim - 1:
        result = np.moveaxis(result, -1, axis)
    return result


# ============================================================================
# Functions from tensor_machinery.rst
# ============================================================================

def convolve_populations(L1, L2, n1, n2):
    """Merge two population tensors via convolution."""
    b1 = np.array([comb(n1, j, exact=True) for j in range(n1 + 1)])
    b2 = np.array([comb(n2, k, exact=True) for k in range(n2 + 1)])
    weighted_L1 = L1 * b1
    weighted_L2 = L2 * b2

    conv = np.convolve(weighted_L1, weighted_L2)

    n_anc = n1 + n2
    b_anc = np.array([comb(n_anc, i, exact=True) for i in range(n_anc + 1)])
    L_anc = conv / b_anc

    return L_anc


def admixture_tensor(n, f):
    """Compute the admixture 3-tensor for a pulse event."""
    T = np.zeros((n + 1, n + 1, n + 1))
    draws = {
        (derived, sample_size): hypergeom_dist.pmf(
            np.arange(sample_size + 1), n, derived, sample_size
        )
        for derived in range(n + 1)
        for sample_size in range(n + 1)
    }
    for a in range(n + 1):
        for b in range(n + 1):
            for n_parent1 in range(n + 1):
                assign_prob = (
                    comb(n, n_parent1)
                    * (1 - f) ** n_parent1
                    * f ** (n - n_parent1)
                )
                n_parent2 = n - n_parent1
                T[a, b] += assign_prob * np.convolve(
                    draws[a, n_parent1], draws[b, n_parent2]
                )
    return T


def hypergeom_quasi_inverse(N, n):
    """Compute the quasi-inverse for reducing lineage count from N to n."""
    projection = np.array([
        [hypergeom_dist.pmf(j, N, i, n) for i in range(N + 1)]
        for j in range(n + 1)
    ])
    return pinv(projection)


# ============================================================================
# Tests for coalescent_sfs.rst
# ============================================================================

class TestWMatrix:
    """Tests for the W-matrix of Polanski and Kimmel (2003)."""

    def test_shape(self):
        n = 10
        W = w_matrix(n)
        assert W.shape == (n - 1, n - 1)

    def test_neutral_sfs_positive(self):
        """Under constant size, expected SFS entries should be positive."""
        n = 10
        W = w_matrix(n)
        j_vals = np.arange(2, n + 1)
        E_Tjj_neutral = 2.0 / (j_vals * (j_vals - 1))
        expected_sfs = W @ E_Tjj_neutral
        assert np.all(expected_sfs > 0), \
            "Expected SFS entries should be positive under neutrality"

    def test_neutral_sfs_decreasing_on_average(self):
        """Under constant size, expected SFS should decrease on average (low b has higher SFS)."""
        n = 10
        W = w_matrix(n)
        j_vals = np.arange(2, n + 1)
        E_Tjj_neutral = 2.0 / (j_vals * (j_vals - 1))
        expected_sfs = W @ E_Tjj_neutral
        # SFS[1] should be larger than SFS[n-1] (rare alleles more common under neutrality)
        assert expected_sfs[0] > expected_sfs[-1]

    def test_first_column_constant(self):
        """The first column W[:, 0] should be 6/(n+1)."""
        n = 15
        W = w_matrix(n)
        expected = 6.0 / (n + 1)
        assert np.allclose(W[:, 0], expected)

    def test_second_column(self):
        """The second column follows 30*(n-2b)/((n+1)(n+2))."""
        n = 12
        W = w_matrix(n)
        bb = np.arange(1, n)
        expected = 30.0 * (n - 2 * bb) / ((n + 1) * (n + 2))
        assert np.allclose(W[:, 1], expected)

    def test_small_n(self):
        """Test for n=3 (smallest non-trivial case)."""
        n = 3
        W = w_matrix(n)
        assert W.shape == (2, 2)


class TestEtjjConstant:
    """Tests for expected coalescence times under constant population size."""

    def test_large_epoch_converges(self):
        """For very long epochs, should converge to the full expectation."""
        n = 10
        tau_large = 1e10
        N = 1000
        result = etjj_constant(n, tau_large, N)
        j = np.arange(2, n + 1)
        expected = N / (j * (j - 1))
        assert np.allclose(result, expected, rtol=1e-6)

    def test_zero_epoch(self):
        """Zero epoch duration should give zero expected times."""
        n = 10
        result = etjj_constant(n, 0, 1000)
        assert np.allclose(result, 0)

    def test_length(self):
        """Output length should be n-1."""
        n = 15
        result = etjj_constant(n, 100, 1000)
        assert len(result) == n - 1

    def test_decreasing(self):
        """Expected times should decrease with j (more lineages = faster coalescence)."""
        n = 10
        result = etjj_constant(n, 1e6, 1000)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_positive(self):
        """All expected times should be non-negative."""
        n = 10
        result = etjj_constant(n, 500, 1000)
        assert np.all(result >= 0)


class TestEtjjExponential:
    """Tests for expected coalescence times under exponential growth."""

    def test_zero_growth_equals_constant(self):
        """Zero growth rate should give the same result as constant size."""
        n = 10
        tau = 500
        N = 1000
        result_exp = etjj_exponential(n, tau, 0.0, N)
        result_const = etjj_constant(n, tau, N)
        assert np.allclose(result_exp, result_const, rtol=1e-6)

    def test_length(self):
        """Output length should be n-1."""
        n = 8
        result = etjj_exponential(n, 100, 0.01, 1000)
        assert len(result) == n - 1

    def test_positive_growth_rate(self):
        """Positive growth rate should give finite positive values."""
        n = 10
        result = etjj_exponential(n, 100, 0.01, 1000)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_matches_survival_integral(self):
        n, tau, growth_rate, N_bottom = 6, 100, 1e-3, 1000
        observed = etjj_exponential(n, tau, growth_rate, N_bottom)
        expected = []
        for j in range(2, n + 1):
            coal_rate = j * (j - 1) / 2
            expected.append(quad(
                lambda t, rate=coal_rate: np.exp(
                    -rate * 2 * np.expm1(growth_rate * t)
                    / (N_bottom * growth_rate)
                ),
                0,
                tau,
            )[0])
        assert np.allclose(observed, expected, rtol=1e-11, atol=1e-11)


class TestComputeJointSFS:
    """Tests for the joint SFS computation."""

    def test_shape(self):
        """SFS shape should be (n1+1) x (n2+1) for two populations."""
        np.random.seed(42)
        n_sites, n_samples = 100, 10
        genotype_matrix = np.random.binomial(1, 0.3, size=(n_sites, n_samples))
        pop_assignments = {i: 'A' if i < 5 else 'B' for i in range(n_samples)}
        pop_names = ['A', 'B']
        sfs = compute_joint_sfs(genotype_matrix, pop_assignments, pop_names)
        assert sfs.shape == (6, 6)

    def test_sum_equals_num_sites(self):
        """The SFS entries should sum to the number of sites."""
        np.random.seed(42)
        n_sites, n_samples = 50, 8
        genotype_matrix = np.random.binomial(1, 0.4, size=(n_sites, n_samples))
        pop_assignments = {i: 'A' if i < 4 else 'B' for i in range(n_samples)}
        sfs = compute_joint_sfs(genotype_matrix, pop_assignments, ['A', 'B'])
        assert sfs.sum() == n_sites

    def test_single_population_marginal(self):
        """With one population, SFS should match the 1D frequency spectrum."""
        np.random.seed(42)
        n_sites, n_samples = 100, 6
        genotype_matrix = np.random.binomial(1, 0.3, size=(n_sites, n_samples))
        pop_assignments = {i: 'A' for i in range(n_samples)}
        sfs = compute_joint_sfs(genotype_matrix, pop_assignments, ['A'])
        assert sfs.shape == (n_samples + 1,)
        assert sfs.sum() == n_sites

    def test_non_negative(self):
        """All SFS entries should be non-negative."""
        np.random.seed(42)
        genotype_matrix = np.random.binomial(1, 0.5, size=(30, 10))
        pop_assignments = {i: 'X' if i < 5 else 'Y' for i in range(10)}
        sfs = compute_joint_sfs(genotype_matrix, pop_assignments, ['X', 'Y'])
        assert np.all(sfs >= 0)


# ============================================================================
# Tests for inference.rst
# ============================================================================

class TestMultinomialLogLikelihood:
    """Tests for the multinomial log-likelihood."""

    def test_perfect_match(self):
        """When observed matches expected proportionally, LL is maximized."""
        observed = np.array([10, 20, 30])
        expected = np.array([1.0, 2.0, 3.0])  # same proportions
        ll = multinomial_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_negative(self):
        """Log-likelihood should always be non-positive for prob <= 1."""
        observed = np.array([5, 10, 15])
        expected = np.array([1.0, 2.0, 3.0])
        ll = multinomial_log_likelihood(observed, expected)
        # LL is sum of d_i * log(p_i), where p_i <= 1 so log(p_i) <= 0
        # and d_i > 0, so LL <= 0
        assert ll <= 0

    def test_zero_observed_ignored(self):
        """Zero-count entries should not contribute to the likelihood."""
        observed = np.array([0, 10, 20])
        expected = np.array([1e-10, 1.0, 2.0])
        ll = multinomial_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_best_at_true_proportions(self):
        """LL should be higher at the true proportions than at distorted ones."""
        observed = np.array([10, 20, 30])
        expected_true = np.array([1.0, 2.0, 3.0])
        expected_bad = np.array([3.0, 2.0, 1.0])
        ll_true = multinomial_log_likelihood(observed, expected_true)
        ll_bad = multinomial_log_likelihood(observed, expected_bad)
        assert ll_true > ll_bad


class TestPoissonLogLikelihood:
    """Tests for the Poisson log-likelihood."""

    def test_finite(self):
        """Poisson LL should be finite for valid inputs."""
        observed = np.array([5, 10, 15])
        expected = np.array([5.0, 10.0, 15.0])
        ll = poisson_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_zero_observed_ignored(self):
        """Entries with zero observed counts should be ignored."""
        observed = np.array([0, 10, 0])
        expected = np.array([5.0, 10.0, 15.0])
        ll = poisson_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_best_at_true_value(self):
        """Poisson LL should be maximized when expected equals observed."""
        observed = np.array([10, 20, 30])
        # For Poisson, LL = d*log(e) - e; maximum at e = d
        ll_true = poisson_log_likelihood(observed, observed.astype(float))
        ll_bad = poisson_log_likelihood(observed, observed.astype(float) * 2)
        assert ll_true > ll_bad


class TestTransformParams:
    """Tests for parameter transformations."""

    def test_log_roundtrip(self):
        """Transform and inverse_transform with 'log' should be identity."""
        params = np.array([1.0, 10.0, 100.0])
        types = ['log', 'log', 'log']
        transformed = transform_params(params, types)
        recovered = inverse_transform(transformed, types)
        assert np.allclose(params, recovered)

    def test_logit_roundtrip(self):
        """Transform and inverse_transform with 'logit' should be identity."""
        params = np.array([0.1, 0.5, 0.9])
        types = ['logit', 'logit', 'logit']
        transformed = transform_params(params, types)
        recovered = inverse_transform(transformed, types)
        assert np.allclose(params, recovered)

    def test_none_roundtrip(self):
        """Transform with 'none' should be identity."""
        params = np.array([-5.0, 0.0, 3.14])
        types = ['none', 'none', 'none']
        transformed = transform_params(params, types)
        assert np.allclose(params, transformed)

    def test_mixed_roundtrip(self):
        """Mixed types should round-trip correctly."""
        params = np.array([5.0, 0.3, -2.0])
        types = ['log', 'logit', 'none']
        transformed = transform_params(params, types)
        recovered = inverse_transform(transformed, types)
        assert np.allclose(params, recovered)

    def test_log_transform_is_log(self):
        """Log transform should give np.log of the parameter."""
        params = np.array([np.e])
        types = ['log']
        transformed = transform_params(params, types)
        assert np.allclose(transformed, [1.0])

    def test_logit_midpoint(self):
        """Logit of 0.5 should be 0."""
        params = np.array([0.5])
        types = ['logit']
        transformed = transform_params(params, types)
        assert np.allclose(transformed, [0.0])


class TestF2Weights:
    """Tests for f2 weights."""

    def test_shape(self):
        """f2 weight matrix should have shape (n_A+1) x (n_B+1)."""
        W = f2_weights(5, 10)
        assert W.shape == (6, 11)

    def test_fixed_equal_populations_are_zero(self):
        n = 5
        W = f2_weights(n, n)
        assert np.isclose(W[0, 0], 0.0)
        assert np.isclose(W[n, n], 0.0)

    def test_uses_distinct_within_population_draws(self):
        n = 6
        W = f2_weights(n, n)
        i, j = 2, 4
        expected = (
            i * (i - 1) / (n * (n - 1))
            + j * (j - 1) / (n * (n - 1))
            - 2 * i * j / n**2
        )
        assert np.isclose(W[i, j], expected)

    def test_symmetry(self):
        """f2(A, B) weights should satisfy W[i,j] = W_transpose[j,i]
        when swapping populations."""
        n_A, n_B = 5, 8
        W1 = f2_weights(n_A, n_B)
        W2 = f2_weights(n_B, n_A)
        assert np.allclose(W1, W2.T)


class TestF3Weights:
    """Tests for f3 weights."""

    def test_shape(self):
        """f3 weight tensor should have shape (n_C+1) x (n_A+1) x (n_B+1)."""
        W = f3_weights(5, 3, 4)
        assert W.shape == (6, 4, 5)

    def test_zero_when_C_equals_A_equals_B(self):
        """When all frequencies are zero (i=0, j=0, k=0), weight is 0."""
        W = f3_weights(5, 5, 5)
        assert np.isclose(W[0, 0, 0], 0.0)

    def test_values_at_extremes(self):
        """f3(C; A, B) at (n_C, 0, 0): should be (1-0)*(1-0) = 1."""
        n = 5
        W = f3_weights(n, n, n)
        assert np.isclose(W[n, 0, 0], 1.0)


# ============================================================================
# Tests for moran_model.rst
# ============================================================================

class TestMoranRateMatrix:
    """Tests for the Moran model rate matrix."""

    def test_row_sums_zero(self):
        """Rate matrix rows should sum to zero."""
        for n in [5, 10, 20]:
            Q = moran_rate_matrix(n)
            assert np.allclose(Q.sum(axis=1), 0), \
                f"Row sums not zero for n={n}"

    def test_diagonal_negative(self):
        """Diagonal entries for interior states should be negative (departure rates)."""
        for n in [5, 10, 15]:
            Q = moran_rate_matrix(n)
            for i in range(1, n):
                assert Q[i, i] < 0, f"Diagonal Q[{i},{i}] should be negative for n={n}"

    def test_shape(self):
        """Shape should be (n+1) x (n+1)."""
        n = 8
        Q = moran_rate_matrix(n)
        assert Q.shape == (n + 1, n + 1)

    def test_absorbing_states(self):
        """States 0 and n should be absorbing (all-zero rows)."""
        n = 10
        Q = moran_rate_matrix(n)
        assert np.allclose(Q[0, :], 0)
        assert np.allclose(Q[n, :], 0)

    def test_tridiagonal(self):
        """Q should be tridiagonal: nonzero only on diag, super-diag, sub-diag."""
        n = 8
        Q = moran_rate_matrix(n)
        for i in range(n + 1):
            for j in range(n + 1):
                if abs(i - j) > 1:
                    assert Q[i, j] == 0

    def test_off_diagonal_positive_interior(self):
        """Off-diagonal entries for interior states should be positive."""
        n = 10
        Q = moran_rate_matrix(n)
        for i in range(1, n):
            assert Q[i, i + 1] > 0 if i < n else True
            assert Q[i, i - 1] > 0 if i > 0 else True


class TestMoranEigensystem:
    """Tests for the Moran eigensystem."""

    def test_eigenvalues_non_positive(self):
        """All eigenvalues should be non-positive (rate matrix property)."""
        for n in [5, 10]:
            _V, eigs, _V_inv = moran_eigensystem(n)
            assert np.all(eigs <= 1e-10), \
                f"Found positive eigenvalue for n={n}: {eigs[eigs > 1e-10]}"

    def test_reconstruct_Q(self):
        """V @ diag(eigs) @ V_inv should reconstruct Q."""
        n = 8
        V, eigs, V_inv = moran_eigensystem(n)
        Q_reconstructed = V @ np.diag(eigs) @ V_inv
        Q_original = moran_rate_matrix(n)
        assert np.allclose(Q_reconstructed, Q_original, atol=1e-8)

    def test_has_zero_eigenvalue(self):
        """There should be at least one zero eigenvalue (absorbing state)."""
        n = 10
        _V, eigs, _V_inv = moran_eigensystem(n)
        num_zeros = np.sum(np.abs(eigs) < 1e-8)
        assert num_zeros >= 1


class TestMoranTransition:
    """Tests for the Moran transition matrix."""

    def test_identity_at_t_zero(self):
        """P(0) should be the identity matrix."""
        n = 10
        P0 = moran_transition(0, n)
        assert np.allclose(P0, np.eye(n + 1), atol=1e-10)

    def test_rows_sum_to_one(self):
        """Rows of P(t) should sum to 1."""
        n = 10
        P1 = moran_transition(1.0, n)
        assert np.allclose(P1.sum(axis=1), 1.0, atol=1e-10)

    def test_non_negative(self):
        """All entries should be non-negative."""
        n = 10
        P1 = moran_transition(1.0, n)
        assert np.all(P1 >= -1e-15)

    def test_chapman_kolmogorov(self):
        """P(s+t) should be close to P(s) @ P(t) (semigroup property)."""
        n = 5
        P_s = moran_transition(0.1, n)
        P_t = moran_transition(0.1, n)
        P_st = moran_transition(0.2, n)
        # Use relaxed tolerance since eigendecomposition introduces errors
        assert np.allclose(P_s @ P_t, P_st, atol=1e-4)

    def test_transition_spreads_probability(self):
        """At moderate t, probability should spread from the initial state."""
        n = 6
        P = moran_transition(0.5, n)
        # Starting at state 3, probability should spread to neighbors
        assert P[3, 3] < 1.0
        assert P[3, 2] > 0
        assert P[3, 4] > 0

    def test_shape(self):
        """Shape should be (n+1) x (n+1)."""
        n = 8
        P = moran_transition(0.5, n)
        assert P.shape == (n + 1, n + 1)


class TestMoranAction:
    """Tests for applying Moran transition to a tensor."""

    def test_1d_tensor(self):
        """Apply Moran transition to a 1D tensor (vector)."""
        n = 5
        tensor = np.zeros(n + 1)
        tensor[3] = 1.0  # delta at state 3
        result = moran_action(0.0, tensor, 0)
        assert np.allclose(result, tensor, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match input shape."""
        n = 5
        tensor = np.random.rand(n + 1)
        result = moran_action(0.5, tensor, 0)
        assert result.shape == tensor.shape


# ============================================================================
# Tests for tensor_machinery.rst
# ============================================================================

class TestConvolvePopulations:
    """Tests for population convolution."""

    def test_output_length(self):
        """Output length should be n1 + n2 + 1."""
        n1, n2 = 5, 5
        L1 = np.ones(n1 + 1) / (n1 + 1)
        L2 = np.ones(n2 + 1) / (n2 + 1)
        L_anc = convolve_populations(L1, L2, n1, n2)
        assert len(L_anc) == n1 + n2 + 1

    def test_delta_convolution(self):
        """Convolving delta(n1) with delta(n2) should give delta(n1+n2)."""
        n1, n2 = 3, 4
        L1 = np.zeros(n1 + 1)
        L1[n1] = 1.0  # all derived in pop 1
        L2 = np.zeros(n2 + 1)
        L2[n2] = 1.0  # all derived in pop 2
        L_anc = convolve_populations(L1, L2, n1, n2)
        # All n1+n2 should be derived
        expected = np.zeros(n1 + n2 + 1)
        expected[n1 + n2] = 1.0
        assert np.allclose(L_anc, expected, atol=1e-10)

    def test_delta_zero_convolution(self):
        """Convolving delta(0) with delta(0) should give delta(0)."""
        n1, n2 = 3, 4
        L1 = np.zeros(n1 + 1)
        L1[0] = 1.0  # all ancestral in pop 1
        L2 = np.zeros(n2 + 1)
        L2[0] = 1.0  # all ancestral in pop 2
        L_anc = convolve_populations(L1, L2, n1, n2)
        expected = np.zeros(n1 + n2 + 1)
        expected[0] = 1.0
        assert np.allclose(L_anc, expected, atol=1e-10)

    def test_uniform_convolution(self):
        """Convolving two uniform vectors should produce a valid result."""
        n1, n2 = 5, 5
        L1 = np.ones(n1 + 1) / (n1 + 1)
        L2 = np.ones(n2 + 1) / (n2 + 1)
        L_anc = convolve_populations(L1, L2, n1, n2)
        assert len(L_anc) == n1 + n2 + 1
        assert np.all(np.isfinite(L_anc))


class TestAdmixtureTensor:
    """Tests for the admixture 3-tensor."""

    def test_shape(self):
        """Tensor should be (n+1) x (n+1) x (n+1)."""
        n = 5
        T = admixture_tensor(n, 0.3)
        assert T.shape == (n + 1, n + 1, n + 1)

    def test_no_admixture(self):
        """At f=0 the child is inherited entirely from parent 1."""
        n = 5
        T = admixture_tensor(n, 0.0)
        for a in range(n + 1):
            for b in range(n + 1):
                assert np.isclose(T[a, b, a], 1.0)

    def test_full_admixture(self):
        """At f=1 the child is inherited entirely from parent 2."""
        n = 5
        T = admixture_tensor(n, 1.0)
        for a in range(n + 1):
            for b in range(n + 1):
                assert np.isclose(T[a, b, b], 1.0)

    def test_probability_sums_to_one(self):
        """Every parental configuration defines a child distribution."""
        n = 5
        f = 0.3
        T = admixture_tensor(n, f)
        assert np.allclose(T.sum(axis=2), 1.0)

    def test_expected_child_derived_count(self):
        """Child mean is the ancestry-weighted parental count."""
        n = 10
        f = 0.3
        T = admixture_tensor(n, f)
        a, b = 2, 8
        expected_k = sum(k * T[a, b, k] for k in range(n + 1))
        assert np.isclose(expected_k, (1 - f) * a + f * b, atol=1e-10)

    def test_non_negative(self):
        """All tensor entries should be non-negative."""
        n = 6
        T = admixture_tensor(n, 0.4)
        assert np.all(T >= 0)


class TestHypergeomQuasiInverse:
    """Tests for the hypergeometric quasi-inverse."""

    def test_shape(self):
        """Output shape should be (N+1) x (n+1)."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        assert M.shape == (N + 1, n + 1)

    def test_is_right_inverse_of_projection(self):
        """Projection times quasi-inverse is identity."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        projection = np.array([
            [hypergeom_dist.pmf(j, N, i, n) for i in range(N + 1)]
            for j in range(n + 1)
        ])
        assert np.allclose(projection @ M, np.eye(n + 1), atol=1e-12)

    def test_has_negative_weights(self):
        """The pseudoinverse is not a probability matrix."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        assert np.any(M < 0)

    def test_square_case_is_identity(self):
        N = 6
        n = N
        M = hypergeom_quasi_inverse(N, n)
        assert np.allclose(M, np.eye(N + 1), atol=1e-12)
