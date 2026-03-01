"""
Tests for watchgen.mini_momi2 module.

Covers all functions extracted from the momi2 RST documentation:
- coalescent_sfs: w_matrix, etjj_constant, etjj_exponential, compute_joint_sfs
- moran_model: moran_rate_matrix, moran_eigensystem, moran_transition, moran_action
- tensor_machinery: convolve_populations, admixture_tensor, hypergeom_quasi_inverse
- inference: multinomial_log_likelihood, poisson_log_likelihood,
             transform_params, inverse_transform, f2_weights, f3_weights
"""

import numpy as np

from watchgen.mini_momi2 import (
    w_matrix,
    etjj_constant,
    etjj_exponential,
    compute_joint_sfs,
    moran_rate_matrix,
    moran_eigensystem,
    moran_transition,
    moran_action,
    convolve_populations,
    admixture_tensor,
    hypergeom_quasi_inverse,
    multinomial_log_likelihood,
    poisson_log_likelihood,
    transform_params,
    inverse_transform,
    f2_weights,
    f3_weights,
)


# ============================================================================
# Tests for coalescent_sfs: w_matrix
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
        """Under constant size, expected SFS should decrease on average."""
        n = 10
        W = w_matrix(n)
        j_vals = np.arange(2, n + 1)
        E_Tjj_neutral = 2.0 / (j_vals * (j_vals - 1))
        expected_sfs = W @ E_Tjj_neutral
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


# ============================================================================
# Tests for coalescent_sfs: etjj_constant
# ============================================================================

class TestEtjjConstant:
    """Tests for expected coalescence times under constant population size."""

    def test_large_epoch_converges(self):
        """For very long epochs, should converge to the full expectation."""
        n = 10
        tau_large = 1e10
        N = 1000
        result = etjj_constant(n, tau_large, N)
        j = np.arange(2, n + 1)
        rate = j * (j - 1) / 2.0
        expected = 1.0 / rate
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
        """Expected times should decrease with j."""
        n = 10
        result = etjj_constant(n, 1e6, 1000)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_positive(self):
        """All expected times should be non-negative."""
        n = 10
        result = etjj_constant(n, 500, 1000)
        assert np.all(result >= 0)


# ============================================================================
# Tests for coalescent_sfs: etjj_exponential
# ============================================================================

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


# ============================================================================
# Tests for coalescent_sfs: compute_joint_sfs
# ============================================================================

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
# Tests for moran_model: moran_rate_matrix
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
        """Diagonal entries for interior states should be negative."""
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
        """Q should be tridiagonal."""
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


# ============================================================================
# Tests for moran_model: moran_eigensystem
# ============================================================================

class TestMoranEigensystem:
    """Tests for the Moran eigensystem."""

    def test_eigenvalues_non_positive(self):
        """All eigenvalues should be non-positive."""
        for n in [5, 10]:
            V, eigs, V_inv = moran_eigensystem(n)
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
        """There should be at least one zero eigenvalue."""
        n = 10
        V, eigs, V_inv = moran_eigensystem(n)
        num_zeros = np.sum(np.abs(eigs) < 1e-8)
        assert num_zeros >= 1


# ============================================================================
# Tests for moran_model: moran_transition
# ============================================================================

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
        """P(s+t) should be close to P(s) @ P(t)."""
        n = 5
        P_s = moran_transition(0.1, n)
        P_t = moran_transition(0.1, n)
        P_st = moran_transition(0.2, n)
        assert np.allclose(P_s @ P_t, P_st, atol=1e-4)

    def test_transition_spreads_probability(self):
        """At moderate t, probability should spread from the initial state."""
        n = 6
        P = moran_transition(0.5, n)
        assert P[3, 3] < 1.0
        assert P[3, 2] > 0
        assert P[3, 4] > 0

    def test_shape(self):
        """Shape should be (n+1) x (n+1)."""
        n = 8
        P = moran_transition(0.5, n)
        assert P.shape == (n + 1, n + 1)


# ============================================================================
# Tests for moran_model: moran_action
# ============================================================================

class TestMoranAction:
    """Tests for applying Moran transition to a tensor."""

    def test_1d_tensor(self):
        """Apply Moran transition to a 1D tensor (vector)."""
        n = 5
        tensor = np.zeros(n + 1)
        tensor[3] = 1.0
        result = moran_action(0.0, tensor, 0)
        assert np.allclose(result, tensor, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match input shape."""
        n = 5
        tensor = np.random.rand(n + 1)
        result = moran_action(0.5, tensor, 0)
        assert result.shape == tensor.shape


# ============================================================================
# Tests for tensor_machinery: convolve_populations
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
        L1[n1] = 1.0
        L2 = np.zeros(n2 + 1)
        L2[n2] = 1.0
        L_anc = convolve_populations(L1, L2, n1, n2)
        expected = np.zeros(n1 + n2 + 1)
        expected[n1 + n2] = 1.0
        assert np.allclose(L_anc, expected, atol=1e-10)

    def test_delta_zero_convolution(self):
        """Convolving delta(0) with delta(0) should give delta(0)."""
        n1, n2 = 3, 4
        L1 = np.zeros(n1 + 1)
        L1[0] = 1.0
        L2 = np.zeros(n2 + 1)
        L2[0] = 1.0
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


# ============================================================================
# Tests for tensor_machinery: admixture_tensor
# ============================================================================

class TestAdmixtureTensor:
    """Tests for the admixture 3-tensor."""

    def test_shape(self):
        """Tensor should be (n+1) x (n+1) x (n+1)."""
        n = 5
        T = admixture_tensor(n, 0.3)
        assert T.shape == (n + 1, n + 1, n + 1)

    def test_no_admixture(self):
        """When f=0, no lineages move."""
        n = 5
        T = admixture_tensor(n, 0.0)
        for k in range(n + 1):
            assert np.isclose(T[k, 0, k], 1.0)
            for j in range(1, k + 1):
                assert np.isclose(T[k - j, j, k], 0.0)

    def test_full_admixture(self):
        """When f=1, all lineages move."""
        n = 5
        T = admixture_tensor(n, 1.0)
        for k in range(n + 1):
            assert np.isclose(T[0, k, k], 1.0)

    def test_probability_sums_to_one(self):
        """For each k, the probabilities over (i,j) with i+j=k should sum to 1."""
        n = 5
        f = 0.3
        T = admixture_tensor(n, f)
        for k in range(n + 1):
            total = sum(T[k - j, j, k] for j in range(k + 1))
            assert np.isclose(total, 1.0), f"Sum not 1 for k={k}: {total}"

    def test_expected_moving_lineages(self):
        """Expected number of lineages moving should be n*f."""
        n = 10
        f = 0.5
        T = admixture_tensor(n, f)
        expected_j = sum(j * T[n - j, j, n] for j in range(n + 1))
        assert np.isclose(expected_j, n * f, atol=1e-10)

    def test_non_negative(self):
        """All tensor entries should be non-negative."""
        n = 6
        T = admixture_tensor(n, 0.4)
        assert np.all(T >= 0)


# ============================================================================
# Tests for tensor_machinery: hypergeom_quasi_inverse
# ============================================================================

class TestHypergeomQuasiInverse:
    """Tests for the hypergeometric quasi-inverse."""

    def test_shape(self):
        """Output shape should be (N+1) x (n+1)."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        assert M.shape == (N + 1, n + 1)

    def test_rows_sum_to_one(self):
        """Each row should be a valid probability distribution."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        for i in range(N + 1):
            assert np.isclose(M[i, :].sum(), 1.0, atol=1e-10), \
                f"Row {i} sums to {M[i, :].sum()}"

    def test_non_negative(self):
        """All entries should be non-negative."""
        N, n = 8, 4
        M = hypergeom_quasi_inverse(N, n)
        assert np.all(M >= -1e-15)

    def test_boundary_zero(self):
        """When i=0 in population of N, j must be 0 when sampling n."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        assert np.isclose(M[0, 0], 1.0)
        assert np.allclose(M[0, 1:], 0.0, atol=1e-15)

    def test_boundary_N(self):
        """When i=N, j must be n."""
        N, n = 10, 5
        M = hypergeom_quasi_inverse(N, n)
        assert np.isclose(M[N, n], 1.0)
        assert np.allclose(M[N, :n], 0.0, atol=1e-15)


# ============================================================================
# Tests for inference: multinomial_log_likelihood
# ============================================================================

class TestMultinomialLogLikelihood:
    """Tests for the multinomial log-likelihood."""

    def test_perfect_match(self):
        """When observed matches expected proportionally, LL is maximized."""
        observed = np.array([10, 20, 30])
        expected = np.array([1.0, 2.0, 3.0])
        ll = multinomial_log_likelihood(observed, expected)
        assert np.isfinite(ll)

    def test_negative(self):
        """Log-likelihood should always be non-positive."""
        observed = np.array([5, 10, 15])
        expected = np.array([1.0, 2.0, 3.0])
        ll = multinomial_log_likelihood(observed, expected)
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


# ============================================================================
# Tests for inference: poisson_log_likelihood
# ============================================================================

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
        ll_true = poisson_log_likelihood(observed, observed.astype(float))
        ll_bad = poisson_log_likelihood(observed, observed.astype(float) * 2)
        assert ll_true > ll_bad


# ============================================================================
# Tests for inference: transform_params / inverse_transform
# ============================================================================

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


# ============================================================================
# Tests for inference: f2_weights
# ============================================================================

class TestF2Weights:
    """Tests for f2 weights."""

    def test_shape(self):
        """f2 weight matrix should have shape (n_A+1) x (n_B+1)."""
        W = f2_weights(5, 10)
        assert W.shape == (6, 11)

    def test_non_negative(self):
        """f2 weights are squared differences, so non-negative."""
        W = f2_weights(10, 10)
        assert np.all(W >= 0)

    def test_zero_on_diagonal_same_n(self):
        """When n_A == n_B and i == j, f2 weight should be zero."""
        n = 5
        W = f2_weights(n, n)
        for i in range(n + 1):
            assert np.isclose(W[i, i], 0.0)

    def test_symmetry(self):
        """f2(A, B) weights should satisfy W[i,j] = W_transpose[j,i]."""
        n_A, n_B = 5, 8
        W1 = f2_weights(n_A, n_B)
        W2 = f2_weights(n_B, n_A)
        assert np.allclose(W1, W2.T)


# ============================================================================
# Tests for inference: f3_weights
# ============================================================================

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
