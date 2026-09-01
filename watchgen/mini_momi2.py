"""
Mini-implementation of the momi2 algorithm for demographic inference.

momi2 is a method for demographic inference -- learning a population's history
(size changes, splits, admixture events) from patterns in its DNA variation. It
works with the site frequency spectrum (SFS) as its summary statistic, but
unlike forward-in-time methods (moments, dadi), momi2 works backward in time
through the coalescent.

The key innovation is that the expected SFS under any demographic model can be
written as a sequence of tensor operations -- matrix multiplications,
convolutions, and antidiagonal summations -- applied to likelihood tensors that
track allele configurations across populations. Population dynamics within each
epoch are governed by the Moran model, a continuous-time Markov chain whose
eigendecomposition allows efficient computation of transition probabilities for
arbitrary time spans.

The four gears of momi2:

1. The Coalescent SFS (the dial) -- The expected frequency spectrum derived
   from coalescent theory: expected branch lengths translate directly into
   expected SFS entries via the W-matrix of Polanski and Kimmel (2003).

2. The Moran Model (the escapement) -- The discrete population model that
   governs lineage dynamics within each epoch. Its eigendecomposition yields
   transition probabilities for any time span in a single matrix operation.

3. Tensor Machinery (the gear train) -- Likelihood tensors that track allele
   configurations across populations, assembled via convolution (for population
   merges), matrix multiplication (for Moran transitions), and the admixture
   3-tensor (for pulse events). A junction tree algorithm processes demographic
   events in the correct order.

4. Automatic Differentiation & Inference (the mainspring) -- Exact gradients
   flow backward through the entire tensor computation, enabling efficient
   maximum-likelihood estimation via TNC, L-BFGS-B, or stochastic methods.

References
----------
Kamm, Terhorst, and Song (2017). Efficient computation of the joint sample
frequency spectra for multiple populations.
Kamm, Terhorst, Durbin, and Song (2020). Efficiently inferring the demographic
history of many populations with allele count data.
Polanski and Kimmel (2003). New explicit expressions for relative frequencies
of single-nucleotide polymorphisms.
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import pinv
from scipy.special import comb, expi
from scipy.stats import hypergeom as hypergeom_dist

# ============================================================================
# Chapter 1: The Coalescent SFS
# ============================================================================

def w_matrix(n):
    """Compute the W-matrix of Polanski and Kimmel (2003).

    Returns W of shape (n-1, n-1), where W[b-1, j-2] gives the
    coefficient for SFS entry b from expected time with j lineages.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")

    W = np.zeros((n - 1, n - 1))
    bb = np.arange(1, n)  # SFS entries 1..n-1

    W[:, 0] = 6.0 / (n + 1)
    if n > 2:
        W[:, 1] = 30.0 * (n - 2 * bb) / ((n + 1) * (n + 2))

    # ``col`` is the zero-based recurrence index used by momi2's Cython
    # implementation.  The corresponding lineage count is col + 2, but the
    # Polanski--Kimmel recurrence itself is parameterized by ``col``.
    for col in range(2, n - 1):
        j = col
        W[:, col] = (
            W[:, col - 1] * (2 * j + 3) * (n - 2 * bb)
            / (j * (n + j + 1))
            - W[:, col - 2] * (j + 1) * (2 * j + 3) * (n - j)
            / (j * (2 * j - 1) * (n + j + 1))
        )
    return W


def etjj_constant(n, tau, N):
    """Expected time with j lineages in an epoch of duration tau and size N.

    tau: epoch duration in generations
    N: population size (constant throughout epoch)

    Returns array of length n-1, indexed by j = 2, ..., n.
    """
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if N <= 0:
        raise ValueError("N must be positive")

    j = np.arange(2, n + 1)
    rate = j * (j - 1) / N
    # Integral of the survival probability over the epoch.  The result is in
    # the same time units as tau, matching momi2's ConstantHistory.etjj.
    return -np.expm1(-rate * tau) / rate


def etjj_exponential(n, tau, growth_rate, N_bottom):
    """Expected time with j lineages under exponential growth.

    N_bottom: population size at the more recent end of the epoch
    growth_rate: exponential growth rate (positive = growing forward)
    tau: epoch duration in generations
    """
    j = np.arange(2, n + 1)
    rate = j * (j - 1) / 2.0
    if abs(growth_rate) < 1e-10:
        return etjj_constant(n, tau, N_bottom)

    if tau < 0:
        raise ValueError("tau must be non-negative")
    if N_bottom <= 0:
        raise ValueError("N_bottom must be positive")

    total_growth = tau * growth_rate
    a = rate * 2.0 / (N_bottom * growth_rate)
    # Integral_0^tau exp[-a * (exp(g*t) - 1)] dt.  momi2 evaluates an
    # algebraically equivalent, stabilized transformed-expi expression so it
    # remains differentiable under autograd.
    result = np.empty_like(rate)
    for idx, c in enumerate(a):
        if abs(c) > 100:
            # The closed form is finite here but its separate exp/Ei factors
            # overflow.  Direct quadrature is a stable educational fallback;
            # production momi2 instead uses transformed_expi.
            result[idx] = quad(
                lambda t, coefficient=c: np.exp(
                    -coefficient * np.expm1(growth_rate * t)
                ),
                0,
                tau,
            )[0]
        else:
            result[idx] = (
                np.exp(c)
                * (expi(-c * np.exp(total_growth)) - expi(-c))
                / growth_rate
            )
    return result


def compute_joint_sfs(genotype_matrix, pop_assignments, pop_names):
    """Compute the joint SFS from a genotype matrix.

    genotype_matrix: (n_sites, n_samples) array of 0/1
    pop_assignments: dict mapping sample index -> population name
    pop_names: list of population names (determines axis order)

    Returns: k-dimensional array of shape (n1+1, n2+1, ..., nk+1)
    """
    # group samples by population
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
# Chapter 2: The Moran Model
# ============================================================================

def moran_rate_matrix(n):
    """Construct the Moran model rate matrix for sample size n.

    Returns a (n+1) x (n+1) tridiagonal matrix Q where:
    - Q[i, i+1] = i*(n-i)/2   (gain one derived copy)
    - Q[i, i-1] = i*(n-i)/2   (lose one derived copy)
    - Q[i, i]   = -i*(n-i)    (total departure rate)
    """
    i = np.arange(n + 1, dtype=float)
    off_diag = i * (n - i) / 2.0
    diag = -2.0 * off_diag
    Q = (np.diag(off_diag[:-1], k=1)
         + np.diag(diag, k=0)
         + np.diag(off_diag[1:], k=-1))
    return Q


def moran_eigensystem(n):
    """Compute the eigendecomposition of the Moran rate matrix.

    Returns (V, eigenvalues, V_inv) where Q = V @ diag(eigenvalues) @ V_inv.
    """
    Q = moran_rate_matrix(n)
    eigenvalues, V = np.linalg.eig(Q)
    idx = np.argsort(eigenvalues.real)
    eigenvalues = eigenvalues[idx].real
    V = V[:, idx].real
    V_inv = np.linalg.inv(V)
    return V, eigenvalues, V_inv


def moran_transition(t, n):
    """Compute the Moran transition matrix P(t) = exp(Q*t).

    t: time (in Moran model units, scaled by 2/N)
    n: sample size

    Returns (n+1) x (n+1) transition probability matrix.
    """
    V, eigs, V_inv = moran_eigensystem(n)
    D = np.diag(np.exp(t * eigs))
    P = V @ D @ V_inv
    # clamp small numerical errors
    P = np.clip(P, 0, None)
    P = P / P.sum(axis=1, keepdims=True)  # normalize rows
    return P


def moran_action(t, tensor, axis):
    """Apply Moran transition matrix to a tensor along a given axis.

    t: scaled time for this epoch
    tensor: multi-dimensional likelihood tensor
    axis: which axis (population) to apply the transition to

    Returns tensor with the Moran transition applied.
    """
    n = tensor.shape[axis] - 1
    P = moran_transition(t, n)
    # Contract along the specified axis and move result back to same position
    result = np.tensordot(tensor, P.T, axes=([axis], [0]))
    # tensordot places the contracted axis last; move it back to original position
    if axis != result.ndim - 1:
        result = np.moveaxis(result, -1, axis)
    return result


# ============================================================================
# Chapter 3: Tensor Machinery
# ============================================================================

def convolve_populations(L1, L2, n1, n2):
    """Merge two population tensors via convolution.

    L1: likelihood vector for child population 1, length n1+1
    L2: likelihood vector for child population 2, length n2+1

    Returns: likelihood vector for ancestral population, length n1+n2+1
    """
    # weight by binomial coefficients
    b1 = np.array([comb(n1, j, exact=True) for j in range(n1 + 1)])
    b2 = np.array([comb(n2, k, exact=True) for k in range(n2 + 1)])
    weighted_L1 = L1 * b1
    weighted_L2 = L2 * b2

    # convolve (polynomial multiplication)
    conv = np.convolve(weighted_L1, weighted_L2)

    # divide out ancestral binomial coefficients
    n_anc = n1 + n2
    b_anc = np.array([comb(n_anc, i, exact=True) for i in range(n_anc + 1)])
    L_anc = conv / b_anc

    return L_anc


def admixture_tensor(n, f):
    """Compute the admixture 3-tensor for a pulse event.

    n: virtual Moran sample size on the child and each parent axis
    f: fraction of child ancestry inherited from parent 2

    Returns T of shape (n+1, n+1, n+1). T[a, b, k] is the probability
    of k derived copies in the child conditional on a and b derived copies in
    parent 1 and parent 2.  It marginalizes over the binomially distributed
    number of child lineages assigned to each parent and the hypergeometric
    sampling of derived copies within each parent.
    """
    if not 0 <= f <= 1:
        raise ValueError("f must lie in [0, 1]")

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
                if assign_prob == 0:
                    continue
                n_parent2 = n - n_parent1
                T[a, b] += assign_prob * np.convolve(
                    draws[a, n_parent1], draws[b, n_parent2]
                )
    return T


def hypergeom_quasi_inverse(N, n):
    """Compute the quasi-inverse for reducing lineage count from N to n.

    Returns the (N+1) x (n+1) Moore--Penrose pseudoinverse of the
    hypergeometric projection from N to n copies.  Unlike the projection
    itself, this quasi-inverse is a linear-algebra operator, not a probability
    matrix, and may contain negative entries.
    """
    if not 0 <= n <= N:
        raise ValueError("sample sizes must satisfy 0 <= n <= N")
    projection = np.zeros((n + 1, N + 1))
    for derived_sample in range(n + 1):
        for derived_population in range(N + 1):
            projection[derived_sample, derived_population] = hypergeom_dist.pmf(
                derived_sample, N, derived_population, n
            )
    return pinv(projection)


# ============================================================================
# Chapter 4: Automatic Differentiation & Inference
# ============================================================================

def multinomial_log_likelihood(observed_sfs, expected_sfs):
    """Composite log-likelihood under the multinomial model.

    observed_sfs: array of observed configuration counts
    expected_sfs: array of expected proportions (normalized to sum to 1)
    """
    # normalize expected SFS to probabilities
    expected_probs = expected_sfs / expected_sfs.sum()
    # avoid log(0) by masking zero-count entries
    mask = observed_sfs > 0
    ll = np.sum(observed_sfs[mask] * np.log(expected_probs[mask]))
    return ll


def poisson_log_likelihood(observed_sfs, expected_sfs):
    """Composite log-likelihood under the Poisson model.

    expected_sfs: array of expected counts (not normalized)
    """
    mask = observed_sfs > 0
    ll = np.sum(
        observed_sfs[mask] * np.log(expected_sfs[mask])
        - expected_sfs[mask]
    )
    # Subtract expected counts for zero-observation bins
    ll -= np.sum(expected_sfs[~mask])
    return ll


def transform_params(params, param_types):
    """Transform parameters to unconstrained space.

    param_types: list of 'log' (positive), 'logit' (0-1), or 'none'
    """
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
    """Weight vector for f2(A, B) = E[(p_A - p_B)^2].

    Uses two draws without replacement within each population, as momi2's
    ordered-probability definition does.  This removes finite-sample binomial
    variance from the plug-in squared frequency difference.
    """
    if n_A < 2 or n_B < 2:
        raise ValueError("f2 requires at least two sampled copies per population")
    p_A = np.arange(n_A + 1) / n_A
    p_B = np.arange(n_B + 1) / n_B
    p_A2 = np.arange(n_A + 1) * (np.arange(n_A + 1) - 1) / (n_A * (n_A - 1))
    p_B2 = np.arange(n_B + 1) * (np.arange(n_B + 1) - 1) / (n_B * (n_B - 1))
    return p_A2[:, None] + p_B2[None, :] - 2 * p_A[:, None] * p_B[None, :]


def f3_weights(n_C, n_A, n_B):
    """Weight vector for f3(C; A, B) = E[(p_C - p_A)(p_C - p_B)].

    Negative f3 indicates admixture of C from A and B.
    """
    if n_C < 2 or n_A < 1 or n_B < 1:
        raise ValueError("f3 requires two C copies and one copy from A and B")
    p_C = np.arange(n_C + 1) / n_C
    p_A = np.arange(n_A + 1) / n_A
    p_B = np.arange(n_B + 1) / n_B
    p_C2 = np.arange(n_C + 1) * (np.arange(n_C + 1) - 1) / (n_C * (n_C - 1))
    # 3-way outer product
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
# Demo
# ============================================================================

def demo():
    """Demonstrate the momi2 mini-implementation."""

    print("=" * 70)
    print("momi2 Mini-Implementation Demo")
    print("=" * 70)

    # --- Coalescent SFS ---
    print("\n--- Chapter 1: The Coalescent SFS ---\n")

    # Demonstrate W-matrix properties
    n = 10
    W = w_matrix(n)
    print(f"W-matrix shape (n={n}): {W.shape}")
    print(f"  First column W[:, 0] = 6/(n+1) = {6.0 / (n + 1):.6f}: "
          f"{'OK' if np.allclose(W[:, 0], 6.0 / (n + 1)) else 'FAIL'}")

    # Expected SFS under constant population: should be positive and decreasing
    j_vals = np.arange(2, n + 1)
    E_Tjj_neutral = 2.0 / (j_vals * (j_vals - 1))
    expected_sfs = W @ E_Tjj_neutral
    assert np.all(expected_sfs > 0), "Expected SFS should be positive"
    assert expected_sfs[0] > expected_sfs[-1], "SFS[1] > SFS[n-1] under neutrality"
    print("  Neutral SFS positive and decreasing: OK")
    print(f"  SFS entries (first 5): {expected_sfs[:5]}")

    # Constant vs exponential growth coalescence times
    tau, N = 500, 1000
    etjj_c = etjj_constant(n, tau, N)
    etjj_e = etjj_exponential(n, tau, 0.0, N)
    assert np.allclose(etjj_c, etjj_e, rtol=1e-6), "Zero growth should equal constant"
    print("  etjj_constant == etjj_exponential(growth=0): OK")

    # Joint SFS demo
    np.random.seed(42)
    G = np.random.binomial(1, 0.3, size=(50, 8))
    pop_assign = {i: 'A' if i < 4 else 'B' for i in range(8)}
    sfs = compute_joint_sfs(G, pop_assign, ['A', 'B'])
    print(f"  Joint SFS shape (4+4 samples): {sfs.shape}, total sites: {sfs.sum()}")

    # --- Moran Model ---
    print("\n--- Chapter 2: The Moran Model ---\n")

    # Verify rate matrix rows sum to zero
    n_m = 10
    Q = moran_rate_matrix(n_m)
    row_sums = Q.sum(axis=1)
    assert np.allclose(row_sums, 0), f"Row sums: {row_sums}"
    print(f"Rate matrix (n={n_m}): rows sum to zero -- OK")

    # Verify absorbing states
    assert np.allclose(Q[0, :], 0), "State 0 should be absorbing"
    assert np.allclose(Q[n_m, :], 0), "State n should be absorbing"
    print(f"Rate matrix (n={n_m}): absorbing states at 0 and n -- OK")

    # Verify eigenvalues match theoretical formula
    _V, eigs, _V_inv = moran_eigensystem(n_m)
    j = np.arange(n_m + 1)
    theoretical_eigs = -j * (j - 1) / 2.0
    assert np.allclose(np.sort(eigs), np.sort(theoretical_eigs), atol=1e-10)
    print(f"Eigenvalues (n={n_m}): match theoretical formula -- OK")

    # Verify P(0) = identity
    P0 = moran_transition(0, n_m)
    assert np.allclose(P0, np.eye(n_m + 1), atol=1e-10)
    print("P(0) = identity -- OK")

    # Verify Chapman-Kolmogorov
    P_s = moran_transition(0.5, n_m)
    P_t = moran_transition(0.3, n_m)
    P_st = moran_transition(0.8, n_m)
    assert np.allclose(P_s @ P_t, P_st, atol=1e-8)
    print("Chapman-Kolmogorov P(0.5)*P(0.3) = P(0.8) -- OK")

    # Fixation probability check
    t_large = 100.0
    P_large = moran_transition(t_large, n_m)
    print(f"\nFixation probabilities (n={n_m}, large t):")
    for i_state in range(n_m + 1):
        fix_prob = P_large[i_state, n_m]
        expected_fix = i_state / n_m
        print(f"  i={i_state}: P(fixation) = {fix_prob:.6f}, expected = {expected_fix:.6f}")

    # --- Tensor Machinery ---
    print("\n--- Chapter 3: Tensor Machinery ---\n")

    # Convolution verification: delta(n1) * delta(n2) = delta(n1+n2)
    n1, n2 = 3, 4
    L1 = np.zeros(n1 + 1)
    L1[n1] = 1.0
    L2 = np.zeros(n2 + 1)
    L2[n2] = 1.0
    L_anc = convolve_populations(L1, L2, n1, n2)
    expected_anc = np.zeros(n1 + n2 + 1)
    expected_anc[n1 + n2] = 1.0
    assert np.allclose(L_anc, expected_anc, atol=1e-10)
    print(f"Convolution delta({n1}) * delta({n2}) = delta({n1+n2}) -- OK")

    # Admixture tensor verification
    n_adm = 4
    f_adm = 0.5
    T = admixture_tensor(n_adm, f_adm)
    print(f"\nAdmixture tensor (n={n_adm}, f={f_adm}):")
    a, b = 1, 3
    total = T[a, b].sum()
    expected_child = np.dot(np.arange(n_adm + 1), T[a, b])
    theoretical_child = (1 - f_adm) * a + f_adm * b
    print(f"  parents ({a}, {b}): sum={total:.6f}, "
          f"E[child]={expected_child:.4f}, expected={theoretical_child:.4f}")

    # Hypergeometric quasi-inverse
    N_hyp, n_hyp = 10, 5
    M = hypergeom_quasi_inverse(N_hyp, n_hyp)
    print(f"\nHypergeometric quasi-inverse ({N_hyp} -> {n_hyp}):")
    projection = np.array([
        [hypergeom_dist.pmf(jj, N_hyp, ii, n_hyp)
         for ii in range(N_hyp + 1)]
        for jj in range(n_hyp + 1)
    ])
    print(f"  max |H H+ - I|: "
          f"{np.max(np.abs(projection @ M - np.eye(n_hyp + 1))):.3e}")

    # --- Inference ---
    print("\n--- Chapter 4: Inference ---\n")

    # Multinomial log-likelihood
    observed = np.array([10, 20, 30])
    expected_sfs_vals = np.array([1.0, 2.0, 3.0])
    ll = multinomial_log_likelihood(observed, expected_sfs_vals)
    print(f"Multinomial LL (proportional match): {ll:.6f}")

    # Parameter transform roundtrip
    params = np.array([5.0, 0.3, -2.0])
    types = ['log', 'logit', 'none']
    transformed = transform_params(params, types)
    recovered = inverse_transform(transformed, types)
    assert np.allclose(params, recovered)
    print(f"Parameter transform roundtrip: {params} -> {transformed} -> {recovered} -- OK")

    # f2 weights
    W_f2 = f2_weights(5, 5)
    print(f"\nf2 weights shape: {W_f2.shape}")
    print("f2 diagonal (signed finite-sample corrections): "
          f"{[W_f2[i, i] for i in range(6)]}")

    # f3 weights
    W_f3 = f3_weights(5, 5, 5)
    print(f"f3 weights shape: {W_f3.shape}")
    print(f"f3[n, 0, 0] = {W_f3[5, 0, 0]:.4f} (expected 1.0)")

    print("\n" + "=" * 70)
    print("All demos passed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
