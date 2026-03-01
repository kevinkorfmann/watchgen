"""
Mini-tsdate: Bayesian node dating for tree sequences.

tsdate is a Bayesian method for estimating the age of every ancestral node
in a tree sequence. Given a genealogy (typically the topology from tsinfer,
which has no meaningful branch lengths), tsdate uses the molecular clock --
the principle that mutations accumulate proportionally to time -- to infer
when each ancestor lived.

The four gears of tsdate:

1. **The Coalescent Prior** -- A prior on node ages derived from coalescent
   theory: nodes with more descendant samples are expected to be younger,
   because large subtrees coalesce quickly.

2. **The Mutation Likelihood** -- A Poisson model for the number of mutations
   on each edge, connecting observed data to branch lengths.

3. **Belief Propagation** -- Message-passing algorithms that combine prior and
   likelihood across the interconnected nodes of the tree sequence:
   - Inside-Outside (discrete time grid)
   - Variational Gamma (continuous time, default)

4. **Rescaling** -- A post-processing step that adjusts node times so the
   inferred mutation rate matches the empirical rate across time windows.

This module extracts all self-contained code from the tsdate documentation
chapters:
  - coalescent_prior.rst
  - mutation_likelihood.rst
  - inside_outside.rst
  - variational_gamma.rst
  - rescaling.rst
"""

import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp, comb, gammaln, digamma, polygamma
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist


# =========================================================================
# Chapter 1: Coalescent Prior (coalescent_prior.rst)
# =========================================================================

def conditional_coalescent_mean(k, n, Ne=1.0):
    """Mean age of a node with k descendants in a sample of n.

    Under the conditional coalescent (Wiuf & Donnelly, 1999), averaged
    over the number of extant ancestors.

    Parameters
    ----------
    k : int
        Number of descendant leaves of this node.
    n : int
        Total number of leaves in the tree.
    Ne : float
        Effective population size (in coalescent units, 2*Ne generations).

    Returns
    -------
    mean : float
        Expected age in units of 2*Ne generations.
    """
    if k == n:
        # The root: must wait for all n lineages to coalesce
        # Mean is sum of 1/(j choose 2) for j = n down to 2
        return sum(2.0 / (j * (j - 1)) for j in range(2, n + 1))

    # P(a ancestors | k descendants coalesce, n total tips)
    # computed recursively
    mean = 0.0
    for a in range(2, n - k + 2):
        # Probability of a ancestors when subtree of size k merges
        p_a = _pr_ancestors(a, k, n)
        # Expected coalescence time given a lineages
        expected_time = 2.0 / (a * (a - 1))
        mean += p_a * expected_time

    return mean


def _pr_ancestors(a, k, n):
    """Probability of a extant ancestors when subtree of size k coalesces.

    This follows Wiuf & Donnelly (1999). For a subtree of size k in a
    tree of n tips, the number of other lineages when k coalesces to 1
    ranges from 1 to n-k. So total ancestors a ranges from 2 to n-k+1.
    """
    if k == 2:
        # Special case: the pair coalesces when a-1 other lineages exist
        # at that time, so a total. This has a known distribution.
        pass
    # In practice, tsdate computes this recursively using the relationship:
    # P(a | k, n) can be computed from P(a | k+1, n) using
    # binomial coefficient identities.
    # For educational purposes, here's a direct simulation approach:
    raise NotImplementedError(
        "See the recursive implementation below for the full computation."
    )


def _transition_prob(a_prime, a):
    """Transition probability in the Wiuf-Donnelly recursion.

    Probability that when one more pair coalesces (decreasing k by 1),
    the number of ancestors changes from a' to a.
    """
    if a > a_prime or a < 2:
        return 0.0
    if a == a_prime:
        # The coalescing pair was entirely within the subtree
        return (a_prime - 1) / (a_prime + 1)
    if a == a_prime - 1:
        # One of the coalescing lineages was in the subtree,
        # the other was not, reducing total ancestors by 1
        return 2.0 / (a_prime + 1)
    return 0.0


def conditional_coalescent_moments(n, Ne=1.0):
    """Compute mean and variance of node age for all possible descendant counts.

    Parameters
    ----------
    n : int
        Total number of tips.
    Ne : float
        Effective population size.

    Returns
    -------
    moments : dict
        {k: (mean, variance)} for k = 2, 3, ..., n.
    """
    # Precompute unconditional coalescence time moments for a lineages
    # E[T | a] = 2/(a*(a-1)),  Var[T | a] = E[T|a]^2 = 4/(a*(a-1))^2
    max_a = n
    t_mean = np.zeros(max_a + 1)
    t_var = np.zeros(max_a + 1)
    for a in range(2, max_a + 1):
        rate = a * (a - 1) / 2.0
        t_mean[a] = 1.0 / rate
        t_var[a] = 1.0 / rate**2

    # Build P(a | k, n) table recursively from k=n-1 down to k=2
    pr_a = {}
    pr_a[n - 1] = np.zeros(max_a + 1)
    pr_a[n - 1][2] = 1.0

    for k in range(n - 2, 1, -1):
        pr_a[k] = np.zeros(max_a + 1)
        for a in range(2, n - k + 2):
            for a_prime in range(a, n - k + 1):
                if pr_a[k + 1][a_prime] > 0:
                    transition = _transition_prob(a_prime, a)
                    pr_a[k][a] += pr_a[k + 1][a_prime] * transition

            # Normalize to ensure probabilities sum to 1
        total = pr_a[k].sum()
        if total > 0:
            pr_a[k] /= total

    # Compute moments by averaging over a (law of total expectation/variance)
    moments = {}
    for k in range(2, n):
        mean = np.sum(pr_a[k] * t_mean)
        e_t_sq = np.sum(pr_a[k] * (t_var + t_mean**2))
        variance = e_t_sq - mean**2
        moments[k] = (mean, variance)

    # Root (k=n): sum of all waiting times from n lineages down to 1
    root_mean = sum(2.0 / (j * (j - 1)) for j in range(2, n + 1))
    root_var = sum(4.0 / (j * (j - 1))**2 for j in range(2, n + 1))
    moments[n] = (root_mean, root_var)

    return moments


def gamma_params_from_moments(mean, variance):
    """Convert mean and variance to gamma distribution parameters.

    Parameters
    ----------
    mean : float
        E[T] from the conditional coalescent.
    variance : float
        Var[T] from the conditional coalescent.

    Returns
    -------
    alpha : float
        Shape parameter (controls peakedness of the distribution).
    beta : float
        Rate parameter (controls how quickly the density decays).
    """
    alpha = mean**2 / variance
    beta = mean / variance
    return alpha, beta


def build_prior_grid(n, Ne=1.0):
    """Build a lookup table of gamma priors indexed by descendant count.

    Parameters
    ----------
    n : int
        Total number of sample leaves.
    Ne : float
        Effective population size.

    Returns
    -------
    prior_grid : np.ndarray, shape (n+1, 4)
        Columns: [alpha, beta, mean, variance]
        Row k gives the prior for a node with k descendants.
        Rows 0 and 1 are unused (no node has 0 or 1 non-self descendants).
    """
    grid = np.zeros((n + 1, 4))
    moments = conditional_coalescent_moments(n, Ne)

    for k in range(2, n + 1):
        mean, var = moments[k]
        alpha, beta = gamma_params_from_moments(mean, var)
        grid[k] = [alpha, beta, mean, var]

    return grid


# =========================================================================
# Chapter 2: Mutation Likelihood (mutation_likelihood.rst)
# =========================================================================

def edge_likelihood(m_e, lambda_e, t_parent, t_child):
    """Poisson likelihood for mutations on a single edge.

    Parameters
    ----------
    m_e : int
        Observed mutation count on this edge.
    lambda_e : float
        Span-weighted mutation rate (mu * span_bp).
    t_parent : float
        Age of parent node.
    t_child : float
        Age of child node.

    Returns
    -------
    likelihood : float
        P(m_e | t_parent, t_child).
    """
    delta_t = t_parent - t_child
    if delta_t <= 0:
        return 0.0
    expected = lambda_e * delta_t
    return poisson.pmf(m_e, expected)


def gamma_poisson_update(alpha_prior, beta_prior, m_e, lambda_e):
    """Update gamma parameters given Poisson observations.

    The gamma-Poisson conjugacy gives a closed-form posterior.

    Parameters
    ----------
    alpha_prior, beta_prior : float
        Prior gamma parameters.
    m_e : int
        Observed mutations.
    lambda_e : float
        Span-weighted mutation rate.

    Returns
    -------
    alpha_post, beta_post : float
        Posterior gamma parameters.
    """
    return alpha_prior + m_e, beta_prior + lambda_e


# =========================================================================
# Chapter 3: Inside-Outside Belief Propagation (inside_outside.rst)
# =========================================================================

def make_time_grid(n, Ne=1.0, num_points=20, grid_type="logarithmic"):
    """Create a time grid for the inside-outside algorithm.

    Parameters
    ----------
    n : int
        Number of samples (sets the expected TMRCA).
    Ne : float
        Effective population size.
    num_points : int
        Number of grid points.
    grid_type : str
        "linear" or "logarithmic".

    Returns
    -------
    grid : np.ndarray
        Array of timepoints, starting at 0.
    """
    # Expected TMRCA under standard coalescent: 2*Ne*(1 - 1/n)
    expected_tmrca = 2 * Ne * (1 - 1.0 / n)
    t_max = expected_tmrca * 4  # go well beyond expected TMRCA

    if grid_type == "linear":
        return np.linspace(0, t_max, num_points)
    else:
        # Log-spaced: more points near 0, fewer far out
        # Start from a small positive number to avoid log(0)
        t_min = t_max / (10 * num_points)
        return np.concatenate([[0], np.geomspace(t_min, t_max, num_points - 1)])


def edge_likelihood_matrix(m_e, lambda_e, grid):
    """Compute the likelihood matrix for an edge on the time grid.

    Parameters
    ----------
    m_e : int
        Mutation count on this edge.
    lambda_e : float
        Span-weighted mutation rate (mu * span_bp).
    grid : np.ndarray
        Time grid.

    Returns
    -------
    L : np.ndarray, shape (K, K)
        L[i, j] = P(m_e | parent_time=grid[i], child_time=grid[j])
        Lower triangular (i >= j).
    """
    K = len(grid)
    L = np.zeros((K, K))

    for i in range(K):
        for j in range(i + 1):  # j <= i (child younger than parent)
            delta_t = grid[i] - grid[j]
            if delta_t > 0:
                expected = lambda_e * delta_t
                L[i, j] = poisson.pmf(m_e, expected)
            elif m_e == 0:
                # delta_t = 0, only possible if no mutations
                L[i, j] = 1.0

    return L


def compute_posteriors(inside, outside):
    """Combine inside and outside to get marginal posteriors.

    Parameters
    ----------
    inside : np.ndarray, shape (num_nodes, K)
    outside : np.ndarray, shape (num_nodes, K)

    Returns
    -------
    posterior : np.ndarray, shape (num_nodes, K)
        posterior[u, :] is the marginal posterior distribution over
        grid points for node u.
    """
    posterior = inside * outside  # element-wise product

    # Normalize each node's posterior to sum to 1
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    posterior /= row_sums

    return posterior


def posterior_mean(posterior, grid):
    """Compute posterior mean age for each node.

    Parameters
    ----------
    posterior : np.ndarray, shape (num_nodes, K)
    grid : np.ndarray, shape (K,)

    Returns
    -------
    means : np.ndarray, shape (num_nodes,)
        E[t_u | D] for each node.
    """
    return posterior @ grid  # weighted sum: sum_i posterior[u,i] * grid[i]


def inside_pass_logspace(inside_log, L_log, K):
    """Compute a single inside message in log space.

    Parameters
    ----------
    inside_log : np.ndarray, shape (K,)
        Log inside values for child node.
    L_log : np.ndarray, shape (K, K)
        Log likelihood matrix.

    Returns
    -------
    msg_log : np.ndarray, shape (K,)
        Log message from child to parent.
    """
    msg_log = np.full(K, -np.inf)    # start at log(0) = -inf
    for i in range(K):
        terms = L_log[i, :i + 1] + inside_log[:i + 1]
        msg_log[i] = logsumexp(terms)
    return msg_log


# =========================================================================
# Chapter 4: Variational Gamma / Expectation Propagation
#             (variational_gamma.rst)
# =========================================================================

class GammaDistribution:
    """A gamma distribution in natural parameterization.

    Natural parameters: eta1 = alpha - 1, eta2 = -beta
    Standard parameters: alpha (shape), beta (rate)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    @property
    def eta1(self):
        """First natural parameter: alpha - 1."""
        return self.alpha - 1

    @property
    def eta2(self):
        """Second natural parameter: -beta."""
        return -self.beta

    @property
    def mean(self):
        """E[t] = alpha / beta."""
        return self.alpha / self.beta

    @property
    def variance(self):
        """Var(t) = alpha / beta^2."""
        return self.alpha / self.beta**2

    @property
    def log_mean(self):
        """E[log t] = digamma(alpha) - log(beta)"""
        return digamma(self.alpha) - np.log(self.beta)

    def multiply(self, other):
        """Multiply two gamma factors (add natural parameters).

        In natural parameter space: (eta1, eta2) + (eta1', eta2')
        In standard parameters: alpha_new = alpha + alpha' - 1,
                                beta_new = beta + beta'
        """
        new_alpha = self.alpha + other.alpha - 1
        new_beta = self.beta + other.beta
        return GammaDistribution(new_alpha, new_beta)

    def divide(self, other):
        """Divide by a gamma factor (subtract natural parameters).

        This is the inverse of multiply: removing a factor's contribution.
        """
        new_alpha = self.alpha - other.alpha + 1
        new_beta = self.beta - other.beta
        return GammaDistribution(new_alpha, new_beta)

    @classmethod
    def from_moments(cls, mean, variance):
        """Create from mean and variance via moment matching.

        Uses the standard method-of-moments estimator:
        beta = mean / variance, alpha = mean * beta
        """
        beta = mean / variance
        alpha = mean * beta
        return cls(alpha, beta)


def numerical_hessian(f, x, eps=1e-5):
    """Compute the Hessian of f at x via finite differences.

    Uses the standard 4-point formula for mixed partial derivatives:
    d^2f/dxidxj ~ (f(+,+) - f(+,-) - f(-,+) + f(-,-)) / (4*eps^2)
    """
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


def compute_tilted_moments(cavity_u, cavity_v, m_e, lambda_e):
    """Compute moments of the tilted distribution via Laplace approximation.

    Parameters
    ----------
    cavity_u, cavity_v : GammaDistribution
        Cavity distributions for parent and child.
    m_e : int
        Mutation count.
    lambda_e : float
        Span-weighted mutation rate.

    Returns
    -------
    mu_u, var_u, mu_v, var_v : float
        Moments of the tilted marginals, or None if numerical failure.
    """
    def neg_log_tilted(params):
        """Negative log of the tilted distribution (to be minimized)."""
        t_u, t_v = params
        if t_u <= t_v or t_u <= 0 or t_v < 0:
            return 1e20

        delta = t_u - t_v

        # Log cavity contributions (gamma log-pdf, unnormalized)
        log_cavity_u = (cavity_u.alpha - 1) * np.log(t_u) - cavity_u.beta * t_u
        log_cavity_v = (cavity_v.alpha - 1) * np.log(max(t_v, 1e-20)) - cavity_v.beta * t_v

        # Log Poisson likelihood: m*log(lambda*delta) - lambda*delta
        log_lik = m_e * np.log(lambda_e * delta) - lambda_e * delta

        return -(log_cavity_u + log_cavity_v + log_lik)

    # Initial guess: cavity means
    t_u_init = max(cavity_u.mean, 1e-6)
    t_v_init = max(cavity_v.mean, 1e-6)
    if t_u_init <= t_v_init:
        t_u_init = t_v_init + 1.0

    result = minimize(neg_log_tilted, [t_u_init, t_v_init],
                      method='Nelder-Mead')

    if not result.success:
        return None

    t_u_hat, t_v_hat = result.x

    # Compute Hessian numerically for the Laplace approximation
    H = numerical_hessian(neg_log_tilted, [t_u_hat, t_v_hat])

    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    mu_u = t_u_hat
    var_u = max(cov[0, 0], 1e-20)
    mu_v = t_v_hat
    var_v = max(cov[1, 1], 1e-20)

    return mu_u, var_u, mu_v, var_v


# =========================================================================
# Chapter 5: Rescaling (rescaling.rst)
# =========================================================================

def compute_scaling_factors(observed, expected, min_count=1.0):
    """Compute per-window scaling factors.

    Parameters
    ----------
    observed, expected : np.ndarray, shape (J,)
    min_count : float
        Minimum mutation count to trust a window.

    Returns
    -------
    scales : np.ndarray, shape (J,)
    """
    scales = np.ones(len(observed))

    for j in range(len(observed)):
        if expected[j] > 0 and observed[j] >= min_count:
            scales[j] = observed[j] / expected[j]

    return scales


def apply_rescaling(node_times, breakpoints, scales, fixed_nodes):
    """Apply piecewise rescaling to node times.

    Parameters
    ----------
    node_times : np.ndarray
        Current node times (will not be modified).
    breakpoints : np.ndarray, shape (J+1,)
    scales : np.ndarray, shape (J,)
    fixed_nodes : set
        Nodes whose times should not change (e.g., samples).

    Returns
    -------
    new_times : np.ndarray
        Rescaled node times.
    """
    new_times = np.zeros_like(node_times)
    J = len(scales)

    # Build cumulative scaling function
    cum_rescaled = np.zeros(J + 1)
    for j in range(J):
        window_width = breakpoints[j + 1] - breakpoints[j]
        cum_rescaled[j + 1] = cum_rescaled[j] + scales[j] * window_width

    for u in range(len(node_times)):
        if u in fixed_nodes:
            new_times[u] = node_times[u]
            continue

        t = node_times[u]

        # Find which window t falls in
        j = np.searchsorted(breakpoints, t, side='right') - 1
        j = min(j, J - 1)
        j = max(j, 0)

        # Rescaled time = cumulative up to window j + fraction within window
        fraction_in_window = t - breakpoints[j]
        new_times[u] = cum_rescaled[j] + scales[j] * fraction_in_window

    return new_times


# =========================================================================
# Demo function
# =========================================================================

def demo():
    """Demonstrate the tsdate mini-implementation.

    Runs code from the RST documentation chapters, illustrating:
    - Coalescent prior computation
    - Gamma-Poisson conjugacy
    - Time grid construction
    - Edge likelihood matrix
    - Inside-outside belief propagation components
    - Variational gamma (GammaDistribution class)
    - Rescaling
    """
    print("=" * 60)
    print("Timepiece IX: tsdate -- Dating Nodes in a Tree Sequence")
    print("=" * 60)

    # --- Chapter 1: Coalescent Prior ---
    print("\n--- Chapter 1: Coalescent Prior ---")

    k, n = 3, 100
    approx_mean = 2.0 / (k * (k - 1))
    approx_var = approx_mean**2

    alpha, beta = gamma_params_from_moments(approx_mean, approx_var)
    print(f"k={k}: mean={approx_mean:.4f}, var={approx_var:.4f}")
    print(f"  Gamma prior: alpha={alpha:.4f}, beta={beta:.4f}")

    # Compute moments for a small tree
    n_small = 10
    moments = conditional_coalescent_moments(n_small)
    print(f"\nConditional coalescent moments for n={n_small}:")
    for k_val in sorted(moments.keys()):
        m, v = moments[k_val]
        print(f"  k={k_val}: mean={m:.4f}, var={v:.4f}")

    # Build prior grid
    prior_grid = build_prior_grid(n_small)
    print(f"\nPrior grid for n={n_small} (first 5 rows with k>=2):")
    for k_val in range(2, min(7, n_small + 1)):
        print(f"  k={k_val}: alpha={prior_grid[k_val, 0]:.4f}, "
              f"beta={prior_grid[k_val, 1]:.4f}, "
              f"mean={prior_grid[k_val, 2]:.4f}")

    # --- Chapter 2: Mutation Likelihood ---
    print("\n--- Chapter 2: Mutation Likelihood ---")

    mu = 1e-8
    span_bp = 10_000
    lambda_e = mu * span_bp
    delta_t = 500
    expected_mutations = lambda_e * delta_t

    print(f"Expected mutations: {expected_mutations:.4f}")
    print(f"P(0 mutations) = {edge_likelihood(0, lambda_e, 500, 0):.6f}")
    print(f"P(1 mutation)  = {edge_likelihood(1, lambda_e, 500, 0):.6f}")
    print(f"P(2 mutations) = {edge_likelihood(2, lambda_e, 500, 0):.6f}")

    # Gamma-Poisson conjugacy
    alpha_prior, beta_prior = 2.0, 3.0
    m_e, lambda_e_conj = 5, 0.01
    alpha_post, beta_post = gamma_poisson_update(alpha_prior, beta_prior,
                                                  m_e, lambda_e_conj)

    print(f"\nPrior:     Gamma({alpha_prior}, {beta_prior})")
    print(f"  mean = {alpha_prior / beta_prior:.4f}")
    print(f"Posterior: Gamma({alpha_post}, {beta_post})")
    print(f"  mean = {alpha_post / beta_post:.4f}")

    # --- Chapter 3: Inside-Outside ---
    print("\n--- Chapter 3: Inside-Outside Belief Propagation ---")

    grid = make_time_grid(n=100, num_points=20)
    print(f"Grid: {grid[:5]} ... {grid[-3:]}")
    print(f"Grid spans [0, {grid[-1]:.2f}] with {len(grid)} points")

    grid_io = make_time_grid(n=50, num_points=10)
    L = edge_likelihood_matrix(m_e=2, lambda_e=0.001, grid=grid_io)
    print(f"Likelihood matrix shape: {L.shape}")
    print(f"Max likelihood at parent_idx, child_idx = "
          f"{np.unravel_index(L.argmax(), L.shape)}")

    # Test compute_posteriors and posterior_mean
    K = 15
    n_nodes = 5
    grid_demo = make_time_grid(n=50, num_points=K, grid_type="logarithmic")
    np.random.seed(42)
    inside = np.random.rand(n_nodes, K) + 0.01
    outside = np.random.rand(n_nodes, K) + 0.01
    post = compute_posteriors(inside, outside)
    means = posterior_mean(post, grid_demo)
    print(f"\nPosterior means for {n_nodes} mock nodes: {means}")

    # --- Chapter 4: Variational Gamma ---
    print("\n--- Chapter 4: Variational Gamma (EP) ---")

    # Verify gamma product rule
    a1, b1 = 3.0, 2.0
    a2, b2 = 2.0, 1.5

    x = np.linspace(0.01, 5.0, 1000)

    f1 = gamma_dist.pdf(x, a=a1, scale=1 / b1)
    f2 = gamma_dist.pdf(x, a=a2, scale=1 / b2)
    product = f1 * f2

    a_new, b_new = a1 + a2 - 1, b1 + b2
    f_new = gamma_dist.pdf(x, a=a_new, scale=1 / b_new)

    ratio = product / f_new
    ratio = ratio[f_new > 1e-10]
    print(f"Product is Gamma({a_new}, {b_new})")
    print(f"Ratio min={ratio.min():.6f}, max={ratio.max():.6f} (should be constant)")

    # GammaDistribution class
    g1 = GammaDistribution(3.0, 2.0)
    g2 = GammaDistribution(2.0, 1.5)
    g_prod = g1.multiply(g2)
    print(f"\nGamma({g1.alpha}, {g1.beta}) * Gamma({g2.alpha}, {g2.beta}) "
          f"= Gamma({g_prod.alpha}, {g_prod.beta})")
    print(f"  mean = {g_prod.mean:.4f}, var = {g_prod.variance:.4f}")

    g_from_mom = GammaDistribution.from_moments(2.0, 0.5)
    print(f"From moments (mean=2.0, var=0.5): "
          f"Gamma({g_from_mom.alpha:.4f}, {g_from_mom.beta:.4f})")

    # --- Chapter 5: Rescaling ---
    print("\n--- Chapter 5: Rescaling ---")

    observed = np.array([10.0, 20.0, 15.0, 5.0])
    expected = np.array([12.0, 18.0, 15.0, 6.0])
    scales = compute_scaling_factors(observed, expected)
    print(f"Observed: {observed}")
    print(f"Expected: {expected}")
    print(f"Scaling factors: {scales}")

    # Demonstrate apply_rescaling
    node_times_demo = np.array([0.0, 0.0, 1.0, 2.5, 4.0])
    breakpoints = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    scales_demo = np.array([1.0, 2.0, 0.5, 1.0])
    fixed = {0, 1}
    new_times = apply_rescaling(node_times_demo, breakpoints, scales_demo, fixed)
    print(f"\nOriginal times: {node_times_demo}")
    print(f"Rescaled times: {new_times}")
    print(f"(Fixed nodes at indices 0,1 unchanged)")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    demo()
