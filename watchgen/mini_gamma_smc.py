"""
Mini Gamma-SMC: Ultrafast pairwise TMRCA inference via continuous-state HMM.

Gamma-SMC (Schweiger & Durbin, 2023) infers the posterior distribution of
the pairwise time to most recent common ancestor (TMRCA) at every position
along the genome.  Unlike PSMC, which discretizes coalescence time into a
finite set of intervals and runs a standard discrete-state HMM, Gamma-SMC
keeps time continuous and tracks the posterior TMRCA as a gamma distribution
at every position.

The key insight is Poisson-gamma conjugacy: if the prior on the TMRCA is a
gamma distribution and the emission model is Poisson, the posterior after
observing a heterozygous or homozygous site is again a gamma distribution.
The transition step (accounting for recombination) breaks this conjugacy,
but the post-transition distribution is very close to gamma.  By
approximating it via a precomputed flow field, the entire forward pass
reduces to a sequence of table lookups and parameter updates.

Components
----------
1. **Gamma Approximation** -- Poisson-gamma conjugate emission update and
   log-coordinate conversions.
2. **Flow Field** -- Precomputed 2D vector field mapping gamma parameters
   through one SMC transition step via bilinear interpolation.
3. **Forward-Backward CS-HMM** -- Forward pass sweeps left-to-right;
   backward pass is a forward pass on the reversed sequence; combination
   yields Gamma(a_fwd + a_bwd - 1, b_fwd + b_bwd - 1).
4. **Segmentation and Caching** -- Consecutive hom/missing positions are
   grouped into segments; entropy clipping prevents approximation drift.

Reference
---------
Schweiger, R. & Durbin, R.  "Ultrafast genome-wide inference of pairwise
coalescence times."  *Genome Research* 33, 1023-1031 (2023).
"""

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, gammaln


# ---------------------------------------------------------------------------
# Gamma Approximation (gamma_approximation.rst)
# ---------------------------------------------------------------------------

def gamma_emission_update(alpha, beta, y, theta):
    """Apply the Poisson-gamma conjugate emission update.

    Parameters
    ----------
    alpha : float
        Current shape parameter.
    beta : float
        Current rate parameter.
    y : int
        Observation: 1 (het), 0 (hom), or -1 (missing).
    theta : float
        Scaled mutation rate.

    Returns
    -------
    alpha_new, beta_new : float
        Updated gamma parameters.
    """
    if y == -1:  # missing
        return alpha, beta
    return alpha + y, beta + theta


def to_log_coords(alpha, beta):
    """Convert (alpha, beta) to log-mean / log-CV coordinates.

    Parameters
    ----------
    alpha : float
        Shape parameter (must be > 0).
    beta : float
        Rate parameter (must be > 0).

    Returns
    -------
    l_mu : float
        log10(alpha/beta), the log-mean TMRCA.
    l_C : float
        log10(1/sqrt(alpha)), the log coefficient of variation.
    """
    l_mu = np.log10(alpha / beta)
    l_C = np.log10(1.0 / np.sqrt(alpha))
    return l_mu, l_C


def from_log_coords(l_mu, l_C):
    """Convert log-coordinates back to (alpha, beta).

    Parameters
    ----------
    l_mu : float
        Log-mean coordinate.
    l_C : float
        Log-CV coordinate.

    Returns
    -------
    alpha, beta : float
    """
    alpha = 10.0 ** (-2 * l_C)
    beta = alpha * 10.0 ** (-l_mu)
    return alpha, beta


# ---------------------------------------------------------------------------
# Flow Field (flow_field.rst)
# ---------------------------------------------------------------------------

def gamma_pdf_partials(x, alpha, beta):
    """Evaluate the gamma PDF and its partial derivatives.

    Parameters
    ----------
    x : ndarray
        Points at which to evaluate.
    alpha : float
        Shape parameter.
    beta : float
        Rate parameter.

    Returns
    -------
    f : ndarray
        Gamma PDF values.
    df_dalpha : ndarray
        Partial derivative with respect to alpha.
    df_dbeta : ndarray
        Partial derivative with respect to beta.
    """
    f = gamma_dist.pdf(x, a=alpha, scale=1.0 / beta)
    # d/dalpha [Gamma PDF] = f * (-psi(alpha) + log(beta) + log(x))
    df_dalpha = f * (-digamma(alpha) + np.log(beta) + np.log(x + 1e-300))
    # d/dbeta [Gamma PDF] = f * (alpha/beta - x)
    df_dbeta = f * (alpha / beta - x)
    return f, df_dalpha, df_dbeta


def compute_flow_at_point(l_mu, l_C, n_eval=2000):
    """Compute the flow displacement at one grid point.

    This is a simplified skeleton showing the least-squares structure.
    The full implementation requires Kummer's hypergeometric function
    M(a, b, z) via the Arb library for the perturbation evaluation.

    Parameters
    ----------
    l_mu, l_C : float
        Log-mean and log-CV coordinates.
    n_eval : int
        Number of evaluation points for least-squares.

    Returns
    -------
    delta_l_mu, delta_l_C : float
        Flow displacement in log-coordinates.
    """
    # Step 1: convert to (alpha, beta)
    alpha = 10.0 ** (-2 * l_C)
    beta = alpha * 10.0 ** (-l_mu)

    # Step 2: set up evaluation grid over the support of Gamma(alpha, beta)
    mean = alpha / beta
    std = np.sqrt(alpha) / beta
    x = np.linspace(max(1e-10, mean - 4 * std), mean + 6 * std, n_eval)

    # Step 3: evaluate partial derivatives
    f, df_da, df_db = gamma_pdf_partials(x, alpha, beta)

    # Step 4: in the full implementation, evaluate the perturbation
    # (p_{alpha,beta}(x) - f_{alpha,beta}(x)) / rho using Kummer's M.
    # Here we use a placeholder zero perturbation for illustration.
    perturbation = np.zeros_like(x)  # placeholder

    # Step 5: solve least-squares for (u, v) in log10(alpha), log10(beta)
    A = np.column_stack([df_da, df_db])
    result = np.linalg.lstsq(A, perturbation, rcond=None)
    delta_log_a, delta_log_b = result[0]

    # Step 6: convert to (delta_l_mu, delta_l_C)
    delta_l_mu = delta_log_a - delta_log_b
    delta_l_C = -0.5 * delta_log_a
    return delta_l_mu, delta_l_C


class FlowField:
    """A precomputed flow field over a (l_mu, l_C) grid.

    Parameters
    ----------
    l_mu_grid : ndarray, shape (n_mu,)
        Log-mean grid values.
    l_C_grid : ndarray, shape (n_C,)
        Log-CV grid values.
    delta_l_mu : ndarray, shape (n_mu, n_C)
        Flow displacement in l_mu at each grid point.
    delta_l_C : ndarray, shape (n_mu, n_C)
        Flow displacement in l_C at each grid point.
    """

    def __init__(self, l_mu_grid, l_C_grid, delta_l_mu, delta_l_C):
        self.l_mu_grid = l_mu_grid
        self.l_C_grid = l_C_grid
        self.delta_l_mu = delta_l_mu
        self.delta_l_C = delta_l_C

    def query(self, l_mu, l_C):
        """Query the flow field via bilinear interpolation with clipping.

        Parameters
        ----------
        l_mu : float
            Log-mean coordinate of the current gamma distribution.
        l_C : float
            Log-CV coordinate of the current gamma distribution.

        Returns
        -------
        dl_mu, dl_C : float
            Interpolated flow displacement.
        """
        # Clip to grid boundaries
        l_mu_c = np.clip(l_mu, self.l_mu_grid[0], self.l_mu_grid[-1])
        l_C_c = np.clip(l_C, self.l_C_grid[0], self.l_C_grid[-1])

        # Find the lower-left grid cell indices
        i = np.searchsorted(self.l_mu_grid, l_mu_c) - 1
        j = np.searchsorted(self.l_C_grid, l_C_c) - 1
        i = np.clip(i, 0, len(self.l_mu_grid) - 2)
        j = np.clip(j, 0, len(self.l_C_grid) - 2)

        # Fractional positions within the cell: s along l_mu, t along l_C
        s = ((l_mu_c - self.l_mu_grid[i])
             / (self.l_mu_grid[i + 1] - self.l_mu_grid[i]))
        t = ((l_C_c - self.l_C_grid[j])
             / (self.l_C_grid[j + 1] - self.l_C_grid[j]))

        # Bilinear interpolation for each displacement component
        def interp(field):
            return ((1 - s) * (1 - t) * field[i, j]
                    + s * (1 - t) * field[i + 1, j]
                    + (1 - s) * t * field[i, j + 1]
                    + s * t * field[i + 1, j + 1])

        return interp(self.delta_l_mu), interp(self.delta_l_C)


# ---------------------------------------------------------------------------
# Forward-Backward CS-HMM (forward_backward.rst)
# ---------------------------------------------------------------------------

def gamma_smc_forward(observations, theta, rho, flow_field):
    """Run the Gamma-SMC forward pass.

    Parameters
    ----------
    observations : list of int
        Observation at each position: 1 (het), 0 (hom), -1 (missing).
    theta : float
        Scaled mutation rate per position.
    rho : float
        Scaled recombination rate per position.
    flow_field : FlowField
        Precomputed flow field with a .query(l_mu, l_C) method.

    Returns
    -------
    alphas : ndarray, shape (N,)
        Forward shape parameter at each position.
    betas : ndarray, shape (N,)
        Forward rate parameter at each position.
    """
    N = len(observations)
    alphas = np.zeros(N)
    betas = np.zeros(N)

    # Initialize with the prior: Gamma(1, 1) = Exp(1)
    alpha, beta = 1.0, 1.0

    for i in range(N):
        # Step 1: Transition via flow field
        l_mu = np.log10(alpha / beta)
        l_C = np.log10(1.0 / np.sqrt(alpha))
        dl_mu, dl_C = flow_field.query(l_mu, l_C)
        l_mu += rho * dl_mu  # displacement scaled by rho
        l_C += rho * dl_C
        alpha = 10.0 ** (-2 * l_C)
        beta = alpha * 10.0 ** (-l_mu)

        # Step 2: Emission update (conjugate)
        y = observations[i]
        if y >= 0:  # not missing
            alpha += y       # +1 for het, +0 for hom
            beta += theta

        alphas[i] = alpha
        betas[i] = beta

    return alphas, betas


def gamma_smc_posterior(observations, theta, rho, flow_field):
    """Compute the full Gamma-SMC posterior at each position.

    Runs forward and backward passes and combines them.

    Parameters
    ----------
    observations : list of int
        Observation at each position: 1 (het), 0 (hom), -1 (missing).
    theta, rho : float
        Scaled mutation and recombination rates.
    flow_field : FlowField
        Precomputed flow field.

    Returns
    -------
    post_alpha : ndarray
        Posterior shape at each position.
    post_beta : ndarray
        Posterior rate at each position.
    """
    # Forward pass (left to right)
    a_fwd, b_fwd = gamma_smc_forward(observations, theta, rho, flow_field)

    # Backward pass = forward pass on reversed sequence
    a_bwd_rev, b_bwd_rev = gamma_smc_forward(
        observations[::-1], theta, rho, flow_field
    )
    # Reverse the backward results to align with original positions
    a_bwd = a_bwd_rev[::-1]
    b_bwd = b_bwd_rev[::-1]

    # Combine: Gamma(a + a' - 1, b + b' - 1)
    post_alpha = a_fwd + a_bwd - 1
    post_beta = b_fwd + b_bwd - 1

    return post_alpha, post_beta


# ---------------------------------------------------------------------------
# Segmentation and Caching (segmentation_and_caching.rst)
# ---------------------------------------------------------------------------

def segment_observations(observations):
    """Segment a sequence into (n_miss, n_hom, final_obs) tuples.

    Consecutive missing and homozygous positions are grouped together.
    Each segment ends at a heterozygous site or the end of the sequence.

    Parameters
    ----------
    observations : list of int
        Observation at each position: 1 (het), 0 (hom), -1 (missing).

    Returns
    -------
    segments : list of (int, int, int)
        Each tuple is (n_miss, n_hom, final_obs).
    """
    segments = []
    n_miss = 0
    n_hom = 0

    for y in observations:
        if y == -1:  # missing
            n_miss += 1
        elif y == 0:  # hom
            n_hom += 1
        else:  # het (y == 1): close the current segment
            segments.append((n_miss, n_hom, 1))
            n_miss = 0
            n_hom = 0

    # Final segment (may end with hom or missing)
    if n_miss > 0 or n_hom > 0:
        segments.append((n_miss, n_hom, 0))

    return segments


def gamma_entropy(alpha, beta):
    """Differential entropy of Gamma(alpha, beta).

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha >= 1).
    beta : float
        Rate parameter.

    Returns
    -------
    h : float
        Differential entropy.
    """
    return (alpha - np.log(beta) + gammaln(alpha)
            + (1 - alpha) * digamma(alpha))


def entropy_clip(alpha, beta, h_max=1.0, tol=1e-8):
    """Clip gamma parameters so that entropy does not exceed h_max.

    Keeps the mean alpha/beta fixed and reduces the CV (increases alpha)
    until the entropy equals h_max.

    Parameters
    ----------
    alpha, beta : float
        Current gamma parameters.
    h_max : float
        Maximum allowed entropy (default 1.0, the prior entropy).
    tol : float
        Bisection tolerance.

    Returns
    -------
    alpha_clipped, beta_clipped : float
        Clipped parameters with h <= h_max.
    """
    if gamma_entropy(alpha, beta) <= h_max:
        return alpha, beta  # no clipping needed

    mean = alpha / beta  # fix the mean

    # Bisect on alpha: increasing alpha decreases entropy (for fixed mean)
    lo, hi = alpha, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2
        b_mid = mid / mean
        if gamma_entropy(mid, b_mid) > h_max:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    alpha_new = (lo + hi) / 2
    beta_new = alpha_new / mean
    return alpha_new, beta_new


def gamma_smc_forward_segmented(observations, theta, rho, flow_field,
                                h_max=1.0):
    """Gamma-SMC forward pass with segmentation and entropy clipping.

    This is the full algorithm combining all four components:
    segmentation, caching (via repeated flow field application),
    flow field queries, and entropy clipping.

    Parameters
    ----------
    observations : list of int
        1 (het), 0 (hom), -1 (missing) at each position.
    theta : float
        Scaled mutation rate.
    rho : float
        Scaled recombination rate.
    flow_field : object
        Flow field with .query(l_mu, l_C) method.
    h_max : float
        Entropy threshold for clipping.

    Returns
    -------
    results : list of (float, float)
        (alpha, beta) at each het position.
    """
    segments = segment_observations(observations)
    alpha, beta = 1.0, 1.0  # prior: Gamma(1, 1)
    results = []

    for n_miss, n_hom, final_obs in segments:
        # Step 1: skip missing positions (flow field only, no emission)
        for _ in range(n_miss):
            l_mu = np.log10(alpha / beta)
            l_C = np.log10(1.0 / np.sqrt(alpha))
            dl_mu, dl_C = flow_field.query(l_mu, l_C)
            l_mu += rho * dl_mu
            l_C += rho * dl_C
            alpha = 10.0 ** (-2 * l_C)
            beta = alpha * 10.0 ** (-l_mu)

        # Step 2: entropy clip after missing block
        alpha, beta = entropy_clip(alpha, beta, h_max)

        # Step 3: skip homozygous positions (flow field + hom emission)
        for _ in range(n_hom):
            l_mu = np.log10(alpha / beta)
            l_C = np.log10(1.0 / np.sqrt(alpha))
            dl_mu, dl_C = flow_field.query(l_mu, l_C)
            l_mu += rho * dl_mu
            l_C += rho * dl_C
            alpha = 10.0 ** (-2 * l_C)
            beta = alpha * 10.0 ** (-l_mu)
            beta += theta  # hom emission

        # Step 4: entropy clip after hom block
        alpha, beta = entropy_clip(alpha, beta, h_max)

        # Step 5: emission update for the final observation
        if final_obs == 1:  # het
            alpha += 1
            beta += theta
        elif final_obs == 0:  # trailing hom (end of sequence)
            beta += theta

        results.append((alpha, beta))

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate all Gamma-SMC components."""

    print("=" * 60)
    print("Gamma-SMC: Continuous-State HMM for TMRCA Inference")
    print("=" * 60)

    # --- Emission update demo ---
    print("\n--- Emission Update (Poisson-Gamma Conjugacy) ---")
    alpha, beta = 1.0, 1.0
    theta = 0.001

    a_het, b_het = gamma_emission_update(alpha, beta, 1, theta)
    print(f"After het: Gamma({a_het}, {b_het:.4f})")
    print(f"  Mean TMRCA = {a_het/b_het:.4f} "
          f"(shifted up -- mutation is evidence of deeper coalescence)")

    a_hom, b_hom = gamma_emission_update(alpha, beta, 0, theta)
    print(f"After hom: Gamma({a_hom}, {b_hom:.4f})")
    print(f"  Mean TMRCA = {a_hom/b_hom:.4f} "
          f"(shifted down -- no mutation favours recent coalescence)")

    a, b = 1.0, 1.0
    for _ in range(100):
        a, b = gamma_emission_update(a, b, 0, theta)
    a, b = gamma_emission_update(a, b, 1, theta)
    print(f"\nAfter 100 hom + 1 het: Gamma({a:.1f}, {b:.4f}), "
          f"mean = {a/b:.4f}")

    # --- Log-coordinate round-trip ---
    print("\n--- Log-Coordinate Conversion ---")
    for alpha, beta in [(1.0, 1.0), (5.0, 2.5), (100.0, 50.0)]:
        l_mu, l_C = to_log_coords(alpha, beta)
        a_back, b_back = from_log_coords(l_mu, l_C)
        print(f"Gamma({alpha}, {beta}) -> (l_mu={l_mu:.4f}, l_C={l_C:.4f}) "
              f"-> Gamma({a_back:.4f}, {b_back:.4f})")

    l_mu_prior, l_C_prior = to_log_coords(1.0, 1.0)
    print(f"\nPrior Gamma(1,1): (l_mu, l_C) = "
          f"({l_mu_prior:.1f}, {l_C_prior:.1f})")

    l_mu_grid = np.linspace(-5, 2, 51)
    l_C_grid = np.linspace(-2, 0, 50)
    print(f"Grid: {len(l_mu_grid)} x {len(l_C_grid)} = "
          f"{len(l_mu_grid) * len(l_C_grid)} points")

    # --- Flow field demo ---
    print("\n--- Flow Field ---")
    dl_mu, dl_C = compute_flow_at_point(0.0, -0.5)
    print(f"Flow at (l_mu=0, l_C=-0.5): "
          f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")
    print("(Zero because we used a placeholder perturbation)")

    # Build a small demonstration flow field
    dl_mu_grid = np.zeros((51, 50))
    dl_C_grid = 0.01 * np.ones((51, 50))
    ff = FlowField(l_mu_grid, l_C_grid, dl_mu_grid, dl_C_grid)
    dl_mu, dl_C = ff.query(0.5, -0.8)
    print(f"Flow at (l_mu=0.5, l_C=-0.8): "
          f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")
    dl_mu, dl_C = ff.query(5.0, -3.0)
    print(f"Flow at (l_mu=5.0, l_C=-3.0) [clipped]: "
          f"delta_l_mu={dl_mu:.6f}, delta_l_C={dl_C:.6f}")

    # --- Forward pass demo ---
    print("\n--- Forward Pass ---")
    obs = [0] * 50 + [1] + [0] * 30 + [1] + [0] * 19
    theta_demo, rho_demo = 0.001, 0.0004

    class ZeroFlow:
        def query(self, l_mu, l_C):
            return 0.0, 0.0

    a_fwd, b_fwd = gamma_smc_forward(obs, theta_demo, rho_demo, ZeroFlow())
    print(f"After {len(obs)} positions ({sum(obs)} hets):")
    print(f"  Final forward: Gamma({a_fwd[-1]:.1f}, {b_fwd[-1]:.4f})")
    print(f"  Mean TMRCA = {a_fwd[-1]/b_fwd[-1]:.2f}")

    # --- Posterior demo ---
    print("\n--- Forward-Backward Posterior ---")
    a_post, b_post = gamma_smc_posterior(obs, theta_demo, rho_demo, ZeroFlow())
    mean_tmrca = a_post / b_post

    a_empty, b_empty = gamma_smc_posterior(
        [-1] * 10, theta_demo, rho_demo, ZeroFlow()
    )
    print(f"No observations: Gamma({a_empty[0]:.1f}, {b_empty[0]:.1f}) "
          f"(should be ~Gamma(1, 1))")
    print(f"\nWith data ({len(obs)} positions):")
    print(f"  Position 0:  mean TMRCA = {mean_tmrca[0]:.2f}")
    print(f"  Position 51 (het): mean TMRCA = {mean_tmrca[51]:.2f}")
    print(f"  Position 99: mean TMRCA = {mean_tmrca[-1]:.2f}")

    # --- Segmentation demo ---
    print("\n--- Segmentation ---")
    obs_seg = [-1] * 3 + [0] * 50 + [1] + [0] * 200 + [1] + [0] * 100
    segs = segment_observations(obs_seg)
    print(f"Sequence: {len(obs_seg)} positions -> {len(segs)} segments")
    for i, (nm, nh, fo) in enumerate(segs):
        label = "het" if fo == 1 else "hom"
        print(f"  Segment {i}: {nm} miss, {nh} hom, ends with {label}")
    print(f"Speedup: {len(obs_seg)}/{len(segs)} = "
          f"{len(obs_seg)/len(segs):.0f}x")

    # --- Entropy clipping demo ---
    print("\n--- Entropy Clipping ---")
    print(f"Entropy of Gamma(1, 1) = {gamma_entropy(1.0, 1.0):.6f}")
    print(f"Entropy of Gamma(5, 5) = {gamma_entropy(5.0, 5.0):.6f}")

    a, b = 1.01, 0.8
    h_before = gamma_entropy(a, b)
    a_clip, b_clip = entropy_clip(a, b)
    h_after = gamma_entropy(a_clip, b_clip)
    print(f"\nBefore clip: Gamma({a}, {b}), entropy = {h_before:.4f}")
    print(f"After clip:  Gamma({a_clip:.4f}, {b_clip:.4f}), "
          f"entropy = {h_after:.4f}")
    print(f"Mean preserved: {a/b:.4f} -> {a_clip/b_clip:.4f}")

    # --- Segmented forward pass demo ---
    print("\n--- Segmented Forward Pass ---")
    np.random.seed(42)
    n_sites = 10000
    obs_large = [0] * n_sites
    het_positions = sorted(np.random.choice(n_sites, size=10, replace=False))
    for pos in het_positions:
        obs_large[pos] = 1

    segments = segment_observations(obs_large)
    results = gamma_smc_forward_segmented(
        obs_large, 0.001, 0.0004, ZeroFlow()
    )
    print(f"Input: {n_sites} positions, {sum(obs_large)} hets")
    print(f"Segments: {len(segments)} (one per het + final)")
    print(f"Speedup: {n_sites}/{len(segments)} = "
          f"{n_sites/len(segments):.0f}x fewer operations")
    a_final, b_final = results[-1]
    print(f"Final posterior: Gamma({a_final:.1f}, {b_final:.4f}), "
          f"mean = {a_final/b_final:.2f}")


if __name__ == "__main__":
    demo()
