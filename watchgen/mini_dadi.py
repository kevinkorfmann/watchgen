"""
Mini-dadi: Diffusion Approximation for Demographic Inference.

This module implements the core algorithms from dadi (Gutenkunst et al. 2009),
which infers demographic history by solving the Wright-Fisher diffusion equation
-- a partial differential equation (PDE) governing the continuous allele
frequency density phi(x, t).

The approach:
1. Start from the equilibrium frequency density phi(x) ~ 1/x (neutral) or
   phi(x) ~ exp(gamma*x) / [x(1-x)] (with selection).
2. Solve the 1D diffusion PDE on a nonuniform frequency grid using finite
   differences (Crank-Nicolson / forward Euler time-stepping).
3. Extract the discrete site frequency spectrum (SFS) from the continuous
   density via binomial projection (trapezoidal integration).
4. Compare the model SFS to observed data using Poisson or multinomial
   composite likelihood, and optimize demographic parameters via BFGS.

Key concepts:
- The Wright-Fisher diffusion PDE:
    dphi/dt = (1/2) d^2/dx^2 [x(1-x)/nu * phi]
              - d/dx [gamma * x(1-x) * (h + (1-2h)x) * phi]
  where nu is the relative population size, gamma is the scaled selection
  coefficient, and h is the dominance coefficient.

- Nonuniform grid: denser spacing near x=0 and x=1 where phi(x) is steep.

- Richardson extrapolation: running at multiple grid sizes and extrapolating
  to the n -> infinity limit to cancel finite-difference bias.

- Poisson composite likelihood: LL = sum_k [D_k * log(M_k) - M_k],
  where D_k is observed and M_k is expected SFS.

Reference:
    Gutenkunst RN, Hernandez RD, Williamson SH, Bustamante CD (2009).
    Inferring the joint demographic history of multiple populations from
    multidimensional SNP frequency data. PLoS Genetics, 5(10): e1000695.
"""

import numpy as np
from scipy.special import comb


# ---------------------------------------------------------------------------
# Gear 1: The Frequency Spectrum -- equilibrium densities
# ---------------------------------------------------------------------------

def equilibrium_sfs_density(xx):
    """Equilibrium frequency density under the standard neutral model.

    Under the coalescent with constant population size, the equilibrium
    density of derived alleles at frequency x is proportional to 1/x.
    Including boundary effects for the diffusion, phi(x) ~ theta / x.

    Parameters
    ----------
    xx : ndarray
        Frequency grid points.

    Returns
    -------
    phi : ndarray
        Equilibrium frequency density on the grid.
    """
    phi = np.zeros_like(xx)
    interior = (xx > 0) & (xx < 1)
    phi[interior] = 1.0 / xx[interior]
    return phi


# ---------------------------------------------------------------------------
# Gear 2: The Diffusion Equation -- grid construction and population splits
# ---------------------------------------------------------------------------

def make_nonuniform_grid(pts):
    """Build a non-uniform grid with denser spacing near boundaries.

    dadi uses a grid that concentrates points near x=0 and x=1 where
    the frequency density has steep gradients. This implementation
    uses a cosine transformation that achieves the same effect.

    Parameters
    ----------
    pts : int
        Number of grid points.

    Returns
    -------
    xx : ndarray
        Frequency grid points in [0, 1] with denser spacing at boundaries.
    """
    # Uniform grid in a transformed space
    zz = np.linspace(0, 1, pts)
    # Apply a transformation that concentrates points at boundaries
    xx = 0.5 * (1 - np.cos(np.pi * zz))
    xx[0] = 0.0
    xx[-1] = 1.0
    return xx


def phi_1d_to_2d(phi_1d, xx):
    """Split a 1D frequency density into a 2D joint density.

    After a population split, the two daughter populations share the same
    ancestral frequency spectrum. The 2D density has mass on the diagonal
    phi_2d[i,j] is nonzero primarily when i ~ j.

    Parameters
    ----------
    phi_1d : ndarray
        1D frequency density.
    xx : ndarray
        Frequency grid points.

    Returns
    -------
    phi_2d : ndarray, shape (n, n)
        2D joint density concentrated on the diagonal.
    """
    n = len(xx)
    phi_2d = np.zeros((n, n))
    # At the moment of split, both populations have identical frequencies
    # so the 2D density is concentrated on the diagonal
    for i in range(n):
        phi_2d[i, i] = phi_1d[i]
    return phi_2d


# ---------------------------------------------------------------------------
# Gear 3: Numerical Integration -- PDE solver and SFS extraction
# ---------------------------------------------------------------------------

def _thomas_solve(lower, diag, upper, rhs):
    """Solve a tridiagonal system Ax = rhs using the Thomas algorithm.

    Parameters
    ----------
    lower : ndarray
        Sub-diagonal coefficients (index 0 unused).
    diag : ndarray
        Main diagonal coefficients.
    upper : ndarray
        Super-diagonal coefficients (last index unused).
    rhs : ndarray
        Right-hand side vector.

    Returns
    -------
    x : ndarray
        Solution vector.
    """
    n = len(rhs)
    c = np.zeros(n)
    d = np.zeros(n)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / m if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / m

    x = np.zeros(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


def crank_nicolson_1d(phi, xx, T, nu=1.0, theta=0.0, n_steps=100):
    """Integrate the 1D diffusion equation using Crank-Nicolson.

    The diffusion equation for the frequency density is:
        dphi/dt = (1/(2*nu)) * d^2/dx^2 [x(1-x) phi]

    with mutation injection at x -> 0.

    Crank-Nicolson averages the spatial operator between the current and
    next time step, yielding an implicit scheme that is unconditionally
    stable and second-order accurate in both time and space:

        (I - 0.5*dt*L) phi^{n+1} = (I + 0.5*dt*L) phi^n + dt*source

    The resulting tridiagonal system is solved at each step using the
    Thomas algorithm.

    Parameters
    ----------
    phi : ndarray
        Initial frequency density on the grid.
    xx : ndarray
        Frequency grid points.
    T : float
        Integration time (in 2*N_ref generations).
    nu : float
        Relative population size (N/N_ref).
    theta : float
        Scaled mutation rate.
    n_steps : int
        Number of time steps.

    Returns
    -------
    phi_new : ndarray
        Evolved frequency density.
    """
    phi = phi.copy()
    dt = T / n_steps
    n = len(xx)

    # Precompute tridiagonal coefficients of spatial operator L
    # L[phi]_i = a_i * phi[i-1] + b_i * phi[i] + c_i * phi[i+1]
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    for i in range(1, n - 1):
        dx_l = xx[i] - xx[i - 1]
        dx_r = xx[i + 1] - xx[i]
        dx_avg = 0.5 * (dx_l + dx_r)

        # Diffusion coefficients at half-grid points
        x_r = 0.5 * (xx[i] + xx[i + 1])
        x_l = 0.5 * (xx[i] + xx[i - 1])
        V_r = x_r * (1 - x_r) / (2.0 * nu)
        V_l = x_l * (1 - x_l) / (2.0 * nu)

        a[i] = V_l / (dx_l * dx_avg)
        c[i] = V_r / (dx_r * dx_avg)
        b[i] = -(a[i] + c[i])

    # Precompute LHS tridiagonal bands: (I - 0.5*dt*L)
    lhs_lower = np.zeros(n)
    lhs_diag = np.ones(n)
    lhs_upper = np.zeros(n)
    for i in range(1, n - 1):
        lhs_lower[i] = -0.5 * dt * a[i]
        lhs_diag[i] = 1.0 - 0.5 * dt * b[i]
        lhs_upper[i] = -0.5 * dt * c[i]

    for _ in range(n_steps):
        # RHS: (I + 0.5*dt*L) phi
        rhs = np.zeros(n)
        for i in range(1, n - 1):
            rhs[i] = (0.5 * dt * a[i] * phi[i - 1]
                      + (1.0 + 0.5 * dt * b[i]) * phi[i]
                      + 0.5 * dt * c[i] * phi[i + 1])

        # Mutation injection at the singleton bin
        if theta > 0 and n > 2:
            rhs[1] += theta / (2.0 * xx[1]) * dt

        # Solve tridiagonal system
        phi = _thomas_solve(lhs_lower, lhs_diag, lhs_upper, rhs)

        # Boundary conditions: phi = 0 at x=0 and x=1
        phi[0] = 0.0
        phi[-1] = 0.0

    return phi


def sfs_from_phi(phi, xx, n_samples):
    """Extract a discrete SFS from the continuous frequency density.

    Uses the binomial projection: the expected number of sites where
    k out of n chromosomes carry the derived allele is obtained by
    integrating phi(x) * C(n,k) * x^k * (1-x)^(n-k) over x.

    Parameters
    ----------
    phi : ndarray
        Frequency density on the grid.
    xx : ndarray
        Frequency grid points.
    n_samples : int
        Sample size (number of chromosomes).

    Returns
    -------
    sfs : ndarray, shape (n_samples + 1,)
        Discrete SFS where sfs[k] is the expected count at frequency k/n.
    """
    sfs = np.zeros(n_samples + 1)
    # Trapezoidal integration
    for k in range(n_samples + 1):
        binom_weight = comb(n_samples, k) * xx**k * (1 - xx)**(n_samples - k)
        integrand = phi * binom_weight
        sfs[k] = np.trapezoid(integrand, xx)
    return sfs


# ---------------------------------------------------------------------------
# Gear 4: Demographic Inference -- likelihoods, scaling, and model functions
# ---------------------------------------------------------------------------

def poisson_log_likelihood(model, data):
    """Poisson composite log-likelihood.

    LL = sum_k [ D_k * log(M_k) - M_k ]

    where D_k is observed and M_k is expected.

    Parameters
    ----------
    model : ndarray
        Expected SFS counts.
    data : ndarray
        Observed SFS counts.

    Returns
    -------
    ll : float
        Poisson composite log-likelihood.
    """
    mask = data > 0
    model_safe = np.maximum(model, 1e-300)
    ll = np.sum(data[mask] * np.log(model_safe[mask]) - model_safe[mask])
    # Subtract contribution from zero-observed entries
    ll -= np.sum(model_safe[~mask])
    return ll


def multinomial_log_likelihood(model, data):
    """Multinomial composite log-likelihood.

    Automatically normalizes the model to probabilities.
    LL = sum_k [ D_k * log(M_k / sum(M)) ]

    Parameters
    ----------
    model : ndarray
        Expected SFS counts (will be normalized to probabilities).
    data : ndarray
        Observed SFS counts.

    Returns
    -------
    ll : float
        Multinomial composite log-likelihood.
    """
    model_probs = model / model.sum()
    model_probs = np.maximum(model_probs, 1e-300)
    mask = data > 0
    return np.sum(data[mask] * np.log(model_probs[mask]))


def optimal_sfs_scaling(model, data):
    """Compute the optimal theta that scales model to best fit data.

    Under the Poisson model, the optimal scaling is:
        theta_opt = sum(D_k) / sum(M_k)

    Parameters
    ----------
    model : ndarray
        Expected SFS counts (unscaled).
    data : ndarray
        Observed SFS counts.

    Returns
    -------
    theta_opt : float
        Optimal scaling factor.
    """
    return data.sum() / max(model.sum(), 1e-300)


def two_epoch_sfs(nu, T, n_samples, pts=60, theta=1.0):
    """Compute the expected SFS under a two-epoch demographic model.

    A population at equilibrium changes size to nu * N_ref at time T ago.

    Parameters
    ----------
    nu : float
        Ratio of new to ancestral population size.
    T : float
        Time of size change (in 2*N_ref generations).
    n_samples : int
        Sample size.
    pts : int
        Number of grid points.
    theta : float
        Scaled mutation rate.

    Returns
    -------
    sfs : ndarray
        Expected SFS.
    """
    xx = make_nonuniform_grid(pts)
    phi = equilibrium_sfs_density(xx) * theta
    phi = crank_nicolson_1d(phi, xx, T, nu=nu, theta=theta)
    sfs = sfs_from_phi(phi, xx, n_samples)
    return sfs


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo():
    """Demonstrate the dadi diffusion approximation pipeline.

    Covers:
    - Building a nonuniform frequency grid
    - Computing the equilibrium frequency density
    - Verifying the 1/x shape of the neutral SFS density
    - Population splits (1D -> 2D density)
    - Solving the diffusion PDE (Crank-Nicolson)
    - Extracting the discrete SFS via binomial projection
    - Poisson and multinomial likelihoods
    - Optimal SFS scaling
    - Two-epoch demographic model
    """
    print("=" * 65)
    print("Mini-dadi: Diffusion Approximation for Demographic Inference")
    print("=" * 65)

    # --- Gear 2: The Diffusion Equation ---
    print("\n--- Gear 2: Nonuniform Grid & Equilibrium Density ---")

    # Build a frequency grid (cf. dadi.Numerics.default_grid)
    pts = 60
    xx = make_nonuniform_grid(pts)

    # Grid spacing near boundaries vs. interior
    print(f"Grid points: {pts}")
    print(f"First spacing:  {xx[1] - xx[0]:.6f}")
    print(f"Middle spacing: {xx[30] - xx[29]:.6f}")
    print(f"Last spacing:   {xx[-1] - xx[-2]:.6f}")

    # Equilibrium density under the standard neutral model
    # (cf. dadi.PhiManip.phi_1D)
    phi = equilibrium_sfs_density(xx)

    # The equilibrium density is proportional to 1/x
    print(f"\nphi at x={xx[1]:.4f}: {phi[1]:.2f}")
    print(f"phi at x={xx[5]:.4f}: {phi[5]:.2f}")
    print(f"Ratio: {phi[1]/phi[5]:.2f}, expected: {xx[5]/xx[1]:.2f}")

    # --- Population split ---
    print("\n--- Population Split (1D -> 2D) ---")
    phi_2d = phi_1d_to_2d(phi, xx)
    print(f"2D density shape: {phi_2d.shape}")
    print(f"Diagonal sum:     {np.diag(phi_2d).sum():.4f}")
    print(f"Off-diagonal sum: {(phi_2d.sum() - np.diag(phi_2d).sum()):.4f}")

    # --- Gear 3: Numerical Integration ---
    print("\n--- Gear 3: Solving the Diffusion PDE ---")

    # Integrate for T=0.5 with doubled population size
    phi_evolved = crank_nicolson_1d(phi, xx, T=0.5, nu=2.0, n_steps=200)
    print(f"Original total density:  {np.trapezoid(phi, xx):.4f}")
    print(f"Evolved total density:   {np.trapezoid(phi_evolved, xx):.4f}")

    # Extract SFS for sample size n=20
    sfs = sfs_from_phi(phi, xx, 20)
    print(f"\nEquilibrium SFS (first 5 entries):")
    for j in range(1, 6):
        print(f"  sfs[{j}] = {sfs[j]:.4f}  (expected ~ 1/{j} = {1.0/j:.4f})")

    # Under neutrality, fs[j] ~ theta/j
    print(f"\nRatio sfs[1]/sfs[2] = {sfs[1]/sfs[2]:.3f} (expected ~2.0)")
    print(f"Ratio sfs[1]/sfs[5] = {sfs[1]/sfs[5]:.3f} (expected ~5.0)")

    # --- Gear 4: Demographic Inference ---
    print("\n--- Gear 4: Likelihoods and Inference ---")

    # Poisson log-likelihood
    model = np.array([5.0, 10.0, 15.0, 20.0])
    data = np.array([5, 10, 15, 20])
    ll = poisson_log_likelihood(model, data)
    print(f"Poisson LL (model=data):     {ll:.4f}")

    ll_bad = poisson_log_likelihood(model * 2, data)
    print(f"Poisson LL (model=2*data):   {ll_bad:.4f}")

    # Multinomial log-likelihood
    ll_multi = multinomial_log_likelihood(model, data)
    print(f"Multinomial LL (true props): {ll_multi:.4f}")

    # Optimal scaling
    model_unscaled = np.array([1.0, 2.0, 3.0])
    data_scaled = np.array([10.0, 20.0, 30.0])
    theta_opt = optimal_sfs_scaling(model_unscaled, data_scaled)
    print(f"Optimal theta scaling: {theta_opt:.2f} (expected 10.0)")

    # --- Two-epoch model ---
    print("\n--- Two-Epoch Demographic Model ---")
    sfs_expand = two_epoch_sfs(nu=5.0, T=0.3, n_samples=15, pts=60)
    sfs_contract = two_epoch_sfs(nu=0.2, T=0.3, n_samples=15, pts=60)
    print(f"Expansion SFS[1:5]:   {sfs_expand[1:5]}")
    print(f"Contraction SFS[1:5]: {sfs_contract[1:5]}")
    print(f"Expansion and contraction produce different spectra: "
          f"{not np.allclose(sfs_expand, sfs_contract, atol=0.01)}")

    print("\n" + "=" * 65)
    print("Demo complete.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
