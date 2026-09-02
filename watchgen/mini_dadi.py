"""
Mini-dadi: Diffusion Approximation for Demographic Inference.

This module implements the core algorithms from dadi (Gutenkunst et al. 2009),
which infers demographic history by solving the Wright-Fisher diffusion equation
-- a partial differential equation (PDE) governing the continuous allele
frequency density phi(x, t).

The approach:
1. Start from the neutral equilibrium frequency density phi(x) ~ 1/x.
2. Solve the neutral 1D diffusion PDE on dadi's default exponential grid
   using the same flux discretisation and backward-Euler update as dadi.
3. Extract the discrete site frequency spectrum (SFS) from the continuous
   density via binomial projection (trapezoidal integration).
4. Compare the model SFS to observed data using Poisson or multinomial
   composite likelihood, and optimize demographic parameters via BFGS.

Key concepts:
- The Wright-Fisher diffusion PDE:
    dphi/dt = (1/2) d^2/dx^2 [x(1-x)/nu * phi]
              - d/dx [2*gamma * x(1-x) * (h + (1-2h)x) * phi]
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
from scipy.special import comb, gammaln

# ---------------------------------------------------------------------------
# Gear 1: The Frequency Spectrum -- equilibrium densities
# ---------------------------------------------------------------------------

def equilibrium_sfs_density(xx, nu=1.0, theta=1.0):
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
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 1 or len(xx) < 3:
        raise ValueError("xx must be a one-dimensional grid with at least 3 points")
    if xx[0] < 0 or xx[-1] > 1 or np.any(np.diff(xx) <= 0):
        raise ValueError("xx must increase strictly within [0, 1]")
    if nu <= 0 or theta < 0:
        raise ValueError("nu must be positive and theta must be non-negative")

    # dadi.PhiManip.phi_1D_snm stores finite endpoint representatives. The
    # mathematical density diverges at zero, so dadi copies the first interior
    # value there; the value at one is the finite 1/x limit.
    phi = np.empty_like(xx)
    if xx[0] == 0:
        phi[1:] = nu * theta / xx[1:]
        phi[0] = phi[1]
    else:
        phi[:] = nu * theta / xx
    return phi


# ---------------------------------------------------------------------------
# Gear 2: The Diffusion Equation -- grid construction and population splits
# ---------------------------------------------------------------------------

def make_nonuniform_grid(pts, crowding=8.0):
    """Build a non-uniform grid with denser spacing near boundaries.

    This is the formula used by ``dadi.Numerics.default_grid`` (currently
    ``exponential_grid``), including its default crowding parameter.

    Parameters
    ----------
    pts : int
        Number of grid points.

    Returns
    -------
    xx : ndarray
        Frequency grid points in [0, 1] with denser spacing at boundaries.
    """
    if not isinstance(pts, (int, np.integer)) or pts < 3:
        raise ValueError("pts must be an integer of at least 3")
    if crowding <= 0:
        raise ValueError("crowding must be positive")
    uniform = np.linspace(-1.0, 1.0, pts)
    grid = 1.0 / (1.0 + np.exp(-crowding * uniform))
    return (grid - grid[0]) / (grid[-1] - grid[0])


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
    phi_1d = np.asarray(phi_1d, dtype=float)
    xx = np.asarray(xx, dtype=float)
    if phi_1d.shape != xx.shape:
        raise ValueError("phi_1d and xx must have the same shape")
    n = len(xx)
    phi_2d = np.zeros((n, n))
    # The split is phi(x) delta(x-y). A sampled delta needs the reciprocal
    # trapezoid weight; copying phi directly onto the diagonal loses mass as
    # the grid is refined. This is dadi.PhiManip.phi_1D_to_2D's convention.
    for i in range(1, n - 1):
        phi_2d[i, i] = phi_1d[i] * 2.0 / (xx[i + 1] - xx[i - 1])
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


def implicit_1d(phi, xx, T, nu=1.0, theta=1.0, n_steps=100):
    """Integrate dadi's neutral 1D diffusion discretisation.

    The diffusion equation for the frequency density is:
        dphi/dt = (1/(2*nu)) * d^2/dx^2 [x(1-x) phi]

    with mutation injection at x -> 0.

    The update is *backward Euler*, matching ``dadi.Integration.one_pop`` for a
    neutral population with constant parameters. dadi discretises the
    derivative of probability flux on the nonuniform grid, injects mutations
    into the first interior point with trapezoid normalization, and solves a
    tridiagonal implicit system. This is not Crank-Nicolson.

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
    phi = np.asarray(phi, dtype=float).copy()
    xx = np.asarray(xx, dtype=float)
    n = len(xx)
    if phi.shape != xx.shape or n < 3 or np.any(np.diff(xx) <= 0):
        raise ValueError("phi and xx must be matching one-dimensional grids")
    if T < 0 or nu <= 0 or theta < 0:
        raise ValueError("T and theta must be non-negative and nu must be positive")
    if not isinstance(n_steps, (int, np.integer)) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")
    if T == 0:
        return phi

    dx = np.diff(xx)
    dfactor = np.empty(n)
    dfactor[1:-1] = 2.0 / (dx[:-1] + dx[1:])
    dfactor[0] = 2.0 / dx[0]
    dfactor[-1] = 2.0 / dx[-1]
    variance = xx * (1.0 - xx) / nu

    # dadi's neutral M=0 coefficients (_one_pop_const_params), with the
    # default centered flux (delj=1/2).
    lower = np.zeros(n)
    upper = np.zeros(n)
    diag = np.zeros(n)
    lower[1:] = -dfactor[1:] * variance[:-1] / (2.0 * dx)
    upper[:-1] = -dfactor[:-1] * variance[1:] / (2.0 * dx)
    diag[:-1] = dfactor[:-1] * variance[:-1] / (2.0 * dx)
    diag[1:] += dfactor[1:] * variance[1:] / (2.0 * dx)
    diag[0] += (0.5 / nu) * 2.0 / dx[0]
    diag[-1] += (0.5 / nu) * 2.0 / dx[-1]

    dt = T / n_steps
    system_diag = diag + 1.0 / dt
    for _ in range(n_steps):
        # dadi normalizes the point injection so its trapezoid integral is
        # theta*dt/(2*x1), not merely its stored array height.
        phi[1] += dt / xx[1] * theta / (xx[2] - xx[0])
        phi = _thomas_solve(lower, system_diag, upper, phi / dt)

    return phi


def crank_nicolson_1d(phi, xx, T, nu=1.0, theta=1.0, n_steps=100):
    """Compatibility wrapper for the old, inaccurate teaching API name.

    New code should call :func:`implicit_1d`; dadi uses backward Euler, not
    Crank-Nicolson.
    """
    return implicit_1d(phi, xx, T, nu=nu, theta=theta, n_steps=n_steps)


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
    model = np.asarray(model, dtype=float)
    data = np.asarray(data, dtype=float)
    if model.shape != data.shape or np.any(model < 0) or np.any(data < 0):
        raise ValueError("model and data must be matching non-negative arrays")
    positive = model > 0
    if np.any((data > 0) & ~positive):
        return -np.inf
    terms = -model - gammaln(data + 1.0)
    terms[positive] += data[positive] * np.log(model[positive])
    return float(np.sum(terms))


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
    scale = optimal_sfs_scaling(model, data)
    return poisson_log_likelihood(scale * np.asarray(model, dtype=float), data)


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
    model = np.asarray(model, dtype=float)
    data = np.asarray(data, dtype=float)
    if model.shape != data.shape or np.any(model < 0) or np.any(data < 0):
        raise ValueError("model and data must be matching non-negative arrays")
    model_sum = model.sum()
    if model_sum == 0:
        raise ValueError("model must have positive total mass")
    return float(data.sum() / model_sum)


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
    phi = implicit_1d(phi, xx, T, nu=nu, theta=theta)
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
    - Solving the neutral diffusion PDE (dadi-style backward Euler)
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
    phi_evolved = implicit_1d(phi, xx, T=0.5, nu=2.0, n_steps=200)
    print(f"Original total density:  {np.trapezoid(phi, xx):.4f}")
    print(f"Evolved total density:   {np.trapezoid(phi_evolved, xx):.4f}")

    # Extract SFS for sample size n=20
    sfs = sfs_from_phi(phi, xx, 20)
    print("\nEquilibrium SFS (first 5 entries):")
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
