"""
Figure: dadi diffusion approximation for demographic inference.

Shows the frequency grid, equilibrium SFS density, diffusion PDE evolution
under different population sizes, and the resulting discrete SFS.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_dadi import (
    equilibrium_sfs_density,
    make_nonuniform_grid,
    crank_nicolson_1d,
    sfs_from_phi,
    two_epoch_sfs,
    poisson_log_likelihood,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("dadi: Diffusion Approximation for Demographic Inference",
             fontsize=14, fontweight="bold")

# --- Panel A: Nonuniform grid and equilibrium density ---
ax = axes[0, 0]
pts = 80
xx = make_nonuniform_grid(pts)
phi = equilibrium_sfs_density(xx)

ax.plot(xx[1:-1], phi[1:-1], color="#1565C0", lw=2, label=r"$\phi(x) \propto 1/x$")
ax.fill_between(xx[1:-1], phi[1:-1], alpha=0.15, color="#1565C0")
# Show grid density with rug plot
ax.scatter(xx, np.zeros_like(xx) - 0.5, marker="|", s=30, color="#E65100", alpha=0.7,
           label="Grid points")
ax.set_xlabel("Derived allele frequency x")
ax.set_ylabel(r"Frequency density $\phi(x)$")
ax.set_title("A. Equilibrium neutral density")
ax.set_ylim(-1.5, 50)
ax.set_xlim(0, 1)
ax.legend(fontsize=8, loc="upper right")

# --- Panel B: Diffusion PDE evolution under expansion/contraction ---
ax = axes[0, 1]
phi_eq = equilibrium_sfs_density(xx) * 1.0  # theta = 1

phi_expand = crank_nicolson_1d(phi_eq, xx, T=0.5, nu=5.0, n_steps=300)
phi_contract = crank_nicolson_1d(phi_eq, xx, T=0.5, nu=0.2, n_steps=300)
phi_const = crank_nicolson_1d(phi_eq, xx, T=0.5, nu=1.0, n_steps=300)

ax.plot(xx[1:-1], phi_eq[1:-1], color="#757575", lw=1.5, ls="--", label="Equilibrium")
ax.plot(xx[1:-1], phi_const[1:-1], color="#4CAF50", lw=2, label=r"Constant ($\nu$=1)")
ax.plot(xx[1:-1], phi_expand[1:-1], color="#2196F3", lw=2, label=r"Expansion ($\nu$=5)")
ax.plot(xx[1:-1], phi_contract[1:-1], color="#F44336", lw=2, label=r"Contraction ($\nu$=0.2)")
ax.set_xlabel("Derived allele frequency x")
ax.set_ylabel(r"$\phi(x)$ after T=0.5")
ax.set_title("B. Diffusion under demographic change")
ax.set_ylim(0, 30)
ax.set_xlim(0, 0.5)
ax.legend(fontsize=8)

# --- Panel C: Discrete SFS under different models ---
ax = axes[1, 0]
n_samples = 20
sfs_eq = sfs_from_phi(phi_eq, xx, n_samples)
sfs_exp = two_epoch_sfs(nu=5.0, T=0.3, n_samples=n_samples, pts=pts)
sfs_con = two_epoch_sfs(nu=0.2, T=0.3, n_samples=n_samples, pts=pts)

k_vals = np.arange(1, n_samples)
w = 0.25
ax.bar(k_vals - w, sfs_eq[1:-1], width=w, label="Equilibrium", color="#757575", alpha=0.8)
ax.bar(k_vals, sfs_exp[1:-1], width=w, label=r"Expansion ($\nu$=5)", color="#2196F3", alpha=0.8)
ax.bar(k_vals + w, sfs_con[1:-1], width=w, label=r"Contraction ($\nu$=0.2)", color="#F44336", alpha=0.8)
ax.set_xlabel("Derived allele count k")
ax.set_ylabel("Expected SFS count")
ax.set_title(f"C. Discrete SFS (n={n_samples})")
ax.legend(fontsize=8)
ax.set_xlim(0.5, 15)

# --- Panel D: Likelihood surface for two-epoch model ---
ax = axes[1, 1]
# Generate "observed" data from a known model
np.random.seed(42)
true_nu, true_T = 2.0, 0.3
sfs_true = two_epoch_sfs(true_nu, true_T, n_samples=20, pts=80, theta=1.0)
sfs_true = np.maximum(sfs_true, 1e-10)
# Add Poisson noise
data = np.random.poisson(sfs_true * 100)

nu_grid = np.linspace(0.2, 5.0, 40)
T_grid = np.linspace(0.05, 0.8, 40)
LL = np.zeros((len(T_grid), len(nu_grid)))

for i, T_val in enumerate(T_grid):
    for j, nu_val in enumerate(nu_grid):
        model = two_epoch_sfs(nu_val, T_val, n_samples=20, pts=60, theta=1.0) * 100
        model = np.maximum(model, 1e-10)
        LL[i, j] = poisson_log_likelihood(model, data)

# Normalize for better visualization
LL_norm = LL - LL.max()
LL_norm = np.maximum(LL_norm, -50)

im = ax.contourf(nu_grid, T_grid, LL_norm, levels=20, cmap="viridis")
ax.plot(true_nu, true_T, "r*", ms=15, mew=2, label="True values")
ax.set_xlabel(r"Population size ratio $\nu$")
ax.set_ylabel("Time T (2N generations)")
ax.set_title("D. Log-likelihood surface (two-epoch)")
ax.legend(fontsize=9, loc="upper right")
plt.colorbar(im, ax=ax, label=r"$\Delta$LL")

plt.tight_layout()
plt.savefig("figures/fig_mini_dadi.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_dadi.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_dadi.png")
