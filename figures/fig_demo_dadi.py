"""
Demo: dadi diffusion approximation on msprime-simulated SFS.

Simulates populations with known demography using msprime, computes
the SFS, and runs dadi's Crank-Nicolson PDE solver to predict the
expected spectrum under different demographic models.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_dadi import (
    make_nonuniform_grid,
    equilibrium_sfs_density,
    crank_nicolson_1d,
    sfs_from_phi,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime ───────────────────────────────────────
Ne = 10_000
mu = 1.25e-8
n_samples = 30
L = 1_000_000

ts = msprime.simulate(
    sample_size=n_samples, Ne=Ne, length=L,
    mutation_rate=mu, random_seed=2024,
)

G = ts.genotype_matrix()
freq_counts = G.sum(axis=1)
observed_sfs = np.bincount(freq_counts, minlength=n_samples + 1)[1:n_samples]

# ── Run dadi diffusion solver ───────────────────────────────────
pts = 100
xx = make_nonuniform_grid(pts)
theta = 4 * Ne * mu * L
theta_unit = 1.0

# Equilibrium density
phi_eq = equilibrium_sfs_density(xx)

# Constant population
phi_const = crank_nicolson_1d(phi_eq.copy(), xx, T=1.0, nu=1.0,
                              theta=theta_unit, n_steps=500)
sfs_const = sfs_from_phi(phi_const, xx, n_samples - 1)

# 5x expansion
phi_expand = crank_nicolson_1d(phi_eq.copy(), xx, T=0.5, nu=5.0,
                               theta=theta_unit, n_steps=500)
sfs_expand = sfs_from_phi(phi_expand, xx, n_samples - 1)

# 5x contraction
phi_contract = crank_nicolson_1d(phi_eq.copy(), xx, T=0.5, nu=0.2,
                                 theta=theta_unit, n_steps=500)
sfs_contract = sfs_from_phi(phi_contract, xx, n_samples - 1)

# Scale to match theta
sfs_const_scaled = sfs_const * theta
sfs_expand_scaled = sfs_expand * theta
sfs_contract_scaled = sfs_contract * theta

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: dadi Diffusion Solver on msprime-simulated SFS ({n_samples} haplotypes, 1 Mb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Observed vs predicted SFS (constant model)
ax = axes[0, 0]
k_vals = np.arange(1, n_samples)
ax.bar(k_vals[:20] - 0.2, observed_sfs[:20], width=0.4, color="#2166AC", alpha=0.7,
       label="Observed (msprime)")
ax.bar(k_vals[:20] + 0.2, sfs_const_scaled[:20], width=0.4, color="#B2182B", alpha=0.7,
       label="dadi constant $N_e$")
ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Number of sites")
ax.set_title("A. Observed vs dadi predicted SFS")
ax.set_xlim(0.5, 20.5)
ax.legend(fontsize=8)

# Panel B: Frequency density evolution
ax = axes[0, 1]
ax.plot(xx, phi_eq, color="#636363", lw=1.5, ls="--", label="Equilibrium")
ax.plot(xx, phi_const, color="#2166AC", lw=2, label="Constant ($\\nu=1$)")
ax.plot(xx, phi_expand, color="#1B7837", lw=2, label="Expansion ($\\nu=5$)")
ax.plot(xx, phi_contract, color="#B2182B", lw=2, label="Contraction ($\\nu=0.2$)")
ax.set_xlabel("Derived allele frequency $x$")
ax.set_ylabel("Frequency density $\\phi(x)$")
ax.set_title("B. Diffusion density under different $N_e$")
ax.set_xlim(0, 1)
ax.set_ylim(0, min(phi_eq.max() * 1.5, 100))
ax.legend(fontsize=8)

# Panel C: Non-uniform grid visualization
ax = axes[1, 0]
ax.plot(range(len(xx)), xx, "o-", color="#2166AC", ms=2, lw=1)
ax.set_xlabel("Grid point index")
ax.set_ylabel("Frequency $x$")
ax.set_title(f"C. Non-uniform frequency grid ({pts} points)")
ax_inset = ax.inset_axes([0.5, 0.15, 0.45, 0.4])
dx = np.diff(xx)
ax_inset.plot(xx[:-1], dx, color="#B2182B", lw=1.5)
ax_inset.set_xlabel("$x$", fontsize=7)
ax_inset.set_ylabel("$\\Delta x$", fontsize=7)
ax_inset.set_title("Grid spacing", fontsize=7)
ax_inset.tick_params(labelsize=6)

# Panel D: SFS under three demographic models
ax = axes[1, 1]
k_show = np.arange(1, min(20, n_samples))
ax.plot(k_show, sfs_const_scaled[:len(k_show)], "o-", color="#2166AC", lw=2,
        ms=4, label="Constant")
ax.plot(k_show, sfs_expand_scaled[:len(k_show)], "^-", color="#1B7837", lw=2,
        ms=4, label="Expansion ($\\nu=5$)")
ax.plot(k_show, sfs_contract_scaled[:len(k_show)], "v-", color="#B2182B", lw=2,
        ms=4, label="Contraction ($\\nu=0.2$)")
ax.plot(k_show, observed_sfs[:len(k_show)], "s", color="#636363", ms=5,
        alpha=0.6, label="Observed")
ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Expected sites")
ax.set_title("D. dadi SFS predictions vs data")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_dadi.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_dadi.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_dadi.png")
