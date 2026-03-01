"""
Demo: moments demographic inference on msprime-simulated SFS.

Simulates a population with known demographic history using msprime,
computes the observed SFS from the VCF-like genotype data, and runs
moments' ODE-based inference to recover the demographic parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_moments import (
    expected_sfs_neutral,
    integrate_sfs,
    tajimas_d,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime under different demographies ──────────
Ne = 10_000
mu = 1.25e-8
n_samples = 40  # 20 diploid individuals
L = 1_000_000   # 1 Mb

# Constant population
ts_const = msprime.simulate(
    sample_size=n_samples, Ne=Ne, length=L,
    mutation_rate=mu, random_seed=2024,
)

# Population with bottleneck
demo_bottle = msprime.Demography()
demo_bottle.add_population(initial_size=Ne)
demo_bottle.add_population_parameters_change(time=500, initial_size=2_000)
demo_bottle.add_population_parameters_change(time=2_000, initial_size=Ne)

ts_bottle = msprime.sim_ancestry(
    samples=n_samples // 2, demography=demo_bottle,
    sequence_length=L, recombination_rate=1e-8, random_seed=2025,
)
ts_bottle = msprime.sim_mutations(ts_bottle, rate=mu, random_seed=2025)

# Population with expansion
demo_expand = msprime.Demography()
demo_expand.add_population(initial_size=50_000)
demo_expand.add_population_parameters_change(time=1_000, initial_size=5_000)

ts_expand = msprime.sim_ancestry(
    samples=n_samples // 2, demography=demo_expand,
    sequence_length=L, recombination_rate=1e-8, random_seed=2026,
)
ts_expand = msprime.sim_mutations(ts_expand, rate=mu, random_seed=2026)

def compute_sfs_from_ts(ts, n):
    """Compute folded SFS from tree sequence."""
    G = ts.genotype_matrix()
    freq_counts = G.sum(axis=1)
    sfs = np.bincount(freq_counts, minlength=n + 1)
    return sfs[1:n]  # exclude fixed

# ── Compute observed SFS ────────────────────────────────────────
sfs_const = compute_sfs_from_ts(ts_const, n_samples)
sfs_bottle = compute_sfs_from_ts(ts_bottle, n_samples)
sfs_expand = compute_sfs_from_ts(ts_expand, n_samples)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: moments on msprime-simulated SFS ({n_samples} haplotypes, {L/1e6:.0f} Mb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Observed SFS from simulated VCF data
ax = axes[0, 0]
k_vals = np.arange(1, n_samples)[:20]
ax.bar(k_vals - 0.25, sfs_const[:20], width=0.25, color="#2166AC", alpha=0.8,
       label="Constant $N_e$")
ax.bar(k_vals, sfs_bottle[:20], width=0.25, color="#B2182B", alpha=0.8,
       label="Bottleneck")
ax.bar(k_vals + 0.25, sfs_expand[:20], width=0.25, color="#1B7837", alpha=0.8,
       label="Expansion")

ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Number of sites")
ax.set_title("A. Observed SFS from simulated VCF")
ax.legend(fontsize=7)

# Panel B: moments ODE prediction vs observed
ax = axes[0, 1]
theta_hat = sfs_const.sum() / sum(1/k for k in range(1, n_samples))
sfs_neutral_pred = expected_sfs_neutral(n_samples, theta=theta_hat)

k_all = np.arange(1, n_samples)
ax.plot(k_all[:20], sfs_const[:20], "o", color="#2166AC", ms=5,
        label="Observed (constant $N_e$)")
ax.plot(k_all[:20], sfs_neutral_pred[1:n_samples][:20], "s-", color="#B2182B",
        ms=4, lw=1.5, label=f"Neutral prediction ($\\hat{{\\theta}}$={theta_hat:.0f})")

ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Expected / observed count")
ax.set_title("B. moments prediction vs observation")
ax.legend(fontsize=8)

# Panel C: SFS distortion signatures (theoretical, using integrate_sfs)
ax = axes[1, 0]
n_theory = 30
theta_theory = 200.0
phi_eq = expected_sfs_neutral(n_theory, theta=theta_theory)

# Two-phase bottleneck: contraction (nu=0.2, T=0.2) then recovery (nu=1.0, T=0.2)
phi_b1 = integrate_sfs(phi_eq.copy(), n_theory, T=0.2, nu_func=lambda t: 0.2, theta=theta_theory)
phi_bottle_theory = integrate_sfs(phi_b1.copy(), n_theory, T=0.2, nu_func=lambda t: 1.0, theta=theta_theory)

# Expansion: nu=5.0 for T=0.5
phi_expand_theory = integrate_sfs(phi_eq.copy(), n_theory, T=0.5, nu_func=lambda t: 5.0, theta=theta_theory)

def normalize_phi(phi, n):
    s = phi[1:n].copy()
    return s / (s.sum() + 1e-30)

k_theory = np.arange(1, n_theory)
norm_const = normalize_phi(phi_eq, n_theory)
norm_bottle = normalize_phi(phi_bottle_theory, n_theory)
norm_expand = normalize_phi(phi_expand_theory, n_theory)

ax.plot(k_theory, norm_const, "o-", color="#2166AC", lw=2, ms=4, label="Constant (equilibrium)")
ax.plot(k_theory, norm_bottle, "s-", color="#B2182B", lw=2, ms=4, label="Bottleneck (theory)")
ax.plot(k_theory, norm_expand, "^-", color="#1B7837", lw=2, ms=4, label="Expansion (theory)")

ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Normalized SFS (proportion)")
ax.set_title("C. Demographic signature in SFS shape")
ax.set_yscale("log")
ax.legend(fontsize=8)

# Panel D: Tajima's D
ax = axes[1, 1]
theta_sfs = 200.0  # use moments SFS scale
n_d = 30
phi_eq = expected_sfs_neutral(n_d, theta=theta_sfs)
T_values = np.linspace(0.02, 1.5, 20)

D_const_m = []
D_expand_m = []
D_contract_m = []
for T_val in T_values:
    phi_c = integrate_sfs(phi_eq.copy(), n_d, T=T_val,
                          nu_func=lambda t: 1.0, theta=theta_sfs)
    D_const_m.append(tajimas_d(phi_c))
    phi_e = integrate_sfs(phi_eq.copy(), n_d, T=T_val,
                          nu_func=lambda t: 5.0, theta=theta_sfs)
    D_expand_m.append(tajimas_d(phi_e))
    phi_r = integrate_sfs(phi_eq.copy(), n_d, T=T_val,
                          nu_func=lambda t: 0.2, theta=theta_sfs)
    D_contract_m.append(tajimas_d(phi_r))

ax.plot(T_values, D_const_m, "o-", color="#2166AC", lw=2, ms=4, label="Constant")
ax.plot(T_values, D_expand_m, "^-", color="#1B7837", lw=2, ms=4, label="Expansion")
ax.plot(T_values, D_contract_m, "v-", color="#B2182B", lw=2, ms=4, label="Contraction")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_xlabel("Time $T$ ($2N_e$ generations)")
ax.set_ylabel("Tajima's $D$")
ax.set_title("D. Tajima's $D$ as demographic diagnostic")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_moments.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_moments.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_moments.png")
