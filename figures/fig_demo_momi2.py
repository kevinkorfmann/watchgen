"""
Demo: momi2 coalescent tensor computation on msprime-simulated SFS.

Simulates data with msprime, computes the observed SFS, and uses
momi2's W-matrix and Moran model machinery to predict it.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_momi2 import (
    w_matrix,
    etjj_constant,
    moran_transition,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime ───────────────────────────────────────
Ne = 10_000
mu = 1.25e-8
n_samples = 20
L = 1_000_000

ts = msprime.simulate(
    sample_size=n_samples, Ne=Ne, length=L,
    mutation_rate=mu, random_seed=2024,
)

G = ts.genotype_matrix()
freq_counts = G.sum(axis=1)
observed_sfs = np.bincount(freq_counts, minlength=n_samples + 1)[1:n_samples]

# ── Compute momi2 predictions ──────────────────────────────────
theta = 4 * Ne * mu * L

# Expected SFS using W-matrix: E[SFS_i] = theta * sum_j W_ij * E[T_{j,j}]
W = w_matrix(n_samples)
# For constant population, E[T_{j,j}] = 1/(j choose 2) in coalescent units
n_int = n_samples
e_tjj = np.zeros(n_int + 1)
for j in range(2, n_int + 1):
    e_tjj[j] = 2.0 / (j * (j - 1))  # standard coalescent

predicted_sfs = np.zeros(n_samples - 1)
for i in range(n_samples - 1):
    for j in range(2, n_int + 1):
        if j - 2 < W.shape[1] and i < W.shape[0]:
            predicted_sfs[i] += W[i, j - 2] * e_tjj[j]
predicted_sfs *= theta

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: momi2 Tensor Computation on msprime SFS ({n_samples} haplotypes, 1 Mb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Observed vs predicted SFS
ax = axes[0, 0]
k_vals = np.arange(1, n_samples)
k_show = min(15, n_samples - 1)
ax.bar(k_vals[:k_show] - 0.2, observed_sfs[:k_show], width=0.4,
       color="#2166AC", alpha=0.7, label="Observed (msprime)")
ax.bar(k_vals[:k_show] + 0.2, predicted_sfs[:k_show], width=0.4,
       color="#B2182B", alpha=0.7, label="momi2 W-matrix prediction")
ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Number of sites")
ax.set_title("A. Observed vs momi2 predicted SFS")
ax.set_xlim(0.5, k_show + 0.5)
ax.legend(fontsize=8)

# Panel B: W-matrix visualization
ax = axes[0, 1]
n_show = min(n_samples, 12)
W_show = w_matrix(n_show)
im = ax.imshow(W_show, aspect="auto", cmap="RdBu_r", interpolation="nearest")
ax.set_xlabel("Column index $j$")
ax.set_ylabel("Row index $i$ (SFS entry)")
ax.set_title(f"B. Polanski-Kimmel $W$-matrix ($n$={n_show})")
plt.colorbar(im, ax=ax, label="$W_{ij}$", shrink=0.8)

# Panel C: E[T_{j,j}] for different epoch durations tau (in coalescent units)
ax = axes[1, 0]
j_vals = np.arange(2, n_samples + 1)
etjj_vals = [2.0 / (j * (j - 1)) for j in j_vals]

# etjj_constant expects tau in generations; convert from coalescent units (1 cu = 2Ne gen)
tau_vals_coal = [0.1, 0.5, 1.0, 2.0]
colors_c = ["#2166AC", "#B2182B", "#1B7837", "#E08214"]
for tau_coal, color in zip(tau_vals_coal, colors_c):
    tau_gen = tau_coal * 2 * Ne  # convert to generations
    sojourn = etjj_constant(n_samples, tau_gen, Ne)  # array length n_samples-1, index j-2
    etjj_tau = [float(sojourn[j - 2]) for j in j_vals]
    ax.plot(j_vals, etjj_tau, "o-", color=color, ms=3, lw=1.5,
            label=f"$\\tau$ = {tau_coal}")

ax.plot(j_vals, etjj_vals, "k--", lw=1, alpha=0.5, label="$\\tau \\to \\infty$: $2/j(j-1)$")
ax.set_xlabel("Number of lineages $j$")
ax.set_ylabel("$E[T_{j,j}]$ (coalescent units)")
ax.set_title("C. Expected sojourn time with $j$ lineages")
ax.legend(fontsize=7)

# Panel D: Moran transition matrix
ax = axes[1, 1]
n_moran = 10
t_moran = 0.5
M = moran_transition(t_moran, n_moran)
im = ax.imshow(M, aspect="auto", cmap="YlOrRd", interpolation="nearest",
               origin="lower")
ax.set_xlabel("Destination state")
ax.set_ylabel("Source state")
ax.set_title(f"D. Moran transition matrix ($n$={n_moran}, $t$={t_moran})")
plt.colorbar(im, ax=ax, label="$P(j \\to k)$", shrink=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_momi2.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_momi2.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_momi2.png")
