"""
Demo: Gamma-SMC pairwise TMRCA inference on msprime-simulated data.

Simulates a diploid genome with msprime, extracts the heterozygosity
sequence, constructs a moderate-resolution canonical flow field, and runs
the mini Gamma-SMC forward-backward approximation.
"""

import matplotlib.pyplot as plt
import msprime
import numpy as np

from watchgen.mini_gamma_smc import (
    FlowField,
    compute_flow_at_point,
    gamma_smc_posterior,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate diploid genome with msprime ────────────────────────
Ne = 10_000
mu = 1.25e-8
rho = 1e-8
L = 500_000  # 500 kb

# Simulate with bottleneck
demo = msprime.Demography()
demo.add_population(initial_size=Ne)
demo.add_population_parameters_change(time=500, initial_size=2_000)
demo.add_population_parameters_change(time=2_000, initial_size=Ne)

ts = msprime.sim_ancestry(
    samples=1, demography=demo, sequence_length=L,
    recombination_rate=rho, random_seed=2024,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=2024)

# Extract heterozygosity sequence (0=hom, 1=het at each bp)
het_positions = set()
for var in ts.variants():
    geno = var.genotypes
    if geno[0] != geno[1]:
        het_positions.add(int(var.position))

# Create observation sequence binned into windows
window_size = 100  # bp per window
n_windows = int(L / window_size)
observations = np.zeros(n_windows, dtype=int)
for pos in het_positions:
    win_idx = min(int(pos / window_size), n_windows - 1)
    observations[win_idx] += 1

# ── Run Gamma-SMC forward pass ──────────────────────────────────
# Rates use coalescent time units, where one unit is 2*Ne generations.
theta_win = 4 * Ne * mu * window_size
rho_win = 4 * Ne * rho * window_size

# Build a compact teaching grid. The official distributed grid is denser and
# was generated with Arb ball arithmetic; see the chapter discussion.
l_mu_grid = np.linspace(-1.2, 1.2, 25)
l_C_grid = np.linspace(-0.8, 0.0, 18)
delta_l_mu = np.empty((len(l_mu_grid), len(l_C_grid)))
delta_l_C = np.empty_like(delta_l_mu)
for i, l_mu in enumerate(l_mu_grid):
    for j, l_C in enumerate(l_C_grid):
        delta_l_mu[i, j], delta_l_C[i, j] = compute_flow_at_point(
            l_mu, l_C, n_eval=300
        )
flow_field = FlowField(l_mu_grid, l_C_grid, delta_l_mu, delta_l_C)

alphas, betas = gamma_smc_posterior(
    observations, theta_win, rho_win, flow_field
)
means = (alphas / betas) * (2 * Ne)
variances = (alphas / betas**2) * (2 * Ne) ** 2

# True TMRCA from tree sequence
true_tmrca = np.zeros(n_windows)
for tree in ts.trees():
    start_win = int(tree.interval[0] / window_size)
    end_win = min(int(tree.interval[1] / window_size), n_windows)
    root = tree.roots[0] if tree.num_roots > 0 else None
    if root is not None:
        tmrca = tree.time(root)
        true_tmrca[start_win:end_win] = tmrca

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: Gamma-SMC on msprime Diploid Genome (500 kb, bottleneck history)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: True TMRCA along genome
ax = axes[0, 0]
positions_kb = np.arange(n_windows) * window_size / 1000
ax.plot(positions_kb, true_tmrca, color="#B2182B", lw=0.8, alpha=0.7,
        label="True TMRCA")
ax.plot(positions_kb, means, color="#2166AC", lw=0.8, alpha=0.7,
        label="Gamma-SMC posterior mean")
ax.set_xlabel("Position (kb)")
ax.set_ylabel("TMRCA (generations)")
ax.set_title("A. TMRCA along the genome")
ax.legend(fontsize=8)

# Panel B: Heterozygosity track
ax = axes[0, 1]
ax.plot(positions_kb, observations, color="#636363", lw=0.3, alpha=0.6)
# Smoothed
window_smooth = 50
het_smooth = np.convolve(observations.astype(float),
                         np.ones(window_smooth)/window_smooth, mode="valid")
ax.plot(positions_kb[window_smooth//2:window_smooth//2+len(het_smooth)],
        het_smooth, color="#2166AC", lw=1.5, label=f"{window_smooth}-window avg")
ax.set_xlabel("Position (kb)")
ax.set_ylabel(f"Het sites per {window_size} bp window")
ax.set_title(f"B. Input heterozygosity ({len(het_positions)} het sites)")
ax.legend(fontsize=8)

# Panel C: Posterior confidence (alpha parameter)
ax = axes[1, 0]
ax.plot(positions_kb, alphas, color="#1B7837", lw=0.8, alpha=0.7)
ax.set_xlabel("Position (kb)")
ax.set_ylabel("Posterior $\\alpha$")
ax.set_title("C. Posterior precision (shape parameter)")

# Inset: relationship between alpha and CI width
ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.35])
posterior_cvs = np.sqrt(variances) / (means + 1e-10)
ax_in.scatter(alphas[:500], posterior_cvs[:500], s=3, alpha=0.3,
              color="#1B7837")
ax_in.set_xlabel("$\\alpha$", fontsize=7)
ax_in.set_ylabel("CV", fontsize=7)
ax_in.set_title("Precision vs CV", fontsize=7)
ax_in.tick_params(labelsize=6)

# Panel D: True vs inferred TMRCA scatter
ax = axes[1, 1]
ax.scatter(true_tmrca, means, s=5, alpha=0.2, color="#2166AC",
           edgecolors="none")
max_val = max(true_tmrca.max(), means.max()) * 1.1
ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.4, label="$y=x$")
ax.set_xlabel("True TMRCA (generations)")
ax.set_ylabel("Mini Gamma-SMC posterior mean")
ax.set_title("D. True vs inferred TMRCA")
corr = np.corrcoef(true_tmrca, means)[0, 1]
ax.text(0.02, 0.95, f"$r$ = {corr:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "gray",
              "alpha": 0.8})
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_gamma_smc.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_gamma_smc.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_gamma_smc.png")
