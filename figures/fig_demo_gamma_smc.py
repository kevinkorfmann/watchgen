"""
Demo: Gamma-SMC pairwise TMRCA inference on msprime-simulated data.

Simulates a diploid genome with msprime, extracts the heterozygosity
sequence, and runs Gamma-SMC's conjugate Bayesian inference to track
TMRCA posteriors along the genome.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_gamma_smc import (
    gamma_emission_update,
    FlowField,
    to_log_coords,
    from_log_coords,
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
# Per-pair Poisson rate: Y ~ Poisson(2*mu*window_size * T) where T is in generations
theta_win = 2 * mu * window_size
rho_win = 4 * Ne * rho * window_size

# Initialize gamma prior
alpha_init = 2.0
beta_init = 2.0 / (2 * Ne)  # mean = 2*Ne

alphas = np.zeros(n_windows)
betas = np.zeros(n_windows)
means = np.zeros(n_windows)
variances = np.zeros(n_windows)

alpha_curr = alpha_init
beta_curr = beta_init

for i in range(n_windows):
    y = observations[i]
    alpha_new, beta_new = gamma_emission_update(alpha_curr, beta_curr, y, theta_win)
    alphas[i] = alpha_new
    betas[i] = beta_new
    means[i] = alpha_new / beta_new
    variances[i] = alpha_new / (beta_new ** 2)

    # Predict step: widen the posterior (approximate transition)
    alpha_curr = max(alpha_new * (1 - rho_win) + alpha_init * rho_win, 1.01)
    beta_curr = beta_new * (1 - rho_win) + beta_init * rho_win

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
    f"Demo: Gamma-SMC on msprime Diploid Genome (500 kb, bottleneck history)",
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
ci_widths = 2 * np.sqrt(variances) / (means + 1e-10)
ax_in.scatter(alphas[:500], ci_widths[:500], s=3, alpha=0.3, color="#1B7837")
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
ax.set_ylabel("Gamma-SMC posterior mean")
ax.set_title("D. True vs inferred TMRCA")
corr = np.corrcoef(true_tmrca, means)[0, 1]
ax.text(0.02, 0.95, f"$r$ = {corr:.3f}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_gamma_smc.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_gamma_smc.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_gamma_smc.png")
