"""
Demo: PSMC on msprime-simulated diploid genome with bottleneck history.

Illustrates the core PSMC HMM machinery: heterozygosity input, time
discretization, posterior decoding of coalescence times, and the
reconstructed N(t) curve from the true parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_psmc import (
    simulate_psmc_input,
    PSMC_HMM,
    scale_psmc_output,
    compute_time_intervals,
    posterior_decoding,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Realistic human-like parameters ─────────────────────────────
Ne0 = 10_000
mu = 1.25e-8
gen_time = 25
theta = 4 * Ne0 * mu
rho = theta / 4.0

# Bottleneck demography: true lambdas (piecewise constant)
n_intervals = 15
t_max = 15.0
alpha = 0.1

t_bounds = compute_time_intervals(n_intervals, t_max, alpha)

true_lambdas = np.ones(n_intervals + 1)
for k in range(n_intervals + 1):
    frac = k / n_intervals
    if frac < 0.15:
        true_lambdas[k] = 1.0
    elif frac < 0.35:
        true_lambdas[k] = 0.25
    elif frac < 0.55:
        true_lambdas[k] = 0.5
    elif frac < 0.75:
        true_lambdas[k] = 1.2
    else:
        true_lambdas[k] = 2.0

def lambda_true(t):
    for k in range(n_intervals + 1):
        if t <= t_bounds[k + 1]:
            return true_lambdas[k]
    return true_lambdas[-1]

# Simulate diploid heterozygosity sequence
L = 100_000
seq, true_coal_times = simulate_psmc_input(L, theta, rho, lambda_true)

# ── Build HMM with true parameters and decode ───────────────────
hmm = PSMC_HMM(n_intervals, theta, rho, true_lambdas, t_max, alpha)
posterior, map_states = posterior_decoding(hmm, seq)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: PSMC on Simulated Human Diploid Genome (Bottleneck History)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Input heterozygosity sequence
ax = axes[0, 0]
window = 500
het_show = seq[:50_000].astype(float)
het_rate = np.convolve(het_show, np.ones(window) / window, mode="valid")
ax.plot(np.arange(len(het_rate)), het_rate, color="#2166AC", lw=0.5, alpha=0.7)
ax.set_xlabel("Genomic position (bp)")
ax.set_ylabel(f"Heterozygosity ({window}-bp window)")
ax.set_title(f"A. Input: diploid consensus ({L:,} bp)")
n_het = seq.sum()
ax.text(0.02, 0.95, f"{n_het:,} het sites ({n_het/L:.4f})",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Panel B: PSMC time discretization and true N(t)
ax = axes[0, 1]
s = 100
N0_true, _, t_years, Nt_true = scale_psmc_output(
    theta, true_lambdas, t_bounds, mu, s, gen_time)

x_steps, y_steps = [], []
for k in range(n_intervals + 1):
    t_lo = max(t_years[k], 1)
    t_hi = t_years[k + 1]
    x_steps.extend([t_lo, t_hi])
    y_steps.extend([Nt_true[k], Nt_true[k]])

ax.plot(x_steps, y_steps, color="#B2182B", lw=2.5, label="True $N(t)$")
for tb in t_years[1:-1]:
    ax.axvline(tb, color='gray', lw=0.4, alpha=0.4)
ax.set_xscale("log")
ax.set_xlabel("Years before present")
ax.set_ylabel("Effective population size $N_e$")
ax.set_title("B. True N(t) and PSMC time discretization")
ax.legend(fontsize=8)

# Panel C: Posterior decoding heatmap (first 2000 bins)
ax = axes[1, 0]
win_len = 2000
post_win = posterior[:win_len]   # shape (2000, N)
im = ax.imshow(post_win.T, aspect="auto", origin="lower",
               cmap="Blues", interpolation="nearest",
               vmin=0, vmax=post_win.max())
het_pos = np.where(seq[:win_len] == 1)[0]
for pos in het_pos:
    ax.axvline(pos, color="#B2182B", lw=0.5, alpha=0.6)
ax.set_xlabel("Genomic position (bin)")
ax.set_ylabel("Time interval (hidden state)")
ax.set_title("C. Posterior decoding of hidden states (first 2 kb)")
plt.colorbar(im, ax=ax, label="Posterior prob", shrink=0.8)
# Mark het sites in legend
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0], color="#B2182B", lw=1, label="Het site")],
          fontsize=7, loc="upper right")

# Panel D: True coalescence times vs MAP state along sequence
ax = axes[1, 1]
# Subsample for clarity
step = 50
pos_sub = np.arange(0, min(L, 50_000), step)
tc_sub = true_coal_times[pos_sub]
map_sub = map_states[pos_sub]

ax.scatter(tc_sub, map_sub, s=2, alpha=0.3, color="#2166AC",
           rasterized=True)
ax.set_xlabel("True coalescence time $t$ (coalescent units)")
ax.set_ylabel("MAP hidden state $k$")
ax.set_title("D. True coalescence time vs MAP state")
# Add a reference line: expected state for each time
ax.set_xlim(left=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_psmc.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_psmc.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_psmc.png")
