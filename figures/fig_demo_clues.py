"""
Demo: CLUES selection coefficient inference.

Uses CLUES' HMM machinery (transition matrices, frequency bins) to
visualize how selection distorts allele frequency dynamics and how
the likelihood surface changes with selection coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt

from watchgen.mini_clues import (
    build_frequency_bins,
    build_transition_matrix,
    forward_algorithm,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Setup ───────────────────────────────────────────────────────
Ne = 10_000
K = 100
freqs = np.asarray(build_frequency_bins(K)[0])

# Simulate Wright-Fisher trajectories
def wf_trajectory(s, Ne, n_gen, x0=0.3):
    x = x0
    traj = [x]
    for _ in range(n_gen):
        x_sel = x * (1 + s) / (1 + s * x)
        x = np.random.binomial(2 * Ne, max(min(x_sel, 1 - 1e-10), 1e-10)) / (2 * Ne)
        if x <= 0 or x >= 1:
            break
        traj.append(x)
    return np.array(traj)

trajs = {
    0.0: wf_trajectory(0.0, Ne, 200, 0.3),
    0.005: wf_trajectory(0.005, Ne, 200, 0.3),
    -0.005: wf_trajectory(-0.005, Ne, 200, 0.3),
    0.01: wf_trajectory(0.01, Ne, 200, 0.1),
}

# Build transition matrices for different s values
# Note: build_transition_matrix returns LOG-probability matrices
logA_neutral = build_transition_matrix(freqs, Ne, s=0.0, h=0.5)
logA_pos = build_transition_matrix(freqs, Ne, s=0.005, h=0.5)

# Convert to probability space for power iteration and visualization
A_neutral = np.exp(logA_neutral)
A_pos = np.exp(logA_pos)

# Compute stationary-like likelihood for a range of s values
s_test_vals = np.linspace(-0.015, 0.015, 61)
ll_surface = []
for s_val in s_test_vals:
    logA = build_transition_matrix(freqs, Ne, s_val, h=0.5)
    A = np.exp(logA)
    # Compute steady state via power iteration
    pi = np.ones(K) / K
    for _ in range(200):
        pi = pi @ A
        pi = pi / (pi.sum() + 1e-30)
    freq_idx = np.argmin(np.abs(freqs - 0.3))
    ll_surface.append(np.log(pi[freq_idx] + 1e-30))

ll_surface = np.array(ll_surface)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: CLUES Selection Inference ($N_e$={Ne:,}, {K} frequency bins)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Allele frequency trajectories
ax = axes[0, 0]
colors = {0.0: "#636363", 0.005: "#B2182B", -0.005: "#2166AC", 0.01: "#1B7837"}
for s_val, traj in trajs.items():
    label = f"$s$ = {s_val}" + (" (neutral)" if s_val == 0 else "")
    ax.plot(range(len(traj)), traj, lw=1.5, alpha=0.8,
            color=colors[s_val], label=label)
ax.set_xlabel("Generation")
ax.set_ylabel("Allele frequency")
ax.set_title("A. Wright-Fisher trajectories under selection")
ax.legend(fontsize=7)
ax.set_ylim(0, 1)

# Panel B: Transition matrix difference (selection vs neutral)
ax = axes[0, 1]
diff = A_pos - A_neutral
show_k = min(40, K)
vmax = np.abs(diff[:show_k, :show_k]).max()
im = ax.imshow(diff[:show_k, :show_k], aspect="auto", cmap="RdBu_r",
               interpolation="nearest", origin="lower",
               vmin=-vmax, vmax=vmax)
ax.set_xlabel("Destination frequency bin")
ax.set_ylabel("Source frequency bin")
ax.set_title("B. Transition: $s$=0.005 minus neutral")
plt.colorbar(im, ax=ax, label="$\\Delta P$", shrink=0.8)

# Panel C: Likelihood surface
ax = axes[1, 0]
valid = ~np.isnan(ll_surface) & ~np.isinf(ll_surface)
if valid.any():
    ax.plot(s_test_vals[valid], ll_surface[valid], color="#2166AC", lw=2)
    best_idx = np.nanargmax(ll_surface[valid])
    valid_s = s_test_vals[valid]
    ax.axvline(valid_s[best_idx], color="#B2182B", ls="--", lw=1.5,
               label=f"MLE $\\hat{{s}}$ = {valid_s[best_idx]:.4f}")
ax.axvline(0, color="#636363", ls=":", lw=1, alpha=0.5, label="Neutral")
ax.set_xlabel("Selection coefficient $s$")
ax.set_ylabel("Log stationary probability")
ax.set_title("C. Likelihood surface at $x$=0.3")
ax.legend(fontsize=8)

# Panel D: Frequency bin discretization
ax = axes[1, 1]
ax.plot(range(K), freqs, "o-", color="#2166AC", ms=2, lw=1)
ax.set_xlabel("Bin index")
ax.set_ylabel("Allele frequency")
ax.set_title(f"D. Beta-quantile frequency bins ($K$={K})")

ax_in = ax.inset_axes([0.5, 0.15, 0.45, 0.35])
bin_widths = np.diff(freqs)
ax_in.bar(range(len(bin_widths)), bin_widths, color="#B2182B", alpha=0.7,
          width=1.0)
ax_in.set_xlabel("Bin", fontsize=7)
ax_in.set_ylabel("Width", fontsize=7)
ax_in.set_title("Finer at boundaries", fontsize=7)
ax_in.tick_params(labelsize=6)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_clues.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_clues.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_clues.png")
