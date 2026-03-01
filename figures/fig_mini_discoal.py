"""
Figure: discoal -- coalescent simulation with selection.

Shows selective sweep signatures: allele frequency trajectories,
diversity reduction around a sweep, SFS distortion, and linkage patterns
(escape probability vs distance from the selected site).
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_discoal import (
    deterministic_trajectory,
    stochastic_trajectory,
    compare_trajectories,
    minimal_discoal,
    hard_sweep_genealogy,
    soft_sweep_standing_variation,
    escape_probability,
    sweep_duration_table,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("discoal: Coalescent Simulation with Selection",
             fontsize=14, fontweight="bold")

# --- Panel A: Allele frequency trajectories (deterministic vs stochastic) ---
ax = axes[0, 0]
N = 500   # small N for tractable stochastic random walk
s = 0.05  # 2Ns = 50

det_traj = deterministic_trajectory(s, N)
det_time = np.arange(len(det_traj))

# Plot multiple stochastic trajectories
rng = np.random.default_rng(42)
stoch_colors = ["#90CAF9", "#81D4FA", "#80DEEA", "#A5D6A7", "#C5E1A5"]
for i in range(5):
    stoch = stochastic_trajectory(s, N, rng=rng)
    stoch_time = np.arange(len(stoch))
    ax.plot(stoch_time, stoch, color=stoch_colors[i], lw=0.8, alpha=0.7,
            label="Stochastic" if i == 0 else None)

ax.plot(det_time, det_traj, color="#D32F2F", lw=2.5, label="Deterministic")

# Mark sweep duration
T_theory = 2 * np.log(2 * N) / s
ax.axvline(T_theory, color="#757575", ls=":", lw=1, alpha=0.6)
ax.text(T_theory + 10, 0.5, f"Theory: {T_theory:.0f} gen",
        fontsize=8, color="#757575", rotation=90, va="center")

ax.set_xlabel("Generation")
ax.set_ylabel("Allele frequency")
ax.set_title(f"A. Sweep trajectories (2Ns = {2*N*s:.0f})")
ax.legend(fontsize=8, loc="center right")
ax.set_ylim(-0.02, 1.02)

# --- Panel B: Diversity profile around a hard sweep ---
ax = axes[1, 0]
# Run the minimal discoal diversity profile for multiple selection strengths
# Use smaller N to keep the structured coalescent tractable
N_b = 500
L = 50_000
n_sample = 6
r_per_site = 2e-7  # higher r to compensate for smaller genome

colors_b = ["#1565C0", "#E65100", "#2E7D32"]
for s_val, color, alpha_label in [
    (0.05, colors_b[0], "50"),
    (0.10, colors_b[1], "100"),
    (0.20, colors_b[2], "200"),
]:
    positions, rel_div = minimal_discoal(
        n=n_sample, N=N_b, s=s_val, r_per_site=r_per_site,
        L=L, n_sites=25, seed=42)
    # Convert positions to distance from center in kb
    dist_kb = (positions - L / 2) / 1000
    ax.plot(dist_kb, rel_div, lw=1.8, color=color,
            label=rf"2Ns = {int(2*N_b*s_val)}", alpha=0.9)

ax.axhline(1.0, color="#757575", ls="--", lw=1, alpha=0.5)
ax.set_xlabel("Distance from selected site (kb)")
ax.set_ylabel("Relative diversity (vs neutral)")
ax.set_title("B. Diversity valley around a sweep")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0, 2.5)

# --- Panel C: SFS distortion from hard vs soft sweeps ---
ax = axes[1, 1]
rng = np.random.default_rng(123)
n_reps = 200
n_sample_c = 8
N_c = 500
s_c = 0.05

# Collect coalescence times and build SFS proxies
# We use TMRCA distribution as a proxy for tree height (diversity)
hard_tmrcas = []
soft_tmrcas = []
neutral_tmrcas = []

for _ in range(n_reps):
    # Hard sweep (perfectly linked)
    hard_ct, _ = hard_sweep_genealogy(n_sample_c, N_c, s_c, 0.0, rng=rng)
    if hard_ct:
        hard_tmrcas.append(max(hard_ct))

    # Soft sweep (x0 = 0.05)
    soft_ct, _ = soft_sweep_standing_variation(
        n_sample_c, N_c, s_c, x0=0.05, r_site=0.0, rng=rng)
    if soft_ct:
        soft_tmrcas.append(max(soft_ct))

    # Neutral
    n_temp = n_sample_c
    t = 0
    while n_temp > 1:
        rate = n_temp * (n_temp - 1) / (2.0 * 2 * N_c)
        t += rng.exponential(1.0 / rate)
        n_temp -= 1
    neutral_tmrcas.append(t)

# Plot TMRCA distributions as proxy for how sweeps distort genealogies
bins = np.linspace(0, max(neutral_tmrcas) * 1.2, 50)
ax.hist(neutral_tmrcas, bins=bins, alpha=0.5, color="#757575",
        label="Neutral", density=True, edgecolor="white", lw=0.5)
ax.hist(soft_tmrcas, bins=bins, alpha=0.5, color="#2196F3",
        label=f"Soft sweep (x$_0$=0.05)", density=True,
        edgecolor="white", lw=0.5)
ax.hist(hard_tmrcas, bins=bins, alpha=0.6, color="#F44336",
        label="Hard sweep", density=True, edgecolor="white", lw=0.5)

ax.set_xlabel("TMRCA (generations)")
ax.set_ylabel("Density")
ax.set_title("C. Genealogical distortion by sweep type")
ax.legend(fontsize=8)

# --- Panel D: Escape probability vs genetic distance ---
ax = axes[0, 1]
N_d = 10_000
r_distances = np.logspace(-6, -2, 200)  # recombination rate to selected site

for s_val, color, ls in [
    (0.005, "#1565C0", "-"),
    (0.01, "#E65100", "--"),
    (0.02, "#2E7D32", "-."),
    (0.05, "#9C27B0", ":"),
]:
    p_escape = [escape_probability(r, s_val, N_d) for r in r_distances]
    # Convert r to physical distance (assuming r = 1e-8 per bp)
    dist_bp = r_distances / 1e-8
    ax.semilogx(dist_bp, p_escape, color=color, lw=2, ls=ls,
                label=f"s = {s_val}")

ax.axhline(0.5, color="#757575", ls=":", lw=0.8, alpha=0.5)
ax.set_xlabel("Physical distance to selected site (bp)")
ax.set_ylabel("Escape probability")
ax.set_title("D. Recombination rescue (escape from sweep)")
ax.legend(fontsize=8, loc="lower right")
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("figures/fig_mini_discoal.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_discoal.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_discoal.png")
