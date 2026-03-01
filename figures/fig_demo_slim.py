"""
Demo: SLiM forward-time simulation with selection.

Runs the mini SLiM forward simulator under neutral, deleterious DFE,
and selective sweep scenarios. Shows allele dynamics and fitness evolution.
"""

import numpy as np
import matplotlib.pyplot as plt

from watchgen.mini_slim import (
    simulate,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulations ─────────────────────────────────────────────────
N = 100
L = 50_000
mu = 5e-6
r = 1e-6
T = 300

pop_neutral, stats_neutral = simulate(N, L, mu, r, T, dfe="neutral", track_every=5)
pop_del, stats_del = simulate(N, L, mu, r, T, dfe="gamma",
                              dfe_params={"shape": 0.3, "scale": 0.05},
                              track_every=5)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: SLiM Forward Simulation ($N$={N}, $L$={L/1e3:.0f} kb, {T} gen)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Mean fitness over time
ax = axes[0, 0]
ax.plot(stats_neutral["generation"], stats_neutral["mean_fitness"],
        color="#1B7837", lw=2, label="Neutral")
ax.plot(stats_del["generation"], stats_del["mean_fitness"],
        color="#B2182B", lw=2, label="Deleterious DFE")
ax.set_xlabel("Generation")
ax.set_ylabel("Mean population fitness")
ax.set_title("A. Fitness evolution")
ax.axhline(1.0, color="k", ls="--", lw=0.8, alpha=0.4)
ax.legend(fontsize=8)

# Panel B: Segregating sites
ax = axes[0, 1]
ax.plot(stats_neutral["generation"], stats_neutral["num_segregating"],
        color="#1B7837", lw=2, label="Neutral")
ax.plot(stats_del["generation"], stats_del["num_segregating"],
        color="#B2182B", lw=2, label="Deleterious DFE")
ax.set_xlabel("Generation")
ax.set_ylabel("Number of segregating sites")
ax.set_title("B. Genetic diversity")
ax.legend(fontsize=8)

# Panel C: Mutations per individual
ax = axes[1, 0]
ax.plot(stats_neutral["generation"], stats_neutral["mean_mutations_per_individual"],
        color="#1B7837", lw=2, label="Neutral")
ax.plot(stats_del["generation"], stats_del["mean_mutations_per_individual"],
        color="#B2182B", lw=2, label="Deleterious DFE")
ax.set_xlabel("Generation")
ax.set_ylabel("Mean mutations per individual")
ax.set_title("C. Mutation accumulation")
ax.legend(fontsize=8)

# Panel D: Fixation probability vs selection coefficient (Kimura formula)
ax = axes[1, 1]
s_range = np.linspace(-0.05, 0.1, 80)
def kimura_fixation(s, N):
    if abs(s) < 1e-10:
        return 1.0 / (2 * N)
    x = 2 * s
    return (1 - np.exp(-x)) / (1 - np.exp(-4 * N * s))
p_fix = [kimura_fixation(s_val, N) for s_val in s_range]

ax.plot(s_range, p_fix, color="#2166AC", lw=2.5)
ax.axhline(1 / (2 * N), color="#636363", ls="--", lw=1,
           label=f"Neutral: $1/(2N)$ = {1/(2*N):.4f}")
ax.axvline(0, color="k", ls=":", lw=0.8, alpha=0.4)
ax.set_xlabel("Selection coefficient $s$")
ax.set_ylabel("Fixation probability")
ax.set_title(f"D. Kimura fixation probability ($N$={N})")
ax.legend(fontsize=8)
ax.set_ylim(0, max(p_fix) * 1.1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_slim.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_slim.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_slim.png")
