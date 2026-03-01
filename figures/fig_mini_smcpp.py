"""
Figure: SMC++ algorithm -- distinguished lineage, ODE system, and emissions.

Shows the coalescence rate h(t) for varying sample sizes, emission probabilities,
ODE lineage-count evolution, and the rate matrix structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_smcpp import (
    build_rate_matrix,
    solve_ode_piecewise,
    compute_h_values,
    emission_unphased,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("SMC++: Distinguished Lineage & ODE System",
             fontsize=14, fontweight="bold")

# ---------------------------------------------------------------------------
# Panel A: Distinguished lineage coalescence rate h(t) for different sample sizes
# ---------------------------------------------------------------------------
ax = axes[0, 0]

time_breaks = np.linspace(0, 4.0, 201)
lambdas = np.ones(200)  # constant population size lambda=1

colors_a = ["#1565C0", "#4CAF50", "#FF9800", "#E91E63"]
sample_configs = [
    (2,  "n=2 (PSMC)"),
    (5,  "n=5"),
    (10, "n=10"),
    (20, "n=20"),
]

for (n_undist, label), color in zip(sample_configs, colors_a):
    p_hist = solve_ode_piecewise(n_undist, time_breaks, lambdas)
    h_vals = compute_h_values(time_breaks, p_hist, lambdas)
    ax.plot(time_breaks, h_vals, lw=2, color=color, label=label)

ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Coalescence rate h(t)")
ax.set_title("A. Distinguished lineage rate h(t)")
ax.legend(fontsize=8)
ax.set_xlim(0, 4.0)
ax.set_ylim(0, None)

# ---------------------------------------------------------------------------
# Panel B: Emission probabilities for unphased genotypes
# ---------------------------------------------------------------------------
ax = axes[0, 1]

theta = 0.5
t_vals = np.linspace(0.001, 5.0, 300)

genotype_info = [
    (0, "Hom-ref (g=0)", "#1565C0"),
    (1, "Het (g=1)",      "#4CAF50"),
    (2, "Hom-alt (g=2)",  "#E91E63"),
]

for g, label, color in genotype_info:
    probs = [emission_unphased(g, t, theta) for t in t_vals]
    ax.plot(t_vals, probs, lw=2, color=color, label=label)

ax.set_xlabel("Coalescence time t")
ax.set_ylabel("Emission probability P(g | t)")
ax.set_title(r"B. Emission probabilities ($\theta$=" + f"{theta})")
ax.legend(fontsize=8)
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 1.05)

# ---------------------------------------------------------------------------
# Panel C: ODE solution -- lineage count probabilities over time
# ---------------------------------------------------------------------------
ax = axes[1, 0]

n_undist_c = 9
time_breaks_c = np.linspace(0, 3.0, 151)
lambdas_c = np.ones(150)

p_hist_c = solve_ode_piecewise(n_undist_c, time_breaks_c, lambdas_c)

# Plot selected lineage-count probabilities: j = 1, 3, 5, 7, 9
colors_c = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#1565C0"]
for j, color in zip([1, 3, 5, 7, 9], colors_c):
    ax.plot(time_breaks_c, p_hist_c[:, j - 1], lw=2, color=color,
            label=f"j={j}")

ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Probability P(J(t)=j)")
ax.set_title(f"C. Lineage count ODE (n-1={n_undist_c})")
ax.legend(fontsize=8, ncol=2)
ax.set_xlim(0, 3.0)
ax.set_ylim(0, 1.05)

# ---------------------------------------------------------------------------
# Panel D: Rate matrix Q heatmap
# ---------------------------------------------------------------------------
ax = axes[1, 1]

n_undist_d = 8
Q = build_rate_matrix(n_undist_d)

# Use a diverging colormap so negative diagonal stands out
vmax = np.max(np.abs(Q))
im = ax.imshow(Q, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal",
               origin="upper")
ax.set_xlabel("State index (j-1)")
ax.set_ylabel("State index (j-1)")
ax.set_title(f"D. Rate matrix Q (n-1={n_undist_d})")
ax.set_xticks(range(n_undist_d))
ax.set_xticklabels(range(1, n_undist_d + 1))
ax.set_yticks(range(n_undist_d))
ax.set_yticklabels(range(1, n_undist_d + 1))
plt.colorbar(im, ax=ax, label="Rate", shrink=0.85)

# Annotate nonzero entries
for i in range(n_undist_d):
    for j_idx in range(n_undist_d):
        val = Q[i, j_idx]
        if abs(val) > 1e-10:
            txt_color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j_idx, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=7, color=txt_color)

plt.tight_layout()
plt.savefig("figures/fig_mini_smcpp.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_smcpp.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_smcpp.png and figures/fig_mini_smcpp.pdf")
