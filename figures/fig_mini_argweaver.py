"""
Figure: ARGweaver algorithm core components.

Four-panel overview of the Discrete SMC machinery underpinning ARGweaver:
time discretization, transition probabilities, re-coalescence distributions,
and MCMC tree sampling behaviour.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from watchgen.mini_argweaver import (
    get_time_points,
    get_time_steps,
    get_coal_times,
    build_simple_transition_matrix,
    recoal_distribution,
    sample_tree,
    harmonic,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("ARGweaver: Discrete SMC Components", fontsize=14, fontweight="bold")

# ============================================================================
# Panel A: Time Discretization
# ============================================================================
ax = axes[0, 0]

ntimes_a = 20
maxtime_a = 160_000
times = get_time_points(ntimes=ntimes_a, maxtime=maxtime_a, delta=0.01)
coal_times_list = get_coal_times(times)
time_steps = get_time_steps(times)

# Plot boundary time points as tall vertical lines
for i, t in enumerate(times):
    ax.axhline(t, color="#2196F3", lw=1.2, alpha=0.7)
    ax.plot(0.15, t, "o", color="#2196F3", ms=5, zorder=5)

# Plot midpoints (coal times) as dashed lines
for i in range(ntimes_a - 1):
    mid = coal_times_list[2 * i + 1]
    ax.axhline(mid, color="#FF5722", lw=0.8, ls="--", alpha=0.5)
    ax.plot(0.85, mid, "D", color="#FF5722", ms=4, zorder=5)

# Annotate time steps on the right side (first 8 intervals for readability)
for i in range(min(8, len(time_steps))):
    y_lo = times[i]
    y_hi = times[i + 1]
    y_mid = (y_lo + y_hi) / 2
    ax.annotate(
        f"{time_steps[i]:.0f}",
        xy=(1.05, y_mid), fontsize=6, color="#616161",
        ha="left", va="center",
    )

# Use a log-ish y-scale: plot only the first 12 intervals for clarity
y_upper = times[12]
ax.set_ylim(-y_upper * 0.02, y_upper * 1.05)
ax.set_xlim(-0.1, 1.3)
ax.set_xticks([0.15, 0.85])
ax.set_xticklabels(["Boundaries\n$t_i$", "Midpoints\n$\\tilde{t}_i$"], fontsize=8)
ax.set_ylabel("Time (generations)")
ax.set_title("A. Time discretization (log-spaced grid)")

# Add a small inset showing log-spacing growth
inset = ax.inset_axes([0.55, 0.55, 0.42, 0.4])
indices = np.arange(len(time_steps))
inset.bar(indices, time_steps, color="#2196F3", alpha=0.7, width=0.8)
inset.set_xlabel("Interval index $i$", fontsize=7)
inset.set_ylabel("$\\Delta t_i$", fontsize=7)
inset.set_title("Step sizes", fontsize=7)
inset.tick_params(labelsize=6)

# ============================================================================
# Panel B: Transition Matrix Heatmap
# ============================================================================
ax = axes[0, 1]

ntimes_b = 20
times_b = get_time_points(ntimes=ntimes_b, maxtime=maxtime_a, delta=0.01)
Ne_b = 10_000.0
nbranches_b = [4] * (ntimes_b - 1)
ncoals_b = [1] * (ntimes_b - 1)
popsizes_b = [Ne_b] * (ntimes_b - 1)
rho_b = 1e-8
treelen_b = sum(nbranches_b[i] * get_time_steps(times_b)[i]
                for i in range(ntimes_b - 1))

T = build_simple_transition_matrix(
    ntimes_b - 1, nbranches_b, ncoals_b, popsizes_b, rho_b, treelen_b, times_b,
)

# Mask zeros for log-scale display
T_plot = np.where(T > 0, T, np.nan)
vmin = np.nanmin(T_plot[T_plot > 0])
vmax = np.nanmax(T_plot)

im = ax.imshow(
    T_plot,
    cmap="viridis",
    norm=LogNorm(vmin=max(vmin, 1e-12), vmax=vmax),
    aspect="auto",
    origin="lower",
    interpolation="nearest",
)
cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03)
cbar.set_label("Transition probability", fontsize=8)
cbar.ax.tick_params(labelsize=7)
ax.set_xlabel("Destination time index $j'$")
ax.set_ylabel("Source time index $j$")
ax.set_title("B. Transition matrix $T(j \\to j')$")

# ============================================================================
# Panel C: Re-coalescence Distribution
# ============================================================================
ax = axes[1, 0]

ntimes_c = 30
times_c = get_time_points(ntimes=ntimes_c, maxtime=maxtime_a, delta=0.01)
Ne_c = 10_000.0

colours_c = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
branch_counts = [2, 4, 8, 16, 32]

for idx, nb in enumerate(branch_counts):
    pmf = recoal_distribution(nb, Ne_c, times_c)
    time_indices = np.arange(len(pmf))
    ax.plot(
        time_indices, pmf,
        "o-", ms=3, lw=1.5, color=colours_c[idx],
        label=f"$k = {nb}$",
    )

ax.set_xlabel("Time index $j$")
ax.set_ylabel("Re-coalescence probability")
ax.set_title("C. Re-coalescence PMF (varying branch count $k$)")
ax.legend(fontsize=8, title="Branches", title_fontsize=8)
ax.set_xlim(-0.5, ntimes_c - 1)

# ============================================================================
# Panel D: MCMC Tree Sampling -- Distribution of Root Times
# ============================================================================
ax = axes[1, 1]

random.seed(42)
np.random.seed(42)

ntimes_d = 20
Ne_d = 10_000.0
times_d = get_time_points(ntimes=ntimes_d, maxtime=maxtime_a, delta=0.01)
popsizes_d = [Ne_d] * (ntimes_d - 1)

k_vals = [5, 10, 20]
colours_d = ["#2196F3", "#FF5722", "#4CAF50"]
n_reps = 2000

for idx, k in enumerate(k_vals):
    root_times = []
    for _ in range(n_reps):
        coal_events = sample_tree(k, popsizes_d, times_d)
        if coal_events:
            root_times.append(max(coal_events))
    root_times = np.array(root_times)

    ax.hist(
        root_times, bins=40, density=True, alpha=0.45,
        color=colours_d[idx], label=f"$k={k}$ (sampled)",
    )

    # Theoretical expected TMRCA = 2*Ne*(1 - 1/k) for standard coalescent
    E_tmrca = 2 * Ne_d * (1 - 1.0 / k)
    ax.axvline(E_tmrca, color=colours_d[idx], ls="--", lw=2,
               label=f"$k={k}$: $E[T_{{MRCA}}]={E_tmrca:.0f}$")

ax.set_xlabel("Tree height (TMRCA, generations)")
ax.set_ylabel("Density")
ax.set_title("D. MCMC tree heights vs coalescent expectation")
ax.legend(fontsize=7, ncol=2, loc="upper right")

# ============================================================================
# Save
# ============================================================================
plt.tight_layout()
plt.savefig("figures/fig_mini_argweaver.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_argweaver.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_argweaver.png and .pdf")
