"""Source-matched diagnostics for the bounded ARGweaver teaching kernels."""

import random

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_argweaver import (
    get_time_points,
    recoal_distribution,
    sample_tree,
)


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

Ne = 10_000.0
mu = 1.25e-8
times = get_time_points(ntimes=20, maxtime=200_000)

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "ARGweaver: Source-Matched Teaching Checks",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

# Panel A: exact Jukes-Cantor probabilities versus low-rate approximations.
ax = axes[0, 0]
branch_times = np.geomspace(1, 1e8, 400)
decay = np.exp(-4 * mu * branch_times / 3)
exact_same = (1 + 3 * decay) / 4
exact_change = (1 - decay) / 4
approx_same = np.exp(-mu * branch_times)
approx_change = (1 - np.exp(-mu * branch_times)) / 3
ax.semilogx(branch_times, exact_same, color="#2166AC", label="same: exact JC")
ax.semilogx(branch_times, approx_same, "--", color="#2166AC",
            label="same: low-rate approximation")
ax.semilogx(branch_times, exact_change, color="#B2182B",
            label="specified change: exact JC")
ax.semilogx(branch_times, approx_change, "--", color="#B2182B",
            label="specified change: approximation")
ax.set_xlabel("Branch length (generations)")
ax.set_ylabel("Transition probability")
ax.set_title("A. Exact Jukes--Cantor branch kernel")
ax.legend(fontsize=7)

# Panel B: logarithmic time discretization.
ax = axes[0, 1]
steps = np.diff(times)
ax.barh(
    range(len(steps)),
    steps,
    left=times[:-1],
    height=0.8,
    color=plt.cm.viridis(np.linspace(0, 1, len(steps))),
    edgecolor="white",
    linewidth=0.3,
)
ax.set_xlabel("Time (generations)")
ax.set_ylabel("Interval index")
ax.set_title("B. Logarithmic time grid")
ax.set_xscale("log")
ax.set_xlim(10, 300_000)

# Panel C: re-coalescence mass under the diploid 2Ne hazard.
ax = axes[1, 0]
colors = ["#2166AC", "#B2182B", "#1B7837", "#E08214"]
for n_lineages, color in zip([2, 4, 6, 8], colors):
    probs = np.asarray(recoal_distribution(n_lineages, Ne, times))
    ax.plot(
        range(len(probs)),
        probs,
        "o-",
        color=color,
        ms=3,
        lw=1.5,
        label=f"$k={n_lineages}$",
    )
ax.set_xlabel("Time interval")
ax.set_ylabel("Probability mass")
ax.set_title("C. Re-coalescence under $k/(2N_e)$")
ax.legend(fontsize=8)

# Panel D: direct coalescent-prior TMRCA calibration, not an MCMC trace.
ax = axes[1, 1]
random.seed(2024)
k = 8
popsizes = [Ne] * (len(times) - 1)
tmrcas = np.asarray([
    sample_tree(k, popsizes, times)[-1]
    for _ in range(4000)
])
expected = 4 * Ne * (1 - 1 / k)
ax.hist(tmrcas, bins=45, density=True, color="#2166AC", alpha=0.7,
        edgecolor="white", linewidth=0.4)
ax.axvline(expected, color="#B2182B", lw=2, ls="--",
           label=f"analytic mean = {expected:,.0f}")
ax.axvline(tmrcas.mean(), color="#1B7837", lw=2,
           label=f"simulated mean = {tmrcas.mean():,.0f}")
ax.set_xlabel("TMRCA (generations)")
ax.set_ylabel("Density")
ax.set_title("D. Coalescent-prior TMRCA calibration")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_argweaver.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_argweaver.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_argweaver.png and .pdf")
