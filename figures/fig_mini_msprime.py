"""
Figure: msprime coalescent simulator -- four-panel overview.

Panel A: Coalescent times -- histogram of simulated T_MRCA for n=2
         vs theoretical Exp(1) density, plus E[T_MRCA] for various n.
Panel B: Site frequency spectrum -- simulated SFS from many coalescent
         replicates vs the classic neutral expectation theta/i.
Panel C: Fenwick tree -- visualisation of cumulative-sum structure and
         weighted random search in the binary indexed tree.
Panel D: Demographic effects -- coalescence times under constant,
         bottleneck, and growth scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from watchgen.mini_msprime import (
    simulate_coalescence_time_continuous,
    simulate_coalescent,
    expected_tmrca,
    expected_total_branch_length,
    expected_sfs,
    FenwickTree,
    Population,
    simulate_coalescent_tmrca,
)

# -- Style ---------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "font.family": "sans-serif",
})

C_BLUE = "#2166AC"
C_RED = "#B2182B"
C_GREEN = "#1B7837"
C_ORANGE = "#E08214"
C_PURPLE = "#7B3294"
C_GREY = "#636363"
C_TEAL = "#01665E"

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle(
    "msprime: Coalescent Simulation with Recombination",
    fontsize=14, fontweight="bold", y=0.98,
)

# =====================================================================
# Panel A -- Coalescent times: simulation vs theory
# =====================================================================
ax = axes[0, 0]

# Simulate many coalescence times for n=2 (Exp(1) in coalescent units)
n_reps = 50000
coal_times = simulate_coalescence_time_continuous(n_replicates=n_reps)

# Histogram of simulated times
bins = np.linspace(0, 6, 60)
ax.hist(coal_times, bins=bins, density=True, alpha=0.5, color=C_BLUE,
        edgecolor="white", linewidth=0.4, label="Simulated $T_2$ (n=2)")

# Theoretical Exp(1) density
t_theory = np.linspace(0.001, 6, 300)
pdf_theory = np.exp(-t_theory)
ax.plot(t_theory, pdf_theory, color=C_RED, lw=2.5,
        label=r"Theory: $f(t) = e^{-t}$")

# Inset: E[T_MRCA] for different n
ax_inset = ax.inset_axes([0.48, 0.38, 0.48, 0.55])
n_values = np.arange(2, 52)
e_tmrca_vals = np.array([expected_tmrca(n) for n in n_values])
ax_inset.plot(n_values, e_tmrca_vals, color=C_GREEN, lw=2, marker="o",
              markersize=2.5, zorder=3)
ax_inset.axhline(2.0, color=C_GREY, ls="--", lw=1, alpha=0.6)
ax_inset.text(35, 2.03, r"$\lim_{n\to\infty} = 2$", fontsize=7,
              color=C_GREY, va="bottom")
ax_inset.set_xlabel("Sample size $n$", fontsize=7)
ax_inset.set_ylabel(r"$\mathbb{E}[T_{\mathrm{MRCA}}]$", fontsize=7)
ax_inset.tick_params(labelsize=6)
ax_inset.set_title(r"$\mathbb{E}[T_{\mathrm{MRCA}}] = 2(1 - 1/n)$",
                    fontsize=7, pad=3)
ax_inset.set_ylim(0, 2.3)
ax_inset.grid(True, alpha=0.3, linewidth=0.5)

ax.set_xlabel("Coalescence time $t$ (coalescent units)")
ax.set_ylabel("Density")
ax.set_title("A.  Coalescence time distribution ($n = 2$)")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, 6)
ax.set_ylim(0, 1.15)

# =====================================================================
# Panel B -- Site Frequency Spectrum
# =====================================================================
ax = axes[0, 1]

n_samples = 20
theta = 50.0
n_coal_reps = 2000

# Simulate many coalescent trees and accumulate empirical SFS
empirical_sfs_total = np.zeros(n_samples - 1)

for rep in range(n_coal_reps):
    results = simulate_coalescent(n_samples, n_replicates=1)
    times_list, pairs_list = results[0]

    # Compute total branch length at each level for this tree.
    # Each coalescent interval k -> k-1 has k lineages and duration dt.
    # The expected number of mutations at frequency i/n is proportional
    # to branch lengths. We use the coalescent structure to build the SFS.
    # The branch above each coalescent event carries a certain number
    # of descendants. For the standard coalescent, we directly compute
    # the SFS from waiting times.
    #
    # In a coalescent tree with n samples, the time with k lineages
    # contributes k * dt to total branch length. The SFS contribution
    # at frequency i is 2*dt for the branch leading to exactly i
    # descendants out of n. For the standard (no recombination)
    # coalescent, we can compute the expected SFS analytically and
    # compare with the simulation below.
    pass

# Use the analytical expected SFS vs the classic 1/k pattern
k_vals = np.arange(1, n_samples)
exp_sfs_vals = expected_sfs(n_samples, theta)

# Bar plot of expected SFS
bars = ax.bar(k_vals - 0.15, exp_sfs_vals, width=0.35, color=C_BLUE,
              alpha=0.85, edgecolor="white", linewidth=0.4,
              label=r"$\mathbb{E}[\xi_i] = \theta / i$")

# Overlay the theoretical 1/k shape (scaled by theta)
k_fine = np.linspace(1, n_samples - 1, 200)
ax.plot(k_fine, theta / k_fine, color=C_RED, lw=2.5, ls="-",
        label=r"$\theta / i$ (continuous)", zorder=5)

# Simulate a few coalescent replicates and compute empirical SFS
# from total branch length contributions
n_sim_sfs = 500
simulated_sfs_runs = []
for _ in range(n_sim_sfs):
    results = simulate_coalescent(n_samples, n_replicates=1)
    coal_times, coal_pairs = results[0]

    # Build a simple tree to count descendants per branch
    # Each node starts as a leaf; track descendant sets
    desc = {i: {i} for i in range(n_samples)}
    next_node = n_samples
    branch_sfs = np.zeros(n_samples - 1)

    prev_time = 0.0
    k = n_samples
    for t_event, (a, b) in zip(coal_times, coal_pairs):
        dt = t_event - prev_time
        # During this interval, each lineage's branch has dt duration
        # and carries some number of descendants. The number of mutations
        # on each branch is proportional to dt. Each mutation on a branch
        # with d descendants produces a variant at frequency d/n.
        for node_id in list(desc.keys()):
            d = len(desc[node_id])
            if 1 <= d <= n_samples - 1:
                # Expected mutations: theta/2 * dt (per branch)
                branch_sfs[d - 1] += dt

        # Merge a and b into a new node
        new_desc = desc[a] | desc[b]
        del desc[a]
        del desc[b]
        desc[next_node] = new_desc
        prev_time = t_event
        next_node += 1
        k -= 1

    # Scale: each branch of duration dt contributes theta/2 * dt mutations
    # at the corresponding frequency
    branch_sfs *= theta / 2.0
    simulated_sfs_runs.append(branch_sfs)

sim_sfs_mean = np.mean(simulated_sfs_runs, axis=0)
sim_sfs_std = np.std(simulated_sfs_runs, axis=0)

ax.bar(k_vals + 0.15, sim_sfs_mean, width=0.35, color=C_GREEN, alpha=0.7,
       edgecolor="white", linewidth=0.4,
       label=f"Simulated mean ({n_sim_sfs} trees)")
ax.errorbar(k_vals + 0.15, sim_sfs_mean, yerr=sim_sfs_std / np.sqrt(n_sim_sfs),
            fmt="none", ecolor=C_GREEN, capsize=2, lw=1, alpha=0.8)

ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel(r"$\mathbb{E}[\xi_i]$ (expected sites)")
ax.set_title("B.  Site frequency spectrum (neutral coalescent)")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_xlim(0.3, n_samples - 0.5)

# =====================================================================
# Panel C -- Fenwick tree structure and operations
# =====================================================================
ax = axes[1, 0]

# Build a Fenwick tree with known values and visualise it
ft_size = 8
ft = FenwickTree(ft_size)
values = [3, 1, 4, 1, 5, 9, 2, 6]
for i, v in enumerate(values):
    ft.set_value(i + 1, v)

# Plot the original values as bars
x_pos = np.arange(1, ft_size + 1)
bar_colors = [C_BLUE] * ft_size
ax.bar(x_pos, values, width=0.4, color=bar_colors, alpha=0.7,
       edgecolor="white", linewidth=0.5, label="Values $a[i]$", zorder=3)

# Plot cumulative prefix sums as a step line
prefix_sums = [ft.get_cumulative_sum(i + 1) for i in range(ft_size)]
ax.step(x_pos - 0.0, prefix_sums, where="mid", color=C_RED, lw=2.5,
        label="Prefix sum $\\sum_{j=1}^{i} a[j]$", zorder=4)
ax.scatter(x_pos, prefix_sums, color=C_RED, s=30, zorder=5, edgecolors="white",
           linewidths=0.8)

# Show the Fenwick tree internal array structure
# The tree[] array stores partial sums covering specific ranges
# tree[i] covers the range (i - lowbit(i), i], where lowbit(i) = i & (-i)
fenwick_ranges = []
for i in range(1, ft_size + 1):
    lowbit = i & (-i)
    lo = i - lowbit + 1
    hi = i
    fenwick_ranges.append((lo, hi, ft.tree[i]))

# Draw the range each Fenwick node covers as coloured brackets below
y_bracket = -2.5
for idx, (lo, hi, val) in enumerate(fenwick_ranges):
    span = hi - lo
    color = plt.cm.Set2(idx / ft_size)
    ax.plot([lo - 0.15, hi + 0.15], [y_bracket - idx * 0.35] * 2,
            color=color, lw=3, solid_capstyle="round", alpha=0.8)
    ax.text((lo + hi) / 2, y_bracket - idx * 0.35 - 0.3,
            f"tree[{hi}]={val}", fontsize=5.5, ha="center", va="top",
            color=color, fontweight="bold")

# Demonstrate find() operation
target = 15
found_idx = ft.find(target)
ax.axhline(target, color=C_ORANGE, ls="--", lw=1.2, alpha=0.7)
ax.annotate(
    f"find({target}) $\\rightarrow$ index {found_idx}",
    xy=(found_idx, prefix_sums[found_idx - 1]),
    xytext=(found_idx + 1.5, target + 2),
    fontsize=8, color=C_ORANGE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_ORANGE, alpha=0.9),
)

ax.set_xlabel("Index $i$")
ax.set_ylabel("Value / Cumulative sum")
ax.set_title("C.  Fenwick tree: $O(\\log n)$ prefix sums and search")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(0.2, ft_size + 1.5)
ax.set_ylim(y_bracket - ft_size * 0.35 - 1, max(prefix_sums) + 5)
ax.set_xticks(x_pos)

# Add complexity annotations
ax.text(ft_size + 0.3, max(prefix_sums) * 0.65,
        "Operations:\n"
        "  update: $O(\\log n)$\n"
        "  prefix sum: $O(\\log n)$\n"
        "  find: $O(\\log n)$",
        fontsize=7, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec=C_GREY, alpha=0.9))

# =====================================================================
# Panel D -- Demographic effects on coalescence times
# =====================================================================
ax = axes[1, 1]

n_demo = 10
n_reps_demo = 3000

# Scenario 1: Constant population (N=1000)
N_const = 1000
tmrca_const = simulate_coalescent_tmrca(n_demo, N_const, n_reps=n_reps_demo)

# Scenario 2: Bottleneck -- use a smaller N to represent a population
# that went through a severe reduction
N_bottle = 200
tmrca_bottle = simulate_coalescent_tmrca(n_demo, N_bottle, n_reps=n_reps_demo)

# Scenario 3: Large ancestral population
N_large = 5000
tmrca_large = simulate_coalescent_tmrca(n_demo, N_large, n_reps=n_reps_demo)

# Violin plots for the three scenarios
positions = [1, 2, 3]
data = [tmrca_const, tmrca_bottle, tmrca_large]
colors = [C_BLUE, C_RED, C_GREEN]
labels = [
    f"Constant\n$N_e = {N_const}$",
    f"Bottleneck\n$N_e = {N_bottle}$",
    f"Large\n$N_e = {N_large}$",
]

parts = ax.violinplot(data, positions=positions, showmeans=True,
                      showmedians=True, showextrema=False)

for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.5)
    pc.set_edgecolor(colors[i])
    pc.set_linewidth(1.2)

parts["cmeans"].set_color("black")
parts["cmeans"].set_linewidth(1.5)
parts["cmedians"].set_color(C_GREY)
parts["cmedians"].set_linewidth(1)
parts["cmedians"].set_linestyle("--")

# Add expected T_MRCA markers
e_tmrca_n = expected_tmrca(n_demo)  # in coalescent units
for i, (N_val, color) in enumerate(zip([N_const, N_bottle, N_large], colors)):
    expected_val = e_tmrca_n * N_val  # convert to generations
    ax.plot(positions[i], expected_val, marker="*", markersize=14,
            color=color, markeredgecolor="black", markeredgewidth=0.8,
            zorder=5)
    ax.annotate(
        f"$E[T_{{\\mathrm{{MRCA}}}}] = {expected_val:.0f}$",
        xy=(positions[i], expected_val),
        xytext=(positions[i] + 0.35, expected_val * 1.05),
        fontsize=7, color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )

ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("$T_{\\mathrm{MRCA}}$ (generations)")
ax.set_title(f"D.  Demographic effects on coalescence ($n = {n_demo}$)")

# Add a note about the relationship T_MRCA ~ N_e
ax.text(0.97, 0.97,
        r"$\bigstar$ = $\mathbb{E}[T_{\mathrm{MRCA}}]$"
        f"\n" + r"$= 2 N_e (1 - 1/n)$",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, alpha=0.9))

ax.set_ylim(bottom=0)

# -- Save ----------------------------------------------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_mini_msprime.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_msprime.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_msprime.png and figures/fig_mini_msprime.pdf")
