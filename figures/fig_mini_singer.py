"""
Figure: SINGER algorithm core components.

Four-panel overview of the key pieces of SINGER (Sampling and Inference of
Genealogies with Recombination): branch joining probabilities, PSMC
transition densities, SPR tree rearrangement, and ARG rescaling factors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from watchgen.mini_singer import (
    f_approx,
    F_bar_approx,
    joining_prob_approx,
    psmc_transition_density,
    SimpleTree,
    spr_move,
    partition_time_axis,
    count_mutations_per_window,
    compute_scaling_factors,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle("SINGER: Sampling & Inference of Genealogies with Recombination",
             fontsize=14, fontweight="bold")

# ============================================================================
# Panel A: Branch Joining Probability Density
# ============================================================================
ax = axes[0, 0]

t_vals = np.linspace(0.001, 5.0, 500)
colours_a = ["#2196F3", "#FF5722", "#4CAF50"]
n_values = [5, 10, 20]

for idx, n in enumerate(n_values):
    density = f_approx(t_vals, n)
    ax.plot(t_vals, density, lw=2.2, color=colours_a[idx], label=f"$n = {n}$")

    # Mark the mode (peak) of each density
    peak_idx = np.argmax(density)
    ax.plot(t_vals[peak_idx], density[peak_idx], "o", color=colours_a[idx],
            ms=6, zorder=5)

# Add the survival function for n=10 as a dashed reference
survival_10 = F_bar_approx(t_vals, 10)
ax.plot(t_vals, survival_10, "--", lw=1.5, color="#FF5722", alpha=0.4,
        label=r"$\bar{F}(t)$, $n=10$")

ax.set_xlabel("Time $t$ (coalescent units)")
ax.set_ylabel("Density $f(t) = \\lambda(t) \\bar{F}(t)$")
ax.set_title("A. Joining time density for varying $n$")
ax.legend(fontsize=8, loc="upper right")
ax.set_xlim(0, 4.0)
ax.set_ylim(bottom=0)

# Inset: joining probability for a branch [x, y] as a function of y
inset_a = ax.inset_axes([0.48, 0.42, 0.48, 0.48])
y_vals = np.linspace(0.05, 3.0, 80)
for idx, n in enumerate(n_values):
    probs = [joining_prob_approx(0.0, y, n) for y in y_vals]
    inset_a.plot(y_vals, probs, lw=1.5, color=colours_a[idx])
inset_a.set_xlabel("Branch upper time $y$", fontsize=7)
inset_a.set_ylabel("$P$(join $[0, y]$)", fontsize=7)
inset_a.set_title("Joining prob. vs branch depth", fontsize=7)
inset_a.tick_params(labelsize=6)

# ============================================================================
# Panel B: PSMC Transition Density q(t | s)
# ============================================================================
ax = axes[0, 1]

rho_b = 0.5
s_values = [0.3, 0.8, 1.5, 3.0]
colours_b = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
t_dense = np.linspace(0.001, 6.0, 600)

for idx, s in enumerate(s_values):
    # Compute continuous density (excluding point mass at t=s)
    q_vals = np.array([psmc_transition_density(t, s, rho_b) for t in t_dense])
    ax.plot(t_dense, q_vals, lw=2, color=colours_b[idx],
            label=f"$s = {s}$")

    # Mark the source time s with a vertical dashed line
    ax.axvline(s, color=colours_b[idx], ls=":", lw=1, alpha=0.5)

    # Show the point mass at t=s as a dot
    p_mass = np.exp(-rho_b * s)
    ax.plot(s, p_mass, "v", color=colours_b[idx], ms=7, zorder=5,
            markeredgecolor="white", markeredgewidth=0.8)

ax.set_xlabel("Target time $t$")
ax.set_ylabel("Density $q_{\\rho}(t \\mid s)$")
ax.set_title(f"B. PSMC transition density ($\\rho = {rho_b}$)")
ax.legend(fontsize=8, title="Source $s$", title_fontsize=8, loc="upper right")
ax.set_xlim(0, 6.0)
ax.set_ylim(bottom=0)

# Annotate the point mass meaning
ax.annotate("$\\blacktriangledown$ = point mass\n(no recombination)",
            xy=(0.98, 0.65), xycoords="axes fraction",
            fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#bdbdbd",
                      alpha=0.9))

# ============================================================================
# Panel C: SPR Move -- Before and After Trees
# ============================================================================
ax = axes[1, 0]
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.15, 2.0)

def draw_tree(ax, tree, leaf_positions, x_offset, color_map=None,
              highlight_edges=None, title_text=None):
    """Draw a simple tree as a stick figure (cladogram style).

    Parameters
    ----------
    ax : matplotlib Axes
    tree : SimpleTree
    leaf_positions : dict mapping leaf node -> x position
    x_offset : float
        Horizontal shift for the entire tree.
    color_map : dict or None
        Map from (child, parent) -> color for individual edges.
    highlight_edges : set of (child, parent) or None
        Edges to draw with thicker, highlighted style.
    title_text : str or None
    """
    if color_map is None:
        color_map = {}
    if highlight_edges is None:
        highlight_edges = set()

    # Compute x positions for internal nodes as midpoint of children
    positions = dict(leaf_positions)
    # Iteratively assign internal node positions
    for _ in range(10):  # iterate until stable
        for node, children_list in tree.children.items():
            if node not in positions and all(c in positions for c in children_list):
                positions[node] = np.mean([positions[c] for c in children_list])

    # Draw edges
    for child, par in tree.parent.items():
        if par is None:
            continue
        cx = positions.get(child, 0) + x_offset
        px = positions.get(par, 0) + x_offset
        ct = tree.time[child]
        pt = tree.time[par]

        edge_key = (child, par)
        col = color_map.get(edge_key, "#455A64")
        lw = 3.0 if edge_key in highlight_edges else 1.8

        # Vertical line from child up to parent height
        ax.plot([cx, cx], [ct, pt], color=col, lw=lw, solid_capstyle="round")
        # Horizontal line connecting to parent x
        ax.plot([cx, px], [pt, pt], color=col, lw=lw, solid_capstyle="round")

    # Draw nodes
    for node, t in tree.time.items():
        x = positions.get(node, 0) + x_offset
        if t == 0:
            ax.plot(x, t, "o", color="#1565C0", ms=8, zorder=5,
                    markeredgecolor="white", markeredgewidth=1)
            ax.text(x, -0.1, str(node), ha="center", va="top", fontsize=8,
                    fontweight="bold")
        else:
            ax.plot(x, t, "s", color="#E65100", ms=6, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)

    if title_text:
        center_x = np.mean([positions[n] + x_offset for n in positions])
        ax.text(center_x, 1.85, title_text, ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#E3F2FD",
                          ec="#90CAF9", alpha=0.9))

# Build the original tree
tree_before = SimpleTree(
    parent={0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6},
    time={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.5, 5: 0.9, 6: 1.5}
)

leaf_pos_before = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}

# Highlight the branch that will be cut (node 0 -> node 4)
highlight_before = {(0, 4)}
color_before = {(0, 4): "#E53935", (1, 4): "#455A64", (2, 5): "#455A64",
                (3, 5): "#455A64", (4, 6): "#E53935", (5, 6): "#455A64"}

draw_tree(ax, tree_before, leaf_pos_before, x_offset=0.0,
          color_map=color_before, highlight_edges=highlight_before,
          title_text="Before SPR")

# Perform the SPR move: cut node 0, re-attach to branch of node 3
tree_after = spr_move(tree_before, cut_node=0, new_parent=3, new_time=0.6)

leaf_pos_after = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}

# Highlight the newly created edges
new_internal = max(tree_after.time.keys())
highlight_after = {(0, new_internal), (3, new_internal)}
color_after = {}
for child, par in tree_after.parent.items():
    if par is not None:
        if (child, par) in highlight_after:
            color_after[(child, par)] = "#43A047"
        else:
            color_after[(child, par)] = "#455A64"

draw_tree(ax, tree_after, leaf_pos_after, x_offset=4.5,
          color_map=color_after, highlight_edges=highlight_after,
          title_text="After SPR")

# Draw the arrow between the two trees
ax.annotate("", xy=(3.8, 0.75), xytext=(3.2, 0.75),
            arrowprops=dict(arrowstyle="-|>", color="#F57C00", lw=2.5,
                            mutation_scale=18))
ax.text(3.5, 0.95, "SPR", ha="center", va="bottom", fontsize=9,
        fontweight="bold", color="#F57C00")
ax.text(3.5, 0.55, "cut 0\nregraft\nonto 3", ha="center", va="top",
        fontsize=7, color="#795548")

ax.set_ylabel("Time (coalescent units)")
ax.set_title("C. Sub-graph pruning & re-grafting (SPR move)")
ax.set_xticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)

# ============================================================================
# Panel D: ARG Rescaling -- Scaling Factors Across Time Windows
# ============================================================================
ax = axes[1, 1]

np.random.seed(42)

# Simulate a simple ARG with known branches
branches_d = [
    (1000, 0.0, 0.2),
    (1000, 0.0, 0.2),
    (1000, 0.0, 0.2),
    (1000, 0.0, 0.5),
    (1000, 0.0, 0.5),
    (1000, 0.2, 0.5),
    (1000, 0.2, 0.5),
    (1000, 0.5, 1.0),
    (1000, 0.5, 1.0),
    (1000, 1.0, 2.0),
]

J_d = 10
boundaries_d = partition_time_axis(branches_d, J=J_d)

# Generate mutations with a non-uniform density: more near the present
# (simulating a bottleneck or rate variation scenario)
n_mut = 80
mut_lowers = np.random.exponential(0.3, size=n_mut)
mut_uppers = mut_lowers + np.random.exponential(0.3, size=n_mut)
mut_uppers = np.clip(mut_uppers, mut_lowers + 0.01, 2.0)
mutations_d = list(zip(mut_lowers, mut_uppers))

counts_d = count_mutations_per_window(mutations_d, boundaries_d)
total_length_d = sum(span * (hi - lo) for span, lo, hi in branches_d)
theta_d = 0.001
scaling_d = compute_scaling_factors(counts_d, total_length_d, theta_d, J_d)

# Plot scaling factors as a step function
window_mids = 0.5 * (boundaries_d[:-1] + boundaries_d[1:])

# Bar chart of scaling factors
bar_widths = np.diff(boundaries_d)
bars = ax.bar(boundaries_d[:-1], scaling_d, width=bar_widths, align="edge",
              color="#2196F3", alpha=0.6, edgecolor="#1565C0", linewidth=0.8,
              label="Scaling factor $c_j$")

# Color bars by value: red for stretched, blue for compressed
for bar, c_val in zip(bars, scaling_d):
    if c_val > 1.3:
        bar.set_facecolor("#E53935")
        bar.set_alpha(0.55)
    elif c_val < 0.7:
        bar.set_facecolor("#1565C0")
        bar.set_alpha(0.55)
    else:
        bar.set_facecolor("#43A047")
        bar.set_alpha(0.55)

ax.axhline(1.0, color="#212121", ls="--", lw=1.5, alpha=0.7, label="$c = 1$ (no change)")

# Add boundary tick marks
for b in boundaries_d:
    ax.axvline(b, color="#BDBDBD", ls=":", lw=0.6, alpha=0.7)

# Legend patches
patch_over = mpatches.Patch(color="#E53935", alpha=0.55, label="$c_j > 1.3$ (time compressed)")
patch_ok = mpatches.Patch(color="#43A047", alpha=0.55, label="$c_j \\approx 1$ (well-calibrated)")
patch_under = mpatches.Patch(color="#1565C0", alpha=0.55, label="$c_j < 0.7$ (time stretched)")
ax.legend(handles=[patch_over, patch_ok, patch_under], fontsize=7,
          loc="upper right")

ax.set_xlabel("Time (coalescent units)")
ax.set_ylabel("Scaling factor $c_j$")
ax.set_title("D. ARG rescaling: mutation-clock correction")
ax.set_xlim(boundaries_d[0], boundaries_d[-1])
ax.set_ylim(0, max(scaling_d) * 1.25)

# Inset: mutation counts per window
inset_d = ax.inset_axes([0.05, 0.55, 0.38, 0.38])
inset_d.bar(np.arange(J_d), counts_d, color="#FF9800", alpha=0.7,
            edgecolor="#E65100", linewidth=0.5)
inset_d.axhline(counts_d.mean(), color="#212121", ls="--", lw=1, alpha=0.6)
inset_d.set_xlabel("Window index $j$", fontsize=7)
inset_d.set_ylabel("Mutation count", fontsize=7)
inset_d.set_title("Observed mutations per window", fontsize=7)
inset_d.tick_params(labelsize=6)

# ============================================================================
# Save
# ============================================================================
plt.tight_layout()
plt.savefig("figures/fig_mini_singer.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_singer.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_singer.png and .pdf")
