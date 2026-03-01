"""
Figure: Mini-Relate algorithm -- four key gears.

Panel A: Asymmetric distance matrix heatmap at a focal SNP.
Panel B: Tree topology built from the distance matrix.
Panel C: MCMC posterior trace and density for a coalescence time.
Panel D: Piecewise-constant N_e(t) estimated via the M-step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec

from watchgen.mini_relate import (
    compute_distance_matrix,
    build_tree,
    to_newick,
    TreeNode,
    map_mutations,
    mcmc_branch_lengths,
    posterior_summary,
    make_epochs,
    m_step,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ============================================================================
# Shared parameters
# ============================================================================
np.random.seed(42)
N_HAPS = 8
L_SITES = 20
MU_PAINT = 0.01
RECOMB_RATE = 1e-4
FOCAL_SNP = 10

# Generate a small haplotype matrix
haplotypes = np.random.binomial(1, 0.3, size=(N_HAPS, L_SITES))
positions = np.arange(L_SITES, dtype=float) * 1000

# ============================================================================
# Panel A data: asymmetric distance matrix
# ============================================================================
D = compute_distance_matrix(haplotypes, positions, RECOMB_RATE, MU_PAINT,
                            focal_snp=FOCAL_SNP)

# ============================================================================
# Panel B data: tree building
# ============================================================================
root, merge_order = build_tree(D, N_HAPS)
newick = to_newick(root) + ";"

# ============================================================================
# Panel C data: MCMC branch length estimation
# ============================================================================
# Build a hand-crafted 4-leaf tree for clean MCMC demonstration
leaf0 = TreeNode(0)
leaf1 = TreeNode(1)
leaf2 = TreeNode(2)
leaf3 = TreeNode(3)
node4 = TreeNode(4, left=leaf0, right=leaf1, is_leaf=False)
node4.leaf_ids = {0, 1}
node5 = TreeNode(5, left=node4, right=leaf2, is_leaf=False)
node5.leaf_ids = {0, 1, 2}
root_mcmc = TreeNode(6, left=node5, right=leaf3, is_leaf=False)
root_mcmc.leaf_ids = {0, 1, 2, 3}

haps_mcmc = np.array([
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

branch_muts, _ = map_mutations(root_mcmc, haps_mcmc, list(range(4)))
samples_mcmc, acc_rate = mcmc_branch_lengths(
    root_mcmc, branch_muts, mu=1.25e-8, span=1e4, N_e=10_000,
    n_samples=2000, burn_in=500, sigma=100.0, seed=42,
)

# ============================================================================
# Panel D data: population size estimation via M-step
# ============================================================================
np.random.seed(7)
TRUE_NE = 10_000
N_TREES_D = 200
N_LEAVES_D = 10

coal_times_all = []
for _ in range(N_TREES_D):
    times = []
    prev_t = 0.0
    for k in range(N_LEAVES_D, 1, -1):
        rate = k * (k - 1) / (2.0 * TRUE_NE)
        dt = np.random.exponential(1.0 / rate)
        prev_t += dt
        times.append(prev_t)
    coal_times_all.append(times)

boundaries_d = make_epochs(50_000, n_epochs=10)
spans_d = np.full(N_TREES_D, 1e4)
N_e_est = m_step(coal_times_all, [N_LEAVES_D] * N_TREES_D,
                 boundaries_d, spans_d)

# ============================================================================
# Figure
# ============================================================================
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.32)

# ---------------------------------------------------------------------------
# Panel A: Asymmetric distance matrix
# ---------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
im = ax_a.imshow(D, cmap="YlOrRd", origin="upper", aspect="equal")
ax_a.set_xticks(range(N_HAPS))
ax_a.set_yticks(range(N_HAPS))
ax_a.set_xlabel("Target haplotype $j$")
ax_a.set_ylabel("Source haplotype $i$")
ax_a.set_title("A.  Asymmetric distance matrix $D_{ij}$", fontweight="bold",
               loc="left")
cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
cbar.set_label("$-\\log\\, p_{ij}$")

# Annotate asymmetry: add small text in each cell
for i in range(N_HAPS):
    for j in range(N_HAPS):
        val = D[i, j]
        color = "white" if val > (D.max() - D.min()) * 0.65 + D.min() else "black"
        ax_a.text(j, i, f"{val:.1f}", ha="center", va="center",
                  fontsize=6.5, color=color)

# ---------------------------------------------------------------------------
# Panel B: Tree topology (manual drawing)
# ---------------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_title("B.  Inferred local tree topology", fontweight="bold",
               loc="left")


def layout_tree(node, x_counter=None, depth=0):
    """Assign (x, y) coordinates to each node via in-order traversal."""
    if x_counter is None:
        x_counter = [0]
    coords = {}
    if node.is_leaf:
        coords[node.id] = (x_counter[0], 0)
        x_counter[0] += 1
    else:
        left_coords = layout_tree(node.left, x_counter, depth + 1)
        right_coords = layout_tree(node.right, x_counter, depth + 1)
        coords.update(left_coords)
        coords.update(right_coords)
        lx, _ = coords[node.left.id]
        rx, _ = coords[node.right.id]
        y = depth + 1
        coords[node.id] = ((lx + rx) / 2.0, y)
    return coords


# Recompute tree depth to set nice y-positions proportional to merge order
def layout_tree_by_merge(root, merge_order, n_leaves):
    """Position nodes: leaves at y=0, internal nodes at y = merge step + 1."""
    coords = {}
    x_counter = [0]

    def assign_x(node):
        if node.is_leaf:
            coords[node.id] = [x_counter[0], 0]
            x_counter[0] += 1
        else:
            assign_x(node.left)
            assign_x(node.right)
            lx = coords[node.left.id][0]
            rx = coords[node.right.id][0]
            coords[node.id] = [(lx + rx) / 2.0, 0]

    assign_x(root)

    # Assign y by merge order (step index)
    for step_idx, (c1, c2, parent) in enumerate(merge_order):
        coords[parent][1] = step_idx + 1

    return {k: tuple(v) for k, v in coords.items()}


coords = layout_tree_by_merge(root, merge_order, N_HAPS)


def draw_tree(ax, node, coords):
    """Draw tree edges recursively with bracket style."""
    if not node.is_leaf:
        px, py = coords[node.id]
        for child in [node.left, node.right]:
            cx, cy = coords[child.id]
            # Vertical line from child up to parent's y
            ax.plot([cx, cx], [cy, py], color="#37474F", lw=1.8,
                    solid_capstyle="round")
            # Horizontal line across to parent's x
            ax.plot([cx, px], [py, py], color="#37474F", lw=1.8,
                    solid_capstyle="round")
            draw_tree(ax, child, coords)


draw_tree(ax_b, root, coords)

# Draw leaf labels
for lid in range(N_HAPS):
    x, y = coords[lid]
    ax_b.text(x, y - 0.35, str(lid), ha="center", va="top", fontsize=9,
              fontweight="bold", color="#1565C0")
    ax_b.plot(x, y, "o", color="#1565C0", ms=5, zorder=5)

# Draw internal nodes
for nid in coords:
    if nid >= N_HAPS:
        x, y = coords[nid]
        ax_b.plot(x, y, "s", color="#C62828", ms=4, zorder=5)

ax_b.set_ylabel("Merge step (coalescence order)")
ax_b.set_xlabel("Leaf index")
ax_b.set_xlim(-0.5, N_HAPS - 0.5)
max_y = max(v[1] for v in coords.values())
ax_b.set_ylim(-0.8, max_y + 0.8)
ax_b.set_xticks(range(N_HAPS))
newick_short = newick if len(newick) <= 45 else newick[:42] + "..."
ax_b.text(0.02, 0.97, f"Newick: {newick_short}", transform=ax_b.transAxes,
          fontsize=7, va="top", fontstyle="italic", color="#555555",
          bbox=dict(boxstyle="round,pad=0.2", fc="#EEEEEE", ec="none"))

# ---------------------------------------------------------------------------
# Panel C: MCMC trace and posterior density
# ---------------------------------------------------------------------------
ax_c_trace = fig.add_subplot(gs[1, 0])

# Extract trace for the root node (node 6 -- the oldest coalescence)
node_id_trace = 6
trace = np.array([s[node_id_trace] for s in samples_mcmc])

ax_c_trace.set_title("C.  MCMC posterior for root coalescence time",
                     fontweight="bold", loc="left")

# Trace plot
color_trace = "#0D47A1"
ax_c_trace.plot(trace, lw=0.4, color=color_trace, alpha=0.7)
mean_val = np.mean(trace)
ci_lo = np.percentile(trace, 2.5)
ci_hi = np.percentile(trace, 97.5)
ax_c_trace.axhline(mean_val, color="#E65100", ls="--", lw=1.5,
                   label=f"Mean = {mean_val:.0f}")
ax_c_trace.axhspan(ci_lo, ci_hi, color="#E65100", alpha=0.08,
                   label=f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}]")
ax_c_trace.set_xlabel("MCMC iteration (post burn-in)")
ax_c_trace.set_ylabel("$t_{\\mathrm{root}}$ (generations)")
ax_c_trace.legend(fontsize=8, loc="upper right")

# Inset: posterior density histogram
ax_inset = ax_c_trace.inset_axes([0.62, 0.52, 0.35, 0.42])
ax_inset.hist(trace, bins=40, density=True, color="#1565C0", alpha=0.7,
              edgecolor="white", lw=0.5)
ax_inset.axvline(mean_val, color="#E65100", ls="--", lw=1.2)
ax_inset.set_xlabel("$t_{\\mathrm{root}}$", fontsize=8)
ax_inset.set_ylabel("Density", fontsize=8)
ax_inset.tick_params(labelsize=7)

# Add acceptance rate annotation
ax_c_trace.text(0.02, 0.03, f"Acceptance rate: {acc_rate:.1%}",
               transform=ax_c_trace.transAxes, fontsize=8,
               color="#555555",
               bbox=dict(boxstyle="round,pad=0.2", fc="#EEEEEE", ec="none"))

# ---------------------------------------------------------------------------
# Panel D: N_e(t) step function
# ---------------------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_title("D.  Estimated $N_e(t)$ via M-step", fontweight="bold",
               loc="left")

n_epochs = len(boundaries_d) - 1
epoch_mids = (boundaries_d[:-1] + boundaries_d[1:]) / 2

# Plot as step function
for j in range(n_epochs):
    ep_start = boundaries_d[j]
    ep_end = boundaries_d[j + 1]
    ax_d.plot([ep_start, ep_end], [N_e_est[j], N_e_est[j]],
              color="#1B5E20", lw=2.5, solid_capstyle="butt")
    # Vertical connectors between epochs (except the last)
    if j < n_epochs - 1:
        ax_d.plot([ep_end, ep_end], [N_e_est[j], N_e_est[j + 1]],
                  color="#1B5E20", lw=1.0, ls=":", alpha=0.5)

# True N_e reference
ax_d.axhline(TRUE_NE, color="#C62828", ls="--", lw=1.5,
             label=f"True $N_e$ = {TRUE_NE:,}")

# Epoch midpoints as dots
ax_d.scatter(epoch_mids, N_e_est, s=25, color="#1B5E20", zorder=5)

ax_d.set_xlabel("Time into the past (generations)")
ax_d.set_ylabel("Effective population size $N_e$")
ax_d.set_xscale("log")
ax_d.set_xlim(boundaries_d[1] * 0.5, boundaries_d[-1] * 1.2)
ylo = min(N_e_est.min(), TRUE_NE) * 0.3
yhi = max(N_e_est.max(), TRUE_NE) * 2.0
ax_d.set_ylim(ylo, yhi)
ax_d.set_yscale("log")
ax_d.legend(fontsize=9, loc="upper left")

# Annotation: number of trees and leaves
ax_d.text(0.98, 0.03,
          f"{N_TREES_D} trees, {N_LEAVES_D} leaves each",
          transform=ax_d.transAxes, fontsize=8, ha="right",
          color="#555555",
          bbox=dict(boxstyle="round,pad=0.2", fc="#EEEEEE", ec="none"))

# ============================================================================
# Save
# ============================================================================
fig.savefig("figures/fig_mini_relate.png", dpi=150, bbox_inches="tight")
fig.savefig("figures/fig_mini_relate.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_relate.png and figures/fig_mini_relate.pdf")
