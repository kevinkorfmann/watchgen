"""
Figure: tsinfer algorithm -- tree sequence inference from variation data.

Shows ancestor generation, Viterbi copying paths, inference site selection,
and the full pipeline output as a tree sequence edge diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from watchgen.mini_tsinfer import (
    select_inference_sites,
    compute_ancestor_times,
    generate_ancestors,
    add_ultimate_ancestor,
    viterbi_ls,
    compute_recombination_probs,
    compute_mismatch_probs,
    tsinfer_pipeline,
)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ============================================================================
# Shared dataset: a small genotype matrix for all panels
# ============================================================================
np.random.seed(42)
n_samples, n_sites = 20, 15

D = np.random.binomial(1, 0.3, size=(n_samples, n_sites))
# Force edge cases to show inference site filtering
D[:, 0] = 1                  # Fixed derived (excluded)
D[:, 1] = 0                  # Fixed ancestral (excluded)
D[0, 2] = 1; D[1:, 2] = 0   # Singleton (excluded)

ancestral_known = np.ones(n_sites, dtype=bool)
ancestral_known[3] = False    # Unknown ancestral (excluded)

positions_all = np.arange(0, n_sites * 1000, 1000, dtype=float)

# ============================================================================
# Figure
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("tsinfer: Tree Sequence Inference from Variation Data",
             fontsize=14, fontweight="bold")

# --- Panel A: Ancestor haplotype matrix ---
ax = axes[0, 0]

ancestors, inf_sites = generate_ancestors(D, ancestral_known)
n_inf = len(inf_sites)
ancestors_full = add_ultimate_ancestor(ancestors, n_inf)

# Build a full ancestor matrix (ancestors x inference sites) for the heatmap
# Skip the ultimate (all-zero) ancestor for visual clarity
n_anc = len(ancestors)
anc_matrix = np.full((n_anc, n_inf), np.nan)
for i, anc in enumerate(ancestors):
    anc_matrix[i, anc["start"]:anc["end"]] = anc["haplotype"]

# Custom colormap: NaN -> light grey, 0 -> blue, 1 -> orange
cmap_anc = mcolors.ListedColormap(["#1565C0", "#FF8F00"])
cmap_anc.set_bad(color="#ECEFF1")

im = ax.imshow(anc_matrix, aspect="auto", cmap=cmap_anc,
               interpolation="nearest", vmin=0, vmax=1,
               extent=[-0.5, n_inf - 0.5, n_anc - 0.5, -0.5])

ax.set_xlabel("Inference site index")
ax.set_ylabel("Ancestor (oldest at top)")
ax.set_title("A. Generated ancestor haplotypes")

# Colorbar -- horizontal, below the heatmap, to avoid overlapping time labels
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.6,
                     orientation="horizontal", pad=0.15, aspect=20)
cbar.ax.set_xticklabels(["Ancestral (0)", "Derived (1)"], fontsize=7)

# Annotate time groups on the y-axis (using ytick labels)
# Show a tick for each ancestor row with its time
ax.set_yticks(range(n_anc))
ax.set_yticklabels([f"t={anc['time']:.2f}" for anc in ancestors], fontsize=6)

# --- Panel B: Viterbi copying path ---
ax = axes[0, 1]

np.random.seed(99)
k_ref = 5
m_vit = 30

# Generate distinct reference haplotypes so segments are identifiable
panel = np.random.binomial(1, 0.3, size=(m_vit, k_ref))

# Create a mosaic query from known segments
true_path = np.zeros(m_vit, dtype=int)
true_path[0:8] = 0
true_path[8:17] = 2
true_path[17:24] = 4
true_path[24:30] = 1
query = np.array([panel[ell, true_path[ell]] for ell in range(m_vit)])

# Use spacing that gives moderate recombination probability
positions_vit = np.arange(m_vit, dtype=float) * 500.0
rho_v = np.full(m_vit, 0.05)   # moderate recombination
rho_v[0] = 0.0
mu_v = np.full(m_vit, 0.001)   # low mismatch

path_v, log_p = viterbi_ls(query, panel, rho_v, mu_v)

# Draw colored segments for each copying source
ref_colors = ["#1565C0", "#4CAF50", "#FF8F00", "#9C27B0", "#F44336"]
ref_labels_used = set()

sites = np.arange(m_vit)
for ell in range(m_vit):
    color = ref_colors[path_v[ell]]
    label = f"Ref h{path_v[ell]}" if path_v[ell] not in ref_labels_used else None
    ref_labels_used.add(path_v[ell])
    ax.barh(0, 1, left=ell, height=0.35, color=color, edgecolor="white",
            linewidth=0.3, label=label)
    # True path on a parallel row
    color_true = ref_colors[true_path[ell]]
    ax.barh(0.5, 1, left=ell, height=0.35, color=color_true,
            edgecolor="white", linewidth=0.3, alpha=0.5)

ax.set_yticks([0, 0.5])
ax.set_yticklabels(["Viterbi path", "True path"], fontsize=8)
ax.set_xlabel("Genomic site")
ax.set_title("B. Viterbi copying path (Li & Stephens HMM)")

accuracy_v = np.mean(path_v == true_path)
ax.text(0.02, 0.95, f"Accuracy: {accuracy_v:.0%}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Mark breakpoints in the true path
for bp in [8, 17, 24]:
    ax.axvline(bp, color="#757575", ls=":", lw=0.8, alpha=0.7)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize=7, ncol=k_ref, loc="lower right",
          framealpha=0.9)
ax.set_xlim(-0.5, m_vit + 0.5)
ax.set_ylim(-0.25, 1.0)

# --- Panel C: Inference site selection ---
ax = axes[1, 0]

# Compute allele frequencies at every site
freqs = D.sum(axis=0) / n_samples
inf_set = set(inf_sites)

bar_colors = []
for j in range(n_sites):
    if j in inf_set:
        bar_colors.append("#4CAF50")  # Selected
    else:
        bar_colors.append("#BDBDBD")  # Excluded

bars = ax.bar(range(n_sites), freqs, color=bar_colors, edgecolor="white",
              linewidth=0.5, width=0.8)

# Annotate exclusion reasons
exclusion_reasons = {}
for j in range(n_sites):
    if j in inf_set:
        continue
    if not ancestral_known[j]:
        exclusion_reasons[j] = "unknown\nancestral"
    elif D[:, j].sum() == n_samples:
        exclusion_reasons[j] = "fixed\nderived"
    elif D[:, j].sum() == 0:
        exclusion_reasons[j] = "fixed\nancestral"
    elif D[:, j].sum() < 2:
        exclusion_reasons[j] = "singleton"
    else:
        num_alleles = len(np.unique(D[:, j]))
        if num_alleles != 2:
            exclusion_reasons[j] = "not\nbiallelic"
        else:
            exclusion_reasons[j] = "excluded"

for j, reason in exclusion_reasons.items():
    ax.annotate(reason, xy=(j, freqs[j] + 0.02), fontsize=5.5,
                ha="center", va="bottom", color="#B71C1C", fontweight="bold")

# Threshold lines
ax.axhline(2 / n_samples, color="#FF5722", ls="--", lw=0.8, alpha=0.5,
           label=f"Min freq = 2/{n_samples}")
ax.axhline(1.0, color="#FF5722", ls="--", lw=0.8, alpha=0.5)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label=f"Inference sites ({len(inf_sites)})"),
    Patch(facecolor="#BDBDBD", label=f"Excluded sites ({n_sites - len(inf_sites)})"),
]
ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

ax.set_xlabel("Site index")
ax.set_ylabel("Derived allele frequency")
ax.set_title("C. Inference site selection")
ax.set_xticks(range(n_sites))
ax.set_ylim(0, 1.15)

# --- Panel D: tsinfer pipeline output -- edge diagram ---
ax = axes[1, 1]

# Run the full pipeline on a compact example
np.random.seed(7)
n_d, m_d = 8, 12
D_d = np.random.binomial(1, 0.35, size=(n_d, m_d))
# Ensure variability: force a couple of high-freq columns
D_d[:6, 4] = 1; D_d[6:, 4] = 0
D_d[:5, 8] = 1; D_d[5:, 8] = 0
D_d[:, 0] = 1  # fixed derived -- to be excluded
anc_known_d = np.ones(m_d, dtype=bool)
pos_d = np.arange(m_d, dtype=float) * 1000.0

builder = tsinfer_pipeline(D_d, pos_d, anc_known_d,
                           recombination_rate=1e-3,
                           mismatch_ratio=1.0)

# Collect all edges and plot them as colored horizontal bars
# Color by parent node; y-axis is the child node
time_map = {node["id"]: node["time"] for node in builder.nodes}
is_sample = {node["id"]: node["is_sample"] for node in builder.nodes}

# Sort nodes by time (top = oldest)
sorted_nodes = sorted(builder.nodes, key=lambda nd: -nd["time"])
node_y = {nd["id"]: i for i, nd in enumerate(sorted_nodes)}

# Color palette for parent nodes
parent_ids = sorted(set(p for _, _, p, _ in builder.edges))
n_parents = len(parent_ids)
cmap_edge = plt.cm.Set2
parent_color = {pid: cmap_edge(i / max(n_parents - 1, 1))
                for i, pid in enumerate(parent_ids)}

for left, right, parent, child in builder.edges:
    y = node_y[child]
    color = parent_color[parent]
    ax.barh(y, right - left, left=left, height=0.6, color=color,
            edgecolor="white", linewidth=0.3, alpha=0.85)

# Label sample nodes vs ancestor nodes
for nd in sorted_nodes:
    y = node_y[nd["id"]]
    if nd["is_sample"]:
        ax.text(-300, y, f"s{nd['id']}", fontsize=7, ha="right", va="center",
                color="#1565C0", fontweight="bold")
    else:
        ax.text(-300, y, f"a{nd['id']}", fontsize=7, ha="right", va="center",
                color="#B71C1C")

# Separator between ancestors and samples
sample_ys = [node_y[nd["id"]] for nd in builder.nodes if nd["is_sample"]]
anc_ys = [node_y[nd["id"]] for nd in builder.nodes if not nd["is_sample"]]
if sample_ys and anc_ys:
    sep_y = (max(anc_ys) + min(sample_ys)) / 2
    ax.axhline(sep_y, color="#90A4AE", ls="--", lw=0.8, alpha=0.6)
    ax.text(pos_d[-1] * 0.95, sep_y - 0.35, "samples", fontsize=7,
            ha="right", color="#1565C0", alpha=0.7)
    ax.text(pos_d[-1] * 0.95, sep_y + 0.35, "ancestors", fontsize=7,
            ha="right", color="#B71C1C", alpha=0.7)

ax.set_xlabel("Genomic position")
ax.set_ylabel("Node (sorted by time)")
ax.set_title("D. Inferred tree sequence edges")
ax.set_yticks([])
ax.invert_yaxis()

# Summary text
n_nodes = len(builder.nodes)
n_edges = len(builder.edges)
n_samp = sum(1 for nd in builder.nodes if nd["is_sample"])
ax.text(0.02, 0.95,
        f"{n_samp} samples, {n_nodes - n_samp} ancestors\n{n_edges} edges",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig("figures/fig_mini_tsinfer.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_tsinfer.pdf", bbox_inches="tight")
print("Saved figures/fig_mini_tsinfer.png and .pdf")
