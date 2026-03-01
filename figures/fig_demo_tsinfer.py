"""
Demo: tsinfer tree sequence inference on msprime-simulated VCF-like data.

Simulates a genomic region with msprime, extracts the genotype matrix
(as would come from a VCF), runs the tsinfer pipeline, and compares
the inferred tree sequence to the truth.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_tsinfer import (
    select_inference_sites,
    generate_ancestors,
    add_ultimate_ancestor,
    tsinfer_pipeline,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate with msprime ───────────────────────────────────────
ts = msprime.simulate(
    sample_size=20,
    Ne=10_000,
    length=100_000,
    recombination_rate=1e-8,
    mutation_rate=1.25e-8,
    random_seed=2024,
)

# Extract genotype matrix and positions (as from VCF)
G = ts.genotype_matrix().T  # samples x sites
positions = np.array([v.position for v in ts.variants()])
n_samples, n_sites = G.shape
ancestral_known = np.ones(n_sites, dtype=bool)

# ── Run tsinfer pipeline ────────────────────────────────────────
builder = tsinfer_pipeline(
    G, positions, ancestral_known,
    recombination_rate=1e-3,
    mismatch_ratio=1.0,
)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: tsinfer on msprime-simulated Data ({n_samples} samples, {n_sites} sites)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Input genotype matrix
ax = axes[0, 0]
show_n = min(n_samples, 20)
show_m = min(n_sites, 80)
im = ax.imshow(G[:show_n, :show_m], aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.set_xlabel(f"Variant site (first {show_m} of {n_sites})")
ax.set_ylabel("Sample")
ax.set_title(f"A. Input: VCF genotype matrix ({n_samples}$\\times${n_sites})")
plt.colorbar(im, ax=ax, label="Allele", shrink=0.8, ticks=[0, 1])

# Panel B: Ancestor haplotypes
ax = axes[0, 1]
ancestors, inf_sites = generate_ancestors(G, ancestral_known)
n_inf = len(inf_sites)
n_anc = len(ancestors)
anc_matrix = np.full((min(n_anc, 30), n_inf), np.nan)
for i, anc in enumerate(ancestors[:30]):
    anc_matrix[i, anc["start"]:anc["end"]] = anc["haplotype"]

import matplotlib.colors as mcolors
cmap_anc = mcolors.ListedColormap(["#1565C0", "#FF8F00"])
cmap_anc.set_bad(color="#ECEFF1")
im = ax.imshow(anc_matrix, aspect="auto", cmap=cmap_anc,
               interpolation="nearest", vmin=0, vmax=1)
ax.set_xlabel("Inference site index")
ax.set_ylabel("Ancestor (oldest at top)")
ax.set_title(f"B. Generated ancestors ({n_anc} total)")

# Panel C: Inferred tree sequence edges
ax = axes[1, 0]
time_map = {node["id"]: node["time"] for node in builder.nodes}
sorted_nodes = sorted(builder.nodes, key=lambda nd: -nd["time"])
node_y = {nd["id"]: i for i, nd in enumerate(sorted_nodes)}

parent_ids = sorted(set(p for _, _, p, _ in builder.edges))
n_parents = len(parent_ids)
cmap_edge = plt.cm.Set2
parent_color = {pid: cmap_edge(i / max(n_parents - 1, 1))
                for i, pid in enumerate(parent_ids)}

for left, right, parent, child in builder.edges:
    y = node_y[child]
    color = parent_color[parent]
    ax.barh(y, right - left, left=left, height=0.6, color=color,
            edgecolor="white", linewidth=0.2, alpha=0.85)

n_samp = sum(1 for nd in builder.nodes if nd["is_sample"])
n_edges = len(builder.edges)
ax.set_xlabel("Genomic position (bp)")
ax.set_ylabel("Node (sorted by time)")
ax.set_title(f"C. Inferred edges ({n_edges} edges, {len(builder.nodes)} nodes)")
ax.set_yticks([])
ax.invert_yaxis()

# Panel D: Allele frequency spectrum comparison
ax = axes[1, 1]
# True SFS from the genotype matrix
freq_counts = G.sum(axis=0)
true_sfs = np.bincount(freq_counts, minlength=n_samples + 1)[1:n_samples]

k_vals = np.arange(1, n_samples)
ax.bar(k_vals, true_sfs, color="#2166AC", alpha=0.7, edgecolor="white",
       linewidth=0.5, label="Observed SFS from VCF")

# Theoretical neutral expectation
n_seg = n_sites
theta_hat = true_sfs.sum() / sum(1/k for k in range(1, n_samples))
expected = np.array([theta_hat / k for k in k_vals])
ax.plot(k_vals, expected, "o-", color="#B2182B", lw=2, ms=3,
        label=f"Neutral expectation ($\\hat{{\\theta}}$={theta_hat:.1f})")

ax.set_xlabel("Derived allele count")
ax.set_ylabel("Number of sites")
ax.set_title("D. SFS of input data")
ax.legend(fontsize=8)
ax.set_xlim(0.5, min(20, n_samples - 1) + 0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_tsinfer.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_tsinfer.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_tsinfer.png")
