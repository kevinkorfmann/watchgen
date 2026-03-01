"""
Demo: Li & Stephens HMM on msprime-simulated haplotype data.

Simulates a reference panel and query haplotype as mosaic of reference
haplotypes (mimicking real phased VCF data), runs forward-backward and
Viterbi algorithms, and shows copying path recovery.
"""

import numpy as np
import matplotlib.pyplot as plt
import msprime

from watchgen.mini_lshmm import (
    estimate_mutation_probability,
    forwards_ls_hap,
    backwards_ls_hap,
    posterior_decoding,
    forwards_viterbi_hap,
    backwards_viterbi_hap,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Simulate realistic haplotype data with msprime ──────────────
ts = msprime.simulate(
    sample_size=42,  # 40 reference + 2 (query pair)
    Ne=10_000,
    length=500_000,   # 500 kb
    recombination_rate=1e-8,
    mutation_rate=1.25e-8,
    random_seed=2024,
)

# Extract haplotype matrix from tree sequence
G = ts.genotype_matrix()  # sites x samples
positions = np.array([v.position for v in ts.variants()])
n_sites_total, n_haps_total = G.shape

# Use first 40 as reference panel, last 2 as query
H = G[:, :40]       # m x n reference panel
query = G[:, 40]    # query haplotype

# Subsample to manageable size for visualization
step = max(1, n_sites_total // 300)
H = H[::step]
query = query[::step]
positions = positions[::step]
m, n = H.shape

# ── Run Li & Stephens algorithms ────────────────────────────────
mu = estimate_mutation_probability(n)
e_mat = np.zeros((m, 2))
e_mat[:, 0] = mu
e_mat[:, 1] = 1 - mu
r_arr = np.full(m, 0.01)
r_arr[0] = 0.0

s_2d = query.reshape(1, -1)

F, c, ll_fwd = forwards_ls_hap(n, m, H, s_2d, e_mat, r_arr, norm=True)
B = backwards_ls_hap(n, m, H, s_2d, e_mat, c, r_arr)
gamma, posterior_path = posterior_decoding(F, B)
V, P, ll_vit = forwards_viterbi_hap(n, m, H, s_2d, e_mat, r_arr)
viterbi_path = backwards_viterbi_hap(m, V, P)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: Li & Stephens HMM on msprime-simulated Haplotypes (500 kb)",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Genotype matrix heatmap
ax = axes[0, 0]
show_sites = min(100, m)
show_haps = min(40, n)
im = ax.imshow(H[:show_sites, :show_haps].T, aspect="auto",
               cmap="YlOrRd", interpolation="nearest")
ax.set_xlabel(f"Variant site (first {show_sites})")
ax.set_ylabel("Reference haplotype")
ax.set_title(f"A. Reference panel ({n} haps, {m} sites from VCF)")
plt.colorbar(im, ax=ax, label="Allele", shrink=0.8, ticks=[0, 1])

# Panel B: Viterbi copying path
ax = axes[0, 1]
sites = np.arange(m)
# Color by copying source
unique_sources = np.unique(viterbi_path)
colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_sources), 1)))
src_color = {s: colors[i % len(colors)] for i, s in enumerate(unique_sources)}

for ell in range(m):
    ax.barh(0, 1, left=ell, height=0.5, color=src_color[viterbi_path[ell]],
            edgecolor="none")

# Mark segment boundaries
prev = viterbi_path[0]
breakpoints = []
for ell in range(1, m):
    if viterbi_path[ell] != prev:
        breakpoints.append(ell)
        prev = viterbi_path[ell]

for bp in breakpoints:
    ax.axvline(bp, color="black", lw=0.5, alpha=0.5)

n_segments = len(breakpoints) + 1
ax.set_xlabel("Variant site")
ax.set_yticks([0])
ax.set_yticklabels(["Viterbi path"])
ax.set_title(f"B. Inferred copying path ({n_segments} segments)")
ax.set_xlim(0, m)

# Panel C: Forward probability heatmap
ax = axes[1, 0]
im = ax.imshow(F[:, :20].T, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.set_xlabel("Variant site")
ax.set_ylabel("Reference haplotype (top 20)")
ax.set_title("C. Forward probabilities $P(Z_\\ell = k | \\text{data}_{1:\\ell})$")
plt.colorbar(im, ax=ax, label="Forward prob", shrink=0.8)

# Panel D: Posterior probabilities for top sources
ax = axes[1, 1]
# Find the top 5 most-copied reference haplotypes
source_counts = np.bincount(viterbi_path, minlength=n)
top5 = np.argsort(-source_counts)[:5]

for rank, hap_idx in enumerate(top5):
    ax.plot(sites, gamma[:, hap_idx], lw=1.2, alpha=0.8,
            label=f"h{hap_idx} ({source_counts[hap_idx]} sites)")

ax.set_xlabel("Variant site")
ax.set_ylabel("Posterior probability")
ax.set_title("D. Posterior decoding (top 5 sources)")
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.set_ylim(0, 1.05)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_lshmm.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_lshmm.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_lshmm.png")
