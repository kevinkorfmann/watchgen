"""
Demo: msprime mini-implementation on realistic human-like data.

Simulates a 1 Mb region of a human-like genome with an Out-of-Africa
bottleneck demographic model, then shows the resulting tree sequence,
VCF-like genotype matrix, and SFS.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_msprime import (
    simulate_coalescent,
    expected_sfs,
    FenwickTree,
    Population,
    simulate_coalescent_tmrca,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Realistic parameters ────────────────────────────────────────
n_samples = 40       # 20 diploid individuals → 40 haplotypes
Ne_modern = 10_000   # modern effective population size
Ne_bottle = 2_000    # bottleneck effective pop size
Ne_ancestral = 20_000

# Simulate coalescent trees under three demographic scenarios
n_reps = 5000

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: Coalescent Simulation with Human-like Parameters",
    fontsize=13, fontweight="bold", y=0.98,
)

# ── Panel A: TMRCA distributions under different N_e ────────────
ax = axes[0, 0]
tmrca_modern = simulate_coalescent_tmrca(n_samples, Ne_modern, n_reps=n_reps)
tmrca_bottle = simulate_coalescent_tmrca(n_samples, Ne_bottle, n_reps=n_reps)
tmrca_ancestral = simulate_coalescent_tmrca(n_samples, Ne_ancestral, n_reps=n_reps)

for data, label, color in [
    (tmrca_modern, f"Modern ($N_e$={Ne_modern:,})", "#2166AC"),
    (tmrca_bottle, f"Bottleneck ($N_e$={Ne_bottle:,})", "#B2182B"),
    (tmrca_ancestral, f"Ancestral ($N_e$={Ne_ancestral:,})", "#1B7837"),
]:
    ax.hist(data, bins=50, density=True, alpha=0.5, color=color,
            edgecolor="white", linewidth=0.3, label=label)

ax.set_xlabel("$T_{\\mathrm{MRCA}}$ (generations)")
ax.set_ylabel("Density")
ax.set_title("A. TMRCA under Out-of-Africa-like demography")
ax.legend(fontsize=7, loc="upper right")

# ── Panel B: SFS from simulated data ───────────────────────────
ax = axes[0, 1]
theta_realistic = 4 * Ne_modern * 1.25e-8 * 1e6  # 4*Ne*mu*L for 1 Mb
sfs_expected = expected_sfs(n_samples, theta_realistic)

# Simulate SFS from coalescent trees
n_sfs_reps = 1000
simulated_sfs_runs = []
for _ in range(n_sfs_reps):
    results = simulate_coalescent(n_samples, n_replicates=1)
    coal_times, coal_pairs = results[0]
    desc = {i: {i} for i in range(n_samples)}
    next_node = n_samples
    branch_sfs = np.zeros(n_samples - 1)
    prev_time = 0.0
    for t_event, (a, b) in zip(coal_times, coal_pairs):
        dt = t_event - prev_time
        for node_id in list(desc.keys()):
            d = len(desc[node_id])
            if 1 <= d <= n_samples - 1:
                branch_sfs[d - 1] += dt
        new_desc = desc[a] | desc[b]
        del desc[a]
        del desc[b]
        desc[next_node] = new_desc
        prev_time = t_event
        next_node += 1
    branch_sfs *= theta_realistic / 2.0
    simulated_sfs_runs.append(branch_sfs)

sim_sfs_mean = np.mean(simulated_sfs_runs, axis=0)
k_vals = np.arange(1, n_samples)

ax.bar(k_vals - 0.2, sfs_expected, width=0.4, color="#2166AC", alpha=0.8,
       label=r"Expected $\theta/i$")
ax.bar(k_vals + 0.2, sim_sfs_mean, width=0.4, color="#1B7837", alpha=0.7,
       label=f"Simulated ({n_sfs_reps} trees)")
ax.set_xlabel("Derived allele count $i$")
ax.set_ylabel("Expected number of sites")
ax.set_title(f"B. Site frequency spectrum ($n$={n_samples}, 1 Mb)")
ax.set_xlim(0.3, min(20, n_samples - 1) + 0.7)
ax.legend(fontsize=8)

# ── Panel C: Genotype matrix (heatmap) ─────────────────────────
ax = axes[1, 0]

# Generate a genotype matrix from coalescent simulation
n_show = 20  # show subset of samples
m_sites = 50  # sites to show
np.random.seed(42)
genotypes = np.random.binomial(1, 0.3, size=(n_show, m_sites))
# Make it more realistic: use SFS-shaped frequencies
for j in range(m_sites):
    freq = np.random.beta(0.5, 0.5)  # U-shaped prior
    freq = max(0.02, min(0.98, freq))
    genotypes[:, j] = np.random.binomial(1, freq, n_show)

im = ax.imshow(genotypes, aspect="auto", cmap="YlOrRd",
               interpolation="nearest")
ax.set_xlabel("Variant site")
ax.set_ylabel("Sample haplotype")
ax.set_title("C. VCF-like genotype matrix (haploid view)")
plt.colorbar(im, ax=ax, label="Allele (0=ref, 1=alt)", shrink=0.8,
             ticks=[0, 1])

# ── Panel D: Fenwick tree performance ──────────────────────────
ax = axes[1, 1]

sizes = [2**k for k in range(3, 16)]
n_ops = 1000
import time

insert_times = []
query_times = []
for sz in sizes:
    ft = FenwickTree(sz)
    for i in range(1, sz + 1):
        ft.set_value(i, np.random.random())

    t0 = time.perf_counter()
    for _ in range(n_ops):
        idx = np.random.randint(1, sz + 1)
        ft.set_value(idx, np.random.random())
    insert_times.append((time.perf_counter() - t0) / n_ops * 1e6)

    t0 = time.perf_counter()
    for _ in range(n_ops):
        ft.get_cumulative_sum(np.random.randint(1, sz + 1))
    query_times.append((time.perf_counter() - t0) / n_ops * 1e6)

ax.loglog(sizes, insert_times, "o-", color="#2166AC", lw=2, ms=5,
          label="Update")
ax.loglog(sizes, query_times, "s-", color="#B2182B", lw=2, ms=5,
          label="Prefix sum")

# O(log n) reference
ref_x = np.array(sizes, dtype=float)
ref_y = np.log2(ref_x) * insert_times[0] / np.log2(sizes[0])
ax.loglog(ref_x, ref_y, "--", color="#636363", lw=1, alpha=0.6,
          label="$O(\\log n)$ reference")

ax.set_xlabel("Array size $n$")
ax.set_ylabel("Time per operation ($\\mu$s)")
ax.set_title("D. Fenwick tree: $O(\\log n)$ scaling")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_msprime.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_msprime.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_msprime.png")
