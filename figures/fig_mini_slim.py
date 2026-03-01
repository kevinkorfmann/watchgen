"""
Figure: SLiM forward-time Wright-Fisher simulation.

Showcases the core capabilities of the mini SLiM forward simulator:
allele frequency trajectories under selection, fixation probability theory,
the site frequency spectrum under neutrality, and background selection.
"""

import numpy as np
import matplotlib.pyplot as plt
from watchgen.mini_slim import simulate_sweep, simulate, simulate_bgs

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("SLiM: Forward-Time Wright-Fisher Simulation",
             fontsize=14, fontweight="bold")

np.random.seed(42)

# ---------------------------------------------------------------------------
# Panel A: Allele frequency trajectory of a beneficial mutation (sweep)
# ---------------------------------------------------------------------------
print("=== Panel A: Selective sweep ===")
ax = axes[0, 0]

N_sweep = 50
L_sweep = 50
mu_sweep = 0.0
r_sweep = 0.0
s_sweep = 0.15

sweep_traj = None
for attempt in range(100):
    traj, fixed = simulate_sweep(
        N=N_sweep, L=L_sweep, mu=mu_sweep, r=r_sweep,
        s_beneficial=s_sweep, position_selected=25,
        T_burnin=0, T_after=400, track_interval=1,
    )
    if fixed:
        sweep_traj = traj
        print(f"Panel A: sweep fixed on attempt {attempt + 1}")
        break

if sweep_traj is not None:
    gens = [g for g, f in sweep_traj]
    freqs = [f for g, f in sweep_traj]
    ax.plot(gens, freqs, color="#D32F2F", lw=2.2)
    ax.fill_between(gens, freqs, alpha=0.10, color="#D32F2F")
    ax.axhline(0.5, ls=":", color="#757575", lw=0.8, alpha=0.6)
    ax.axhline(1.0, ls="--", color="#757575", lw=0.8, alpha=0.6)
else:
    ax.text(0.5, 0.5, "No fixation observed", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="gray")

ax.set_xlabel("Generation after introduction")
ax.set_ylabel("Beneficial allele frequency")
ax.set_title(f"A. Selective sweep (N={N_sweep}, s={s_sweep})")
ax.set_ylim(-0.02, 1.05)
ax.set_xlim(left=0)

# ---------------------------------------------------------------------------
# Panel B: Fixation probability vs selection coefficient
# ---------------------------------------------------------------------------
print("\n=== Panel B: Fixation probability ===")
ax = axes[0, 1]

N_fix = 50
L_fix = 50
mu_fix = 0.0
r_fix = 0.0
n_trials = 30

s_values = [0.01, 0.05, 0.1, 0.2]
p_fix_obs = []

for s_val in s_values:
    n_fixed = 0
    for trial in range(n_trials):
        _, fixed = simulate_sweep(
            N=N_fix, L=L_fix, mu=mu_fix, r=r_fix,
            s_beneficial=s_val, position_selected=25,
            T_burnin=0, T_after=1000, track_interval=99999,
        )
        if fixed:
            n_fixed += 1
    p_obs = n_fixed / n_trials
    p_fix_obs.append(p_obs)
    print(f"  s={s_val:.3f}, p_fix={p_obs:.4f} ({n_fixed}/{n_trials})")

# Theoretical curve: Kimura (1962) -- p_fix = 2s / (1 - exp(-4Ns))
s_theory = np.linspace(0.001, 0.3, 200)
p_theory = (2 * s_theory) / (1 - np.exp(-4 * N_fix * s_theory))

ax.plot(s_theory, p_theory, color="#1565C0", lw=2.2,
        label="Kimura theory", zorder=2)
ax.scatter(s_values, p_fix_obs, color="#FF6F00", s=70, edgecolors="black",
           linewidth=0.8, zorder=3, label=f"Simulated ({n_trials} trials)")
ax.set_xlabel("Selection coefficient s")
ax.set_ylabel("Fixation probability")
ax.set_title(f"B. Fixation probability (N={N_fix})")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(-0.005, 0.3)
ax.set_ylim(bottom=0)

# ---------------------------------------------------------------------------
# Panel C: Site frequency spectrum under neutrality
# ---------------------------------------------------------------------------
print("\n=== Panel C: Site frequency spectrum ===")
ax = axes[1, 0]

N_sfs = 50
L_sfs = 2000
mu_sfs = 5e-4
r_sfs = 1e-5
T_sfs = 300

np.random.seed(123)
pop_sfs, _ = simulate(
    N=N_sfs, L=L_sfs, mu=mu_sfs, r=r_sfs, T=T_sfs,
    dfe="neutral", track_every=T_sfs,
)

# Compute allele frequency spectrum from the final population
position_counts = {}
for ind in pop_sfs:
    for m in ind.haplosome_1:
        position_counts[m.position] = position_counts.get(m.position, 0) + 1
    for m in ind.haplosome_2:
        position_counts[m.position] = position_counts.get(m.position, 0) + 1

total_haps = 2 * N_sfs
allele_freqs = np.array([c / total_haps for c in position_counts.values()
                         if 0 < c < total_haps])

# Fold the spectrum
allele_freqs_folded = np.minimum(allele_freqs, 1.0 - allele_freqs)

n_bins = 15
bins = np.linspace(0, 0.5, n_bins + 1)
counts_hist, bin_edges = np.histogram(allele_freqs_folded, bins=bins)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

ax.bar(bin_centers, counts_hist, width=bin_edges[1] - bin_edges[0],
       color="#00897B", edgecolor="white", linewidth=0.5, alpha=0.85,
       label="Simulated")

# Theoretical folded SFS density: proportional to 1/x + 1/(1-x)
x_th = np.linspace(0.01, 0.49, 200)
sfs_density = 1.0 / x_th + 1.0 / (1.0 - x_th)
# Scale to match histogram area using trapezoidal integration
hist_area = np.sum(counts_hist * (bin_edges[1] - bin_edges[0]))
theory_area = np.trapezoid(sfs_density, x_th)
sfs_scaled = sfs_density * (hist_area / theory_area)
ax.plot(x_th, sfs_scaled, color="#D32F2F", lw=2, ls="--",
        label=r"Neutral theory $\propto 1/x + 1/(1{-}x)$")

ax.set_xlabel("Minor allele frequency")
ax.set_ylabel("Number of sites")
ax.set_title(f"C. Folded SFS (N={N_sfs}, neutral)")
ax.legend(fontsize=8)
ax.set_xlim(0, 0.5)

# ---------------------------------------------------------------------------
# Panel D: Background selection -- neutral diversity reduction
# ---------------------------------------------------------------------------
print("\n=== Panel D: Background selection ===")
ax = axes[1, 1]

N_bgs = 50
L_bgs = 2000
T_bgs = 300

# With deleterious mutations (background selection)
np.random.seed(99)
stats_bgs = simulate_bgs(
    N=N_bgs, L=L_bgs,
    mu_neutral=2e-4, mu_deleterious=2e-3,
    s_deleterious=-0.1, r=1e-5,
    T=T_bgs, track_interval=5,
)

# Pure neutral: use simulate_bgs with zero deleterious rate
print("\n  Running neutral comparison...")
np.random.seed(99)
stats_neutral = simulate_bgs(
    N=N_bgs, L=L_bgs,
    mu_neutral=2e-4, mu_deleterious=0.0,
    s_deleterious=0.0, r=1e-5,
    T=T_bgs, track_interval=5,
)

ax.plot(stats_bgs["generation"], stats_bgs["neutral_diversity"],
        color="#7B1FA2", lw=2, label="With deleterious (s=-0.1)")
ax.plot(stats_neutral["generation"], stats_neutral["neutral_diversity"],
        color="#388E3C", lw=2, ls="--", label="Neutral only")
ax.set_xlabel("Generation")
ax.set_ylabel("Neutral segregating sites")
ax.set_title(f"D. Background selection (N={N_bgs})")
ax.legend(fontsize=8)
ax.set_xlim(0, T_bgs)

# ---------------------------------------------------------------------------
plt.tight_layout()
plt.savefig("figures/fig_mini_slim.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_mini_slim.pdf", bbox_inches="tight")
print("\nSaved figures/fig_mini_slim.png and figures/fig_mini_slim.pdf")
