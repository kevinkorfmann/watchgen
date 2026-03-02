"""
Demo: discoal selective sweep simulation.

Runs the mini discoal simulator to generate genealogies under hard sweeps,
and shows the characteristic signatures in sweep dynamics and coalescence.
"""

import numpy as np
import matplotlib.pyplot as plt

from watchgen.mini_discoal import (
    deterministic_trajectory,
    structured_coalescent_sweep,
    minimal_discoal,
)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "font.family": "sans-serif",
})

np.random.seed(2024)

# ── Parameters ──────────────────────────────────────────────────
N = 2_000
s_strong = 0.01
s_weak = 0.003

# ── Deterministic trajectories ───────────────────────────────────
x_det_strong = deterministic_trajectory(s_strong, N, x0=1/(2*N), dt=1.0)
x_det_weak = deterministic_trajectory(s_weak, N, x0=1/(2*N), dt=1.0)

# ── Diversity profile around a hard sweep (Panel B) ──────────────
L_profile = 2_000_000  # 2 Mb — large enough to show sweep recovery
n_sample_profile = 20
# Average over replicates for a smoother diversity curve
n_div_reps = 100
div_reps = []
for rep_seed in range(n_div_reps):
    pos_rep, div_rep = minimal_discoal(
        n=n_sample_profile, N=N, s=s_strong,
        r_per_site=1e-8, L=L_profile, n_sites=100, seed=rep_seed,
    )
    div_reps.append(div_rep)
positions = pos_rep
rel_div = np.mean(div_reps, axis=0)

# ── Structured coalescent under sweep for TMRCA (Panel C) ────────
n_samples = 20
n_reps = 100

coal_times_sweep = []
coal_times_neutral = []

traj_for_coal = deterministic_trajectory(s_strong, N, x0=1/(2*N))
rng_coal = np.random.default_rng(42)
for _ in range(n_reps):
    try:
        # Use a closely linked site (5 kb away) for clear sweep signal
        coal_list, n_B, n_b = structured_coalescent_sweep(
            traj_for_coal, n_samples, r_site=1e-8 * 5_000, N=N, rng=rng_coal
        )
        if len(coal_list) > 0:
            coal_times_sweep.append(max(coal_list))
    except Exception:
        pass

for _ in range(n_reps):
    k = n_samples
    t = 0
    while k > 1:
        rate = k * (k - 1) / (4 * N)
        t += np.random.exponential(1 / rate)
        k -= 1
    coal_times_neutral.append(t)

# ── Sweep duration vs selection coefficient (Panel D) ────────────
s_range = np.logspace(-3, -1, 30)  # avoid very small s (slow logistic)
durations = []
for s_val in s_range:
    x = deterministic_trajectory(s_val, N, x0=1/(2*N), dt=1.0)
    durations.append(len(x))

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    f"Demo: discoal Selective Sweep Simulation ($N$={N:,})",
    fontsize=13, fontweight="bold", y=0.98,
)

# Panel A: Deterministic trajectories
ax = axes[0, 0]
ax.plot(range(len(x_det_strong)), x_det_strong, color="#B2182B", lw=2.5,
        label=f"Strong ($s$={s_strong})")
ax.plot(range(len(x_det_weak)), x_det_weak, color="#2166AC", lw=2.5,
        label=f"Weak ($s$={s_weak})")
ax.set_xlabel("Generation")
ax.set_ylabel("Beneficial allele frequency")
ax.set_title("A. Deterministic sweep trajectory")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)

# Panel B: Diversity reduction profile around the sweep
ax = axes[0, 1]
ax.scatter(positions / 1000, rel_div, s=20, color="#1B7837", alpha=0.7, zorder=5)
ax.plot(positions / 1000, rel_div, "-", color="#1B7837", lw=1, alpha=0.5)
# Theoretical diversity reduction: f(r) ≈ r / (r + s)  [Maynard Smith & Haigh]
d_arr = np.abs(positions - L_profile / 2)
r_arr = 1e-8 * d_arr
f_theory = r_arr / (r_arr + s_strong)
ax.plot(positions / 1000, f_theory, "--", color="#B2182B", lw=2,
        label="Theory: $r/(r+s)$")
ax.axvline(L_profile / 2 / 1000, color="#B2182B", lw=1, ls=":", alpha=0.5)
ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.5, label="Neutral (ref)")
ax.set_xlabel("Position (Mb)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}"))
ax.set_ylabel("Relative diversity")
ax.set_title(f"B. Sweep signature ($s$={s_strong}, $n$={n_sample_profile}, {n_div_reps} reps)")
ax.legend(fontsize=8)

# Panel C: TMRCA distribution — zoom to show both sweep and neutral
ax = axes[1, 0]
if coal_times_sweep and coal_times_neutral:
    # Use 80th percentile of neutral as upper limit to keep sweep visible
    upper = np.percentile(coal_times_neutral, 80)
    bins = np.linspace(0, upper, 25)
    neutral_in_range = [t for t in coal_times_neutral if t <= upper]
    ax.hist(neutral_in_range, bins=bins, density=True, alpha=0.5,
            color="#2166AC", edgecolor="white", linewidth=0.3, label="Neutral")
    sweep_in_range = [t for t in coal_times_sweep if t <= upper]
    if sweep_in_range:
        ax.hist(sweep_in_range, bins=bins, density=True, alpha=0.5,
                color="#B2182B", edgecolor="white", linewidth=0.3,
                label=f"Sweep ($s$={s_strong})")
    ax.set_xlabel("$T_{\\mathrm{MRCA}}$ (generations)")
    ax.set_ylabel("Density")
    ax.set_title(f"C. TMRCA distribution ($n$={n_samples}, linked site)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, upper)
else:
    ax.text(0.5, 0.5, "No sweep coalescence data", ha="center", va="center",
            transform=ax.transAxes, fontsize=10)
    ax.set_title(f"C. TMRCA distribution ($n$={n_samples})")

# Panel D: Sweep duration vs selection coefficient
ax = axes[1, 1]
ax.loglog(s_range, durations, "o-", color="#2166AC", lw=2, ms=4)
ax.set_xlabel("Selection coefficient $s$")
ax.set_ylabel("Sweep duration (generations)")
ax.set_title("D. Sweep speed: $\\tau \\approx (4/s) \\ln(2Ns)$")

tau_theory = 4 / s_range * np.log(2 * N * s_range + 1)
ax.loglog(s_range, tau_theory, "--", color="#B2182B", lw=1.5,
          label="Theory: $4 \\ln(2Ns) / s$")
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_discoal.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_discoal.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_discoal.png")
