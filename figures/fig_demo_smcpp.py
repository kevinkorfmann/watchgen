"""Demo: verified SMC++ CSFS, transition, and likelihood mechanisms."""

import matplotlib.pyplot as plt
import numpy as np

from watchgen.mini_smcpp import (
    constant_csc_transition_matrix,
    emission_probabilities,
    forward_log_likelihood,
    pair_interval_probabilities,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "font.family": "sans-serif",
    }
)

BREAKS = np.array([0.0, 0.25, 0.75, 2.0, np.inf])
N_UNDISTINGUISHED = 2
THETA = 0.02
RHO = 0.1
emissions = emission_probabilities(
    BREAKS, N_UNDISTINGUISHED, THETA, quadrature_order=12
)
transitions = constant_csc_transition_matrix(BREAKS, 1.0, RHO)
initial = pair_interval_probabilities(BREAKS)
state_labels = ["0–0.25", "0.25–0.75", "0.75–2", "2–∞"]

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
fig.suptitle(
    "Demo: SMC++ distinguished-pair building blocks",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

ax = axes[0, 0]
x = np.arange(len(initial))
ax.bar(x, initial, color="#2166AC", alpha=0.85)
ax.set_xticks(x, state_labels)
ax.set(
    xlabel="Distinguished-pair TMRCA interval",
    ylabel="Probability",
    title="A. Hidden-state distribution",
)

ax = axes[0, 1]
categories = [
    ((0, 0), "No mutation"),
    ((1, 0), "Distinguished singleton"),
    ((0, 1), "Undistinguished singleton"),
    ((2, 0), "Pair branch"),
]
for (a, b), label in categories:
    ax.plot(x, emissions[:, a, b], "o-", lw=2, ms=4, label=label)
ax.set_xticks(x, state_labels)
ax.set(
    xlabel="TMRCA interval",
    ylabel="Emission probability",
    title="B. Conditioned-SFS emissions",
)
ax.legend(fontsize=7)

ax = axes[1, 0]
image = ax.imshow(transitions, cmap="Blues", vmin=0, vmax=1, aspect="equal")
ax.set_xticks(x, state_labels, rotation=35, ha="right")
ax.set_yticks(x, state_labels)
ax.set(
    xlabel="Next TMRCA interval",
    ylabel="Current TMRCA interval",
    title="C. Exact constant-demography transition",
)
plt.colorbar(image, ax=ax, label="Probability", shrink=0.82)

ax = axes[1, 1]
datasets = [
    np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 1], [0, 0]]),
    np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [1, 0]]),
    np.array([[0, 0], [0, 1], [0, 0], [1, 0], [0, 0], [0, 0]]),
]
for i, data in enumerate(datasets, start=1):
    cumulative = [
        forward_log_likelihood(data[:end], transitions, emissions, initial)
        for end in range(1, len(data) + 1)
    ]
    ax.plot(range(1, len(data) + 1), cumulative, "o-", lw=1.8, label=f"Pair {i}")
ax.set(
    xlabel="Sites included",
    ylabel="Cumulative log likelihood",
    title="D. Scaled forward likelihoods",
)
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/fig_demo_smcpp.png", dpi=150, bbox_inches="tight")
plt.savefig("figures/fig_demo_smcpp.pdf", bbox_inches="tight")
print("Saved figures/fig_demo_smcpp.png and figures/fig_demo_smcpp.pdf")
