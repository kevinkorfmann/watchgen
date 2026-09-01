"""Generate independently checkable mini-Relate exercises."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from watchgen import mini_relate as relate


OUT = Path(__file__).parent


def balanced_tree():
    left = relate.TreeNode(4, relate.TreeNode(0), relate.TreeNode(1))
    right = relate.TreeNode(5, relate.TreeNode(2), relate.TreeNode(3))
    return relate.TreeNode(6, left, right)


def main():
    target = np.array([1, 1, 0, 1, 0, 1])
    panel = np.array(
        [[1, 0, 1], [1, 0, 1], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]]
    )
    posterior = relate.copying_posterior(target, panel, [0, 0.05, 0.05, 0.4, 0.05, 0.05])

    root = balanced_tree()
    exposure = {branch: 1.0 for branch in relate.branch_lengths(
        root, relate.node_times_from_intervals(root, [4, 5, 6], [0.2, 0.3, 0.5])
    )}
    draws, acceptance = relate.sample_ranked_branch_lengths(
        root, [4, 5, 6], {0: 1, 1: 1, 2: 1}, exposure, theta=1.0,
        iterations=3_000, burn_in=500, seed=19,
    )
    rates, events, times = relate.piecewise_coalescence_rate_mle(
        [0.2, 0.4, 1.2, 1.8], [0.0, 0.5, 1.0, 2.0]
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for reference in range(panel.shape[1]):
        axes[0].plot(np.arange(target.size), posterior[:, reference], marker="o", label=f"reference {reference}")
    axes[0].set(xlabel="site", ylabel="copying posterior", ylim=(0, 1), title="A  Directional painting")
    axes[0].legend(frameon=False, fontsize=8)

    root_ages = draws.sum(axis=1)
    axes[1].hist(root_ages, bins=35, color="#6a4c93", alpha=0.8)
    axes[1].axvline(root_ages.mean(), color="black", ls="--", lw=1)
    axes[1].set(xlabel="root age (coalescent units)", ylabel="draws", title=f"B  Fixed-order MCMC\nacceptance={acceptance:.2f}")

    epochs = np.arange(3)
    width = 0.25
    axes[2].bar(epochs - width, events, width, label="events", color="#1982c4")
    axes[2].bar(epochs, times, width, label="exposure", color="#8ac926")
    axes[2].bar(epochs + width, rates, width, label="rate", color="#ff595e")
    axes[2].set(xticks=epochs, xticklabels=["0-.5", ".5-1", "1-2"], xlabel="time epoch", title="C  Events / exposure")
    axes[2].legend(frameon=False, fontsize=8)

    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig_demo_relate.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
