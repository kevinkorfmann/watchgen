"""
Tests for the Mainspring dating module.

Covers: EdgeDatingMLP, dating_loss, bottom-up propagation,
training loop convergence, and predicted time reasonableness.
"""

import numpy as np
import pytest
import torch


def _has_msprime():
    try:
        import msprime
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# EdgeDatingMLP tests
# ---------------------------------------------------------------------------


class TestEdgeDatingMLP:
    """Tests for the EdgeDatingMLP network."""

    def test_output_shapes(self):
        """Alpha and beta have shape (E,)."""
        from timepieces.mainspring_piece.dating import EdgeDatingMLP

        mlp = EdgeDatingMLP(hidden_dim=32, n_hidden=2)
        features = torch.randn(20, 5)
        alpha, beta = mlp(features)
        assert alpha.shape == (20,)
        assert beta.shape == (20,)

    def test_output_positivity(self):
        """Alpha and beta are strictly positive."""
        from timepieces.mainspring_piece.dating import EdgeDatingMLP

        torch.manual_seed(0)
        mlp = EdgeDatingMLP(hidden_dim=32, n_hidden=2)
        features = torch.randn(50, 5)
        alpha, beta = mlp(features)
        assert (alpha > 0).all()
        assert (beta > 0).all()

    def test_single_edge(self):
        """Works with a single edge."""
        from timepieces.mainspring_piece.dating import EdgeDatingMLP

        mlp = EdgeDatingMLP()
        features = torch.randn(1, 5)
        alpha, beta = mlp(features)
        assert alpha.shape == (1,)
        assert beta.shape == (1,)


# ---------------------------------------------------------------------------
# Dating loss tests
# ---------------------------------------------------------------------------


class TestDatingLoss:
    """Tests for the Gamma NLL dating loss."""

    def test_gradient_flow(self):
        """Loss propagates gradients to MLP parameters."""
        from timepieces.mainspring_piece.dating import EdgeDatingMLP
        from timepieces.mainspring_piece.losses import dating_loss

        mlp = EdgeDatingMLP(hidden_dim=16, n_hidden=1)
        features = torch.randn(10, 5)
        alpha, beta = mlp(features)
        true_inc = torch.ones(10) * 100.0
        loss = dating_loss(alpha, beta, true_inc)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in mlp.parameters()
        )
        assert has_grad, "No gradients flowed to MLP parameters"

    def test_loss_is_scalar(self):
        """Loss returns a scalar tensor."""
        from timepieces.mainspring_piece.losses import dating_loss

        alpha = torch.tensor([2.0, 3.0])
        beta = torch.tensor([1.0, 1.5])
        inc = torch.tensor([1.0, 2.0])
        loss = dating_loss(alpha, beta, inc)
        assert loss.dim() == 0

    def test_loss_finite(self):
        """Loss is finite for reasonable inputs."""
        from timepieces.mainspring_piece.losses import dating_loss

        alpha = torch.tensor([1.0, 5.0, 10.0])
        beta = torch.tensor([0.01, 0.1, 1.0])
        inc = torch.tensor([100.0, 50.0, 10.0])
        loss = dating_loss(alpha, beta, inc)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Bottom-up propagation tests
# ---------------------------------------------------------------------------


class TestPropagation:
    """Tests for bottom-up time propagation."""

    def test_parent_greater_than_child(self):
        """Propagated times enforce parent > child ordering."""
        from timepieces.mainspring_piece.dating import propagate_times_bottom_up

        # Simple tree: 0->2, 1->2, 2->3
        edge_list = [
            (0, 2, 0.0, 100.0),
            (1, 2, 0.0, 100.0),
            (2, 3, 0.0, 100.0),
        ]
        gamma_means = np.array([50.0, 60.0, 100.0])
        node_times = propagate_times_bottom_up(edge_list, gamma_means, N=2)

        assert node_times[0] == 0.0
        assert node_times[1] == 0.0
        assert node_times[2] > node_times[0]
        assert node_times[2] > node_times[1]
        assert node_times[3] > node_times[2]

    def test_leaves_at_zero(self):
        """Sample nodes are at time 0."""
        from timepieces.mainspring_piece.dating import propagate_times_bottom_up

        edge_list = [(0, 2, 0.0, 100.0), (1, 2, 0.0, 100.0)]
        gamma_means = np.array([10.0, 20.0])
        node_times = propagate_times_bottom_up(edge_list, gamma_means, N=2)

        assert node_times[0] == 0.0
        assert node_times[1] == 0.0

    def test_deeper_tree(self):
        """Deeper trees maintain strict ordering."""
        from timepieces.mainspring_piece.dating import propagate_times_bottom_up

        # 0->3, 1->3, 2->4, 3->5, 4->5
        edge_list = [
            (0, 3, 0.0, 100.0),
            (1, 3, 0.0, 100.0),
            (2, 4, 0.0, 100.0),
            (3, 5, 0.0, 100.0),
            (4, 5, 0.0, 100.0),
        ]
        gamma_means = np.array([10.0, 20.0, 15.0, 30.0, 25.0])
        node_times = propagate_times_bottom_up(edge_list, gamma_means, N=3)

        assert node_times[5] > node_times[3]
        assert node_times[5] > node_times[4]
        assert node_times[3] > node_times[0]
        assert node_times[4] > node_times[2]


# ---------------------------------------------------------------------------
# Training convergence test
# ---------------------------------------------------------------------------


class TestDatingTraining:
    """Tests for dating training loop convergence."""

    @pytest.mark.skipif(
        not _has_msprime(), reason="msprime not installed"
    )
    def test_loss_decreases(self):
        """Dating loss decreases over 50 epochs on msprime edge data."""
        from timepieces.mainspring_piece.dating import EdgeDatingMLP
        from timepieces.mainspring_piece.training import train_dating_epoch

        torch.manual_seed(42)
        mlp = EdgeDatingMLP(hidden_dim=32, n_hidden=2)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        losses = []
        for epoch in range(50):
            avg_loss = train_dating_epoch(
                mlp, optimizer,
                n_samples=6, seq_length=1e4,
                recomb_rate=1.5e-8, mut_rate=1.5e-8,
                n_batches=3, random_seed=epoch * 100,
            )
            losses.append(avg_loss)

        # Loss should decrease from early to late epochs
        early = np.mean(losses[:10])
        late = np.mean(losses[-10:])
        assert late < early, (
            f"Loss did not decrease: early={early:.3f}, late={late:.3f}"
        )

    @pytest.mark.skipif(
        not _has_msprime(), reason="msprime not installed"
    )
    def test_predicted_times_reasonable(self):
        """Predicted times are same order of magnitude as true times."""
        import msprime
        from timepieces.mainspring_piece.dating import EdgeDatingMLP
        from timepieces.mainspring_piece.training import (
            extract_dating_targets, train_dating_epoch,
        )
        from timepieces.mainspring_piece.losses import dating_loss

        torch.manual_seed(42)
        mlp = EdgeDatingMLP(hidden_dim=32, n_hidden=2)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        # Train for a bit
        for epoch in range(30):
            train_dating_epoch(
                mlp, optimizer,
                n_samples=6, seq_length=1e4,
                recomb_rate=1.5e-8, mut_rate=1.5e-8,
                n_batches=3, random_seed=epoch * 100,
            )

        # Generate test data and check predictions
        ts = msprime.sim_ancestry(
            samples=6, sequence_length=1e4,
            recombination_rate=1.5e-8, population_size=1e4,
            random_seed=999,
        )
        ts = msprime.sim_mutations(ts, rate=1.5e-8, random_seed=999)

        edge_features, true_increments = extract_dating_targets(ts)
        if len(true_increments) == 0:
            pytest.skip("No edges with positive increments")

        feat_t = torch.from_numpy(edge_features).float()
        mlp.eval()
        with torch.no_grad():
            alpha, beta = mlp(feat_t)
        # MLP predicts in normalized time; rescale by time_scale
        pred_means = (alpha / beta).numpy() * mlp.time_scale

        # Check same order of magnitude (within 100x)
        ratio = pred_means.mean() / true_increments.mean()
        assert 0.01 < ratio < 100, (
            f"Predicted/true ratio {ratio:.3f} out of range"
        )
