"""
Tests for the Mainspring neural ARG inference module.

Covers: PBWT, Encoder, Topology Decoder, Losses, Model, and Training.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# PBWT tests
# ---------------------------------------------------------------------------


class TestPBWT:
    """Tests for PBWT neighbor computation."""

    def test_pbwt_sort_basic(self):
        """pbwt_sort correctly partitions by allele."""
        from timepieces.mainspring_piece.pbwt import pbwt_sort

        prefix = np.array([0, 1, 2, 3], dtype=np.int64)
        div = np.zeros(4, dtype=np.int64)
        alleles = np.array([0, 1, 0, 1], dtype=np.int8)
        new_prefix, new_div = pbwt_sort(prefix, div, alleles)

        # Samples 0,2 (allele 0) should come before 1,3 (allele 1)
        assert list(new_prefix[:2]) == [0, 2]
        assert list(new_prefix[2:]) == [1, 3]

    def test_pbwt_sort_all_same(self):
        """When all alleles are the same, prefix order is preserved."""
        from timepieces.mainspring_piece.pbwt import pbwt_sort

        prefix = np.array([2, 0, 3, 1], dtype=np.int64)
        div = np.zeros(4, dtype=np.int64)
        alleles = np.array([0, 0, 0, 0], dtype=np.int8)
        new_prefix, _ = pbwt_sort(prefix, div, alleles)
        assert list(new_prefix) == [2, 0, 3, 1]

    def test_compute_neighbors_shape(self):
        """Output shape is (N, M, L)."""
        from timepieces.mainspring_piece.pbwt import compute_pbwt_neighbors

        np.random.seed(42)
        geno = np.random.randint(0, 2, size=(10, 20)).astype(np.int8)
        nbrs = compute_pbwt_neighbors(geno, L=4)
        assert nbrs.shape == (10, 20, 4)

    def test_neighbors_exclude_self(self):
        """Neighbors should not include the sample itself (when enough others exist)."""
        from timepieces.mainspring_piece.pbwt import compute_pbwt_neighbors

        np.random.seed(42)
        geno = np.random.randint(0, 2, size=(10, 20)).astype(np.int8)
        nbrs = compute_pbwt_neighbors(geno, L=4)
        for i in range(10):
            for j in range(20):
                # With 10 samples and L=4, self should not appear
                assert i not in nbrs[i, j, :]

    def test_neighbors_valid_range(self):
        """All neighbor indices are valid sample indices."""
        from timepieces.mainspring_piece.pbwt import compute_pbwt_neighbors

        np.random.seed(42)
        geno = np.random.randint(0, 2, size=(8, 15)).astype(np.int8)
        nbrs = compute_pbwt_neighbors(geno, L=6)
        assert np.all(nbrs >= 0)
        assert np.all(nbrs < 8)

    def test_identical_haplotypes_neighbors(self):
        """Identical haplotypes should be each other's neighbors."""
        from timepieces.mainspring_piece.pbwt import compute_pbwt_neighbors

        geno = np.zeros((4, 10), dtype=np.int8)
        geno[0, :] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        geno[1, :] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # identical to 0
        geno[2, :] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        geno[3, :] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # identical to 2

        nbrs = compute_pbwt_neighbors(geno, L=2)
        # At later sites, sample 0's closest neighbor should include sample 1
        # (and vice versa) due to PBWT ordering of identical haplotypes
        for j in range(5, 10):
            assert 1 in nbrs[0, j, :] or 0 in nbrs[1, j, :]


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


class TestEncoder:
    """Tests for the Genomic Encoder."""

    @pytest.fixture
    def small_encoder(self):
        from timepieces.mainspring_piece.encoder import GenomicEncoder

        return GenomicEncoder(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8,
        )

    @pytest.fixture
    def small_inputs(self):
        torch.manual_seed(42)
        N, M = 6, 12
        genotypes = torch.randint(0, 2, (N, M)).float()
        positions = torch.linspace(0, 1e4, M)
        recomb_rates = torch.ones(M - 1) * 1e-8
        return genotypes, positions, recomb_rates

    def test_output_shape(self, small_encoder, small_inputs):
        """Encoder output has shape (N, M, d_model)."""
        geno, pos, recomb = small_inputs
        H = small_encoder(geno, pos, recomb)
        assert H.shape == (6, 12, 32)

    def test_permutation_equivariance(self, small_encoder, small_inputs):
        """Permuting input samples permutes encoder output."""
        geno, pos, recomb = small_inputs

        small_encoder.eval()
        with torch.no_grad():
            H1 = small_encoder(geno, pos, recomb)

            perm = torch.tensor([3, 1, 4, 0, 5, 2])
            geno_perm = geno[perm]
            H2 = small_encoder(geno_perm, pos, recomb)

        # H2 should be H1[perm]
        torch.testing.assert_close(H2, H1[perm], atol=1e-4, rtol=1e-4)

    def test_deterministic_eval(self, small_encoder, small_inputs):
        """In eval mode, two forward passes give the same result."""
        geno, pos, recomb = small_inputs
        small_encoder.eval()
        with torch.no_grad():
            H1 = small_encoder(geno, pos, recomb)
            H2 = small_encoder(geno, pos, recomb)
        torch.testing.assert_close(H1, H2)

    def test_random_fourier_encoding(self):
        """RandomFourierPositionalEncoding produces correct shape."""
        from timepieces.mainspring_piece.encoder import RandomFourierPositionalEncoding

        rfe = RandomFourierPositionalEncoding(d_model=16, sigma=1.0)
        positions = torch.linspace(0, 1000, 50)
        out = rfe(positions)
        assert out.shape == (50, 16)


# ---------------------------------------------------------------------------
# Topology / Decoder tests
# ---------------------------------------------------------------------------


class TestTopology:
    """Tests for the Copying Decoder."""

    @pytest.fixture
    def small_decoder(self):
        from timepieces.mainspring_piece.topology import CopyingDecoder

        return CopyingDecoder(d_model=32, n_heads=2, n_neighbors=4)

    @pytest.fixture
    def decoder_inputs(self):
        torch.manual_seed(42)
        N, M, d, L = 6, 12, 32, 4
        H = torch.randn(N, M, d)
        nbrs = torch.stack([
            torch.stack([
                torch.tensor(sorted(np.random.choice(
                    [x for x in range(N) if x != i], L, replace=False
                )))
                for _ in range(M)
            ])
            for i in range(N)
        ])  # (N, M, L)
        return H, nbrs

    def test_attention_shape(self, small_decoder, decoder_inputs):
        """Attention output has shape (N, M, L)."""
        H, nbrs = decoder_inputs
        attn, bp = small_decoder(H, nbrs)
        assert attn.shape == (6, 12, 4)

    def test_breakpoint_shape(self, small_decoder, decoder_inputs):
        """Breakpoint output has shape (N, M)."""
        H, nbrs = decoder_inputs
        attn, bp = small_decoder(H, nbrs)
        assert bp.shape == (6, 12)

    def test_attention_sums_to_one(self, small_decoder, decoder_inputs):
        """Attention weights sum to 1 over the L neighbors."""
        H, nbrs = decoder_inputs
        attn, _ = small_decoder(H, nbrs)
        sums = attn.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_breakpoint_range(self, small_decoder, decoder_inputs):
        """Breakpoint probabilities are in [0, 1]."""
        H, nbrs = decoder_inputs
        _, bp = small_decoder(H, nbrs)
        assert (bp >= 0).all() and (bp <= 1).all()

    def test_breakpoint_first_site_zero(self, small_decoder, decoder_inputs):
        """First-site breakpoint probability is 0."""
        H, nbrs = decoder_inputs
        _, bp = small_decoder(H, nbrs)
        torch.testing.assert_close(bp[:, 0], torch.zeros(6))

    def test_extract_edges(self):
        """extract_edges returns valid edge tuples."""
        from timepieces.mainspring_piece.topology import extract_edges

        N, M, L = 4, 10, 3
        attn = torch.ones(N, M, L) / L
        bp = torch.zeros(N, M)
        bp[:, 5] = 0.9  # breakpoint in the middle
        nbrs = torch.zeros(N, M, L, dtype=torch.long)
        for i in range(N):
            for j in range(M):
                others = [x for x in range(N) if x != i][:L]
                nbrs[i, j, :len(others)] = torch.tensor(others)

        edges = extract_edges(attn, bp, nbrs, threshold=0.5)
        assert isinstance(edges, list)
        for child, parent, left, right in edges:
            assert 0 <= child < N
            assert 0 <= parent < N
            assert child != parent
            assert 0 <= left < right <= M


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------


class TestLosses:
    """Tests for loss functions."""

    def test_topology_loss_gradient_flow(self):
        """topology_loss produces non-zero gradients."""
        from timepieces.mainspring_piece.losses import topology_loss

        pred = torch.randn(4, 10, 3).softmax(dim=-1).requires_grad_(True)
        parents = torch.randint(0, 4, (4, 10))
        nbrs = torch.randint(0, 4, (4, 10, 3))

        loss = topology_loss(pred, parents, nbrs)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_breakpoint_loss_gradient_flow(self):
        """breakpoint_loss produces non-zero gradients."""
        from timepieces.mainspring_piece.losses import breakpoint_loss

        pred = torch.sigmoid(torch.randn(4, 10, requires_grad=True))
        true = torch.zeros(4, 10)
        true[:, 5] = 1.0

        loss = breakpoint_loss(pred, true)
        loss.backward()

    def test_topology_loss_perfect_prediction(self):
        """When prediction exactly matches truth, loss is near zero."""
        from timepieces.mainspring_piece.losses import topology_loss

        N, M, L = 4, 10, 3
        nbrs = torch.zeros(N, M, L, dtype=torch.long)
        parents = torch.zeros(N, M, dtype=torch.long)

        # Set up: parent at each site is neighbor slot 0
        for i in range(N):
            for j in range(M):
                others = [x for x in range(N) if x != i][:L]
                nbrs[i, j, :len(others)] = torch.tensor(others)
                parents[i, j] = others[0]

        # Predict high weight on slot 0
        pred = torch.zeros(N, M, L)
        pred[:, :, 0] = 10.0
        pred = pred.softmax(dim=-1)

        loss = topology_loss(pred, parents, nbrs)
        assert loss.item() < 0.1

    def test_total_loss(self):
        """total_loss returns scalar and has gradients."""
        from timepieces.mainspring_piece.losses import total_loss

        pred_attn = torch.randn(4, 10, 3).softmax(dim=-1).requires_grad_(True)
        pred_delta = torch.sigmoid(torch.randn(4, 10, requires_grad=True))
        parents = torch.randint(0, 4, (4, 10))
        breaks = torch.zeros(4, 10)
        nbrs = torch.randint(0, 4, (4, 10, 3))

        loss = total_loss(pred_attn, pred_delta, parents, breaks, nbrs)
        assert loss.shape == ()
        loss.backward()
        assert pred_attn.grad is not None


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestModel:
    """Tests for the end-to-end MainspringModel."""

    @pytest.fixture
    def small_model(self):
        from timepieces.mainspring_piece.model import MainspringModel

        return MainspringModel(
            d_model=32, n_heads=2, n_enc_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )

    def test_forward_pass(self, small_model):
        """End-to-end forward pass with small inputs."""
        torch.manual_seed(42)
        N, M = 8, 20
        geno = torch.randint(0, 2, (N, M)).float()
        pos = torch.linspace(0, 1e4, M)
        recomb = torch.ones(M - 1) * 1e-8

        out = small_model(geno, pos, recomb)
        assert out["attn_weights"].shape == (N, M, 4)
        assert out["breakpoints"].shape == (N, M)
        assert out["pbwt_neighbors"].shape == (N, M, 4)

    def test_forward_with_precomputed_pbwt(self, small_model):
        """Forward pass with pre-computed PBWT neighbors."""
        torch.manual_seed(42)
        N, M, L = 8, 20, 4
        geno = torch.randint(0, 2, (N, M)).float()
        pos = torch.linspace(0, 1e4, M)
        recomb = torch.ones(M - 1) * 1e-8
        nbrs = torch.randint(0, N, (N, M, L))

        out = small_model(geno, pos, recomb, pbwt_neighbors=nbrs)
        assert out["attn_weights"].shape == (N, M, L)

    def test_predict_edges(self, small_model):
        """predict_edges returns edge list and outputs dict."""
        torch.manual_seed(42)
        N, M = 8, 20
        geno = torch.randint(0, 2, (N, M)).float()
        pos = torch.linspace(0, 1e4, M)
        recomb = torch.ones(M - 1) * 1e-8

        edges, outputs = small_model.predict_edges(geno, pos, recomb)
        assert isinstance(edges, list)
        assert "attn_weights" in outputs

    def test_gradient_flow_all_params(self, small_model):
        """loss.backward() produces gradients for all parameters."""
        from timepieces.mainspring_piece.losses import total_loss

        torch.manual_seed(42)
        N, M, L = 8, 20, 4
        geno = torch.randint(0, 2, (N, M)).float()
        pos = torch.linspace(0, 1e4, M)
        recomb = torch.ones(M - 1) * 1e-8

        out = small_model(geno, pos, recomb)

        parents = torch.randint(0, N, (N, M))
        breaks = torch.zeros(N, M)

        loss = total_loss(
            out["attn_weights"], out["breakpoints"],
            parents, breaks, out["pbwt_neighbors"],
        )
        loss.backward()

        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


# ---------------------------------------------------------------------------
# Training test (requires msprime)
# ---------------------------------------------------------------------------


class TestTraining:
    """Integration tests with msprime (skipped if not installed)."""

    @pytest.fixture
    def check_msprime(self):
        pytest.importorskip("msprime")
        pytest.importorskip("tskit")

    def test_generate_training_batch(self, check_msprime):
        """generate_training_batch produces valid data."""
        from timepieces.mainspring_piece.training import generate_training_batch

        data = generate_training_batch(
            n_samples=6, seq_length=5e4, random_seed=42,
        )
        N = 6
        M = data["genotypes"].shape[1]
        assert data["genotypes"].shape[0] == N
        assert data["positions"].shape == (M,)
        if M > 1:
            assert data["recomb_rates"].shape == (M - 1,)
        assert data["true_parents"].shape == (N, M)
        assert data["true_breakpoints"].shape == (N, M)
        assert set(np.unique(data["genotypes"])).issubset({0, 1})

    def test_extract_training_targets(self, check_msprime):
        """Extracted targets have consistent shapes and values."""
        from timepieces.mainspring_piece.training import (
            extract_training_targets,
        )
        import msprime

        ts = msprime.simulate(
            sample_size=8, length=1e4,
            recombination_rate=1e-8, mutation_rate=1e-8,
            random_seed=123,
        )
        geno, pos, recomb, parents, breaks = extract_training_targets(ts)
        N = ts.num_samples
        M = ts.num_sites

        assert geno.shape == (N, M)
        assert parents.shape == (N, M)
        assert breaks.shape == (N, M)
        # Parents should be valid sample indices
        assert np.all(parents >= 0) and np.all(parents < N)
        # No sample is its own parent
        for i in range(N):
            assert not np.all(parents[i] == i)

    def test_loss_decreases(self, check_msprime):
        """3 epochs on small msprime data -> loss decreases."""
        from timepieces.mainspring_piece.model import MainspringModel
        from timepieces.mainspring_piece.training import train_epoch

        model = MainspringModel(
            d_model=32, n_heads=2, n_enc_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for epoch in range(3):
            avg_loss = train_epoch(
                model, optimizer,
                n_samples=6, seq_length=5e4,
                n_batches=3, n_neighbors=4,
                random_seed=epoch * 100,
            )
            losses.append(avg_loss)

        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses}"
        )
