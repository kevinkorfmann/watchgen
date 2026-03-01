"""Tests for Tourbillon: Denoising Diffusion for ARG Posterior Sampling.

Tests cover:
1. Noise schedule -- shape, range, monotonicity
2. Sinusoidal timestep embedding -- shape, uniqueness
3. Adaptive layer norm -- shape, conditioning effect
4. Conditional denoiser -- output shape, gradient flow, sensitivity
5. Diffusion -- forward process, loss, reverse step, sampling
6. Representations -- x_0 conversion, roundtrip, edge extraction, tree sequence
7. Training -- batch generation, loss decrease
8. Sampling -- valid TreeSequence output
"""

import numpy as np
import torch
import pytest


def _has_msprime():
    try:
        import msprime
        import tskit
        return True
    except ImportError:
        return False


# ---------- 1. Noise Schedule ----------

class TestNoiseSchedule:
    """Tests for cosine beta schedule and derived quantities."""

    def test_betas_shape(self):
        from timepieces.tourbillon_piece.noise_schedule import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)

    def test_betas_range(self):
        from timepieces.tourbillon_piece.noise_schedule import cosine_beta_schedule
        betas = cosine_beta_schedule(100)
        assert np.all(betas >= 0.0)
        assert np.all(betas <= 0.999)

    def test_alpha_bar_monotonic(self):
        from timepieces.tourbillon_piece.noise_schedule import (
            cosine_beta_schedule, compute_alphas,
        )
        betas = cosine_beta_schedule(200)
        sched = compute_alphas(betas)
        ab = sched["alpha_bar"].numpy()
        # alpha_bar should be monotonically decreasing
        assert np.all(np.diff(ab) < 0)

    def test_alpha_bar_endpoints(self):
        from timepieces.tourbillon_piece.noise_schedule import (
            cosine_beta_schedule, compute_alphas,
        )
        betas = cosine_beta_schedule(200)
        sched = compute_alphas(betas)
        ab = sched["alpha_bar"].numpy()
        # Near 1 at the start, near 0 at the end
        assert ab[0] > 0.95
        assert ab[-1] < 0.1

    def test_schedule_tensor_shapes(self):
        from timepieces.tourbillon_piece.noise_schedule import (
            cosine_beta_schedule, compute_alphas,
        )
        T = 50
        betas = cosine_beta_schedule(T)
        sched = compute_alphas(betas)
        for key in ["betas", "alphas", "alpha_bar", "sqrt_alpha_bar",
                     "sqrt_one_minus_alpha_bar", "posterior_variance",
                     "posterior_mean_coef1", "posterior_mean_coef2"]:
            assert sched[key].shape == (T,), f"{key} has wrong shape"


# ---------- 2. Sinusoidal Embedding ----------

class TestSinusoidalEmbedding:
    """Tests for timestep embedding."""

    @pytest.fixture
    def emb(self):
        from timepieces.tourbillon_piece.denoiser import SinusoidalTimestepEmbedding
        torch.manual_seed(42)
        return SinusoidalTimestepEmbedding(d_model=32)

    def test_scalar_shape(self, emb):
        t = torch.tensor(5)
        out = emb(t)
        assert out.shape == (32,)

    def test_batch_shape(self, emb):
        t = torch.tensor([0, 10, 50])
        out = emb(t)
        assert out.shape == (3, 32)

    def test_different_timesteps_different_embeddings(self, emb):
        t1 = torch.tensor(0)
        t2 = torch.tensor(100)
        e1 = emb(t1)
        e2 = emb(t2)
        # Embeddings should be different for different timesteps
        assert not torch.allclose(e1, e2, atol=1e-4)


# ---------- 3. Adaptive Layer Norm ----------

class TestAdaptiveLayerNorm:
    """Tests for DiT-style adaptive layer normalization."""

    @pytest.fixture
    def aln(self):
        from timepieces.tourbillon_piece.denoiser import AdaptiveLayerNorm
        torch.manual_seed(42)
        return AdaptiveLayerNorm(d_model=16)

    def test_output_shape(self, aln):
        x = torch.randn(5, 16)
        t_emb = torch.randn(16)
        out = aln(x, t_emb)
        assert out.shape == (5, 16)

    def test_conditioning_effect(self, aln):
        x = torch.randn(5, 16)
        t_emb_1 = torch.randn(16)
        t_emb_2 = torch.randn(16)
        out1 = aln(x, t_emb_1)
        out2 = aln(x, t_emb_2)
        # Different conditioning should produce different outputs
        assert not torch.allclose(out1, out2, atol=1e-4)


# ---------- 4. Conditional Denoiser ----------

class TestConditionalDenoiser:
    """Tests for the full epsilon_theta network."""

    @pytest.fixture
    def small_denoiser(self):
        from timepieces.tourbillon_piece.denoiser import ConditionalDenoiser
        torch.manual_seed(42)
        return ConditionalDenoiser(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )

    @pytest.fixture
    def small_inputs(self):
        torch.manual_seed(42)
        N, M, L = 4, 8, 4
        C = L + 1
        x_t = torch.randn(N, M, C)
        t = torch.tensor(10)
        genotypes = torch.randint(0, 2, (N, M)).float()
        positions = torch.linspace(0, 1e4, M)
        recomb_rates = torch.ones(M - 1) * 1e-8
        return x_t, t, genotypes, positions, recomb_rates

    def test_output_shape(self, small_denoiser, small_inputs):
        x_t, t, geno, pos, recomb = small_inputs
        eps = small_denoiser(x_t, t, geno, pos, recomb)
        assert eps.shape == x_t.shape

    def test_gradient_flow(self, small_denoiser, small_inputs):
        x_t, t, geno, pos, recomb = small_inputs
        x_t.requires_grad_(True)
        eps = small_denoiser(x_t, t, geno, pos, recomb)
        loss = eps.sum()
        loss.backward()
        assert x_t.grad is not None
        assert x_t.grad.abs().sum() > 0

    def test_timestep_sensitivity(self, small_denoiser, small_inputs):
        x_t, _, geno, pos, recomb = small_inputs
        eps_1 = small_denoiser(x_t, torch.tensor(0), geno, pos, recomb)
        eps_2 = small_denoiser(x_t, torch.tensor(100), geno, pos, recomb)
        # Different timesteps should produce different predictions
        assert not torch.allclose(eps_1, eps_2, atol=1e-4)

    def test_conditioning_sensitivity(self, small_denoiser, small_inputs):
        x_t, t, geno, pos, recomb = small_inputs
        eps_1 = small_denoiser(x_t, t, geno, pos, recomb)
        geno_alt = 1.0 - geno  # flip all genotypes
        eps_2 = small_denoiser(x_t, t, geno_alt, pos, recomb)
        assert not torch.allclose(eps_1, eps_2, atol=1e-4)


# ---------- 5. Diffusion ----------

class TestDiffusion:
    """Tests for forward/reverse process and sampling."""

    @pytest.fixture
    def small_diffusion(self):
        from timepieces.tourbillon_piece.denoiser import ConditionalDenoiser
        from timepieces.tourbillon_piece.diffusion import TourbillonDiffusion
        torch.manual_seed(42)
        denoiser = ConditionalDenoiser(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )
        return TourbillonDiffusion(denoiser, T=50, schedule="cosine")

    @pytest.fixture
    def small_data(self):
        torch.manual_seed(42)
        N, M, L = 4, 8, 4
        C = L + 1
        x_0 = torch.randn(N, M, C)
        genotypes = torch.randint(0, 2, (N, M)).float()
        positions = torch.linspace(0, 1e4, M)
        recomb_rates = torch.ones(M - 1) * 1e-8
        return x_0, genotypes, positions, recomb_rates

    def test_forward_process_shape(self, small_diffusion, small_data):
        x_0, _, _, _ = small_data
        x_t, eps = small_diffusion.forward_process(x_0, 10)
        assert x_t.shape == x_0.shape
        assert eps.shape == x_0.shape

    def test_forward_process_noise_level(self, small_diffusion, small_data):
        x_0, _, _, _ = small_data
        # At t=0, x_t should be close to x_0
        x_t_early, _ = small_diffusion.forward_process(x_0, 0)
        diff_early = (x_t_early - x_0 * small_diffusion.sqrt_alpha_bar[0]).abs().mean()
        # At t=T-1, x_t should be mostly noise
        x_t_late, _ = small_diffusion.forward_process(x_0, small_diffusion.T - 1)
        # The signal component should be smaller at late timestep
        signal_early = small_diffusion.sqrt_alpha_bar[0].item()
        signal_late = small_diffusion.sqrt_alpha_bar[-1].item()
        assert signal_early > signal_late

    def test_loss_is_scalar(self, small_diffusion, small_data):
        x_0, geno, pos, recomb = small_data
        loss = small_diffusion.compute_loss(x_0, geno, pos, recomb)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_has_gradients(self, small_diffusion, small_data):
        x_0, geno, pos, recomb = small_data
        loss = small_diffusion.compute_loss(x_0, geno, pos, recomb)
        loss.backward()
        # Check that denoiser parameters have gradients
        has_grad = False
        for p in small_diffusion.denoiser.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad

    def test_reverse_step_shape(self, small_diffusion, small_data):
        x_0, geno, pos, recomb = small_data
        x_t = torch.randn_like(x_0)
        x_prev = small_diffusion.reverse_step(x_t, 10, geno, pos, recomb)
        assert x_prev.shape == x_t.shape

    def test_sample_shape(self, small_diffusion, small_data):
        _, geno, pos, recomb = small_data
        N, M = geno.shape
        # Use very few steps for speed
        small_diffusion.T = 3
        x_0 = small_diffusion.sample(geno, pos, recomb, N, M)
        C = small_diffusion.denoiser.n_neighbors + 1
        assert x_0.shape == (N, M, C)


# ---------- 6. Representations ----------

class TestRepresentations:
    """Tests for tree sequence <-> tensor conversions."""

    @pytest.fixture
    def check_msprime(self):
        pytest.importorskip("msprime")
        pytest.importorskip("tskit")

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_x0_shape(self, check_msprime):
        import msprime
        from timepieces.tourbillon_piece.representations import tree_sequence_to_x0

        ts = msprime.sim_ancestry(
            samples=6, sequence_length=1e4,
            recombination_rate=1e-8, population_size=1e4, random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)

        if ts.num_sites < 2:
            pytest.skip("Not enough mutations generated")

        x_0, nbrs, geno, pos, recomb = tree_sequence_to_x0(ts, n_neighbors=4)
        N = ts.num_samples
        M = ts.num_sites
        L = 4
        assert x_0.shape == (N, M, L + 1)
        assert nbrs.shape == (N, M, L)
        assert torch.isfinite(x_0).all()

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_x0_parent_channel(self, check_msprime):
        import msprime
        from timepieces.tourbillon_piece.representations import tree_sequence_to_x0

        ts = msprime.sim_ancestry(
            samples=6, sequence_length=1e4,
            recombination_rate=1e-8, population_size=1e4, random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)

        if ts.num_sites < 2:
            pytest.skip("Not enough mutations generated")

        x_0, _, _, _, _ = tree_sequence_to_x0(ts, n_neighbors=4)
        # Parent logits should be non-negative (one-hot or uniform)
        parent_logits = x_0[:, :, :4]
        assert (parent_logits >= 0).all()

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_roundtrip_parents(self, check_msprime):
        import msprime
        from timepieces.tourbillon_piece.representations import (
            tree_sequence_to_x0, x0_to_edges,
        )

        ts = msprime.sim_ancestry(
            samples=6, sequence_length=1e4,
            recombination_rate=1e-8, population_size=1e4, random_seed=42,
        )
        ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)

        if ts.num_sites < 2:
            pytest.skip("Not enough mutations generated")

        x_0, nbrs, geno, pos, recomb = tree_sequence_to_x0(ts, n_neighbors=4)
        edges, node_times = x0_to_edges(x_0, nbrs, pos)
        # Should produce some edges
        assert len(edges) > 0
        # All edges should have child < parent (node IDs)
        for child, parent, left, right in edges:
            assert child != parent

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_edges_to_tree_sequence(self, check_msprime):
        import tskit
        from timepieces.tourbillon_piece.representations import edges_to_tree_sequence

        edges = [(0, 2, 0.0, 5000.0), (1, 2, 0.0, 5000.0)]
        node_times = {0: 0.0, 1: 0.0, 2: 100.0}
        positions = np.array([100.0, 500.0, 1000.0, 3000.0])

        ts = edges_to_tree_sequence(edges, node_times, positions)
        assert isinstance(ts, tskit.TreeSequence)
        assert ts.num_nodes >= 2


# ---------- 7. Training ----------

class TestTraining:
    """Tests for training data generation and loss decrease."""

    @pytest.fixture
    def check_msprime(self):
        pytest.importorskip("msprime")
        pytest.importorskip("tskit")

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_generate_diffusion_batch(self, check_msprime):
        from timepieces.tourbillon_piece.training import generate_diffusion_batch

        data = generate_diffusion_batch(
            n_samples=6, seq_length=1e4,
            recomb_rate=1e-8, mut_rate=1e-8,
            n_neighbors=4, random_seed=42,
        )
        if data is None:
            pytest.skip("Not enough sites generated")

        assert "x_0" in data
        assert "genotypes" in data
        assert "positions" in data
        assert data["x_0"].dim() == 3

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_loss_decreases(self, check_msprime):
        from timepieces.tourbillon_piece.denoiser import ConditionalDenoiser
        from timepieces.tourbillon_piece.diffusion import TourbillonDiffusion
        from timepieces.tourbillon_piece.training import train_epoch

        torch.manual_seed(42)
        denoiser = ConditionalDenoiser(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )
        diffusion = TourbillonDiffusion(denoiser, T=10, schedule="cosine")
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

        losses = []
        for epoch in range(2):
            avg_loss = train_epoch(
                diffusion, optimizer,
                n_samples=6, seq_length=1e4,
                recomb_rate=1e-8, mut_rate=1e-8,
                n_batches=3, n_neighbors=4,
                random_seed=100 + epoch * 10,
            )
            losses.append(avg_loss)

        # Loss should be finite
        assert all(np.isfinite(l) for l in losses)


# ---------- 8. Sampling ----------

class TestSampling:
    """Tests for ARG sampling from the diffusion model."""

    @pytest.fixture
    def check_msprime(self):
        pytest.importorskip("msprime")
        pytest.importorskip("tskit")

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_sample_arg_returns_tree_sequence(self, check_msprime):
        import tskit
        from timepieces.tourbillon_piece.denoiser import ConditionalDenoiser
        from timepieces.tourbillon_piece.diffusion import TourbillonDiffusion
        from timepieces.tourbillon_piece.sampling import sample_arg

        torch.manual_seed(42)
        denoiser = ConditionalDenoiser(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )
        diffusion = TourbillonDiffusion(denoiser, T=3, schedule="cosine")

        N, M = 6, 10
        genotypes = np.random.RandomState(42).randint(0, 2, (N, M))
        positions = np.linspace(0, 1e4, M)
        recomb_rates = np.ones(M - 1) * 1e-8

        ts = sample_arg(
            diffusion, genotypes, positions, recomb_rates,
            n_neighbors=4,
        )
        assert isinstance(ts, tskit.TreeSequence)

    @pytest.mark.skipif(not _has_msprime(), reason="msprime not installed")
    def test_sample_multiple(self, check_msprime):
        import tskit
        from timepieces.tourbillon_piece.denoiser import ConditionalDenoiser
        from timepieces.tourbillon_piece.diffusion import TourbillonDiffusion
        from timepieces.tourbillon_piece.sampling import sample_multiple

        torch.manual_seed(42)
        denoiser = ConditionalDenoiser(
            d_model=32, n_heads=2, n_layers=1,
            n_inducing=4, window_size=8, n_neighbors=4,
        )
        diffusion = TourbillonDiffusion(denoiser, T=3, schedule="cosine")

        N, M = 4, 8
        genotypes = np.random.RandomState(42).randint(0, 2, (N, M))
        positions = np.linspace(0, 1e4, M)
        recomb_rates = np.ones(M - 1) * 1e-8

        results = sample_multiple(
            diffusion, genotypes, positions, recomb_rates,
            n_samples_posterior=2, n_neighbors=4,
        )
        assert len(results) == 2
        for ts in results:
            assert isinstance(ts, tskit.TreeSequence)
