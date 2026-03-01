"""
Tests for watchgen.mini_discoal module.

Imports all functions from watchgen.mini_discoal and tests their
mathematical properties. Adapted from tests/test_timepieces_discoal.py.

Covers:
- allele_trajectory.rst: deterministic_trajectory, fixation_probability,
  stochastic_trajectory, sweep_duration_table
- structured_coalescent.rst: coalescence_rate, migration_rates,
  structured_coalescent_sweep, escape_probability
- sweep_types.rst: expected_independent_origins
- msprime_comparison.rst: discoal_to_msprime, msprime_to_discoal
"""

import numpy as np
import pytest

from watchgen.mini_discoal import (
    deterministic_trajectory,
    fixation_probability,
    stochastic_trajectory,
    sweep_duration_table,
    coalescence_rate,
    migration_rates,
    structured_coalescent_sweep,
    escape_probability,
    expected_independent_origins,
    discoal_to_msprime,
    msprime_to_discoal,
)


# ============================================================================
# Test classes: allele_trajectory.rst
# ============================================================================

class TestDeterministicTrajectory:
    """Tests for the deterministic (logistic) allele frequency trajectory."""

    def test_starts_at_x0(self):
        """Trajectory should start at x0 = 1/(2N) by default."""
        N = 1000
        s = 0.01
        traj = deterministic_trajectory(s, N)
        assert traj[0] == pytest.approx(1.0 / (2 * N), rel=1e-6)

    def test_ends_at_fixation(self):
        """Trajectory should end at 1.0."""
        traj = deterministic_trajectory(0.01, 1000)
        assert traj[-1] == 1.0

    def test_monotonically_increasing(self):
        """Trajectory should be strictly increasing."""
        traj = deterministic_trajectory(0.01, 1000)
        diffs = np.diff(traj)
        assert np.all(diffs >= 0), "Trajectory should be non-decreasing"

    def test_bounded_zero_one(self):
        """All frequencies should be in [0, 1]."""
        traj = deterministic_trajectory(0.05, 5000)
        assert np.all(traj >= 0)
        assert np.all(traj <= 1.0)

    def test_custom_x0(self):
        """Trajectory with custom initial frequency."""
        traj = deterministic_trajectory(0.01, 1000, x0=0.1)
        assert traj[0] == pytest.approx(0.1, rel=1e-6)
        assert traj[-1] == 1.0

    def test_stronger_selection_shorter_sweep(self):
        """Stronger selection should produce a shorter sweep."""
        N = 5000
        traj_weak = deterministic_trajectory(0.001, N)
        traj_strong = deterministic_trajectory(0.05, N)
        assert len(traj_strong) < len(traj_weak)

    def test_midpoint_approximately_logistic(self):
        """At t = ln((1-x0)/x0) / s, frequency should be ~0.5."""
        N = 10000
        s = 0.01
        x0 = 1.0 / (2 * N)
        t_half = np.log((1 - x0) / x0) / s
        x_half = 1.0 / (1.0 + ((1 - x0) / x0) * np.exp(-s * t_half))
        assert x_half == pytest.approx(0.5, abs=0.01)

    def test_sweep_duration_scales_with_log_N(self):
        """Sweep duration should scale approximately as 2*ln(2N)/s."""
        s = 0.01
        for N in [1000, 5000, 10000]:
            traj = deterministic_trajectory(s, N)
            expected_T = 2 * np.log(2 * N) / s
            # Allow 20% tolerance
            assert len(traj) == pytest.approx(expected_T, rel=0.2)


class TestFixationProbability:
    """Tests for the fixation probability function h(x)."""

    def test_boundary_zero(self):
        """h(0) should be 0."""
        assert fixation_probability(0, 100) == pytest.approx(0, abs=1e-10)

    def test_boundary_one(self):
        """h(1) should be 1."""
        assert fixation_probability(1, 100) == pytest.approx(1, abs=1e-6)

    def test_neutral_case(self):
        """Under neutrality (2Ns=0), h(x) = x."""
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert fixation_probability(x, 0) == pytest.approx(x, abs=1e-6)

    def test_monotonically_increasing(self):
        """h(x) should increase with x for positive selection."""
        xs = np.linspace(0.01, 0.99, 50)
        hs = [fixation_probability(x, 100) for x in xs]
        assert all(hs[i] <= hs[i+1] for i in range(len(hs)-1))

    def test_strong_selection_new_mutation(self):
        """For strong selection, h(1/(2N)) ~ s."""
        N = 10000
        s = 0.01
        two_N_s = 2 * N * s
        x = 1.0 / (2 * N)
        h = fixation_probability(x, two_N_s)
        # For strong selection: h(1/(2N)) ~ 2s (diploid) or s (haploid-like)
        assert h == pytest.approx(s, rel=0.5)  # within 50% for approximation

    def test_positive_selection_above_neutral(self):
        """Under positive selection, h(x) > x for small x."""
        for x in [0.01, 0.05, 0.1]:
            h_sel = fixation_probability(x, 100)
            assert h_sel > x, \
                f"h({x}) = {h_sel} should be > {x} under positive selection"

    def test_symmetry_with_sign_flip(self):
        """h(x, 2Ns) + h(1-x, -2Ns) should equal 1 (probability complement)."""
        for x in [0.1, 0.3, 0.5]:
            h_pos = fixation_probability(x, 100)
            h_neg = fixation_probability(1 - x, -100)
            assert h_pos + h_neg == pytest.approx(1.0, abs=0.01)


class TestStochasticTrajectory:
    """Tests for the stochastic (conditioned) trajectory."""

    def test_starts_near_x0(self):
        """Stochastic trajectory should start near 1/(2N)."""
        N = 2000
        s = 0.02
        traj = stochastic_trajectory(s, N, rng=np.random.default_rng(42))
        assert traj[0] <= 0.05  # generous bound for jump process

    def test_ends_at_fixation(self):
        """Stochastic trajectory should end at or near 1.0."""
        N = 2000
        s = 0.02
        traj = stochastic_trajectory(s, N, rng=np.random.default_rng(42))
        assert traj[-1] >= 0.95

    def test_bounded_zero_one(self):
        """All frequencies should be in [0, 1]."""
        N = 2000
        s = 0.02
        traj = stochastic_trajectory(s, N, rng=np.random.default_rng(42))
        assert np.all(traj >= 0)
        assert np.all(traj <= 1.0)

    def test_reproducible_with_seed(self):
        """Same seed should produce same trajectory."""
        N = 2000
        s = 0.02
        traj1 = stochastic_trajectory(s, N, rng=np.random.default_rng(123))
        traj2 = stochastic_trajectory(s, N, rng=np.random.default_rng(123))
        np.testing.assert_array_equal(traj1, traj2)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different
        trajectories."""
        N = 2000
        s = 0.02
        traj1 = stochastic_trajectory(s, N, rng=np.random.default_rng(1))
        traj2 = stochastic_trajectory(s, N, rng=np.random.default_rng(2))
        if len(traj1) > 2 and len(traj2) > 2:
            assert not np.array_equal(traj1, traj2)

    def test_length_greater_than_one(self):
        """Trajectory should have more than one entry (the allele sweeps)."""
        N = 2000
        s = 0.02
        traj = stochastic_trajectory(s, N, rng=np.random.default_rng(42))
        assert len(traj) >= 2


class TestSweepDuration:
    """Tests for sweep duration calculations."""

    def test_stronger_selection_shorter(self):
        """Stronger selection should give shorter sweeps."""
        N = 10000
        results = sweep_duration_table(N, [10, 100, 1000])
        assert results[0]['T_gen'] > results[1]['T_gen'] > results[2]['T_gen']

    def test_duration_formula(self):
        """Duration should match 2*ln(2N)/s."""
        N = 10000
        for alpha in [100, 500, 1000]:
            s = alpha / (2 * N)
            expected = 2 * np.log(2 * N) / s
            results = sweep_duration_table(N, [alpha])
            assert results[0]['T_gen'] == pytest.approx(expected, rel=1e-6)

    def test_coalescent_units_conversion(self):
        """T_coal should equal T_gen / (2N)."""
        N = 10000
        results = sweep_duration_table(N, [100, 500])
        for r in results:
            assert r['T_coal'] == pytest.approx(
                r['T_gen'] / (2 * N), rel=1e-10
            )

    def test_4N_units_conversion(self):
        """T_4N should equal T_gen / (4N)."""
        N = 10000
        results = sweep_duration_table(N, [100, 500])
        for r in results:
            assert r['T_4N'] == pytest.approx(
                r['T_gen'] / (4 * N), rel=1e-10
            )


# ============================================================================
# Test classes: structured_coalescent.rst
# ============================================================================

class TestCoalescenceRate:
    """Tests for the pairwise coalescence rate function."""

    def test_zero_lineages(self):
        """No lineages means zero rate."""
        assert coalescence_rate(0, 1000) == 0.0
        assert coalescence_rate(1, 1000) == 0.0

    def test_two_lineages(self):
        """Two lineages: rate = 1 / background_size."""
        assert coalescence_rate(2, 1000) == pytest.approx(1.0 / 1000.0)

    def test_scales_with_k_choose_2(self):
        """Rate should be k(k-1)/2 divided by background size."""
        bg = 5000
        for k in [2, 5, 10, 20]:
            expected = k * (k - 1) / (2.0 * bg)
            assert coalescence_rate(k, bg) == pytest.approx(expected)

    def test_inversely_proportional_to_size(self):
        """Doubling background size should halve the rate."""
        rate1 = coalescence_rate(5, 1000)
        rate2 = coalescence_rate(5, 2000)
        assert rate1 == pytest.approx(2 * rate2)

    def test_bottleneck_speedup(self):
        """Small background should give much higher rate than full
        population."""
        two_N = 20000
        x = 0.01
        rate_bottleneck = coalescence_rate(5, two_N * x)
        rate_neutral = coalescence_rate(5, two_N)
        assert rate_bottleneck / rate_neutral == pytest.approx(1.0 / x)

    def test_zero_background_size(self):
        """Zero-size background should give zero rate (protected)."""
        assert coalescence_rate(5, 0) == 0.0


class TestMigrationRates:
    """Tests for the recombination-as-migration rate function."""

    def test_zero_recombination(self):
        """No recombination means no migration."""
        m_Bb, m_bB = migration_rates(10, 5, 0, 0.5)
        assert m_Bb == 0.0
        assert m_bB == 0.0

    def test_zero_lineages(self):
        """No lineages means no migration."""
        m_Bb, m_bB = migration_rates(0, 0, 0.001, 0.5)
        assert m_Bb == 0.0
        assert m_bB == 0.0

    def test_x_equals_half(self):
        """At x = 0.5, migration rates should be symmetric (given same n)."""
        n = 10
        r = 0.001
        m_Bb, m_bB = migration_rates(n, n, r, 0.5)
        assert m_Bb == pytest.approx(m_bB)

    def test_migration_B_to_b_scales_with_1_minus_x(self):
        """B->b migration rate should increase as x decreases."""
        n_B = 10
        r = 0.001
        rates = [migration_rates(n_B, 0, r, x)[0] for x in [0.9, 0.5, 0.1]]
        assert rates[0] < rates[1] < rates[2]

    def test_migration_b_to_B_scales_with_x(self):
        """b->B migration rate should increase as x increases."""
        n_b = 10
        r = 0.001
        rates = [migration_rates(0, n_b, r, x)[1] for x in [0.1, 0.5, 0.9]]
        assert rates[0] < rates[1] < rates[2]

    def test_rate_formula_exact(self):
        """Verify exact formula: m_Bb = n_B * r * (1-x)."""
        assert migration_rates(5, 3, 0.002, 0.3) == pytest.approx(
            (5 * 0.002 * 0.7, 3 * 0.002 * 0.3), abs=1e-10
        )


class TestEscapeProbability:
    """Tests for the approximate escape probability."""

    def test_at_selected_site(self):
        """At r = 0, no lineage escapes."""
        assert escape_probability(0, 0.01, 10000) == pytest.approx(0.0)

    def test_far_away(self):
        """Very far from selected site, nearly all lineages escape."""
        p = escape_probability(0.1, 0.01, 10000)
        assert p > 0.99

    def test_monotonically_increasing_with_r(self):
        """Escape probability should increase with recombination distance."""
        s = 0.01
        N = 10000
        r_values = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        p_values = [escape_probability(r, s, N) for r in r_values]
        assert all(p_values[i] <= p_values[i+1]
                    for i in range(len(p_values)-1))

    def test_bounded_zero_one(self):
        """Escape probability should be in [0, 1]."""
        for r in np.logspace(-8, -2, 20):
            p = escape_probability(r, 0.01, 10000)
            assert 0 <= p <= 1

    def test_stronger_selection_less_escape(self):
        """Stronger selection should reduce escape probability at given r."""
        r = 1e-5
        N = 10000
        p_weak = escape_probability(r, 0.001, N)
        p_strong = escape_probability(r, 0.1, N)
        assert p_strong < p_weak


class TestStructuredCoalescentSweep:
    """Tests for the structured coalescent sweep simulation."""

    def test_returns_correct_types(self):
        """Should return list of times and two integers."""
        traj = deterministic_trajectory(0.05, 500)
        coal_times, n_B, n_b = structured_coalescent_sweep(
            traj, 10, 0.0, 500, rng=np.random.default_rng(42)
        )
        assert isinstance(coal_times, list)
        assert isinstance(n_B, int)
        assert isinstance(n_b, int)

    def test_total_events_equals_n_minus_1(self):
        """Total coalescence events should be n-1 (all must coalesce
        eventually)."""
        N = 500
        s = 0.05
        traj = deterministic_trajectory(s, N)
        n = 10
        coal_times, n_B, n_b = structured_coalescent_sweep(
            traj, n, 0.0, N, rng=np.random.default_rng(42)
        )
        # n_B + n_b remaining lineages + len(coal_times) coalescences = n
        assert len(coal_times) + n_B + n_b == n

    def test_no_recombination_all_coalesce_in_B(self):
        """With r=0, no lineages migrate to b, all coalesce in B."""
        N = 500
        s = 0.05
        traj = deterministic_trajectory(s, N)
        coal_times, n_B, n_b = structured_coalescent_sweep(
            traj, 10, 0.0, N, rng=np.random.default_rng(42)
        )
        assert n_b == 0  # no lineages escaped to b
        assert n_B <= 1  # all coalesced (or 1 remaining)

    def test_high_recombination_some_escape(self):
        """With high r, some lineages should migrate to background b."""
        N = 500
        s = 0.01
        traj = deterministic_trajectory(s, N)
        found_escape = False
        for seed in range(100):
            _, _, n_b = structured_coalescent_sweep(
                traj, 20, 0.01, N, rng=np.random.default_rng(seed)
            )
            if n_b > 0:
                found_escape = True
                break
        assert found_escape, "Expected at least one lineage to escape to b"

    def test_reproducible(self):
        """Same seed should give same result."""
        N = 500
        traj = deterministic_trajectory(0.05, N)
        ct1, nB1, nb1 = structured_coalescent_sweep(
            traj, 10, 0.001, N, rng=np.random.default_rng(99)
        )
        ct2, nB2, nb2 = structured_coalescent_sweep(
            traj, 10, 0.001, N, rng=np.random.default_rng(99)
        )
        assert ct1 == ct2
        assert nB1 == nB2
        assert nb1 == nb2

    def test_coal_times_positive(self):
        """All coalescence times should be positive."""
        N = 500
        traj = deterministic_trajectory(0.05, N)
        coal_times, _, _ = structured_coalescent_sweep(
            traj, 10, 0.0, N, rng=np.random.default_rng(42)
        )
        assert all(t > 0 for t in coal_times)

    def test_single_lineage_no_events(self):
        """A single lineage should produce no coalescence events."""
        N = 500
        traj = deterministic_trajectory(0.05, N)
        coal_times, n_B, n_b = structured_coalescent_sweep(
            traj, 1, 0.0, N, rng=np.random.default_rng(42)
        )
        assert len(coal_times) == 0
        assert n_B + n_b == 1


# ============================================================================
# Test classes: sweep_types.rst
# ============================================================================

class TestExpectedIndependentOrigins:
    """Tests for the expected number of independent origins in a soft sweep."""

    def test_low_mutation_hard_sweep(self):
        """Very low mutation rate should give < 1 origin (hard sweep)."""
        E_K = expected_independent_origins(0.01, 10000, 1e-9)
        assert E_K < 1.0

    def test_high_mutation_soft_sweep(self):
        """High mutation rate should give > 1 origin (soft sweep)."""
        E_K = expected_independent_origins(0.01, 10000, 1e-5)
        assert E_K > 1.0

    def test_positive(self):
        """Expected origins should always be positive."""
        for s in [0.001, 0.01, 0.1]:
            for mu_a in [1e-9, 1e-7, 1e-5]:
                E_K = expected_independent_origins(s, 10000, mu_a)
                assert E_K > 0

    def test_scales_with_mutation_rate(self):
        """Doubling the mutation rate should approximately double E[K]."""
        E1 = expected_independent_origins(0.01, 10000, 1e-7)
        E2 = expected_independent_origins(0.01, 10000, 2e-7)
        assert E2 == pytest.approx(2 * E1, rel=1e-6)

    def test_stronger_selection_fewer_origins(self):
        """Stronger selection (fewer generations to sweep) -> fewer origins."""
        N = 10000
        mu_a = 1e-7
        E_weak = expected_independent_origins(0.01, N, mu_a)
        E_strong = expected_independent_origins(0.1, N, mu_a)
        assert E_strong < E_weak


# ============================================================================
# Test classes: msprime_comparison.rst
# ============================================================================

class TestParameterConversion:
    """Tests for parameter conversion between discoal and msprime."""

    def test_roundtrip_discoal_msprime(self):
        """Converting discoal -> msprime -> discoal should be identity."""
        N = 10000
        L = 100000
        theta = 50.0
        rho = 40.0
        alpha = 200.0
        n = 100

        msp = discoal_to_msprime(theta, rho, alpha, n, L, N)
        disc = msprime_to_discoal(
            msp['samples'], msp['sequence_length'],
            msp['mutation_rate'], msp['recombination_rate'],
            msp['selection_coefficient'], msp['population_size']
        )
        assert disc['theta'] == pytest.approx(theta, rel=1e-10)
        assert disc['rho'] == pytest.approx(rho, rel=1e-10)
        assert disc['alpha'] == pytest.approx(alpha, rel=1e-10)

    def test_roundtrip_msprime_discoal(self):
        """Converting msprime -> discoal -> msprime should be identity."""
        N = 10000
        L = 100000
        mu = 1.25e-8
        r = 1e-8
        s = 0.01
        n = 100

        disc = msprime_to_discoal(n, L, mu, r, s, N)
        msp = discoal_to_msprime(
            disc['theta'], disc['rho'], disc['alpha'],
            disc['n'], disc['L'], N
        )
        assert msp['mutation_rate'] == pytest.approx(mu, rel=1e-10)
        assert msp['recombination_rate'] == pytest.approx(r, rel=1e-10)
        assert msp['selection_coefficient'] == pytest.approx(s, rel=1e-10)

    def test_theta_formula(self):
        """theta = 4 * N * mu * L."""
        N = 10000
        L = 100000
        mu = 1.25e-8
        r = 1e-8
        s = 0.01
        result = msprime_to_discoal(100, L, mu, r, s, N)
        assert result['theta'] == pytest.approx(4 * N * mu * L)

    def test_rho_formula(self):
        """rho = 4 * N * r * L."""
        N = 10000
        L = 100000
        mu = 1.25e-8
        r = 1e-8
        s = 0.01
        result = msprime_to_discoal(100, L, mu, r, s, N)
        assert result['rho'] == pytest.approx(4 * N * r * L)

    def test_alpha_formula(self):
        """alpha = 2 * N * s."""
        N = 10000
        s = 0.01
        result = msprime_to_discoal(100, 100000, 1e-8, 1e-8, s, N)
        assert result['alpha'] == pytest.approx(2 * N * s)

    def test_mu_formula(self):
        """mu = theta / (4 * N * L)."""
        N = 10000
        L = 100000
        theta = 50.0
        result = discoal_to_msprime(theta, 40, 200, 100, L, N)
        assert result['mutation_rate'] == pytest.approx(theta / (4 * N * L))

    def test_selection_coefficient_formula(self):
        """s = alpha / (2N)."""
        N = 10000
        alpha = 200
        result = discoal_to_msprime(50, 40, alpha, 100, 100000, N)
        assert result['selection_coefficient'] == pytest.approx(
            alpha / (2 * N)
        )

    def test_known_values(self):
        """Test with known human-like parameters."""
        N = 10000
        L = 100000
        mu = 1.25e-8
        r = 1e-8
        s = 0.01
        disc = msprime_to_discoal(100, L, mu, r, s, N)
        assert disc['theta'] == pytest.approx(50.0)
        assert disc['rho'] == pytest.approx(40.0)
        assert disc['alpha'] == pytest.approx(200.0)


# ============================================================================
# Integration tests
# ============================================================================

class TestSweepIntegration:
    """Integration tests combining trajectory + structured coalescent."""

    def test_hard_sweep_reduces_tmrca(self):
        """A hard sweep should compress the TMRCA compared to neutral."""
        N = 500
        s = 0.05
        n = 10
        rng = np.random.default_rng(42)

        # Neutral TMRCA: simulate standard coalescent
        neutral_tmrcas = []
        for _ in range(50):
            k = n
            t = 0
            while k > 1:
                rate = k * (k - 1) / (2.0 * 2 * N)
                t += rng.exponential(1.0 / rate)
                k -= 1
            neutral_tmrcas.append(t)

        # Sweep TMRCA (r = 0, maximally linked)
        sweep_tmrcas = []
        for _ in range(50):
            traj = deterministic_trajectory(s, N)
            coal_times, _, _ = structured_coalescent_sweep(
                traj, n, 0.0, N, rng=rng
            )
            if coal_times:
                sweep_tmrcas.append(max(coal_times))

        # Sweep TMRCA should be much smaller than neutral
        if sweep_tmrcas:
            assert np.mean(sweep_tmrcas) < np.mean(neutral_tmrcas)

    def test_soft_sweep_less_reduction_than_hard(self):
        """A soft sweep from standing variation should preserve more
        diversity."""
        N = 500
        s = 0.05
        n = 10
        rng = np.random.default_rng(42)

        # Hard sweep: from 1/(2N)
        hard_events = []
        traj_hard = deterministic_trajectory(s, N)
        for _ in range(30):
            ct, _, _ = structured_coalescent_sweep(
                traj_hard, n, 0.0, N, rng=rng
            )
            hard_events.append(len(ct))

        # Soft sweep: from x0 = 0.1
        soft_events = []
        traj_soft = deterministic_trajectory(s, N, x0=0.1)
        for _ in range(30):
            ct, nB, nb = structured_coalescent_sweep(
                traj_soft, n, 0.0, N, rng=rng
            )
            soft_events.append(len(ct))

        # Soft sweep trajectory is shorter (starts at higher x0)
        assert len(traj_soft) < len(traj_hard)

    def test_lineage_conservation(self):
        """Total lineages (coalesced + remaining) should equal initial
        sample."""
        N = 500
        s = 0.05
        n = 15
        traj = deterministic_trajectory(s, N)

        for seed in range(20):
            coal_times, n_B, n_b = structured_coalescent_sweep(
                traj, n, 0.001, N, rng=np.random.default_rng(seed)
            )
            assert len(coal_times) + n_B + n_b == n, \
                f"Seed {seed}: {len(coal_times)} + {n_B} + {n_b} != {n}"

    def test_escape_probability_matches_simulation(self):
        """Simulated escape fraction should roughly match the formula."""
        N = 500
        s = 0.05
        r_site = 0.001
        n_reps = 200

        escaped = 0
        total = 0
        for seed in range(n_reps):
            traj = deterministic_trajectory(s, N)
            coal_times, n_B, n_b = structured_coalescent_sweep(
                traj, 1, r_site, N, rng=np.random.default_rng(seed)
            )
            total += 1
            if n_b > 0:
                escaped += 1

        sim_escape = escaped / total
        formula_escape = escape_probability(r_site, s, N)
        assert abs(sim_escape - formula_escape) < 0.3, \
            f"Simulated escape {sim_escape:.2f} vs formula {formula_escape:.2f}"
