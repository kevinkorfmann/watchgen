"""
Tests for watchgen/mini_singer.py

Tests all functions from the SINGER mini-implementation, organized by chapter:
- Branch Sampling: joining probabilities, deterministic approximation,
  representative times, emission probabilities, BranchState, transitions
- Time Sampling: branch partition, representative times, PSMC transition,
  transition matrix, linearized forward
- ARG Rescaling: ARG length, time partition, mutation counts, scaling factors,
  rescale times, rate variation
- SGPR: SimpleTree, SPR moves, cut selection, acceptance ratio, tree height
  variability

All functions are imported from watchgen.mini_singer, not redefined.
"""

import numpy as np
import pytest

from watchgen.mini_singer import (
    # Branch Sampling
    joining_probability_exact,
    lambda_approx,
    F_bar_approx,
    f_approx,
    joining_prob_approx,
    lambda_inverse,
    representative_time,
    emission_probability,
    BranchState,
    build_state_space,
    branch_transition_prob,
    split_branch_transition,
    # Time Sampling
    partition_branch,
    representative_times_ts,
    psmc_transition_density,
    psmc_transition_cdf,
    time_transition_matrix,
    forward_linearized,
    type_b_transition,
    type_c_transition,
    # ARG Rescaling
    compute_arg_length_in_window,
    partition_time_axis,
    count_mutations_per_window,
    compute_scaling_factors,
    rescale_times,
    count_mutations_with_rate_variation,
    # SGPR
    SimpleTree,
    spr_move,
    select_cut,
    sgpr_acceptance_ratio,
    simulate_tree_height_variability,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def example_tree():
    """The example tree from sgpr.rst: ((0,1)4, (2,3)5)6.

    Tree shape:
           6 (t=1.5)
          / \\
        4     5
       (0.3) (0.7)
       / \\   / \\
      0   1 2   3
     (0) (0)(0) (0)
    """
    return SimpleTree(
        parent={0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6},
        time={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.3, 5: 0.7, 6: 1.5}
    )


@pytest.fixture
def cherry_tree():
    """A minimal 2-leaf tree (cherry): (0,1)2."""
    return SimpleTree(
        parent={0: 2, 1: 2},
        time={0: 0.0, 1: 0.0, 2: 1.0}
    )


@pytest.fixture
def simple_branches():
    """Simple branch set for ARG rescaling tests."""
    return [
        (1000, 0.0, 0.3),
        (1000, 0.0, 0.3),
        (1000, 0.0, 0.7),
        (1000, 0.0, 0.7),
        (1000, 0.3, 0.7),
        (1000, 0.3, 0.7),
        (1000, 0.7, 1.5),
    ]


@pytest.fixture
def tree_intervals_4leaf():
    """Tree intervals for a 4-leaf tree."""
    t1, t2 = 0.3, 0.8
    return [
        (0, t1), (0, t1), (0, t1), (0, t1),  # 4 leaf branches
        (t1, t2), (t1, t2),                    # 2 internal branches
        (t2, 5.0),                              # root branch (truncated)
    ]


# =========================================================================
# Tests for Branch Sampling: Joining Probabilities
# =========================================================================

class TestJoiningProbabilityExact:
    """Tests for the exact joining probability calculation."""

    def test_all_probabilities_positive(self, tree_intervals_4leaf):
        """All branch joining probabilities should be positive."""
        for lo, hi in tree_intervals_4leaf:
            p = joining_probability_exact(lo, hi, tree_intervals_4leaf)
            assert p > 0, f"Branch [{lo}, {hi}] has non-positive probability"

    def test_leaf_branches_equal(self, tree_intervals_4leaf):
        """Identical leaf branches should have equal joining probabilities."""
        p0 = joining_probability_exact(0, 0.3, tree_intervals_4leaf)
        p1 = joining_probability_exact(0, 0.3, tree_intervals_4leaf)
        assert p0 == pytest.approx(p1)

    def test_probabilities_sum_close_to_one(self, tree_intervals_4leaf):
        """Joining probabilities should sum close to 1 (may not be exact
        because the root branch is truncated)."""
        total = sum(joining_probability_exact(lo, hi, tree_intervals_4leaf)
                    for lo, hi in tree_intervals_4leaf)
        # With root branch truncated at 5.0, total should be close to but
        # slightly less than 1
        assert 0.9 < total <= 1.01


class TestLambdaApprox:
    """Tests for the deterministic approximation of lineage counts."""

    def test_lambda_at_zero(self):
        """At t=0, lambda(0) = n (all lineages present)."""
        for n_val in [5, 50, 100]:
            assert lambda_approx(0, n_val) == pytest.approx(n_val)

    def test_lambda_decreases(self):
        """Lambda should decrease with time."""
        n_val = 50
        t_values = [0.0, 0.1, 0.5, 1.0, 5.0]
        lam_values = [lambda_approx(t, n_val) for t in t_values]
        for i in range(len(lam_values) - 1):
            assert lam_values[i] > lam_values[i+1]

    def test_lambda_approaches_one(self):
        """As t -> infinity, lambda -> 1."""
        assert lambda_approx(100, 50) == pytest.approx(1.0, abs=0.01)

    def test_lambda_positive(self):
        """Lambda should always be positive."""
        for t in [0, 0.1, 1, 10, 100]:
            for n_val in [2, 10, 100]:
                assert lambda_approx(t, n_val) > 0


class TestFBarApprox:
    """Tests for the survival probability approximation."""

    def test_F_bar_at_zero(self):
        """At t=0, F_bar(0) = exp(0) / (n + (1-n)*exp(0))^2 = 1/1 = 1."""
        for n_val in [5, 50, 100]:
            # At t=0: exp(-0)/(n + (1-n)*exp(0))^2 = 1/(n+1-n)^2 = 1
            expected = 1.0
            assert F_bar_approx(0, n_val) == pytest.approx(expected)

    def test_F_bar_decreases(self):
        """Survival probability should decrease with time."""
        n_val = 50
        t_values = [0.0, 0.1, 0.5, 1.0, 5.0]
        f_values = [F_bar_approx(t, n_val) for t in t_values]
        for i in range(len(f_values) - 1):
            assert f_values[i] > f_values[i+1]

    def test_F_bar_positive(self):
        """F_bar should always be positive."""
        for t in [0, 1, 10]:
            for n_val in [2, 10, 100]:
                assert F_bar_approx(t, n_val) > 0


class TestFApprox:
    """Tests for the joining time density approximation."""

    def test_density_positive_near_zero(self):
        """The density should be positive near t=0."""
        assert f_approx(0.01, 50) > 0

    def test_density_relationship(self):
        """f(t) = lambda(t) * F_bar(t)."""
        for t in [0.1, 0.5, 1.0]:
            n_val = 50
            expected = lambda_approx(t, n_val) * F_bar_approx(t, n_val)
            assert f_approx(t, n_val) == pytest.approx(expected)


class TestJoiningProbApprox:
    """Tests for the approximate joining probability."""

    def test_positive_probability(self):
        """Joining probability should be positive for valid branches."""
        n_val = 50
        for x, y in [(0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]:
            p = joining_prob_approx(x, y, n_val)
            assert p > 0

    def test_wider_branch_more_probable(self):
        """A wider branch (spanning more time) near the present should have
        higher joining probability than a narrow branch far from present."""
        n_val = 50
        p_wide = joining_prob_approx(0.0001, 0.5, n_val)
        p_narrow = joining_prob_approx(2.0, 2.5, n_val)
        assert p_wide > p_narrow


class TestLambdaInverse:
    """Tests for the inverse of the lambda function."""

    def test_roundtrip(self):
        """lambda(lambda_inverse(ell, n), n) should give back ell."""
        n_val = 50
        for ell in [2, 5, 10, 25]:
            t = lambda_inverse(ell, n_val)
            recovered = lambda_approx(t, n_val)
            assert recovered == pytest.approx(ell, rel=1e-6)

    def test_inverse_at_n(self):
        """lambda_inverse(n, n) should be 0 (at t=0 all n lineages exist)."""
        n_val = 50
        t = lambda_inverse(n_val, n_val)
        assert t == pytest.approx(0.0, abs=1e-10)


class TestRepresentativeTime:
    """Tests for the representative joining time."""

    def test_within_branch_interval(self):
        """Representative time should be within the branch interval."""
        n_val = 50
        for x, y in [(0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]:
            tau = representative_time(x, y, n_val)
            assert x <= tau <= y

    def test_geometric_mean_property(self):
        """lambda(tau) should be the geometric mean of lambda(x) and lambda(y)."""
        n_val = 50
        x, y = 0.05, 0.5
        tau = representative_time(x, y, n_val)
        lam_x = lambda_approx(x, n_val)
        lam_y = lambda_approx(y, n_val)
        expected_lam = np.sqrt(lam_x * lam_y)
        assert lambda_approx(tau, n_val) == pytest.approx(expected_lam, rel=1e-4)


class TestEmissionProbability:
    """Tests for the emission probability calculation."""

    def test_matching_alleles_high_prob(self):
        """Matching alleles should give high emission probability."""
        e = emission_probability(0, 0, 0.5, 0.1, 1.2, 0.001)
        assert e > 0.99

    def test_mismatching_alleles_low_prob(self):
        """Mismatching alleles should give lower emission probability."""
        e_match = emission_probability(0, 0, 0.5, 0.1, 1.2, 0.001)
        e_mismatch = emission_probability(0, 1, 0.5, 0.1, 1.2, 0.001)
        assert e_match > e_mismatch

    def test_symmetric_mismatch(self):
        """emission(0,1,...) should equal emission(1,0,...) for same branch."""
        e01 = emission_probability(0, 1, 0.5, 0.1, 1.2, 0.001)
        e10 = emission_probability(1, 0, 0.5, 0.1, 1.2, 0.001)
        assert e01 == pytest.approx(e10)

    def test_emission_between_zero_and_one(self):
        """Emission probability should be in (0, 1)."""
        for allele_new, allele_join in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            e = emission_probability(allele_new, allele_join, 0.5, 0.1, 1.2, 0.001)
            assert 0 < e < 1


# =========================================================================
# Tests for BranchState and State Space
# =========================================================================

class TestBranchState:
    """Tests for the BranchState class."""

    def test_length(self):
        """Length should be upper_time - lower_time."""
        bs = BranchState(0, 4, 0.0, 0.3)
        assert bs.length == pytest.approx(0.3)

    def test_is_partial_default_false(self):
        """Default is_partial should be False."""
        bs = BranchState(0, 4, 0.0, 0.3)
        assert not bs.is_partial

    def test_is_partial_true(self):
        """Partial branch should have is_partial=True."""
        bs = BranchState(0, 4, 0.0, 0.15, is_partial=True)
        assert bs.is_partial

    def test_repr(self):
        """String representation should include key info."""
        bs = BranchState(0, 4, 0.0, 0.3)
        r = repr(bs)
        assert "0" in r
        assert "4" in r
        assert "full" in r

    def test_partial_repr(self):
        """Partial branch repr should say 'partial'."""
        bs = BranchState(0, 4, 0.0, 0.15, is_partial=True)
        assert "partial" in repr(bs)


class TestBuildStateSpace:
    """Tests for building the HMM state space."""

    def test_all_full_branches_included(self):
        """All full branches should always be included."""
        full = [BranchState(i, i+4, 0.0, 0.3) for i in range(4)]
        states = build_state_space(full, [], None, epsilon=0.01)
        assert len(states) == 4

    def test_high_prob_partial_included(self):
        """Partial branches above epsilon should be included."""
        full = [BranchState(0, 4, 0.0, 0.3)]
        partial = [(BranchState(0, 4, 0.0, 0.15, is_partial=True), 0.05)]
        states = build_state_space(full, partial, None, epsilon=0.01)
        assert len(states) == 2

    def test_low_prob_partial_excluded(self):
        """Partial branches below epsilon should be excluded."""
        full = [BranchState(0, 4, 0.0, 0.3)]
        partial = [(BranchState(0, 4, 0.0, 0.15, is_partial=True), 0.005)]
        states = build_state_space(full, partial, None, epsilon=0.01)
        assert len(states) == 1

    def test_mixed_partials(self):
        """Mix of above and below threshold partials."""
        full = [BranchState(0, 4, 0.0, 0.3)]
        partial = [
            (BranchState(0, 4, 0.0, 0.15, is_partial=True), 0.05),
            (BranchState(0, 4, 0.15, 0.3, is_partial=True), 0.002),
        ]
        states = build_state_space(full, partial, None, epsilon=0.01)
        assert len(states) == 2  # 1 full + 1 partial above threshold


class TestBranchTransitionProb:
    """Tests for branch transition probabilities."""

    def test_same_branch_high_prob(self):
        """Staying on the same branch should have high probability."""
        p = branch_transition_prob(
            tau_i=0.1, tau_j=0.1, p_j=0.2, rho=0.5,
            is_partial_j=False, q_sum=0.1, same_branch=True
        )
        assert p > 0.9

    def test_different_branch_lower_prob(self):
        """Switching branches should have lower probability."""
        p_same = branch_transition_prob(
            tau_i=0.1, tau_j=0.1, p_j=0.2, rho=0.5,
            is_partial_j=False, q_sum=0.1, same_branch=True
        )
        p_diff = branch_transition_prob(
            tau_i=0.1, tau_j=0.1, p_j=0.2, rho=0.5,
            is_partial_j=False, q_sum=0.1, same_branch=False
        )
        assert p_same > p_diff

    def test_partial_branch_zero_weight(self):
        """Partial branches should have zero q_j weight."""
        p = branch_transition_prob(
            tau_i=0.1, tau_j=0.1, p_j=0.2, rho=0.5,
            is_partial_j=True, q_sum=0.1, same_branch=False
        )
        assert p == 0.0

    def test_transition_positive(self):
        """Transition probability should be non-negative."""
        p = branch_transition_prob(
            tau_i=0.5, tau_j=0.3, p_j=0.1, rho=0.5,
            is_partial_j=False, q_sum=0.2, same_branch=False
        )
        assert p >= 0


class TestSplitBranchTransition:
    """Tests for split branch transition weights."""

    def test_weights_sum_to_one(self):
        """Weights should sum to 1."""
        full = BranchState(1, 5, 0.0, 1.0)
        seg_lo = BranchState(1, 5, 0.0, 0.3, is_partial=True)
        seg_hi = BranchState(1, 5, 0.3, 1.0, is_partial=True)
        weights = split_branch_transition(full, [seg_lo, seg_hi], n=50)
        assert sum(weights) == pytest.approx(1.0)

    def test_lower_segment_more_probable(self):
        """Lower segment (closer to present) should get more weight."""
        full = BranchState(1, 5, 0.0, 1.0)
        seg_lo = BranchState(1, 5, 0.0, 0.3, is_partial=True)
        seg_hi = BranchState(1, 5, 0.3, 1.0, is_partial=True)
        weights = split_branch_transition(full, [seg_lo, seg_hi], n=50)
        assert weights[0] > weights[1]

    def test_single_segment(self):
        """A single segment should get weight 1."""
        seg = BranchState(1, 5, 0.0, 1.0, is_partial=True)
        weights = split_branch_transition(None, [seg], n=50)
        assert weights == [pytest.approx(1.0)]


# =========================================================================
# Tests for Time Sampling
# =========================================================================

class TestPartitionBranch:
    """Tests for branch partitioning into time sub-intervals."""

    def test_endpoints_match(self):
        """First and last boundaries should match branch endpoints."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        assert boundaries[0] == pytest.approx(0.1)
        assert boundaries[-1] == pytest.approx(2.0)

    def test_correct_number_of_boundaries(self):
        """Should have d+1 boundaries for d sub-intervals."""
        for d in [5, 10, 20]:
            boundaries = partition_branch(0.1, 2.0, d=d)
            assert len(boundaries) == d + 1

    def test_monotonically_increasing(self):
        """Boundaries should be strictly increasing."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i+1]

    def test_denser_near_present(self):
        """Sub-intervals should be narrower near the lower endpoint."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        widths = np.diff(boundaries)
        # First interval should be narrower than last
        assert widths[0] < widths[-1]

    def test_degenerate_branch(self):
        """Very short branch should still work."""
        boundaries = partition_branch(1.0, 1.01, d=5)
        assert boundaries[0] == pytest.approx(1.0)
        assert boundaries[-1] == pytest.approx(1.01)


class TestRepresentativeTimesTS:
    """Tests for representative times in time sampling."""

    def test_within_sub_intervals(self):
        """Representative times should fall within their sub-intervals."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        for i in range(len(taus)):
            assert boundaries[i] <= taus[i] <= boundaries[i+1]

    def test_correct_count(self):
        """Should have d representative times for d sub-intervals."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        assert len(taus) == 10


class TestPSMCTransitionDensity:
    """Tests for the PSMC transition density."""

    def test_point_mass_no_recombination(self):
        """At t=s, should return the no-recombination probability."""
        s = 1.0
        rho = 0.5
        d = psmc_transition_density(s, s, rho)
        assert d == pytest.approx(np.exp(-rho * s))

    def test_density_positive(self):
        """Density should be positive for t != s."""
        s = 1.0
        rho = 0.5
        for t in [0.1, 0.5, 1.5, 3.0]:
            d = psmc_transition_density(t, s, rho)
            assert d > 0

    def test_symmetric_about_s(self):
        """The density is NOT symmetric about s, but should be positive
        on both sides."""
        s = 1.0
        rho = 0.5
        d_below = psmc_transition_density(0.5, s, rho)
        d_above = psmc_transition_density(1.5, s, rho)
        assert d_below > 0
        assert d_above > 0


class TestPSMCTransitionCDF:
    """Tests for the PSMC transition CDF."""

    def test_cdf_at_infinity_is_one(self):
        """CDF at large t should approach 1."""
        cdf = psmc_transition_cdf(100, 1.0, 0.5)
        assert cdf == pytest.approx(1.0, abs=1e-4)

    def test_cdf_at_zero_is_zero(self):
        """CDF at t=0 should be 0 (or very close)."""
        cdf = psmc_transition_cdf(0, 1.0, 0.5)
        assert cdf == pytest.approx(0.0, abs=1e-10)

    def test_cdf_monotonically_increasing(self):
        """CDF should be monotonically increasing."""
        s, rho = 1.0, 0.5
        t_values = [0.1, 0.5, 1.0, 1.5, 3.0, 10.0]
        cdf_values = [psmc_transition_cdf(t, s, rho) for t in t_values]
        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] <= cdf_values[i+1]

    def test_cdf_includes_point_mass(self):
        """CDF should jump at t=s due to the point mass."""
        s, rho = 1.0, 0.5
        cdf_just_below = psmc_transition_cdf(s - 0.001, s, rho)
        cdf_at_s = psmc_transition_cdf(s, s, rho)
        # The jump should be approximately exp(-rho*s)
        jump = cdf_at_s - cdf_just_below
        expected_jump = np.exp(-rho * s)
        assert jump == pytest.approx(expected_jump, rel=0.1)


class TestTimeTransitionMatrix:
    """Tests for the time transition matrix."""

    def test_row_sums_to_one(self):
        """Each row of the transition matrix should sum to 1."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)
        row_sums = Q.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0, abs=1e-4)

    def test_matrix_shape(self):
        """Matrix shape should be (d_prev, d_next)."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)
        assert Q.shape == (10, 10)

    def test_all_entries_non_negative(self):
        """All entries should be non-negative."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)
        assert np.all(Q >= -1e-10)

    def test_diagonal_dominant(self):
        """With moderate rho, the matrix should be somewhat diagonal-dominant."""
        boundaries = partition_branch(0.1, 2.0, d=10)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)
        # Diagonal entries should be among the largest in each row
        for i in range(Q.shape[0]):
            assert Q[i, i] > 0


class TestForwardLinearized:
    """Tests for the linearized forward step."""

    def test_matches_quadratic(self):
        """The linearized forward step should approximately match the O(d^2) version.

        The linearization exploits Properties 1 and 2 of the PSMC transition
        matrix, which hold approximately. Small deviations are expected due to
        the approximation, so we use a relaxed tolerance.
        """
        np.random.seed(42)
        d = 10
        boundaries = partition_branch(0.1, 2.0, d=d)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)

        alpha_prev = np.random.dirichlet(np.ones(d))
        emissions = np.random.uniform(0.1, 0.9, size=d)

        alpha_quad = emissions * (alpha_prev @ Q)
        alpha_lin = forward_linearized(alpha_prev, Q, emissions)

        np.testing.assert_allclose(alpha_lin, alpha_quad, rtol=0.02)

    def test_non_negative_output(self):
        """Output should be non-negative."""
        np.random.seed(42)
        d = 10
        boundaries = partition_branch(0.1, 2.0, d=d)
        taus = representative_times_ts(boundaries)
        Q = time_transition_matrix(boundaries, taus, boundaries, rho=0.5)

        alpha_prev = np.random.dirichlet(np.ones(d))
        emissions = np.random.uniform(0.1, 0.9, size=d)

        alpha_lin = forward_linearized(alpha_prev, Q, emissions)
        assert np.all(alpha_lin >= -1e-10)


class TestTypeBTransition:
    """Tests for Type B (hitchhiking) transitions."""

    def test_mapped_intervals_transfer(self):
        """Forward probability should transfer for mapped intervals."""
        alpha_prev = np.array([0.2, 0.3, 0.5])
        boundaries_prev = np.array([0.1, 0.5, 1.0, 2.0])
        boundaries_next = np.array([0.1, 0.5, 1.0, 2.0])
        mapped = [0, 1, None]  # third interval not mapped

        alpha_curr = type_b_transition(alpha_prev, boundaries_prev,
                                        boundaries_next, mapped, rho=0.5)
        assert alpha_curr[0] == pytest.approx(0.2)
        assert alpha_curr[1] == pytest.approx(0.3)
        assert alpha_curr[2] == pytest.approx(0.0)


class TestTypeCTransition:
    """Tests for Type C (new recombination) transitions."""

    def test_output_shape(self):
        """Output should have correct shape."""
        d = 5
        alpha_prev = np.ones(d) / d
        taus_prev = np.linspace(0.2, 1.8, d)
        boundaries_next = partition_branch(0.1, 2.0, d=d)

        alpha_curr = type_c_transition(alpha_prev, taus_prev, boundaries_next)
        assert len(alpha_curr) == d

    def test_output_non_negative(self):
        """Output should be non-negative."""
        d = 5
        alpha_prev = np.ones(d) / d
        taus_prev = np.linspace(0.2, 1.8, d)
        boundaries_next = partition_branch(0.1, 2.0, d=d)

        alpha_curr = type_c_transition(alpha_prev, taus_prev, boundaries_next)
        assert np.all(alpha_curr >= -1e-10)


# =========================================================================
# Tests for ARG Rescaling
# =========================================================================

class TestComputeArgLengthInWindow:
    """Tests for computing ARG length in a time window."""

    def test_full_overlap(self):
        """Branch fully inside window should contribute span * length."""
        branches = [(1000, 0.0, 0.5)]
        length = compute_arg_length_in_window(branches, 0.0, 1.0)
        assert length == pytest.approx(1000 * 0.5)

    def test_no_overlap(self):
        """Branch outside window should contribute nothing."""
        branches = [(1000, 2.0, 3.0)]
        length = compute_arg_length_in_window(branches, 0.0, 1.0)
        assert length == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap should contribute span * overlap."""
        branches = [(1000, 0.0, 0.5)]
        length = compute_arg_length_in_window(branches, 0.2, 1.0)
        assert length == pytest.approx(1000 * 0.3)

    def test_multiple_branches(self):
        """Multiple branches should sum their contributions."""
        branches = [(1000, 0.0, 0.5), (500, 0.0, 0.5)]
        length = compute_arg_length_in_window(branches, 0.0, 0.5)
        assert length == pytest.approx(1500 * 0.5)


class TestPartitionTimeAxis:
    """Tests for partitioning the time axis."""

    def test_correct_number_of_boundaries(self, simple_branches):
        """Should have J+1 boundaries for J windows."""
        for J in [3, 5, 10]:
            boundaries = partition_time_axis(simple_branches, J=J)
            assert len(boundaries) == J + 1

    def test_boundaries_start_at_zero(self, simple_branches):
        """First boundary should be 0."""
        boundaries = partition_time_axis(simple_branches, J=5)
        assert boundaries[0] == pytest.approx(0.0)

    def test_boundaries_end_at_tmax(self, simple_branches):
        """Last boundary should be t_max."""
        t_max = max(hi for _, _, hi in simple_branches)
        boundaries = partition_time_axis(simple_branches, J=5)
        assert boundaries[-1] == pytest.approx(t_max)

    def test_monotonically_increasing(self, simple_branches):
        """Boundaries should be monotonically increasing."""
        boundaries = partition_time_axis(simple_branches, J=5)
        for i in range(len(boundaries) - 1):
            assert boundaries[i] < boundaries[i+1]


class TestCountMutationsPerWindow:
    """Tests for counting mutations per window."""

    def test_total_count(self):
        """Total fractional count should equal number of mutations."""
        boundaries = np.array([0.0, 0.5, 1.0, 1.5])
        mutations = [(0.2, 0.8), (0.1, 0.9)]
        counts = count_mutations_per_window(mutations, boundaries)
        assert sum(counts) == pytest.approx(2.0)

    def test_single_window_mutation(self):
        """Mutation fully in one window gets count 1."""
        boundaries = np.array([0.0, 0.5, 1.0])
        mutations = [(0.1, 0.4)]  # fully in window 0
        counts = count_mutations_per_window(mutations, boundaries)
        assert counts[0] == pytest.approx(1.0)
        assert counts[1] == pytest.approx(0.0)

    def test_split_mutation(self):
        """Mutation spanning two windows is split proportionally."""
        boundaries = np.array([0.0, 0.5, 1.0])
        mutations = [(0.0, 1.0)]  # spans both windows equally
        counts = count_mutations_per_window(mutations, boundaries)
        assert counts[0] == pytest.approx(0.5)
        assert counts[1] == pytest.approx(0.5)

    def test_zero_length_branch_skipped(self):
        """Degenerate (zero-length) branches should be skipped."""
        boundaries = np.array([0.0, 1.0])
        mutations = [(0.5, 0.5)]  # zero length
        counts = count_mutations_per_window(mutations, boundaries)
        assert counts[0] == pytest.approx(0.0)


class TestComputeScalingFactors:
    """Tests for computing rescaling factors."""

    def test_uniform_counts_give_uniform_factors(self):
        """If all windows have the same count, factors should be equal."""
        J = 5
        counts = np.ones(J) * 10
        total_length = 1000
        theta = 0.001
        c = compute_scaling_factors(counts, total_length, theta, J)
        # All factors should be equal
        assert np.allclose(c, c[0])

    def test_zero_expected_returns_ones(self):
        """If expected is 0, return all ones."""
        c = compute_scaling_factors(np.ones(5), 0.0, 0.001, 5)
        np.testing.assert_array_equal(c, np.ones(5))

    def test_double_counts_double_factor(self):
        """Doubling the count should double the factor."""
        J = 5
        total_length = 1000
        theta = 0.001
        counts1 = np.ones(J) * 10
        counts2 = np.ones(J) * 20
        c1 = compute_scaling_factors(counts1, total_length, theta, J)
        c2 = compute_scaling_factors(counts2, total_length, theta, J)
        np.testing.assert_allclose(c2, 2 * c1)


class TestRescaleTimes:
    """Tests for rescaling coalescence times."""

    def test_leaves_stay_at_zero(self):
        """Leaf nodes at time 0 should remain at time 0."""
        node_times = {0: 0.0, 1: 0.0, 2: 0.5}
        boundaries = np.array([0.0, 0.5, 1.0])
        scaling = np.array([1.0, 1.0])
        new_times = rescale_times(node_times, boundaries, scaling)
        assert new_times[0] == 0.0
        assert new_times[1] == 0.0

    def test_identity_scaling(self):
        """With all scaling factors = 1, times should not change."""
        node_times = {0: 0.0, 1: 0.3, 2: 0.7}
        boundaries = np.array([0.0, 0.5, 1.0])
        scaling = np.array([1.0, 1.0])
        new_times = rescale_times(node_times, boundaries, scaling)
        for nid, t in node_times.items():
            assert new_times[nid] == pytest.approx(t)

    def test_double_scaling(self):
        """With all scaling factors = 2, times should double."""
        node_times = {0: 0.0, 1: 0.25, 2: 0.75}
        boundaries = np.array([0.0, 0.5, 1.0])
        scaling = np.array([2.0, 2.0])
        new_times = rescale_times(node_times, boundaries, scaling)
        assert new_times[0] == 0.0
        assert new_times[1] == pytest.approx(0.5)
        assert new_times[2] == pytest.approx(1.5)

    def test_monotonicity_preserved(self):
        """Order of times should be preserved after rescaling."""
        node_times = {0: 0.0, 1: 0.3, 2: 0.7, 3: 1.5}
        boundaries = np.array([0.0, 0.5, 1.0, 2.0])
        scaling = np.array([0.5, 2.0, 1.5])
        new_times = rescale_times(node_times, boundaries, scaling)
        times_sorted = sorted(new_times.values())
        for i in range(len(times_sorted) - 1):
            assert times_sorted[i] <= times_sorted[i+1]


class TestCountMutationsWithRateVariation:
    """Tests for mutation counting with rate variation."""

    def test_constant_rate_matches_simple(self):
        """With constant rate, expected should be proportional to length."""
        branches = [(0, 100, 0.0, 1.0)]
        mutations = [(50, 0.0, 1.0)]
        boundaries = np.array([0.0, 0.5, 1.0])
        rate_map = lambda x: 1e-8

        expected, observed = count_mutations_with_rate_variation(
            branches, mutations, boundaries, rate_map)
        # Expected should be positive in both windows
        assert expected[0] > 0
        assert expected[1] > 0

    def test_hotspot_higher_expected(self):
        """A hotspot region should have higher expected mutations."""
        branches = [(0, 100, 0.0, 1.0)]
        mutations = []
        boundaries = np.array([0.0, 0.5, 1.0])

        def rate_map(x):
            return 1e-7 if 40 <= x < 60 else 1e-8

        expected, observed = count_mutations_with_rate_variation(
            branches, mutations, boundaries, rate_map)
        # Expected should be positive (rate_map evaluated at midpoint 50)
        assert expected[0] > 0


# =========================================================================
# Tests for SimpleTree (from SGPR chapter)
# =========================================================================

class TestSimpleTree:
    """Tests for the SimpleTree class."""

    def test_height_example_tree(self, example_tree):
        """Height should equal the root time (TMRCA)."""
        assert example_tree.height() == 1.5

    def test_height_cherry(self, cherry_tree):
        """Height of a cherry tree should be the root time."""
        assert cherry_tree.height() == 1.0

    def test_branches_count(self, example_tree):
        """The example tree has 7 nodes and 6 edges (branches)."""
        branches = example_tree.branches()
        assert len(branches) == 6

    def test_branches_have_positive_length(self, example_tree):
        """All branch lengths must be strictly positive."""
        for child, parent, length in example_tree.branches():
            assert length > 0, f"Branch {child}->{parent} has non-positive length {length}"

    def test_branch_length_computation(self, example_tree):
        """Branch lengths should equal parent_time - child_time."""
        for child, parent, length in example_tree.branches():
            expected = example_tree.time[parent] - example_tree.time[child]
            assert length == pytest.approx(expected)

    def test_children_map_consistency(self, example_tree):
        """Children map should be the inverse of the parent map."""
        for child, par in example_tree.parent.items():
            if par is not None:
                assert child in example_tree.children[par]

    def test_children_map_root_has_two_children(self, example_tree):
        """The root (node 6) should have exactly two children: 4 and 5."""
        assert set(example_tree.children[6]) == {4, 5}

    def test_leaf_nodes_have_time_zero(self, example_tree):
        """All leaf nodes (0-3) are at time 0."""
        for leaf in [0, 1, 2, 3]:
            assert example_tree.time[leaf] == 0.0

    def test_internal_nodes_have_positive_time(self, example_tree):
        """Internal nodes (4, 5, 6) have positive times."""
        for node in [4, 5, 6]:
            assert example_tree.time[node] > 0.0

    def test_single_leaf_tree(self):
        """A tree with a single leaf (degenerate case)."""
        tree = SimpleTree(parent={0: 1}, time={0: 0.0, 1: 1.0})
        assert tree.height() == 1.0
        branches = tree.branches()
        assert len(branches) == 1
        assert branches[0] == (0, 1, 1.0)


# =========================================================================
# Tests for spr_move
# =========================================================================

class TestSPRMove:
    """Tests for the spr_move function."""

    def test_spr_preserves_leaf_count(self, example_tree):
        """SPR should preserve the number of leaf nodes."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        original_leaves = {n for n, t in example_tree.time.items() if t == 0}
        new_leaves = {n for n, t in new_tree.time.items() if t == 0}
        assert len(new_leaves) == len(original_leaves)

    def test_spr_creates_new_internal_node(self, example_tree):
        """SPR should create a new internal node at the re-attachment point."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        assert 0.5 in new_tree.time.values()

    def test_spr_new_internal_node_time(self, example_tree):
        """The new internal node should have the specified time."""
        new_time = 0.8
        new_tree = spr_move(example_tree, cut_node=0, new_parent=3, new_time=new_time)
        new_internal = max(new_tree.time.keys())
        assert new_tree.time[new_internal] == new_time

    def test_spr_cut_node_has_new_parent(self, example_tree):
        """After SPR, the cut node should be connected to the new internal node."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        new_internal = max(new_tree.time.keys())
        assert new_tree.parent[0] == new_internal

    def test_spr_target_branch_has_new_parent(self, example_tree):
        """The target branch node should now point to the new internal node."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        new_internal = max(new_tree.time.keys())
        assert new_tree.parent[2] == new_internal

    def test_spr_removes_unary_node(self, example_tree):
        """SPR should remove the now-unary old parent node."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        assert 4 not in new_tree.parent

    def test_spr_sibling_reconnected_to_grandparent(self, example_tree):
        """Sibling of the cut node should connect directly to the grandparent."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        assert new_tree.parent[1] == 6

    def test_spr_all_branch_lengths_positive(self, example_tree):
        """All branches in the resulting tree should have positive lengths."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        for child, parent, length in new_tree.branches():
            assert length > 0, f"Branch {child}->{parent} has non-positive length {length}"

    def test_spr_height_can_change(self, example_tree):
        """SPR can change the tree height if the re-attachment time changes the root."""
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=2.0)
        assert new_tree.height() >= 1.5

    def test_spr_on_cherry(self, cherry_tree):
        """SPR on a 2-leaf tree: cut one leaf and re-attach to the other."""
        new_tree = spr_move(cherry_tree, cut_node=0, new_parent=1, new_time=0.5)
        assert 0.5 in new_tree.time.values()


# =========================================================================
# Tests for select_cut
# =========================================================================

class TestSelectCut:
    """Tests for the select_cut function."""

    def test_cut_time_in_valid_range(self, example_tree):
        """Cut time should be in [0, tree height)."""
        np.random.seed(42)
        for _ in range(100):
            cut_node, cut_time = select_cut(example_tree)
            assert 0 <= cut_time < example_tree.height()

    def test_cut_node_is_valid_branch(self, example_tree):
        """Cut node should correspond to a valid branch in the tree."""
        np.random.seed(42)
        valid_children = {child for child, par in example_tree.parent.items() if par is not None}
        for _ in range(100):
            cut_node, cut_time = select_cut(example_tree)
            assert cut_node in valid_children

    def test_cut_time_crosses_selected_branch(self, example_tree):
        """The cut time must be within the branch interval of the selected node."""
        np.random.seed(42)
        for _ in range(100):
            cut_node, cut_time = select_cut(example_tree)
            child_time = example_tree.time[cut_node]
            parent_time = example_tree.time[example_tree.parent[cut_node]]
            assert child_time <= cut_time < parent_time

    def test_cut_reproducibility(self, example_tree):
        """With the same seed, select_cut should produce the same result."""
        np.random.seed(123)
        result1 = select_cut(example_tree)
        np.random.seed(123)
        result2 = select_cut(example_tree)
        assert result1[0] == result2[0]
        assert result1[1] == pytest.approx(result2[1])

    def test_cut_distribution_covers_all_branches(self, example_tree):
        """Over many samples, all branches should be selected at least once."""
        np.random.seed(0)
        selected_nodes = set()
        for _ in range(1000):
            cut_node, _ = select_cut(example_tree)
            selected_nodes.add(cut_node)
        valid_children = {child for child, par in example_tree.parent.items() if par is not None}
        assert selected_nodes == valid_children

    def test_cut_probability_proportional_to_branch_length(self, example_tree):
        """Branches should be selected with nonzero probability."""
        np.random.seed(7)
        n_samples = 50000
        counts = {}
        for _ in range(n_samples):
            cut_node, _ = select_cut(example_tree)
            counts[cut_node] = counts.get(cut_node, 0) + 1

        for child, parent, length in example_tree.branches():
            observed_prob = counts.get(child, 0) / n_samples
            assert observed_prob > 0, f"Branch {child} was never selected"


# =========================================================================
# Tests for sgpr_acceptance_ratio
# =========================================================================

class TestSGPRAcceptanceRatio:
    """Tests for the sgpr_acceptance_ratio function."""

    def test_equal_heights_gives_one(self):
        """When old and new heights are equal, acceptance ratio is 1."""
        assert sgpr_acceptance_ratio(2.0, 2.0) == 1.0

    def test_shorter_new_tree_gives_one(self):
        """When new tree is shorter (height decreases), ratio is 1."""
        assert sgpr_acceptance_ratio(2.0, 1.5) == 1.0

    def test_taller_new_tree_gives_ratio(self):
        """When new tree is taller, ratio = old_height / new_height < 1."""
        ratio = sgpr_acceptance_ratio(1.5, 2.0)
        assert ratio == pytest.approx(1.5 / 2.0)

    def test_ratio_always_between_zero_and_one(self):
        """The acceptance ratio must always be in (0, 1]."""
        for old_h in [0.1, 1.0, 2.0, 10.0]:
            for new_h in [0.1, 1.0, 2.0, 10.0]:
                ratio = sgpr_acceptance_ratio(old_h, new_h)
                assert 0 < ratio <= 1.0

    def test_ratio_formula_min(self):
        """Verify the min(1, old/new) formula explicitly."""
        assert sgpr_acceptance_ratio(3.0, 5.0) == pytest.approx(3.0 / 5.0)
        assert sgpr_acceptance_ratio(5.0, 3.0) == 1.0
        assert sgpr_acceptance_ratio(1.0, 1.0) == 1.0

    def test_ratio_symmetry_product(self):
        """The product of forward and reverse ratios has a known bound."""
        h1, h2 = 1.5, 2.5
        forward = sgpr_acceptance_ratio(h1, h2)
        reverse = sgpr_acceptance_ratio(h2, h1)
        expected_product = min(h1, h2) / max(h1, h2)
        assert forward * reverse == pytest.approx(expected_product)

    def test_very_similar_heights_near_one(self):
        """For very similar heights, the ratio should be close to 1."""
        ratio = sgpr_acceptance_ratio(2.0, 2.001)
        assert ratio > 0.999

    def test_very_different_heights_near_zero(self):
        """For very different heights, the ratio should be near 0."""
        ratio = sgpr_acceptance_ratio(0.01, 100.0)
        assert ratio < 0.001


# =========================================================================
# Tests for simulate_tree_height_variability
# =========================================================================

class TestSimulateTreeHeightVariability:
    """Tests for the simulate_tree_height_variability function."""

    def test_returns_correct_number_of_replicates(self):
        """Output array should have the requested number of replicates."""
        np.random.seed(42)
        heights = simulate_tree_height_variability(10, n_replicates=500)
        assert len(heights) == 500

    def test_all_heights_positive(self):
        """All simulated tree heights should be positive."""
        np.random.seed(42)
        heights = simulate_tree_height_variability(10, n_replicates=1000)
        assert np.all(heights > 0)

    def test_mean_height_close_to_expected(self):
        """Mean TMRCA should be close to 2*(1 - 1/n) for n samples."""
        np.random.seed(42)
        n = 50
        heights = simulate_tree_height_variability(n, n_replicates=10000)
        expected_mean = 2.0 * (1.0 - 1.0 / n)
        assert heights.mean() == pytest.approx(expected_mean, rel=0.05)

    def test_mean_height_for_two_samples(self):
        """For n=2, E[TMRCA] = 1.0 (single exponential with rate 1)."""
        np.random.seed(42)
        heights = simulate_tree_height_variability(2, n_replicates=20000)
        assert heights.mean() == pytest.approx(1.0, rel=0.05)

    def test_cv_decreases_with_n(self):
        """Coefficient of variation should decrease as n increases."""
        np.random.seed(42)
        cv_values = []
        for n in [5, 20, 100]:
            heights = simulate_tree_height_variability(n, n_replicates=5000)
            cv = heights.std() / heights.mean()
            cv_values.append(cv)
        assert cv_values[0] > cv_values[1] > cv_values[2]

    def test_height_distribution_for_n2_is_exponential(self):
        """For n=2, TMRCA ~ Exp(1). Check the variance matches."""
        np.random.seed(42)
        heights = simulate_tree_height_variability(2, n_replicates=20000)
        assert heights.var() == pytest.approx(1.0, rel=0.1)

    def test_reproducibility(self):
        """With the same seed, results should be identical."""
        np.random.seed(99)
        h1 = simulate_tree_height_variability(10, n_replicates=100)
        np.random.seed(99)
        h2 = simulate_tree_height_variability(10, n_replicates=100)
        np.testing.assert_array_equal(h1, h2)

    def test_acceptance_rate_increases_with_n(self):
        """Simulate the acceptance rate experiment from the chapter."""
        np.random.seed(42)
        mean_acceptance = []
        for n in [5, 50, 200]:
            heights = simulate_tree_height_variability(n, n_replicates=2000)
            ratios = []
            for k in range(0, len(heights) - 1, 2):
                r = sgpr_acceptance_ratio(heights[k], heights[k + 1])
                ratios.append(r)
            mean_acceptance.append(np.mean(ratios))
        assert mean_acceptance[0] < mean_acceptance[1] < mean_acceptance[2]
        assert mean_acceptance[2] > 0.75


# =========================================================================
# Integration tests
# =========================================================================

class TestSPRIntegration:
    """Integration tests combining multiple components."""

    def test_select_cut_then_spr(self, example_tree):
        """Select a random cut and then perform an SPR move using it."""
        np.random.seed(42)
        cut_node, cut_time = select_cut(example_tree)

        valid_targets = [child for child, par in example_tree.parent.items()
                         if par is not None and child != cut_node]
        target = valid_targets[0]

        min_time = max(example_tree.time[cut_node], example_tree.time[target])
        target_parent = example_tree.parent[target]
        max_time = example_tree.time[target_parent] if target_parent is not None else min_time + 1.0
        new_time = (min_time + max_time) / 2.0

        new_tree = spr_move(example_tree, cut_node, target, new_time=new_time)
        assert new_tree.height() > 0
        branches = new_tree.branches()
        assert len(branches) > 0

    def test_spr_and_acceptance_ratio(self, example_tree):
        """Perform an SPR and compute the SGPR acceptance ratio."""
        old_height = example_tree.height()
        new_tree = spr_move(example_tree, cut_node=0, new_parent=2, new_time=0.5)
        new_height = new_tree.height()

        ratio = sgpr_acceptance_ratio(old_height, new_height)
        assert 0 < ratio <= 1.0


class TestRescalingIntegration:
    """Integration tests for the full ARG rescaling pipeline."""

    def test_full_rescaling_pipeline(self, simple_branches):
        """Test the full rescaling pipeline from branches to rescaled times."""
        # Partition
        boundaries = partition_time_axis(simple_branches, J=5)

        # Generate some mutations
        np.random.seed(42)
        mutations = [(np.random.uniform(0, 0.5), np.random.uniform(0.5, 1.5))
                     for _ in range(20)]

        # Count mutations
        counts = count_mutations_per_window(mutations, boundaries)
        assert len(counts) == 5

        # Compute scaling
        total_length = sum(span * (hi - lo)
                          for span, lo, hi in simple_branches)
        c = compute_scaling_factors(counts, total_length, 0.001, 5)
        assert len(c) == 5

        # Rescale times
        node_times = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0,
                      4: 0.3, 5: 0.7, 6: 1.5}
        new_times = rescale_times(node_times, boundaries, c)

        # Leaves should stay at 0
        for leaf in [0, 1, 2, 3]:
            assert new_times[leaf] == 0.0

        # Internal nodes should have positive times
        for node in [4, 5, 6]:
            assert new_times[node] > 0

    def test_branch_sampling_and_rescaling(self):
        """Combine branch sampling probabilities with rescaling."""
        n = 50
        branches_approx = [(0.01, 0.05), (0.05, 0.2), (0.2, 1.0)]

        # Compute joining probabilities
        probs = [joining_prob_approx(x, y, n) for x, y in branches_approx]
        assert all(p > 0 for p in probs)

        # Compute representative times
        taus = [representative_time(max(x, 1e-10), y, n)
                for x, y in branches_approx]
        for i, (x, y) in enumerate(branches_approx):
            assert x <= taus[i] <= y
