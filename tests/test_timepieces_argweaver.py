"""
Tests for ARGweaver timepiece code blocks.

Re-defines each self-contained function from the RST documentation
and tests its mathematical properties.
"""

import numpy as np
import random
from math import exp, log
import pytest


# ============================================================
# Functions from time_discretization.rst
# ============================================================

def get_time_point(i, ntimes, maxtime, delta=0.01):
    """Compute the i-th discretized time point."""
    return (exp(i / float(ntimes) * log(1 + delta * maxtime)) - 1) / delta


def get_time_points(ntimes=20, maxtime=160000, delta=0.01):
    """Compute all discretized time points."""
    return [get_time_point(i, ntimes - 1, maxtime, delta)
            for i in range(ntimes)]


def get_time_steps(times):
    """Compute time step sizes from time points."""
    ntimes = len(times) - 1
    return [times[i+1] - times[i] for i in range(ntimes)]


def get_coal_times(times):
    """Compute coal times (geometric mean midpoints) for each interval."""
    ntimes = len(times) - 1
    times2 = []
    for i in range(ntimes):
        times2.append(times[i])
        times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
    times2.append(times[ntimes])
    return times2


def get_coal_time_steps(times):
    """Compute the effective time step for coalescence at each time point."""
    ntimes = len(times) - 1
    times2 = []
    for i in range(ntimes):
        times2.append(times[i])
        times2.append(((times[i+1] + 1) * (times[i] + 1)) ** 0.5 - 1)
    times2.append(times[ntimes])

    coal_time_steps = []
    for i in range(0, len(times2), 2):
        coal_time_steps.append(
            times2[min(i + 1, len(times2) - 1)] -
            times2[max(i - 1, 0)]
        )
    return coal_time_steps


# ============================================================
# Functions from transition_probabilities.rst
# ============================================================

def logsumexp(x):
    """Numerically stable log-sum-exp."""
    m = np.max(x)
    if m == -np.inf:
        return -np.inf
    return m + np.log(np.sum(np.exp(x - m)))


def verify_transition_matrix(trans, tol=1e-6):
    """Verify that transition matrix rows sum to 1."""
    row_sums = trans.sum(axis=1)
    ok = np.allclose(row_sums, 1.0, atol=tol)
    return ok


# ============================================================
# Functions from mcmc_sampling.rst
# ============================================================

def sample_next_recomb(treelen, rho):
    """Sample the distance to the next recombination event."""
    rate = max(treelen * rho, rho)
    return random.expovariate(rate)


def sample_recomb_time(nbranches, time_steps, root_age_index):
    """Sample which time interval the recombination falls in."""
    weights = [nbranches[i] * time_steps[i]
               for i in range(root_age_index + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]

    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return i
    return len(probs) - 1


def sample_tree(k, popsizes, times):
    """Sample a coalescent tree using a discrete-time coalescent."""
    ntimes = len(times)
    coal_times = []

    timei = 0
    n = popsizes[timei]
    t = 0.0
    k2 = k

    while k2 > 1:
        coal_rate = (k2 * (k2 - 1) / 2) / float(n)
        t2 = random.expovariate(coal_rate)

        if timei < ntimes - 2 and t + t2 > times[timei + 1]:
            timei += 1
            t = times[timei]
            n = popsizes[timei]
            continue

        t += t2
        coal_times.append(t)
        k2 -= 1

    return coal_times


# ============================================================
# Tests: Time Discretization
# ============================================================

class TestGetTimePoint:
    def test_first_point_is_zero(self):
        """t_0 should always be 0."""
        assert get_time_point(0, 19, 160000, 0.01) == 0.0

    def test_last_point_is_maxtime(self):
        """t_{ntimes} should be maxtime."""
        result = get_time_point(19, 19, 160000, 0.01)
        assert abs(result - 160000.0) < 1e-6

    def test_monotonically_increasing(self):
        """Time points should increase monotonically."""
        ntimes = 19
        maxtime = 160000
        for i in range(ntimes):
            assert get_time_point(i, ntimes, maxtime) < get_time_point(i + 1, ntimes, maxtime)

    def test_different_deltas_produce_different_grids(self):
        """Smaller delta should produce wider initial spacing."""
        t_small = get_time_point(1, 19, 160000, 0.001)
        t_large = get_time_point(1, 19, 160000, 0.1)
        assert t_small > t_large

    def test_positive_values(self):
        """All time points should be non-negative."""
        for i in range(20):
            assert get_time_point(i, 19, 160000, 0.01) >= 0.0


class TestGetTimePoints:
    def test_length(self):
        """Should return ntimes points."""
        times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
        assert len(times) == 20

    def test_first_is_zero(self):
        times = get_time_points(ntimes=20)
        assert times[0] == 0.0

    def test_last_is_maxtime(self):
        times = get_time_points(ntimes=20, maxtime=160000)
        assert abs(times[-1] - 160000.0) < 1e-6

    def test_increasing(self):
        times = get_time_points(ntimes=20)
        for i in range(len(times) - 1):
            assert times[i] < times[i + 1]

    def test_steps_grow(self):
        """Time steps should generally increase (log spacing)."""
        times = get_time_points(ntimes=20, maxtime=160000, delta=0.01)
        steps = [times[i+1] - times[i] for i in range(len(times) - 1)]
        # Steps should be mostly increasing (except possibly the last)
        for i in range(len(steps) - 2):
            assert steps[i] < steps[i + 1]


class TestGetTimeSteps:
    def test_length(self):
        times = get_time_points(ntimes=20)
        steps = get_time_steps(times)
        assert len(steps) == 19

    def test_all_positive(self):
        times = get_time_points(ntimes=20)
        steps = get_time_steps(times)
        for s in steps:
            assert s > 0

    def test_sum_equals_maxtime(self):
        """Sum of steps should equal the total time range."""
        maxtime = 160000
        times = get_time_points(ntimes=20, maxtime=maxtime)
        steps = get_time_steps(times)
        assert abs(sum(steps) - maxtime) < 1e-6


class TestGetCoalTimes:
    def test_length(self):
        """Interleaved structure should have 2*ntimes - 1 entries."""
        times = [0.0, 100.0, 1000.0, 10000.0]
        ct = get_coal_times(times)
        assert len(ct) == 7  # 2*3 + 1

    def test_first_is_zero(self):
        times = [0.0, 100.0, 1000.0]
        ct = get_coal_times(times)
        assert ct[0] == 0.0

    def test_last_is_last_time(self):
        times = [0.0, 100.0, 1000.0]
        ct = get_coal_times(times)
        assert ct[-1] == 1000.0

    def test_midpoints_between_boundaries(self):
        """Midpoints should lie between their surrounding time points."""
        times = get_time_points(ntimes=20)
        ct = get_coal_times(times)
        ntimes = len(times) - 1
        for i in range(ntimes):
            t_lo = ct[2 * i]
            t_mid = ct[2 * i + 1]
            t_hi = ct[2 * i + 2]
            assert t_lo <= t_mid <= t_hi

    def test_geometric_midpoint_below_arithmetic(self):
        """Geometric mean midpoint should be <= arithmetic mean for non-negative values."""
        times = get_time_points(ntimes=20)
        ct = get_coal_times(times)
        ntimes = len(times) - 1
        for i in range(ntimes):
            t_lo = times[i]
            t_hi = times[i + 1]
            geo_mid = ct[2 * i + 1]
            arith_mid = (t_lo + t_hi) / 2
            assert geo_mid <= arith_mid + 1e-10

    def test_monotonically_increasing(self):
        """The interleaved structure should be monotonically increasing."""
        times = get_time_points(ntimes=20)
        ct = get_coal_times(times)
        for i in range(len(ct) - 1):
            assert ct[i] <= ct[i + 1]


class TestGetCoalTimeSteps:
    def test_length(self):
        """Should have one entry per time point."""
        times = get_time_points(ntimes=20)
        steps = get_coal_time_steps(times)
        assert len(steps) == 20

    def test_all_positive(self):
        """Coal time steps should be positive (except possibly the first which starts at 0)."""
        times = get_time_points(ntimes=20)
        steps = get_coal_time_steps(times)
        for s in steps:
            assert s >= 0


# ============================================================
# Tests: Transition Probabilities
# ============================================================

class TestLogsumexp:
    def test_single_value(self):
        x = np.array([2.0])
        assert abs(logsumexp(x) - 2.0) < 1e-10

    def test_known_result(self):
        """log(exp(1) + exp(2)) = 2 + log(1 + exp(-1))"""
        x = np.array([1.0, 2.0])
        expected = 2.0 + np.log(1 + np.exp(-1.0))
        assert abs(logsumexp(x) - expected) < 1e-10

    def test_large_values(self):
        """Should handle large values without overflow."""
        x = np.array([1000.0, 1001.0])
        result = logsumexp(x)
        expected = 1001.0 + np.log(1 + np.exp(-1.0))
        assert abs(result - expected) < 1e-10

    def test_very_negative_values(self):
        """Should handle very negative values."""
        x = np.array([-1000.0, -999.0])
        result = logsumexp(x)
        expected = -999.0 + np.log(1 + np.exp(-1.0))
        assert abs(result - expected) < 1e-10

    def test_all_negative_inf(self):
        x = np.array([-np.inf, -np.inf])
        assert logsumexp(x) == -np.inf

    def test_matches_scipy(self):
        """Should match scipy's logsumexp."""
        from scipy.special import logsumexp as sp_logsumexp
        np.random.seed(42)
        x = np.random.randn(10) * 5
        assert abs(logsumexp(x) - sp_logsumexp(x)) < 1e-10


class TestVerifyTransitionMatrix:
    def test_valid_stochastic_matrix(self):
        T = np.array([[0.7, 0.2, 0.1],
                       [0.3, 0.5, 0.2],
                       [0.1, 0.1, 0.8]])
        assert verify_transition_matrix(T)

    def test_invalid_matrix(self):
        T = np.array([[0.5, 0.3, 0.1],
                       [0.3, 0.5, 0.2],
                       [0.1, 0.1, 0.8]])
        assert not verify_transition_matrix(T)

    def test_identity_is_valid(self):
        T = np.eye(5)
        assert verify_transition_matrix(T)

    def test_uniform_is_valid(self):
        n = 4
        T = np.ones((n, n)) / n
        assert verify_transition_matrix(T)


# ============================================================
# Tests: MCMC Sampling
# ============================================================

class TestSampleNextRecomb:
    def test_positive(self):
        """Distance to next recombination should be positive."""
        random.seed(42)
        for _ in range(100):
            d = sample_next_recomb(1000.0, 1e-8)
            assert d > 0

    def test_mean_approximately_correct(self):
        """Mean should approximate 1/(treelen * rho)."""
        random.seed(42)
        treelen = 1000.0
        rho = 1e-4
        samples = [sample_next_recomb(treelen, rho) for _ in range(10000)]
        expected_mean = 1.0 / (treelen * rho)
        sample_mean = sum(samples) / len(samples)
        assert abs(sample_mean - expected_mean) / expected_mean < 0.1

    def test_guards_against_zero_treelen(self):
        """Should not crash with zero tree length."""
        random.seed(42)
        d = sample_next_recomb(0.0, 1e-8)
        assert d > 0


class TestSampleRecombTime:
    def test_returns_valid_index(self):
        """Should return an index within the valid range."""
        random.seed(42)
        nbranches = [4, 4, 3, 2, 1]
        time_steps = [100, 200, 300, 400, 500]
        root_age_index = 3
        for _ in range(100):
            idx = sample_recomb_time(nbranches, time_steps, root_age_index)
            assert 0 <= idx <= root_age_index

    def test_distribution_weighted_by_branch_material(self):
        """Intervals with more branch material should be sampled more often."""
        random.seed(42)
        nbranches = [10, 1, 1, 1]
        time_steps = [1.0, 1.0, 1.0, 1.0]
        root_age_index = 3
        counts = [0, 0, 0, 0]
        n_samples = 10000
        for _ in range(n_samples):
            idx = sample_recomb_time(nbranches, time_steps, root_age_index)
            counts[idx] += 1
        # First interval has 10x the weight, so should get ~10x the counts
        # Total weight = 10 + 1 + 1 + 1 = 13, expected fraction for first = 10/13
        expected_frac = 10.0 / 13.0
        observed_frac = counts[0] / n_samples
        assert abs(observed_frac - expected_frac) < 0.05

    def test_probabilities_sum_to_one(self):
        """Implicitly tested: the function always returns a valid index."""
        random.seed(42)
        nbranches = [3, 3, 2, 1]
        time_steps = [50.0, 100.0, 200.0, 400.0]
        root_age_index = 3
        for _ in range(1000):
            idx = sample_recomb_time(nbranches, time_steps, root_age_index)
            assert isinstance(idx, int)
            assert 0 <= idx <= root_age_index


class TestSampleTree:
    def test_correct_number_of_coalescences(self):
        """k lineages should produce k-1 coalescence events."""
        random.seed(42)
        k = 5
        popsizes = [10000.0] * 20
        times = get_time_points(ntimes=20, maxtime=160000)
        coal_times = sample_tree(k, popsizes, times)
        assert len(coal_times) == k - 1

    def test_coalescence_times_increasing(self):
        """Coalescence times should be non-decreasing."""
        random.seed(42)
        k = 10
        popsizes = [10000.0] * 20
        times = get_time_points(ntimes=20, maxtime=160000)
        coal_times = sample_tree(k, popsizes, times)
        for i in range(len(coal_times) - 1):
            assert coal_times[i] <= coal_times[i + 1]

    def test_coalescence_times_positive(self):
        """All coalescence times should be positive."""
        random.seed(42)
        k = 6
        popsizes = [5000.0] * 20
        times = get_time_points(ntimes=20, maxtime=160000)
        coal_times = sample_tree(k, popsizes, times)
        for t in coal_times:
            assert t > 0

    def test_two_lineages_one_coalescence(self):
        """Two lineages should produce exactly one coalescence event."""
        random.seed(42)
        popsizes = [10000.0] * 20
        times = get_time_points(ntimes=20, maxtime=160000)
        coal_times = sample_tree(2, popsizes, times)
        assert len(coal_times) == 1

    def test_smaller_population_coalesces_faster(self):
        """Smaller population should generally lead to earlier coalescence."""
        random.seed(42)
        times = get_time_points(ntimes=20, maxtime=160000)

        small_pop = [100.0] * 20
        large_pop = [100000.0] * 20

        n_trials = 200
        small_tmrca = []
        large_tmrca = []
        for _ in range(n_trials):
            ct = sample_tree(5, small_pop, times)
            small_tmrca.append(ct[-1])
            ct = sample_tree(5, large_pop, times)
            large_tmrca.append(ct[-1])

        assert np.mean(small_tmrca) < np.mean(large_tmrca)


# ============================================================
# Tests: Emission (five cases)
# ============================================================

class TestEmissionCases:
    """Test the five emission cases from emission_probabilities.rst."""

    def test_case1_no_mutation(self):
        """Case 1: v = x = p => log P = -mu * t."""
        mu = 1e-8
        t = 1000.0
        log_p = -mu * t
        assert log_p < 0  # log prob should be negative
        assert abs(log_p - (-1e-5)) < 1e-10

    def test_case2_mutation_on_new_branch(self):
        """Case 2: v != p = x => log P = log(1/3 * (1 - exp(-mu*t)))."""
        mu = 1e-8
        t = 1000.0
        log_p = log(1.0/3 - 1.0/3 * exp(-mu * t))
        assert log_p < 0
        # For small mu*t, should be approximately log(mu*t/3)
        approx = log(mu * t / 3)
        assert abs(log_p - approx) / abs(approx) < 0.01

    def test_case1_dominates(self):
        """Case 1 should have higher probability than Case 2."""
        mu = 1e-8
        t = 1000.0
        case1 = -mu * t
        case2 = log(1.0/3 - 1.0/3 * exp(-mu * t))
        assert case1 > case2

    def test_case2_increases_with_time(self):
        """Case 2 probability should increase with time (more mutation opportunity)."""
        mu = 1e-8
        t1 = 500.0
        t2 = 5000.0
        case2_t1 = log(1.0/3 - 1.0/3 * exp(-mu * t1))
        case2_t2 = log(1.0/3 - 1.0/3 * exp(-mu * t2))
        assert case2_t2 > case2_t1

    def test_case3_mutation_on_lower_segment(self):
        """Case 3: v = p != x => involves ratio of segment probabilities."""
        mu = 1e-8
        parent_age = 5000.0
        node_age = 0.0
        time = 2500.0  # join point in the middle
        mintime = 1.0

        t1 = max(parent_age - node_age, mintime)  # full branch
        t2 = max(time - node_age, mintime)  # lower segment

        log_p = log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                     * exp(-mu * (time + t2 - t1)))
        assert np.isfinite(log_p)
        assert log_p < 0

    def test_case4_mutation_on_upper_segment(self):
        """Case 4: v = x != p => similar form to Case 3."""
        mu = 1e-8
        parent_age = 5000.0
        node_age = 0.0
        time = 2500.0
        mintime = 1.0

        t1 = max(parent_age - node_age, mintime)
        t2 = max(parent_age - time, mintime)

        log_p = log((1 - exp(-mu * t2)) / (1 - exp(-mu * t1))
                     * exp(-mu * (time + t2 - t1)))
        assert np.isfinite(log_p)
        assert log_p < 0

    def test_case5_two_mutations_rarest(self):
        """Case 5 (two mutations) should be rarer than Case 1 (no mutation)
        when the branch lengths are large enough for the two-mutation
        probability to be well-defined and small."""
        mu = 1e-8
        time = 1000.0
        parent_age = 5000.0
        node_age = 0.0
        mintime = 1.0

        t1 = max(parent_age - node_age, mintime)
        t2a = max(parent_age - time, mintime)
        t2b = max(time - node_age, mintime)
        t2 = max(t2a, t2b)
        t3 = time

        case5 = log((1 - exp(-mu * t2)) * (1 - exp(-mu * t3))
                     / (1 - exp(-mu * t1))
                     * exp(-mu * (time + t2 + t3 - t1)))

        case1 = -mu * time

        # Case 5 (two mutations) should be rarer than case 1 (no mutation)
        assert case5 < case1
        # Case 5 should be finite and negative (valid log-probability)
        assert case5 < 0
        assert not np.isinf(case5)


# ============================================================
# Tests: Overall consistency
# ============================================================

class TestTimeGridConsistency:
    def test_coal_times_contain_boundaries(self):
        """Coal times should contain the original boundary times."""
        times = get_time_points(ntimes=10)
        ct = get_coal_times(times)
        ntimes = len(times) - 1
        for i in range(ntimes + 1):
            assert abs(ct[2 * i] - times[i]) < 1e-10

    def test_different_ntimes(self):
        """Grid should work for various numbers of time points."""
        for nt in [5, 10, 20, 50]:
            times = get_time_points(ntimes=nt, maxtime=100000)
            assert len(times) == nt
            assert times[0] == 0.0
            assert abs(times[-1] - 100000.0) < 1e-6

            steps = get_time_steps(times)
            assert len(steps) == nt - 1
            assert all(s > 0 for s in steps)

    def test_uniform_limit(self):
        """For very small delta, spacing should approach uniform."""
        times = get_time_points(ntimes=10, maxtime=1000, delta=1e-10)
        steps = get_time_steps(times)
        # All steps should be approximately equal
        mean_step = sum(steps) / len(steps)
        for s in steps:
            assert abs(s - mean_step) / mean_step < 0.05
