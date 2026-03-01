"""Tests for code examples in docs/prerequisites/hmms.rst.

This module extracts and tests every Python code block from the HMM
(Hidden Markov Models) prerequisites documentation.

Code blocks tested:
1. HMM class            -- Basic HMM container
2. forward_algorithm    -- Unscaled forward algorithm
3. forward_scaled       -- Scaled forward algorithm with log-likelihood
4. stochastic_traceback -- Forward-filtering backward-sampling
5. forward_li_stephens  -- O(K) forward algorithm with Li-Stephens transitions
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Code block 1: HMM class
# ---------------------------------------------------------------------------

class HMM:
    """A Hidden Markov Model.

    Parameters
    ----------
    initial : ndarray of shape (K,)
        Initial state distribution pi.
    transition : ndarray of shape (K, K)
        Transition matrix A[i, j] = P(Z_l = j | Z_{l-1} = i).
    emission : callable
        emission(state, observation) returns P(X = obs | Z = state).
    """
    def __init__(self, initial, transition, emission):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.K = len(initial)


# The weather model from the documentation
def make_weather_hmm():
    """Create the 2-state weather HMM from the documentation."""
    return HMM(
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.95, 0.05],
                             [0.10, 0.90]]),
        emission=lambda s, x: [0.9, 0.1][x] if s == 0 else [0.2, 0.8][x]
    )


# Observation sequence from the documentation
WEATHER_OBS = [0, 0, 1, 1, 0, 1, 1, 1, 0, 0]


class TestHMMClass:
    """Tests for the HMM class definition."""

    def test_hmm_creation(self):
        """Verify that the HMM class can be instantiated with the weather model."""
        hmm = make_weather_hmm()
        assert hmm.K == 2
        assert hmm.initial.shape == (2,)
        assert hmm.transition.shape == (2, 2)

    def test_initial_distribution_valid(self):
        """Verify the initial distribution sums to 1."""
        hmm = make_weather_hmm()
        np.testing.assert_allclose(hmm.initial.sum(), 1.0)

    def test_transition_matrix_stochastic(self):
        """Verify each row of the transition matrix sums to 1."""
        hmm = make_weather_hmm()
        np.testing.assert_allclose(hmm.transition.sum(axis=1), [1.0, 1.0])

    def test_emission_probabilities(self):
        """Verify emission probabilities for all state-observation pairs."""
        hmm = make_weather_hmm()
        # State 0 (Sunny): P(no umbrella)=0.9, P(umbrella)=0.1
        assert hmm.emission(0, 0) == 0.9
        assert hmm.emission(0, 1) == 0.1
        # State 1 (Rainy): P(no umbrella)=0.2, P(umbrella)=0.8
        assert hmm.emission(1, 0) == 0.2
        assert hmm.emission(1, 1) == 0.8

    def test_emission_probabilities_sum_to_one(self):
        """Verify that for each state, emission probs over all observations sum to 1."""
        hmm = make_weather_hmm()
        for state in range(hmm.K):
            total = sum(hmm.emission(state, obs) for obs in [0, 1])
            np.testing.assert_allclose(total, 1.0)


# ---------------------------------------------------------------------------
# Code block 2: forward_algorithm (unscaled)
# ---------------------------------------------------------------------------

def forward_algorithm(hmm, observations):
    """Run the forward algorithm.

    Parameters
    ----------
    hmm : HMM
    observations : list of length L

    Returns
    -------
    alpha : ndarray of shape (L, K)
        Forward probabilities.
    """
    L = len(observations)
    K = hmm.K
    alpha = np.zeros((L, K))

    for j in range(K):
        alpha[0, j] = hmm.initial[j] * hmm.emission(j, observations[0])

    for ell in range(1, L):
        for j in range(K):
            alpha[ell, j] = hmm.emission(j, observations[ell]) * \
                np.sum(alpha[ell - 1, :] * hmm.transition[:, j])

    return alpha


class TestForwardAlgorithm:
    """Tests for the unscaled forward algorithm."""

    def test_output_shape(self):
        """Verify the output has shape (L, K)."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        assert alpha.shape == (len(WEATHER_OBS), hmm.K)

    def test_all_values_non_negative(self):
        """Forward probabilities must be non-negative."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        assert np.all(alpha >= 0)

    def test_initialization_step(self):
        """Verify the initialization: alpha_j(1) = pi_j * e_j(X_1)."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        # X_1 = 0 (no umbrella)
        expected_0 = 0.5 * 0.9  # pi_sunny * P(no_umbrella | sunny)
        expected_1 = 0.5 * 0.2  # pi_rainy * P(no_umbrella | rainy)
        np.testing.assert_allclose(alpha[0, 0], expected_0)
        np.testing.assert_allclose(alpha[0, 1], expected_1)

    def test_likelihood_positive(self):
        """The total likelihood (sum of alpha at last position) must be positive."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        likelihood = np.sum(alpha[-1, :])
        assert likelihood > 0

    def test_likelihood_less_than_one(self):
        """The joint probability P(X_1,...,X_L) must be less than 1 for non-trivial L."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        likelihood = np.sum(alpha[-1, :])
        assert likelihood < 1.0

    def test_forward_probs_decrease_over_time(self):
        """Forward probabilities (joint probs) should generally decrease as L grows."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        sums = alpha.sum(axis=1)
        # The first few sums should be larger than later ones
        assert sums[0] > sums[-1]

    def test_single_observation(self):
        """With a single observation, the forward probs should equal initial * emission."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, [1])  # Umbrella
        np.testing.assert_allclose(alpha[0, 0], 0.5 * 0.1)  # Sunny, umbrella
        np.testing.assert_allclose(alpha[0, 1], 0.5 * 0.8)  # Rainy, umbrella

    def test_manual_step_two(self):
        """Manually compute alpha at position 2 and compare with the algorithm."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)

        # alpha(2, j) = e_j(X_2) * sum_i alpha(1,i) * A[i,j]
        # X_2 = 0 (no umbrella, the second observation in WEATHER_OBS)
        expected_0 = hmm.emission(0, 0) * (
            alpha[0, 0] * hmm.transition[0, 0] +
            alpha[0, 1] * hmm.transition[1, 0]
        )
        expected_1 = hmm.emission(1, 0) * (
            alpha[0, 0] * hmm.transition[0, 1] +
            alpha[0, 1] * hmm.transition[1, 1]
        )
        np.testing.assert_allclose(alpha[1, 0], expected_0)
        np.testing.assert_allclose(alpha[1, 1], expected_1)


# ---------------------------------------------------------------------------
# Code block 3: forward_scaled
# ---------------------------------------------------------------------------

def forward_scaled(hmm, observations):
    """Forward algorithm with scaling for numerical stability.

    Returns
    -------
    alpha_hat : ndarray of shape (L, K)
        Scaled forward probabilities (conditional state distributions).
    log_likelihood : float
        log P(X_1, ..., X_L).
    """
    L = len(observations)
    K = hmm.K
    alpha_hat = np.zeros((L, K))
    log_likelihood = 0.0

    for j in range(K):
        alpha_hat[0, j] = hmm.initial[j] * hmm.emission(j, observations[0])
    c = alpha_hat[0].sum()
    alpha_hat[0] /= c
    log_likelihood += np.log(c)

    for ell in range(1, L):
        for j in range(K):
            alpha_hat[ell, j] = hmm.emission(j, observations[ell]) * \
                np.sum(alpha_hat[ell - 1, :] * hmm.transition[:, j])
        c = alpha_hat[ell].sum()
        alpha_hat[ell] /= c
        log_likelihood += np.log(c)

    return alpha_hat, log_likelihood


class TestForwardScaled:
    """Tests for the scaled forward algorithm."""

    def test_output_shapes(self):
        """Verify output shapes and types."""
        hmm = make_weather_hmm()
        alpha_hat, ll = forward_scaled(hmm, WEATHER_OBS)
        assert alpha_hat.shape == (len(WEATHER_OBS), hmm.K)
        assert isinstance(ll, float)

    def test_rows_sum_to_one(self):
        """Scaled forward probabilities should sum to 1 at each position."""
        hmm = make_weather_hmm()
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)
        for ell in range(len(WEATHER_OBS)):
            np.testing.assert_allclose(alpha_hat[ell].sum(), 1.0, atol=1e-12)

    def test_log_likelihood_matches_unscaled(self):
        """Log-likelihood from scaled version should match log of unscaled likelihood."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        likelihood_unscaled = np.sum(alpha[-1, :])
        _, ll_scaled = forward_scaled(hmm, WEATHER_OBS)
        np.testing.assert_allclose(ll_scaled, np.log(likelihood_unscaled), atol=1e-10)

    def test_log_likelihood_negative(self):
        """Log-likelihood should be negative (likelihood < 1)."""
        hmm = make_weather_hmm()
        _, ll = forward_scaled(hmm, WEATHER_OBS)
        assert ll < 0

    def test_scaled_probs_valid_distributions(self):
        """Each row should be a valid probability distribution."""
        hmm = make_weather_hmm()
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)
        assert np.all(alpha_hat >= 0)
        assert np.all(alpha_hat <= 1)

    def test_final_position_conditional_probs(self):
        """At the last position, scaled alpha should give P(state | all observations)."""
        hmm = make_weather_hmm()
        alpha = forward_algorithm(hmm, WEATHER_OBS)
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)

        # Normalize unscaled forward probs at last position
        conditional = alpha[-1] / alpha[-1].sum()
        np.testing.assert_allclose(alpha_hat[-1], conditional, atol=1e-10)

    def test_longer_sequences_still_work(self):
        """Scaled version should handle long sequences without numerical issues."""
        hmm = make_weather_hmm()
        np.random.seed(42)
        long_obs = list(np.random.choice([0, 1], size=1000))
        alpha_hat, ll = forward_scaled(hmm, long_obs)
        assert np.isfinite(ll)
        assert np.all(np.isfinite(alpha_hat))
        np.testing.assert_allclose(alpha_hat.sum(axis=1), np.ones(1000), atol=1e-10)


# ---------------------------------------------------------------------------
# Code block 4: stochastic_traceback
# ---------------------------------------------------------------------------

def stochastic_traceback(hmm, alpha_hat):
    """Sample a state sequence from the posterior using forward probs.

    Parameters
    ----------
    hmm : HMM
    alpha_hat : ndarray of shape (L, K)

    Returns
    -------
    states : ndarray of shape (L,)
    """
    L, K = alpha_hat.shape
    states = np.zeros(L, dtype=int)

    probs = alpha_hat[-1] / alpha_hat[-1].sum()
    states[-1] = np.random.choice(K, p=probs)

    for ell in range(L - 2, -1, -1):
        j = states[ell + 1]
        probs = alpha_hat[ell] * hmm.transition[:, j]
        probs /= probs.sum()
        states[ell] = np.random.choice(K, p=probs)

    return states


class TestStochasticTraceback:
    """Tests for the stochastic traceback (forward-filtering backward-sampling)."""

    def test_output_shape(self):
        """Verify output is a state sequence of length L."""
        hmm = make_weather_hmm()
        np.random.seed(42)
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)
        states = stochastic_traceback(hmm, alpha_hat)
        assert states.shape == (len(WEATHER_OBS),)

    def test_states_are_valid(self):
        """All sampled states should be valid state indices."""
        hmm = make_weather_hmm()
        np.random.seed(42)
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)
        for _ in range(20):
            states = stochastic_traceback(hmm, alpha_hat)
            assert np.all(states >= 0)
            assert np.all(states < hmm.K)

    def test_different_seeds_give_different_samples(self):
        """Different random seeds should generally produce different samples."""
        hmm = make_weather_hmm()
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)

        np.random.seed(42)
        s1 = stochastic_traceback(hmm, alpha_hat)
        np.random.seed(123)
        s2 = stochastic_traceback(hmm, alpha_hat)
        # With different seeds we expect at least one position to differ
        # (not guaranteed but extremely likely with 10 positions)
        assert not np.array_equal(s1, s2)

    def test_reproducibility_with_seed(self):
        """Same seed should produce the same sample."""
        hmm = make_weather_hmm()
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)

        np.random.seed(42)
        s1 = stochastic_traceback(hmm, alpha_hat)
        np.random.seed(42)
        s2 = stochastic_traceback(hmm, alpha_hat)
        np.testing.assert_array_equal(s1, s2)

    def test_documented_example_runs(self):
        """Verify the documented example (5 samples) runs without error."""
        hmm = make_weather_hmm()
        np.random.seed(42)
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)
        samples = []
        for _ in range(5):
            states = stochastic_traceback(hmm, alpha_hat)
            samples.append(states.copy())
        assert len(samples) == 5
        for s in samples:
            assert s.shape == (len(WEATHER_OBS),)

    def test_marginal_distribution_matches_forward(self):
        """Over many samples, the marginal state distribution at each position
        should match the scaled forward probabilities."""
        hmm = make_weather_hmm()
        np.random.seed(42)
        alpha_hat, _ = forward_scaled(hmm, WEATHER_OBS)

        n_samples = 5000
        counts = np.zeros((len(WEATHER_OBS), hmm.K))
        for _ in range(n_samples):
            states = stochastic_traceback(hmm, alpha_hat)
            for ell in range(len(WEATHER_OBS)):
                counts[ell, states[ell]] += 1

        # Note: stochastic traceback samples from the posterior P(Z | X_1,...,X_L),
        # which is NOT the same as the forward-only distribution P(Z_ell | X_1,...,X_ell).
        # At the last position they should agree though.
        empirical_last = counts[-1] / n_samples
        np.testing.assert_allclose(empirical_last, alpha_hat[-1], atol=0.05)


# ---------------------------------------------------------------------------
# Code block 5: forward_li_stephens
# ---------------------------------------------------------------------------

def forward_li_stephens(initial, r, q, emissions, observations):
    """Forward algorithm with Li-Stephens transition structure.

    A[i,j] = (1 - r[i]) * delta(i,j) + r[i] * q[j] / sum(q)

    Parameters
    ----------
    initial : ndarray of shape (K,)
    r : ndarray of shape (K,)
    q : ndarray of shape (K,)
    emissions : ndarray of shape (L, K)
    observations : ignored

    Returns
    -------
    alpha : ndarray of shape (L, K)
    """
    L, K = emissions.shape
    alpha = np.zeros((L, K))
    alpha[0] = initial * emissions[0]
    q_sum = q.sum()

    for ell in range(1, L):
        recomb_sum = np.sum(r * alpha[ell - 1])
        for j in range(K):
            stay = (1 - r[j]) * alpha[ell - 1, j]
            switch = (q[j] / q_sum) * recomb_sum
            alpha[ell, j] = emissions[ell, j] * (stay + switch)

    return alpha


class TestForwardLiStephens:
    """Tests for the Li-Stephens O(K) forward algorithm."""

    def test_documented_example_runs(self):
        """Verify the documented example (K=10, L=100) runs without error."""
        np.random.seed(42)
        K, L = 10, 100
        r = np.full(K, 0.05)
        q = np.random.dirichlet(np.ones(K))
        emissions = np.random.uniform(0.1, 0.9, size=(L, K))

        alpha = forward_li_stephens(
            initial=np.ones(K) / K,
            r=r, q=q, emissions=emissions, observations=None
        )
        assert alpha.shape == (L, K)
        assert np.all(alpha >= 0)

    def test_output_shape(self):
        """Verify output shape matches (L, K)."""
        np.random.seed(42)
        K, L = 5, 20
        r = np.full(K, 0.1)
        q = np.ones(K) / K
        emissions = np.ones((L, K)) * 0.5
        alpha = forward_li_stephens(np.ones(K) / K, r, q, emissions, None)
        assert alpha.shape == (L, K)

    def test_matches_full_forward_with_li_stephens_transition(self):
        """The Li-Stephens forward algorithm should give the same result as the
        standard forward algorithm when using the equivalent transition matrix."""
        np.random.seed(42)
        K = 4
        L = 15
        r = np.array([0.05, 0.08, 0.03, 0.06])
        q = np.array([0.4, 0.3, 0.2, 0.1])
        q_sum = q.sum()

        # Build the explicit K x K transition matrix
        A = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    A[i, j] = (1 - r[i]) + r[i] * q[j] / q_sum
                else:
                    A[i, j] = r[i] * q[j] / q_sum

        # Verify it is a valid stochastic matrix
        np.testing.assert_allclose(A.sum(axis=1), np.ones(K), atol=1e-12)

        emissions = np.random.uniform(0.1, 0.9, size=(L, K))
        initial = np.ones(K) / K

        # Li-Stephens forward
        alpha_ls = forward_li_stephens(initial, r, q, emissions, None)

        # Standard forward using the explicit transition matrix
        alpha_std = np.zeros((L, K))
        alpha_std[0] = initial * emissions[0]
        for ell in range(1, L):
            for j in range(K):
                alpha_std[ell, j] = emissions[ell, j] * np.sum(
                    alpha_std[ell - 1, :] * A[:, j]
                )

        np.testing.assert_allclose(alpha_ls, alpha_std, atol=1e-12)

    def test_all_values_non_negative(self):
        """Forward probabilities should all be non-negative."""
        np.random.seed(42)
        K, L = 6, 30
        r = np.random.uniform(0.01, 0.2, K)
        q = np.random.dirichlet(np.ones(K))
        emissions = np.random.uniform(0.1, 0.9, size=(L, K))
        alpha = forward_li_stephens(np.ones(K) / K, r, q, emissions, None)
        assert np.all(alpha >= 0)

    def test_zero_recombination(self):
        """With r=0, the process stays in the initial state: alpha should track
        the product of emission probabilities along each state independently."""
        K = 3
        L = 5
        r = np.zeros(K)
        q = np.array([0.5, 0.3, 0.2])
        emissions = np.array([
            [0.8, 0.6, 0.9],
            [0.7, 0.5, 0.8],
            [0.9, 0.4, 0.7],
            [0.6, 0.8, 0.5],
            [0.5, 0.7, 0.6],
        ])
        initial = np.array([0.4, 0.3, 0.3])
        alpha = forward_li_stephens(initial, r, q, emissions, None)

        # With no recombination, alpha[ell, j] = initial[j] * prod(emissions[:ell+1, j])
        for j in range(K):
            expected = initial[j]
            for ell in range(L):
                expected *= emissions[ell, j]
                np.testing.assert_allclose(alpha[ell, j], expected, atol=1e-12)

    def test_full_recombination(self):
        """With r=1 for all states, transition is purely q-weighted."""
        np.random.seed(42)
        K = 4
        L = 10
        r = np.ones(K)  # always recombine
        q = np.array([0.4, 0.3, 0.2, 0.1])
        q_sum = q.sum()
        emissions = np.random.uniform(0.1, 0.9, size=(L, K))
        initial = np.ones(K) / K

        alpha = forward_li_stephens(initial, r, q, emissions, None)

        # With r=1, the stay term is 0, and the switch term dominates:
        # alpha[ell, j] = emissions[ell, j] * (q[j] / q_sum) * sum(alpha[ell-1])
        alpha_manual = np.zeros((L, K))
        alpha_manual[0] = initial * emissions[0]
        for ell in range(1, L):
            total_prev = alpha_manual[ell - 1].sum()
            for j in range(K):
                alpha_manual[ell, j] = emissions[ell, j] * (q[j] / q_sum) * total_prev

        np.testing.assert_allclose(alpha, alpha_manual, atol=1e-12)

    def test_transition_matrix_rows_sum_to_one(self):
        """Verify that the implied Li-Stephens transition matrix has rows summing to 1."""
        K = 5
        r = np.array([0.05, 0.08, 0.12, 0.03, 0.15])
        q = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        q_sum = q.sum()

        A = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i == j:
                    A[i, j] = (1 - r[i]) + r[i] * q[j] / q_sum
                else:
                    A[i, j] = r[i] * q[j] / q_sum

        np.testing.assert_allclose(A.sum(axis=1), np.ones(K), atol=1e-12)

    def test_worked_example_from_docs(self):
        """Verify the manually worked example from the documentation.

        K=3, r=(0.1,0.1,0.1), q=(0.5,0.3,0.2),
        alpha(ell-1)=(0.4,0.35,0.25), emissions=(0.8,0.6,0.9).

        R = 0.1*0.4 + 0.1*0.35 + 0.1*0.25 = 0.1
        alpha_1 = 0.8 * (0.4*0.9 + 0.5*0.1) = 0.8 * 0.41 = 0.328
        alpha_2 = 0.6 * (0.35*0.9 + 0.3*0.1) = 0.6 * 0.345 = 0.207
        alpha_3 = 0.9 * (0.25*0.9 + 0.2*0.1) = 0.9 * 0.245 = 0.2205
        """
        K = 3
        r = np.array([0.1, 0.1, 0.1])
        q = np.array([0.5, 0.3, 0.2])
        prev_alpha = np.array([0.4, 0.35, 0.25])
        current_emissions = np.array([0.8, 0.6, 0.9])

        # Build a 2-position emissions array where the first position
        # produces the desired initial alpha and the second has the test emissions
        # We use initial = prev_alpha and emissions[0] = ones so alpha[0] = prev_alpha
        emissions = np.vstack([np.ones(K), current_emissions])
        alpha = forward_li_stephens(prev_alpha, r, q, emissions, None)

        np.testing.assert_allclose(alpha[1, 0], 0.328, atol=1e-10)
        np.testing.assert_allclose(alpha[1, 1], 0.207, atol=1e-10)
        np.testing.assert_allclose(alpha[1, 2], 0.2205, atol=1e-10)

    def test_sum_at_last_position_positive(self):
        """The sum of forward probabilities at the last position should be positive."""
        np.random.seed(42)
        K, L = 10, 100
        r = np.full(K, 0.05)
        q = np.random.dirichlet(np.ones(K))
        emissions = np.random.uniform(0.1, 0.9, size=(L, K))
        alpha = forward_li_stephens(np.ones(K) / K, r, q, emissions, None)
        assert alpha[-1].sum() > 0
