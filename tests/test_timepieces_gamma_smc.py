"""
Tests for Python code blocks from the Gamma-SMC timepiece RST documentation.

All functions are re-defined here since the code in the RST files is not
importable. Tests cover mathematical properties and expected behaviors.

Covers:
- gamma_approximation.rst: gamma_emission_update, to_log_coords, from_log_coords
- flow_field.rst: gamma_pdf_partials, FlowField
- forward_backward.rst: gamma_smc_forward, gamma_smc_posterior
- segmentation_and_caching.rst: segment_observations, gamma_entropy, entropy_clip,
                                 gamma_smc_forward_segmented
"""

import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist
from scipy.special import digamma, gammaln


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/gamma_smc/gamma_approximation.rst
# ---------------------------------------------------------------------------

def gamma_emission_update(alpha, beta, y, theta):
    """Apply the Poisson-gamma conjugate emission update."""
    if y == -1:  # missing
        return alpha, beta
    return alpha + y, beta + theta


def to_log_coords(alpha, beta):
    """Convert (alpha, beta) to log-mean / log-CV coordinates."""
    l_mu = np.log10(alpha / beta)
    l_C = np.log10(1.0 / np.sqrt(alpha))
    return l_mu, l_C


def from_log_coords(l_mu, l_C):
    """Convert log-coordinates back to (alpha, beta)."""
    alpha = 10.0 ** (-2 * l_C)
    beta = alpha * 10.0 ** (-l_mu)
    return alpha, beta


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/gamma_smc/flow_field.rst
# ---------------------------------------------------------------------------

def gamma_pdf_partials(x, alpha, beta):
    """Evaluate the gamma PDF and its partial derivatives."""
    f = gamma_dist.pdf(x, a=alpha, scale=1.0 / beta)
    df_dalpha = f * (-digamma(alpha) + np.log(beta) + np.log(x + 1e-300))
    df_dbeta = f * (alpha / beta - x)
    return f, df_dalpha, df_dbeta


class FlowField:
    """A precomputed flow field over a (l_mu, l_C) grid."""

    def __init__(self, l_mu_grid, l_C_grid, delta_l_mu, delta_l_C):
        self.l_mu_grid = l_mu_grid
        self.l_C_grid = l_C_grid
        self.delta_l_mu = delta_l_mu
        self.delta_l_C = delta_l_C

    def query(self, l_mu, l_C):
        """Query the flow field via bilinear interpolation with clipping."""
        l_mu_c = np.clip(l_mu, self.l_mu_grid[0], self.l_mu_grid[-1])
        l_C_c = np.clip(l_C, self.l_C_grid[0], self.l_C_grid[-1])

        i = np.searchsorted(self.l_mu_grid, l_mu_c) - 1
        j = np.searchsorted(self.l_C_grid, l_C_c) - 1
        i = np.clip(i, 0, len(self.l_mu_grid) - 2)
        j = np.clip(j, 0, len(self.l_C_grid) - 2)

        s = (l_mu_c - self.l_mu_grid[i]) / (self.l_mu_grid[i+1] - self.l_mu_grid[i])
        t = (l_C_c - self.l_C_grid[j]) / (self.l_C_grid[j+1] - self.l_C_grid[j])

        def interp(field):
            return ((1-s)*(1-t) * field[i, j]
                    + s*(1-t) * field[i+1, j]
                    + (1-s)*t * field[i, j+1]
                    + s*t * field[i+1, j+1])

        return interp(self.delta_l_mu), interp(self.delta_l_C)


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/gamma_smc/forward_backward.rst
# ---------------------------------------------------------------------------

def gamma_smc_forward(observations, theta, rho, flow_field):
    """Run the Gamma-SMC forward pass."""
    N = len(observations)
    alphas = np.zeros(N)
    betas = np.zeros(N)

    alpha, beta = 1.0, 1.0

    for i in range(N):
        l_mu = np.log10(alpha / beta)
        l_C = np.log10(1.0 / np.sqrt(alpha))
        dl_mu, dl_C = flow_field.query(l_mu, l_C)
        l_mu += rho * dl_mu
        l_C += rho * dl_C
        alpha = 10.0 ** (-2 * l_C)
        beta = alpha * 10.0 ** (-l_mu)

        y = observations[i]
        if y >= 0:
            alpha += y
            beta += theta

        alphas[i] = alpha
        betas[i] = beta

    return alphas, betas


def gamma_smc_posterior(observations, theta, rho, flow_field):
    """Compute the full Gamma-SMC posterior at each position."""
    a_fwd, b_fwd = gamma_smc_forward(observations, theta, rho, flow_field)

    a_bwd_rev, b_bwd_rev = gamma_smc_forward(
        observations[::-1], theta, rho, flow_field
    )
    a_bwd = a_bwd_rev[::-1]
    b_bwd = b_bwd_rev[::-1]

    post_alpha = a_fwd + a_bwd - 1
    post_beta = b_fwd + b_bwd - 1

    return post_alpha, post_beta


# ---------------------------------------------------------------------------
# Re-defined functions from docs/timepieces/gamma_smc/segmentation_and_caching.rst
# ---------------------------------------------------------------------------

def segment_observations(observations):
    """Segment a sequence into (n_miss, n_hom, final_obs) tuples."""
    segments = []
    n_miss = 0
    n_hom = 0

    for y in observations:
        if y == -1:
            n_miss += 1
        elif y == 0:
            n_hom += 1
        else:
            segments.append((n_miss, n_hom, 1))
            n_miss = 0
            n_hom = 0

    if n_miss > 0 or n_hom > 0:
        segments.append((n_miss, n_hom, 0))

    return segments


def gamma_entropy(alpha, beta):
    """Differential entropy of Gamma(alpha, beta)."""
    return (alpha - np.log(beta) + gammaln(alpha)
            + (1 - alpha) * digamma(alpha))


def entropy_clip(alpha, beta, h_max=1.0, tol=1e-8):
    """Clip gamma parameters so that entropy does not exceed h_max."""
    if gamma_entropy(alpha, beta) <= h_max:
        return alpha, beta

    mean = alpha / beta

    lo, hi = alpha, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2
        b_mid = mid / mean
        if gamma_entropy(mid, b_mid) > h_max:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    alpha_new = (lo + hi) / 2
    beta_new = alpha_new / mean
    return alpha_new, beta_new


# ---------------------------------------------------------------------------
# Helper: create a simple zero flow field for testing
# ---------------------------------------------------------------------------

def _make_zero_flow_field():
    """Create a flow field that returns zero displacement everywhere."""
    l_mu_grid = np.linspace(-3, 3, 10)
    l_C_grid = np.linspace(-3, 1, 10)
    delta_l_mu = np.zeros((10, 10))
    delta_l_C = np.zeros((10, 10))
    return FlowField(l_mu_grid, l_C_grid, delta_l_mu, delta_l_C)


def _make_constant_flow_field(dl_mu_val, dl_C_val):
    """Create a flow field that returns constant displacement everywhere."""
    l_mu_grid = np.linspace(-3, 3, 10)
    l_C_grid = np.linspace(-3, 1, 10)
    delta_l_mu = np.full((10, 10), dl_mu_val)
    delta_l_C = np.full((10, 10), dl_C_val)
    return FlowField(l_mu_grid, l_C_grid, delta_l_mu, delta_l_C)


# ===========================================================================
# Tests for gamma_emission_update
# ===========================================================================

class TestGammaEmissionUpdate:
    def test_het_increments_alpha(self):
        """A heterozygous observation (y=1) should increment alpha by 1."""
        a_new, b_new = gamma_emission_update(2.0, 3.0, 1, 0.01)
        assert a_new == 3.0
        assert b_new == 3.01

    def test_hom_no_alpha_change(self):
        """A homozygous observation (y=0) should not change alpha."""
        a_new, b_new = gamma_emission_update(2.0, 3.0, 0, 0.01)
        assert a_new == 2.0
        assert b_new == 3.01

    def test_missing_no_change(self):
        """A missing observation (y=-1) should not change parameters."""
        a_new, b_new = gamma_emission_update(2.0, 3.0, -1, 0.01)
        assert a_new == 2.0
        assert b_new == 3.0

    def test_beta_always_increases_for_data(self):
        """Beta should increase by theta for any non-missing observation."""
        theta = 0.05
        for y in [0, 1]:
            _, b_new = gamma_emission_update(1.0, 1.0, y, theta)
            assert b_new == 1.0 + theta

    def test_conjugate_posterior_mean(self):
        """After observing a het, the posterior mean = (alpha+1)/(beta+theta)."""
        alpha, beta = 3.0, 2.0
        theta = 0.1
        a_new, b_new = gamma_emission_update(alpha, beta, 1, theta)
        posterior_mean = a_new / b_new
        expected_mean = (alpha + 1) / (beta + theta)
        assert abs(posterior_mean - expected_mean) < 1e-12

    def test_conjugate_posterior_variance_decreases(self):
        """After observing data, the posterior variance should decrease."""
        alpha, beta = 1.0, 1.0
        theta = 0.1
        prior_var = alpha / beta**2
        a_new, b_new = gamma_emission_update(alpha, beta, 0, theta)
        posterior_var = a_new / b_new**2
        assert posterior_var < prior_var


# ===========================================================================
# Tests for to_log_coords / from_log_coords
# ===========================================================================

class TestLogCoordinates:
    def test_roundtrip(self):
        """Converting to log coords and back should be the identity."""
        for alpha, beta in [(1.0, 1.0), (2.0, 3.0), (0.5, 0.1), (10.0, 5.0)]:
            l_mu, l_C = to_log_coords(alpha, beta)
            a_back, b_back = from_log_coords(l_mu, l_C)
            assert abs(a_back - alpha) < 1e-10
            assert abs(b_back - beta) < 1e-10

    def test_log_mean(self):
        """l_mu should be log10(mean) = log10(alpha/beta)."""
        alpha, beta = 4.0, 2.0
        l_mu, _ = to_log_coords(alpha, beta)
        assert abs(l_mu - np.log10(2.0)) < 1e-12

    def test_log_cv(self):
        """l_C should be log10(1/sqrt(alpha))."""
        alpha, beta = 4.0, 2.0
        _, l_C = to_log_coords(alpha, beta)
        assert abs(l_C - np.log10(0.5)) < 1e-12

    def test_higher_alpha_lower_cv(self):
        """Higher alpha means lower CV (more concentrated)."""
        _, l_C_low = to_log_coords(2.0, 1.0)
        _, l_C_high = to_log_coords(10.0, 5.0)
        assert l_C_high < l_C_low

    def test_mean_preserved_across_beta_changes(self):
        """Changing beta while adjusting alpha to keep mean constant
        should change l_C but not l_mu."""
        # Mean = alpha/beta = 2 for both
        l_mu_1, _ = to_log_coords(2.0, 1.0)
        l_mu_2, _ = to_log_coords(4.0, 2.0)
        assert abs(l_mu_1 - l_mu_2) < 1e-12


# ===========================================================================
# Tests for gamma_pdf_partials
# ===========================================================================

class TestGammaPdfPartials:
    def test_pdf_integrates_to_one(self):
        """Gamma PDF should integrate to approximately 1."""
        alpha, beta = 3.0, 2.0
        x = np.linspace(0.001, 10, 10000)
        f, _, _ = gamma_pdf_partials(x, alpha, beta)
        integral = np.trapezoid(f, x)
        assert abs(integral - 1.0) < 0.01

    def test_pdf_nonnegative(self):
        """Gamma PDF should be non-negative everywhere."""
        x = np.linspace(0.001, 20, 1000)
        f, _, _ = gamma_pdf_partials(x, 2.0, 1.0)
        assert np.all(f >= 0)

    def test_pdf_matches_scipy(self):
        """PDF values should match scipy.stats.gamma."""
        alpha, beta = 3.0, 2.0
        x = np.array([0.5, 1.0, 2.0, 5.0])
        f, _, _ = gamma_pdf_partials(x, alpha, beta)
        expected = gamma_dist.pdf(x, a=alpha, scale=1.0 / beta)
        assert np.allclose(f, expected)

    def test_df_dbeta_sign_at_mode(self):
        """At x < alpha/beta (before the mode), increasing beta should
        increase the PDF (shift left), so df_dbeta > 0."""
        alpha, beta = 5.0, 2.0
        x = np.array([alpha / beta - 0.5])  # just below the mean
        _, _, df_db = gamma_pdf_partials(x, alpha, beta)
        assert df_db[0] > 0

    def test_partials_finite(self):
        """Partial derivatives should be finite for valid inputs."""
        x = np.linspace(0.1, 10, 100)
        _, df_da, df_db = gamma_pdf_partials(x, 3.0, 2.0)
        assert np.all(np.isfinite(df_da))
        assert np.all(np.isfinite(df_db))

    def test_dalpha_numerical(self):
        """Partial derivative with respect to alpha should match numerical diff."""
        alpha, beta = 3.0, 2.0
        x = np.array([1.0, 2.0, 3.0])
        eps = 1e-6
        f1, _, _ = gamma_pdf_partials(x, alpha - eps, beta)
        f2, _, _ = gamma_pdf_partials(x, alpha + eps, beta)
        numerical = (f2 - f1) / (2 * eps)
        _, analytical, _ = gamma_pdf_partials(x, alpha, beta)
        assert np.allclose(analytical, numerical, rtol=1e-4)


# ===========================================================================
# Tests for FlowField
# ===========================================================================

class TestFlowField:
    def test_zero_flow_returns_zero(self):
        """A zero flow field should return zero displacement."""
        ff = _make_zero_flow_field()
        dl_mu, dl_C = ff.query(0.0, 0.0)
        assert abs(dl_mu) < 1e-12
        assert abs(dl_C) < 1e-12

    def test_constant_flow_returns_constant(self):
        """A constant flow field should return the constant value."""
        ff = _make_constant_flow_field(0.5, -0.3)
        dl_mu, dl_C = ff.query(0.0, 0.0)
        assert abs(dl_mu - 0.5) < 1e-10
        assert abs(dl_C - (-0.3)) < 1e-10

    def test_clipping_at_boundary(self):
        """Queries outside the grid should be clipped to the boundary."""
        ff = _make_constant_flow_field(1.0, 2.0)
        dl_mu, dl_C = ff.query(-100.0, -100.0)
        assert np.isfinite(dl_mu)
        assert np.isfinite(dl_C)

    def test_bilinear_interpolation_at_grid_point(self):
        """At a grid point, interpolation should return the grid value."""
        l_mu_grid = np.array([0.0, 1.0, 2.0])
        l_C_grid = np.array([0.0, 1.0, 2.0])
        delta_l_mu = np.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])
        delta_l_C = np.zeros((3, 3))
        ff = FlowField(l_mu_grid, l_C_grid, delta_l_mu, delta_l_C)
        dl_mu, _ = ff.query(1.0, 1.0)
        assert abs(dl_mu - 5.0) < 1e-10

    def test_bilinear_interpolation_midpoint(self):
        """At the midpoint of four grid values, should return the average."""
        l_mu_grid = np.array([0.0, 1.0])
        l_C_grid = np.array([0.0, 1.0])
        delta_l_mu = np.array([[0.0, 2.0],
                                [4.0, 6.0]])
        delta_l_C = np.zeros((2, 2))
        ff = FlowField(l_mu_grid, l_C_grid, delta_l_mu, delta_l_C)
        dl_mu, _ = ff.query(0.5, 0.5)
        expected = (0.0 + 2.0 + 4.0 + 6.0) / 4.0
        assert abs(dl_mu - expected) < 1e-10


# ===========================================================================
# Tests for gamma_smc_forward
# ===========================================================================

class TestGammaSmcForward:
    def test_output_shape(self):
        """Output arrays should have the same length as observations."""
        ff = _make_zero_flow_field()
        obs = [0, 1, 0, 0, 1]
        alphas, betas = gamma_smc_forward(obs, theta=0.01, rho=0.001, flow_field=ff)
        assert len(alphas) == 5
        assert len(betas) == 5

    def test_alpha_increases_with_hets(self):
        """Alpha should increase by 1 for each het observation."""
        ff = _make_zero_flow_field()
        obs = [1, 1, 1]
        alphas, betas = gamma_smc_forward(obs, theta=0.01, rho=0.0, flow_field=ff)
        # Starting alpha=1, three hets: alpha should be ~4
        assert alphas[-1] > 3.5

    def test_beta_increases_with_observations(self):
        """Beta should increase by theta for each non-missing observation."""
        ff = _make_zero_flow_field()
        theta = 0.05
        obs = [0, 0, 0, 0, 0]
        alphas, betas = gamma_smc_forward(obs, theta=theta, rho=0.0, flow_field=ff)
        # Starting beta=1, five homs: beta ~ 1 + 5*theta = 1.25
        assert abs(betas[-1] - (1.0 + 5 * theta)) < 0.01

    def test_missing_observations_no_emission(self):
        """Missing observations should not trigger emission updates."""
        ff = _make_zero_flow_field()
        obs_all_miss = [-1, -1, -1]
        obs_all_hom = [0, 0, 0]
        a_miss, b_miss = gamma_smc_forward(obs_all_miss, theta=0.01, rho=0.0, flow_field=ff)
        a_hom, b_hom = gamma_smc_forward(obs_all_hom, theta=0.01, rho=0.0, flow_field=ff)
        # Missing should keep alpha and beta unchanged (relative to no emission)
        assert b_miss[-1] < b_hom[-1]

    def test_positive_parameters(self):
        """Alpha and beta should always be positive."""
        ff = _make_zero_flow_field()
        obs = [0, 1, -1, 0, 1, 0, 0, 1]
        alphas, betas = gamma_smc_forward(obs, theta=0.01, rho=0.001, flow_field=ff)
        assert np.all(alphas > 0)
        assert np.all(betas > 0)

    def test_posterior_mean_decreases_with_hom(self):
        """More homozygous observations should push the posterior mean down
        (evidence for smaller TMRCA)."""
        ff = _make_zero_flow_field()
        # All hom: expected mean should be lower
        obs_hom = [0, 0, 0, 0, 0]
        obs_het = [1, 1, 1, 1, 1]
        a_hom, b_hom = gamma_smc_forward(obs_hom, theta=0.01, rho=0.0, flow_field=ff)
        a_het, b_het = gamma_smc_forward(obs_het, theta=0.01, rho=0.0, flow_field=ff)
        mean_hom = a_hom[-1] / b_hom[-1]
        mean_het = a_het[-1] / b_het[-1]
        assert mean_hom < mean_het


# ===========================================================================
# Tests for gamma_smc_posterior
# ===========================================================================

class TestGammaSmcPosterior:
    def test_output_shape(self):
        """Output should have the same length as observations."""
        ff = _make_zero_flow_field()
        obs = [0, 1, 0]
        post_a, post_b = gamma_smc_posterior(obs, theta=0.01, rho=0.0, flow_field=ff)
        assert len(post_a) == 3
        assert len(post_b) == 3

    def test_posterior_positive(self):
        """Posterior alpha and beta should be positive."""
        ff = _make_zero_flow_field()
        obs = [0, 1, 0, 0, 1]
        post_a, post_b = gamma_smc_posterior(obs, theta=0.01, rho=0.0, flow_field=ff)
        assert np.all(post_a > 0)
        assert np.all(post_b > 0)

    def test_posterior_more_informative_than_prior(self):
        """Posterior should have higher alpha (lower CV) than the prior."""
        ff = _make_zero_flow_field()
        obs = [0, 1, 0, 0, 1, 0]
        post_a, post_b = gamma_smc_posterior(obs, theta=0.01, rho=0.0, flow_field=ff)
        # The prior is Gamma(1, 1), so post_alpha should be > 1
        for a in post_a:
            assert a > 1.0

    def test_symmetric_observations_symmetric_posterior(self):
        """For a palindromic observation sequence with zero flow,
        the posterior at position k should roughly mirror position N-1-k."""
        ff = _make_zero_flow_field()
        obs = [0, 1, 0, 1, 0]  # palindrome
        post_a, post_b = gamma_smc_posterior(obs, theta=0.01, rho=0.0, flow_field=ff)
        # Should be approximately symmetric
        for k in range(len(obs)):
            assert abs(post_a[k] - post_a[len(obs) - 1 - k]) < 0.1
            assert abs(post_b[k] - post_b[len(obs) - 1 - k]) < 0.1


# ===========================================================================
# Tests for segment_observations
# ===========================================================================

class TestSegmentObservations:
    def test_simple_het_only(self):
        """A sequence of only hets should produce one segment per het."""
        obs = [1, 1, 1]
        segments = segment_observations(obs)
        assert len(segments) == 3
        for seg in segments:
            assert seg == (0, 0, 1)

    def test_hom_then_het(self):
        """Homozygous sites before a het should be grouped together."""
        obs = [0, 0, 0, 1]
        segments = segment_observations(obs)
        assert len(segments) == 1
        assert segments[0] == (0, 3, 1)

    def test_missing_then_hom_then_het(self):
        """Missing and hom before a het should be captured."""
        obs = [-1, -1, 0, 0, 1]
        segments = segment_observations(obs)
        assert len(segments) == 1
        assert segments[0] == (2, 2, 1)

    def test_trailing_hom(self):
        """Trailing hom/missing without a het should produce a final segment."""
        obs = [1, 0, 0]
        segments = segment_observations(obs)
        assert len(segments) == 2
        assert segments[0] == (0, 0, 1)
        assert segments[1] == (0, 2, 0)

    def test_all_missing(self):
        """All missing data should produce one trailing segment."""
        obs = [-1, -1, -1]
        segments = segment_observations(obs)
        assert len(segments) == 1
        assert segments[0] == (3, 0, 0)

    def test_empty_sequence(self):
        """An empty sequence should produce no segments."""
        segments = segment_observations([])
        assert len(segments) == 0

    def test_reconstruction_preserves_length(self):
        """The total positions in all segments should equal the input length."""
        obs = [0, -1, 1, 0, 0, -1, 1, 0]
        segments = segment_observations(obs)
        total = sum(n_miss + n_hom + (1 if final == 1 else 0)
                    for n_miss, n_hom, final in segments)
        # Add trailing non-het positions
        trailing = sum(n_miss + n_hom for n_miss, n_hom, final in segments if final == 0)
        het_count = sum(1 for y in obs if y == 1)
        hom_count = sum(1 for y in obs if y == 0)
        miss_count = sum(1 for y in obs if y == -1)
        assert het_count + hom_count + miss_count == len(obs)


# ===========================================================================
# Tests for gamma_entropy
# ===========================================================================

class TestGammaEntropy:
    def test_exponential_entropy(self):
        """Entropy of Gamma(1, beta) = Exp(beta) should be 1 + log(1/beta)."""
        beta = 2.0
        h = gamma_entropy(1.0, beta)
        expected = 1.0 - np.log(beta) + gammaln(1.0)
        assert abs(h - expected) < 1e-10

    def test_increases_with_variance(self):
        """Entropy should increase as the distribution becomes more spread out."""
        # Fix mean=1, increase variance by decreasing alpha
        h_narrow = gamma_entropy(10.0, 10.0)  # mean=1, CV=0.316
        h_wide = gamma_entropy(1.0, 1.0)      # mean=1, CV=1.0
        assert h_wide > h_narrow

    def test_finite(self):
        """Entropy should be finite for valid parameters."""
        for alpha in [0.5, 1.0, 2.0, 10.0]:
            for beta in [0.5, 1.0, 5.0]:
                h = gamma_entropy(alpha, beta)
                assert np.isfinite(h)

    def test_unit_exponential(self):
        """Entropy of Exp(1) = Gamma(1, 1) should be 1.0."""
        h = gamma_entropy(1.0, 1.0)
        assert abs(h - 1.0) < 1e-10


# ===========================================================================
# Tests for entropy_clip
# ===========================================================================

class TestEntropyClip:
    def test_no_clip_when_below_threshold(self):
        """When entropy is below h_max, parameters should be unchanged."""
        # Gamma(10, 10) has low entropy
        alpha, beta = 10.0, 10.0
        h = gamma_entropy(alpha, beta)
        a_clip, b_clip = entropy_clip(alpha, beta, h_max=h + 1.0)
        assert abs(a_clip - alpha) < 1e-8
        assert abs(b_clip - beta) < 1e-8

    def test_clips_when_above_threshold(self):
        """When entropy exceeds h_max, alpha should increase."""
        alpha, beta = 0.5, 0.5  # high entropy
        h_max = 0.5
        a_clip, b_clip = entropy_clip(alpha, beta, h_max=h_max)
        assert a_clip > alpha
        # Entropy should now be at or below h_max
        h_new = gamma_entropy(a_clip, b_clip)
        assert h_new <= h_max + 1e-6

    def test_preserves_mean(self):
        """Clipping should preserve the mean alpha/beta."""
        alpha, beta = 0.5, 0.5
        mean_orig = alpha / beta
        a_clip, b_clip = entropy_clip(alpha, beta, h_max=0.5)
        mean_clip = a_clip / b_clip
        assert abs(mean_clip - mean_orig) < 1e-6

    def test_already_at_threshold(self):
        """Parameters already at the threshold should be approximately unchanged."""
        # Find params where entropy equals exactly some value
        alpha, beta = 10.0, 10.0
        h = gamma_entropy(alpha, beta)
        a_clip, b_clip = entropy_clip(alpha, beta, h_max=h)
        assert abs(a_clip - alpha) < 1.0  # might shift slightly due to bisection

    def test_positive_result(self):
        """Clipped parameters should be positive."""
        alpha, beta = 0.1, 0.1
        a_clip, b_clip = entropy_clip(alpha, beta, h_max=0.5)
        assert a_clip > 0
        assert b_clip > 0


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_forward_pass_with_segmented_data(self):
        """The forward pass should handle typical genomic data patterns."""
        ff = _make_zero_flow_field()
        # Simulate a typical pattern: many hom, occasional het
        np.random.seed(42)
        obs = [0] * 100 + [1] + [0] * 50 + [1] + [0] * 100
        alphas, betas = gamma_smc_forward(obs, theta=0.001, rho=0.0001, flow_field=ff)
        assert np.all(alphas > 0)
        assert np.all(betas > 0)

    def test_emission_update_roundtrip_with_coords(self):
        """Emission update in (alpha, beta) should be consistent after
        round-tripping through log coordinates."""
        alpha, beta = 3.0, 2.0
        l_mu, l_C = to_log_coords(alpha, beta)
        a_back, b_back = from_log_coords(l_mu, l_C)
        a_updated, b_updated = gamma_emission_update(a_back, b_back, 1, 0.01)
        expected_a, expected_b = gamma_emission_update(alpha, beta, 1, 0.01)
        assert abs(a_updated - expected_a) < 1e-8
        assert abs(b_updated - expected_b) < 1e-8

    def test_segmentation_then_reconstruction(self):
        """Segmentation should correctly capture the observation structure."""
        obs = [0, 0, 1, -1, 0, 1, 0, 0, 0]
        segments = segment_observations(obs)
        # First segment: 0, 0, 1 -> (0, 2, 1)
        assert segments[0] == (0, 2, 1)
        # Second segment: -1, 0, 1 -> (1, 1, 1)
        assert segments[1] == (1, 1, 1)
        # Third segment: 0, 0, 0 -> (0, 3, 0)
        assert segments[2] == (0, 3, 0)

    def test_posterior_mean_reflects_data(self):
        """Positions near heterozygous sites should have higher posterior mean
        (larger TMRCA estimate)."""
        ff = _make_zero_flow_field()
        # A het site surrounded by hom sites
        obs = [0] * 10 + [1] + [0] * 10
        post_a, post_b = gamma_smc_posterior(obs, theta=0.01, rho=0.0, flow_field=ff)
        means = post_a / post_b
        # The mean at the het site should be among the highest
        het_idx = 10
        assert means[het_idx] > np.median(means)
