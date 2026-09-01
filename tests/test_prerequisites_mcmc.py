"""Independent numerical and source-parity checks for the MCMC prerequisite."""

from pathlib import Path

import numpy as np
import pytest


RST = Path(__file__).parents[1] / "docs" / "prerequisites" / "mcmc.rst"


def execute_rst_python_blocks(path=RST):
    """Execute every published Python block in one shared namespace."""
    lines = Path(path).read_text().splitlines()
    namespace = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() != ".. code-block:: python":
            i += 1
            continue
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        indent = len(lines[i]) - len(lines[i].lstrip())
        block = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) < indent:
                break
            block.append(line[indent:] if line.strip() else "")
            i += 1
        exec("\n".join(block), namespace)
    return namespace


def random_walk_metropolis(log_target, initial, proposal_scale, n_samples, rng):
    if n_samples < 2 or proposal_scale <= 0 or not np.isfinite(initial):
        raise ValueError("invalid sampler arguments")
    current_logp = float(log_target(initial))
    if not np.isfinite(current_logp):
        raise ValueError("initial state must have finite target density")
    samples = np.empty(n_samples)
    samples[0] = initial
    accepted = 0
    for t in range(1, n_samples):
        proposal = samples[t - 1] + rng.normal(scale=proposal_scale)
        proposal_logp = float(log_target(proposal))
        log_ratio = proposal_logp - current_logp
        if np.log(rng.random()) < min(0.0, log_ratio):
            samples[t] = proposal
            current_logp = proposal_logp
            accepted += 1
        else:
            samples[t] = samples[t - 1]
    return samples, accepted / (n_samples - 1)


def beta_log_kernel(theta, alpha, beta):
    if not 0.0 < theta < 1.0:
        return -np.inf
    return (alpha - 1) * np.log(theta) + (beta - 1) * np.log1p(-theta)


def compute_acf_and_ess(chain, max_lag=200):
    chain = np.asarray(chain, dtype=float)
    if chain.ndim != 1 or len(chain) < 4 or not np.all(np.isfinite(chain)):
        raise ValueError("chain must be a finite one-dimensional array")
    n = len(chain)
    max_lag = min(int(max_lag), n - 1)
    if max_lag < 2:
        raise ValueError("max_lag must allow at least one lag pair")
    centered = chain - chain.mean()
    var = np.dot(centered, centered) / n
    if var == 0:
        return np.ones(max_lag + 1), 1.0
    acf = np.ones(max_lag + 1)
    for k in range(1, max_lag + 1):
        acf[k] = np.dot(centered[:-k], centered[k:]) / ((n - k) * var)
    paired_sum = 0.0
    previous_pair = np.inf
    for k in range(1, max_lag, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        pair = min(pair, previous_pair)
        paired_sum += pair
        previous_pair = pair
    iat = max(1.0, 1.0 + 2.0 * paired_sum)
    return acf, min(float(n), n / iat)


def gibbs_bivariate_normal(n_samples, rho, seed):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, 2))
    cond_std = np.sqrt(1 - rho**2)
    for t in range(1, n_samples):
        samples[t, 0] = rng.normal(rho * samples[t - 1, 1], cond_std)
        samples[t, 1] = rng.normal(rho * samples[t, 0], cond_std)
    return samples


class TestAnalyticOracles:
    def test_beta_binomial_posterior_parameters_and_moments(self):
        a, b = 2 + 7, 2 + (20 - 7)
        assert (a, b) == (9, 15)
        assert a / (a + b) == pytest.approx(0.375)
        assert a * b / ((a + b) ** 2 * (a + b + 1)) == pytest.approx(0.009375)

    def test_finite_chain_stationarity_and_detailed_balance(self):
        pi = np.array([0.2, 0.3, 0.5])
        transition = np.array(
            [[0.50, 0.20, 0.30], [2 / 15, 17 / 30, 0.30], [0.12, 0.18, 0.70]]
        )
        np.testing.assert_allclose(transition.sum(axis=1), 1)
        np.testing.assert_allclose(pi @ transition, pi)
        np.testing.assert_allclose(pi[:, None] * transition,
                                   (pi[:, None] * transition).T)

    def test_finite_chain_empirical_distribution(self):
        pi = np.array([0.2, 0.3, 0.5])
        transition = np.array(
            [[0.50, 0.20, 0.30], [2 / 15, 17 / 30, 0.30], [0.12, 0.18, 0.70]]
        )
        rng = np.random.default_rng(310)
        state = 0
        counts = np.zeros(3)
        for _ in range(100_000):
            state = rng.choice(3, p=transition[state])
            counts[state] += 1
        np.testing.assert_allclose(counts / counts.sum(), pi, atol=0.008)


class TestMetropolisHastings:
    @pytest.mark.parametrize("args", [(0.5, 0, 10), (0.5, 0.1, 1), (np.nan, 0.1, 10)])
    def test_invalid_sampler_arguments_rejected(self, args):
        initial, scale, n = args
        with pytest.raises(ValueError):
            random_walk_metropolis(lambda x: -x * x, initial, scale, n,
                                   np.random.default_rng(1))

    def test_initial_state_outside_support_rejected(self):
        with pytest.raises(ValueError):
            random_walk_metropolis(lambda x: beta_log_kernel(x, 9, 15), -1, 0.1,
                                   10, np.random.default_rng(1))

    def test_beta_target_matches_exact_moments_across_chains(self):
        a, b = 9, 15
        exact_mean = a / (a + b)
        exact_var = a * b / ((a + b) ** 2 * (a + b + 1))
        pooled = []
        rates = []
        for seed, start in zip([17, 31, 47], [0.1, 0.5, 0.9]):
            chain, rate = random_walk_metropolis(
                lambda x: beta_log_kernel(x, a, b), start, 0.15, 40_000,
                np.random.default_rng(seed),
            )
            pooled.append(chain[5_000:])
            rates.append(rate)
        draws = np.concatenate(pooled)
        assert 0.2 < min(rates) < max(rates) < 0.9
        assert draws.mean() == pytest.approx(exact_mean, abs=0.004)
        assert draws.var() == pytest.approx(exact_var, abs=0.0005)

    def test_beta_quantiles_match_independent_direct_sampler(self):
        chain, _ = random_walk_metropolis(
            lambda x: beta_log_kernel(x, 9, 15), 0.5, 0.15, 80_000,
            np.random.default_rng(92),
        )
        oracle = np.random.default_rng(93).beta(9, 15, size=500_000)
        np.testing.assert_allclose(
            np.quantile(chain[10_000:], [0.05, 0.5, 0.95]),
            np.quantile(oracle, [0.05, 0.5, 0.95]), atol=0.008,
        )

    def test_hastings_correction_restores_asymmetric_proposal_balance(self):
        target = np.array([0.15, 0.35, 0.50])
        proposal = np.array([[0.2, 0.7, 0.1], [0.1, 0.2, 0.7], [0.6, 0.2, 0.2]])
        transition = np.zeros_like(proposal)
        for i in range(3):
            for j in range(3):
                if i != j:
                    alpha = min(1.0, target[j] * proposal[j, i]
                                / (target[i] * proposal[i, j]))
                    transition[i, j] = proposal[i, j] * alpha
            transition[i, i] = 1 - transition[i].sum()
        np.testing.assert_allclose(target @ transition, target)
        np.testing.assert_allclose(target[:, None] * transition,
                                   (target[:, None] * transition).T)


class TestGibbsAndDiagnostics:
    def test_gibbs_recovers_bivariate_normal_covariance(self):
        rho = 0.8
        samples = gibbs_bivariate_normal(100_000, rho, 73)[5_000:]
        np.testing.assert_allclose(samples.mean(axis=0), 0, atol=0.025)
        np.testing.assert_allclose(np.cov(samples, rowvar=False, ddof=0),
                                   [[1, rho], [rho, 1]], atol=0.035)

    def test_white_noise_ess_is_near_sample_count(self):
        draws = np.random.default_rng(4).normal(size=50_000)
        acf, ess = compute_acf_and_ess(draws, 400)
        assert acf[0] == 1
        assert ess / len(draws) > 0.90

    @pytest.mark.parametrize("rho", [0.3, 0.7])
    def test_ar1_ess_matches_closed_form(self, rho):
        rng = np.random.default_rng(round(100 * rho))
        n = 100_000
        x = np.empty(n)
        x[0] = rng.normal()
        innovations = rng.normal(scale=np.sqrt(1 - rho**2), size=n - 1)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + innovations[i - 1]
        _, ess = compute_acf_and_ess(x, 1_000)
        exact = n * (1 - rho) / (1 + rho)
        assert ess == pytest.approx(exact, rel=0.13)

    @pytest.mark.parametrize("bad", [[1, 1, 1], [[1, 2], [3, 4]], [1, np.nan, 2, 3]])
    def test_diagnostic_rejects_invalid_input(self, bad):
        with pytest.raises(ValueError):
            compute_acf_and_ess(bad)


def test_every_published_python_block_executes_and_exports_correct_helpers():
    namespace = execute_rst_python_blocks()
    assert namespace["alpha_post"] == 9
    assert namespace["beta_post"] == 15
    assert "markov_chain_convergence" in namespace
    docs_chain, _ = namespace["random_walk_metropolis"](
        lambda x: namespace["beta_log_kernel"](x, 9, 15), 0.5, 0.15, 100,
        np.random.default_rng(99),
    )
    test_chain, _ = random_walk_metropolis(
        lambda x: beta_log_kernel(x, 9, 15), 0.5, 0.15, 100,
        np.random.default_rng(99),
    )
    np.testing.assert_array_equal(docs_chain, test_chain)
    docs_acf, docs_ess = namespace["compute_acf_and_ess"](test_chain, 20)
    test_acf, test_ess = compute_acf_and_ess(test_chain, 20)
    np.testing.assert_allclose(docs_acf, test_acf)
    assert docs_ess == test_ess
