"""Small, source-aligned kernels from PHLASH.

This is not a demographic inference program. It exposes the numerical ideas that
can be checked without PHLASH's JAX/CUDA data pipeline: its coalescent-rate
parameterisation, endpoint-randomised grid, AFS score, structured SMC' matrix-vector
product, a linear-memory Fisher score recursion, and an SVGD particle step.

Ground truth: PHLASH 1.0.6 (``jthlab/phlash``) and Terhorst (2025).
"""

import numpy as np


def effective_population_size(coalescent_rate):
    """Convert diploid coalescent rate ``eta(t)`` to ``N_e(t)``."""
    rate = np.asarray(coalescent_rate, dtype=float)
    if np.any(rate <= 0):
        raise ValueError("coalescent rates must be positive")
    return 0.5 / rate


def coalescence_probabilities(times, rates):
    """Discretise a piecewise-constant coalescent-time distribution.

    ``times`` contains left endpoints, beginning at zero. The final interval
    extends to infinity. The output order matches ``SizeHistory.p_coal`` in the
    official implementation: chronological finite intervals, then the open tail.
    """
    times = np.asarray(times, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates):
        raise ValueError("times and rates must be one-dimensional and equally sized")
    if len(times) == 0 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must start at zero and be strictly increasing")
    if np.any(rates <= 0):
        raise ValueError("rates must be positive")
    survival = np.r_[1.0, np.exp(-np.cumsum(rates[:-1] * np.diff(times))), 0.0]
    interval_mass = -np.diff(survival)
    return interval_mass


def logarithmic_grid(log_t1, log_tM, intervals=16):
    """Construct PHLASH's grid from its two random endpoint parameters."""
    if intervals < 2:
        raise ValueError("at least two intervals are required")
    t1, tM = np.exp([log_t1, log_tM])
    if t1 >= tM:
        raise ValueError("t1 must be smaller than tM")
    return np.r_[0.0, np.geomspace(t1, tM, intervals - 1)]


def afs_log_score(observed, expected, transform=None):
    """Return the AFS contribution used by ``phlash.model.log_density``.

    Both spectra may be linearly transformed (for example, folding and binning),
    after which the expected spectrum is normalized. Parameter-independent
    multinomial constants are omitted, exactly as in the source.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape or observed.ndim != 1:
        raise ValueError("observed and expected spectra must have the same 1-D shape")
    if np.any(observed < 0) or np.any(expected < 0) or expected.sum() <= 0:
        raise ValueError("spectra must be nonnegative and expected must have mass")
    expected = expected / expected.sum()
    if transform is not None:
        transform = np.asarray(transform, dtype=float)
        observed = transform @ observed
        expected = transform @ expected
    if np.any((observed > 0) & (expected <= 0)):
        return -np.inf
    positive = observed > 0
    return float(np.sum(observed[positive] * np.log(expected[positive])))


def composite_log_density(
    prior,
    psmc_log_likelihoods,
    afs_score=0.0,
    prior_weight=1.0,
    sequence_weight=1.0,
    afs_weight=1.0,
):
    """Combine the three terms used by the released sequence-data model."""
    return float(
        prior_weight * prior
        + sequence_weight * np.sum(psmc_log_likelihoods)
        + afs_weight * afs_score
    )


def structured_smc_matvec(vector, lower, diagonal, upper_scale, column_scale):
    """Compute ``vector @ transition`` using PHLASH's structured O(M) product.

    The four vectors are the ``b``, ``d``, ``u`` and ``v`` fields of the official
    ``PSMCParams`` object. This is a NumPy transcription of
    ``phlash.hmm.matvec_smc``.
    """
    vector, lower, diagonal, upper_scale, column_scale = (
        np.asarray(x, dtype=float)
        for x in (vector, lower, diagonal, upper_scale, column_scale)
    )
    n = len(vector)
    if any(x.shape != (n,) for x in (lower, diagonal, upper_scale, column_scale)):
        raise ValueError("all arguments must have the same one-dimensional shape")
    reverse_suffix = np.cumsum(np.r_[vector, 0.0][1:][::-1])[::-1]
    lower_part = reverse_suffix * lower
    upper_part = np.empty(n)
    state = 0.0
    for i, (u_i, value) in enumerate(zip(upper_scale, vector)):
        upper_part[i] = state * column_scale[i]
        state += u_i * value
    return lower_part + diagonal * vector + upper_part


def forward_log_likelihood(observations, transition, emission, initial):
    """Scaled HMM forward likelihood, retaining only O(M) state."""
    observations = np.asarray(observations, dtype=int)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    alpha = np.asarray(initial, dtype=float).copy()
    log_likelihood = 0.0
    for symbol in observations:
        alpha = (alpha @ transition) * emission[:, symbol]
        scale = alpha.sum()
        if scale <= 0:
            return -np.inf
        alpha /= scale
        log_likelihood += np.log(scale)
    return float(log_likelihood)


def fisher_transition_score(observations, transition, emission, initial, feature):
    """Evaluate one additive transition score with linear sequence memory.

    ``feature[i, j]`` is ``d log T[i,j] / d parameter``. Repeating this
    recursion for structured transition coordinates gives the PHLASH score
    construction; the released GPU kernel evaluates those coordinates together.
    """
    observations = np.asarray(observations, dtype=int)
    transition = np.asarray(transition, dtype=float)
    emission = np.asarray(emission, dtype=float)
    alpha = np.asarray(initial, dtype=float).copy()
    feature = np.asarray(feature, dtype=float)
    m = len(alpha)
    if transition.shape != (m, m) or feature.shape != (m, m):
        raise ValueError("transition and feature must be M by M")
    additive = np.zeros(m)
    log_likelihood = 0.0
    for symbol in observations:
        weighted = transition * emission[:, symbol][None, :]
        new_alpha = alpha @ weighted
        new_additive = (
            additive @ weighted
            + (alpha[:, None] * weighted * feature).sum(axis=0)
        )
        scale = new_alpha.sum()
        if scale <= 0:
            return -np.inf, np.nan
        alpha = new_alpha / scale
        additive = new_additive / scale
        log_likelihood += np.log(scale)
    return float(log_likelihood), float(additive.sum())


def rbf_kernel(particles, bandwidth=None):
    """Return the RBF Gram matrix and the median-heuristic bandwidth."""
    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 2 or len(particles) < 2:
        raise ValueError("particles must be a J by D array with J >= 2")
    delta = particles[:, None, :] - particles[None, :, :]
    squared_distance = np.sum(delta**2, axis=-1)
    if bandwidth is None:
        positive = squared_distance[np.triu_indices(len(particles), 1)]
        bandwidth = np.sqrt(np.median(positive) / (2 * np.log(len(particles) + 1)))
        bandwidth = max(float(bandwidth), 1e-5)
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-squared_distance / (2 * bandwidth**2)), float(bandwidth)


def svgd_direction(particles, score, bandwidth=None):
    """Compute the standard RBF-SVGD attraction-plus-repulsion direction."""
    particles = np.asarray(particles, dtype=float)
    score = np.asarray(score, dtype=float)
    if particles.shape != score.shape:
        raise ValueError("particles and score must have the same shape")
    kernel, bandwidth = rbf_kernel(particles, bandwidth)
    attraction = kernel.T @ score
    delta = particles[:, None, :] - particles[None, :, :]
    repulsion = -(kernel[:, :, None] * delta).sum(axis=0) / bandwidth**2
    return (attraction + repulsion) / len(particles)


def demo():
    """Run deterministic parity examples used in the PHLASH chapter."""
    times = logarithmic_grid(np.log(1e-4), np.log(15.0), intervals=4)
    rates = np.array([1.0, 0.5, 2.0, 1.0])
    masses = coalescence_probabilities(times, rates)
    afs = afs_log_score([12, 7, 3], [0.5, 0.3, 0.2])
    print("grid:", np.array2string(times, precision=6))
    print("coalescence masses:", np.array2string(masses, precision=6))
    print(f"mass sum: {masses.sum():.12f}")
    print(f"AFS log score: {afs:.6f}")


if __name__ == "__main__":
    demo()
