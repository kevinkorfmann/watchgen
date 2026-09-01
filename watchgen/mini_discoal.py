"""Source-guided pedagogical kernels for :program:`discoal`.

This module deliberately does *not* call itself a Python translation of the full
program. Production discoal simulates a discrete ancestral recombination graph,
including crossover, gene conversion, demography, and several sweep models. The
code below isolates two pieces that can be checked directly against the published
algorithm and the C implementation:

* the backward conditional jump process for the selected-allele trajectory; and
* the two-background structured coalescent for one neutral locus at a fixed
  recombination distance from the selected site.

Times in :class:`SweepTrajectory` are stored in units of ``2 * sweep_N``
generations, matching discoal's internal coalescent clock. Command-line sweep
times are in ``4 * N`` generations and are converted internally by discoal.

Ground truth: Kern and Schrider (2016), Coop and Griffiths (2004), the msprime
1.0 sweep description, and discoal source commit
``7d0955f4107053c135d2086790b0426457147a8e``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DISCOAL_DEFAULT_SWEEP_N = 1_000_000
DISCOAL_DEFAULT_DT_SCALAR = 40.0


@dataclass(frozen=True)
class SweepTrajectory:
    """Allele frequencies followed backward from the sweep endpoint."""

    frequencies: np.ndarray
    dt_2N: float
    sweep_N: int
    selected_steps: int

    def __post_init__(self) -> None:
        f = np.asarray(self.frequencies, dtype=float)
        if f.ndim != 1 or len(f) < 2:
            raise ValueError("frequencies must be a one-dimensional trajectory")
        if np.any(~np.isfinite(f)) or np.any((f <= 0) | (f >= 1)):
            raise ValueError("trajectory frequencies must lie strictly in (0, 1)")
        if self.dt_2N <= 0 or self.sweep_N <= 0:
            raise ValueError("dt_2N and sweep_N must be positive")
        object.__setattr__(self, "frequencies", f)

    @property
    def dt_generations(self) -> float:
        """Length of one trajectory step in generations."""

        return self.dt_2N * 2.0 * self.sweep_N

    @property
    def duration_2N(self) -> float:
        """Trajectory duration in ``2 * sweep_N`` units."""

        return (len(self.frequencies) - 1) * self.dt_2N

    @property
    def duration_generations(self) -> float:
        """Trajectory duration in generations."""

        return (len(self.frequencies) - 1) * self.dt_generations


def _validate_trajectory_parameters(
    alpha: float,
    start_frequency: float,
    end_frequency: float,
    sweep_N: int,
    dt_scalar: float,
) -> None:
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive")
    if not 0 < start_frequency < end_frequency < 1:
        raise ValueError("require 0 < start_frequency < end_frequency < 1")
    if int(sweep_N) != sweep_N or sweep_N <= 0:
        raise ValueError("sweep_N must be a positive integer")
    if not np.isfinite(dt_scalar) or dt_scalar <= 0:
        raise ValueError("dt_scalar must be positive")


def fixation_probability(x: float | np.ndarray, alpha: float):
    """Diffusion fixation probability under genic selection.

    ``alpha`` is ``2 * N * s``. The implementation uses ``expm1`` so that the
    neutral limit is numerically stable.
    """

    x_arr = np.asarray(x, dtype=float)
    if np.any((x_arr < 0) | (x_arr > 1)):
        raise ValueError("x must lie in [0, 1]")
    if abs(alpha) < 1e-8:
        answer = x_arr
    else:
        answer = np.expm1(-alpha * x_arr) / np.expm1(-alpha)
    return float(answer) if answer.ndim == 0 else answer


def discoal_deterministic_frequency(time_2N: float | np.ndarray, alpha: float):
    """Evaluate discoal's production ``detSweepFreq`` curve exactly.

    Time increases backward from the high-frequency endpoint. The C code's
    cutoff parameter is ``epsilon = 0.05 / alpha``; it is not the single-copy
    frequency.
    """

    if alpha <= 0.05:
        raise ValueError("discoal's deterministic curve requires alpha > 0.05")
    t = np.asarray(time_2N, dtype=float)
    if np.any(t < 0):
        raise ValueError("time_2N must be non-negative")
    epsilon = 0.05 / alpha
    t_s = -2.0 * math.log(epsilon) / alpha
    denominator = epsilon + (1.0 - epsilon) * np.exp(alpha * (t - t_s))
    answer = epsilon / denominator
    return float(answer) if answer.ndim == 0 else answer


def deterministic_trajectory(
    alpha: float,
    *,
    start_frequency: float | None = None,
    end_frequency: float | None = None,
    sweep_N: int = DISCOAL_DEFAULT_SWEEP_N,
    dt_scalar: float = DISCOAL_DEFAULT_DT_SCALAR,
) -> SweepTrajectory:
    """Generate discoal's deterministic trajectory backward in time.

    Examples should use a modest ``sweep_N``; the production default of one
    million deliberately creates a very fine grid.
    """

    if int(sweep_N) != sweep_N or sweep_N <= 0:
        raise ValueError("sweep_N must be a positive integer")
    if start_frequency is None:
        start_frequency = 1.0 / (2.0 * sweep_N)
    if end_frequency is None:
        end_frequency = 1.0 - 1.0 / (2.0 * sweep_N)
    _validate_trajectory_parameters(
        alpha, start_frequency, end_frequency, sweep_N, dt_scalar
    )
    dt = 1.0 / (dt_scalar * sweep_N)
    values = [float(end_frequency)]
    step = 1
    while values[-1] > start_frequency:
        x = discoal_deterministic_frequency(step * dt, alpha)
        values.append(
            min(values[-1], max(float(start_frequency), float(x)))
        )
        step += 1
        if step > 500_000_000:
            raise RuntimeError("trajectory exceeded discoal's production step limit")
    return SweepTrajectory(np.asarray(values), dt, sweep_N, len(values) - 1)


def _selected_backward_step(x: float, alpha: float, dt_2N: float, rng) -> float:
    """One Coop--Griffiths two-point jump, followed backward in time."""

    q = 1.0 - x
    z = alpha * q
    ratio = 1.0 if abs(z) < 1e-8 else z / math.tanh(z)
    drift = x * ratio
    noise = math.sqrt(max(x * (1.0 - x) * dt_2N, 0.0))
    return x - drift * dt_2N + (-noise if rng.random() < 0.5 else noise)


def _neutral_backward_step(x: float, dt_2N: float, rng) -> float:
    """One backward jump for a neutral allele conditioned on its origin."""

    drift = -x * dt_2N
    noise = math.sqrt(max(x * (1.0 - x) * dt_2N, 0.0))
    return x + drift + (-noise if rng.random() < 0.5 else noise)


def stochastic_trajectory(
    alpha: float,
    *,
    start_frequency: float | None = None,
    end_frequency: float | None = None,
    selection_start_frequency: float | None = None,
    sweep_N: int = DISCOAL_DEFAULT_SWEEP_N,
    dt_scalar: float = DISCOAL_DEFAULT_DT_SCALAR,
    rng=None,
) -> SweepTrajectory:
    """Generate the conditional jump trajectory used by discoal.

    The process starts at the high-frequency endpoint and runs backward. Above
    ``selection_start_frequency`` it uses the selected conditional diffusion.
    Below that frequency it uses discoal's neutral conditional process, which is
    how the program represents a sweep from standing variation.
    """

    if int(sweep_N) != sweep_N or sweep_N <= 0:
        raise ValueError("sweep_N must be a positive integer")
    if start_frequency is None:
        start_frequency = 1.0 / (2.0 * sweep_N)
    if end_frequency is None:
        end_frequency = 1.0 - 1.0 / (2.0 * sweep_N)
    if selection_start_frequency is None:
        selection_start_frequency = start_frequency
    _validate_trajectory_parameters(
        alpha, start_frequency, end_frequency, sweep_N, dt_scalar
    )
    if not start_frequency <= selection_start_frequency < end_frequency:
        raise ValueError(
            "selection_start_frequency must be between start and end frequencies"
        )
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / (dt_scalar * sweep_N)
    x = float(end_frequency)
    values = [x]
    selected_steps = 0
    for _ in range(1, 500_000_001):
        if x <= start_frequency:
            break
        if x > selection_start_frequency:
            candidate = _selected_backward_step(x, alpha, dt, rng)
            selected_steps += 1
        else:
            candidate = _neutral_backward_step(x, dt, rng)
        # Production discoal rejects invalid whole trajectories. Reflecting a
        # local overshoot avoids introducing point mass by clipping in this mini.
        if candidate >= 1.0:
            candidate = 2.0 - candidate
        if candidate <= 0.0:
            candidate = -candidate
        x = max(float(start_frequency), candidate)
        values.append(x)
    else:
        raise RuntimeError("trajectory exceeded discoal's production step limit")
    return SweepTrajectory(np.asarray(values), dt, sweep_N, selected_steps)


def coalescence_rate(n_lineages: int, background_frequency: float, N: int) -> float:
    """Total coalescence rate per generation within one background."""

    if n_lineages < 0 or N <= 0:
        raise ValueError("n_lineages must be non-negative and N positive")
    if not 0 < background_frequency <= 1:
        return 0.0
    return math.comb(n_lineages, 2) / (2.0 * N * background_frequency)


def migration_rates(n_B: int, n_b: int, r: float, x: float) -> tuple[float, float]:
    """Single-locus background-switch rates per generation.

    Within-locus discoal recombination is richer: it splits ancestral material
    and cannot be represented by these two lineage counts.
    """

    if n_B < 0 or n_b < 0 or r < 0 or not 0 <= x <= 1:
        raise ValueError("invalid lineage count, recombination rate, or frequency")
    return n_B * r * (1.0 - x), n_b * r * x


def structured_event_probabilities(
    n_B: int,
    n_b: int,
    x: float,
    r: float,
    N: int,
    dt_generations: float,
) -> np.ndarray:
    """Per-step probabilities for the four single-locus sweep events."""

    rates = np.asarray(
        [
            coalescence_rate(n_B, x, N),
            coalescence_rate(n_b, 1.0 - x, N),
            *migration_rates(n_B, n_b, r, x),
        ]
    )
    probabilities = rates * dt_generations
    if probabilities.sum() >= 1.0:
        raise ValueError(
            "time step is too coarse for the discrete sweep event approximation"
        )
    return probabilities


def structured_coalescent_sweep(
    trajectory: SweepTrajectory,
    n_sample: int,
    r_site: float,
    N: int,
    rng=None,
    *,
    n_B_init: int | None = None,
    n_b_init: int | None = None,
    single_origin: bool = True,
) -> tuple[list[float], int, int]:
    """Run discoal's two-background kernel for one linked neutral locus.

    This follows the production rejection-loop approximation: each small time
    interval has either no event or one event with probability proportional to
    its current rate. Returned times are generations backward from the sweep
    endpoint.
    """

    if n_sample < 1 or N <= 0 or r_site < 0:
        raise ValueError("invalid sample size, population size, or recombination")
    if rng is None:
        rng = np.random.default_rng()
    if (n_B_init is None) != (n_b_init is None):
        raise ValueError("provide both initial background counts or neither")
    if n_B_init is None:
        n_B, n_b = n_sample, 0
    else:
        n_B, n_b = int(n_B_init), int(n_b_init)
        if n_B < 0 or n_b < 0 or n_B + n_b != n_sample:
            raise ValueError("initial background counts must sum to n_sample")

    coal_times: list[float] = []
    dt = trajectory.dt_generations
    for step, x in enumerate(trajectory.frequencies[1:], start=1):
        if n_B + n_b <= 1:
            break
        probabilities = structured_event_probabilities(n_B, n_b, x, r_site, N, dt)
        u = rng.random()
        cumulative = np.cumsum(probabilities)
        if u < cumulative[0]:
            n_B -= 1
            coal_times.append(step * dt)
        elif u < cumulative[1]:
            n_b -= 1
            coal_times.append(step * dt)
        elif u < cumulative[2]:
            n_B -= 1
            n_b += 1
        elif u < cumulative[3]:
            n_b -= 1
            n_B += 1

    if single_origin and n_B > 1:
        coal_times.extend([trajectory.duration_generations] * (n_B - 1))
        n_B = 1
    return coal_times, n_B, n_b


def escape_probability(r_site: float, trajectory: SweepTrajectory) -> float:
    """Probability that one B lineage switches background along a trajectory."""

    if r_site < 0:
        raise ValueError("r_site must be non-negative")
    hazards = r_site * (1.0 - trajectory.frequencies[1:]) * trajectory.dt_generations
    if np.any(hazards >= 1):
        raise ValueError("time step is too coarse for escape calculation")
    return float(-np.expm1(np.log1p(-hazards).sum()))


def neutral_coalescent(
    n_lineages: int,
    N: int,
    rng,
    *,
    start_time: float = 0.0,
    end_time: float = math.inf,
) -> tuple[list[float], int]:
    """Simulate a standard diploid neutral coalescent over a time interval."""

    times: list[float] = []
    t = float(start_time)
    k = int(n_lineages)
    while k > 1:
        rate = math.comb(k, 2) / (2.0 * N)
        proposed = t + rng.exponential(1.0 / rate)
        if proposed > end_time:
            break
        t = proposed
        times.append(t)
        k -= 1
    return times, k


def simulate_linked_locus_genealogy(
    n_sample: int,
    N: int,
    trajectory: SweepTrajectory,
    r_site: float,
    *,
    tau_generations: float = 0.0,
    endpoint_frequency: float = 1.0,
    rng=None,
) -> list[float]:
    """Simulate one marginal genealogy linked to a completed or partial sweep."""

    if rng is None:
        rng = np.random.default_rng()
    recent, n_at_sweep = neutral_coalescent(
        n_sample, N, rng, end_time=tau_generations
    )
    if n_at_sweep <= 1:
        return recent
    if endpoint_frequency >= 1.0:
        n_B, n_b = n_at_sweep, 0
    elif 0 < endpoint_frequency < 1:
        n_B = int(rng.binomial(n_at_sweep, endpoint_frequency))
        n_b = n_at_sweep - n_B
    else:
        raise ValueError("endpoint_frequency must lie in (0, 1]")

    sweep_times, n_B, n_b = structured_coalescent_sweep(
        trajectory,
        n_at_sweep,
        r_site,
        N,
        rng,
        n_B_init=n_B,
        n_b_init=n_b,
        single_origin=True,
    )
    all_times = recent + [tau_generations + t for t in sweep_times]
    ancient, _ = neutral_coalescent(
        n_B + n_b,
        N,
        rng,
        start_time=tau_generations + trajectory.duration_generations,
    )
    return all_times + ancient


def pairwise_diversity_profile(
    N: int,
    trajectory: SweepTrajectory,
    recombination_rate: float,
    positions,
    selected_position: float,
    *,
    replicates: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Estimate relative pairwise diversity around a deterministic sweep.

    Sites are simulated independently; this is not a chromosome-scale ARG
    simulator.
    """

    if recombination_rate < 0 or replicates < 1:
        raise ValueError("invalid recombination rate or replicate count")
    positions = np.asarray(positions, dtype=float)
    rng = np.random.default_rng(seed)
    answer = np.empty_like(positions)
    for j, position in enumerate(positions):
        distance = abs(position - selected_position)
        r_site = min(0.5, recombination_rate * distance)
        tmrcas = []
        for _ in range(replicates):
            times = simulate_linked_locus_genealogy(
                2, N, trajectory, r_site, rng=rng
            )
            tmrcas.append(max(times) if times else 0.0)
        answer[j] = np.mean(tmrcas) / (2.0 * N)
    return answer


def discoal_to_msprime(theta, rho, alpha, n, L, N) -> dict:
    """Translate scaling conventions while preserving haploid sample count."""

    if min(theta, rho, alpha, n, L, N) < 0 or min(n, L, N) == 0:
        raise ValueError("rates must be non-negative and sizes positive")
    return {
        "samples": int(n),
        "ploidy": 1,
        "sequence_length": float(L),
        "mutation_rate": theta / (4.0 * N * L),
        "recombination_rate": rho / (4.0 * N * L),
        "population_size": float(N),
        "selection_coefficient": alpha / (2.0 * N),
        "start_frequency": 1.0 / (2.0 * N),
        "end_frequency": 1.0 - 1.0 / (2.0 * N),
    }


def msprime_to_discoal(n, L, mu, r, s, N) -> dict:
    """Translate raw per-generation parameters to discoal's CLI scaling."""

    if min(mu, r, s, n, L, N) < 0 or min(n, L, N) == 0:
        raise ValueError("rates must be non-negative and sizes positive")
    return {
        "n": int(n),
        "L": int(L),
        "theta": 4.0 * N * mu * L,
        "rho": 4.0 * N * r * L,
        "alpha": 2.0 * N * s,
    }


def demo() -> None:
    """Run a small, reproducible source-guided example."""

    N = 500
    trajectory = deterministic_trajectory(50.0, sweep_N=N)
    print(f"trajectory steps: {len(trajectory.frequencies):,}")
    print(f"duration: {trajectory.duration_generations:.1f} generations")
    print(f"escape probability at r=1e-3: {escape_probability(1e-3, trajectory):.3f}")
    positions = np.linspace(0, 100_000, 11)
    profile = pairwise_diversity_profile(
        N, trajectory, 1e-8, positions, 50_000, replicates=20
    )
    for position, diversity in zip(positions, profile):
        print(f"{position:9.0f}  {diversity:7.3f}")


if __name__ == "__main__":
    demo()
