"""Small, explicit mechanisms from SLiM's diploid Wright--Fisher model.

This is a teaching model, not a reimplementation of SLiM. It follows the
default SLiM 5.2 semantics for mutation identity, multiplicative mutation
fitness, relative-fitness parent sampling, and a uniform recombination map.
Callbacks, genomic elements, stacking policies, migration, sex, pedigrees,
nonWF survival, and tree-sequence recording remain SLiM's job.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import count
from math import log1p

import numpy as np

_MUTATION_IDS = count()


@dataclass(frozen=True)
class Mutation:
    """One mutational lineage; identity is distinct from genomic position."""

    position: int
    s: float = 0.0
    h: float = 0.5
    origin_tick: int = 0
    mutation_id: int = field(default_factory=lambda: next(_MUTATION_IDS))

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("position must be non-negative")


@dataclass
class Individual:
    """A diploid individual represented by two mutation-lineage lists."""

    haplosome_1: list[Mutation] = field(default_factory=list)
    haplosome_2: list[Mutation] = field(default_factory=list)
    fitness: float = 1.0


WF_TICK_STAGES = (
    "first events",
    "early events",
    "offspring generation",
    "offspring become parents",
    "fixed-mutation processing",
    "late events",
    "fitness recalculation",
    "tick increment",
)


def wf_tick_stages() -> tuple[str, ...]:
    """Return the default WF tick stages in SLiM 5.2 execution order."""

    return WF_TICK_STAGES


def _by_id(haplosome: Iterable[Mutation]) -> dict[int, Mutation]:
    mutations: dict[int, Mutation] = {}
    for mutation in haplosome:
        previous = mutations.setdefault(mutation.mutation_id, mutation)
        if previous != mutation:
            raise ValueError("one mutation_id cannot describe two mutations")
    return mutations


def calculate_fitness(individual: Individual) -> float:
    """Calculate default multiplicative mutation fitness.

    A lineage present once contributes ``1 + h*s``; the same lineage on both
    haplosomes contributes ``1 + s``. Distinct recurrent mutations at one
    position each contribute as heterozygotes. SLiM callbacks may replace
    these defaults; this bounded function does not implement callbacks.
    """

    first = _by_id(individual.haplosome_1)
    second = _by_id(individual.haplosome_2)
    fitness = 1.0
    for mutation_id in first.keys() | second.keys():
        mutation = first.get(mutation_id, second.get(mutation_id))
        assert mutation is not None
        fitness *= (
            1.0 + mutation.s
            if mutation_id in first and mutation_id in second
            else 1.0 + mutation.h * mutation.s
        )
    individual.fitness = max(0.0, float(fitness))
    return individual.fitness


def parent_probabilities(population: Sequence[Individual]) -> np.ndarray:
    """Normalize cached non-negative WF fitnesses into parent probabilities."""

    if not population:
        raise ValueError("population must not be empty")
    fitness = np.asarray([individual.fitness for individual in population], dtype=float)
    if np.any(~np.isfinite(fitness)) or np.any(fitness < 0):
        raise ValueError("fitness values must be finite and non-negative")
    total = float(fitness.sum())
    if total <= 0:
        raise RuntimeError("population extinction: total fitness is zero")
    return fitness / total


def select_parents(
    population: Sequence[Individual], rng: np.random.Generator | None = None
) -> tuple[int, int]:
    """Choose two parents independently using cached relative fitness."""

    rng = np.random.default_rng() if rng is None else rng
    parents = rng.choice(
        len(population), size=2, replace=True, p=parent_probabilities(population)
    )
    return int(parents[0]), int(parents[1])


def breakpoint_intensity(probability: float) -> float:
    """Convert adjacent-base breakpoint probability to Poisson intensity."""

    if not 0.0 <= probability <= 0.5:
        raise ValueError("breakpoint probability must be between 0 and 0.5")
    return -log1p(-probability)


def draw_breakpoints(
    probability: float,
    length: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Draw breakpoints for a uniform SLiM-style recombination map.

    Positions 0 through ``length - 1`` have ``length - 1`` possible breaks.
    Coordinate ``j`` means immediately left of base ``j``. Duplicate raw
    events are collapsed, matching SLiM's crossover path.
    """

    if length < 1:
        raise ValueError("length must be at least one")
    rng = np.random.default_rng() if rng is None else rng
    intervals = length - 1
    if intervals == 0 or probability == 0:
        return []
    count_ = int(rng.poisson(breakpoint_intensity(probability) * intervals))
    if count_ == 0:
        return []
    return sorted({int(x) for x in rng.integers(1, length, size=count_)})


def gamete_from_breakpoints(
    parent: Individual,
    breakpoints: Iterable[int],
    *,
    start_haplosome: int,
    length: int,
) -> list[Mutation]:
    """Assemble one gamete from explicit breakpoint coordinates."""

    if length < 1:
        raise ValueError("length must be at least one")
    if start_haplosome not in (0, 1):
        raise ValueError("start_haplosome must be 0 or 1")
    points = sorted({int(point) for point in breakpoints})
    if any(point <= 0 or point >= length for point in points):
        raise ValueError("breakpoints must be in [1, length - 1]")
    child: list[Mutation] = []
    for source, haplosome in enumerate((parent.haplosome_1, parent.haplosome_2)):
        for mutation in haplosome:
            if mutation.position >= length:
                raise ValueError("mutation position lies outside the chromosome")
            switches = sum(point <= mutation.position for point in points)
            active = start_haplosome ^ (switches % 2)
            if source == active:
                child.append(mutation)
    return sorted(child, key=lambda mutation: (mutation.position, mutation.mutation_id))


def recombine(
    parent: Individual,
    r: float,
    L: int,
    rng: np.random.Generator | None = None,
) -> list[Mutation]:
    """Generate a gamete under a uniform SLiM-style recombination map."""

    rng = np.random.default_rng() if rng is None else rng
    return gamete_from_breakpoints(
        parent,
        draw_breakpoints(r, L, rng),
        start_haplosome=int(rng.integers(0, 2)),
        length=L,
    )


recombine_v2 = recombine


def add_mutations(
    haplosome: Iterable[Mutation],
    mu: float,
    L: int,
    tick: int,
    dfe: str = "neutral",
    dfe_params: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Mutation]:
    """Add independent lineages under a simple uniform mutation map."""

    if mu < 0 or L < 1:
        raise ValueError("mu must be non-negative and L must be positive")
    rng = np.random.default_rng() if rng is None else rng
    params = {} if dfe_params is None else dfe_params
    child = list(haplosome)
    for _ in range(int(rng.poisson(mu * L))):
        if dfe == "neutral":
            selection = 0.0
        elif dfe == "fixed":
            selection = float(params.get("s", 0.0))
        elif dfe == "exponential_beneficial":
            selection = float(rng.exponential(params.get("mean", 0.01)))
        elif dfe == "gamma_deleterious":
            selection = -float(
                rng.gamma(params.get("shape", 0.3), params.get("scale", 0.05))
            )
        else:
            raise ValueError(f"unknown DFE: {dfe}")
        child.append(
            Mutation(
                position=int(rng.integers(0, L)),
                s=selection,
                h=float(params.get("h", 0.5)),
                origin_tick=tick,
            )
        )
    return sorted(child, key=lambda mutation: (mutation.position, mutation.mutation_id))


def wright_fisher_generation(
    population: Sequence[Individual],
    N: int,
    L: int,
    mu: float,
    r: float,
    tick: int,
    dfe: str = "neutral",
    dfe_params: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[Individual]:
    """Produce one fixed-size WF offspring generation.

    This first establishes the cached fitness SLiM would calculate at the end
    of the preceding tick, then generates offspring. Callbacks and fixed-
    mutation/substitution processing are intentionally omitted.
    """

    if N < 1:
        raise ValueError("N must be positive")
    rng = np.random.default_rng() if rng is None else rng
    for individual in population:
        calculate_fitness(individual)
    offspring: list[Individual] = []
    for _ in range(N):
        first, second = select_parents(population, rng)
        haplosome_1 = add_mutations(
            recombine(population[first], r, L, rng),
            mu, L, tick, dfe, dfe_params, rng,
        )
        haplosome_2 = add_mutations(
            recombine(population[second], r, L, rng),
            mu, L, tick, dfe, dfe_params, rng,
        )
        offspring.append(Individual(haplosome_1, haplosome_2))
    return offspring


def simulate(
    N: int,
    L: int,
    mu: float,
    r: float,
    T: int,
    dfe: str = "neutral",
    dfe_params: dict[str, float] | None = None,
    seed: int | None = None,
) -> tuple[list[Individual], dict[str, list[float]]]:
    """Run the bounded teaching model and return population and summaries."""

    if T < 0:
        raise ValueError("T must be non-negative")
    rng = np.random.default_rng(seed)
    population = [Individual() for _ in range(N)]
    statistics: dict[str, list[float]] = {
        "tick": [], "mean_fitness": [], "segregating_mutations": []
    }
    for tick in range(1, T + 1):
        population = wright_fisher_generation(
            population, N, L, mu, r, tick, dfe, dfe_params, rng
        )
        fitness = [calculate_fitness(individual) for individual in population]
        counts: dict[int, int] = {}
        for individual in population:
            for mutation in individual.haplosome_1 + individual.haplosome_2:
                counts[mutation.mutation_id] = counts.get(mutation.mutation_id, 0) + 1
        statistics["tick"].append(float(tick))
        statistics["mean_fitness"].append(float(np.mean(fitness)))
        statistics["segregating_mutations"].append(
            float(sum(copies < 2 * N for copies in counts.values()))
        )
    return population, statistics


def mutation_frequency(population: Sequence[Individual], mutation: Mutation) -> float:
    """Return one mutation lineage's frequency among diploid haplosomes."""

    if not population:
        raise ValueError("population must not be empty")
    copies = sum(
        candidate.mutation_id == mutation.mutation_id
        for individual in population
        for candidate in individual.haplosome_1 + individual.haplosome_2
    )
    return copies / (2 * len(population))


def demo() -> None:
    """Print a deterministic smoke-test run."""

    population, statistics = simulate(
        N=20, L=1_000, mu=1e-5, r=1e-4, T=10, seed=23
    )
    print(f"individuals: {len(population)}")
    print(f"final mean fitness: {statistics['mean_fitness'][-1]:.3f}")


if __name__ == "__main__":
    demo()
