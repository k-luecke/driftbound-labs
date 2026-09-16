"""Parent selection strategies over a Population."""

from __future__ import annotations

from numpy.random import Generator

from driftbound.smartdream.agent import Agent
from driftbound.smartdream.population import Population


def select_parents(
    population: Population,
    rng: Generator,
    *,
    n: int = 2,
    tournament_k: int = 3,
) -> list[Agent]:
    """Tournament selection by recent correctness among living agents."""
    living = population.living()
    if len(living) < n:
        raise ValueError(f"Need at least {n} living agents, found {len(living)}")
    parents: list[Agent] = []
    for _ in range(n):
        k = min(tournament_k, len(living))
        idxs = rng.choice(len(living), size=k, replace=False)
        contestants = [living[int(i)] for i in idxs]
        winner = max(contestants, key=lambda a: a.recent_correctness)
        parents.append(winner)
    return parents
