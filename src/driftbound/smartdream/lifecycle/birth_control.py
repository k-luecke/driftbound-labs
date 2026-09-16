"""Birth Control: atomic birth/death on the canonical Population.

Eligibility, parent selection, max population, cull, replacement,
offspring registration, lineage tracking, and extinction protection.
"""

from __future__ import annotations

from dataclasses import dataclass

from numpy.random import Generator

from driftbound.smartdream.agent import Agent
from driftbound.smartdream.lifecycle.genome import Genome
from driftbound.smartdream.lifecycle.selection import select_parents
from driftbound.smartdream.population import Population


@dataclass
class BirthControl:
    """Lifecycle controller bound to one canonical Population instance."""

    population: Population
    fitness_threshold: float = 0.15
    mutation_rate: float = 0.2
    min_population: int = 2  # extinction protection floor
    tournament_k: int = 3

    def eligible(self, agent: Agent) -> bool:
        return (
            agent.alive
            and agent.fitness() >= self.fitness_threshold
            and self.population.size() < self.population.max_size
        )

    def can_birth(self) -> bool:
        return self.population.size() < self.population.max_size

    def select_parents(self, rng: Generator, n: int = 2) -> list[Agent]:
        return select_parents(
            self.population, rng, n=n, tournament_k=self.tournament_k
        )

    def reproduce(self, rng: Generator) -> Agent | None:
        """Atomic birth: maybe cull if at capacity, then register offspring."""
        if self.population.size() < self.min_population:
            # Prefer cloning a survivor over failing when near extinction
            living = self.population.living()
            if not living:
                return None
            parent = living[0]
            child_genome = parent.genome.mutate(rng, rate=self.mutation_rate)
            return self._register_offspring(
                child_genome, parents=(parent,), generation=parent.generation + 1
            )

        if not self.can_birth():
            self.cull_least_fit(rng)
            if not self.can_birth():
                return None

        eligible = [a for a in self.population.living() if self.eligible(a)]
        if len(eligible) < 2:
            # Fall back to any living pair if eligibility is sparse
            if self.population.size() < 2:
                return None
            parents = self.select_parents(rng, n=2)
        else:
            # Tournament among eligible only via temporary view: use select on full pop
            parents = self.select_parents(rng, n=2)

        child_genome = parents[0].genome.crossover(parents[1].genome, rng)
        child_genome = child_genome.mutate(rng, rate=self.mutation_rate)
        gen = max(p.generation for p in parents) + 1
        return self._register_offspring(child_genome, parents=tuple(parents), generation=gen)

    def _register_offspring(
        self,
        genome: Genome,
        *,
        parents: tuple[Agent, ...],
        generation: int,
    ) -> Agent:
        child = Agent(
            genome=genome,
            generation=generation,
            parent_ids=tuple(p.id for p in parents),
        )
        self.population.register(child)
        return child

    def cull_least_fit(self, rng: Generator | None = None) -> Agent | None:
        """Remove lowest-fitness living agent, respecting extinction floor."""
        living = self.population.living()
        if len(living) <= self.min_population:
            return None
        victim = min(living, key=lambda a: a.fitness())
        self.population.mark_dead(victim.id, reason="culled_low_fitness")
        return victim

    def replace(
        self,
        rng: Generator,
        *,
        deaths: int = 1,
        births: int = 1,
    ) -> list[Agent]:
        """Atomic replacement cycle: cull then birth on the same Population."""
        for _ in range(deaths):
            self.cull_least_fit(rng)
        born: list[Agent] = []
        for _ in range(births):
            child = self.reproduce(rng)
            if child is not None:
                born.append(child)
        return born

    def seed_population(self, n: int, rng: Generator) -> list[Agent]:
        agents: list[Agent] = []
        for _ in range(n):
            if self.population.size() >= self.population.max_size:
                break
            agent = Agent(genome=Genome.random(rng), generation=0)
            self.population.register(agent)
            agents.append(agent)
        return agents
