"""Canonical Population registry.

Birth Control, selection, reproduction, death, controller hooks, and metrics
ALL operate on this single object. There is no parallel agent list.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from driftbound.smartdream.agent import Agent


@dataclass
class Population:
    """Single source of truth for living and historical agents."""

    _agents: dict[str, Agent] = field(default_factory=dict)
    _birth_count: int = 0
    _death_count: int = 0
    max_size: int = 50

    def __len__(self) -> int:
        return len(self.living())

    def __iter__(self) -> Iterator[Agent]:
        return iter(self.living())

    def register(self, agent: Agent) -> Agent:
        if agent.id in self._agents:
            raise ValueError(f"Agent {agent.id} already registered")
        self._agents[agent.id] = agent
        self._birth_count += 1
        return agent

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def all_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def living(self) -> list[Agent]:
        return [a for a in self._agents.values() if a.alive]

    def dead(self) -> list[Agent]:
        return [a for a in self._agents.values() if not a.alive]

    def mark_dead(self, agent_id: str, reason: str = "culled") -> None:
        agent = self._agents[agent_id]
        if agent.alive:
            agent.die(reason)
            self._death_count += 1

    def size(self) -> int:
        return len(self.living())

    def summary(self) -> dict[str, int | float]:
        living = self.living()
        avg_fit = (
            sum(a.fitness() for a in living) / len(living) if living else 0.0
        )
        return {
            "living": len(living),
            "dead": len(self.dead()),
            "total_registered": len(self._agents),
            "births": self._birth_count,
            "deaths": self._death_count,
            "max_size": self.max_size,
            "avg_fitness": avg_fit,
        }
