"""Agent: genome-bearing individual registered in a canonical Population."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from driftbound.smartdream.lifecycle.genome import Genome, Phenotype
from driftbound.smartdream.memory import Memory


@dataclass
class Agent:
    """Canonical agent fields used by Birth Control, selection, and metrics."""

    genome: Genome
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    vitality: float = 1.0
    health: float = 1.0
    cumulative_correctness: float = 0.0
    recent_correctness: float = 0.0
    cumulative_consequence: float = 0.0
    cumulative_utility: float = 0.0
    generation: int = 0
    parent_ids: tuple[str, ...] = ()
    alive: bool = True
    controller_params: dict[str, Any] = field(default_factory=dict)
    expert_params: dict[str, Any] = field(default_factory=dict)
    memory: Memory | None = None
    _correct_window: list[bool] = field(default_factory=list, repr=False)
    _window_size: int = 20

    def __post_init__(self) -> None:
        if self.memory is None:
            self.memory = Memory(agent_id=self.id)

    @property
    def phenotype(self) -> Phenotype:
        return self.genome.phenotype()

    @property
    def dead(self) -> bool:
        return not self.alive

    def record_outcome(
        self,
        *,
        correct: bool,
        consequence: float,
        utility: float,
        step: int | None = None,
    ) -> None:
        self.cumulative_correctness += 1.0 if correct else 0.0
        self.cumulative_consequence += consequence
        self.cumulative_utility += utility
        self._correct_window.append(correct)
        if len(self._correct_window) > self._window_size:
            self._correct_window.pop(0)
        if self._correct_window:
            self.recent_correctness = sum(self._correct_window) / len(self._correct_window)
        if step is not None and self.memory is not None:
            self.memory.record(
                step,
                "outcome",
                {"correct": correct, "consequence": consequence, "utility": utility},
            )

    def fitness(self) -> float:
        """Scalar fitness for cull/selection: recent correctness × vitality + utility."""
        return self.recent_correctness * max(self.vitality, 0.0) + 0.1 * self.cumulative_utility

    def die(self, reason: str = "unspecified") -> None:
        self.alive = False
        if self.memory is not None:
            self.memory.record(0, "death", {"reason": reason})

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alive": self.alive,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "vitality": self.vitality,
            "health": self.health,
            "recent_correctness": self.recent_correctness,
            "cumulative_correctness": self.cumulative_correctness,
            "cumulative_consequence": self.cumulative_consequence,
            "cumulative_utility": self.cumulative_utility,
            "genome": self.genome.as_dict(),
            "phenotype": self.phenotype.as_dict(),
        }
