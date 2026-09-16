"""Event and reward decomposition types.

net = correctness_reward + consequence_utility
      - observation_cost - intervention_cost - fallback_cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RewardComponents:
    """Canonical reward decomposition kept on every scored step."""

    correctness_reward: float = 0.0
    consequence_utility: float = 0.0
    observation_cost: float = 0.0
    intervention_cost: float = 0.0
    fallback_cost: float = 0.0

    @property
    def net(self) -> float:
        return (
            self.correctness_reward
            + self.consequence_utility
            - self.observation_cost
            - self.intervention_cost
            - self.fallback_cost
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "correctness_reward": self.correctness_reward,
            "consequence_utility": self.consequence_utility,
            "observation_cost": self.observation_cost,
            "intervention_cost": self.intervention_cost,
            "fallback_cost": self.fallback_cost,
            "net": self.net,
        }


@dataclass
class StepEvent:
    """One environment/controller step with reward decomposition and metadata."""

    step: int
    agent_id: str
    action: str
    predicted: Any
    outcome: Any
    correct: bool
    reward: RewardComponents
    regime: str | None = None
    observed: bool = False
    intervened: bool = False
    fallback: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
