"""Reward helpers mapping outcomes to RewardComponents."""

from __future__ import annotations

from driftbound.core.events import RewardComponents


def decompose_reward(
    *,
    correct: bool,
    consequence: float,
    observation_cost: float = 0.0,
    intervention_cost: float = 0.0,
    fallback_cost: float = 0.0,
    correctness_value: float = 1.0,
) -> RewardComponents:
    """Build the canonical reward decomposition for a step."""
    return RewardComponents(
        correctness_reward=correctness_value if correct else 0.0,
        consequence_utility=consequence if correct else -abs(consequence),
        observation_cost=observation_cost,
        intervention_cost=intervention_cost,
        fallback_cost=fallback_cost,
    )
