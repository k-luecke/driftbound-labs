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
    success: bool | None = None,
) -> RewardComponents:
    """Build the canonical reward decomposition for a step.

    ``correct`` drives *decision* correctness reward (e.g. action == best_action).
    Physical survival/consequence uses ``success`` when provided; otherwise it
    falls back to ``correct`` for backward compatibility. These stay distinct.
    """
    physical_ok = correct if success is None else success
    return RewardComponents(
        correctness_reward=correctness_value if correct else 0.0,
        consequence_utility=consequence if physical_ok else -abs(consequence),
        observation_cost=observation_cost,
        intervention_cost=intervention_cost,
        fallback_cost=fallback_cost,
    )
