"""Aggregate metrics from step events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from driftbound.core.events import StepEvent


@dataclass(frozen=True)
class MetricSummary:
    accuracy: float
    net_reward: float
    consequence_weighted_loss: float
    observation_cost: float
    intervention_count: int
    fallback_count: int
    false_promotion_rate: float
    time_to_detect_oom: float | None
    recovery_time: float | None
    calibration_error: float | None
    n_steps: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "accuracy": self.accuracy,
            "net_reward": self.net_reward,
            "consequence_weighted_loss": self.consequence_weighted_loss,
            "observation_cost": self.observation_cost,
            "intervention_count": self.intervention_count,
            "fallback_count": self.fallback_count,
            "false_promotion_rate": self.false_promotion_rate,
            "time_to_detect_oom": self.time_to_detect_oom,
            "recovery_time": self.recovery_time,
            "calibration_error": self.calibration_error,
            "n_steps": self.n_steps,
        }


def summarize_metrics(
    events: Sequence[StepEvent],
    *,
    false_promotions: int = 0,
    promotions: int = 0,
    time_to_detect_oom: float | None = None,
    recovery_time: float | None = None,
    calibration_error: float | None = None,
) -> MetricSummary:
    if not events:
        return MetricSummary(
            accuracy=0.0,
            net_reward=0.0,
            consequence_weighted_loss=0.0,
            observation_cost=0.0,
            intervention_count=0,
            fallback_count=0,
            false_promotion_rate=0.0,
            time_to_detect_oom=time_to_detect_oom,
            recovery_time=recovery_time,
            calibration_error=calibration_error,
            n_steps=0,
        )

    n = len(events)
    correct = sum(1 for e in events if e.correct)
    net = sum(e.reward.net for e in events)
    obs_cost = sum(e.reward.observation_cost for e in events)
    interventions = sum(1 for e in events if e.intervened)
    fallbacks = sum(1 for e in events if e.fallback)
    # Consequence-weighted loss: wrong predictions weighted by |consequence_utility|
    cwl = sum(
        abs(e.reward.consequence_utility) for e in events if not e.correct
    )
    fpr = (false_promotions / promotions) if promotions > 0 else 0.0

    return MetricSummary(
        accuracy=correct / n,
        net_reward=float(net),
        consequence_weighted_loss=float(cwl),
        observation_cost=float(obs_cost),
        intervention_count=interventions,
        fallback_count=fallbacks,
        false_promotion_rate=float(fpr),
        time_to_detect_oom=time_to_detect_oom,
        recovery_time=recovery_time,
        calibration_error=calibration_error,
        n_steps=n,
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))
