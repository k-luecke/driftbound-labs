"""Experiment configuration and replication helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from driftbound.core.metrics import MetricSummary, mean_std
from driftbound.core.provenance import ProvenanceRecord
from driftbound.core.random import make_rng, spawn_rng


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    replications: int = 1
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    metrics: list[MetricSummary]
    provenance: ProvenanceRecord
    extras: dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> dict[str, Any]:
        keys = [
            "accuracy",
            "net_reward",
            "consequence_weighted_loss",
            "observation_cost",
            "false_promotion_rate",
        ]
        out: dict[str, Any] = {"n_replications": len(self.metrics)}
        for k in keys:
            vals = [float(getattr(m, k)) for m in self.metrics]
            mean, std = mean_std(vals)
            out[f"{k}_mean"] = mean
            out[f"{k}_std"] = std
        return out


def run_replications(
    config: ExperimentConfig,
    runner: Callable[..., MetricSummary | tuple[MetricSummary, dict[str, Any]]],
    *,
    package_version: str = "0.1.0",
) -> ExperimentResult:
    """Run ``config.replications`` independent trials with spawned RNGs."""
    parent = make_rng(config.seed)
    child_rngs = spawn_rng(parent, config.replications)
    metrics: list[MetricSummary] = []
    extras_acc: list[dict[str, Any]] = []
    for i, rng in enumerate(child_rngs):
        result = runner(rng=rng, replication=i, **config.params)
        if isinstance(result, tuple):
            metric, extra = result
            metrics.append(metric)
            extras_acc.append(extra)
        else:
            metrics.append(result)
    prov = ProvenanceRecord(
        experiment_name=config.name,
        seed=config.seed,
        config={"replications": config.replications, **config.params},
        package_version=package_version,
    )
    return ExperimentResult(
        config=config,
        metrics=metrics,
        provenance=prov,
        extras={"per_replication": extras_acc} if extras_acc else {},
    )
