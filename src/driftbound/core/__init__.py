"""Core experiment primitives: RNG, events, metrics, provenance."""

from driftbound.core.events import RewardComponents, StepEvent
from driftbound.core.experiment import ExperimentConfig, ExperimentResult, run_replications
from driftbound.core.metrics import MetricSummary, summarize_metrics
from driftbound.core.provenance import ProvenanceRecord
from driftbound.core.random import make_rng, spawn_rng

__all__ = [
    "RewardComponents",
    "StepEvent",
    "ExperimentConfig",
    "ExperimentResult",
    "run_replications",
    "MetricSummary",
    "summarize_metrics",
    "ProvenanceRecord",
    "make_rng",
    "spawn_rng",
]
