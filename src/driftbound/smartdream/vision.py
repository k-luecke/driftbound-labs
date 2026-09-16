"""Vision: observational evidence channel.

Observation ≠ automatic correctness. Evidence records who saw whom do what,
under what context, with independent observability and cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VisionEvidence:
    """One observational evidence record."""

    observer_id: str
    subject_id: str
    context: str
    observed_action: str
    outcome: Any
    independently_observable: bool
    cost: float
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionChannel:
    """Collects observational evidence; does not assert correctness by itself."""

    evidences: list[VisionEvidence] = field(default_factory=list)
    total_cost: float = 0.0

    def observe(
        self,
        *,
        observer_id: str,
        subject_id: str,
        context: str,
        observed_action: str,
        outcome: Any,
        independently_observable: bool,
        cost: float,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> VisionEvidence:
        ev = VisionEvidence(
            observer_id=observer_id,
            subject_id=subject_id,
            context=context,
            observed_action=observed_action,
            outcome=outcome,
            independently_observable=independently_observable,
            cost=cost,
            step=step,
            metadata=metadata or {},
        )
        self.evidences.append(ev)
        self.total_cost += cost
        return ev

    def independent_evidence(self) -> list[VisionEvidence]:
        return [e for e in self.evidences if e.independently_observable]

    def for_subject(self, subject_id: str) -> list[VisionEvidence]:
        return [e for e in self.evidences if e.subject_id == subject_id]

    def in_window(self, start: int, end: int) -> list[VisionEvidence]:
        """Evidence with step in [start, end)."""
        return [e for e in self.evidences if start <= e.step < end]
