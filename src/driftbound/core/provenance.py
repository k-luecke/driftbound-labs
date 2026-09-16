"""Lightweight provenance records for experiment reproducibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ProvenanceRecord:
    experiment_name: str
    seed: int
    config: dict[str, Any]
    package_version: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
