"""Bounded episodic memory for agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    step: int
    signal: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Memory:
    """Fixed-capacity event buffer (no pandas dependency)."""

    agent_id: str
    capacity: int = 256
    _events: deque[MemoryRecord] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._events = deque(maxlen=self.capacity)

    def record(self, step: int, signal: str, metadata: dict[str, Any] | None = None) -> None:
        self._events.append(
            MemoryRecord(step=step, signal=signal, metadata=metadata or {})
        )

    def recent(self, n: int = 20) -> list[MemoryRecord]:
        items = list(self._events)
        return items[-n:]

    def count(self, signal: str | None = None) -> int:
        if signal is None:
            return len(self._events)
        return sum(1 for e in self._events if e.signal == signal)

    def clear(self) -> None:
        self._events.clear()
