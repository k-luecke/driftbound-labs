"""Shared protocols / interfaces for controllers and environments."""

from __future__ import annotations

from typing import Any, Protocol

from numpy.random import Generator


class Controller(Protocol):
    """Minimal controller interface used by experiments."""

    name: str

    def reset(self, rng: Generator) -> None: ...

    def act(self, observation: Any, rng: Generator) -> Any: ...

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None: ...
