"""Deterministic NumPy RNG helpers. Always pass RNG explicitly; never use global state."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def make_rng(seed: int | None = None) -> Generator:
    """Create a NumPy Generator from an integer seed (or non-deterministic if None)."""
    return np.random.default_rng(seed)


def spawn_rng(parent: Generator, n: int = 1) -> list[Generator]:
    """Spawn ``n`` child generators from ``parent`` for independent streams."""
    seeds = parent.integers(0, 2**31 - 1, size=n, dtype=np.int64)
    return [np.random.default_rng(int(s)) for s in seeds]
