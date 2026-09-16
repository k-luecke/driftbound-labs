"""Page-Hinkley change detector (classic baseline)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkleyDetector:
    """Detect mean shifts in a univariate stream."""

    delta: float = 0.005
    threshold: float = 0.05
    min_instances: int = 10
    _mean: float = 0.0
    _sum: float = 0.0
    _min_sum: float = 0.0
    _n: int = 0
    changed: bool = False

    def reset(self) -> None:
        self._mean = 0.0
        self._sum = 0.0
        self._min_sum = 0.0
        self._n = 0
        self.changed = False

    def update(self, value: float) -> bool:
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum += value - self._mean - self.delta
        self._min_sum = min(self._min_sum, self._sum)
        ph = self._sum - self._min_sum
        if self._n >= self.min_instances and ph > self.threshold:
            self.changed = True
        return self.changed
