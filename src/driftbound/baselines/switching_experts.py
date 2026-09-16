"""Fixed Share switching experts (Herbster & Warmuth style baseline).

Honest label: classic Fixed Share over a small discrete expert pool.
Specialist pools / sleeping experts / Bayesian model averaging: future work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass
class FixedShareExperts:
    """Weighted experts with Fixed Share mixing."""

    actions: tuple[str, ...] = ("straw", "wood", "brick")
    alpha: float = 0.1  # share parameter
    eta: float = 2.0  # learning rate
    weights: np.ndarray | None = None

    def reset(self, rng: Generator) -> None:
        n = len(self.actions)
        self.weights = np.ones(n) / n

    def act(self, observation: object, rng: Generator) -> str:
        assert self.weights is not None
        # Softmax sampling from weights
        p = self.weights / self.weights.sum()
        idx = int(rng.choice(len(self.actions), p=p))
        return self.actions[idx]

    def update(
        self,
        observation: object,
        action: object,
        outcome: object,
        rng: Generator,
    ) -> None:
        assert self.weights is not None
        n = len(self.actions)
        # Loss: 0 if action matched a truth encoded in outcome dict/str
        losses = np.ones(n)
        truth = outcome
        if isinstance(outcome, dict):
            truth = outcome.get("best_action")
        for i, a in enumerate(self.actions):
            losses[i] = 0.0 if a == truth else 1.0
        # Multiplicative weights
        self.weights *= np.exp(-self.eta * losses)
        # Fixed share mix
        pool = self.alpha / n
        self.weights = (1 - self.alpha) * self.weights + pool
        self.weights /= self.weights.sum()
