"""Numeric genome and phenotype expression.

Traits (bounded):
  observation_sensitivity, caution (intervention threshold), learning_rate,
  exploration_tendency, validation_threshold
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

TRAIT_BOUNDS: dict[str, tuple[float, float]] = {
    "observation_sensitivity": (0.0, 1.0),
    "caution": (0.0, 1.0),
    "learning_rate": (0.01, 0.5),
    "exploration_tendency": (0.0, 1.0),
    "validation_threshold": (0.5, 0.99),
}

TRAIT_NAMES = tuple(TRAIT_BOUNDS.keys())


def _clip_trait(name: str, value: float) -> float:
    lo, hi = TRAIT_BOUNDS[name]
    return float(np.clip(value, lo, hi))


@dataclass(frozen=True)
class Phenotype:
    """Expressed controller-facing parameters derived from genome traits."""

    observe_prob: float
    intervention_threshold: float
    learning_rate: float
    explore_prob: float
    validation_threshold: float

    def as_dict(self) -> dict[str, float]:
        return {
            "observe_prob": self.observe_prob,
            "intervention_threshold": self.intervention_threshold,
            "learning_rate": self.learning_rate,
            "explore_prob": self.explore_prob,
            "validation_threshold": self.validation_threshold,
        }


@dataclass
class Genome:
    """Numeric trait vector with crossover, mutation, and phenotype expression."""

    traits: dict[str, float]

    def __post_init__(self) -> None:
        clipped = {k: _clip_trait(k, float(self.traits[k])) for k in TRAIT_NAMES}
        # Preserve any unexpected keys only if present; canonical set is TRAIT_NAMES
        self.traits = clipped

    @classmethod
    def random(cls, rng: Generator) -> Genome:
        traits = {
            name: float(rng.uniform(lo, hi)) for name, (lo, hi) in TRAIT_BOUNDS.items()
        }
        return cls(traits=traits)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> Genome:
        return cls(traits={k: float(mapping[k]) for k in TRAIT_NAMES})

    def crossover(self, other: Genome, rng: Generator) -> Genome:
        """Uniform blend crossover with equal parent weight plus small noise."""
        child: dict[str, float] = {}
        for name in TRAIT_NAMES:
            a, b = self.traits[name], other.traits[name]
            blend = 0.5 * (a + b)
            noise = float(rng.normal(0.0, 0.02))
            child[name] = _clip_trait(name, blend + noise)
        return Genome(traits=child)

    def mutate(self, rng: Generator, rate: float = 0.2, scale: float = 0.08) -> Genome:
        """Return a mutated copy; each trait mutates independently with ``rate``."""
        new_traits = dict(self.traits)
        for name in TRAIT_NAMES:
            if rng.random() < rate:
                lo, hi = TRAIT_BOUNDS[name]
                delta = float(rng.normal(0.0, scale * (hi - lo)))
                new_traits[name] = _clip_trait(name, new_traits[name] + delta)
        return Genome(traits=new_traits)

    def phenotype(self) -> Phenotype:
        t = self.traits
        return Phenotype(
            observe_prob=t["observation_sensitivity"],
            intervention_threshold=t["caution"],
            learning_rate=t["learning_rate"],
            explore_prob=t["exploration_tendency"],
            validation_threshold=t["validation_threshold"],
        )

    def as_dict(self) -> dict[str, float]:
        return dict(self.traits)
