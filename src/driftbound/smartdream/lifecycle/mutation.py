"""Mutation operators (thin wrappers for clarity in lifecycle pipelines)."""

from __future__ import annotations

from numpy.random import Generator

from driftbound.smartdream.lifecycle.genome import Genome


def mutate_genome(
    genome: Genome,
    rng: Generator,
    *,
    rate: float = 0.2,
    scale: float = 0.08,
) -> Genome:
    return genome.mutate(rng, rate=rate, scale=scale)
