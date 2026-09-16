"""Genome mutation stays in bounds and changes traits."""

from __future__ import annotations

from driftbound.core.random import make_rng
from driftbound.smartdream.lifecycle.genome import TRAIT_BOUNDS, TRAIT_NAMES, Genome


def test_random_genome_in_bounds() -> None:
    rng = make_rng(0)
    g = Genome.random(rng)
    for name, (lo, hi) in TRAIT_BOUNDS.items():
        assert lo <= g.traits[name] <= hi


def test_mutation_changes_traits_within_bounds() -> None:
    rng = make_rng(1)
    g0 = Genome.random(rng)
    # Force high mutation rate / scale to ensure change
    g1 = g0.mutate(rng, rate=1.0, scale=0.2)
    assert any(g0.traits[n] != g1.traits[n] for n in TRAIT_NAMES)
    for name, (lo, hi) in TRAIT_BOUNDS.items():
        assert lo <= g1.traits[name] <= hi


def test_crossover_in_bounds() -> None:
    rng = make_rng(2)
    a, b = Genome.random(rng), Genome.random(rng)
    child = a.crossover(b, rng)
    for name, (lo, hi) in TRAIT_BOUNDS.items():
        assert lo <= child.traits[name] <= hi


def test_phenotype_maps_traits() -> None:
    g = Genome.from_mapping(
        {
            "observation_sensitivity": 0.7,
            "caution": 0.4,
            "learning_rate": 0.1,
            "exploration_tendency": 0.2,
            "validation_threshold": 0.8,
        }
    )
    p = g.phenotype()
    assert p.observe_prob == 0.7
    assert p.intervention_threshold == 0.4
