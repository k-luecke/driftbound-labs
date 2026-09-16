"""CI-small statistical / determinism checks with fixed seeds."""

from __future__ import annotations

from driftbound.core.random import make_rng
from driftbound.experiments.three_little_pigs import run_benchmark
from driftbound.smartdream.lifecycle.genome import TRAIT_BOUNDS, Genome


def test_benchmark_seed_42_stable() -> None:
    a = run_benchmark(seed=42, replications=2, n_steps=60, controllers=("hmm", "fixed_policy"))
    b = run_benchmark(seed=42, replications=2, n_steps=60, controllers=("hmm", "fixed_policy"))
    assert a["controllers"]["hmm"]["per_replication"] == b["controllers"]["hmm"]["per_replication"]


def test_genome_distribution_in_bounds_many_draws() -> None:
    rng = make_rng(99)
    for _ in range(200):
        g = Genome.random(rng)
        for name, (lo, hi) in TRAIT_BOUNDS.items():
            assert lo <= g.traits[name] <= hi
