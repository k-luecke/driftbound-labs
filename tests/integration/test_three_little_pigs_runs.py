from driftbound.core.random import make_rng
from driftbound.experiments.three_little_pigs import run_benchmark, run_once


def test_run_once_deterministic() -> None:
    m1, _ = run_once(make_rng(42), controller="hmm", n_steps=50)
    m2, _ = run_once(make_rng(42), controller="hmm", n_steps=50)
    assert m1.as_dict() == m2.as_dict()


def test_benchmark_smoke() -> None:
    results = run_benchmark(seed=0, replications=1, n_steps=40)
    assert "hmm" in results["controllers"]
    assert results["controllers"]["fixed_policy"]["aggregate"]["n_replications"] == 1
