from driftbound.core.random import make_rng, spawn_rng


def test_spawn_independent() -> None:
    parent = make_rng(42)
    children = spawn_rng(parent, 2)
    a = [float(children[0].random()) for _ in range(5)]
    b = [float(children[1].random()) for _ in range(5)]
    assert a != b
