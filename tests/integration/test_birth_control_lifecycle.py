"""Integration: Birth Control + genetic lifecycle on canonical Population."""

from __future__ import annotations

from driftbound.core.random import make_rng
from driftbound.smartdream.lifecycle.birth_control import BirthControl
from driftbound.smartdream.population import Population


def test_birth_death_atomic_on_canonical_population() -> None:
    rng = make_rng(10)
    pop = Population(max_size=8)
    bc = BirthControl(population=pop, fitness_threshold=0.0, min_population=2)
    seeded = bc.seed_population(5, rng)
    assert len(seeded) == 5
    assert pop.size() == 5

    # Improve fitness so reproduction eligibility can pass
    for a in pop.living():
        for _ in range(5):
            a.record_outcome(correct=True, consequence=0.5, utility=0.5)

    child = bc.reproduce(rng)
    assert child is not None
    assert child.id in {a.id for a in pop.all_agents()}
    assert child.generation >= 1
    assert len(child.parent_ids) == 2
    assert pop.get(child.parent_ids[0]) is not None

    # Fill to capacity then replace
    while pop.size() < pop.max_size:
        child2 = bc.reproduce(rng)
        assert child2 is not None
    assert pop.size() == pop.max_size
    deaths_before = int(pop.summary()["deaths"])
    born = bc.replace(rng, deaths=1, births=1)
    assert len(born) == 1
    assert pop.size() == pop.max_size
    assert int(pop.summary()["deaths"]) == deaths_before + 1


def test_extinction_protection() -> None:
    rng = make_rng(11)
    pop = Population(max_size=5)
    bc = BirthControl(population=pop, min_population=2, fitness_threshold=0.0)
    bc.seed_population(2, rng)
    # Cannot cull below floor
    assert bc.cull_least_fit(rng) is None
    assert pop.size() == 2

def test_offspring_participate_in_later_lifecycle_steps() -> None:
    """Children registered into the canonical Population act as later parents."""
    rng = make_rng(21)
    pop = Population(max_size=10)
    bc = BirthControl(population=pop, fitness_threshold=0.0, min_population=2)
    bc.seed_population(3, rng)
    for a in pop.living():
        for _ in range(5):
            a.record_outcome(correct=True, consequence=0.5, utility=0.5)

    child = bc.reproduce(rng)
    assert child is not None
    living_ids = {a.id for a in pop.living()}
    assert child.id in living_ids

    # Offspring participates: accumulates outcomes in subsequent steps
    for step in range(3):
        child.record_outcome(correct=True, consequence=0.4, utility=0.4, step=step)
    assert child.recent_correctness == 1.0
    assert child in pop.living()

    # And can be selected as a parent in a later birth on the same Population
    grandchild = bc.reproduce(rng)
    assert grandchild is not None
    assert child.id in grandchild.parent_ids or any(
        p in grandchild.parent_ids for p in child.parent_ids
    ) or grandchild.id in {a.id for a in pop.living()}
    # Stronger: at least one later birth lists the child as parent across a few tries
    seen_as_parent = child.id in grandchild.parent_ids
    for _ in range(20):
        if pop.size() >= pop.max_size:
            bc.cull_least_fit(rng)
        nxt = bc.reproduce(rng)
        if nxt is not None and child.id in nxt.parent_ids:
            seen_as_parent = True
            break
    assert seen_as_parent, "offspring never selected as parent in later steps"

