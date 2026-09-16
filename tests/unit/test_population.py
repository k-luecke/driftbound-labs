from driftbound.core.random import make_rng
from driftbound.smartdream.agent import Agent
from driftbound.smartdream.lifecycle.genome import Genome
from driftbound.smartdream.population import Population


def test_canonical_population_register_and_death() -> None:
    rng = make_rng(3)
    pop = Population(max_size=10)
    a = Agent(genome=Genome.random(rng))
    pop.register(a)
    assert pop.size() == 1
    pop.mark_dead(a.id, reason="test")
    assert pop.size() == 0
    assert len(pop.dead()) == 1
    assert pop.get(a.id) is a
