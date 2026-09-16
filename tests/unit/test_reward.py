from driftbound.smartdream.reward import decompose_reward


def test_reward_decomposition_net() -> None:
    r = decompose_reward(
        correct=True,
        consequence=0.5,
        observation_cost=0.1,
        intervention_cost=0.2,
        fallback_cost=0.0,
        correctness_value=1.0,
    )
    assert r.net == 1.0 + 0.5 - 0.1 - 0.2 - 0.0


def test_incorrect_negates_consequence() -> None:
    r = decompose_reward(correct=False, consequence=0.8)
    assert r.consequence_utility == -0.8
    assert r.correctness_reward == 0.0
