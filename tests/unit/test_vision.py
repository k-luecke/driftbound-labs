from driftbound.smartdream.vision import VisionChannel


def test_observation_not_automatic_correctness() -> None:
    v = VisionChannel()
    ev = v.observe(
        observer_id="o1",
        subject_id="s1",
        context="wolf",
        observed_action="straw",
        outcome={"success": False},
        independently_observable=True,
        cost=0.02,
        step=1,
    )
    # Evidence exists but does not imply correctness
    assert ev.outcome["success"] is False
    assert v.total_cost == 0.02
    assert len(v.independent_evidence()) == 1
