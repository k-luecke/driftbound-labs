import pytest

from driftbound.smartdream.seer import GuardedSeer
from driftbound.smartdream.vision import VisionChannel


def test_no_propose_outside_discovery() -> None:
    seer = GuardedSeer(discovery_end=10, validation_end=20, deployment_end=30)
    with pytest.raises(ValueError):
        seer.propose("h1", "claim", 1.0, step=15)


def test_validation_rejects_discovery_leakage_path() -> None:
    seer = GuardedSeer(discovery_end=5, validation_end=15, deployment_end=25, min_evidence=2)
    seer.propose("h1", "brick_under_threat", 1.0, step=2)
    seer.lock_discovery()
    vision = VisionChannel()
    for s in range(5, 12):
        vision.observe(
            observer_id="o",
            subject_id="s",
            context="wolf",
            observed_action="brick",
            outcome={"action": "brick", "success": True},
            independently_observable=True,
            cost=0.01,
            step=s,
        )

    def checker(hyp, ev):  # type: ignore[no-untyped-def]
        return ev.outcome.get("success") is True

    promoted = seer.validate_from_vision(vision, step=10, claim_checker=checker)
    assert isinstance(promoted, list)
    # Authority is temporary
    action, used = seer.advisory_action(16, "wood", "brick")
    assert action in ("wood", "brick")
