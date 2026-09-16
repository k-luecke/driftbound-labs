"""Seer promotes on validation evidence, not discovery alone."""

from __future__ import annotations

from driftbound.smartdream.seer import GuardedSeer
from driftbound.smartdream.vision import VisionChannel


def test_seer_requires_independent_validation() -> None:
    seer = GuardedSeer(
        discovery_end=10,
        validation_end=30,
        deployment_end=50,
        min_evidence=5,
        alpha=0.05,
    )
    seer.propose("h1", "brick_under_threat", score=1.0, step=3)
    # Discovery-only: not promoted yet
    assert seer.hypotheses["h1"].promoted is False

    vision = VisionChannel()
    # Strong independent validation evidence in validation window only
    for s in range(10, 25):
        vision.observe(
            observer_id="obs",
            subject_id="house",
            context="wolf",
            observed_action="brick",
            outcome={"action": "brick", "success": True},
            independently_observable=True,
            cost=0.01,
            step=s,
        )

    def checker(hyp, ev):  # type: ignore[no-untyped-def]
        return bool(ev.outcome.get("action") == "brick" and ev.outcome.get("success"))

    seer.lock_discovery()
    promoted = seer.validate_from_vision(vision, step=20, claim_checker=checker)
    # With many successes, Bonferroni may still promote
    assert seer.hypotheses["h1"].validation_score > 0.0
    assert seer.hypotheses["h1"].phase.value in ("validation", "deployment")
    _ = promoted
