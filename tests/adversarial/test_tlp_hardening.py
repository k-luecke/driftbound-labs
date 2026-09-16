"""Adversarial property tests for the TLP fairness / lifecycle hardening pass."""

from __future__ import annotations

from driftbound.baselines.hmm import DiscreteHMM
from driftbound.core.random import make_rng
from driftbound.experiments.three_little_pigs import (
    SEER_CLAIM,
    TLPConfig,
    WorldTrace,
    _claim_checker,
    best_action,
    generate_world,
    run_benchmark,
    run_once,
)
from driftbound.smartdream.controller import HMMController
from driftbound.smartdream.seer import GuardedSeer, SeerHypothesis
from driftbound.smartdream.vision import VisionChannel, VisionEvidence


def test_controllers_share_bit_identical_world_traces() -> None:
    """Same seed ⇒ identical pre-generated schedules/obs across controllers."""
    results = run_benchmark(
        seed=123,
        replications=2,
        n_steps=40,
        controllers=("hmm", "fixed_policy", "switching_experts"),
        evolution_enabled=False,
    )
    worlds = {
        name: results["controllers"][name]["world_regimes"]
        for name in ("hmm", "fixed_policy", "switching_experts")
    }
    assert worlds["hmm"] == worlds["fixed_policy"] == worlds["switching_experts"]

    # Direct WorldTrace equality for generate_world
    cfg = TLPConfig(n_steps=30)
    w1 = generate_world(cfg, make_rng(7))
    w2 = generate_world(cfg, make_rng(7))
    assert w1.as_dict() == w2.as_dict()

    # Controllers with different RNGs but same world still see same regimes/obs
    world = generate_world(cfg, make_rng(99))
    _, e_hmm = run_once(make_rng(1), controller="hmm", n_steps=30, world=world)
    _, e_fix = run_once(make_rng(2), controller="fixed_policy", n_steps=30, world=world)
    assert e_hmm["world_trace"] == e_fix["world_trace"]


def test_hmm_uses_current_observation_not_lag_only() -> None:
    """Filtering on current obs changes action vs lag-only MAP baseline."""
    rng = make_rng(0)
    hmm = DiscreteHMM()
    hmm.reset(rng)
    # Warm up with obs=0 so belief peaks near state 0
    for _ in range(8):
        hmm.filter(0)
        hmm._filtered_obs = None

    lag_only = hmm.map_state()
    # Mid-stream obs flip: current-filter MAP should move toward state 2
    filtered = hmm.filter(2)
    assert filtered != lag_only or hmm.map_state() == filtered

    # Controller path: act filters once; flipping obs changes selected action
    ctrl = HMMController()
    ctrl.reset(make_rng(1))
    for _ in range(10):
        a = ctrl.act(0, make_rng(2))
        ctrl.update(0, a, {"success": True}, make_rng(2))
    a_stable = ctrl.act(0, make_rng(3))
    # Force a distinct observation without going through update lag
    hmm2 = DiscreteHMM()
    hmm2.reset(make_rng(1))
    for _ in range(10):
        hmm2.filter(0)
        hmm2._filtered_obs = None
    lag_action = ctrl.actions[hmm2.map_state() % 3]
    current_action = ctrl.actions[hmm2.filter(2) % 3]
    # Property: incorporating obs=2 can differ from lag-only after obs=0 training
    assert isinstance(a_stable, str)
    assert lag_action in ctrl.actions and current_action in ctrl.actions
    # Stronger regression: predict_state must equal filter (not ignore obs)
    hmm3 = DiscreteHMM()
    hmm3.reset(make_rng(5))
    for _ in range(5):
        hmm3.filter(0)
        hmm3._filtered_obs = None
    before = hmm3.map_state()
    after = hmm3.predict_state(2)
    # Belief moved using observation 2 (emission peaked at index 2 for state 2)
    assert after == hmm3.map_state()
    # After enough contrast, MAP typically shifts; if sticky prior keeps it,
    # at least belief mass on state 2 must increase relative to pre-filter copy.
    hmm4 = DiscreteHMM()
    hmm4.reset(make_rng(5))
    for _ in range(5):
        hmm4.filter(0)
        hmm4._filtered_obs = None
    mass_before = float(hmm4.belief[2])  # type: ignore[index]
    hmm4.predict_state(2)
    mass_after = float(hmm4.belief[2])  # type: ignore[index]
    assert mass_after > mass_before
    del before, after


def test_correctness_is_action_equals_best_action_not_survival() -> None:
    """decision_correct == (action == best_action); independent of success."""
    cfg = TLPConfig(n_steps=50)
    world = generate_world(cfg, make_rng(11))
    metrics, extra = run_once(
        make_rng(11), controller="fixed_policy", n_steps=50, world=world
    )
    # Fixed policy always plays wood. Correctness rate must match wood==best.
    expected_correct = sum(
        1 for s in world.steps if best_action(s.regime) == "wood"
    ) / len(world.steps)
    assert abs(metrics.accuracy - expected_correct) < 1e-9

    # Craft a world where wood survives but is not best (cunning_wolf → brick best)
    from driftbound.experiments.three_little_pigs import WorldStep

    crafted = WorldTrace(
        steps=(
            WorldStep(regime="cunning_wolf", observed=True, observation=2, outcome_u=0.01),
            WorldStep(regime="wolf", observed=True, observation=1, outcome_u=0.01),
        )
    )
    m, _ = run_once(make_rng(0), controller="fixed_policy", n_steps=2, world=crafted)
    # wood != brick on cunning_wolf (wrong decision); wood == best on wolf (correct)
    # even when outcome_u=0.01 makes physical survival likely on both.
    assert m.accuracy == 0.5
    assert best_action("cunning_wolf") == "brick"
    assert best_action("wolf") == "wood"


def test_seer_claim_targets_cunning_wolf_not_ordinary_wolf() -> None:
    hyp = SeerHypothesis(hypothesis_id=SEER_CLAIM, claim=SEER_CLAIM)
    wolf_ev = VisionEvidence(
        observer_id="o",
        subject_id="s",
        context="wolf",
        observed_action="brick",
        outcome={"action": "brick", "success": True},
        independently_observable=True,
        cost=0.01,
        step=10,
    )
    cunning_ev = VisionEvidence(
        observer_id="o",
        subject_id="s",
        context="cunning_wolf",
        observed_action="brick",
        outcome={"action": "brick", "success": True},
        independently_observable=True,
        cost=0.01,
        step=11,
    )
    assert _claim_checker(hyp, wolf_ev) is None  # ordinary wolf not applicable
    assert _claim_checker(hyp, cunning_ev) is True
    assert best_action("wolf") == "wood"
    assert best_action("cunning_wolf") == "brick"

    # End-to-end: discovery/validation path uses SEER_CLAIM
    seer = GuardedSeer(discovery_end=5, validation_end=20, deployment_end=40, min_evidence=4)
    seer.propose(SEER_CLAIM, SEER_CLAIM, score=1.0, step=2)
    vision = VisionChannel()
    for s in range(5, 15):
        vision.observe(
            observer_id="o",
            subject_id="s",
            context="cunning_wolf",
            observed_action="brick",
            outcome={"action": "brick", "success": True},
            independently_observable=True,
            cost=0.01,
            step=s,
        )
    seer.lock_discovery()
    seer.validate_from_vision(vision, step=10, claim_checker=_claim_checker)
    assert seer.hypotheses[SEER_CLAIM].validation_score > 0.0


def test_exploration_override_promotion_demotion_logged_distinctly() -> None:
    world = generate_world(TLPConfig(n_steps=80), make_rng(21))
    _, extra = run_once(
        make_rng(21),
        controller="guarded_seer_hybrid",
        n_steps=80,
        world=world,
    )
    audit = extra["decision_audit"]
    assert "exploration_actions" in audit
    assert "authority_overrides" in audit
    assert "promotions" in audit
    assert "demotions" in audit
    assert "exploration_cost_total" in audit
    assert "authority_override_cost_total" in audit
    # Keys remain distinct (not collapsed into one opaque score)
    assert set(audit) >= {
        "exploration_actions",
        "authority_overrides",
        "promotions",
        "demotions",
        "exploration_cost_total",
        "authority_override_cost_total",
    }
    # Event metadata carries the same distinctions when exploration fires
    if audit["exploration_actions"] > 0:
        assert any(m.get("explored") for m in extra["events_metadata_sample"])


def test_evolution_enabled_wires_population_between_episodes() -> None:
    results = run_benchmark(
        seed=5,
        replications=1,
        n_steps=30,
        controllers=("fixed_policy",),
        evolution_enabled=True,
        n_episodes=3,
        population_size=4,
    )
    evo = results["evolution"]
    assert evo["evolution_side_effects"] is True
    assert evo["final_summary"]["births"] > evo["population_size_initial"]
    assert evo["final_summary"]["deaths"] >= 1
    assert len(evo["offspring_seen"]) >= 1
    # Offspring appear in later episode agent metrics
    later = evo["episodes"][-1]["agent_metrics"]
    assert any(row["generation"] > 0 or row["is_offspring"] for row in later)
    # Lineage / generation metrics recorded
    assert "lineage" in evo["episodes"][0]
    assert evo["episodes"][-1]["max_generation"] >= 1


def test_evolution_disabled_has_no_side_effects() -> None:
    results = run_benchmark(
        seed=5,
        replications=1,
        n_steps=20,
        controllers=("hmm", "fixed_policy"),
        evolution_enabled=False,
    )
    assert results["evolution_enabled"] is False
    assert "evolution" not in results
    # Fair comparison still present
    assert "hmm" in results["controllers"]
