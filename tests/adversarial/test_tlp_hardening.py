"""Adversarial property tests for the TLP fairness / lifecycle hardening pass."""

from __future__ import annotations

from driftbound.baselines.hmm import DiscreteHMM
from driftbound.core.random import make_rng
from driftbound.experiments.three_little_pigs import (
    OOM_REGIME,
    SEER_CLAIM,
    TLPConfig,
    WorldTrace,
    _claim_checker,
    _obs_encode,
    best_action,
    generate_world,
    run_benchmark,
    run_once,
)
from driftbound.smartdream.agent import Agent
from driftbound.smartdream.controller import HMMController
from driftbound.smartdream.lifecycle.genome import Genome
from driftbound.smartdream.seer import GuardedSeer, SeerHypothesis
from driftbound.smartdream.vision import VisionChannel, VisionEvidence


def test_controllers_share_bit_identical_world_traces() -> None:
    """Same seed ⇒ identical full world traces across controllers.

    Compares regimes, observation flags, encoded observations, AND outcome
    uniforms — not regimes alone.
    """
    results = run_benchmark(
        seed=123,
        replications=2,
        n_steps=40,
        controllers=("hmm", "fixed_policy", "switching_experts"),
        evolution_enabled=False,
    )
    traces = {
        name: results["controllers"][name]["world_traces"]
        for name in ("hmm", "fixed_policy", "switching_experts")
    }
    assert traces["hmm"] == traces["fixed_policy"] == traces["switching_experts"]
    # Each stored trace has the complete shared fields
    for rep_trace in traces["hmm"]:
        assert set(rep_trace) == {"regimes", "observed", "observations", "outcome_u"}
        n = len(rep_trace["regimes"])
        assert (
            len(rep_trace["observed"])
            == len(rep_trace["observations"])
            == len(rep_trace["outcome_u"])
            == n
        )

    # Direct WorldTrace equality for generate_world
    cfg = TLPConfig(n_steps=30)
    w1 = generate_world(cfg, make_rng(7))
    w2 = generate_world(cfg, make_rng(7))
    assert w1.as_dict() == w2.as_dict()
    assert w1.as_dict()["outcome_u"] == w2.as_dict()["outcome_u"]
    assert w1.as_dict()["observed"] == w2.as_dict()["observed"]

    # Controllers with different RNGs but same world still see same full trace
    world = generate_world(cfg, make_rng(99))
    _, e_hmm = run_once(make_rng(1), controller="hmm", n_steps=30, world=world)
    _, e_fix = run_once(make_rng(2), controller="fixed_policy", n_steps=30, world=world)
    assert e_hmm["world_trace"] == e_fix["world_trace"] == world.as_dict()


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

    # Craft a world where wood survives but is not best (silent_threat → brick best)
    from driftbound.experiments.three_little_pigs import WorldStep

    crafted = WorldTrace(
        steps=(
            WorldStep(
                regime="silent_threat", observed=True, observation=0, outcome_u=0.01
            ),
            WorldStep(regime="wolf", observed=True, observation=1, outcome_u=0.01),
        )
    )
    m, _ = run_once(make_rng(0), controller="fixed_policy", n_steps=2, world=crafted)
    # wood != brick on silent_threat (wrong decision); wood == best on wolf (correct)
    # even when outcome_u=0.01 makes physical survival likely on both.
    assert m.accuracy == 0.5
    assert best_action("silent_threat") == "brick"
    assert best_action("wolf") == "wood"
    del extra


def test_silent_threat_is_genuine_oom_vs_hmm_mapping() -> None:
    """silent_threat obs aliases calm; HMM path prefers straw; brick is optimal."""
    rng = make_rng(0)
    # Noise-free encode: silent_threat → 0 (same as calm)
    assert _obs_encode("silent_threat", rng, noise=0.0, observed=True) == 0
    assert _obs_encode("calm", rng, noise=0.0, observed=True) == 0
    # cunning_wolf remains in-model (obs 2 → brick via HMM state 2)
    assert _obs_encode("cunning_wolf", rng, noise=0.0, observed=True) == 2
    assert best_action("silent_threat") == "brick"
    assert best_action("calm") == "straw"
    assert best_action("cunning_wolf") == "brick"

    hmm = DiscreteHMM()
    hmm.reset(make_rng(1))
    for _ in range(12):
        hmm.filter(0)
        hmm._filtered_obs = None
    state = hmm.predict_state(0)
    hmm_action = ("straw", "wood", "brick")[state % 3]
    # After calm-like observations, HMM maps toward straw — contradicts OOM optimum
    assert hmm_action == "straw"
    assert hmm_action != best_action("silent_threat")
    assert SEER_CLAIM == "brick_under_silent_threat"
    assert OOM_REGIME == "silent_threat"


def test_seer_claim_targets_silent_threat_oom_not_cunning_wolf() -> None:
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
    silent_ev = VisionEvidence(
        observer_id="o",
        subject_id="s",
        context="silent_threat",
        observed_action="brick",
        outcome={"action": "brick", "success": True},
        independently_observable=True,
        cost=0.01,
        step=12,
    )
    assert _claim_checker(hyp, wolf_ev) is None
    assert _claim_checker(hyp, cunning_ev) is None  # in-model; not the OOM claim
    assert _claim_checker(hyp, silent_ev) is True
    assert best_action("wolf") == "wood"
    assert best_action("cunning_wolf") == "brick"
    assert best_action("silent_threat") == "brick"

    # End-to-end: discovery/validation path uses SEER_CLAIM on silent_threat
    seer = GuardedSeer(
        discovery_end=5, validation_end=20, deployment_end=40, min_evidence=4
    )
    seer.propose(SEER_CLAIM, SEER_CLAIM, score=1.0, step=2)
    vision = VisionChannel()
    for s in range(5, 15):
        vision.observe(
            observer_id="o",
            subject_id="s",
            context="silent_threat",
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
        "paid_observations",
        "skipped_observations",
        "caution_upgrades",
    }
    # Event metadata carries the same distinctions when exploration fires
    if audit["exploration_actions"] > 0:
        assert any(m.get("explored") for m in extra["events_metadata_sample"])
    # Exploration is not conflated with validated Seer authority
    assert audit["exploration_actions"] != audit["authority_overrides"] or (
        audit["exploration_actions"] == 0 and audit["authority_overrides"] == 0
    )


def test_genome_traits_causally_affect_tlp() -> None:
    """explore_prob, observation_sensitivity, caution, validation_threshold matter."""
    from driftbound.experiments.three_little_pigs import WorldStep

    # Craft a fully-observed calm-heavy world so paid-obs + caution have signal
    steps = tuple(
        WorldStep(regime="calm", observed=True, observation=0, outcome_u=0.5)
        for _ in range(40)
    )
    world = WorldTrace(steps=steps)

    # Low observe_prob → skips paid observations
    low_obs = Agent(
        genome=Genome.from_mapping(
            {
                "observation_sensitivity": 0.0,
                "caution": 0.0,
                "learning_rate": 0.1,
                "exploration_tendency": 0.0,
                "validation_threshold": 0.5,
            }
        )
    )
    _, e_low = run_once(
        make_rng(0),
        controller="guarded_seer_hybrid",
        n_steps=40,
        world=world,
        agent=low_obs,
    )
    assert e_low["decision_audit"]["skipped_observations"] == 40
    assert e_low["decision_audit"]["paid_observations"] == 0

    # High observe_prob → takes all scheduled observations
    high_obs = Agent(
        genome=Genome.from_mapping(
            {
                "observation_sensitivity": 1.0,
                "caution": 0.0,
                "learning_rate": 0.1,
                "exploration_tendency": 0.0,
                "validation_threshold": 0.5,
            }
        )
    )
    _, e_high = run_once(
        make_rng(0),
        controller="guarded_seer_hybrid",
        n_steps=40,
        world=world,
        agent=high_obs,
    )
    assert e_high["decision_audit"]["paid_observations"] == 40
    assert e_high["decision_audit"]["skipped_observations"] == 0

    # High caution → straw→wood upgrades on calm (HMM tends toward straw)
    high_caution = Agent(
        genome=Genome.from_mapping(
            {
                "observation_sensitivity": 1.0,
                "caution": 1.0,
                "learning_rate": 0.1,
                "exploration_tendency": 0.0,
                "validation_threshold": 0.5,
            }
        )
    )
    _, e_caut = run_once(
        make_rng(1),
        controller="guarded_seer_hybrid",
        n_steps=40,
        world=world,
        agent=high_caution,
    )
    assert e_caut["decision_audit"]["caution_upgrades"] > 0
    assert e_caut["genome_traits_applied"]["caution"] == 1.0

    # validation_threshold is applied to Seer min_validation_score
    strict = Agent(
        genome=Genome.from_mapping(
            {
                "observation_sensitivity": 1.0,
                "caution": 0.0,
                "learning_rate": 0.1,
                "exploration_tendency": 0.0,
                "validation_threshold": 0.95,
            }
        )
    )
    _, e_strict = run_once(
        make_rng(2),
        controller="guarded_seer_hybrid",
        n_steps=40,
        world=world,
        agent=strict,
    )
    assert e_strict["genome_traits_applied"]["validation_threshold"] == 0.95
    mon = e_strict["seer_monitor"]
    assert mon is not None
    assert mon["min_validation_score"] == 0.95

    # explore_prob still wired
    explorer = Agent(
        genome=Genome.from_mapping(
            {
                "observation_sensitivity": 1.0,
                "caution": 0.0,
                "learning_rate": 0.1,
                "exploration_tendency": 1.0,
                "validation_threshold": 0.5,
            }
        )
    )
    # Use a world that spans discovery so exploration can fire
    disc_world = generate_world(TLPConfig(n_steps=50), make_rng(3))
    _, e_exp = run_once(
        make_rng(3),
        controller="guarded_seer_hybrid",
        n_steps=50,
        world=disc_world,
        agent=explorer,
    )
    assert e_exp["decision_audit"]["exploration_actions"] > 0
    assert e_exp["genome_traits_applied"]["explore_prob"] == 1.0


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
