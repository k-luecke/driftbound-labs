"""Three Little Pigs adaptive assurance benchmark.

Agents / roles (symbolic names — not literal claims):
  Straw, Wood, Brick — defensive materials / policies of increasing cost & strength
  Wolf, CunningWolf, SilentThreat — adversarial pressure regimes

Controllers compared under equal budgets:
  HMM, HMM+change detection, switching experts, guarded Seer hybrid, fixed policy

Fairness: environment schedules and observation/noise draws are pre-generated
independently of controller RNG so every controller faces identical worlds.
Decision correctness is action == best_action(regime), separate from physical
survival/consequence.

Out-of-model (OOM) claim: ``brick_under_silent_threat``. Regime ``silent_threat``
encodes like calm (obs 0) so the HMM's obs-class mapping prefers straw, but
brick is optimal — a genuine contradiction of the in-model mapping. By contrast,
``cunning_wolf`` (obs 2 → brick) is an *in-model* hard regime, not OOM.
Hard-coded brick exploration is logged as exploration, never as Seer-discovered
competence; only validated authority overrides count as Seer competence.

Observation is partial/noisy and costly. Genome traits (when an Agent is
supplied) causally affect explore_prob, paid observation access, caution
upgrades, and Seer validation_threshold.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator

from driftbound import __version__
from driftbound.core.events import StepEvent
from driftbound.core.experiment import ExperimentConfig
from driftbound.core.metrics import MetricSummary, summarize_metrics
from driftbound.core.random import make_rng, spawn_rng
from driftbound.smartdream.agent import Agent
from driftbound.smartdream.controller import (
    FixedPolicyController,
    GuardedSeerHybridController,
    HMMChangeController,
    HMMController,
    SwitchingExpertsController,
)
from driftbound.smartdream.lifecycle.birth_control import BirthControl
from driftbound.smartdream.population import Population
from driftbound.smartdream.reward import decompose_reward
from driftbound.smartdream.seer import GuardedSeer, SeerHypothesis
from driftbound.smartdream.vision import VisionChannel, VisionEvidence

MATERIALS = ("straw", "wood", "brick")
MATERIAL_STRENGTH = {"straw": 0.2, "wood": 0.55, "brick": 0.9}
MATERIAL_COST = {"straw": 0.05, "wood": 0.15, "brick": 0.35}
WOLF_STRENGTH = {
    "wolf": 0.4,
    "cunning_wolf": 0.75,
    "calm": 0.1,
    "silent_threat": 0.8,
    "storm": 0.85,
}
OBS_COST = 0.02
INTERVENTION_COST = 0.1
FALLBACK_COST = 0.25
EXPLORATION_COST = 0.03
AUTHORITY_OVERRIDE_COST = 0.05

# Genuine OOM: silent_threat looks like calm (obs 0) but requires brick.
SEER_CLAIM = "brick_under_silent_threat"
OOM_REGIME = "silent_threat"


@dataclass
class TLPConfig:
    n_steps: int = 200
    discovery_end: int = 60
    validation_end: int = 120
    held_out_start: int = 120
    oom_start: int = 160
    noise: float = 0.15
    observe_prob: float = 0.4
    controller: str = "hmm"
    evolution_enabled: bool = False


@dataclass(frozen=True)
class WorldStep:
    """One pre-generated environment step (controller-independent)."""

    regime: str
    observed: bool
    observation: int
    outcome_u: float  # uniform draw for physical success Bernoulli


@dataclass(frozen=True)
class WorldTrace:
    """Full pre-generated schedule + obs/noise so controllers share identical worlds."""

    steps: tuple[WorldStep, ...]

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def regimes(self) -> list[str]:
        return [s.regime for s in self.steps]

    def observations(self) -> list[int]:
        return [s.observation for s in self.steps]

    def as_dict(self) -> dict[str, Any]:
        return {
            "regimes": self.regimes(),
            "observed": [s.observed for s in self.steps],
            "observations": self.observations(),
            "outcome_u": [s.outcome_u for s in self.steps],
        }


def _regime_at(step: int, cfg: TLPConfig, rng: Generator) -> str:
    """Sample regime label.

    Discovery/validation include low-rate ``silent_threat`` so the Seer can
    propose/validate the OOM claim. The OOM segment injects ``silent_threat``
    densely. ``cunning_wolf`` remains an in-model hard regime (obs→brick).
    """
    if step >= cfg.oom_start:
        return OOM_REGIME
    if step >= cfg.held_out_start:
        return str(
            rng.choice(
                ["wolf", "calm", "cunning_wolf", "silent_threat"],
                p=[0.35, 0.25, 0.25, 0.15],
            )
        )
    if step < cfg.discovery_end:
        return str(
            rng.choice(
                ["calm", "wolf", "cunning_wolf", "silent_threat"],
                p=[0.45, 0.35, 0.12, 0.08],
            )
        )
    return str(
        rng.choice(
            ["calm", "wolf", "cunning_wolf", "silent_threat"],
            p=[0.30, 0.30, 0.25, 0.15],
        )
    )


def best_action(regime: str) -> str:
    """Optimal material for a regime (decision-correctness ground truth)."""
    if regime in ("storm", "cunning_wolf", "silent_threat"):
        return "brick"
    if regime == "wolf":
        return "wood"
    return "straw"


# Public alias used by tests
_best_action = best_action


def _outcome_success_from_u(
    action: str, regime: str, u: float, noise: float
) -> bool:
    strength = MATERIAL_STRENGTH[action]
    threat = WOLF_STRENGTH.get(regime, 0.85)  # storm defaults high
    margin = strength - threat
    p = 1.0 / (1.0 + np.exp(-6.0 * margin))
    p = float(np.clip(p * (1.0 - noise) + 0.5 * noise, 0.01, 0.99))
    return bool(u < p)


def _obs_encode(regime: str, rng: Generator, noise: float, observed: bool) -> int:
    """Partial observation code in {0,1,2,3}; 3 = unobserved/noise.

    ``silent_threat`` deliberately aliases to calm (0) so the HMM's peaked
    emission→state→action path prefers straw while brick is optimal — the
    OOM contradiction. ``cunning_wolf``/``storm`` share code 2 (in-model brick).
    """
    if not observed:
        return 3
    base = {
        "calm": 0,
        "wolf": 1,
        "cunning_wolf": 2,
        "storm": 2,
        "silent_threat": 0,  # OOM alias: looks calm, needs brick
    }.get(regime, 3)
    if rng.random() < noise:
        return int(rng.integers(0, 4))
    return base


def generate_world(cfg: TLPConfig, world_rng: Generator) -> WorldTrace:
    """Pre-generate regimes, observe flags, obs codes, and outcome uniforms.

    Uses only ``world_rng`` — never the controller RNG — so every controller
    with the same world seed faces bit-identical environment traces.
    """
    steps: list[WorldStep] = []
    for step in range(cfg.n_steps):
        regime = _regime_at(step, cfg, world_rng)
        observed = bool(world_rng.random() < cfg.observe_prob)
        obs = _obs_encode(regime, world_rng, cfg.noise, observed)
        outcome_u = float(world_rng.random())
        steps.append(
            WorldStep(
                regime=regime,
                observed=observed,
                observation=obs,
                outcome_u=outcome_u,
            )
        )
    return WorldTrace(steps=tuple(steps))


def _make_controller(
    name: str,
    cfg: TLPConfig,
    *,
    explore_prob: float = 0.25,
    caution: float = 0.0,
    validation_threshold: float = 0.5,
) -> Any:
    if name == "fixed_policy":
        return FixedPolicyController(action="wood")
    if name == "hmm":
        return HMMController()
    if name == "hmm_change":
        return HMMChangeController()
    if name == "switching_experts":
        return SwitchingExpertsController()
    if name == "guarded_seer_hybrid":
        seer = GuardedSeer(
            discovery_end=cfg.discovery_end,
            validation_end=cfg.validation_end,
            deployment_end=cfg.n_steps,
            min_evidence=4,
            alpha=0.05,
            authority_horizon=25,
            min_validation_score=float(validation_threshold),
        )
        return GuardedSeerHybridController(
            seer=seer,
            explore_prob=explore_prob,
            caution=float(caution),
        )
    raise ValueError(f"Unknown controller: {name}")


def _claim_checker(hyp: SeerHypothesis, ev: VisionEvidence) -> bool | None:
    """Validation for brick_under_silent_threat (genuine OOM claim).

    Applicable only under ``silent_threat`` context with brick action.
    Ordinary wolf / cunning_wolf evidence is non-applicable (None) so the
    denominator is relevant OOM trials only.
    """
    if hyp.claim != SEER_CLAIM:
        return False
    if not isinstance(ev.outcome, dict):
        return None
    if ev.outcome.get("action") != "brick":
        return None
    if ev.context != OOM_REGIME:
        return None
    return bool(ev.outcome.get("success"))


def run_once(
    rng: Generator,
    *,
    replication: int = 0,
    controller: str = "hmm",
    n_steps: int = 200,
    world: WorldTrace | None = None,
    evolution_enabled: bool = False,
    agent: Agent | None = None,
    **_kwargs: Any,
) -> tuple[MetricSummary, dict[str, Any]]:
    """Run one TLP episode.

    ``rng`` is the *controller* RNG only. World schedules/obs/noise come from
    ``world`` (pre-generated) or are generated from a spawned world stream so
    controller draws never perturb the environment.
    """
    cfg = TLPConfig(
        n_steps=n_steps, controller=controller, evolution_enabled=evolution_enabled
    )
    if world is None:
        world_rng, ctrl_rng = spawn_rng(rng, 2)
        world = generate_world(cfg, world_rng)
    else:
        ctrl_rng = rng
        if world.n_steps != cfg.n_steps:
            cfg = TLPConfig(
                n_steps=world.n_steps,
                controller=controller,
                evolution_enabled=evolution_enabled,
            )

    explore_prob = 0.25
    caution = 0.0
    validation_threshold = 0.5
    observe_prob_trait = 1.0  # default: always take scheduled observations
    if agent is not None:
        ph = agent.phenotype
        explore_prob = float(ph.explore_prob)
        caution = float(ph.intervention_threshold)
        validation_threshold = float(ph.validation_threshold)
        observe_prob_trait = float(ph.observe_prob)

    ctrl = _make_controller(
        controller,
        cfg,
        explore_prob=explore_prob,
        caution=caution,
        validation_threshold=validation_threshold,
    )
    ctrl.reset(ctrl_rng)
    vision = VisionChannel()
    events: list[StepEvent] = []
    seer: GuardedSeer | None = getattr(ctrl, "seer", None)

    detect_oom_step: int | None = None
    recover_step: int | None = None
    post_oom_correct = 0
    post_oom_total = 0

    exploration_actions = 0
    authority_overrides = 0
    promotions_logged = 0
    demotions_logged = 0
    exploration_cost_total = 0.0
    override_cost_total = 0.0
    paid_observations = 0
    skipped_observations = 0
    caution_upgrades = 0
    promotions_before = seer.promotions if seer is not None else 0
    demotions_before = seer.demotions if seer is not None else 0

    for step, wstep in enumerate(world.steps):
        regime = wstep.regime
        # World schedule is shared; observation_sensitivity decides paid access.
        scheduled_observed = wstep.observed
        obs = wstep.observation
        observed = scheduled_observed
        if scheduled_observed and agent is not None:
            if float(ctrl_rng.random()) >= observe_prob_trait:
                observed = False
                obs = 3  # treat as unobserved / no paid access
                skipped_observations += 1
            else:
                paid_observations += 1
        elif scheduled_observed:
            paid_observations += 1
        action = ctrl.act(obs, ctrl_rng)
        if action not in MATERIALS:
            action = "wood"
        success = _outcome_success_from_u(action, regime, wstep.outcome_u, cfg.noise)
        best = best_action(regime)
        decision_correct = action == best
        consequence = MATERIAL_STRENGTH[action] * (1.0 if success else 0.5)

        decision = getattr(ctrl, "last_decision", None)
        explored = bool(decision.explored) if decision is not None else False
        override = bool(decision.authority_override) if decision is not None else False
        demoted_now = bool(decision.demoted) if decision is not None else False

        obs_cost = OBS_COST if observed else 0.0
        intervened = action == "brick"
        int_cost = INTERVENTION_COST if intervened else 0.0
        fallback = (not success) and regime != "calm"
        fb_cost = FALLBACK_COST if fallback else 0.0
        exp_cost = EXPLORATION_COST if explored else 0.0
        ovr_cost = AUTHORITY_OVERRIDE_COST if override else 0.0

        if explored:
            exploration_actions += 1
            exploration_cost_total += exp_cost
        if override:
            authority_overrides += 1
            override_cost_total += ovr_cost
        if hasattr(ctrl, "caution_upgrades"):
            caution_upgrades = int(ctrl.caution_upgrades)

        reward = decompose_reward(
            correct=decision_correct,
            consequence=consequence,
            observation_cost=obs_cost,
            intervention_cost=int_cost + exp_cost + ovr_cost,
            fallback_cost=fb_cost,
            success=success,
        )

        if observed:
            vision.observe(
                observer_id="env",
                subject_id="house",
                context=regime,
                observed_action=action,
                outcome={"action": action, "success": success, "regime": regime},
                independently_observable=True,
                cost=obs_cost,
                step=step,
            )

        promotion_now = False
        if seer is not None:
            phase = seer.phase_at(step)
            if (
                phase.value == "discovery"
                and SEER_CLAIM not in seer.hypotheses
                and action == "brick"
                and success
                and regime == OOM_REGIME
            ):
                # Proposal from exploratory brick success under OOM regime —
                # not yet validated authority / Seer competence.
                seer.propose(SEER_CLAIM, SEER_CLAIM, score=1.0, step=step)
            if step == cfg.validation_end - 1:
                seer.lock_discovery()
                promoted_ids = seer.validate_from_vision(
                    vision, step=step, claim_checker=_claim_checker
                )
                if promoted_ids:
                    promotion_now = True
                    promotions_logged += len(promoted_ids)

        if demoted_now:
            demotions_logged += 1

        ctrl.update(
            obs,
            action,
            {
                "best_action": best,
                "success": success,
                "decision_correct": decision_correct,
            },
            ctrl_rng,
        )

        if agent is not None:
            agent.record_outcome(
                correct=decision_correct,
                consequence=consequence if success else -abs(consequence),
                utility=reward.net,
                step=step,
            )

        events.append(
            StepEvent(
                step=step,
                agent_id=agent.id if agent is not None else "tlp",
                action=action,
                predicted=action,
                outcome=regime,
                correct=decision_correct,
                reward=reward,
                regime=regime,
                observed=observed,
                intervened=intervened,
                fallback=fallback,
                metadata={
                    "success": success,
                    "best_action": best,
                    "explored": explored,
                    "authority_override": override,
                    "promotion": promotion_now,
                    "demotion": demoted_now,
                    "exploration_cost": exp_cost,
                    "authority_override_cost": ovr_cost,
                },
            )
        )

        if step >= cfg.oom_start:
            post_oom_total += 1
            if decision_correct:
                post_oom_correct += 1
            if detect_oom_step is None and action == "brick":
                detect_oom_step = step
            if detect_oom_step is not None and recover_step is None and success:
                recover_step = step

    false_promotions = 0
    promotions = 0
    demotions = 0
    if seer is not None:
        promotions = seer.promotions
        demotions = seer.demotions
        for hid, hyp in seer.hypotheses.items():
            if (
                (hyp.promoted or hyp.phase.value == "deployment")
                and hyp.validation_score < 0.55
                and hyp.discovery_score >= 0.9
            ):
                seer.record_false_promotion(hid)
        false_promotions = seer.false_promotions
        # Count promotions/demotions that occurred during this episode
        promotions_logged = max(promotions_logged, promotions - promotions_before)
        demotions_logged = max(demotions_logged, demotions - demotions_before)

    ttd = (
        float(detect_oom_step - cfg.oom_start) if detect_oom_step is not None else None
    )
    recovery = (
        float(recover_step - detect_oom_step)
        if detect_oom_step is not None and recover_step is not None
        else None
    )

    metrics = summarize_metrics(
        events,
        false_promotions=false_promotions,
        promotions=promotions,
        time_to_detect_oom=ttd,
        recovery_time=recovery,
        calibration_error=None,
    )
    extra: dict[str, Any] = {
        "replication": replication,
        "controller": controller,
        "seer_monitor": seer.monitor() if seer else None,
        "vision_cost": vision.total_cost,
        "n_evidence": len(vision.evidences),
        # Complete shared world trace (regimes, obs flags, encodings, uniforms)
        "world_trace": world.as_dict(),
        "decision_audit": {
            "exploration_actions": exploration_actions,
            "authority_overrides": authority_overrides,
            "promotions": promotions_logged,
            "demotions": demotions_logged,
            "exploration_cost_total": exploration_cost_total,
            "authority_override_cost_total": override_cost_total,
            "paid_observations": paid_observations,
            "skipped_observations": skipped_observations,
            "caution_upgrades": caution_upgrades,
        },
        "genome_traits_applied": {
            "explore_prob": explore_prob,
            "caution": caution,
            "validation_threshold": validation_threshold,
            "observe_prob": observe_prob_trait,
        },
        "seer_claim": SEER_CLAIM,
        "oom_regime": OOM_REGIME,
        "evolution_enabled": evolution_enabled,
        "agent_id": agent.id if agent is not None else None,
        "events_metadata_sample": [
            e.metadata
            for e in events
            if e.metadata.get("explored") or e.metadata.get("authority_override")
        ][:5],
    }
    return metrics, extra


CONTROLLERS = (
    "hmm",
    "hmm_change",
    "switching_experts",
    "guarded_seer_hybrid",
    "fixed_policy",
)


def run_benchmark(
    seed: int = 42,
    replications: int = 3,
    n_steps: int = 200,
    controllers: tuple[str, ...] = CONTROLLERS,
    evolution_enabled: bool = False,
    n_episodes: int = 3,
    population_size: int = 4,
) -> dict[str, Any]:
    """Compare controllers under identical pre-generated worlds.

    When ``evolution_enabled`` is False (default), runs the fair non-evolving
    comparison: one shared WorldTrace per replication across controllers.
    When True, wires Population / BirthControl / Genome between episodes and
    records lineage / generation metrics.
    """
    results: dict[str, Any] = {
        "seed": seed,
        "replications": replications,
        "evolution_enabled": evolution_enabled,
        "controllers": {},
    }

    if evolution_enabled:
        results["evolution"] = _run_evolution(
            seed=seed,
            n_episodes=n_episodes,
            n_steps=n_steps,
            population_size=population_size,
            controller="guarded_seer_hybrid",
        )
        # Still run fair non-evolving controller table for comparison baselines
        # under the same seed (no evolution side effects on that table).

    from driftbound.core.experiment import ExperimentResult
    from driftbound.core.provenance import ProvenanceRecord

    # Pre-generate one WorldTrace per replication (controller-independent seeds).
    worlds: list[WorldTrace] = []
    for rep in range(replications):
        w_rng = make_rng(seed + 10_000 + rep * 97)
        worlds.append(generate_world(TLPConfig(n_steps=n_steps), w_rng))

    for name in controllers:
        per_rep_metrics: list[MetricSummary] = []
        per_rep_extra: list[dict[str, Any]] = []
        name_offset = sum(ord(c) for c in name)
        for rep in range(replications):
            c_rng = make_rng(seed + 50_000 + rep * 91 + name_offset)
            metrics, extra = run_once(
                c_rng,
                replication=rep,
                controller=name,
                n_steps=n_steps,
                world=worlds[rep],
                evolution_enabled=False,
            )
            per_rep_metrics.append(metrics)
            per_rep_extra.append(extra)

        cfg_exp = ExperimentConfig(
            name=f"three_little_pigs:{name}",
            seed=seed,
            replications=replications,
            params={"controller": name, "n_steps": n_steps},
        )
        er = ExperimentResult(
            config=cfg_exp,
            metrics=per_rep_metrics,
            provenance=ProvenanceRecord(
                experiment_name=cfg_exp.name,
                seed=seed,
                config={"replications": replications, **cfg_exp.params},
                package_version=__version__,
            ),
            extras={"per_replication": per_rep_extra},
        )
        results["controllers"][name] = {
            "aggregate": er.aggregate(),
            "provenance": er.provenance.as_dict(),
            "per_replication": [m.as_dict() for m in er.metrics],
            "decision_audit": [e.get("decision_audit") for e in per_rep_extra],
            "world_traces": [e["world_trace"] for e in per_rep_extra],
            # Backward-compatible alias (regimes only)
            "world_regimes": [e["world_trace"]["regimes"] for e in per_rep_extra],
        }

    return results


def _run_evolution(
    *,
    seed: int,
    n_episodes: int,
    n_steps: int,
    population_size: int,
    controller: str,
) -> dict[str, Any]:
    """Between-episode Birth Control on the canonical Population."""
    rng = make_rng(seed)
    pop = Population(max_size=max(population_size * 2, 8))
    bc = BirthControl(population=pop, fitness_threshold=0.0, min_population=2)
    bc.seed_population(population_size, rng)
    episode_logs: list[dict[str, Any]] = []
    offspring_ids: set[str] = set()

    for ep in range(n_episodes):
        world_rng, ctrl_stream = spawn_rng(rng, 2)
        cfg = TLPConfig(n_steps=n_steps, controller=controller, evolution_enabled=True)
        world = generate_world(cfg, world_rng)
        ctrl_rngs = spawn_rng(ctrl_stream, max(len(pop.living()), 1))
        living = list(pop.living())
        ep_metrics: list[dict[str, Any]] = []
        for i, agent in enumerate(living):
            c_rng = ctrl_rngs[i % len(ctrl_rngs)]
            metrics, extra = run_once(
                c_rng,
                replication=ep,
                controller=controller,
                n_steps=n_steps,
                world=world,
                evolution_enabled=True,
                agent=agent,
            )
            ep_metrics.append(
                {
                    "agent_id": agent.id,
                    "generation": agent.generation,
                    "parent_ids": list(agent.parent_ids),
                    "accuracy": metrics.accuracy,
                    "net_reward": metrics.net_reward,
                    "is_offspring": agent.id in offspring_ids or agent.generation > 0,
                }
            )

        summary_before = pop.summary()
        born = bc.replace(rng, deaths=1, births=1)
        for child in born:
            offspring_ids.add(child.id)
        summary_after = pop.summary()
        gens = [a.generation for a in pop.living()]
        episode_logs.append(
            {
                "episode": ep,
                "population_before": summary_before,
                "population_after": summary_after,
                "births_this_episode": len(born),
                "born_ids": [c.id for c in born],
                "born_generations": [c.generation for c in born],
                "born_parent_ids": [list(c.parent_ids) for c in born],
                "living_generations": gens,
                "max_generation": max(gens) if gens else 0,
                "agent_metrics": ep_metrics,
                "lineage": [
                    {
                        "id": a.id,
                        "generation": a.generation,
                        "parent_ids": list(a.parent_ids),
                        "alive": a.alive,
                    }
                    for a in pop.all_agents()
                ],
            }
        )

    return {
        "n_episodes": n_episodes,
        "population_size_initial": population_size,
        "final_summary": pop.summary(),
        "offspring_seen": sorted(offspring_ids),
        "episodes": episode_logs,
        "evolution_side_effects": True,
    }


def _format_table(results: dict[str, Any]) -> str:
    headers = (
        "controller",
        "acc_mean",
        "net_mean",
        "cwl_mean",
        "obs_cost",
        "fpr_mean",
    )
    lines = [" | ".join(headers), "-+-".join("-" * len(h) for h in headers)]
    for name, block in results["controllers"].items():
        agg = block["aggregate"]
        row = [
            f"{name:22s}",
            f"{agg['accuracy_mean']:.3f}",
            f"{agg['net_reward_mean']:.2f}",
            f"{agg['consequence_weighted_loss_mean']:.2f}",
            f"{agg['observation_cost_mean']:.2f}",
            f"{agg['false_promotion_rate_mean']:.3f}",
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Three Little Pigs benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument(
        "--evolution-enabled",
        action="store_true",
        help="Wire Population/BirthControl between episodes",
    )
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("generated-results/three_little_pigs.json"),
    )
    args = parser.parse_args(argv)

    results = run_benchmark(
        seed=args.seed,
        replications=args.replications,
        n_steps=args.n_steps,
        evolution_enabled=args.evolution_enabled,
        n_episodes=args.n_episodes,
        population_size=args.population_size,
    )
    table = _format_table(results)
    print(table)
    if args.evolution_enabled and "evolution" in results:
        evo = results["evolution"]
        print(
            f"\n[evolution] episodes={evo['n_episodes']} "
            f"final_living={evo['final_summary']['living']} "
            f"births={evo['final_summary']['births']} "
            f"deaths={evo['final_summary']['deaths']} "
            f"offspring={len(evo['offspring_seen'])}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
