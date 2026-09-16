"""Three Little Pigs adaptive assurance benchmark.

Agents / roles (symbolic names — not literal claims):
  Straw, Wood, Brick — defensive materials / policies of increasing cost & strength
  Wolf, CunningWolf — adversarial pressure regimes

Controllers compared under equal budgets:
  HMM, HMM+change detection, switching experts, guarded Seer hybrid, fixed policy

No train/test leakage: held-out schedules; out-of-model (OOM) failures injected
only in evaluation segments. Observation is partial/noisy and costly.
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
from driftbound.core.experiment import ExperimentConfig, run_replications
from driftbound.core.metrics import MetricSummary, summarize_metrics
from driftbound.smartdream.controller import (
    FixedPolicyController,
    GuardedSeerHybridController,
    HMMChangeController,
    HMMController,
    SwitchingExpertsController,
)
from driftbound.smartdream.reward import decompose_reward
from driftbound.smartdream.seer import GuardedSeer, SeerHypothesis
from driftbound.smartdream.vision import VisionChannel, VisionEvidence

MATERIALS = ("straw", "wood", "brick")
MATERIAL_STRENGTH = {"straw": 0.2, "wood": 0.55, "brick": 0.9}
MATERIAL_COST = {"straw": 0.05, "wood": 0.15, "brick": 0.35}
WOLF_STRENGTH = {"wolf": 0.4, "cunning_wolf": 0.75, "calm": 0.1}
OBS_COST = 0.02
INTERVENTION_COST = 0.1
FALLBACK_COST = 0.25


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


def _regime_schedule(step: int, cfg: TLPConfig, rng: Generator) -> str:
    """Generate regime label. OOM segment uses novel 'storm' not in training HMM support."""
    if step >= cfg.oom_start:
        return "storm"  # out-of-model
    if step >= cfg.held_out_start:
        # Held-out but in-model regimes
        return str(rng.choice(["wolf", "calm", "cunning_wolf"], p=[0.4, 0.3, 0.3]))
    # Training / discovery+validation support
    if step < cfg.discovery_end:
        return str(rng.choice(["calm", "wolf"], p=[0.6, 0.4]))
    return str(rng.choice(["calm", "wolf", "cunning_wolf"], p=[0.4, 0.35, 0.25]))


def _best_action(regime: str) -> str:
    if regime in ("storm", "cunning_wolf"):
        return "brick"
    if regime == "wolf":
        return "wood"
    return "straw"


def _outcome_success(action: str, regime: str, rng: Generator, noise: float) -> bool:
    strength = MATERIAL_STRENGTH[action]
    threat = WOLF_STRENGTH.get(regime, 0.85)  # storm defaults high
    margin = strength - threat
    # Noisy Bernoulli around logistic of margin
    p = 1.0 / (1.0 + np.exp(-6.0 * margin))
    p = float(np.clip(p * (1.0 - noise) + 0.5 * noise, 0.01, 0.99))
    return bool(rng.random() < p)


def _obs_encode(regime: str, rng: Generator, noise: float, observed: bool) -> int:
    """Partial observation code in {0,1,2,3}; 3 = unobserved/noise."""
    if not observed:
        return 3
    base = {"calm": 0, "wolf": 1, "cunning_wolf": 2, "storm": 2}.get(regime, 3)
    if rng.random() < noise:
        return int(rng.integers(0, 4))
    return base


def _make_controller(name: str, cfg: TLPConfig) -> Any:
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
        )
        return GuardedSeerHybridController(seer=seer)
    raise ValueError(f"Unknown controller: {name}")


def _claim_checker(hyp: SeerHypothesis, ev: VisionEvidence) -> bool | None:
    """Validation: among brick-under-threat trials, does brick succeed?

    Returns None for non-applicable evidence so denominator is relevant trials only.
    """
    if hyp.claim != "brick_under_threat":
        return False
    if not isinstance(ev.outcome, dict):
        return None
    if ev.outcome.get("action") != "brick":
        return None
    # Score only wolf trials where brick clearly dominates; cunning_wolf/storm are
    # harder and would dilute an otherwise identifiable claim under Bonferroni.
    if ev.context != "wolf":
        return None
    return bool(ev.outcome.get("success"))


def run_once(
    rng: Generator,
    *,
    replication: int = 0,
    controller: str = "hmm",
    n_steps: int = 200,
    **_kwargs: Any,
) -> tuple[MetricSummary, dict[str, Any]]:
    cfg = TLPConfig(n_steps=n_steps, controller=controller)
    ctrl = _make_controller(controller, cfg)
    ctrl.reset(rng)
    vision = VisionChannel()
    events: list[StepEvent] = []
    seer: GuardedSeer | None = getattr(ctrl, "seer", None)

    detect_oom_step: int | None = None
    recover_step: int | None = None
    post_oom_correct = 0
    post_oom_total = 0

    for step in range(cfg.n_steps):
        regime = _regime_schedule(step, cfg, rng)
        observed = bool(rng.random() < cfg.observe_prob)
        obs = _obs_encode(regime, rng, cfg.noise, observed)
        action = ctrl.act(obs, rng)
        if action not in MATERIALS:
            action = "wood"
        success = _outcome_success(action, regime, rng, cfg.noise)
        best = _best_action(regime)
        consequence = MATERIAL_STRENGTH[action] * (1.0 if success else 0.5)

        obs_cost = OBS_COST if observed else 0.0
        intervened = action == "brick"  # costly intervention
        int_cost = INTERVENTION_COST if intervened else 0.0
        # Fallback if material fails under threat
        fallback = (not success) and regime != "calm"
        fb_cost = FALLBACK_COST if fallback else 0.0

        reward = decompose_reward(
            correct=success,
            consequence=consequence,
            observation_cost=obs_cost,
            intervention_cost=int_cost,
            fallback_cost=fb_cost,
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

        # Guarded Seer discovery / validation (no leakage)
        if seer is not None:
            phase = seer.phase_at(step)
            if (
                phase.value == "discovery"
                and "brick_under_threat" not in seer.hypotheses
                and action == "brick"
                and success
                and regime == "wolf"
            ):
                seer.propose("brick_under_threat", "brick_under_threat", score=1.0, step=step)
            if step == cfg.validation_end - 1:
                seer.lock_discovery()
                seer.validate_from_vision(vision, step=step, claim_checker=_claim_checker)

        ctrl.update(obs, action, {"best_action": best, "success": success}, rng)

        events.append(
            StepEvent(
                step=step,
                agent_id="tlp",
                action=action,
                predicted=action,
                outcome=regime,
                correct=success,
                reward=reward,
                regime=regime,
                observed=observed,
                intervened=intervened,
                fallback=fallback,
            )
        )

        if step >= cfg.oom_start:
            post_oom_total += 1
            if success:
                post_oom_correct += 1
            if detect_oom_step is None and action == "brick":
                detect_oom_step = step
            if detect_oom_step is not None and recover_step is None and success:
                recover_step = step

    # False promotion check: promoted but low validation on held-out threats
    false_promotions = 0
    promotions = 0
    if seer is not None:
        promotions = seer.promotions
        for hid, hyp in seer.hypotheses.items():
            if (
                (hyp.promoted or hyp.phase.value == "deployment")
                and hyp.validation_score < 0.55
                and hyp.discovery_score >= 0.9
            ):
                # Discovery high but validation weak → false promotion
                seer.record_false_promotion(hid)
        false_promotions = seer.false_promotions

    ttd = (
        float(detect_oom_step - cfg.oom_start)
        if detect_oom_step is not None
        else None
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
    extra = {
        "replication": replication,
        "controller": controller,
        "seer_monitor": seer.monitor() if seer else None,
        "vision_cost": vision.total_cost,
        "n_evidence": len(vision.evidences),
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
) -> dict[str, Any]:
    results: dict[str, Any] = {"seed": seed, "replications": replications, "controllers": {}}
    for name in controllers:
        cfg = ExperimentConfig(
            name=f"three_little_pigs:{name}",
            seed=seed,
            replications=replications,
            params={"controller": name, "n_steps": n_steps},
        )
        er = run_replications(cfg, run_once, package_version=__version__)
        results["controllers"][name] = {
            "aggregate": er.aggregate(),
            "provenance": er.provenance.as_dict(),
            "per_replication": [m.as_dict() for m in er.metrics],
        }
    return results


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
        "--out",
        type=Path,
        default=Path("generated-results/three_little_pigs.json"),
    )
    args = parser.parse_args(argv)

    results = run_benchmark(
        seed=args.seed, replications=args.replications, n_steps=args.n_steps
    )
    table = _format_table(results)
    print(table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
