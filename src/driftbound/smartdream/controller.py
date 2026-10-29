"""Controller wrappers combining baselines and optional Guarded Seer advice."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.random import Generator

from driftbound.baselines.change_detection import PageHinkleyDetector
from driftbound.baselines.hmm import DiscreteHMM
from driftbound.baselines.switching_experts import FixedShareExperts
from driftbound.smartdream.seer import GuardedSeer


@dataclass
class DecisionInfo:
    """Per-step decision audit trail (exploration vs authority vs base)."""

    action: str
    explored: bool = False
    authority_override: bool = False
    demoted: bool = False
    base_action: str | None = None


@dataclass
class FixedPolicyController:
    """Fixed action policy (baseline)."""

    name: str = "fixed_policy"
    action: str = "wood"
    last_decision: DecisionInfo | None = None

    def reset(self, rng: Generator) -> None:
        del rng
        self.last_decision = None

    def act(self, observation: Any, rng: Generator) -> str:
        del observation, rng
        self.last_decision = DecisionInfo(action=self.action)
        return self.action

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        del observation, action, outcome, rng
        return None


@dataclass
class HMMController:
    name: str = "hmm"
    hmm: DiscreteHMM = field(default_factory=DiscreteHMM)
    actions: tuple[str, ...] = ("straw", "wood", "brick")
    last_decision: DecisionInfo | None = None

    def reset(self, rng: Generator) -> None:
        self.hmm.reset(rng)
        self.last_decision = None

    def act(self, observation: Any, rng: Generator) -> str:
        del rng
        # Filter on *current* observation exactly once; action uses filtered MAP.
        state = self.hmm.predict_state(observation)
        action = self.actions[int(state) % len(self.actions)]
        self.last_decision = DecisionInfo(action=action, base_action=action)
        return action

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        del action, outcome
        self.hmm.update(observation, rng)


@dataclass
class HMMChangeController:
    name: str = "hmm_change"
    hmm: DiscreteHMM = field(default_factory=DiscreteHMM)
    detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    actions: tuple[str, ...] = ("straw", "wood", "brick")
    _last_change: int = -1
    last_decision: DecisionInfo | None = None

    def reset(self, rng: Generator) -> None:
        self.hmm.reset(rng)
        self.detector.reset()
        self._last_change = -1
        self.last_decision = None

    def act(self, observation: Any, rng: Generator) -> str:
        del rng
        state = self.hmm.predict_state(observation)
        action = (
            "brick"
            if self.detector.changed
            else self.actions[int(state) % len(self.actions)]
        )
        self.last_decision = DecisionInfo(action=action, base_action=action)
        return action

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        del action
        # Correctness proxy from bool or TLP outcome dict (dict is always truthy).
        if isinstance(outcome, dict):
            score = 1.0 if outcome.get("success") else 0.0
        else:
            score = 1.0 if outcome else 0.0
        self.detector.update(score)
        if self.detector.changed:
            self.hmm.reset(rng)
            self.detector.reset()
            # After reset, filter current obs so next belief is not empty prior lag
            self.hmm.filter(observation)
            self.hmm._filtered_obs = None
        else:
            self.hmm.update(observation, rng)


@dataclass
class SwitchingExpertsController:
    name: str = "switching_experts"
    experts: FixedShareExperts = field(default_factory=FixedShareExperts)
    last_decision: DecisionInfo | None = None

    def reset(self, rng: Generator) -> None:
        self.experts.reset(rng)
        self.last_decision = None

    def act(self, observation: Any, rng: Generator) -> str:
        action = self.experts.act(observation, rng)
        self.last_decision = DecisionInfo(action=action, base_action=action)
        return action

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        self.experts.update(observation, action, outcome, rng)


@dataclass
class GuardedSeerHybridController:
    """HMM base with temporary Guarded Seer advisory overrides.

    ``explore_prob`` (genome exploration_tendency) controls hard-coded brick
    exploration in discovery/validation — logged as ``explored``, never as
    Seer-discovered competence. Promoted authority overrides are separate.
    ``caution`` (genome intervention_threshold) causally upgrades straw→wood
    when neither exploring nor under Seer authority.
    """

    seer: GuardedSeer
    name: str = "guarded_seer_hybrid"
    base: HMMController = field(default_factory=HMMController)
    seer_preferred_action: str = "brick"
    explore_prob: float = 0.25
    caution: float = 0.0
    _step: int = 0
    last_decision: DecisionInfo | None = None
    exploration_count: int = 0
    authority_override_count: int = 0
    demotion_count: int = 0
    caution_upgrades: int = 0

    def reset(self, rng: Generator) -> None:
        self.base.reset(rng)
        self._step = 0
        self.last_decision = None
        self.exploration_count = 0
        self.authority_override_count = 0
        self.demotion_count = 0
        self.caution_upgrades = 0

    def act(self, observation: Any, rng: Generator) -> str:
        base_action = self.base.act(observation, rng)
        phase = self.seer.phase_at(self._step).value
        explored = False
        demoted = False
        # Hard-coded brick exploration (not Seer competence). Enables Vision to
        # gather evidence for the OOM claim; logged distinctly as explored.
        if phase in ("discovery", "validation") and float(rng.random()) < self.explore_prob:
            explored = True
            self.exploration_count += 1
            action = self.seer_preferred_action
            override = False
        else:
            demotions_before = self.seer.demotions
            action, override = self.seer.advisory_action(
                self._step, base_action, self.seer_preferred_action
            )
            if override:
                self.authority_override_count += 1
            if self.seer.demotions > demotions_before:
                demoted = True
                self.demotion_count += 1
            # Caution trait: upgrade straw→wood when not exploring / overriding
            if (
                not override
                and str(action) == "straw"
                and self.caution > 0.0
                and float(rng.random()) < self.caution
            ):
                action = "wood"
                self.caution_upgrades += 1
        self.last_decision = DecisionInfo(
            action=str(action),
            explored=explored,
            authority_override=override if not explored else False,
            demoted=demoted,
            base_action=base_action,
        )
        return str(action)

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        self.base.update(observation, action, outcome, rng)
        self._step += 1
