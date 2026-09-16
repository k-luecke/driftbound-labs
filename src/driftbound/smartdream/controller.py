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
class FixedPolicyController:
    """Fixed action policy (baseline)."""

    name: str = "fixed_policy"
    action: str = "wood"

    def reset(self, rng: Generator) -> None:
        return None

    def act(self, observation: Any, rng: Generator) -> str:
        return self.action

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        return None


@dataclass
class HMMController:
    name: str = "hmm"
    hmm: DiscreteHMM = field(default_factory=DiscreteHMM)
    actions: tuple[str, ...] = ("straw", "wood", "brick")

    def reset(self, rng: Generator) -> None:
        self.hmm.reset(rng)

    def act(self, observation: Any, rng: Generator) -> str:
        state = self.hmm.predict_state(observation)
        # Map latent state index to defensive action
        return self.actions[int(state) % len(self.actions)]

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        self.hmm.update(observation, rng)


@dataclass
class HMMChangeController:
    name: str = "hmm_change"
    hmm: DiscreteHMM = field(default_factory=DiscreteHMM)
    detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    actions: tuple[str, ...] = ("straw", "wood", "brick")
    _last_change: int = -1

    def reset(self, rng: Generator) -> None:
        self.hmm.reset(rng)
        self.detector.reset()
        self._last_change = -1

    def act(self, observation: Any, rng: Generator) -> str:
        state = self.hmm.predict_state(observation)
        if self.detector.changed:
            # On detected change, prefer more cautious action
            return "brick"
        return self.actions[int(state) % len(self.actions)]

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        # Correctness proxy from bool or TLP outcome dict (dict is always truthy).
        if isinstance(outcome, dict):
            score = 1.0 if outcome.get("success") else 0.0
        else:
            score = 1.0 if outcome else 0.0
        self.detector.update(score)
        if self.detector.changed:
            self.hmm.reset(rng)
            self.detector.reset()
        self.hmm.update(observation, rng)


@dataclass
class SwitchingExpertsController:
    name: str = "switching_experts"
    experts: FixedShareExperts = field(default_factory=FixedShareExperts)

    def reset(self, rng: Generator) -> None:
        self.experts.reset(rng)

    def act(self, observation: Any, rng: Generator) -> str:
        return self.experts.act(observation, rng)

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        self.experts.update(observation, action, outcome, rng)


@dataclass
class GuardedSeerHybridController:
    """HMM base with temporary Guarded Seer advisory overrides."""

    seer: GuardedSeer
    name: str = "guarded_seer_hybrid"
    base: HMMController = field(default_factory=HMMController)
    seer_preferred_action: str = "brick"
    _step: int = 0

    def reset(self, rng: Generator) -> None:
        self.base.reset(rng)
        self._step = 0

    def act(self, observation: Any, rng: Generator) -> str:
        base_action = self.base.act(observation, rng)
        # Explore preferred action in discovery/validation so Vision can both
        # propose and independently validate (otherwise promotion never fires).
        phase = self.seer.phase_at(self._step).value
        if phase in ("discovery", "validation") and float(rng.random()) < 0.25:
            return self.seer_preferred_action
        action, _used = self.seer.advisory_action(
            self._step, base_action, self.seer_preferred_action
        )
        return str(action)

    def update(self, observation: Any, action: Any, outcome: Any, rng: Generator) -> None:
        self.base.update(observation, action, outcome, rng)
        self._step += 1
