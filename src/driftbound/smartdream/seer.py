"""Guarded Seer: discovery / validation / deployment windows with bounded authority.

Promotion happens on *validation* evidence, never on discovery alone.
Multiple-testing correction: Bonferroni (conservative; documented).
Temporary authority with expiry/demotion; no window leakage.
Does NOT replace the controller globally — advisory/temporary only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from driftbound.smartdream.vision import VisionChannel


class SeerPhase(StrEnum):
    DISCOVERY = "discovery"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    DEMOTED = "demoted"
    EXPIRED = "expired"


@dataclass
class SeerHypothesis:
    hypothesis_id: str
    claim: str
    discovery_score: float = 0.0
    validation_score: float = 0.0
    p_value: float = 1.0
    promoted: bool = False
    false_promotion: bool = False
    authority_remaining: int = 0
    phase: SeerPhase = SeerPhase.DISCOVERY


@dataclass
class GuardedSeer:
    """Guarded Seer with separated windows and Bonferroni correction."""

    discovery_end: int
    validation_end: int
    deployment_end: int
    min_evidence: int = 5
    alpha: float = 0.05
    authority_horizon: int = 20
    hypotheses: dict[str, SeerHypothesis] = field(default_factory=dict)
    promotions: int = 0
    false_promotions: int = 0
    _discovery_locked: bool = False
    _validation_used_steps: set[int] = field(default_factory=set)

    def phase_at(self, step: int) -> SeerPhase:
        if step < self.discovery_end:
            return SeerPhase.DISCOVERY
        if step < self.validation_end:
            return SeerPhase.VALIDATION
        if step < self.deployment_end:
            return SeerPhase.DEPLOYMENT
        return SeerPhase.EXPIRED

    def propose(self, hypothesis_id: str, claim: str, score: float, step: int) -> None:
        """Record a discovery-window proposal. No promotion here."""
        if self.phase_at(step) != SeerPhase.DISCOVERY:
            raise ValueError("propose() only allowed in discovery window (no leakage)")
        self.hypotheses[hypothesis_id] = SeerHypothesis(
            hypothesis_id=hypothesis_id,
            claim=claim,
            discovery_score=score,
            phase=SeerPhase.DISCOVERY,
        )

    def lock_discovery(self) -> None:
        self._discovery_locked = True

    def _bonferroni_threshold(self) -> float:
        m = max(len(self.hypotheses), 1)
        return self.alpha / m

    def validate_from_vision(
        self,
        vision: VisionChannel,
        *,
        step: int,
        claim_checker: Any,
    ) -> list[str]:
        """Validate hypotheses using independently observable Vision evidence.

        ``claim_checker(hypothesis, evidence) -> bool`` scores validation hits.
        Uses only validation-window evidence; discovery evidence is ignored.
        """
        if self.phase_at(step) != SeerPhase.VALIDATION:
            raise ValueError("validate_from_vision only in validation window")
        if not self._discovery_locked:
            self.lock_discovery()

        val_evidence = [
            e
            for e in vision.independent_evidence()
            if self.discovery_end <= e.step < self.validation_end
        ]
        # Guard against reusing discovery steps
        for e in val_evidence:
            if e.step < self.discovery_end:
                raise RuntimeError("Window leakage: discovery evidence in validation")
            self._validation_used_steps.add(e.step)

        promoted: list[str] = []
        threshold = self._bonferroni_threshold()
        m = max(len(self.hypotheses), 1)

        for hid, hyp in self.hypotheses.items():
            # claim_checker may return None for non-applicable evidence rows
            applicable: list[Any] = []
            hits = 0
            for e in val_evidence:
                result = claim_checker(hyp, e)
                if result is None:
                    continue
                applicable.append(e)
                if result:
                    hits += 1
            n = len(applicable)
            if n < self.min_evidence:
                hyp.phase = SeerPhase.VALIDATION
                hyp.validation_score = hits / n if n else 0.0
                continue
            # One-sided binomial test under H0: p=0.5 (simplified, honest baseline)
            # Normal approximation; one-sided upper tail (phat > 0.5)
            phat = hits / n
            se = np.sqrt(0.25 / n)
            z = (phat - 0.5) / se if se > 0 else 0.0
            from math import erfc, sqrt

            p_raw = 0.5 * erfc(z / sqrt(2.0)) if z > 0 else 1.0
            hyp.p_value = min(1.0, p_raw * m)  # Bonferroni-adjusted
            hyp.validation_score = phat
            hyp.phase = SeerPhase.VALIDATION
            if hyp.p_value < threshold and hits >= self.min_evidence // 2:
                hyp.promoted = True
                hyp.authority_remaining = self.authority_horizon
                hyp.phase = SeerPhase.DEPLOYMENT
                self.promotions += 1
                promoted.append(hid)
        return promoted

    def advisory_action(
        self,
        step: int,
        default_action: Any,
        seer_action: Any,
    ) -> tuple[Any, bool]:
        """Bounded authority: may override default only while authority remains."""
        if self.phase_at(step) != SeerPhase.DEPLOYMENT:
            return default_action, False
        active = [h for h in self.hypotheses.values() if h.promoted and h.authority_remaining > 0]
        if not active:
            return default_action, False
        # Temporary advisory override
        for h in active:
            h.authority_remaining -= 1
            if h.authority_remaining <= 0:
                h.phase = SeerPhase.DEMOTED
                h.promoted = False
        return seer_action, True

    def record_false_promotion(self, hypothesis_id: str) -> None:
        hyp = self.hypotheses[hypothesis_id]
        hyp.false_promotion = True
        self.false_promotions += 1

    def false_promotion_rate(self) -> float:
        if self.promotions == 0:
            return 0.0
        return self.false_promotions / self.promotions

    def monitor(self) -> dict[str, Any]:
        return {
            "n_hypotheses": len(self.hypotheses),
            "promotions": self.promotions,
            "false_promotions": self.false_promotions,
            "false_promotion_rate": self.false_promotion_rate(),
            "correction": "bonferroni",
            "min_evidence": self.min_evidence,
            "windows": {
                "discovery_end": self.discovery_end,
                "validation_end": self.validation_end,
                "deployment_end": self.deployment_end,
            },
        }
