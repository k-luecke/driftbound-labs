"""Baseline algorithms: discrete HMM, change detection, switching experts.

Labels:
  - discrete HMM: simplified baseline (fixed small state space, online EM-lite)
  - Page-Hinkley change detector: classic baseline
  - Fixed Share switching experts: classic baseline (Herbster & Warmuth)
  - Full Baum-Welch / Bayesian HMM / specialist pools: future work
"""

from driftbound.baselines.change_detection import PageHinkleyDetector
from driftbound.baselines.hmm import DiscreteHMM
from driftbound.baselines.switching_experts import FixedShareExperts

__all__ = ["DiscreteHMM", "PageHinkleyDetector", "FixedShareExperts"]
