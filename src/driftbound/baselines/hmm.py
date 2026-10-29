"""Discrete HMM baseline (simplified online variant).

Honest label: this is a *simplified* discrete HMM with fixed transition prior
and online emission counting — not a full Baum-Welch implementation.
Full unsupervised learning of transitions is future work.

Action selection must filter on the *current* observation exactly once per step
(predict-then-update with that obs), not lag-only MAP of the previous belief.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass
class DiscreteHMM:
    """Small-state discrete HMM for regime tracking."""

    n_states: int = 3
    n_obs: int = 4
    transition: np.ndarray | None = None
    emission: np.ndarray | None = None
    belief: np.ndarray | None = None
    # Tracks whether the current step's observation was already filtered (act path).
    _filtered_obs: int | None = None

    def reset(self, rng: Generator) -> None:
        # Mildly sticky transitions
        t = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(t, 0.9)
        self.transition = t
        self.emission = np.full((self.n_states, self.n_obs), 1.0 / self.n_obs)
        # Slightly peaked emissions per state
        for s in range(self.n_states):
            self.emission[s, s % self.n_obs] = 0.55
            self.emission[s] /= self.emission[s].sum()
        self.belief = np.ones(self.n_states) / self.n_states
        self._filtered_obs = None

    def filter(self, observation: int | float) -> int:
        """Predict-update filter with the current observation; return MAP state.

        Calling this advances belief exactly once for ``observation``. Controllers
        should use this (via ``predict_state``) at act time so action selection
        is not lag-only.
        """
        assert self.transition is not None and self.emission is not None
        assert self.belief is not None
        obs = int(observation) % self.n_obs
        prior = self.transition.T @ self.belief
        likelihood = self.emission[:, obs]
        post = prior * likelihood
        s = post.sum()
        self.belief = post / s if s > 0 else np.ones(self.n_states) / self.n_states
        state = int(np.argmax(self.belief))
        # Lightweight online emission adaptation tied to this filter step
        self.emission[state] *= 0.95
        self.emission[state, obs] += 0.05
        self.emission[state] /= self.emission[state].sum()
        self._filtered_obs = obs
        return state

    def map_state(self) -> int:
        """MAP of current belief without incorporating a new observation."""
        assert self.belief is not None
        return int(np.argmax(self.belief))

    def predict_state(self, observation: int | float) -> int:
        """Filter on ``observation`` once and return MAP (current-obs action path)."""
        return self.filter(observation)

    def update(self, observation: int | float, rng: Generator) -> None:
        """Post-action hook: filter only if act did not already filter this obs.

        Ensures exactly one filter per step when used as act(obs) → update(obs).
        """
        del rng  # reserved for future stochastic adaptation
        obs = int(observation) % self.n_obs
        if self._filtered_obs != obs:
            self.filter(observation)
        self._filtered_obs = None
