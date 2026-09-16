"""Discrete HMM baseline (simplified online variant).

Honest label: this is a *simplified* discrete HMM with fixed transition prior
and online emission counting — not a full Baum-Welch implementation.
Full unsupervised learning of transitions is future work.
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

    def predict_state(self, observation: int | float) -> int:
        assert self.belief is not None
        return int(np.argmax(self.belief))

    def update(self, observation: int | float, rng: Generator) -> None:
        assert self.transition is not None and self.emission is not None
        assert self.belief is not None
        obs = int(observation) % self.n_obs
        # Predict
        prior = self.transition.T @ self.belief
        likelihood = self.emission[:, obs]
        post = prior * likelihood
        s = post.sum()
        self.belief = post / s if s > 0 else np.ones(self.n_states) / self.n_states
        # Lightweight online emission adaptation
        state = int(np.argmax(self.belief))
        self.emission[state] *= 0.95
        self.emission[state, obs] += 0.05
        self.emission[state] /= self.emission[state].sum()
