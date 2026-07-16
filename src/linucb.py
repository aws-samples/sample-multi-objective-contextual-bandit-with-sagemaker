"""Multi-objective disjoint LinUCB contextual bandit.

This module implements the core algorithm described in the AWS blog post
"Driving double-digit conversion lift with contextual bandits on AWS".

It contains two classes:

* ``LinUCBDisjoint`` - a standard disjoint LinUCB bandit (one independent
  linear model per arm), based on Li et al. (2010),
  "A Contextual-Bandit Approach to Personalized News Article Recommendation".
* ``MultiObjectiveLinUCB`` - composes one ``LinUCBDisjoint`` per funnel stage
  (start, submit, approve) and selects an arm using a weighted sum of the
  per-stage UCB scores. This is the "optimize the whole funnel" extension.

The implementation uses only NumPy so it can be read, run, and tested without
any AWS dependencies. See ``examples/demo.py`` for a runnable walkthrough.
"""

from __future__ import annotations

import numpy as np


class LinUCBDisjoint:
    """LinUCB with disjoint (independent) parameters per arm.

    Each arm ``a`` maintains two running tallies that are updated on every
    impression:

    * ``A[a]`` - the "experience ledger": ``A += x xᵀ`` (starts at identity).
    * ``b[a]`` - the "reward ledger":     ``b += reward · x``.

    The per-arm reward estimate is ``theta = A⁻¹ b`` and the score for a given
    context ``x`` is::

        score(a, x) = thetaᵀ x  +  alpha · sqrt(xᵀ A⁻¹ x)
                      \\_______/    \\__________________/
                       estimate         uncertainty bonus
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha  # exploration strength
        # One (d x d) matrix and one d-vector per arm.
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def ucb_scores(self, context: np.ndarray) -> np.ndarray:
        """Return the UCB score for every arm given a context vector."""
        context = np.asarray(context, dtype=float)
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ context + self.alpha * np.sqrt(
                context @ A_inv @ context
            )
        return scores

    def select_arm(self, context: np.ndarray) -> int:
        """Select the highest-scoring arm for this context."""
        return int(np.argmax(self.ucb_scores(context)))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Exact, incremental update for the chosen arm."""
        context = np.asarray(context, dtype=float)
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context


class MultiObjectiveLinUCB:
    """Three disjoint bandits combined with objective weights.

    One bandit per funnel stage (start, submit, approve). The final arm
    selection is the argmax of a weighted sum of the three per-stage UCB
    scores. Weights encode the business decision of how much each funnel
    stage matters and should sum to 1.
    """

    STAGES = ("start", "submit", "approve")

    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        if len(weights) != 3:
            raise ValueError("weights must have exactly 3 entries "
                             "(start, submit, approve)")
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.weights = weights  # [start, submit, approve]
        self.app_start_bandit = LinUCBDisjoint(n_arms, n_features, alpha)
        self.app_submit_bandit = LinUCBDisjoint(n_arms, n_features, alpha)
        self.app_approved_bandit = LinUCBDisjoint(n_arms, n_features, alpha)

    @property
    def _bandits(self) -> list[LinUCBDisjoint]:
        return [
            self.app_start_bandit,
            self.app_submit_bandit,
            self.app_approved_bandit,
        ]

    def combined_scores(self, context: np.ndarray) -> np.ndarray:
        """Weighted-sum UCB score across all three objectives, per arm."""
        context = np.asarray(context, dtype=float)
        combined = np.zeros(self.n_arms)
        for w, bandit in zip(self.weights, self._bandits):
            combined += w * bandit.ucb_scores(context)
        return combined

    def select_arm(self, context: np.ndarray) -> int:
        """Select the arm that maximizes the weighted funnel objective."""
        return int(np.argmax(self.combined_scores(context)))

    def update(
        self,
        arm: int,
        context: np.ndarray,
        r_start: float,
        r_submit: float,
        r_approve: float,
    ) -> None:
        """Update all three stage bandits from one observed outcome."""
        self.app_start_bandit.update(arm, context, r_start)
        self.app_submit_bandit.update(arm, context, r_submit)
        self.app_approved_bandit.update(arm, context, r_approve)

    # --- Serialization ------------------------------------------------------

    def get_state(self) -> dict[str, np.ndarray]:
        """Return model state as a flat dict of NumPy arrays."""
        state = {
            "n_arms": np.array(self.n_arms),
            "n_features": np.array(self.n_features),
            "alpha": np.array(self.alpha, dtype=float),
            "weights": np.asarray(self.weights, dtype=float),
        }
        for name, bandit in zip(self.STAGES, self._bandits):
            state[f"{name}_A"] = np.stack(bandit.A)
            state[f"{name}_b"] = np.stack(bandit.b)
        return state

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray]) -> "MultiObjectiveLinUCB":
        """Rebuild a model from the dict produced by ``get_state``."""
        model = cls(
            n_arms=int(state["n_arms"]),
            n_features=int(state["n_features"]),
            alpha=float(state["alpha"]),
            weights=tuple(float(w) for w in state["weights"]),
        )
        for name, bandit in zip(cls.STAGES, model._bandits):
            A = np.asarray(state[f"{name}_A"], dtype=float)
            b = np.asarray(state[f"{name}_b"], dtype=float)
            bandit.A = [A[i].copy() for i in range(A.shape[0])]
            bandit.b = [b[i].copy() for i in range(b.shape[0])]
        return model
