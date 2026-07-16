"""Synthetic data generator for the LinUCB bandit demo and tests.

This is entirely made-up data - no real customers, features, or content. It
exists only so the sample is runnable end to end without any AWS resources or
proprietary data.

Model of the world:
    * Each entity (prospect) has a random context vector of behavioral signals.
    * Each arm (content variation) has hidden per-stage weight vectors.
    * The probability that an entity starts / submits / gets approved on a given
      arm is a logistic function of (context . hidden_weights), with the funnel
      constraint that submit implies start, and approve implies submit.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


class SyntheticFunnelEnv:
    """A synthetic multi-stage funnel environment with a known best arm."""

    def __init__(self, n_arms: int, n_features: int, seed: int = 0):
        self.n_arms = n_arms
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)
        # Hidden per-arm, per-stage weight vectors the bandit must learn.
        scale = 1.0 / np.sqrt(n_features)
        self.w_start = self.rng.normal(0, scale, (n_arms, n_features))
        self.w_submit = self.rng.normal(0, scale, (n_arms, n_features))
        self.w_approve = self.rng.normal(0, scale, (n_arms, n_features))
        # Bias so base rates are realistic (low, like a marketing funnel).
        self.bias = np.array([-1.5, -2.5, -3.5])

    def sample_context(self) -> np.ndarray:
        """Draw one entity context vector."""
        return self.rng.normal(0, 1, self.n_features)

    def stage_probs(self, arm: int, context: np.ndarray) -> tuple[float, float, float]:
        """Return (p_start, p_submit, p_approve) for an arm and context."""
        p_start = float(_sigmoid(self.w_start[arm] @ context + self.bias[0]))
        p_submit = float(_sigmoid(self.w_submit[arm] @ context + self.bias[1]))
        p_approve = float(_sigmoid(self.w_approve[arm] @ context + self.bias[2]))
        return p_start, p_submit, p_approve

    def step(self, arm: int, context: np.ndarray) -> tuple[int, int, int]:
        """Sample funnel outcomes with monotonic constraint start>=submit>=approve."""
        p_start, p_submit, p_approve = self.stage_probs(arm, context)
        start = int(self.rng.random() < p_start)
        submit = int(start and self.rng.random() < p_submit)
        approve = int(submit and self.rng.random() < p_approve)
        return start, submit, approve

    def best_arm(self, context: np.ndarray, weights) -> int:
        """Oracle: the truly best arm for a context under the objective weights."""
        scores = []
        for a in range(self.n_arms):
            p = self.stage_probs(a, context)
            scores.append(weights[0] * p[0] + weights[1] * p[1] + weights[2] * p[2])
        return int(np.argmax(scores))


def make_batch_file(
    path: str,
    env: SyntheticFunnelEnv,
    n_feedback: int = 500,
    n_inference: int = 200,
    seed: int = 123,
) -> None:
    """Write a batch.npz compatible with src/batch_job.py.

    Feedback rows have a (random) arm and observed outcomes; inference rows are
    new entities that need a recommendation.
    """
    rng = np.random.default_rng(seed)

    fb_arm = rng.integers(0, env.n_arms, n_feedback)
    fb_context = np.stack([env.sample_context() for _ in range(n_feedback)])
    r_start, r_submit, r_approve = [], [], []
    for i in range(n_feedback):
        s, su, ap = env.step(int(fb_arm[i]), fb_context[i])
        r_start.append(s)
        r_submit.append(su)
        r_approve.append(ap)

    inf_entity_id = np.array([f"entity-{i:05d}" for i in range(n_inference)])
    inf_context = np.stack([env.sample_context() for _ in range(n_inference)])

    np.savez(
        path,
        fb_arm=fb_arm,
        fb_context=fb_context,
        fb_reward_start=np.array(r_start),
        fb_reward_submit=np.array(r_submit),
        fb_reward_approve=np.array(r_approve),
        inf_entity_id=inf_entity_id,
        inf_context=inf_context,
    )
