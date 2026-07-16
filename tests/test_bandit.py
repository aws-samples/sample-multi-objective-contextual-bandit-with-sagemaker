"""Unit tests for the LinUCB bandit and batch scoring helpers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from linucb import LinUCBDisjoint, MultiObjectiveLinUCB  # noqa: E402
from batch_job import score_prospects, update_from_feedback  # noqa: E402
from synthetic_data import SyntheticFunnelEnv  # noqa: E402


def test_shapes_and_init():
    b = LinUCBDisjoint(n_arms=3, n_features=4)
    assert len(b.A) == 3 and len(b.b) == 3
    assert b.A[0].shape == (4, 4)
    # Starts from identity, so scores are finite for any context.
    scores = b.ucb_scores(np.ones(4))
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_update_changes_estimate():
    b = LinUCBDisjoint(n_arms=2, n_features=3, alpha=0.0)
    ctx = np.array([1.0, 0.0, 0.0])
    before = b.ucb_scores(ctx)[0]
    b.update(0, ctx, reward=1.0)
    after = b.ucb_scores(ctx)[0]
    # With alpha=0 the score is pure estimate; a positive reward raises it.
    assert after > before


def test_exploration_bonus_shrinks_with_experience():
    b = LinUCBDisjoint(n_arms=1, n_features=2, alpha=1.0)
    ctx = np.array([1.0, 1.0])
    A_inv0 = np.linalg.inv(b.A[0])
    bonus0 = np.sqrt(ctx @ A_inv0 @ ctx)
    for _ in range(50):
        b.update(0, ctx, reward=0.0)
    A_inv1 = np.linalg.inv(b.A[0])
    bonus1 = np.sqrt(ctx @ A_inv1 @ ctx)
    assert bonus1 < bonus0


def test_multiobjective_weights_validation():
    with pytest.raises(ValueError):
        MultiObjectiveLinUCB(n_arms=2, n_features=2, weights=(0.5, 0.5))


def test_multiobjective_select_matches_combined_argmax():
    m = MultiObjectiveLinUCB(n_arms=5, n_features=4, weights=(0.33, 0.33, 0.34))
    ctx = np.array([0.5, -0.2, 1.0, 0.1])
    assert m.select_arm(ctx) == int(np.argmax(m.combined_scores(ctx)))


def test_score_prospects_matches_serial_select():
    m = MultiObjectiveLinUCB(n_arms=6, n_features=5)
    rng = np.random.default_rng(0)
    # Train a little so arms differ.
    for _ in range(100):
        ctx = rng.normal(0, 1, 5)
        arm = m.select_arm(ctx)
        m.update(arm, ctx, rng.integers(0, 2), rng.integers(0, 2), rng.integers(0, 2))

    contexts = rng.normal(0, 1, (30, 5))
    parallel = score_prospects(m, contexts, n_workers=4)
    serial = np.array([m.select_arm(x) for x in contexts])
    assert np.array_equal(parallel, serial)


def test_update_from_feedback_runs():
    m = MultiObjectiveLinUCB(n_arms=4, n_features=3)
    feedback = {
        "arm": np.array([0, 1, 2, 3]),
        "context": np.random.default_rng(1).normal(0, 1, (4, 3)),
        "reward_start": np.array([1, 0, 1, 0]),
        "reward_submit": np.array([1, 0, 0, 0]),
        "reward_approve": np.array([0, 0, 0, 0]),
    }
    update_from_feedback(m, feedback)
    # Arm 0 saw a start+submit, so its start-bandit b vector is non-zero.
    assert np.linalg.norm(m.app_start_bandit.b[0]) > 0


def test_get_state_from_state_roundtrip():
    """State survives a save/load round-trip and reproduces the same choices."""
    m = MultiObjectiveLinUCB(n_arms=5, n_features=4, alpha=0.7, weights=(0.3, 0.3, 0.4))
    rng = np.random.default_rng(11)
    for _ in range(80):
        ctx = rng.normal(0, 1, 4)
        arm = m.select_arm(ctx)
        m.update(arm, ctx, rng.integers(0, 2), rng.integers(0, 2), rng.integers(0, 2))

    restored = MultiObjectiveLinUCB.from_state(m.get_state())
    assert restored.n_arms == m.n_arms
    assert restored.weights == m.weights
    for ctx in rng.normal(0, 1, (20, 4)):
        assert restored.select_arm(ctx) == m.select_arm(ctx)


def test_bandit_beats_random_on_synthetic_funnel():
    """End-to-end learning check on synthetic data."""
    env = SyntheticFunnelEnv(n_arms=15, n_features=6, seed=3)
    weights = (0.33, 0.33, 0.34)
    m = MultiObjectiveLinUCB(15, 6, alpha=1.0, weights=weights)
    rng = np.random.default_rng(3)

    def reward(o):
        return weights[0] * o[0] + weights[1] * o[1] + weights[2] * o[2]

    bandit_r, random_r = [], []
    for _ in range(3000):
        ctx = env.sample_context()
        arm = m.select_arm(ctx)
        o = env.step(arm, ctx)
        m.update(arm, ctx, *o)
        bandit_r.append(reward(o))
        random_r.append(reward(env.step(int(rng.integers(0, 15)), ctx)))

    late = slice(-750, None)
    assert np.mean(bandit_r[late]) > np.mean(random_r[late])
