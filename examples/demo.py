"""Runnable demo: multi-objective LinUCB learning on a synthetic funnel.

Run locally (no AWS needed):

    python examples/demo.py

The demo streams synthetic entities through the bandit, updating it online, and
compares its funnel reward against a random-arm baseline. Because the bandit
learns which arms fit which contexts, its average reward should climb above
random over time.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linucb import MultiObjectiveLinUCB  # noqa: E402
from synthetic_data import SyntheticFunnelEnv  # noqa: E402

N_ARMS = 20
N_FEATURES = 8
N_ROUNDS = 4000
WEIGHTS = (0.33, 0.33, 0.34)  # start, submit, approve
ALPHA = 1.0


def reward(outcome, weights):
    start, submit, approve = outcome
    return weights[0] * start + weights[1] * submit + weights[2] * approve


def main() -> None:
    env = SyntheticFunnelEnv(N_ARMS, N_FEATURES, seed=7)
    model = MultiObjectiveLinUCB(N_ARMS, N_FEATURES, alpha=ALPHA, weights=WEIGHTS)
    rng = np.random.default_rng(7)

    bandit_rewards, random_rewards, oracle_hits = [], [], []

    for t in range(N_ROUNDS):
        context = env.sample_context()

        # Bandit choice (learns online).
        arm = model.select_arm(context)
        outcome = env.step(arm, context)
        model.update(arm, context, *outcome)
        bandit_rewards.append(reward(outcome, WEIGHTS))
        oracle_hits.append(int(arm == env.best_arm(context, WEIGHTS)))

        # Random baseline (for comparison only; does not learn).
        rnd_arm = int(rng.integers(0, N_ARMS))
        random_rewards.append(reward(env.step(rnd_arm, context), WEIGHTS))

    def window_mean(x, frac=0.25):
        k = int(len(x) * frac)
        return np.mean(x[:k]), np.mean(x[-k:])

    b_early, b_late = window_mean(bandit_rewards)
    r_early, r_late = window_mean(random_rewards)
    hit_early, hit_late = window_mean(oracle_hits)

    print("Multi-objective LinUCB demo")
    print(f"  arms={N_ARMS} features={N_FEATURES} rounds={N_ROUNDS} alpha={ALPHA}")
    print(f"  weights (start, submit, approve) = {WEIGHTS}")
    print()
    print(f"  Avg reward  first 25% -> last 25%")
    print(f"    bandit :  {b_early:.4f} -> {b_late:.4f}")
    print(f"    random :  {r_early:.4f} -> {r_late:.4f}")
    print(f"  Oracle best-arm hit rate:  {hit_early:.2%} -> {hit_late:.2%}")
    print()
    lift = (b_late - r_late) / r_late * 100 if r_late else float("nan")
    print(f"  Late-window bandit vs random reward lift: {lift:+.1f}%")

    if b_late > r_late and b_late > b_early:
        print("\n  PASS: the bandit learned and beats the random baseline.")
    else:
        print("\n  NOTE: bandit did not clearly beat random - try more rounds.")


if __name__ == "__main__":
    main()
