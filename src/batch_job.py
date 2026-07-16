"""Batch entry point run inside an Amazon SageMaker AI Processing job.

This is the "model update and inference" step of the batch architecture:

    1. Load the most recent model state from Amazon S3 (or cold-start).
    2. Split the input data into *feedback* rows (impressions with observed
       funnel outcomes) and *inference* rows (new prospects needing an arm).
    3. Update the bandit on the feedback rows (exact, incremental updates).
    4. Score every prospect and select an arm (parallelized).
    5. Persist the updated model state back to S3 under a new dated prefix,
       and write per-customer recommendations for the serving layer.

Why a Processing job (and not a Training job or Batch Transform)?
    The workload is a single self-contained batch step that BOTH updates the
    model and scores prospects, with custom read/write to S3. A Training job
    is oriented toward emitting a model artifact; Batch Transform toward
    inference against an already-deployed model. A Processing job gives a
    general-purpose, container-managed environment for arbitrary Python with
    straightforward S3 I/O, which fits this update-and-score step cleanly.

When run inside SageMaker AI Processing, input data is mounted under
``/opt/ml/processing/input`` and outputs written to
``/opt/ml/processing/output`` are copied back to S3 by the platform. See
``sagemaker/run_processing_job.py`` for how those channels are wired.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np

from linucb import MultiObjectiveLinUCB
from s3_model_registry import load_latest_model_state, save_model_state

# Default SageMaker AI Processing container paths.
INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_DIR = "/opt/ml/processing/output"


def _precompute_inverses(model: MultiObjectiveLinUCB):
    """Precompute per-arm inverse matrices once per batch.

    Arm matrices do not change during a scoring batch, so inverting them once
    up front (rather than once per customer) is the key inference-time
    optimization. Returns a list of (theta, A_inv) pairs per bandit.
    """
    cached = []
    for bandit in model._bandits:
        per_arm = []
        for a in range(bandit.n_arms):
            A_inv = np.linalg.inv(bandit.A[a])
            theta = A_inv @ bandit.b[a]
            per_arm.append((theta, A_inv))
        cached.append(per_arm)
    return cached


def _score_chunk(contexts: np.ndarray, weights, alpha: float, cached) -> np.ndarray:
    """Score a chunk of contexts and return the selected arm per row."""
    n_arms = len(cached[0])
    selected = np.empty(len(contexts), dtype=int)
    for i, x in enumerate(contexts):
        combined = np.zeros(n_arms)
        for w, per_arm in zip(weights, cached):
            for a, (theta, A_inv) in enumerate(per_arm):
                combined[a] += w * (
                    theta @ x + alpha * np.sqrt(x @ A_inv @ x)
                )
        selected[i] = int(np.argmax(combined))
    return selected


def score_prospects(
    model: MultiObjectiveLinUCB,
    contexts: np.ndarray,
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """Select an arm for every prospect, in parallel.

    Two optimizations keep large-population scoring within budget:
    precomputing all arm matrix inversions once, and splitting the prospect
    set into chunks scored in parallel across the vCPUs allocated to the job.
    """
    cached = _precompute_inverses(model)
    n_workers = n_workers or os.cpu_count() or 1
    if n_workers <= 1 or len(contexts) < n_workers:
        return _score_chunk(contexts, model.weights, model.alpha, cached)

    chunks = np.array_split(contexts, n_workers)
    worker = partial(
        _score_chunk, weights=model.weights, alpha=model.alpha, cached=cached
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, chunks))
    return np.concatenate(results)


def update_from_feedback(model: MultiObjectiveLinUCB, feedback: dict) -> None:
    """Apply exact incremental updates from observed funnel outcomes."""
    arms = feedback["arm"]
    contexts = feedback["context"]
    r_start = feedback["reward_start"]
    r_submit = feedback["reward_submit"]
    r_approve = feedback["reward_approve"]
    for i in range(len(arms)):
        model.update(
            int(arms[i]),
            contexts[i],
            float(r_start[i]),
            float(r_submit[i]),
            float(r_approve[i]),
        )


def load_batch(input_dir: str) -> dict:
    """Load the batch dataset written by the upstream data-extraction step.

    Expects a single ``batch.npz`` with feedback and inference arrays. Replace
    with your own reader (Parquet/CSV from the data warehouse) as needed.
    """
    path = os.path.join(input_dir, "batch.npz")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket for model state")
    parser.add_argument("--model-prefix", default="personalizer/models/")
    parser.add_argument("--n-arms", type=int, required=True)
    parser.add_argument("--n-features", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--weights",
        default="0.33,0.33,0.34",
        help="Comma-separated start,submit,approve weights.",
    )
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(","))

    # 1. Load latest model state, or cold-start a fresh model.
    model = load_latest_model_state(args.bucket, args.model_prefix)
    if model is None:
        print("No prior model found; cold-starting a fresh model.")
        model = MultiObjectiveLinUCB(
            n_arms=args.n_arms,
            n_features=args.n_features,
            alpha=args.alpha,
            weights=weights,
        )
    else:
        print("Loaded latest model state from S3.")

    # 2/3. Split feedback vs inference and update on feedback.
    batch = load_batch(args.input_dir)
    if "fb_arm" in batch and len(batch["fb_arm"]) > 0:
        update_from_feedback(
            model,
            {
                "arm": batch["fb_arm"],
                "context": batch["fb_context"],
                "reward_start": batch["fb_reward_start"],
                "reward_submit": batch["fb_reward_submit"],
                "reward_approve": batch["fb_reward_approve"],
            },
        )
        print(f"Updated model on {len(batch['fb_arm'])} feedback rows.")

    # 4. Score all prospects (inference rows).
    prospect_ids = batch["inf_entity_id"]
    prospect_contexts = batch["inf_context"]
    selected = score_prospects(model, prospect_contexts)
    print(f"Scored {len(prospect_ids)} prospects.")

    # 5a. Persist updated model state under a new dated prefix (versioned).
    key = save_model_state(model, args.bucket, prefix=args.model_prefix)
    print(f"Saved model state to s3://{args.bucket}/{key}")

    # 5b. Write per-entity recommendations for the serving layer.
    #     entity_id is an opaque routing key used only for serving lookup;
    #     it is never a model feature.
    os.makedirs(args.output_dir, exist_ok=True)
    recs_path = os.path.join(args.output_dir, "recommendations.jsonl")
    with open(recs_path, "w") as f:
        for eid, arm in zip(prospect_ids, selected):
            f.write(json.dumps({"entity_id": str(eid), "arm": int(arm)}) + "\n")
    print(f"Wrote recommendations to {recs_path}")


if __name__ == "__main__":
    main()
