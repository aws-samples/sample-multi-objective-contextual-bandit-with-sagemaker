"""Run the batch bandit update as an Amazon SageMaker AI Processing job.

This launches ``src/batch_job.py`` inside a SageMaker AI-managed scikit-learn
container using ``SKLearnProcessor`` (SageMaker Python SDK). SageMaker AI pulls
the container, copies the input data from S3 to ``/opt/ml/processing/input``,
runs the script, and copies ``/opt/ml/processing/output`` back to S3.

Why SKLearnProcessor?
    Our entry point is plain NumPy/Python. The managed scikit-learn image
    already bundles NumPy, so we avoid building a custom container. Swap in
    ``FrameworkProcessor`` / ``PyTorchProcessor`` etc. if your entry point
    needs a different base image.

This runs the job once. To run it on a recurring cadence (the bandit updates
weekly, since approval feedback is delayed), wrap this Processing job in a
SageMaker AI Pipeline and attach a schedule with ``PipelineSchedule``, or
trigger it from Amazon EventBridge Scheduler. See "Schedule your ML workflows"
in the SageMaker AI docs.

Prereqs: ``pip install sagemaker`` and AWS credentials with a SageMaker
execution role. This script is illustrative - set the bucket/role/paths for
your account.

Usage:
    python deploy/run_processing_job.py \
        --bucket my-bucket \
        --role arn:aws:iam::123456789012:role/MySageMakerRole \
        --input-s3 s3://my-bucket/personalizer/batches/2026-02-09/ \
        --output-s3 s3://my-bucket/personalizer/recommendations/2026-02-09/ \
        --n-arms 92 --n-features 21
"""

from __future__ import annotations

import argparse
import os

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
INPUT_DEST = "/opt/ml/processing/input"
OUTPUT_SRC = "/opt/ml/processing/output"


def build_processor(role: str, instance_type: str, instance_count: int) -> SKLearnProcessor:
    return SKLearnProcessor(
        framework_version="1.2-1",  # managed scikit-learn container (bundles NumPy)
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name="mab-linucb-batch",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--role", required=True, help="SageMaker AI execution role ARN")
    p.add_argument("--input-s3", required=True, help="S3 URI with batch.npz")
    p.add_argument("--output-s3", required=True, help="S3 URI for recommendations")
    p.add_argument("--n-arms", type=int, required=True)
    p.add_argument("--n-features", type=int, required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--weights", default="0.33,0.33,0.34")
    p.add_argument("--instance-type", default="ml.m5.xlarge")
    p.add_argument("--instance-count", type=int, default=1)
    args = p.parse_args()

    processor = build_processor(args.role, args.instance_type, args.instance_count)

    processor.run(
        code="batch_job.py",
        source_dir=SRC_DIR,  # ships linucb.py + s3_model_registry.py alongside
        inputs=[ProcessingInput(source=args.input_s3, destination=INPUT_DEST)],
        outputs=[ProcessingOutput(source=OUTPUT_SRC, destination=args.output_s3)],
        arguments=[
            "--bucket", args.bucket,
            "--n-arms", str(args.n_arms),
            "--n-features", str(args.n_features),
            "--alpha", str(args.alpha),
            "--weights", args.weights,
            "--input-dir", INPUT_DEST,
            "--output-dir", OUTPUT_SRC,
        ],
    )
    print("Processing job submitted.")


if __name__ == "__main__":
    main()
