# Multi-Objective Contextual Bandit with Amazon SageMaker AI

> Repository: `sample-multi-objective-contextual-bandit-with-sagemaker`

A small, self-contained sample that shows how to run a **multi-objective
contextual bandit** (LinUCB) as a batch job with **Amazon SageMaker AI**. It
accompanies the AWS blog post on driving conversion lift with contextual
bandits.

The bandit personalizes content by choosing, per entity, the content variation
(*arm*) most likely to succeed across an **entire conversion funnel** — start,
submit, and approve — rather than optimizing a single metric. This is the
"seesaw problem" fix described in the blog: optimizing one funnel stage in
isolation can quietly regress another.

> This is an illustrative sample built on **synthetic data**. It contains no
> real customer data, features, or content.

## What's here

```
src/
  linucb.py            # Core algorithm: LinUCBDisjoint + MultiObjectiveLinUCB
  s3_model_registry.py # S3 as a versioned model registry (dated prefixes)
  batch_job.py         # Batch entry point: load -> update -> score -> persist
examples/
  synthetic_data.py    # Synthetic funnel environment + batch generator
  demo.py              # Runnable local demo (no AWS needed)
notebooks/
  walkthrough.ipynb    # Guided, explained walkthrough of the whole sample
deploy/
  run_processing_job.py# Launch batch_job.py as a SageMaker AI Processing job
tests/
  test_bandit.py       # Unit + end-to-end learning tests
```

## Quickstart (local, no AWS)

```bash
pip install -r requirements.txt
python examples/demo.py     # watch the bandit learn and beat a random baseline
pytest -q                   # run the tests
```

The demo streams synthetic entities through the bandit and reports how its
funnel reward climbs above a random-arm baseline as it learns.

### Prefer a guided walkthrough?

Open `notebooks/walkthrough.ipynb` for a step-by-step, explained tour: the
intuition behind LinUCB, optimizing the whole funnel, watching the bandit learn,
tuning `alpha`, and how the same model runs as a SageMaker AI batch job. It runs
on synthetic data with just NumPy (plots appear if matplotlib is installed).

```bash
pip install jupyter matplotlib
jupyter notebook notebooks/walkthrough.ipynb
```

## The algorithm in brief

Each arm keeps two running tallies, updated on every impression:

- `A` — the experience ledger (`A += x·xᵀ`, starts at identity)
- `b` — the reward ledger (`b += reward · x`)

The score for an arm given context `x` balances an estimate against an
uncertainty bonus:

```
score(a, x) = θᵀ·x  +  α · sqrt(xᵀ · A⁻¹ · x)
              estimate       uncertainty bonus       (θ = A⁻¹·b)
```

`MultiObjectiveLinUCB` runs one such bandit per funnel stage and selects the arm
that maximizes a weighted sum of the three UCB scores. `alpha` controls
exploration strength (higher = explore more).

## Running on Amazon SageMaker AI

`deploy/run_processing_job.py` launches `src/batch_job.py` inside a SageMaker AI
managed scikit-learn container using `SKLearnProcessor`. SageMaker AI copies
input from Amazon S3 to the job, runs the update-and-score step, and copies the
output recommendations back to S3. Amazon S3 also serves as a simple versioned
model registry via dated prefixes, giving a built-in audit trail and rollback.

```bash
pip install sagemaker
python deploy/run_processing_job.py \
    --bucket my-bucket \
    --role arn:aws:iam::123456789012:role/MySageMakerRole \
    --input-s3 s3://my-bucket/personalizer/batches/2026-02-09/ \
    --output-s3 s3://my-bucket/personalizer/recommendations/2026-02-09/ \
    --n-arms 92 --n-features 21
```

Because approval feedback is delayed by days, the bandit is designed to update
on a **weekly cadence**. You can schedule this Processing job to run weekly by
wrapping it in a SageMaker AI Pipeline with a `PipelineSchedule`, or by
triggering it from Amazon EventBridge Scheduler — see
[Schedule your ML workflows](https://docs.aws.amazon.com/sagemaker/latest/dg/workflow-scheduling.html).

## A note on `entity_id`

Recommendations are keyed by an opaque `entity_id`. It is used **only** to route
a precomputed recommendation back to the right place at serving time — it is
never a model feature.

## Reference

Li, Chu, Langford, Schapire (2010),
["A Contextual-Bandit Approach to Personalized News Article Recommendation"](https://arxiv.org/abs/1003.0146).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
