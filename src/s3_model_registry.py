"""S3-backed model registry for the batch bandit.

Amazon S3 doubles as the model registry. Each batch run persists its model
state under a dated prefix, for example::

    s3://<bucket>/personalizer/models/2026-02-09/model.npz
    s3://<bucket>/personalizer/models/2026-02-16/model.npz

Persisting under a dated prefix gives a built-in version history and a
straightforward rollback (point at an earlier date). The job discovers the
latest dated prefix automatically at startup, so no configuration change is
needed between runs.

Model state is stored as NumPy arrays (.npz) and loaded with
``allow_pickle=False``. These helpers are thin wrappers around boto3 so the
pattern is easy to read and adapt to your environment.
"""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import Any, Optional

import boto3
import numpy as np

from linucb import MultiObjectiveLinUCB

DEFAULT_PREFIX = "personalizer/models/"


def _parse_date_from_key(key: str, prefix: str) -> Optional[date]:
    """Extract a YYYY-MM-DD date embedded in an S3 key under ``prefix``."""
    tail = key[len(prefix):] if key.startswith(prefix) else key
    date_str = tail.split("/", 1)[0]
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def find_latest_model_date(
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    s3_client: Any = None,
) -> Optional[date]:
    """Return the most recent dated prefix under ``prefix``, or None."""
    s3 = s3_client or boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    dates: set[date] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            parsed = _parse_date_from_key(obj["Key"], prefix)
            if parsed is not None:
                dates.add(parsed)
    return max(dates) if dates else None


def load_latest_model_state(
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    filename: str = "model.npz",
    s3_client: Any = None,
) -> Optional[MultiObjectiveLinUCB]:
    """Discover and load the most recent dated model.

    Returns ``None`` if no prior model exists (cold start).
    """
    s3 = s3_client or boto3.client("s3")
    latest = find_latest_model_date(bucket, prefix, s3_client=s3)
    if latest is None:
        return None
    key = f"{prefix}{latest.isoformat()}/{filename}"
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with np.load(io.BytesIO(body), allow_pickle=False) as npz:
        state = {k: npz[k] for k in npz.files}
    return MultiObjectiveLinUCB.from_state(state)


def save_model_state(
    model: MultiObjectiveLinUCB,
    bucket: str,
    run_date: Optional[date] = None,
    prefix: str = DEFAULT_PREFIX,
    filename: str = "model.npz",
    s3_client: Any = None,
) -> str:
    """Persist model state under a new dated prefix; return the S3 key."""
    s3 = s3_client or boto3.client("s3")
    run_date = run_date or date.today()
    key = f"{prefix}{run_date.isoformat()}/{filename}"
    buffer = io.BytesIO()
    np.savez(buffer, **model.get_state())
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    return key
