"""Parallelism utilities shared across optimization flows."""

import os
from typing import Optional, Tuple

from joblib import cpu_count as joblib_cpu_count


def resolve_worker_counts(
    configured_n_jobs: int,
    default_workers: int = 4,
) -> Tuple[Optional[int], int, int, int]:
    """Resolve detected CPU counts and effective worker count.

    Returns:
        Tuple of (os_detected_cpus, joblib_detected_cpus,
        selected_detected_cpus, effective_workers)
    """
    os_detected_cpus = os.cpu_count()
    joblib_detected_cpus = joblib_cpu_count()
    detected_cpus = os_detected_cpus or joblib_detected_cpus or default_workers
    effective_workers = (
        detected_cpus if configured_n_jobs == -1 else configured_n_jobs
    )
    return os_detected_cpus, joblib_detected_cpus, detected_cpus, effective_workers
