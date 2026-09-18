"""In-process background job registry for long-running trading cycles.

Design (Phase 4):
- Single-process, in-memory registry — matches the containerized deployment
  (one health server, one process). No Celery/Redis.
- Exactly one job may run at a time. Requests for a new job while one is
  running are rejected by the caller (HTTP 409).
- Live progress is exposed via a callback (``progress_cb(percent, message)``)
  that long-running stages (optimizer, walk-forward) invoke at natural
  boundaries. Final outcomes are mirrored into ``shared_state['last_result']``
  by main.py so the dashboard keeps working across restarts (job history is
  intentionally not persisted).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]

# How many finished jobs to keep in the registry (in-memory history).
MAX_FINISHED_JOBS = 10


@dataclass
class Job:
    """State of one background run (trading cycle / backtest batch)."""
    kind: str = "full_cycle"          # full_cycle | run_session
    status: str = "queued"            # queued | running | done | failed
    progress: int = 0                 # 0-100
    message: str = ""                 # current stage, e.g. "Optimizing AAPL"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_summary: Dict = field(default_factory=dict)
    job_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
        }


class JobManager:
    """Thread-safe registry for background jobs (one active at a time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: Optional[Job] = None
        self._history: List[Job] = []

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def has_active(self) -> bool:
        with self._lock:
            return self._active is not None

    def get_active(self) -> Optional[Job]:
        with self._lock:
            return self._active

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            if self._active and self._active.job_id == job_id:
                return self._active
            for job in self._history:
                if job.job_id == job_id:
                    return job
        return None

    def list_jobs(self, limit: int = 20) -> List[Dict]:
        """Newest-first snapshot (active job first, then finished history)."""
        with self._lock:
            jobs: List[Job] = []
            if self._active:
                jobs.append(self._active)
            jobs.extend(reversed(self._history))
            return [j.to_dict() for j in jobs[:limit]]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_job(self, kind: str = "full_cycle") -> Optional[Job]:
        """Register a new queued job. Returns None if a job is active."""
        with self._lock:
            if self._active is not None:
                return None
            job = Job(
                kind=kind,
                status="queued",
                job_id=uuid.uuid4().hex[:12],
            )
            self._active = job
            return job

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            if self._active and self._active.job_id == job_id:
                self._active.status = "running"
                self._active.started_at = time.time()

    def update_progress(self, job_id: str, percent: int, message: str) -> None:
        with self._lock:
            if self._active and self._active.job_id == job_id:
                self._active.progress = max(0, min(100, int(percent)))
                self._active.message = str(message)

    def finish_job(self, job_id: str, success: bool,
                   result_summary: Optional[Dict] = None,
                   error: Optional[str] = None) -> None:
        with self._lock:
            if self._active and self._active.job_id == job_id:
                job = self._active
                job.status = "done" if success else "failed"
                job.progress = 100 if success else job.progress
                job.finished_at = time.time()
                job.result_summary = result_summary or {}
                job.error = error
                self._active = None
                self._history.append(job)
                if len(self._history) > MAX_FINISHED_JOBS:
                    self._history = self._history[-MAX_FINISHED_JOBS:]

    def make_progress_callback(self, job_id: str,
                               base: int = 0, span: int = 100) -> ProgressCallback:
        """Build a callback that scales sub-progress into [base, base+span]."""

        def _cb(percent: int, message: str) -> None:
            try:
                scaled = base + (span * max(0, min(100, int(percent))) // 100)
                self.update_progress(job_id, scaled, message)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        return _cb


# Module-level singleton used by main.py and health_server.py.
job_manager = JobManager()
