"""In-memory job store.

Fine for a single-process dev server. For production with multiple workers
or a GPU queue, swap this for Redis (e.g. RQ/Celery) so job state is shared
across processes and survives restarts.
"""

from __future__ import annotations

import uuid
from enum import Enum
from threading import Lock

from pydantic import BaseModel


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    id: str
    status: JobStatus = JobStatus.QUEUED
    glb_url: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update=fields)


store = JobStore()
