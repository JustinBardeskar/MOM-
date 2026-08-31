"""
app.infrastructure
==================
Persistence Adapters, SQLite WAL Database Driver, and Async Dispatcher.

This module provides:
- SQLiteDatabase: High-performance SQLite manager with WAL journal mode, connection pooling, and auto-migrations.
- InMemoryJobRepository: In-memory dictionary repository for isolated unit testing.
- SQLiteJobRepository: Disk-backed SQLite repository storing jobs, transcripts, and structured deliverables.
- AsyncJobDispatcher: Concurrent asynchronous queue processor managing background worker tasks.
"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Generator
from uuid import UUID

from app.domain import JobRecord


# ==========================================
# Default Database Path & Connection Manager
# ==========================================

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "runtime" / "mom_ai_agent.db"


class SQLiteDatabase:
    """Manages SQLite database connections, schema migrations, and WAL configuration."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes an optimized SQLite connection with Write-Ahead Logging (WAL)."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a thread-safe connection."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initializes tables and indexes for job tracking, transcripts, and memory."""
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_jobs (
                    id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    meeting_id TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    selected_path TEXT,
                    current_stage TEXT,
                    progress_percent INTEGER DEFAULT 0,
                    error_code TEXT,
                    error_message TEXT,
                    failed_milestone TEXT,
                    request_json TEXT NOT NULL,
                    preprocessed_json TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_meeting_jobs_idempotency
                ON meeting_jobs(idempotency_key);
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meeting_memory (
                    meeting_id TEXT PRIMARY KEY,
                    title TEXT,
                    client_id TEXT,
                    occurred_at TEXT,
                    summary TEXT,
                    decisions_json TEXT,
                    pending_actions_json TEXT,
                    participants_json TEXT,
                    created_at TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_meeting_memory_client
                ON meeting_memory(client_id);
                """
            )

    async def execute_write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Executes an INSERT/UPDATE/DELETE statement asynchronously off the main event loop."""
        def _write() -> None:
            with self.connect() as conn:
                conn.execute(sql, params)
        await asyncio.to_thread(_write)

    # Alias for commit execution
    execute_commit = execute_write

    async def execute_query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Executes a SELECT query asynchronously and returns row records."""
        def _read() -> list[sqlite3.Row]:
            with self.connect() as conn:
                cursor = conn.execute(sql, params)
                return cursor.fetchall()
        return await asyncio.to_thread(_read)


# ==========================================
# In-Memory Repository (Testing Adapter)
# ==========================================

class InMemoryJobRepository:
    """In-memory adapter used for unit tests and ephemeral execution."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, JobRecord] = {}
        self._idempotency_index: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def create_or_get(self, job: JobRecord) -> tuple[JobRecord, bool]:
        """Creates a new job or returns existing if idempotency key is matched."""
        async with self._lock:
            existing_id = self._idempotency_index.get(job.idempotency_key)
            if existing_id is not None:
                return self._jobs[existing_id], True
            self._jobs[job.id] = job
            self._idempotency_index[job.idempotency_key] = job.id
            return job, False

    async def get(self, job_id: UUID) -> JobRecord | None:
        """Fetches job by ID."""
        return self._jobs.get(job_id)

    async def save(self, job: JobRecord) -> None:
        """Persists updated job state."""
        async with self._lock:
            self._jobs[job.id] = job

    async def list_all(self) -> list[JobRecord]:
        """Lists all active and completed jobs sorted by creation date."""
        async with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)


# ==========================================
# SQLite Persistent Repository
# ==========================================

class SQLiteJobRepository:
    """Persistent SQLite adapter that stores all meeting jobs and deliverables to disk."""

    def __init__(self, db: SQLiteDatabase | None = None) -> None:
        self.db = db or SQLiteDatabase()
        self._lock = asyncio.Lock()

    async def create_or_get(self, job: JobRecord) -> tuple[JobRecord, bool]:
        """Atomically retrieves existing job by idempotency key or inserts new job."""
        async with self._lock:
            existing_rows = await self.db.execute_query(
                "SELECT id, request_json, result_json, preprocessed_json FROM meeting_jobs WHERE idempotency_key = ?",
                (job.idempotency_key,),
            )
            if existing_rows:
                existing_id = UUID(existing_rows[0]["id"])
                existing_job = await self.get(existing_id)
                if existing_job:
                    return existing_job, True

            # Serialize and insert new job record
            job_json = job.model_dump_json()
            await self.db.execute_write(
                """
                INSERT INTO meeting_jobs (
                    id, event_id, meeting_id, idempotency_key, status,
                    selected_path, current_stage, progress_percent,
                    error_code, error_message, failed_milestone,
                    request_json, preprocessed_json, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job.id),
                    job.event_id,
                    job.meeting_id,
                    job.idempotency_key,
                    job.status.value,
                    job.selected_path.value if job.selected_path else None,
                    job.current_stage.value if job.current_stage else None,
                    job.progress_percent,
                    job.error_code,
                    job.error_message,
                    job.failed_milestone.value if job.failed_milestone else None,
                    job_json,
                    job.preprocessed_transcript.model_dump_json() if job.preprocessed_transcript else None,
                    json.dumps(job.result) if job.result else None,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )
            return job, False

    async def get(self, job_id: UUID) -> JobRecord | None:
        """Fetches full job aggregate from SQLite database."""
        rows = await self.db.execute_query(
            "SELECT request_json FROM meeting_jobs WHERE id = ?",
            (str(job_id),),
        )
        if not rows:
            return None
        return JobRecord.model_validate_json(rows[0]["request_json"])

    async def save(self, job: JobRecord) -> None:
        """Saves updated job aggregate state to SQLite database."""
        async with self._lock:
            job_json = job.model_dump_json()
            await self.db.execute_write(
                """
                UPDATE meeting_jobs SET
                    status = ?,
                    selected_path = ?,
                    current_stage = ?,
                    progress_percent = ?,
                    error_code = ?,
                    error_message = ?,
                    failed_milestone = ?,
                    request_json = ?,
                    preprocessed_json = ?,
                    result_json = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    job.status.value,
                    job.selected_path.value if job.selected_path else None,
                    job.current_stage.value if job.current_stage else None,
                    job.progress_percent,
                    job.error_code,
                    job.error_message,
                    job.failed_milestone.value if job.failed_milestone else None,
                    job_json,
                    job.preprocessed_transcript.model_dump_json() if job.preprocessed_transcript else None,
                    json.dumps(job.result) if job.result else None,
                    job.updated_at.isoformat(),
                    str(job.id),
                ),
            )

    async def list_all(self) -> list[JobRecord]:
        """Lists all jobs in reverse chronological order."""
        rows = await self.db.execute_query(
            "SELECT request_json FROM meeting_jobs ORDER BY created_at DESC"
        )
        return [JobRecord.model_validate_json(row["request_json"]) for row in rows]


# ==========================================
# Asynchronous Background Job Dispatcher
# ==========================================

class AsyncJobDispatcher:
    """Concurrent in-process asyncio task queue processor managing background pipelines."""

    def __init__(
        self,
        handler: Callable[[UUID], Awaitable[None]],
        concurrency: int,
    ) -> None:
        self._handler = handler
        self._concurrency = concurrency
        self._queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Launches worker coroutine tasks."""
        self._workers = [
            asyncio.create_task(self._worker(), name=f"automation-worker-{index}")
            for index in range(self._concurrency)
        ]

    async def stop(self) -> None:
        """Cancels and cleans up worker tasks gracefully."""
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def publish(self, job_id: UUID) -> None:
        """Publishes a job ID to the background queue."""
        await self._queue.put(job_id)

    async def _worker(self) -> None:
        """Background worker consuming job IDs and invoking pipeline handler."""
        while True:
            job_id = await self._queue.get()
            try:
                await self._handler(job_id)
            finally:
                self._queue.task_done()


class InMemoryJobDispatcher:
    """In-memory dispatcher recording published jobs for unit tests."""

    def __init__(self) -> None:
        self.published_jobs: list[UUID] = []

    async def publish(self, job_id: UUID) -> None:
        """Appends published job ID to local list."""
        self.published_jobs.append(job_id)
