import asyncio
from datetime import datetime, timezone
from pathlib import Path
import pytest
from uuid import uuid4

from app.ai_brain.models import ActionItem, MemoryRecord
from app.ai_brain.memory import SQLiteMemoryStore
from app.domain import (
    JobRecord,
    JobStatus,
    MeetingProvider,
    MeetingReadyRequest,
    Participant,
    PipelineStage,
    ProcessingPath,
)
from app.infrastructure import SQLiteDatabase, SQLiteJobRepository


@pytest.fixture
def temp_db(tmp_path: Path) -> SQLiteDatabase:
    db_file = tmp_path / "test_mom.db"
    return SQLiteDatabase(db_path=db_file)


def test_sqlite_job_repository_crud(temp_db: SQLiteDatabase) -> None:
    async def _test() -> None:
        repo = SQLiteJobRepository(db=temp_db)

        job_id = uuid4()
        req = MeetingReadyRequest(
            event_id="evt-sql-1",
            meeting_id="mtg-sql-1",
            title="Database Architecture Review",
            ended_at=datetime.now(timezone.utc),
            provider=MeetingProvider.TEAMS,
            transcript={"url": "https://example.com/test.txt", "content_type": "text/plain"},
            participants=[Participant(display_name="Alex"), Participant(display_name="Sarah")],
        )
        job = JobRecord(
            id=job_id,
            event_id=req.event_id,
            meeting_id=req.meeting_id,
            idempotency_key=req.event_id,
            status=JobStatus.AWAITING_ANALYSIS,
            selected_path=ProcessingPath.DIRECT_TRANSCRIPT,
            current_stage=PipelineStage.PREPROCESSED_TRANSCRIPT_READY,
            progress_percent=60,
            planned_steps=[],
            request=req,
        )

        # 1. Create Job
        saved_job, duplicate = await repo.create_or_get(job)
        assert not duplicate
        assert saved_job.id == job_id

        # 2. Duplicate Idempotency Check
        dup_job, duplicate_flag = await repo.create_or_get(job)
        assert duplicate_flag
        assert dup_job.id == job_id

        # 3. Retrieve Job
        retrieved = await repo.get(job_id)
        assert retrieved is not None
        assert retrieved.meeting_id == "mtg-sql-1"
        assert retrieved.request.title == "Database Architecture Review"

        # 4. Update Job with Result
        retrieved.status = JobStatus.COMPLETED
        retrieved.result = {"meeting_summary": "Database was successfully reviewed.", "key_points": ["Point 1"]}
        await repo.save(retrieved)

        # 5. Simulate Server Restart (New repo instance on same DB file)
        repo2 = SQLiteJobRepository(db=temp_db)
        reloaded = await repo2.get(job_id)
        assert reloaded is not None
        assert reloaded.status == JobStatus.COMPLETED
        assert reloaded.result is not None
        assert reloaded.result["meeting_summary"] == "Database was successfully reviewed."

        # 6. List All
        all_jobs = await repo2.list_all()
        assert len(all_jobs) == 1
        assert all_jobs[0].id == job_id

    asyncio.run(_test())


def test_sqlite_memory_store_persistence(temp_db: SQLiteDatabase) -> None:
    async def _test() -> None:
        store = SQLiteMemoryStore(db=temp_db)

        record = MemoryRecord(
            meeting_id="meeting-101",
            client_id="client-acme",
            occurred_at=datetime.now(timezone.utc),
            summary="Discussed Q3 migration roadmap.",
            pending_action_items=[
                ActionItem(description="Implement Redis connection pool", owner="David", status="pending")
            ],
            metadata={"client_id": "client-acme"},
        )

        # Save memory
        await store.save(record)

        # Recall memory with fresh store instance
        store2 = SQLiteMemoryStore(db=temp_db)
        recalled = await store2.recall(client_id="client-acme", limit=5)
        assert len(recalled) == 1
        assert recalled[0].meeting_id == "meeting-101"
        assert recalled[0].summary == "Discussed Q3 migration roadmap."
        assert len(recalled[0].pending_action_items) == 1
        assert recalled[0].pending_action_items[0].owner == "David"

    asyncio.run(_test())
