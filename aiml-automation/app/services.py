"""
app.services
============
Application Layer Coordination, Job Planning, and Ingestion Services.

This module provides:
- PipelineRunner: Abstract interface for milestone pipeline executions.
- AutomationPipelineCoordinator: Executes sequential milestones (M1+M2 -> M3 -> M4 -> M5 -> M6).
- PipelinePlanner: Determines optimal processing path (recording vs transcript) and builds step list.
- MeetingIngestionService: Validates, deduplicates via idempotency keys, enqueues, and monitors jobs.
- Exceptions: JobNotFoundError, ResultNotReadyError, TranscriptNotReadyError, PreprocessedTranscriptNotReadyError.
"""

from typing import Protocol
from uuid import UUID

from app.domain import (
    InputPreference,
    JobAcceptedResponse,
    JobDispatcher,
    JobRecord,
    JobRepository,
    JobStatus,
    MeetingReadyRequest,
    PipelineStage,
    PipelineStep,
    ProcessingPath,
)


# ==========================================
# Domain Service Exceptions
# ==========================================

class JobNotFoundError(Exception):
    """Raised when a requested Job ID cannot be found in the repository."""
    pass


class ResultNotReadyError(Exception):
    """Raised when job result is queried before analysis completion."""
    pass


class TranscriptNotReadyError(Exception):
    """Raised when M1 transcript is queried before transcription completion."""
    pass


class PreprocessedTranscriptNotReadyError(Exception):
    """Raised when M2 preprocessed transcript is queried before preprocessing completion."""
    pass


# ==========================================
# Pipeline Coordination & Planning
# ==========================================

class PipelineRunner(Protocol):
    """Abstract protocol for executing milestone pipeline stages."""
    async def run(self, job_id: UUID) -> None:
        """Executes the pipeline stage for the given job."""
        ...


class AutomationPipelineCoordinator:
    """
    Coordinates sequential execution of milestone pipelines without tight coupling:
    1. Runs M1 (Ingestion/Transcription) and M2 (Preprocessing/Chunking).
    2. Runs M3 (AI Brain Multi-Agent Intelligence).
    3. Runs M4 (Analysis Validation & Golden Verification).
    4. Runs M5 (Memory Validation & Cross-Meeting Consistency).
    5. Runs M6 (Final Deliverables Packaging & Delivery Readiness).
    """

    def __init__(
        self,
        m1_m2: PipelineRunner,
        model3: PipelineRunner | None,
        model4: PipelineRunner | None,
        model5: PipelineRunner | None,
        model6: PipelineRunner | None,
    ) -> None:
        self._m1_m2 = m1_m2
        self._model3 = model3
        self._model4 = model4
        self._model5 = model5
        self._model6 = model6

    async def run(self, job_id: UUID) -> None:
        """Executes all available milestone pipelines sequentially."""
        # Execute M1 & M2
        await self._m1_m2.run(job_id)

        # Execute M3 AI Brain if configured
        if self._model3 is not None:
            await self._model3.run(job_id)

        # Execute M4 Analysis Stage
        if self._model4 is not None:
            await self._model4.run(job_id)

        # Execute M5 Memory Validation Stage
        if self._model5 is not None:
            await self._model5.run(job_id)

        # Execute M6 Deliverable Finalization Stage
        if self._model6 is not None:
            await self._model6.run(job_id)


class PipelinePlanner:
    """
    Determines the appropriate processing strategy based on input preferences and available assets:
    - DIRECT_TRANSCRIPT: Direct ingestion of provided VTT/SRT/JSON transcript.
    - RECORDING_TO_TRANSCRIPT: Remote media download, FFmpeg audio extraction, and Whisper STT.
    """

    def select_path(self, request: MeetingReadyRequest) -> ProcessingPath:
        """Selects either direct transcript processing or recording audio transcription."""
        if request.input_preference == InputPreference.RECORDING:
            return ProcessingPath.RECORDING_TO_TRANSCRIPT
        if request.input_preference == InputPreference.TRANSCRIPT:
            return ProcessingPath.DIRECT_TRANSCRIPT
        if request.transcript is not None:
            return ProcessingPath.DIRECT_TRANSCRIPT
        return ProcessingPath.RECORDING_TO_TRANSCRIPT

    def build_steps(self, path: ProcessingPath) -> list[PipelineStep]:
        """Constructs the sequence of planned stages for frontend progress tracking."""
        shared_steps = [
            PipelineStep(stage=PipelineStage.NORMALIZE_TRANSCRIPT),
            PipelineStep(stage=PipelineStage.VALIDATE_M1_OUTPUT),
            PipelineStep(stage=PipelineStage.UNIFIED_TRANSCRIPT_READY),
            PipelineStep(stage=PipelineStage.M1_TO_M2_HANDOFF),
            PipelineStep(stage=PipelineStage.REMOVE_FILLERS),
            PipelineStep(stage=PipelineStage.NORMALIZE_SPEAKERS),
            PipelineStep(stage=PipelineStage.CLEAN_TIMESTAMPS),
            PipelineStep(stage=PipelineStage.REMOVE_TRANSCRIPT_NOISE),
            PipelineStep(stage=PipelineStage.CHUNK_TRANSCRIPT),
            PipelineStep(stage=PipelineStage.BUILD_CONTEXT),
            PipelineStep(stage=PipelineStage.VALIDATE_M2_OUTPUT),
            PipelineStep(stage=PipelineStage.M2_TO_M3_HANDOFF),
            PipelineStep(stage=PipelineStage.MEETING_UNDERSTANDING),
            PipelineStep(stage=PipelineStage.PARALLEL_AGENT_ANALYSIS),
            PipelineStep(stage=PipelineStage.VALIDATE_AGENT_OUTPUTS),
            PipelineStep(stage=PipelineStage.DETECT_AGENT_CONFLICTS),
            PipelineStep(stage=PipelineStage.VALIDATE_MEMORY),
            PipelineStep(stage=PipelineStage.SCORE_CONFIDENCE),
            PipelineStep(stage=PipelineStage.M3_TO_M4_HANDOFF),
            PipelineStep(stage=PipelineStage.M4_ANALYSIS),
            PipelineStep(stage=PipelineStage.VALIDATE_M4_OUTPUT),
            PipelineStep(stage=PipelineStage.M4_TO_M5_HANDOFF),
            PipelineStep(stage=PipelineStage.M5_VALIDATION),
            PipelineStep(stage=PipelineStage.M5_TO_M6_HANDOFF),
            PipelineStep(stage=PipelineStage.M6_FINALIZATION),
            PipelineStep(stage=PipelineStage.FINAL_STRUCTURED_JSON_READY),
            PipelineStep(stage=PipelineStage.DELIVER_RESULTS),
        ]
        if path == ProcessingPath.DIRECT_TRANSCRIPT:
            return shared_steps
        return [
            PipelineStep(stage=PipelineStage.DOWNLOAD_RECORDING),
            PipelineStep(stage=PipelineStage.EXTRACT_AUDIO),
            PipelineStep(stage=PipelineStage.SPEECH_TO_TEXT),
            *shared_steps,
        ]


# ==========================================
# Meeting Ingestion Application Service
# ==========================================

class MeetingIngestionService:
    """
    High-level application service handling meeting ingestion requests,
    idempotent job deduplication, queue dispatching, and status retrieval.
    """

    def __init__(
        self,
        repository: JobRepository,
        dispatcher: JobDispatcher,
        planner: PipelinePlanner,
    ) -> None:
        self._repository = repository
        self._dispatcher = dispatcher
        self._planner = planner

    @property
    def repository(self) -> JobRepository:
        return self._repository

    async def accept(
        self,
        request: MeetingReadyRequest,
        idempotency_key: str,
    ) -> JobAcceptedResponse:
        """
        Accepts an incoming meeting payload, creates or recovers an existing job,
        and triggers background pipeline execution.
        """
        selected_path = self._planner.select_path(request)
        candidate = JobRecord(
            event_id=request.event_id,
            meeting_id=request.meeting_id,
            idempotency_key=idempotency_key,
            selected_path=selected_path,
            planned_steps=self._planner.build_steps(selected_path),
            request=request,
        )
        job, duplicate = await self._repository.create_or_get(candidate)

        # Only dispatch background worker if job is freshly created
        if not duplicate:
            await self._dispatcher.publish(job.id)

        return JobAcceptedResponse(
            job_id=job.id,
            meeting_id=job.meeting_id,
            status=job.status,
            selected_path=job.selected_path,
            status_url=f"/api/v1/jobs/{job.id}",
            transcript_url=f"/api/v1/jobs/{job.id}/transcript",
            preprocessed_url=f"/api/v1/jobs/{job.id}/preprocessed",
            result_url=f"/api/v1/jobs/{job.id}/result",
            duplicate=duplicate,
        )

    async def get_job(self, job_id: UUID) -> JobRecord:
        """Retrieves raw job record by ID or raises JobNotFoundError."""
        job = await self._repository.get(job_id)
        if job is None:
            raise JobNotFoundError(str(job_id))
        return job

    async def get_result(self, job_id: UUID) -> JobRecord:
        """Retrieves job results or formatted error deliverables if failed."""
        job = await self.get_job(job_id)
        if job.status == JobStatus.FAILED:
            if not job.result:
                job.result = {
                    "status": "failed",
                    "error_code": job.error_code or "processing_failed",
                    "error_message": job.error_message or "Meeting analysis failed",
                    "meeting_summary": f"Analysis failed: {job.error_message or job.error_code}",
                }
            return job
        if job.result is None:
            raise ResultNotReadyError(str(job_id))
        return job

    async def get_transcript(self, job_id: UUID) -> JobRecord:
        """Retrieves M1 unified transcript or raises TranscriptNotReadyError."""
        job = await self.get_job(job_id)
        if job.unified_transcript is None:
            raise TranscriptNotReadyError(str(job_id))
        return job

    async def get_preprocessed_transcript(self, job_id: UUID) -> JobRecord:
        """Retrieves M2 preprocessed transcript or raises PreprocessedTranscriptNotReadyError."""
        job = await self.get_job(job_id)
        if job.preprocessed_transcript is None:
            raise PreprocessedTranscriptNotReadyError(str(job_id))
        return job
