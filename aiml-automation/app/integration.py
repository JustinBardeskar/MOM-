"""
app.integration
===============
Milestone Integration, Boundary Contracts, and M1+M2 Pipeline Orchestrator.

This module provides:
- M1ToM2Contract: Validates structural integrity of M1 UnifiedTranscript before M2 ingestion.
- ValidatedM2Output: Validates chunking, token counts, and context bundles produced by M2.
- MilestoneContractValidator: Throws ContractValidationError on schema or semantic boundary violations.
- M1M2PipelineOrchestrator: Executes M1 (transcription) and M2 (preprocessing) with stage progress reporting.
- Milestone2Adapter: Adapts preprocessing pipeline to integration runner protocol.
"""

from collections.abc import Awaitable, Callable
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.domain import (
    JobRecord,
    JobRepository,
    JobStatus,
    MeetingReadyRequest,
    MilestoneName,
    PipelineStage,
    PreprocessedTranscript,
    ProcessingPath,
    UnifiedTranscript,
)

if TYPE_CHECKING:
    from app.processing import TranscriptPreprocessingPipeline

logger = logging.getLogger("automation.pipeline")
StageReporter = Callable[[PipelineStage, int], Awaitable[None]]


# ==========================================
# Contract Validation Exceptions
# ==========================================

class ContractValidationError(Exception):
    """Raised when an inter-milestone handoff artifact fails schema or business rules."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# ==========================================
# Milestone Runner Protocols & Adapter
# ==========================================

class Milestone1Runner(Protocol):
    """Port for Milestone 1 ingestion and transcription execution."""
    async def execute(
        self,
        job: JobRecord,
        job_directory: Path,
        report_stage: StageReporter,
    ) -> UnifiedTranscript: ...


class Milestone2Runner(Protocol):
    """Port for Milestone 2 text cleaning and chunking execution."""
    async def execute(
        self,
        contract: "M1ToM2Contract",
        report_stage: StageReporter,
    ) -> PreprocessedTranscript: ...


class Milestone2Adapter:
    """Adapts the deterministic M2 preprocessing pipeline to the Milestone2Runner protocol."""

    def __init__(self, pipeline: "TranscriptPreprocessingPipeline") -> None:
        self._pipeline = pipeline

    async def execute(
        self,
        contract: "M1ToM2Contract",
        report_stage: StageReporter,
    ) -> PreprocessedTranscript:
        """Executes M2 text cleaning, speaker normalization, and chunking pipeline."""
        return await self._pipeline.process(
            contract.transcript,
            contract.meeting,
            report_stage,
        )


# ==========================================
# Inter-Milestone Boundary Data Contracts
# ==========================================

class M1ToM2Contract(BaseModel):
    """Boundary contract validating M1 UnifiedTranscript before M2 preprocessing."""
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    meeting: MeetingReadyRequest
    selected_path: ProcessingPath
    transcript: UnifiedTranscript

    @model_validator(mode="after")
    def validate_m1_output(self) -> "M1ToM2Contract":
        """Ensures non-empty text, valid segments, and consistent timestamps."""
        if not self.transcript.text.strip():
            raise ValueError("M1 transcript text is empty")
        if not self.transcript.segments:
            raise ValueError("M1 transcript contains no segments")
        if self.transcript.source_path != self.selected_path:
            raise ValueError("M1 source path does not match the selected processing path")
        for index, segment in enumerate(self.transcript.segments):
            if not segment.text.strip():
                raise ValueError(f"M1 segment {index} is empty")
            if segment.end_seconds < segment.start_seconds:
                raise ValueError(f"M1 segment {index} has an invalid time range")
        return self


class ValidatedM2Output(BaseModel):
    """Boundary contract validating M2 preprocessed chunks and context bundles."""
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["1.0"] = "1.0"
    job_id: UUID
    preprocessing: PreprocessedTranscript

    @model_validator(mode="after")
    def validate_m2_output(self) -> "ValidatedM2Output":
        """Ensures chunk-to-context 1:1 mapping, token bounds, and unique IDs."""
        output = self.preprocessing
        if not output.text.strip() or not output.segments:
            raise ValueError("M2 output has no preprocessed transcript")
        if not output.chunks:
            raise ValueError("M2 output has no chunks")
        if len(output.contexts) != len(output.chunks):
            raise ValueError("M2 must produce exactly one context for each chunk")
        if output.statistics.chunk_count != len(output.chunks):
            raise ValueError("M2 chunk statistics do not match the chunk output")

        chunk_ids = [chunk.id for chunk in output.chunks]
        context_chunk_ids = [context.chunk_id for context in output.contexts]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("M2 chunk identifiers are not unique")
        if context_chunk_ids != chunk_ids:
            raise ValueError("M2 contexts do not preserve chunk order and identity")
        for chunk in output.chunks:
            if not chunk.text.strip() or chunk.token_count <= 0:
                raise ValueError(f"M2 chunk {chunk.id} is empty")
            if chunk.end_seconds < chunk.start_seconds:
                raise ValueError(f"M2 chunk {chunk.id} has an invalid time range")
            if any(index < 0 or index >= len(output.segments) for index in chunk.source_segment_indexes):
                raise ValueError(f"M2 chunk {chunk.id} references an invalid segment")
        return self


class MilestoneContractValidator:
    """Validates boundary contracts between pipeline milestones."""

    def validate_m1(self, job: JobRecord, transcript: UnifiedTranscript) -> M1ToM2Contract:
        """Validates M1 transcript and packages into M1ToM2Contract."""
        try:
            return M1ToM2Contract(
                job_id=job.id,
                meeting=job.request,
                selected_path=job.selected_path,
                transcript=transcript,
            )
        except ValidationError as exc:
            raise ContractValidationError("invalid_m1_output", "M1 output failed the M1-to-M2 contract") from exc

    def validate_m2(self, contract: M1ToM2Contract, preprocessing: PreprocessedTranscript) -> ValidatedM2Output:
        """Validates M2 preprocessed output and packages into ValidatedM2Output."""
        try:
            return ValidatedM2Output(job_id=contract.job_id, preprocessing=preprocessing)
        except ValidationError as exc:
            raise ContractValidationError("invalid_m2_output", "M2 output failed validation") from exc


# ==========================================
# Milestone 1 & 2 Sequential Orchestrator
# ==========================================

class M1M2PipelineOrchestrator:
    """Executes M1 (Transcription) and M2 (Preprocessing) in order with validated boundaries."""

    def __init__(
        self,
        repository: JobRepository,
        milestone1: Milestone1Runner,
        milestone2: Milestone2Runner,
        validator: MilestoneContractValidator,
        work_directory: Path,
        keep_work_files: bool,
    ) -> None:
        self._repository = repository
        self._milestone1 = milestone1
        self._milestone2 = milestone2
        self._validator = validator
        self._work_directory = work_directory
        self._keep_work_files = keep_work_files

    async def run(self, job_id: UUID) -> None:
        """Executes full M1 and M2 lifecycle with progress tracking and workspace cleanup."""
        job = await self._repository.get(job_id)
        if job is None:
            logger.error("pipeline_job_not_found", extra={"request_id": str(job_id)})
            return

        job_directory = self._work_directory / str(job.id)
        try:
            job_directory.mkdir(parents=True, exist_ok=True)
            job.status = JobStatus.PROCESSING
            job.error_code = None
            job.error_message = None
            job.failed_milestone = None
            await self._save_stage(job, PipelineStage.MEETING_READY, 2)

            # Stage Reporter Callback
            async def report_stage(stage: PipelineStage, progress_percent: int) -> None:
                await self._save_stage(job, stage, progress_percent)

            # Step 1: Execute Milestone 1 (Ingestion & Transcription)
            transcript = await self._milestone1.execute(job, job_directory, report_stage)
            await self._save_stage(job, PipelineStage.VALIDATE_M1_OUTPUT, 28)
            contract = self._validator.validate_m1(job, transcript)
            job.unified_transcript = transcript
            await self._save_stage(job, PipelineStage.UNIFIED_TRANSCRIPT_READY, 30)

            # Step 2: Execute Milestone 2 (Preprocessing & Chunking)
            await self._save_stage(job, PipelineStage.M1_TO_M2_HANDOFF, 32)
            preprocessed = await self._milestone2.execute(contract, report_stage)
            await self._save_stage(job, PipelineStage.VALIDATE_M2_OUTPUT, 52)
            self._validator.validate_m2(contract, preprocessed)
            job.preprocessed_transcript = preprocessed
            job.status = JobStatus.AWAITING_ANALYSIS
            await self._save_stage(job, PipelineStage.PREPROCESSED_TRANSCRIPT_READY, 55)

        except ContractValidationError as exc:
            logger.error("Contract validation error in job %s: %s", job_id, exc.message)
            job.status = JobStatus.FAILED
            job.error_code = exc.code
            job.error_message = exc.message
            job.failed_milestone = MilestoneName.M1_VALIDATION if exc.code == "invalid_m1_output" else MilestoneName.M2_VALIDATION
            await self._repository.save(job)
        except Exception as exc:
            logger.exception("Pipeline processing failure in job %s: %s", job_id, exc)
            job.status = JobStatus.FAILED
            job.error_code = getattr(exc, "code", "pipeline_processing_failed")
            job.error_message = str(exc)
            if job.unified_transcript is not None:
                job.failed_milestone = getattr(exc, "milestone", MilestoneName.M2)
            else:
                job.failed_milestone = getattr(exc, "milestone", MilestoneName.M1)
            await self._repository.save(job)
        finally:
            if not self._keep_work_files and job_directory.exists():
                shutil.rmtree(job_directory, ignore_errors=True)

    async def _save_stage(self, job: JobRecord, stage: PipelineStage, progress_percent: int) -> None:
        """Updates and persists the current pipeline stage and progress percentage."""
        job.current_stage = stage
        job.progress_percent = progress_percent
        await self._repository.save(job)
